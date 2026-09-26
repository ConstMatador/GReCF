from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.models.attention_processor import Attention
except Exception:  # pragma: no cover - only used for type checking friendliness.
    Attention = object  # type: ignore


class PreferenceIPAttnProcessor2_0:
    """Parameter-free IP-Adapter-style extra attention branch for SD cross-attention.

    The base text prompt keeps using the original SD cross-attention path.  Optional
    user-preference and interest-delta tokens are supplied through
    ``cross_attention_kwargs`` and attend through separate branches. This
    version intentionally reuses the frozen SD attention projections; the trainable
    part is the upstream projector that produces the extra condition tokens.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        preference_user_tokens: torch.Tensor | None = None,
        preference_user_mask: torch.Tensor | None = None,
        preference_delta_tokens: torch.Tensor | None = None,
        preference_user_scale: float | torch.Tensor = 1.0,
        preference_delta_scale: float | torch.Tensor = 1.0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            kwargs.pop("scale", None)

        user_tokens = preference_user_tokens
        delta_tokens = preference_delta_tokens
        user_scale = preference_user_scale
        delta_scale = preference_delta_scale

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            use_preference = False
        else:
            use_preference = user_tokens is not None or delta_tokens is not None
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if use_preference:
            if user_tokens is not None and user_tokens.shape[0] == batch_size:
                attended = attended + self._preference_attention(
                    attn, query, user_tokens, batch_size, head_dim, user_scale
                )
            if delta_tokens is not None and delta_tokens.shape[0] == batch_size:
                attended = attended + self._preference_attention(
                    attn, query, delta_tokens, batch_size, head_dim, delta_scale
                )

        hidden_states = attended.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor

    @staticmethod
    def _preference_attention(
        attn: Attention,
        query: torch.Tensor,
        tokens: torch.Tensor,
        batch_size: int,
        head_dim: int,
        scale: float | torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens.to(device=query.device, dtype=query.dtype)
        pref_key = attn.to_k(tokens)
        pref_value = attn.to_v(tokens)
        pref_key = pref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        pref_value = pref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_k is not None:
            pref_key = attn.norm_k(pref_key)
        output = F.scaled_dot_product_attention(
            query, pref_key, pref_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        if torch.is_tensor(scale):
            scale = scale.to(device=query.device, dtype=query.dtype)
        return output * scale


def install_preference_ip_processors(unet) -> None:
    """Install the parameter-free extra-branch processor on every UNet attention layer."""

    unet.set_attn_processor({
        name: PreferenceIPAttnProcessor2_0()
        for name in unet.attn_processors.keys()
    })


class PreferenceIPTrainableKVAttnProcessor2_0(nn.Module):
    """Trainable-K/V Preference-IP branch for SD cross-attention.

    The base text prompt still uses the original frozen SD text cross-attention
    path.  User and delta preference tokens use their own trainable K/V
    projections, initialized from the frozen SD text K/V projections.  This keeps
    the first forward pass close to the parameter-free behavior while giving
    the preference branch enough capacity to adapt image-space preference tokens
    to each UNet attention layer.
    """

    def __init__(
        self,
        attn: Attention,
        layer_gates: bool = False,
        user_gate_init: float = 1.0,
        delta_gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        inner_dim = int(attn.to_k.out_features)
        cross_dim = int(attn.to_k.in_features)
        use_bias = attn.to_k.bias is not None
        self.user_to_k = nn.Linear(cross_dim, inner_dim, bias=use_bias)
        self.user_to_v = nn.Linear(cross_dim, inner_dim, bias=attn.to_v.bias is not None)
        self.delta_to_k = nn.Linear(cross_dim, inner_dim, bias=use_bias)
        self.delta_to_v = nn.Linear(cross_dim, inner_dim, bias=attn.to_v.bias is not None)
        self.layer_gates = bool(layer_gates)
        if self.layer_gates:
            self.user_layer_gate_logit = nn.Parameter(torch.logit(torch.tensor(float(user_gate_init)).clamp(1e-4, 1 - 1e-4)))
            self.delta_layer_gate_logit = nn.Parameter(torch.logit(torch.tensor(float(delta_gate_init)).clamp(1e-4, 1 - 1e-4)))
        self._copy_from_sd(attn)

    def _copy_from_sd(self, attn: Attention) -> None:
        device = attn.to_k.weight.device
        self.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            for layer in (self.user_to_k, self.delta_to_k):
                layer.weight.copy_(attn.to_k.weight.detach().float())
                if layer.bias is not None and attn.to_k.bias is not None:
                    layer.bias.copy_(attn.to_k.bias.detach().float())
            for layer in (self.user_to_v, self.delta_to_v):
                layer.weight.copy_(attn.to_v.weight.detach().float())
                if layer.bias is not None and attn.to_v.bias is not None:
                    layer.bias.copy_(attn.to_v.bias.detach().float())

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        preference_user_tokens: torch.Tensor | None = None,
        preference_user_mask: torch.Tensor | None = None,
        preference_delta_tokens: torch.Tensor | None = None,
        preference_user_scale: float | torch.Tensor = 1.0,
        preference_delta_scale: float | torch.Tensor = 1.0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # diffusers filters cross_attention_kwargs by inspecting
        # ``processor.__call__`` rather than ``forward``.  Keep an explicit
        # signature here, but delegate to nn.Module.__call__ so hooks and module
        # semantics remain intact.
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            preference_user_tokens=preference_user_tokens,
            preference_user_mask=preference_user_mask,
            preference_delta_tokens=preference_delta_tokens,
            preference_user_scale=preference_user_scale,
            preference_delta_scale=preference_delta_scale,
            *args,
            **kwargs,
        )

    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        preference_user_tokens: torch.Tensor | None = None,
        preference_user_mask: torch.Tensor | None = None,
        preference_delta_tokens: torch.Tensor | None = None,
        preference_user_scale: float | torch.Tensor = 1.0,
        preference_delta_scale: float | torch.Tensor = 1.0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            kwargs.pop("scale", None)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            use_preference = False
        else:
            use_preference = preference_user_tokens is not None or preference_delta_tokens is not None
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if use_preference:
            if preference_user_tokens is not None and preference_user_tokens.shape[0] == batch_size:
                attended = attended + self._preference_attention(
                    attn,
                    query,
                    preference_user_tokens,
                    self.user_to_k,
                    self.user_to_v,
                    batch_size,
                    head_dim,
                    self._apply_layer_gate(preference_user_scale, "user", query),
                    preference_user_mask,
                )
            if preference_delta_tokens is not None and preference_delta_tokens.shape[0] == batch_size:
                attended = attended + self._preference_attention(
                    attn,
                    query,
                    preference_delta_tokens,
                    self.delta_to_k,
                    self.delta_to_v,
                    batch_size,
                    head_dim,
                    self._apply_layer_gate(preference_delta_scale, "delta", query),
                    None,
                )

        hidden_states = attended.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor

    def _apply_layer_gate(
        self,
        scale: float | torch.Tensor,
        branch: str,
        reference: torch.Tensor,
    ) -> float | torch.Tensor:
        if not self.layer_gates:
            return scale
        gate = (
            torch.sigmoid(self.user_layer_gate_logit)
            if branch == "user"
            else torch.sigmoid(self.delta_layer_gate_logit)
        ).to(device=reference.device, dtype=reference.dtype)
        if torch.is_tensor(scale):
            return scale.to(device=reference.device, dtype=reference.dtype) * gate
        return torch.as_tensor(float(scale), device=reference.device, dtype=reference.dtype) * gate

    @staticmethod
    def _preference_attention(
        attn: Attention,
        query: torch.Tensor,
        tokens: torch.Tensor,
        to_k: nn.Linear,
        to_v: nn.Linear,
        batch_size: int,
        head_dim: int,
        scale: float | torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens.to(device=query.device, dtype=to_k.weight.dtype)
        pref_key = to_k(tokens).to(dtype=query.dtype)
        pref_value = to_v(tokens).to(dtype=query.dtype)
        pref_key = pref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        pref_value = pref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_k is not None:
            pref_key = attn.norm_k(pref_key)
        preference_mask = None
        if token_mask is not None:
            token_mask = token_mask.to(device=query.device, dtype=torch.bool)
            if token_mask.shape != (batch_size, pref_key.shape[-2]):
                raise ValueError(
                    f"preference token mask must have shape {(batch_size, pref_key.shape[-2])}, "
                    f"got {tuple(token_mask.shape)}"
                )
            preference_mask = token_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query, pref_key, pref_value, attn_mask=preference_mask, dropout_p=0.0, is_causal=False
        )
        if torch.is_tensor(scale):
            scale = scale.to(device=query.device, dtype=query.dtype)
        return output * scale


def _attention_module_from_processor_name(unet, processor_name: str):
    module_name = processor_name.rsplit(".processor", 1)[0]
    return unet.get_submodule(module_name)


def _gate_init_for_processor(processor_name: str) -> tuple[float, float]:
    if processor_name.startswith("mid_block"):
        return 0.85, 0.90
    if processor_name.startswith("down_blocks"):
        return 0.70, 0.75
    if processor_name.startswith("up_blocks"):
        return 0.45, 0.55
    return 0.65, 0.70


def install_preference_ip_trainable_kv_processors(unet, layer_gates: bool = False) -> None:
    """Install trainable-K/V Preference-IP processors on cross-attention layers.

    Self-attention layers keep the parameter-free processor because preference
    tokens are only meaningful when ``encoder_hidden_states`` are present.
    """

    processors = {}
    for name in unet.attn_processors.keys():
        if ".attn2." in name:
            user_init, delta_init = _gate_init_for_processor(name)
            processors[name] = PreferenceIPTrainableKVAttnProcessor2_0(
                _attention_module_from_processor_name(unet, name),
                layer_gates=layer_gates,
                user_gate_init=user_init,
                delta_gate_init=delta_init,
            )
        else:
            processors[name] = PreferenceIPAttnProcessor2_0()
    unet.set_attn_processor(processors)


def preference_ip_trainable_parameters(unet) -> list[nn.Parameter]:
    return [
        parameter
        for module in unet.attn_processors.values()
        if isinstance(module, PreferenceIPTrainableKVAttnProcessor2_0)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def preference_ip_processor_state_dict(unet) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: module.state_dict()
        for name, module in unet.attn_processors.items()
        if isinstance(module, PreferenceIPTrainableKVAttnProcessor2_0)
    }


def load_preference_ip_processor_state_dict(unet, state: dict[str, dict[str, torch.Tensor]]) -> None:
    missing = []
    for name, payload in state.items():
        module = unet.attn_processors.get(name)
        if module is None:
            missing.append(name)
            continue
        module.load_state_dict(payload)
    if missing:
        raise KeyError(f"missing Preference-IP processors in UNet: {missing[:3]}")


def preference_ip_processor_gate_summary(unet) -> dict[str, float]:
    user_values = []
    delta_values = []
    for module in unet.attn_processors.values():
        if isinstance(module, PreferenceIPTrainableKVAttnProcessor2_0) and module.layer_gates:
            user_values.append(float(torch.sigmoid(module.user_layer_gate_logit.detach()).cpu()))
            delta_values.append(float(torch.sigmoid(module.delta_layer_gate_logit.detach()).cpu()))
    if not user_values:
        return {}
    user_tensor = torch.tensor(user_values)
    delta_tensor = torch.tensor(delta_values)
    return {
        "preference_layer_user_gate_mean": float(user_tensor.mean()),
        "preference_layer_user_gate_min": float(user_tensor.min()),
        "preference_layer_user_gate_max": float(user_tensor.max()),
        "preference_layer_delta_gate_mean": float(delta_tensor.mean()),
        "preference_layer_delta_gate_min": float(delta_tensor.min()),
        "preference_layer_delta_gate_max": float(delta_tensor.max()),
    }
