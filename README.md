# GReCF

Anonymous implementation of **Generative Recommendation via Continuous-Space
Collaborative Filtering**.

## Abstract

Generative recommendation aims to create and recommend novel items beyond
fixed catalogs, addressing users' evolving or unrepresented preferences that
existing catalogs cannot adequately capture. It is important to generate
content aligned with users' historical preferences while reliably
extrapolating beyond observed history. We formulate this task as collaborative
preference-field completion in continuous item space, with user-based and
item-based collaborative filtering providing two completion mechanisms.

We instantiate this formulation in GReCF, a reference-free generative
recommender in the image modality. GReCF represents users with adaptive
multi-interest themes, anchors each generation to one theme, and injects user
and theme conditions into a pretrained diffusion model through residual
preference attention. Shared training transfers evidence across users, while
masked-interest training learns relations among co-occurring themes to support
extrapolation beyond observed history.

We further introduce an extrapolation-focused evaluation protocol covering
full-pool ranking, personalization, image quality, multi-interest coverage, and
preference-source attribution. The evaluation demonstrates competitive
ranking, strong personalization, favorable image quality, high coverage, and
reliable CF extrapolation with few unsupported generations. Cold-start
evaluation further shows that GReCF preserves initial interests while enabling
CF extrapolation over iterative generation rounds.

## Repository Structure

```text
GReCF/
  grecf/                         Core method implementation
    data.py                      Shared input and cache interfaces
    multi_interest.py            Adaptive themes and SD1.5 conditioning
    preference_ip_adapter.py     Residual preference-attention branches
    reference.py                 User representations and Item Average
    image_embedder.py            Differentiable CLIP image encoding
    sdxl.py                      SDXL preference adapter
    pixart.py                    PixArt caption-prefix adapter

  experiments/                   Training and generation entry points
    sd15/                        SD1.5 training and generation
    sdxl/                        SDXL context, training, and generation
    pixart/                      PixArt context, training, and generation
    cold_start/                  Five-round cold-start generation and evaluation
    generation_utils.py          Shared generation utilities
    merge_generation_shards.py  Generation-shard consolidation

  baselines/                     Baseline adaptation code
    viper/                       ViPer preference extraction and generation
    rebeca/                      REBECA preparation, prior training, and generation
    gnr/                         GNR SFT preparation, training, and generation
    navigen/                     NaviGen CID/TID, SFT, GRPO, and generation

  evaluation/                    Evaluation implementations
    ranking/                     Full-pool HitRate, Recall, and NDCG
    personalization/             CLIP Similarity, LPIPS, and Coverage
    image_quality/               LAION Aesthetic evaluation
    interest_source/             History/CF/Other attribution and cross-judge checks
    human_validation.py          Human verification aggregation

  configs/
    experiment_settings.yaml     Experiment settings used by GReCF

  requirements.txt               Python dependencies
```

## Environment

Run all commands from the repository root:

```bash
cd GReCF
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the paths used by the commands below:

```bash
export INPUT_ROOT=/path/to/input
export SELECTED_USERS=/path/to/selected_users.json
export CLIP_MODEL=/path/to/clip-vit-base-patch32
export CLIP_CACHE=/path/to/public_clip_features.float16.npy
export LATENT_CACHE=/path/to/image_latents.float16.npy
export SD15_MODEL=/path/to/stable-diffusion-v1-5
export SDXL_MODEL=/path/to/stable-diffusion-xl-base-1.0
export PIXART_MODEL=/path/to/PixArt-XL-2-512x512
export QWEN3_VL_MODEL=/path/to/Qwen3-VL-32B-Instruct
export IP_ADAPTER=/path/to/IP-Adapter
export JANUS_MODEL=/path/to/Janus-Pro-1B
export JANUS_SOURCE=/path/to/Janus
export REBECA_SOURCE=/path/to/REBECA
```

The settings stated in the paper are centralized in
`configs/experiment_settings.yaml` and are also the defaults of the experiment
entry points.

## Main Experiments

### SD1.5

```bash
torchrun --standalone --nproc_per_node=8 -m experiments.sd15.train \
  --dataset-root "$INPUT_ROOT" \
  --sd15-model "$SD15_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --latent-cache "$LATENT_CACHE" \
  --user-rep-cache cache/user_representations.float32.npy \
  --interest-cache-dir cache/adaptive_interests \
  --results-dir outputs \
  --run-name grecf_sd15

python -m experiments.sd15.generate \
  --dataset-root "$INPUT_ROOT" \
  --sd15-model "$SD15_MODEL" \
  --clip-model "$CLIP_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --user-rep-cache cache/user_representations.float32.npy \
  --interest-cache-dir cache/adaptive_interests \
  --checkpoint outputs/grecf_sd15/checkpoints/last.pt \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/grecf_sd15_generation \
  --skip-contact-sheets
```

### SDXL1.0

```bash
python -m experiments.sdxl.prepare_context \
  --model "$SDXL_MODEL" \
  --output cache/sdxl_context.pt

torchrun --standalone --nproc_per_node=8 -m experiments.sdxl.train \
  --model "$SDXL_MODEL" \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --latent-cache "$LATENT_CACHE" \
  --interest-cache-dir cache/adaptive_interests \
  --base-context cache/sdxl_context.pt \
  --output-dir outputs/grecf_sdxl

python -m experiments.sdxl.generate \
  --model "$SDXL_MODEL" \
  --checkpoint outputs/grecf_sdxl/last.pt \
  --base-context cache/sdxl_context.pt \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --clip-model "$CLIP_MODEL" \
  --interest-cache-dir cache/adaptive_interests \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/grecf_sdxl_generation
```

### PixArt

```bash
python -m experiments.pixart.prepare_context \
  --model "$PIXART_MODEL" \
  --output cache/pixart_context.pt

torchrun --standalone --nproc_per_node=8 -m experiments.pixart.train \
  --model "$PIXART_MODEL" \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --latent-cache "$LATENT_CACHE" \
  --interest-cache-dir cache/adaptive_interests \
  --base-context cache/pixart_context.pt \
  --output-dir outputs/grecf_pixart

python -m experiments.pixart.generate \
  --model "$PIXART_MODEL" \
  --checkpoint outputs/grecf_pixart/last.pt \
  --base-context cache/pixart_context.pt \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --clip-model "$CLIP_MODEL" \
  --interest-cache-dir cache/adaptive_interests \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/grecf_pixart_generation
```

## Baseline Experiments

### ViPer

```bash
python baselines/viper/extract_preferences.py \
  --dataset-root "$INPUT_ROOT" \
  --selected-users-json "$SELECTED_USERS" \
  --model "$QWEN3_VL_MODEL" \
  --output-jsonl outputs/viper/preferences.jsonl

python baselines/viper/generate.py \
  --preferences-jsonl outputs/viper/preferences.jsonl \
  --dataset-root "$INPUT_ROOT" \
  --sd15-model "$SD15_MODEL" \
  --clip-model "$CLIP_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/viper/generation
```

### REBECA

```bash
python baselines/rebeca/prepare.py \
  --dataset-root "$INPUT_ROOT" \
  --output-dir outputs/rebeca/prepared

python baselines/rebeca/encode_catalog.py \
  --dataset-root "$INPUT_ROOT" \
  --sd15-model "$SD15_MODEL" \
  --ip-adapter-dir "$IP_ADAPTER" \
  --output-file outputs/rebeca/catalog_features.pt

python baselines/rebeca/train_prior.py \
  --upstream-root "$REBECA_SOURCE" \
  --ratings-csv outputs/rebeca/prepared/processed/ratings.csv \
  --catalog-features outputs/rebeca/catalog_features.pt \
  --output-dir outputs/rebeca/prior

python baselines/rebeca/generate.py \
  --upstream-root "$REBECA_SOURCE" \
  --weights-dir outputs/rebeca/prior \
  --dataset-root "$INPUT_ROOT" \
  --sd15-model "$SD15_MODEL" \
  --ip-adapter-dir "$IP_ADAPTER" \
  --clip-model "$CLIP_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/rebeca/generation
```

### GNR

```bash
python baselines/gnr/prepare_sft_data.py \
  --dataset-root "$INPUT_ROOT" \
  --output-jsonl outputs/gnr/train.jsonl

torchrun --standalone --nproc_per_node=8 baselines/gnr/train_sft.py \
  --janus-root "$JANUS_SOURCE" \
  --model "$JANUS_MODEL" \
  --train-jsonl outputs/gnr/train.jsonl \
  --output-dir outputs/gnr/sft

python baselines/gnr/generate.py \
  --janus-root "$JANUS_SOURCE" \
  --model "$JANUS_MODEL" \
  --sft-checkpoint outputs/gnr/sft/checkpoints/sft_full_bf16.pth \
  --dataset-root "$INPUT_ROOT" \
  --clip-model "$CLIP_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/gnr/generation
```

### NaviGen

```bash
python baselines/navigen/build_cids.py \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --output-dir outputs/navigen/cids

python baselines/navigen/build_tids.py \
  --dataset-root "$INPUT_ROOT" \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --image-cache-root /path/to/image_cache \
  --output-jsonl outputs/navigen/tids.jsonl

python baselines/navigen/prepare.py \
  --dataset-root "$INPUT_ROOT" \
  --tids-jsonl outputs/navigen/tids.jsonl \
  --cid-assignments outputs/navigen/cids/cid_assignments.int16.npy \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/navigen/prepared \
  --teacher-input-dir outputs/navigen/teacher_inputs

python baselines/navigen/build_teacher.py \
  --input-jsonl outputs/navigen/teacher_inputs/train_teacher_input.jsonl \
  --output-jsonl outputs/navigen/train_teacher.jsonl \
  --output-parquet outputs/navigen/prepared/train_cid2ins.parquet

python baselines/navigen/build_teacher.py \
  --input-jsonl outputs/navigen/teacher_inputs/valid_teacher_input.jsonl \
  --output-jsonl outputs/navigen/valid_teacher.jsonl \
  --output-parquet outputs/navigen/prepared/valid_cid2ins.parquet

torchrun --standalone --nproc_per_node=8 \
  baselines/navigen/run_cached_stage.py \
  --official-script /path/to/navigen/train/sft_aigc_stage1_embed.py \
  --output_dir outputs/navigen/prepared \
  --sft_data_dir outputs/navigen/sft_data_stage1 \
  --model_name /path/to/navigen_expanded_model \
  --train_out_dir outputs/navigen/stage1 \
  --max_seq_len 2048 --epochs 3 --max_steps 200 --lr 5e-4

torchrun --standalone --nproc_per_node=8 \
  baselines/navigen/run_cached_stage.py \
  --official-script /path/to/navigen/train/sft_aigc_stage2_full_ft.py \
  --output_dir outputs/navigen/prepared \
  --sft_data_dir outputs/navigen/sft_data_stage2 \
  --model_name outputs/navigen/stage1/final \
  --train_out_dir outputs/navigen/stage2 \
  --max_seq_len 2048 --epochs 1 --lr 5e-4

torchrun --standalone --nproc_per_node=8 \
  baselines/navigen/run_official_grpo.py \
  /path/to/navigen_grpo.py \
  --grpo-max-steps 600 \
  --base_model_dir outputs/navigen/stage2/final \
  --pid2cid2tid_path outputs/navigen/prepared/pid2cid2tid.parquet \
  --train_cid2cid_path outputs/navigen/prepared/train_cid2cid.parquet \
  --val_cid2cid_path outputs/navigen/prepared/valid_cid2cid.parquet \
  --train_cid2ins_path outputs/navigen/prepared/train_cid2ins.parquet \
  --val_cid2ins_path outputs/navigen/prepared/valid_cid2ins.parquet \
  --output_dir outputs/navigen/grpo

torchrun --standalone --nproc_per_node=8 \
  baselines/navigen/run_compat_infer.py \
  --official-script /path/to/navigen_inference.py \
  --input_dir outputs/navigen/prepared \
  --model_dir outputs/navigen/grpo/final_rl_lora \
  --base_model_dir outputs/navigen/stage2/final \
  --pred_dir outputs/navigen/raw_predictions \
  --max_rows 1000 --generation_mode two_stage

python baselines/navigen/normalize_predictions.py \
  --prediction-dir outputs/navigen/raw_predictions \
  --output-jsonl outputs/navigen/predictions.jsonl

python baselines/navigen/generate.py \
  --predictions-jsonl outputs/navigen/predictions.jsonl \
  --dataset-root "$INPUT_ROOT" \
  --sd15-model "$SD15_MODEL" \
  --clip-model "$CLIP_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --selected-users-json "$SELECTED_USERS" \
  --output-dir outputs/navigen/generation
```

## Ablations

The ablations share the SD1.5 training and generation implementation:

- `Item Average` replaces the adaptive multi-interest representation with the
  mean embedding of all historical items.
- `w/o Anchor` retains the complete user representation but removes the
  single-theme generation anchor.
- `w/o Mask` disables masked-interest training while preserving the remaining
  architecture.

```bash
for ABLATION in item_average no_theme_anchor no_mask; do
  torchrun --standalone --nproc_per_node=8 -m experiments.sd15.train \
    --dataset-root "$INPUT_ROOT" \
    --sd15-model "$SD15_MODEL" \
    --clip-cache "$CLIP_CACHE" \
    --latent-cache "$LATENT_CACHE" \
    --user-rep-cache cache/user_representations.float32.npy \
    --interest-cache-dir cache/adaptive_interests \
    --results-dir outputs \
    --run-name "grecf_${ABLATION}" \
    --ablation "$ABLATION"

  python -m experiments.sd15.generate \
    --dataset-root "$INPUT_ROOT" \
    --sd15-model "$SD15_MODEL" \
    --clip-model "$CLIP_MODEL" \
    --clip-cache "$CLIP_CACHE" \
    --user-rep-cache cache/user_representations.float32.npy \
    --interest-cache-dir cache/adaptive_interests \
    --checkpoint "outputs/grecf_${ABLATION}/checkpoints/last.pt" \
    --selected-users-json "$SELECTED_USERS" \
    --output-dir "outputs/grecf_${ABLATION}_generation" \
    --skip-contact-sheets
done
```

## Evaluation

The following commands evaluate one completed generation directory:

```bash
export GENERATION_DIR=outputs/grecf_sd15_generation
export EVAL_DIR=outputs/grecf_sd15_evaluation

python -m evaluation.ranking.retrieval_rank \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --method "GReCF=$GENERATION_DIR" \
  --output-dir "$EVAL_DIR/ranking"

python -m evaluation.personalization.clip_similarity \
  --generation-metrics "$GENERATION_DIR/tables/generation_metrics.csv" \
  --method GReCF \
  --output "$EVAL_DIR/clip_similarity.json"

python -m evaluation.personalization.lpips \
  --generation-metrics "$GENERATION_DIR/tables/generation_metrics.csv" \
  --dataset-root "$INPUT_ROOT" \
  --method GReCF \
  --output "$EVAL_DIR/lpips.json"

python -m evaluation.personalization.coverage \
  --generation-dir "$GENERATION_DIR" \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --test-interest-cache-dir cache/test_adaptive_interests \
  --method GReCF \
  --output-dir "$EVAL_DIR/coverage"

python -m evaluation.image_quality.laion_aesthetic \
  --image-dir "$GENERATION_DIR/images" \
  --method GReCF \
  --predictor-weights /path/to/laion_aesthetic_predictor.pth \
  --clip-checkpoint /path/to/ViT-L-14.pt \
  --output-dir "$EVAL_DIR/aesthetic"
```

### Preference-Source Attribution

```bash
python -m evaluation.interest_source.build_tasks \
  --dataset-root "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --user-rep-cache cache/user_representations.float32.npy \
  --selected-users-json "$SELECTED_USERS" \
  --generated-features "$GENERATION_DIR/tables/generated_clip_features.float32.npy" \
  --generated-manifest "$GENERATION_DIR/tables/generation_metrics.csv" \
  --eval-method GReCF \
  --output-jsonl "$EVAL_DIR/judge_tasks.jsonl"

python -m evaluation.interest_source.run_judge \
  --tasks-jsonl "$EVAL_DIR/judge_tasks.jsonl" \
  --model /path/to/Qwen3-VL-32B-Instruct \
  --output-jsonl "$EVAL_DIR/judge_shard0.jsonl"

python -m evaluation.interest_source.merge_judge \
  --tasks-jsonl "$EVAL_DIR/judge_tasks.jsonl" \
  --input-glob "$EVAL_DIR/judge_shard*.jsonl" \
  --output-jsonl "$EVAL_DIR/judge_merged.jsonl" \
  --summary-json "$EVAL_DIR/judge_summary.json"
```

### Cold Start

For Round 1, build the balanced user context and generate 10 images per user:

```bash
python -m experiments.cold_start.prepare_round \
  --base-dataset "$INPUT_ROOT" \
  --cold-root /path/to/cold_start_input \
  --clip-cache "$CLIP_CACHE" \
  --base-user-reps cache/user_representations.float32.npy \
  --base-interest-cache cache/adaptive_interests \
  --output-root outputs/cold_start/context_r1

python -m experiments.sd15.generate \
  --dataset-root outputs/cold_start/context_r1/dataset \
  --sd15-model "$SD15_MODEL" \
  --clip-model "$CLIP_MODEL" \
  --clip-cache "$CLIP_CACHE" \
  --user-rep-cache outputs/cold_start/context_r1/user_reps.float32.npy \
  --interest-cache-dir outputs/cold_start/context_r1/interests \
  --checkpoint outputs/grecf_sd15/checkpoints/last.pt \
  --selected-users-json outputs/cold_start/context_r1/cold_users.json \
  --output-dir outputs/cold_start/generation_r1 \
  --skip-contact-sheets
```

For each later round, rerun `prepare_round` with all preceding generation
directories supplied through repeated `--previous-round` arguments, then run
the same SD1.5 generation command with the new context directory. Continue
until five rounds are complete.

Evaluate all five rounds after generation:

```bash
for ROUND in 1 2 3 4 5; do
  python -m experiments.cold_start.lpips_to_history \
    --generation-dir "outputs/cold_start/generation_r${ROUND}" \
    --context-root "outputs/cold_start/context_r${ROUND}" \
    --output-dir "outputs/cold_start/lpips_r${ROUND}"

  python -m experiments.cold_start.lpips_to_history \
    --generation-dir "outputs/cold_start/generation_r${ROUND}" \
    --context-root "outputs/cold_start/context_r${ROUND}" \
    --output-dir "outputs/cold_start/lpips_r${ROUND}" \
    --merge-only
done

python -m experiments.cold_start.build_judge_tasks \
  --base-dataset "$INPUT_ROOT" \
  --clip-cache "$CLIP_CACHE" \
  --base-user-reps cache/user_representations.float32.npy \
  --context-root outputs/cold_start/context_r1 \
  --context-root outputs/cold_start/context_r2 \
  --context-root outputs/cold_start/context_r3 \
  --context-root outputs/cold_start/context_r4 \
  --context-root outputs/cold_start/context_r5 \
  --generation-dir outputs/cold_start/generation_r1 \
  --generation-dir outputs/cold_start/generation_r2 \
  --generation-dir outputs/cold_start/generation_r3 \
  --generation-dir outputs/cold_start/generation_r4 \
  --generation-dir outputs/cold_start/generation_r5 \
  --output outputs/cold_start/judge_tasks.jsonl

python -m evaluation.interest_source.run_judge \
  --tasks-jsonl outputs/cold_start/judge_tasks.jsonl \
  --model "$QWEN3_VL_MODEL" \
  --output-jsonl outputs/cold_start/judge_shard0.jsonl

python -m evaluation.interest_source.merge_judge \
  --tasks-jsonl outputs/cold_start/judge_tasks.jsonl \
  --input-glob "outputs/cold_start/judge_shard*.jsonl" \
  --output-jsonl outputs/cold_start/judge_merged.jsonl \
  --summary-json outputs/cold_start/judge_summary.json

python -m experiments.cold_start.summarize \
  --base-dataset "$INPUT_ROOT" \
  --judge-merged outputs/cold_start/judge_merged.jsonl \
  --generation-dir outputs/cold_start/generation_r1 \
  --generation-dir outputs/cold_start/generation_r2 \
  --generation-dir outputs/cold_start/generation_r3 \
  --generation-dir outputs/cold_start/generation_r4 \
  --generation-dir outputs/cold_start/generation_r5 \
  --context-root outputs/cold_start/context_r1 \
  --context-root outputs/cold_start/context_r2 \
  --context-root outputs/cold_start/context_r3 \
  --context-root outputs/cold_start/context_r4 \
  --context-root outputs/cold_start/context_r5 \
  --lpips-summary outputs/cold_start/lpips_r1/lpips_to_history_summary.json \
  --lpips-summary outputs/cold_start/lpips_r2/lpips_to_history_summary.json \
  --lpips-summary outputs/cold_start/lpips_r3/lpips_to_history_summary.json \
  --lpips-summary outputs/cold_start/lpips_r4/lpips_to_history_summary.json \
  --lpips-summary outputs/cold_start/lpips_r5/lpips_to_history_summary.json \
  --output outputs/cold_start/summary.json
```
