#!/bin/bash
# Launch script for train_ttt_simsiam.py with convenient hyperparameter knobs.
# Mirrors the data pipeline from train.py while enabling SimSiam + TTT tuning.

# set -euo pipefail

# -----------------------------------------------------------------------------
# Basic experiment metadata
# -----------------------------------------------------------------------------
PROJECT=VGPS
NAME=TTT_SimSiam_fractal_base
SAVE_DIR=/home/yruan/V-GPS/save
mkdir -p "$SAVE_DIR"

# -----------------------------------------------------------------------------
# Data configuration (reuses existing OXE data config)
# -----------------------------------------------------------------------------
data_cfg=experiments/configs/data_config.py
DATA_DIR=gs://gresearch/robotics
data_mix=fractal
DISCOUNT=0.98
BATCH_SIZE=256

# -----------------------------------------------------------------------------
# SimSiam + TTT knobs you likely want to sweep
# -----------------------------------------------------------------------------
LAMBDA_SIM=1.0          # weight on auxiliary cosine loss
SIMSIAM_LR=3e-4         # step size for encoder updates driven by SimSiam loss

HIDDEN_DIMS=(512 512)   # predictor MLP widths
PRED_NORM="layer"      # layer|group|none
PRED_DROPOUT=0.0
NUM_GROUPS=32
STOP_GRAD_TARGET=true   # freeze next-observation branch
NORMALIZE_LATENTS=true

TTT_STEPS=1             # number of gradient steps per test batch
TTT_LR=1e-3             # learning rate for those steps
TTT_MODE="online"      # standard|online|off
EVAL_BATCHES=4          # number of eval batches per eval_interval

# -----------------------------------------------------------------------------
# Compose hidden dim flags for ml_collections (pass each entry separately)
# -----------------------------------------------------------------------------
HIDDEN_FLAGS=()
for width in "${HIDDEN_DIMS[@]}"; do
  HIDDEN_FLAGS+=(--config.simsiam_hidden_dims "$width")
done

# -----------------------------------------------------------------------------
# Kick off training
# -----------------------------------------------------------------------------
python experiments/train_ttt_simsiam.py \
  --config experiments/configs/ttt_simsiam_config.py:lc_cql_simsiam \
  --oxedata_config "$data_cfg" \
  --name "$NAME" \
  --project "$PROJECT" \
  --config.save_dir "$SAVE_DIR" \
  --config.batch_size "$BATCH_SIZE" \
  --config.lambda_sim "$LAMBDA_SIM" \
  --config.simsiam_lr "$SIMSIAM_LR" \
  --config.predictor_dropout_rate "$PRED_DROPOUT" \
  --config.simsiam_norm "$PRED_NORM" \
  --config.simsiam_num_groups "$NUM_GROUPS" \
  --config.stop_grad_target "$STOP_GRAD_TARGET" \
  --config.normalize_latents "$NORMALIZE_LATENTS" \
  --config.ttt_steps "$TTT_STEPS" \
  --config.ttt_lr "$TTT_LR" \
  --config.ttt_mode "$TTT_MODE" \
  --config.eval_batches "$EVAL_BATCHES" \
  "${HIDDEN_FLAGS[@]}" \
  --oxedata_config.batch_size "$BATCH_SIZE" \
  --oxedata_config.oxe_kwargs.data_dir "$DATA_DIR" \
  --oxedata_config.oxe_kwargs.data_mix "$data_mix" \
  --oxedata_config.oxe_kwargs.discount "$DISCOUNT"
