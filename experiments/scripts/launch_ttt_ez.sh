#!/bin/bash
# Launch script for unified CQL+SimSiam training.
#
# This script uses the new unified agent that handles both RL and SimSiam
# losses in a single update call. Much simpler than the original!

# =============================================================================
# Experiment Configuration
# =============================================================================
PROJECT="VGPS"
NAME="TTT_EZ_fractal_base"
SAVE_DIR="${SAVE_DIR:-/home/yruan/V-GPS/save}"
mkdir -p "$SAVE_DIR"

# =============================================================================
# Data Configuration
# =============================================================================
DATA_CONFIG="experiments/configs/data_config.py"
DATA_DIR="gs://gresearch/robotics"
DATA_MIX="fractal"

# =============================================================================
# Training Configuration
# =============================================================================
# Choose config variant:
#   - lc_cql_simsiam      : Standard CQL+SimSiam with AdamW (recommended)
#   - lc_cql_simsiam_conv : With convolutional dynamics (for spatial features)
#   - lc_cql_simsiam_sgd  : With SGD optimizer (closer to EfficientZero)
#   - lc_cql_baseline     : CQL only, no SimSiam (for comparison)
CONFIG_VARIANT="${CONFIG_VARIANT:-lc_cql_simsiam}"

# Training schedule
BATCH_SIZE=512
NUM_STEPS=1000000
LOG_INTERVAL=1000
EVAL_INTERVAL=20000
SAVE_INTERVAL=100000
SEED=42

# =============================================================================
# SimSiam Configuration (override defaults if needed)
# =============================================================================
LAMBDA_SIM=2.0  # SimSiam loss weight (EfficientZero uses 2.0)

# =============================================================================
# TTT Configuration
# =============================================================================
TTT_MODE="online"   # "standard", "online", or "off"
TTT_STEPS=1
TTT_LR=1e-2

# =============================================================================
# Print Configuration
# =============================================================================
echo "=========================================="
echo "Unified CQL+SimSiam Training"
echo "=========================================="
echo ""
echo "Experiment:"
echo "  Project: $PROJECT"
echo "  Name: $NAME"
echo "  Config: $CONFIG_VARIANT"
echo ""
echo "Data:"
echo "  Mix: $DATA_MIX"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "Training:"
echo "  Steps: $NUM_STEPS"
echo "  λ_sim: $LAMBDA_SIM"
echo ""
echo "TTT:"
echo "  Mode: $TTT_MODE"
echo "  Steps: $TTT_STEPS"
echo "  LR: $TTT_LR"
echo ""
echo "=========================================="
echo ""

# =============================================================================
# Launch Training
# =============================================================================
python experiments/train_ttt_ez.py \
    --config "experiments/configs/ttt_ez_config.py:${CONFIG_VARIANT}" \
    --oxedata_config "$DATA_CONFIG" \
    --name "$NAME" \
    --project "$PROJECT" \
    \
    `# Training schedule` \
    --config.batch_size "$BATCH_SIZE" \
    --config.num_steps "$NUM_STEPS" \
    --config.log_interval "$LOG_INTERVAL" \
    --config.eval_interval "$EVAL_INTERVAL" \
    --config.save_interval "$SAVE_INTERVAL" \
    --config.seed "$SEED" \
    --config.save_dir "$SAVE_DIR" \
    \
    `# SimSiam loss weight` \
    --config.agent_kwargs.lambda_sim "$LAMBDA_SIM" \
    \
    `# TTT configuration` \
    --config.ttt_mode "$TTT_MODE" \
    --config.ttt_steps "$TTT_STEPS" \
    --config.ttt_lr "$TTT_LR" \
    \
    `# Data configuration` \
    --oxedata_config.batch_size "$BATCH_SIZE" \
    --oxedata_config.oxe_kwargs.data_dir "$DATA_DIR" \
    --oxedata_config.oxe_kwargs.data_mix "$DATA_MIX"

echo ""
echo "Training complete!"