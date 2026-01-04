#!/bin/bash
# V-GPS Training on TPU with Fractal dataset

# Restrict visible TPU cores to just one for speed benchmarking
export TPU_VISIBLE_DEVICES=0

# Verify TPU is available (should now report a single device)
python -c "import jax; print(f'TPU devices: {jax.devices()}'); assert len(jax.devices()) > 0"

# Paths for TPU
export HF_HOME=/home/yruan/huggingface_cache
mkdir -p $HF_HOME

# Stream data directly from GCS (fractal is already there)
data_dir=gs://gresearch/robotics

# Save checkpoints to local disk (or create GCS bucket for persistence)
save_dir=/home/yruan/V-GPS/save
mkdir -p $save_dir

# Training config
PROJECT=VGPS
data_mix=fractal
discount=0.98
traj_len=120  # Trajectory length
octo_chunk=1024  # Max frames per OCTO encoder batch
ttt_adapt_mode=windowed
ttt_adapt_window=8
ttt_adapt_steps=5
ttt_adapt_reset=False

# Number of trajectories per batch (B). Setable when launching.
batch_size=4

NAME=TTT_paper_${data_mix}_traj${traj_len}

# Append batch size to the run name for easier tracking
NAME=${NAME}_B${batch_size}

python experiments/train_ttt_paper.py \
  --config=experiments/configs/ttt_paper_config.py \
  --oxedata_config experiments/configs/ttt_data_config.py \
  --name $NAME \
  --project $PROJECT \
  --config.num_steps 1000000 \
  --config.eval_interval 20 \
  --config.log_interval 100 \
  --config.save_dir $save_dir \
  --oxedata_config.oxe_kwargs.data_dir $data_dir \
  --oxedata_config.oxe_kwargs.data_mix $data_mix \
  --oxedata_config.oxe_kwargs.discount $discount \
  --oxedata_config.batch_size $batch_size \