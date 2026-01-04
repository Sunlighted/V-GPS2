#!/bin/bash
# V-GPS Training on TPU with Fractal dataset

# Verify TPU is available
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
batch_size=256
data_mix=fractal  # Changed from bridge_fractal to just fractal
discount=0.98
NAME=VGPS_CalQLFIX_fractal_b${batch_size}_tpu

# Optional: Set JAX config for TPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

cd /home/yruan/V-GPS

python experiments/train_embedding.py \
  --config experiments/configs/train_config.py:lc_cqlfix \
  --oxedata_config experiments/configs/data_config.py \
  --name $NAME \
  --project $PROJECT \
  --config.num_steps 500000 \
  --config.agent_kwargs.cql_alpha 5.0 \
  --config.agent_kwargs.use_calql=True \
  --config.save_dir $save_dir \
  --oxedata_config.batch_size $batch_size \
  --oxedata_config.oxe_kwargs.data_dir $data_dir \
  --oxedata_config.oxe_kwargs.data_mix $data_mix \
  --config.agent_kwargs.discount $discount \
  --oxedata_config.oxe_kwargs.discount $discount