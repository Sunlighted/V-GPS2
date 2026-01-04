#!/bin/bash
# TTT-Predict Training on TPU with next-state prediction objective

# Restrict visible TPU cores to just one for speed benchmarking
# export TPU_VISIBLE_DEVICES=0,1,2,3

# Verify TPU is available
python -c "import jax; print(f'TPU devices: {jax.devices()}'); assert len(jax.devices()) > 0"

# Paths for TPU
export HF_HOME=/home/yruan/huggingface_cache
mkdir -p $HF_HOME

# Stream data directly from GCS
data_dir=/data/Chenyang/OXE_download

# Save checkpoints to local disk
save_dir=/data/Chenyang/value_learning/V-GPS/save
mkdir -p $save_dir

# Training config
PROJECT=VGPS
data_mix=fractal
discount=0.98

# Transition-level batch size
batch_size=512

# TTT-Predict adaptation params
ttt_adapt_steps=1
ttt_adapt_lr=0.1
ttt_adapt_reset=False
lambda_self=0.5
projection_dim=256
projection_num_layers=2
projection_hidden_dim=128
adapt_during_training=False

NAME=TTT_Predict_CalQL_${data_mix}_B${batch_size}

python experiments/train_ttt_predict.py \
    --config experiments/configs/ttt_predict_config.py:ttt_predict_calql \
    --oxedata_config experiments/configs/data_config.py \
    --name $NAME \
    --project $PROJECT \
    --config.num_steps 1000000 \
    --config.eval_interval 20000 \
    --config.save_dir $save_dir \
    --config.batch_size $batch_size \
    --config.lambda_self $lambda_self \
    --config.ttt_adapt_lr $ttt_adapt_lr \
    --config.projection_hidden_dim $projection_hidden_dim \
    --config.projection_num_layers $projection_num_layers \
    --config.agent_kwargs.projection_dim $projection_dim \
    --config.ttt_adapt_steps $ttt_adapt_steps \
    --config.ttt_adapt_reset $ttt_adapt_reset \
    --oxedata_config.oxe_kwargs.data_dir $data_dir \
    --oxedata_config.oxe_kwargs.data_mix $data_mix \
    --oxedata_config.oxe_kwargs.discount $discount \
    --oxedata_config.batch_size $batch_size