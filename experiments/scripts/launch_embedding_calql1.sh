#!/bin/bash
# Launch script for training with pre-computed embeddings

# Configuration - FILL IN THESE PATHS
data_dir=/data/Chenyang/OXE_embedding_new  # Directory containing embedding datasets
save_dir=/data/Chenyang/value_learning/V-GPS/save  # Directory to save checkpoints

# Experiment settings
PROJECT=VGPS
batch_size=4096
data_mix=bridge_fractal_embedding  # Must be in OXE_EMBEDDING_CONFIGS
discount=0.98

NAME=VGPS_CalQL_Embedding_${data_mix}_b${batch_size}

python experiments/train_emb.py \
    --config experiments/configs/train_config.py:lc_cqlfix \
    --oxedata_config experiments/configs/emb_data_config.py \
    --name $NAME \
    --project $PROJECT \
    --config.num_steps 500000 \
    --config.batch_size $batch_size \
    --config.save_dir $save_dir \
    --config.agent_kwargs.cql_alpha 5.0 \
    --config.agent_kwargs.use_calql=True \
    --config.agent_kwargs.discount $discount \
    --oxedata_config.oxe_kwargs.data_dir $data_dir \
    --oxedata_config.oxe_kwargs.data_mix $data_mix \
    --oxedata_config.oxe_kwargs.discount $discount \
