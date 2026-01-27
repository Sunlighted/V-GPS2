#!/bin/bash
# Launch script for training with pre-computed embeddings

# Configuration - FILL IN THESE PATHS
embedding_data_dir=/data/Chenyang/OXE_embedding  # Directory containing embedding datasets
save_dir=/data/Chenyang/value_learning/V-GPS/save  # Directory to save checkpoints

# Experiment settings
PROJECT=VGPS
batch_size=512
dataset_name=bridge_fractal_embedding  # Must be in OXE_EMBEDDING_CONFIGS
discount=0.98

NAME=VGPS_CalQL_Embedding_${dataset_name}_b${batch_size}

python experiments/train_from_embedding.py \
    --config experiments/configs/cqlfix_embedding_config.py \
    --embedding_data_config experiments/configs/embedding_data_config.py \
    --name $NAME \
    --project $PROJECT \
    --config.num_steps 500000 \
    --config.batch_size $batch_size \
    --config.save_dir $save_dir \
    --config.agent_kwargs.cql_alpha 5.0 \
    --config.agent_kwargs.use_calql=True \
    --config.agent_kwargs.discount $discount \
    --embedding_data_config.dataset_name $dataset_name \
    --embedding_data_config.data_dir $embedding_data_dir \
    --embedding_data_config.skip_unlabeled=True \
    --embedding_data_config.shuffle_buffer_size 50000
