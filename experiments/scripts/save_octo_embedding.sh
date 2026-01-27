python experiments/Octo_embedding_save.py \
    --encoder octo-small \
    --output_dir /data/Chenyang/OXE_embedding \
    --batch_size 64 \
    --oxedata_config experiments/configs/data_config.py \
    --data_dir /data/Chenyang/OXE_download \
    --data_mix bridge \
    --episodes_per_shard 50 \
    --split train