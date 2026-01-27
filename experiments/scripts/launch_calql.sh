data_dir=/data/Chenyang/OXE_download # FILL IN
save_dir=/data/Chenyang/value_learning/V-GPS/save # FILL IN


PROJECT=VGPS
batch_size=512
data_mix=bridge
discount=0.98

NAME=VGPS_CalQL_${data_mix}_b${batch_size}_both

python experiments/train.py \
    --config experiments/configs/train_config.py:lc_cql \
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
    --oxedata_config.oxe_kwargs.discount $discount \
