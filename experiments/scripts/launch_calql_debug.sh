data_dir="gs://leaf-lab-bucket-0/OXE" # FILL IN
save_dir="/home/chenyangcao/V-GPS/save" # FILL IN


PROJECT=VGPS
batch_size=16
data_mix=bridge_fractal
discount=0.98

NAME=VGPS_CalQL_${data_mix}_b${batch_size}_debug

python experiments/train_ttt_debug.py \
    --config experiments/configs/train_config_debug.py:lc_cql \
    --oxedata_config experiments/configs/data_config_ttte2e.py \
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
