export HF_HOME=$SCRATCH/huggingface_cache
export TF_CPP_MIN_LOG_LEVEL=2
data_dir=/home/yifanr/projects/def-rhinehar/RT
save_dir=/home/yifanr/projects/def-rhinehar/V-GPS/checkpoints
PROJECT=VGPS_TTT
data_mix=fractal
discount=0.98
traj_len=120  # Trajectory length
octo_chunk=16  # Max frames per OCTO encoder batch
ttt_adapt_mode=windowed
ttt_adapt_window=8
ttt_adapt_steps=5
ttt_adapt_reset=True

NAME=TTT_CalQL_${data_mix}_traj${traj_len}

python experiments/train_ttt.py \
  --config experiments/configs/ttt_train_config.py:ttt_calql \
  --oxedata_config experiments/configs/ttt_data_config.py \
  --name $NAME \
  --project $PROJECT \
  --config.num_steps 1000000 \
  --config.save_dir $save_dir \
  --oxedata_config.oxe_kwargs.data_dir $data_dir \
  --oxedata_config.oxe_kwargs.data_mix $data_mix \
  --oxedata_config.oxe_kwargs.discount $discount \
  --oxedata_config.traj_transform_kwargs.window_size $traj_len \
  --config.octo_max_frames_per_batch $octo_chunk \
  --config.ttt_adapt_mode $ttt_adapt_mode \
  --config.ttt_adapt_window $ttt_adapt_window \
  --config.ttt_adapt_steps $ttt_adapt_steps \
  --config.ttt_adapt_reset $ttt_adapt_reset