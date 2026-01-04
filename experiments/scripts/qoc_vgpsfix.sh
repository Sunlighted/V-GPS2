export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

model_name=octo-small

# WidowX tasks: widowx_put_eggplant_in_basket, widowx_spoon_on_towel, widowx_carrot_on_plate, widowx_stack_cube
# Google Robot tasks: google_robot_pick_coke_can, google_robot_move_near
# 'google_robot_pick_coke_can', 'google_robot_pick_horizontal_coke_can', 'google_robot_pick_vertical_coke_can', 
# 'google_robot_pick_standing_coke_can', 'google_robot_pick_object', 'google_robot_move_near_v0', 'google_robot_move_near_v1', 
# 'google_robot_move_near', 'google_robot_open_drawer', 'google_robot_open_top_drawer', 'google_robot_open_middle_drawer', 
# 'google_robot_open_bottom_drawer', 'google_robot_close_drawer', 'google_robot_close_top_drawer', 'google_robot_close_middle_drawer', 
# 'google_robot_close_bottom_drawer', 'google_robot_place_in_closed_drawer', 'google_robot_place_in_closed_top_drawer', 'google_robot_place_in_closed_middle_drawer', 
# 'google_robot_place_in_closed_bottom_drawer', 'google_robot_place_apple_in_closed_top_drawer', 'widowx_spoon_on_towel', 
# 'widowx_carrot_on_plate', 'widowx_stack_cube', 'widowx_put_eggplant_in_basket'

for task_name in google_robot widowx
do
python experiments/plot_qoc.py \
    --seed=0 \
    --config="experiments/configs/train_config.py:lc_cqlfix" \
    --oxedata_config="experiments/configs/data_config.py" \
    --model_name=$model_name \
    --task_name=$task_name \
    --use_vgps=True \
    --vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQLFIX_bridge_fractal_b256_octo-small_20251121_151543/checkpoint_500000" \
    --num_samples=50 \
    --action_temp=1.0 \
    --pretrain_method_name="vgpsfix"\
    --config.save_dir="/home/Chenyang/value_learning/V-GPS/save_qvoc" \
    --oxedata_config.batch_size=16 \
    --oxedata_config.oxe_kwargs.data_dir="/data/Chenyang/OXE_download" \
    --oxedata_config.oxe_kwargs.data_mix="bridge_fractal" \
    --config.agent_kwargs.discount=0.98 \
    --oxedata_config.oxe_kwargs.discount=0.98
done
