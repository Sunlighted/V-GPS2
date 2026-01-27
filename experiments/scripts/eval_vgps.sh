export CUDA_VISIBLE_DEVICES=0
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
# widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube

for task_name in google_robot_pick_coke_can google_robot_move_near widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube # google_robot_close_top_drawer # google_robot_open_top_drawer
    do
    for seed in 0 1 2 3 4
        do
        python experiments/eval_vgps.py \
        --seed=$seed \
        --model_name=$model_name \
        --task_name=$task_name \
        --use_vgps=True \
        --vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/dyn_loss_srd/checkpoint_500000" \
        --num_samples=50 \
        --action_temp=1.0 \
        --add_actions=False \
        --num_eval_episodes=100 \
        --pretrain_method_name="vgps"
    done
done

# --vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_only-bridge_20251124_162649/checkpoint_500000" \
# --vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_only-fractal_20251119_194638/checkpoint_500000" \