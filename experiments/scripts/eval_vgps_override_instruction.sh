export CUDA_VISIBLE_DEVICES=7
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

for task_name in widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube google_robot_pick_coke_can google_robot_move_near
    do
    if [ "$task_name" == "widowx_put_eggplant_in_basket" ]; then
        instruction="put the apple into yellow basket"
    elif [ "$task_name" == "widowx_spoon_on_towel" ]; then
        instruction="place the potato on the towel"
    elif [ "$task_name" == "widowx_carrot_on_plate" ]; then
        instruction="place the carrot on the basket"
    elif [ "$task_name" == "widowx_stack_cube" ]; then
        instruction="pick the cube"
    elif [ "$task_name" == "google_robot_pick_coke_can" ]; then
        instruction="move the coke can"
    elif [ "$task_name" == "google_robot_move_near" ]; then
        instruction="pick the object near the plate"
    fi
    for seed in 1 2 3
        do
        python experiments/eval_vgps.py \
        --seed=$seed \
        --model_name=$model_name \
        --task_name=$task_name \
        --override_instruction="$instruction" \
        --use_vgps=True \
        --vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_20251115_054407/checkpoint_500000" \
        --num_samples=50 \
        --action_temp=1.0 \
        --num_eval_episodes=100 \
        --pretrain_method_name="vgps"
    done
done