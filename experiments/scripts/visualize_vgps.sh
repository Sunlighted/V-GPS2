#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

model_name=octo-small

# WidowX tasks: widowx_put_eggplant_in_basket, widowx_spoon_on_towel, widowx_carrot_on_plate, widowx_stack_cube
# Google Robot tasks: google_robot_pick_coke_can, google_robot_move_near

for task_name in widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube
do
python experiments/visualize_vgps_actions.py \
--seed=0 \
--model_name=$model_name \
--task_name=$task_name \
--vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_20251115_054407/checkpoint_500000" \
--num_samples=50 \
--action_temp=1.0 \
--num_eval_episodes=20 \
--max_timesteps=120 \
--show_gripper=True \
--show_rotation=False
done
