export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

model_name="octo-small"

# WidowX tasks: widowx_put_eggplant_in_basket, widowx_spoon_on_towel, widowx_carrot_on_plate, widowx_stack_cube
# Google Robot tasks: google_robot_pick_coke_can, google_robot_move_near

for task_name in widowx_put_eggplant_in_basket
do
python experiments/eval_vgps.py \
--seed=0 \
--model_name=$model_name \
--task_name=$task_name \
--use_vgps=False \
--num_eval_episodes=20 
done