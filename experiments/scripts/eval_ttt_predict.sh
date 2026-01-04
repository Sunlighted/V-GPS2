export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Headless rendering setup (for servers without display)
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SAPIEN_RENDERER=egl

model_name=octo-small

# TTT adaptation settings
USE_TTT=True
TTT_STEPS=1
TTT_LR=0.01
TTT_RESET=False

# Checkpoint path
CHECKPOINT=/home/yruan/V-GPS/save/VGPS/TTT_Predict_CalQL_fractal_B256_vrkewc6v/checkpoint_500000

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
python experiments/eval_ttt_predict.py \
--seed=0 \
--model_name=$model_name \
--task_name=$task_name \
--use_ttt=$USE_TTT \
--ttt_steps=$TTT_STEPS \
--ttt_lr=$TTT_LR \
--ttt_reset=$TTT_RESET \
--checkpoint_path=$CHECKPOINT \
--num_samples=50 \
--action_temp=1.0 \
--num_eval_episodes=100
done
