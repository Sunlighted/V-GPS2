#!/usr/bin/env bash
set -euo pipefail

# 环境变量
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 定义要运行的6个任务列表
TASKS=(
    # "widowx_put_eggplant_in_basket"
    # "widowx_spoon_on_towel"
    # "widowx_carrot_on_plate"
    "widowx_stack_cube"
    # "google_robot_pick_coke_can"
    # "google_robot_move_near"
)

# 基础配置 (支持命令行参数覆盖，如果未提供则使用默认值)
BASE_OUT_DIR=${1:-sim_data}
EPISODES=${2:-3000}

echo "开始批量运行 ${#TASKS[@]} 个任务..."

# 遍历任务列表
for task in "${TASKS[@]}"; do
    echo "----------------------------------------------------"
    echo "正在处理任务: ${task}"
    echo "----------------------------------------------------"

    python experiments/collect_vgps_data.py \
      --task_name="${task}" \
      --output_dir="${BASE_OUT_DIR}/${task}" \
      --max_episodes="${EPISODES}" \
      --model_name=octo-small \
      --use_vgps=False \
      --vgps_checkpoint="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_20251115_054407/checkpoint_500000" \
      --num_samples=50 \
      --action_temp=0.0 \
      --episodes_per_shard=50 \
      --record_videos=False \
      --save_npz=True
      
    echo "任务 ${task} 完成。"
done

echo "所有任务已完成！"