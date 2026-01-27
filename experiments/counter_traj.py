from __future__ import annotations

import json
import math
import os
# 设置 GPU 即使在 import tensorflow 前也不可见，防止占用
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import collections

# Ensure repo root is importable
# 假设脚本位于 octo/scripts/ 或类似位置，根据实际文件位置可能需要调整
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OCTO_ROOT = REPO_ROOT / "octo"
if str(OCTO_ROOT) not in sys.path:
    sys.path.insert(0, str(OCTO_ROOT))

# 即使不使用 GPU，这些 import 也保留以防依赖检查
import flax.linen as nn
import jax
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from ml_collections import config_flags

# 确保这些模块在你的环境中可以被正确 import
from octo.data.dataset import make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
# 移除未使用的模型引用，防止报错
# from octo.model.octo_model import OctoModel
# from octo.model.octo_module import OctoModule
# from jaxrl_m.data.text_processing import text_processors

FLAGS = flags.FLAGS

# Model flags
flags.DEFINE_string("output_dir", "./", "Directory where stats are stored.")

# Data processing flags
flags.DEFINE_integer("batch_size", 64, "Batch size (not strictly used for stats but kept for config).")
flags.DEFINE_bool("overwrite", False, "If True, delete output_dir before processing.")
flags.DEFINE_string("split", "train", "Dataset split: train or val")

# Data config
config_flags.DEFINE_config_file(
    "oxedata_config",
    "./experiments/configs/data_config.py",
    "File path to the OXE data configuration.",
    lock_config=False,
)

# Data overrides
flags.DEFINE_string("data_dir", "/data/Chenyang/OXE_download", "Override data directory in config.")
flags.DEFINE_string("data_mix", "fractal", "Override data mix in config (e.g., 'bridge', 'bridge_fractal').")
flags.DEFINE_string("dataset_name", None, "Filter to specific dataset(s), comma-separated.")

def main(_):
    # 1. 设置仅使用 CPU
    tf.config.set_visible_devices([], "GPU")
    logging.set_verbosity(logging.INFO)

    # 2. 准备输出目录
    output_dir = Path(FLAGS.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计文件的路径
    stats_file_path = output_dir / "trajectory_length_stats.txt"
    
    # 初始化文件头
    with open(stats_file_path, "w") as f:
        f.write("Trajectory Length Statistics (Sorted by Length Descending)\n")
        f.write("=======================================================\n\n")

    config = FLAGS.oxedata_config
    if FLAGS.data_dir:
        config["oxe_kwargs"]["data_dir"] = FLAGS.data_dir
    if FLAGS.data_mix:
        config["oxe_kwargs"]["data_mix"] = FLAGS.data_mix

    logging.info(f"Data config: {config}")

    if "oxe_kwargs" in config:
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(**config["oxe_kwargs"])
    else:
        raise ValueError("oxe_kwargs not found in config")

    # 3. 筛选数据集
    if FLAGS.dataset_name:
        dataset_names = [n.strip() for n in FLAGS.dataset_name.split(",")]
        filtered = [(kw, w) for kw, w in zip(dataset_kwargs_list, sample_weights)
                    if any(dn in kw.get("name", "") for dn in dataset_names)]
        if not filtered:
            raise ValueError(f"No datasets matched filter: {FLAGS.dataset_name}")
        dataset_kwargs_list, sample_weights = zip(*filtered)

    traj_transform_kwargs = dict(config.get("traj_transform_kwargs", {}))
    # 强制 window_size=1 以获取原始长度（取决于 Dataset 实现，如果 Dataset 已经切片，这里可能需要注意）
    traj_transform_kwargs["window_size"] = 1 
    frame_transform_kwargs = dict(config.get("frame_transform_kwargs", {}))

    for dataset_idx, dataset_kwargs in enumerate(dataset_kwargs_list):
        dataset_name = dataset_kwargs.get("name", f"dataset_{dataset_idx}")
        logging.info(f"Processing dataset {dataset_idx + 1}/{len(dataset_kwargs_list)}: {dataset_name}")
        
        try:
            # 创建数据集
            dataset = make_single_dataset(
                dataset_kwargs,
                train=(FLAGS.split == "train"),
                traj_transform_kwargs=traj_transform_kwargs,
                frame_transform_kwargs=frame_transform_kwargs,
            )
        except Exception as e:
            logging.warning(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # 4. 初始化计数器
        length_counter = collections.Counter()
        
        logging.info(f"Computing statistics for {dataset_name}...")

        # 尝试获取数据集大小用于进度条（如果未知则为 None）
        cardinality = dataset.cardinality().numpy()
        total_steps = cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None

        # 使用 as_numpy_iterator() 通常比直接 iter(dataset) 更安全且兼容性更好
        iterator = dataset.as_numpy_iterator()

        for traj in tqdm.tqdm(iterator, total=total_steps, desc=f"Stats {dataset_name}"):
            # 获取轨迹长度
            traj_len = 0
            if 'action' in traj:
                traj_len = int(traj['action'].shape[0])
            elif 'observations' in traj:
                # 如果没有 action，取 observation 中第一个键的长度
                # 兼容字典结构
                first_key = next(iter(traj['observations']))
                traj_len = int(traj['observations'][first_key].shape[0])
            
            length_counter[traj_len] += 1

        # 5. 排序并写入文件
        # 如果 dataset 为空，length_counter 为空，处理一下
        if not length_counter:
            logging.warning(f"Dataset {dataset_name} yielded no trajectories.")
            continue

        # 按照轨迹长度从大到小排序
        sorted_stats = sorted(length_counter.items(), key=lambda x: x[0], reverse=True)
        
        # 实时追加写入文件
        with open(stats_file_path, "a") as f:
            f.write(f"Dataset: {dataset_name}\n")
            
            # 计算统计量
            all_lengths = []
            for l, c in length_counter.items():
                all_lengths.extend([l] * c)
            
            total_trajs_count = len(all_lengths)
            f.write(f"Total Trajectories: {total_trajs_count}\n")

            if all_lengths:
                f.write(f"Min: {min(all_lengths)}, Max: {max(all_lengths)}, Mean: {np.mean(all_lengths):.2f}\n")
            
            f.write("-" * 35 + "\n")
            f.write(f"{'Length':<10} | {'Count':<10}\n")
            f.write("-" * 35 + "\n")
            
            for length, count in sorted_stats:
                f.write(f"{length:<10} | {count:<10}\n")
            
            f.write("\n" + "="*50 + "\n\n")

    logging.info(f"All done. Statistics saved to {stats_file_path}")

if __name__ == "__main__":
    app.run(main)