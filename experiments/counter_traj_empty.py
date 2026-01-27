from __future__ import annotations

import json
import math
import os
# 设置 GPU 即使在 import tensorflow 前也不可见，防止占用
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import collections

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OCTO_ROOT = REPO_ROOT / "octo"
if str(OCTO_ROOT) not in sys.path:
    sys.path.insert(0, str(OCTO_ROOT))

import flax.linen as nn
import jax
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from ml_collections import config_flags

from octo.data.dataset import make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights

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
flags.DEFINE_string("data_mix", "fractal", "Override data mix in config.")
flags.DEFINE_string("dataset_name", None, "Filter to specific dataset(s), comma-separated.")

def get_instruction_string(traj) -> str:
    """Helper to safely extract and decode instruction string from trajectory."""
    inst = ""
    # 常见的存放位置：traj['task']['language_instruction']
    if 'task' in traj and 'language_instruction' in traj['task']:
        inst = traj['task']['language_instruction']
    elif 'language_instruction' in traj:
        inst = traj['language_instruction']
    else:
        # 如果找不到指令字段，视为空
        return ""

    # 处理 numpy array 包裹的情况
    if isinstance(inst, (np.ndarray, np.generic)):
        # 如果是标量数组
        if inst.ndim == 0 or inst.size == 1:
            inst = inst.item()
        # 如果是序列（每个timestep都有），取第一个
        elif inst.ndim > 0:
            inst = inst[0]

    # 处理 bytes 转 string
    if isinstance(inst, bytes):
        try:
            inst = inst.decode('utf-8')
        except:
            inst = str(inst)
            
    return str(inst).strip()

def main(_):
    tf.config.set_visible_devices([], "GPU")
    logging.set_verbosity(logging.INFO)

    output_dir = Path(FLAGS.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 修改输出文件名
    stats_file_path = output_dir / "empty_instruction_stats.txt"
    
    with open(stats_file_path, "w") as f:
        f.write("Empty Instruction Trajectory Length Stats\n")
        f.write("=========================================\n\n")

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

    if FLAGS.dataset_name:
        dataset_names = [n.strip() for n in FLAGS.dataset_name.split(",")]
        filtered = [(kw, w) for kw, w in zip(dataset_kwargs_list, sample_weights)
                    if any(dn in kw.get("name", "") for dn in dataset_names)]
        if not filtered:
            raise ValueError(f"No datasets matched filter: {FLAGS.dataset_name}")
        dataset_kwargs_list, sample_weights = zip(*filtered)

    traj_transform_kwargs = dict(config.get("traj_transform_kwargs", {}))
    traj_transform_kwargs["window_size"] = 1 
    frame_transform_kwargs = dict(config.get("frame_transform_kwargs", {}))

    for dataset_idx, dataset_kwargs in enumerate(dataset_kwargs_list):
        dataset_name = dataset_kwargs.get("name", f"dataset_{dataset_idx}")
        logging.info(f"Processing dataset {dataset_idx + 1}/{len(dataset_kwargs_list)}: {dataset_name}")
        
        try:
            dataset = make_single_dataset(
                dataset_kwargs,
                train=(FLAGS.split == "train"),
                traj_transform_kwargs=traj_transform_kwargs,
                frame_transform_kwargs=frame_transform_kwargs,
            )
        except Exception as e:
            logging.warning(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # 计数器
        empty_length_counter = collections.Counter()
        total_trajs_scanned = 0
        empty_trajs_found = 0
        
        logging.info(f"Scanning for empty instructions in {dataset_name}...")

        cardinality = dataset.cardinality().numpy()
        total_steps = cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None

        iterator = dataset.as_numpy_iterator()

        for traj in tqdm.tqdm(iterator, total=total_steps, desc=f"Scanning {dataset_name}"):
            total_trajs_scanned += 1
            
            # 1. 获取指令文本
            inst_str = get_instruction_string(traj)
            
            # 2. 判断是否为空 (空字符串或只有空格)
            if not inst_str:
                empty_trajs_found += 1
                
                # 3. 计算长度
                traj_len = 0
                if 'action' in traj:
                    traj_len = int(traj['action'].shape[0])
                elif 'observations' in traj:
                    first_key = next(iter(traj['observations']))
                    traj_len = int(traj['observations'][first_key].shape[0])
                
                empty_length_counter[traj_len] += 1

        if empty_trajs_found == 0:
            logging.info(f"No empty instructions found in {dataset_name}.")
            with open(stats_file_path, "a") as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Scanned {total_trajs_scanned} trajectories. NO empty instructions found.\n")
                f.write("\n" + "="*50 + "\n\n")
            continue

        # 排序并写入
        sorted_stats = sorted(empty_length_counter.items(), key=lambda x: x[0], reverse=True)
        
        with open(stats_file_path, "a") as f:
            f.write(f"Dataset: {dataset_name}\n")
            
            # 计算统计量
            all_lengths = []
            for l, c in empty_length_counter.items():
                all_lengths.extend([l] * c)
            
            f.write(f"Total Scanned: {total_trajs_scanned}\n")
            f.write(f"Empty Instructions Found: {empty_trajs_found} ({empty_trajs_found/total_trajs_scanned*100:.2f}%)\n")

            if all_lengths:
                f.write(f"Min: {min(all_lengths)}, Max: {max(all_lengths)}, Mean: {np.mean(all_lengths):.2f}\n")
            
            f.write("-" * 35 + "\n")
            f.write(f"{'Length':<10} | {'Count':<10}\n")
            f.write("-" * 35 + "\n")
            
            for length, count in sorted_stats:
                f.write(f"{length:<10} | {count:<10}\n")
            
            f.write("\n" + "="*50 + "\n\n")

    logging.info(f"Done. Check {stats_file_path} for results.")

if __name__ == "__main__":
    app.run(main)