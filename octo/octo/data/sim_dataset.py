"""Utility loaders for simulated V-GPS datasets stored as TFRecords."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import dlimp as dl
import numpy as np
import tensorflow as tf
from absl import logging

from octo.data.dataset import apply_frame_transforms, apply_trajectory_transforms

_SIM_OBS_IMAGE_KEY = "observation.image_primary"
_SIM_OBS_TIMESTEP_KEY = "observation.timestep"
_SIM_TASK_LANGUAGE_KEY = "task.language_instruction"


def _unflatten_sim_traj(traj: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Convert flattened TFRecord features back to the nested Octo trajectory schema."""
    observation = {
        "image_primary": tf.cast(traj.pop(_SIM_OBS_IMAGE_KEY), tf.uint8),
        "timestep": tf.cast(traj.pop(_SIM_OBS_TIMESTEP_KEY), tf.int32),
    }
    task = {
        "language_instruction": tf.cast(
            traj.pop(_SIM_TASK_LANGUAGE_KEY), tf.string
        )
    }
    traj["observation"] = observation
    traj["task"] = task
    traj["action"] = tf.cast(traj["action"], tf.float32)
    if "action_pad_mask" not in traj:
        traj["action_pad_mask"] = tf.ones_like(traj["action"], dtype=tf.bool)
    return traj


def _load_metadata_if_present(data_dir: str) -> Optional[Dict]:
    meta_path = os.path.join(data_dir, "metadata.json")
    if tf.io.gfile.exists(meta_path):
        with tf.io.gfile.GFile(meta_path, "r") as f:
            return json.load(f)
    return None


def make_simulated_dataset(
    data_dir: str,
    *,
    train: bool,
    traj_transform_kwargs: Dict,
    frame_transform_kwargs: Dict,
    shuffle_buffer_size: int,
    batch_size: Optional[int],
    num_parallel_calls: int = tf.data.AUTOTUNE,
    num_parallel_reads: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Create a DLataset from TFRecords collected via collect_vgps_data.py."""
    records_dir = data_dir
    if tf.io.gfile.isdir(data_dir):
        pattern = os.path.join(data_dir, "*.tfrecord")
        if not tf.io.gfile.glob(pattern):
            candidate = os.path.join(data_dir, "tfrecords")
            if tf.io.gfile.isdir(candidate):
                records_dir = candidate

    dataset = dl.DLataset.from_tfrecords(
        records_dir,
        shuffle=train,
        num_parallel_reads=num_parallel_reads,
    )
    dataset = dataset.traj_map(_unflatten_sim_traj, num_parallel_calls)
    dataset = apply_trajectory_transforms(
        dataset.repeat(),
        **traj_transform_kwargs,
        num_parallel_calls=num_parallel_calls,
        train=train,
    ).flatten(num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = apply_frame_transforms(
        dataset,
        **frame_transform_kwargs,
        train=train,
    )
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.with_ram_budget(1)
    dataset = dataset.ignore_errors(log_warning=True)

    metadata = _load_metadata_if_present(data_dir)
    if metadata is None and records_dir != data_dir:
        metadata = _load_metadata_if_present(os.path.dirname(records_dir))
    if metadata is not None:
        dataset.dataset_statistics = {
            "num_transitions": metadata.get("num_transitions", 0),
            "num_trajectories": metadata.get("num_episodes", 0),
            "action": {
                "mean": np.array(metadata.get("action_mean", []), dtype=np.float32),
                "std": np.array(metadata.get("action_std", []), dtype=np.float32),
            },
        }
        logging.info(
            "Loaded simulated dataset metadata from %s (episodes=%s transitions=%s)",
            data_dir,
            metadata.get("num_episodes", "?"),
            metadata.get("num_transitions", "?"),
        )
    else:
        logging.warning("No metadata.json found in %s; proceeding without statistics", data_dir)

    return dataset
