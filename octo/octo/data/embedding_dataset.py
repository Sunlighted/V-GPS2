"""
Logic for loading pre-computed OXE embedding datasets.
"""

from functools import partial
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np
import tensorflow as tf

from octo.data.oxe.oxe_dataset_configs import OXE_EMBEDDING_CONFIGS
from octo.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from octo.data.utils.data_utils import NormalizationType, normalize_action_and_proprio, tree_map


def _parse_embedding_example(serialized_example: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parse a single TFRecord example from an embedding dataset."""
    feature_description = {
        "steps/observation/embedding": tf.io.FixedLenFeature([], tf.string),
        "steps/next_observation/embedding": tf.io.FixedLenFeature([], tf.string),
        "steps/action": tf.io.FixedLenFeature([], tf.string),
        "steps/language_instruction": tf.io.FixedLenFeature([], tf.string),
        "steps/language_embedding": tf.io.FixedLenFeature([], tf.string),
        "steps/reward": tf.io.FixedLenFeature([], tf.string),
        "steps/td_mask": tf.io.FixedLenFeature([], tf.string),
        "steps/mc_return": tf.io.FixedLenFeature([], tf.string),
        "steps/is_first": tf.io.FixedLenFeature([], tf.string),
        "steps/is_last": tf.io.FixedLenFeature([], tf.string),
        "steps/is_terminal": tf.io.FixedLenFeature([], tf.string),
        "steps/discount": tf.io.FixedLenFeature([], tf.string),
        "episode_metadata/episode_id": tf.io.FixedLenFeature([], tf.int64),
        "episode_metadata/trajectory_length": tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(serialized_example, feature_description)

    # Deserialize tensors
    obs_embedding = tf.io.parse_tensor(parsed["steps/observation/embedding"], out_type=tf.float32)
    next_obs_embedding = tf.io.parse_tensor(parsed["steps/next_observation/embedding"], out_type=tf.float32)
    action = tf.io.parse_tensor(parsed["steps/action"], out_type=tf.float32)
    
    # --- Fix for Language Embedding Dimensions ---
    # Raw tensor from file might be (1, 16, 768) or (16, 768)
    language_embedding = tf.io.parse_tensor(parsed["steps/language_embedding"], out_type=tf.float32) 
    
    # Ensure it is (16, 768) by flattening the batch dimension if it exists
    lang_shape = tf.shape(language_embedding)
    lang_content_shape = lang_shape[-2:] # Take last two dims (16, 768)
    language_embedding_squeezed = tf.reshape(language_embedding, lang_content_shape)
    # ---------------------------------------------

    reward = tf.io.parse_tensor(parsed["steps/reward"], out_type=tf.float32)
    td_mask = tf.io.parse_tensor(parsed["steps/td_mask"], out_type=tf.float32)
    mc_return = tf.io.parse_tensor(parsed["steps/mc_return"], out_type=tf.float32)
    discount = tf.io.parse_tensor(parsed["steps/discount"], out_type=tf.float32)
    is_first = tf.io.parse_tensor(parsed["steps/is_first"], out_type=tf.bool)
    is_last = tf.io.parse_tensor(parsed["steps/is_last"], out_type=tf.bool)
    is_terminal = tf.io.parse_tensor(parsed["steps/is_terminal"], out_type=tf.bool)

    traj_len = tf.shape(action)[0]
    language_instruction = parsed["steps/language_instruction"]

    episode_id = parsed["episode_metadata/episode_id"]

    return {
        "observation": {
            "embedding": obs_embedding,
            "timestep": tf.range(traj_len),
        },
        "next_observation": {
            "embedding": next_obs_embedding,
        },
        "action": action,
        "task": {
            "language_instruction": tf.fill([traj_len], language_instruction),
            "language_embedding": tf.repeat(language_embedding_squeezed[None, ...], traj_len, axis=0)
        },
        "reward": reward,
        "td_mask": td_mask,
        "mc_return": mc_return,
        "discount": discount,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "episode_id": tf.fill([traj_len], episode_id),
    }


def _load_metadata(data_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Load metadata.json for an embedding dataset."""
    metadata_path = data_dir / dataset_name / "1.0.0" / "metadata.json"
    if not metadata_path.exists():
        logging.warning(f"Metadata file not found: {metadata_path}")
        return {}

    with open(metadata_path, "r") as f:
        return json.load(f)


def _get_tfrecord_files(data_dir: Path, dataset_name: str, split: str) -> list:
    """Get list of TFRecord files for an embedding dataset."""
    tfrecord_dir = data_dir / dataset_name / "1.0.0"
    pattern = f"{dataset_name}-{split}.tfrecord-*"

    files = sorted(tfrecord_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found matching pattern: {tfrecord_dir / pattern}"
        )

    return [str(f) for f in files]


def make_embedding_dataset(
    name: str,
    data_dir: str,
    *,
    train: bool,
    shuffle: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
    dataset_statistics: Optional[Union[dict, str]] = None,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    skip_unlabeled: bool = True,
    skip_norm: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    discount: float = 0.98,
    num_final_repeat: int = 3,
) -> Tuple[tf.data.Dataset, dict]:
    """Load a pre-computed embedding dataset."""
    if name not in OXE_EMBEDDING_CONFIGS:
        raise ValueError(
            f"Unknown embedding dataset: {name}. "
            f"Available: {list(OXE_EMBEDDING_CONFIGS.keys())}"
        )

    config = OXE_EMBEDDING_CONFIGS[name]
    data_path = Path(data_dir)
    split = "train" if train else "val"

    tfrecord_files = _get_tfrecord_files(data_path, name, split)
    logging.info(f"Found {len(tfrecord_files)} TFRecord files for {name}/{split}")

    metadata = _load_metadata(data_path, name)

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        if "action" in metadata:
            action_stats = metadata["action"]
            dataset_statistics = {
                "action": {
                    "mean": np.array(action_stats.get("mean", [0.0] * config["action_dim"])),
                    "std": np.array(action_stats.get("std", [1.0] * config["action_dim"])),
                    "min": np.array(action_stats.get("min", [-1.0] * config["action_dim"])),
                    "max": np.array(action_stats.get("max", [1.0] * config["action_dim"])),
                    "p01": np.array(action_stats.get("p01", [-1.0] * config["action_dim"])),
                    "p99": np.array(action_stats.get("p99", [1.0] * config["action_dim"])),
                    "p001": np.array(action_stats.get("p001", action_stats.get("p01", [-1.0] * config["action_dim"]))),
                    "p999": np.array(action_stats.get("p999", action_stats.get("p99", [1.0] * config["action_dim"]))),
                },
                "num_transitions": metadata.get("num_transitions", 0),
                "num_trajectories": metadata.get("num_trajectories", 0),
            }
        else:
            dataset_statistics = {
                "action": {
                    "mean": np.zeros(config["action_dim"]),
                    "std": np.ones(config["action_dim"]),
                    "min": np.full(config["action_dim"], -1.0),
                    "max": np.full(config["action_dim"], 1.0),
                    "p01": np.full(config["action_dim"], -1.0),
                    "p99": np.full(config["action_dim"], 1.0),
                    "p001": np.full(config["action_dim"], -1.0),
                    "p999": np.full(config["action_dim"], 1.0),
                },
                "num_transitions": 0,
                "num_trajectories": 0,
            }
            logging.warning(f"No action statistics found for {name}, using default values")

    dataset_statistics = tree_map(np.array, dataset_statistics)

    if "bridge" in name.lower():
        dataset_statistics["action"]["min"] = np.array(
            [-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0]
        )
        dataset_statistics["action"]["max"] = np.array(
            [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]
        )

    if action_normalization_mask is not None:
        if len(action_normalization_mask) != config["action_dim"]:
            raise ValueError(
                f"Length of action_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({config['action_dim']})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    if shuffle:
        np.random.shuffle(tfrecord_files)

    # 1. Create Standard TF Dataset
    tf_dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=num_parallel_reads,
    )

    # 2. Map parsing function
    tf_dataset = tf_dataset.map(
        _parse_embedding_example,
        num_parallel_calls=num_parallel_calls,
    )

    # 3. Calculate Returns (Standard TF map)
    def calculate_returns(traj):
        traj_len = tf.shape(traj["action"])[0]
        num_pos = tf.minimum(num_final_repeat, traj_len)
        reward = tf.concat(
             [-tf.ones(traj_len - num_pos, dtype=tf.float32), tf.zeros(num_pos, dtype=tf.float32)], axis=0
        )
        td_mask = tf.concat(
            [tf.ones(traj_len - num_pos, dtype=tf.float32), tf.zeros(num_pos, dtype=tf.float32)], axis=0
        )
        mc_return = tf.scan(
            lambda prev_return, x: x[0] + discount * prev_return * x[1],
            [reward, td_mask],
            initializer=0.0,
            reverse=True
        )
        traj["reward"] = reward
        traj["td_mask"] = td_mask
        traj["mc_return"] = mc_return
        return traj
    
    tf_dataset = tf_dataset.map(calculate_returns, num_parallel_calls=num_parallel_calls)

    def add_dataset_name(traj):
        traj_len = tf.shape(traj["action"])[0]
        traj["dataset_name"] = tf.fill([traj_len], name)
        return traj

    tf_dataset = tf_dataset.map(add_dataset_name, num_parallel_calls=num_parallel_calls)

    tf_dataset = tf_dataset.filter(lambda x: tf.shape(x["action"])[0] > 0)

    if skip_unlabeled:
        if "language_instruction" not in tf_dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")
        tf_dataset = tf_dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    if not skip_norm:
        tf_dataset = tf_dataset.map(
            partial(
                normalize_action_and_proprio,
                metadata=dataset_statistics,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls=num_parallel_calls,
        )
    else:
        logging.warning("Dataset normalization turned off -- set skip_norm=False to apply normalization.")

    return tf_dataset, dataset_statistics


def make_embedding_dataset_kwargs(
    name: str,
    data_dir: str,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
) -> Dict[str, Any]:
    """Generate kwargs for make_embedding_dataset."""
    if name not in OXE_EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown embedding dataset: {name}")

    config = OXE_EMBEDDING_CONFIGS[name]
    action_normalization_mask = [True] * (config["action_dim"] - 1) + [False]

    return {
        "name": name,
        "data_dir": data_dir,
        "action_proprio_normalization_type": action_proprio_normalization_type,
        "action_normalization_mask": action_normalization_mask,
    }


def make_embedding_dataset_mix(
    name: str,
    data_dir: str,
    *,
    train: bool,
    shuffle: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
    dataset_statistics: Optional[Union[dict, str]] = None,
    skip_unlabeled: bool = True,
    skip_norm: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    discount: float = 0.98,
    num_final_repeat: int = 3,
) -> Tuple[tf.data.Dataset, dict]:
    """Load a dataset mix of pre-computed embedding datasets."""
    if name not in OXE_NAMED_MIXES:
        raise ValueError(f"Unknown dataset mix: {name}")

    mix_spec = OXE_NAMED_MIXES[name]
    logging.info(f"Loading embedding dataset mix '{name}' with {len(mix_spec)} components")

    datasets = []
    weights = []
    combined_stats = None

    for dataset_name, weight in mix_spec:
        if dataset_name not in OXE_EMBEDDING_CONFIGS:
            raise ValueError(f"Dataset '{dataset_name}' not in OXE_EMBEDDING_CONFIGS.")

        logging.info(f"  Loading component: {dataset_name} (weight={weight})")

        # Returns a dl.DLataset now (checked)
        component_dataset, component_stats = make_embedding_dataset(
            name=dataset_name,
            data_dir=data_dir,
            train=train,
            shuffle=shuffle,
            action_proprio_normalization_type=action_proprio_normalization_type,
            dataset_statistics=dataset_statistics,
            skip_unlabeled=skip_unlabeled,
            skip_norm=True, # Skip norm inside components, apply globally
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            discount=discount,
            num_final_repeat=num_final_repeat,
        )

        datasets.append(component_dataset)
        weights.append(weight)

        if combined_stats is None:
            combined_stats = component_stats

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    logging.info(f"  Sampling probabilities: {probabilities}")

    # Use standard tf.data.Dataset.sample_from_datasets since we have tf.data.Dataset objects
    mixed_dataset = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=probabilities,
        seed=None if shuffle else 42,
        stop_on_empty_dataset=False,
    )

    if not skip_norm:
        mixed_dataset = mixed_dataset.map(
            partial(
                normalize_action_and_proprio,
                metadata=combined_stats,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls=num_parallel_calls,
        )
    else:
        logging.warning("Dataset normalization turned off.")

    return mixed_dataset, combined_stats


def is_embedding_dataset_mix(name: str) -> bool:
    if name not in OXE_NAMED_MIXES:
        return False
    mix_spec = OXE_NAMED_MIXES[name]
    return all(ds_name in OXE_EMBEDDING_CONFIGS for ds_name, _ in mix_spec)