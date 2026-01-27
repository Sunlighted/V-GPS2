#!/usr/bin/env python3
"""
Save Octo embeddings from OXE datasets to TFRecord format.

Produces TFDS-compatible TFRecords that match the bridge_dataset structure,
with image observations replaced by Octo embeddings.

Usage:
    python experiments/Octo_embedding_save.py \
        --encoder=octo-small \
        --output_dir=logs/embeddings/bridge \
        --batch_size=128 \
        --oxedata_config=experiments/configs/data_config.py \
        --data_dir=/path/to/oxe_data \
        --data_mix=bridge
"""

from __future__ import annotations

import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from octo.model.octo_model import OctoModel
from octo.model.octo_module import OctoModule
from jaxrl_m.data.text_processing import text_processors

FLAGS = flags.FLAGS

# Model flags
flags.DEFINE_string("encoder", "octo-small", "Octo model name: octo-small or octo-base")
flags.DEFINE_string("output_dir", "logs/embeddings", "Directory where embeddings are stored.")

# Data processing flags
flags.DEFINE_integer("batch_size", 64, "Batch size for embedding generation.")
flags.DEFINE_integer("max_T", 600, "Maximum trajectory length.")
flags.DEFINE_integer("episodes_per_shard", 50, "Number of trajectories per TFRecord shard.")
flags.DEFINE_bool("overwrite", False, "If True, delete output_dir before processing.")
flags.DEFINE_string("split", "train", "Dataset split: train or val")

# Data config
config_flags.DEFINE_config_file(
    "oxedata_config",
    None,
    "File path to the OXE data configuration.",
    lock_config=False,
)

# Data overrides
flags.DEFINE_string("data_dir", None, "Override data directory in config.")
flags.DEFINE_string("data_mix", None, "Override data mix in config (e.g., 'bridge', 'bridge_fractal').")
flags.DEFINE_string("dataset_name", None, "Filter to specific dataset(s), comma-separated.")


class OctoEncoderModule(nn.Module):
    """Wraps Octo's transformer to extract embeddings from the action readout."""
    octo_module: OctoModule
    readout_name: str = "action"
    pool_type: str = "mean"

    @nn.compact
    def __call__(self, observations, tasks, timestep_pad_mask, train: bool = False):
        transformer_outputs = self.octo_module.octo_transformer(
            observations, tasks, timestep_pad_mask, train=train, verbose=False
        )
        tg = transformer_outputs[f"readout_{self.readout_name}"]
        if self.pool_type == "mean":
            emb = tg.tokens.mean(axis=-2)
        elif self.pool_type == "last_timestep":
            emb = tg.tokens[:, -1].mean(axis=-2)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        return emb


class OctoEmbeddingGenerator:
    """Generates embeddings using Octo model with optimized JAX JIT."""

    def __init__(self, model_type: str = "octo-small", image_size: int = 256):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # 加载官方推荐的 OctoModel
        if model_type in ["octo-base", "octo-small", "octo-base-1.5", "octo-small-1.5"]:
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.model = OctoModel.load_pretrained(self.model_type)
        else:
            raise NotImplementedError(f"{model_type} not supported yet.")

        self.image_size = image_size
        self.embedding_dim = self._get_embedding_dim()
        self.pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        logging.info(f"Loaded Octo model: {model_type}, embedding_dim={self.embedding_dim}")

        # --- 定义核心 JIT 函数 ---
        # 我们定义一个专门提取 Embedding 的函数，包裹 model.run_transformer
        @jax.jit
        def _extract_embeddings(params, observations, tasks, pad_mask):
            # 调用 OctoModel 自带的 run_transformer (它处理了大部分逻辑)
            # 注意：这里直接调用 module.apply 会更稳，因为 run_transformer 有 check 可能会慢
            transformer_outputs = self.model.module.apply(
                {"params": params},
                observations,
                tasks,
                pad_mask,
                train=False,
                method="octo_transformer",
            )
            # 提取 readout_action 并取平均
            return transformer_outputs['readout_action'].tokens.mean(axis=-2)[:, 0, :]

        self._extract_fn = _extract_embeddings

        # --- Warmup (预热) ---
        logging.info("Warming up JAX JIT...")
        # 构造一个 batch_size=64 的虚假输入进行编译
        dummy_batch_size = 64
        # 1. 创建任务 (使用文本创建，获取 Token IDs)
        dummy_tasks = self.model.create_tasks(texts=["dummy"] * dummy_batch_size)
        # 2. 创建图像
        dummy_img = np.zeros((dummy_batch_size, 1, self.image_size, self.image_size, 3), dtype=np.uint8)
        dummy_pad = np.ones((dummy_batch_size, 1), dtype=bool)
        # 3. 构造 observations 字典
        dummy_obs = {"image_primary": dummy_img, self.pad_key: dummy_pad}
        
        # 编译
        self._extract_fn(self.model.params, dummy_obs, dummy_tasks, dummy_pad).block_until_ready()
        logging.info("JIT compilation finished.")

    def _get_embedding_dim(self) -> int:
        # 保持原样，只运行一次
        dummy_img = np.zeros((1, 1, self.image_size, self.image_size, 3), dtype=np.uint8)
        pad_mask = np.ones((1, 1), dtype=bool)
        self.pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": dummy_img, self.pad_key: pad_mask}
        tasks = self.model.create_tasks(texts=["dummy"])
        transformer_outputs = self.model.run_transformer(
            input_observation, tasks, timestep_pad_mask=pad_mask, train=False
        )
        return transformer_outputs['readout_action'].tokens.shape[-1]

    def encode_trajectory(self, images: np.ndarray, language_instruction: str, batch_size: int = 64) -> np.ndarray:
        """
        Args:
            images: (T, 256, 256, 3)
            language_instruction: str (原始文本，不再是 embedding)
        """
        T = images.shape[0]
        
        # 1. 准备 Task (Token IDs)
        # 为了避免重复 Tokenize 64 次，我们先创建一个 size=1 的 task，然后复制
        # 这比 tokenizer 跑 64 次要快得多
        single_task = self.model.create_tasks(texts=[language_instruction])
        
        # 将 task 中的所有数组在第 0 维复制 batch_size 份
        # 使用 jax.tree_map 自动处理嵌套字典
        batch_tasks = jax.tree_map(
            lambda x: np.tile(x, (batch_size,) + (1,) * (x.ndim - 1)), 
            single_task
        )
        
        # 2. 准备 Pad Mask (全 1)
        batch_pad_mask = np.ones((batch_size, 1), dtype=bool)

        all_embeddings = []
        
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_imgs = images[start:end]
            actual_n = batch_imgs.shape[0]
            
            # --- Padding Logic ---
            if actual_n < batch_size:
                pad_len = batch_size - actual_n
                batch_imgs_padded = np.pad(batch_imgs, ((0, pad_len), (0,0), (0,0), (0,0)), mode='constant')
            else:
                batch_imgs_padded = batch_imgs

            # 构造 input_observation
            input_obs = {
                "image_primary": batch_imgs_padded[:, None], # (B, 1, H, W, C)
                self.pad_key: batch_pad_mask
            }

            # --- Run JIT ---
            # 这里的 batch_tasks 永远是 64 大小，input_obs 也是 64 大小
            # 所以 JAX 不会重新编译
            embeddings_padded = self._extract_fn(
                self.model.params, 
                input_obs, 
                batch_tasks, 
                batch_pad_mask
            )
            
            # Unpad
            if actual_n < batch_size:
                # 转换回 numpy 并切片
                embeddings = np.array(embeddings_padded[:actual_n])
            else:
                embeddings = np.array(embeddings_padded)
            
            all_embeddings.append(embeddings)

        if not all_embeddings:
            return np.zeros((0, self.embedding_dim))
            
        return np.concatenate(all_embeddings, axis=0)


# ============================================================================
# TFDS-Compatible TFRecord Writing
# ============================================================================

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _tensor_feature(arr: np.ndarray) -> tf.train.Feature:
    """Serialize tensor to bytes for TFDS compatibility."""
    return _bytes_feature(tf.io.serialize_tensor(tf.constant(arr)).numpy())


class TFDSCompatibleWriter:
    """Writes TFRecords in TFDS-compatible format matching bridge_dataset structure."""

    def __init__(
        self,
        out_dir: Path,
        dataset_name: str,
        split_name: str,
        episodes_per_shard: int,
        embedding_dim: int,
    ) -> None:
        self.out_dir = out_dir
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.episodes_per_shard = episodes_per_shard
        self.embedding_dim = embedding_dim

        self._writer: Optional[tf.io.TFRecordWriter] = None
        self._episodes_in_shard = 0
        self._shard_idx = 0
        self.shard_lengths: List[int] = []
        self.num_trajectories = 0
        self.num_transitions = 0

    def _get_filename(self, total_shards: int) -> str:
        # Match TFDS naming: {DATASET}-{SPLIT}.tfrecord-{SHARD_X_OF_Y}
        return f"{self.dataset_name}-{self.split_name}.tfrecord-{self._shard_idx:05d}-of-{total_shards:05d}"

    def _open_new_shard(self, total_shards: int) -> None:
        if self._writer is not None:
            self._writer.close()
            self.shard_lengths.append(self._episodes_in_shard)

        filename = self._get_filename(total_shards)
        path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = tf.io.TFRecordWriter(str(path))
        self._episodes_in_shard = 0
        self._shard_idx += 1
        logging.info("Opened shard: %s", filename)

    def write_trajectory(
        self,
        obs_embeddings: np.ndarray,      # (T, D) - Octo embeddings
        next_obs_embeddings: np.ndarray, # (T, D) - next observation embeddings
        actions: np.ndarray,              # (T, action_dim)
        rewards: np.ndarray,              # (T,)
        td_mask: np.ndarray,              # (T,)
        mc_return: np.ndarray,            # (T,)
        language_instruction: str,
        language_embedding: np.ndarray,   # (512,) or similar
        total_shards: int,
    ) -> None:
        if self._writer is None or self._episodes_in_shard >= self.episodes_per_shard:
            self._open_new_shard(total_shards)

        T = obs_embeddings.shape[0]

        # Build steps sequence - matching bridge_dataset structure
        # Each step is serialized as a SequenceExample or nested Features
        steps_features = []
        for t in range(T):
            step_features = {
                # Observation: embedding instead of image
                "steps/observation/embedding": _tensor_feature(obs_embeddings[t].astype(np.float32)),
                "steps/next_observation/embedding": _tensor_feature(next_obs_embeddings[t].astype(np.float32)),

                # Action
                "steps/action": _tensor_feature(actions[t].astype(np.float32)),

                # Language (repeated per step for compatibility)
                "steps/language_instruction": _bytes_feature(language_instruction.encode("utf-8")),
                "steps/language_embedding": _tensor_feature(language_embedding.astype(np.float32)),

                # Rewards and masks
                "steps/reward": _float_feature(float(rewards[t])),
                "steps/td_mask": _float_feature(float(td_mask[t])),
                "steps/mc_return": _float_feature(float(mc_return[t])),
                "steps/discount": _float_feature(0.98),

                # Step markers
                "steps/is_first": _int64_feature(1 if t == 0 else 0),
                "steps/is_last": _int64_feature(1 if t == T - 1 else 0),
                "steps/is_terminal": _int64_feature(1 if t == T - 1 else 0),
            }
            steps_features.append(step_features)

        # Flatten all steps into a single feature dict with indexed keys
        # This matches how TFDS stores sequences
        all_features = {}

        # Store steps as serialized tensor sequences
        all_features["steps/observation/embedding"] = _tensor_feature(obs_embeddings.astype(np.float32))
        all_features["steps/next_observation/embedding"] = _tensor_feature(next_obs_embeddings.astype(np.float32))
        all_features["steps/action"] = _tensor_feature(actions.astype(np.float32))
        all_features["steps/language_instruction"] = _bytes_feature(language_instruction.encode("utf-8"))
        all_features["steps/language_embedding"] = _tensor_feature(language_embedding.astype(np.float32))
        all_features["steps/reward"] = _tensor_feature(rewards.astype(np.float32))
        all_features["steps/td_mask"] = _tensor_feature(td_mask.astype(np.float32))
        all_features["steps/mc_return"] = _tensor_feature(mc_return.astype(np.float32))

        # Scalars per step
        is_first = np.zeros(T, dtype=np.bool_)
        is_first[0] = True
        is_last = np.zeros(T, dtype=np.bool_)
        is_last[-1] = True
        is_terminal = np.zeros(T, dtype=np.bool_)
        is_terminal[-1] = True
        discount = np.ones(T, dtype=np.float32) * 0.98

        all_features["steps/is_first"] = _tensor_feature(is_first)
        all_features["steps/is_last"] = _tensor_feature(is_last)
        all_features["steps/is_terminal"] = _tensor_feature(is_terminal)
        all_features["steps/discount"] = _tensor_feature(discount)

        # Episode metadata
        all_features["episode_metadata/episode_id"] = _int64_feature(self.num_trajectories)
        all_features["episode_metadata/trajectory_length"] = _int64_feature(T)

        example = tf.train.Example(features=tf.train.Features(feature=all_features))

        assert self._writer is not None
        self._writer.write(example.SerializeToString())
        self._episodes_in_shard += 1
        self.num_trajectories += 1
        self.num_transitions += T

    def close(self) -> None:
        if self._writer is not None:
            self.shard_lengths.append(self._episodes_in_shard)
            self._writer.close()
            self._writer = None


def save_features_json(output_dir: Path, embedding_dim: int, action_dim: int, lang_emb_dim: int = 512):
    """Save features.json matching TFDS format."""
    features_schema = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "action": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {
                                            "shape": {"dimensions": [str(action_dim)]},
                                            "dtype": "float32",
                                            "encoding": "none"
                                        },
                                        "description": "Robot action"
                                    },
                                    "language_embedding": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {
                                            "shape": {"dimensions": [str(lang_emb_dim)]},
                                            "dtype": "float32",
                                            "encoding": "none"
                                        },
                                        "description": "Language embedding (MUSE)"
                                    },
                                    "language_instruction": {
                                        "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                        "text": {},
                                        "description": "Language Instruction"
                                    },
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "embedding": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": [str(embedding_dim)]},
                                                        "dtype": "float32",
                                                        "encoding": "none"
                                                    },
                                                    "description": "Octo embedding for observation"
                                                }
                                            }
                                        }
                                    },
                                    "next_observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "embedding": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": [str(embedding_dim)]},
                                                        "dtype": "float32",
                                                        "encoding": "none"
                                                    },
                                                    "description": "Octo embedding for next observation"
                                                }
                                            }
                                        }
                                    },
                                    "reward": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Reward"
                                    },
                                    "td_mask": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "TD mask (1 for valid, 0 for terminal)"
                                    },
                                    "mc_return": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Monte Carlo return"
                                    },
                                    "discount": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Discount factor"
                                    },
                                    "is_first": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on first step"
                                    },
                                    "is_last": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step"
                                    },
                                    "is_terminal": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on terminal step"
                                    }
                                }
                            }
                        },
                        "length": "-1"
                    }
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "episode_id": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"},
                                "description": "Episode ID"
                            },
                            "trajectory_length": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"},
                                "description": "Trajectory length"
                            }
                        }
                    }
                }
            }
        }
    }

    with open(output_dir / "features.json", "w") as f:
        json.dump(features_schema, f, indent=2)
    logging.info("Saved features.json")


def save_dataset_info_json(
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    shard_lengths: List[int],
    num_bytes: int = 0,
):
    """Save dataset_info.json matching TFDS format."""
    dataset_info = {
        "citation": "Octo embedding dataset",
        "description": f"Precomputed Octo embeddings for {dataset_name}",
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}_embedding.{dataset_name}_embedding_dataset_builder",
        "name": f"{dataset_name}_embedding",
        "releaseNotes": {"1.0.0": "Initial release with Octo embeddings"},
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": split_name,
                "numBytes": str(num_bytes),
                "shardLengths": [str(l) for l in shard_lengths]
            }
        ],
        "version": "1.0.0"
    }

    with open(output_dir / f"dataset_info_{split_name}.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    logging.info("Saved dataset_info.json")


def save_metadata_json(
    output_dir: Path,
    embedding_dim: int,
    num_trajectories: int,
    num_transitions: int,
    encoder_name: str,
    action_statistics: Optional[Dict] = None,
):
    """Save additional metadata for training."""
    metadata = {
        "embedding_dim": embedding_dim,
        "num_trajectories": num_trajectories,
        "num_transitions": num_transitions,
        "encoder": encoder_name,
    }
    if action_statistics:
        metadata["action"] = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in action_statistics.items()
        }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Saved metadata.json")


# ============================================================================
# Trajectory Processing
# ============================================================================

def process_trajectory(
    traj: Dict[str, Any],
    encoder: OctoEmbeddingGenerator,
    text_processor,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Process a single trajectory and compute embeddings."""
    # Extract images
    images = np.array(traj["observation"]["image_primary"])
    if images.ndim == 5:
        images = images.squeeze(axis=1)

    # Get language instruction
    lang_instr = traj["task"]["language_instruction"]
    if isinstance(lang_instr, bytes):
        lang_instr = lang_instr.decode("utf-8")
    elif isinstance(lang_instr, np.ndarray):
        lang_instr = lang_instr.flat[0]
        if isinstance(lang_instr, bytes):
            lang_instr = lang_instr.decode("utf-8")

    # Encode language
    lang_emb = text_processor.encode([lang_instr]) #(1,16,768)

    # Encode observations
    T = images.shape[0]
    obs_emb = encoder.encode_trajectory(images, lang_instr, batch_size)

    # Next observation embeddings (shift by 1, repeat last)
    if "next_observation" in traj and "image_primary" in traj["next_observation"]:
        next_images = np.array(traj["next_observation"]["image_primary"])
        if next_images.ndim == 5:
            next_images = next_images.squeeze(axis=1)
        next_obs_emb = encoder.encode_trajectory(next_images, lang_instr, batch_size)
    else:
        next_obs_emb = np.concatenate([obs_emb[1:], obs_emb[-1:]], axis=0)

    # Extract other fields
    actions = np.array(traj["action"])
    if actions.ndim == 3:
        actions = actions.squeeze(axis=1)

    def safe_array(val, default_shape, dtype=np.float32):
        arr = np.array(val) if val is not None else np.zeros(default_shape, dtype=dtype)
        if arr.ndim == 0:
            arr = np.array([arr])
        arr = arr.flatten()[:T]
        if len(arr) < T:
            arr = np.pad(arr, (0, T - len(arr)), mode='edge')
        return arr.astype(dtype)

    rewards = safe_array(traj.get("reward"), (T,))
    td_mask = safe_array(traj.get("td_mask"), (T,))
    mc_return = safe_array(traj.get("mc_return"), (T,))

    return {
        "obs_emb": obs_emb,
        "next_obs_emb": next_obs_emb,
        "actions": actions,
        "rewards": rewards,
        "td_mask": td_mask,
        "mc_return": mc_return,
        "language_instruction": lang_instr,
        "language_embedding": lang_emb,
    }


def count_trajectories(dataset) -> int:
    count = 0
    for _ in dataset.iterator():
        count += 1
    return count


# ============================================================================
# Main
# ============================================================================

def main(_):
    tf.config.set_visible_devices([], "GPU")
    logging.set_verbosity(logging.INFO)

    output_dir = Path(FLAGS.output_dir).expanduser()
    if output_dir.exists() and FLAGS.overwrite:
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    logging.info(f"Loading Octo encoder: {FLAGS.encoder}")
    encoder = OctoEmbeddingGenerator(model_type=FLAGS.encoder)
    text_processor = text_processors["t5"]()

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
            logging.warning(f"Failed to create dataset {dataset_name}: {e}")
            continue

        logging.info("Counting trajectories...")
        total_trajs = count_trajectories(dataset)
        logging.info(f"Total trajectories: {total_trajs}")

        if total_trajs == 0:
            logging.warning(f"No trajectories in {dataset_name}, skipping")
            continue

        # Re-create dataset
        dataset = make_single_dataset(
            dataset_kwargs,
            train=(FLAGS.split == "train"),
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transform_kwargs=frame_transform_kwargs,
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        dataset_output_dir = output_dir / f"{dataset_name}_embedding" / "1.0.0"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        total_shards = max(1, math.ceil(total_trajs / FLAGS.episodes_per_shard))

        writer = TFDSCompatibleWriter(
            out_dir=dataset_output_dir,
            dataset_name=f"{dataset_name}_embedding",
            split_name=FLAGS.split,
            episodes_per_shard=FLAGS.episodes_per_shard,
            embedding_dim=encoder.embedding_dim,
        )

        action_dim = None
        action_stats = getattr(dataset, 'dataset_statistics', {}).get("action", None)

        for traj in tqdm.tqdm(dataset.iterator(), total=total_trajs, desc=f"Processing {dataset_name}"):
            # try:
            processed = process_trajectory(traj, encoder, text_processor, FLAGS.batch_size)
            if action_dim is None:
                action_dim = processed["actions"].shape[-1]

            writer.write_trajectory(
                obs_embeddings=processed["obs_emb"],
                next_obs_embeddings=processed["next_obs_emb"],
                actions=processed["actions"],
                rewards=processed["rewards"],
                td_mask=processed["td_mask"],
                mc_return=processed["mc_return"],
                language_instruction=processed["language_instruction"],
                language_embedding=processed["language_embedding"],
                total_shards=total_shards,
            )
            # except Exception as e:
            #     logging.warning(f"Failed to process trajectory: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     continue

        writer.close()

        # Save TFDS metadata files
        if action_dim:
            save_features_json(dataset_output_dir, encoder.embedding_dim, action_dim)
        save_dataset_info_json(
            dataset_output_dir,
            f"{dataset_name}_embedding",
            FLAGS.split,
            writer.shard_lengths
        )
        save_metadata_json(
            dataset_output_dir,
            encoder.embedding_dim,
            writer.num_trajectories,
            writer.num_transitions,
            FLAGS.encoder,
            action_stats,
        )

        logging.info(f"Completed {dataset_name}: {writer.num_trajectories} trajectories, {writer.num_transitions} transitions")

    logging.info(f"All done. Embeddings saved to {output_dir}")


if __name__ == "__main__":
    app.run(main)
