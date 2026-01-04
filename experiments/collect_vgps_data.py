#!/usr/bin/env python3
"""Collect simulated trajectories compatible with RLDS/TFDS structure.

Output:
1. TFRecords named: {dataset_name}-{split}.tfrecord-{shard_index}-of-{total_shards}
2. dataset_info.json
3. features.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OCTO_ROOT = REPO_ROOT / "octo"
if str(OCTO_ROOT) not in sys.path:
    sys.path.insert(0, str(OCTO_ROOT))

import imageio
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

import simpler_env
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)
from simpler_env.policies.octo.octo_model import OctoInference
# Note: vgps_utils import is kept but optional based on flags
try:
    from vgps_utils import load_vgps_checkpoint
except ImportError:
    pass

FLAGS = flags.FLAGS

# Evaluation / policy flags
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Octo model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "ManiSkill task.")
flags.DEFINE_integer("num_samples", 50, "Number of V-GPS action samples.")
flags.DEFINE_float("action_temp", 1.0, "Softmax temperature for V-GPS sampler.")
flags.DEFINE_boolean("use_vgps", False, "Whether to enable the V-GPS critic.")
flags.DEFINE_string("vgps_checkpoint", "", "Checkpoint path for critic.")
flags.DEFINE_string("vgps_wandb", "", "Optional W&B run for critic config.")
flags.DEFINE_float("target_success_rate", 0.7, "Desired success rate in the final dataset.")
flags.DEFINE_integer("max_attempts_multiplier", 5, "Safety limit.")
flags.DEFINE_integer("image_size", 256, "Resolution to resize images to.")

# Collection-specific flags
flags.DEFINE_string("output_dir", "logs/rlds_data", "Directory where data is stored.")
flags.DEFINE_integer("max_episodes", 50, "Number of episodes to collect.")
flags.DEFINE_integer("episodes_per_shard", 10, "Trajectories per TFRecord shard.") # Reduced default for testing
flags.DEFINE_bool("record_videos", False, "Whether to dump MP4s.")
flags.DEFINE_bool("save_npz", False, "Whether to dump npz payloads (disabled by default for RLDS focus).")
flags.DEFINE_float("discount", 0.98, "Discount used for Monte-Carlo returns.")
flags.DEFINE_string("dataset_name", "my_custom_task_data", "Name stamped into filenames.")
flags.DEFINE_string("split_name", "train", "Split name (e.g., train, val).")
flags.DEFINE_float("success_reward", 1.0, "Sparse reward.")
flags.DEFINE_bool("overwrite", False, "If True, delete output_dir before collecting.")


def _serialize_tensor(value: np.ndarray) -> bytes:
    """Serialize a numpy array into the TFRecord format."""
    tensor = tf.convert_to_tensor(value)
    return tf.io.serialize_tensor(tensor).numpy()

def _feature_from_array(value: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[_serialize_tensor(value)]))

def _feature_int64(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _feature_bytes(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    image_tensor = tf.convert_to_tensor(image)
    resized_tensor = tf.image.resize(image_tensor, [size, size], method='bilinear')
    return tf.cast(tf.clip_by_value(resized_tensor, 0, 255), tf.uint8).numpy()

class TFRecordShardWriter:
    """Writes TFRecords with the naming convention: {dataset}-{split}.tfrecord-{shard}-of-{total}"""

    def __init__(self, out_dir: Path, dataset_name: str, split_name: str, 
                 episodes_per_shard: int, total_episodes: int) -> None:
        self.out_dir = out_dir
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.episodes_per_shard = episodes_per_shard
        self.total_shards = math.ceil(total_episodes / episodes_per_shard)
        
        self._writer: Optional[tf.io.TFRecordWriter] = None
        self._episodes_in_shard = 0
        self._shard_idx = 0
        self.paths: List[str] = []
        self.shard_lengths: List[str] = []

    def _get_filename(self) -> str:
        # Format: dataset_name-split.tfrecord-00000-of-00010
        return f"{self.dataset_name}-{self.split_name}.tfrecord-{self._shard_idx:05d}-of-{self.total_shards:05d}"

    def _open_new_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self.shard_lengths.append(str(self._episodes_in_shard))
        
        filename = self._get_filename()
        path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self._writer = tf.io.TFRecordWriter(str(path))
        self.paths.append(str(path))
        self._episodes_in_shard = 0
        self._shard_idx += 1
        logging.info("Opened shard: %s", filename)

    def write(self, example: tf.train.Example) -> None:
        if self._writer is None or self._episodes_in_shard >= self.episodes_per_shard:
            self._open_new_shard()
        assert self._writer is not None
        self._writer.write(example.SerializeToString())
        self._episodes_in_shard += 1

    def close(self) -> None:
        if self._writer is not None:
            self.shard_lengths.append(str(self._episodes_in_shard))
            self._writer.close()
            self._writer = None

class EpisodeBuffer:
    def __init__(self, instruction: str, discount: float, success_reward: float):
        self.instruction = instruction
        self.discount = discount
        self.success_reward = success_reward
        
        # Buffers for time-series data
        self.images: List[np.ndarray] = []
        
        # Action components
        self.world_vectors: List[np.ndarray] = []
        self.rotation_deltas: List[np.ndarray] = []
        self.gripper_actions: List[np.ndarray] = []
        self.terminate_actions: List[np.ndarray] = []

    def append_observation(self, image: np.ndarray) -> None:
        self.images.append(image)

    def append_action(self, world_vector: np.ndarray, rotation_delta: np.ndarray, gripper_closedness: np.ndarray) -> None:
        self.world_vectors.append(world_vector.astype(np.float32))
        self.rotation_deltas.append(rotation_delta.astype(np.float32))
        self.gripper_actions.append(gripper_closedness.astype(np.float32))
        # Dummy terminate action (0,0,1) or similar, using int32 for now as placeholder
        self.terminate_actions.append(np.array([1, 0, 0], dtype=np.int32)) 

    def finalize(self, success: bool) -> Dict[str, Any]:
        if len(self.world_vectors) == 0:
            raise ValueError("Cannot finalize empty trajectory")

        # Observations (T+1 frames, last one is terminal state)
        # Note: In standard RLDS step-based datasets, we typically align obs[t] with action[t].
        # The last observation is usually part of a terminal step with dummy action or handled via is_last.
        # Here we follow the convention: Steps = length of actions.
        
        n_steps = len(self.world_vectors)
        
        # Slice observations to match action length (remove last obs if we strictly want Step(obs, act, rew))
        # OR keep all and repeat last action. Standard RLDS usually has T steps.
        # We will use the first T observations for the steps.
        obs_images = np.stack(self.images[:-1], axis=0) # Shape: (T, H, W, 3)
        
        actions_world = np.stack(self.world_vectors, axis=0)
        actions_rot = np.stack(self.rotation_deltas, axis=0)
        actions_gripper = np.stack(self.gripper_actions, axis=0)
        actions_terminate = np.stack(self.terminate_actions, axis=0)
        
        # Rewards
        rewards = np.zeros(n_steps, dtype=np.float32)
        if success:
            rewards[-1] = self.success_reward
            
        # Is First/Last/Terminal
        is_first = np.zeros(n_steps, dtype=bool)
        is_first[0] = True
        is_last = np.zeros(n_steps, dtype=bool)
        is_last[-1] = True
        is_terminal = np.zeros(n_steps, dtype=bool)
        # If success, it is terminal. If failed but truncated, technically not terminal in MDP sense but last in file.
        # Let's mark successful end as terminal.
        if success:
            is_terminal[-1] = True

        # Language
        # Repeating for each step is standard for "flat" TFRecord features representing a Sequence
        language = np.array([self.instruction] * n_steps, dtype=object)
        
        # Dummy Embedding (512,) float32 - Required by Schema
        dummy_embedding = np.zeros((n_steps, 512), dtype=np.float32)

        # Construct flat dictionary with "/" notation matching features.json structure
        data = {
            "steps/observation/image": obs_images,
            "steps/observation/natural_language_instruction": language,
            "steps/observation/natural_language_embedding": dummy_embedding,
            
            # Missing obs from Schema placeholders (filling with zeros to match schema if needed)
            # "steps/observation/wrist_image": ..., 
            
            "steps/action/world_vector": actions_world,
            "steps/action/rotation_delta": actions_rot,
            "steps/action/gripper_closedness_action": actions_gripper,
            "steps/action/terminate_episode": actions_terminate,
            
            "steps/reward": rewards,
            "steps/is_first": is_first,
            "steps/is_last": is_last,
            "steps/is_terminal": is_terminal,
            
            # Episode level metadata (broadcasted or single, but usually broadcasted for simple readers)
            "steps/observation/pad_mask": np.ones(n_steps, dtype=bool) # Optional helper
        }
        
        # Add Episode-level context (attributes)
        data["attributes/success"] = success
        return data

def _trajectory_to_example(traj_dict: Dict[str, Any]) -> tf.train.Example:
    features = {}
    
    # Process steps
    for key, value in traj_dict.items():
        if key.startswith("attributes/"):
            # Scalar context features (not sequence)
            if isinstance(value, bool):
                 features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
            else:
                 # Extend logic if needed for other attribute types
                 pass
        else:
            # Sequence features (T, ...) -> Serialize entire array as tensor
            features[key] = _feature_from_array(value)
            
    return tf.train.Example(features=tf.train.Features(feature=features))

def save_features_json(output_dir: Path):
    """Writes the features.json file matching the user's schema."""
    features_schema = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "attributes": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "success": {
                                # 属性数据你没有用 serialize_tensor，而是存成了 Int64List
                                # TFDS 可以自动把 Int64 读成 Bool，所以这里保持 bool 没问题
                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}
                            }
                        }
                    }
                },
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "natural_language_instruction": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                                },
                                                "natural_language_embedding": {
                                                    # 关键修改：Shape 改为 {} (标量)，dtype 改为 string
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                                },
                                                "image": {
                                                    # 关键修改：不再伪装成 ImageFeature，直接承认是 serialized string
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                                }
                                            }
                                        }
                                    },
                                    "action": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "terminate_episode": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                                },
                                                "gripper_closedness_action": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                                                    "description": "continuous gripper position"
                                                },
                                                "rotation_delta": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                                                    "description": "rpy commanded orientation displacement"
                                                },
                                                "world_vector": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                                                    "description": "commanded end-effector displacement"
                                                }
                                            }
                                        }
                                    },
                                    "is_first": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                    },
                                    "is_terminal": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                    },
                                    "is_last": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                    },
                                    "reward": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"}
                                    }
                                }
                            }
                        },
                        "length": "-1"
                    }
                }
            }
        }
    }
    
    with open(output_dir / "features.json", "w") as f:
        json.dump(features_schema, f, indent=4)
    logging.info("Saved features.json")

def save_dataset_info(output_dir: Path, shard_lengths: List[str], total_episodes: int):
    """Writes dataset_info.json."""
    info = {
        "name": FLAGS.dataset_name,
        "description": f"Simulated data for task {FLAGS.task_name}",
        "version": "0.1.0",
        "fileFormat": "tfrecord",
        "citation": "Todo",
        "location": {
            "urls": ["http://localhost"]
        },
        "releaseNotes": {
            "0.1.0": "Initial release."
        },
        "splits": [
            {
                "name": FLAGS.split_name,
                "shardLengths": shard_lengths,
                "numShards": str(len(shard_lengths)),
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"
            }
        ]
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logging.info("Saved dataset_info.json")

def main(_):
    tf.config.set_visible_devices([], "GPU")
    logging.set_verbosity(logging.INFO)

    # Setup Output Directory
    base_dir = Path(FLAGS.output_dir).expanduser()
    if base_dir.exists() and FLAGS.overwrite:
        import shutil
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = base_dir / "videos"
    if FLAGS.record_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Shard Writer
    shard_writer = TFRecordShardWriter(
        out_dir=base_dir,
        dataset_name=FLAGS.dataset_name,
        split_name=FLAGS.split_name,
        episodes_per_shard=FLAGS.episodes_per_shard,
        total_episodes=FLAGS.max_episodes
    )

    # Initialize Environment and Model
    env = simpler_env.make(FLAGS.task_name)
    
    if "google" in FLAGS.task_name:
        policy_setup = "google_robot"
        sticky_steps = 15
    else:
        policy_setup = "widowx_bridge"
        sticky_steps = 3

    model = OctoInference(
        model_type=FLAGS.model_name,
        policy_setup=policy_setup,
        init_rng=FLAGS.seed,
        sticky_step=sticky_steps,
    )

    # V-GPS setup (omitted for brevity, assume logic works if enabled)
    if FLAGS.use_vgps and FLAGS.vgps_checkpoint:
        get_values, critic_text_processor = load_vgps_checkpoint(
            FLAGS.vgps_checkpoint, FLAGS.vgps_wandb
        )
        model.init_vgps(
            FLAGS.num_samples, get_values, critic_text_processor,
            action_temp=FLAGS.action_temp, max_episode_steps=env._max_episode_steps
        )

    # Quota Calculation
    target_total = FLAGS.max_episodes
    target_success_count = int(target_total * FLAGS.target_success_rate)
    target_failure_count = target_total - target_success_count
    
    saved_success_count = 0
    saved_failure_count = 0
    total_attempts = 0
    
    # Collection Loop
    while (saved_success_count + saved_failure_count) < target_total:
        total_attempts += 1
        if total_attempts > target_total * FLAGS.max_attempts_multiplier:
            logging.warning("Max attempts reached.")
            break

        obs, _ = env.reset()
        instruction = env.get_language_instruction()
        model.reset(instruction)

        image = get_image_from_maniskill2_obs_dict(env, obs)
        image = _resize_image(image, FLAGS.image_size)

        buffer = EpisodeBuffer(
            instruction=instruction,
            discount=FLAGS.discount,
            success_reward=FLAGS.success_reward,
        )
        buffer.append_observation(image)
        episode_frames = [image]

        success = False
        truncated = False
        
        while not (success or truncated):
            _, action = model.step(image)
            
            # Extract action components for Octo policy return structure
            # Note: OctoInference usually returns a dict. We need to parse it back 
            # into the components expected by our new Schema.
            # Assuming action keys: 'world_vector', 'rot_axangle', 'gripper'
            world_vec = np.asarray(action["world_vector"]).reshape(3)
            rot_delta = np.asarray(action["rot_axangle"]).reshape(3)
            gripper = np.asarray(action["gripper"]).reshape(1)
            
            # Combine for Env Step
            action_vec = np.concatenate([world_vec, rot_delta, gripper], axis=0)
            
            # Add to buffer components
            buffer.append_action(world_vec, rot_delta, gripper)
            
            # Step Env
            obs, reward, success, truncated, info = env.step(action_vec)
            
            image = get_image_from_maniskill2_obs_dict(env, obs)
            image = _resize_image(image, FLAGS.image_size)
            
            buffer.append_observation(image)
            if FLAGS.record_videos:
                episode_frames.append(image)

        # Decide to Save
        is_success = bool(success)
        should_save = False

        if is_success:
            if saved_success_count < target_success_count:
                should_save = True
                saved_success_count += 1
        else:
            if saved_failure_count < target_failure_count:
                should_save = True
                saved_failure_count += 1

        if should_save:
            traj_dict = buffer.finalize(success=is_success)
            example = _trajectory_to_example(traj_dict)
            shard_writer.write(example)
            
            current_saved_idx = saved_success_count + saved_failure_count
            logging.info(f"Saved {current_saved_idx}/{target_total} (Success: {is_success})")

            if FLAGS.record_videos:
                vid_path = video_dir / f"{current_saved_idx:05d}_succ{int(is_success)}.mp4"
                imageio.mimsave(vid_path, episode_frames, fps=10)

    # Cleanup and Meta Generation
    env.close()
    shard_writer.close()
    
    # Generate Metadata JSONs
    save_features_json(base_dir)
    save_dataset_info(base_dir, shard_writer.shard_lengths, target_total)

    logging.info("Done. Data saved to %s", base_dir)

if __name__ == "__main__":
    app.run(main)