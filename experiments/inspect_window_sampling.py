#!/usr/bin/env python3
"""Utility to visualize which timesteps populate each trajectory window."""

import argparse
from typing import Sequence

import numpy as np
import tensorflow as tf

from ml_collections import ConfigDict

from configs import ttt_data_config
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights


def _apply_overrides(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    cfg = config.copy_and_resolve_references()
    cfg.oxe_kwargs["data_dir"] = args.data_dir
    cfg.oxe_kwargs["data_mix"] = args.data_mix
    cfg.batch_size = args.batch_size
    cfg.traj_transform_kwargs["window_size"] = args.window_size
    cfg.traj_transform_kwargs["subsample_length"] = args.subsample_length
    cfg.traj_transform_kwargs["goal_relabeling_strategy"] = args.goal_relabeling_strategy
    return cfg


def _finalize_dataset_kwargs(cfg: ConfigDict) -> ConfigDict:
    (cfg["dataset_kwargs_list"], cfg["sample_weights"]) = make_oxe_dataset_kwargs_and_weights(
        **cfg["oxe_kwargs"]
    )
    del cfg["oxe_kwargs"]
    return cfg


def _to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _format_timesteps(timesteps: np.ndarray, pad_mask: np.ndarray) -> str:
    valid = timesteps[pad_mask]
    if valid.size == 0:
        return "(empty)"
    preview = ", ".join(map(str, valid[:6]))
    if valid.size > 6:
        preview += ", ..."
    return f"len={valid.size}, range=[{valid[0]}, {valid[-1]}], seq=({preview})"


def _format_rewards(rewards: np.ndarray, pad_mask: np.ndarray) -> str:
    rewards = np.asarray(rewards)
    if rewards.ndim == 0:
        return f"scalar={float(rewards):.3f}"
    valid = rewards[pad_mask]
    if valid.size == 0:
        return "(empty)"
    mean_val = float(np.mean(valid))
    preview = ", ".join(f"{r:.2f}" for r in valid[:6])
    if valid.size > 6:
        preview += ", ..."
    return f"len={valid.size}, mean={mean_val:.3f}, seq=({preview})"


def _format_language(lang: np.ndarray) -> str:
    if isinstance(lang, bytes):
        return lang.decode("utf-8", "ignore")
    if hasattr(lang, "decode"):
        return lang.decode("utf-8", "ignore")
    return str(lang)


def _select_window(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    indices = np.asarray(indices).astype(int)
    if values.ndim == 0:
        return np.full_like(indices, fill_value=float(values), dtype=float)
    if values.shape == indices.shape:
        return values
    return np.take(values, indices, mode="clip")


def summarize_windows(batch: dict, batch_index: int) -> Sequence[str]:
    observations = batch["observation"]
    timesteps = _to_numpy(observations["timestep"])
    pad_mask = _to_numpy(observations["timestep_pad_mask"]).astype(bool)
    languages = batch["task"].get("language_instruction")

    rewards = None
    for key in ("reward", "rewards"):
        if key in batch and batch[key] is not None:
            rewards = batch[key]
            break

    returns = None
    for key in ("mc_return", "mc_returns"):
        if key in batch and batch[key] is not None:
            returns = batch[key]
            break
    if languages is not None:
        languages = _to_numpy(languages)
    if rewards is not None:
        rewards = _to_numpy(rewards)
    if returns is not None:
        returns = _to_numpy(returns)
    lines = []
    for i in range(timesteps.shape[0]):
        lang = _format_language(languages[i]) if languages is not None else "<none>"
        description = _format_timesteps(timesteps[i], pad_mask[i])
        reward_desc = ""
        return_desc = ""
        if rewards is not None:
            reward_window = _select_window(rewards[i], timesteps[i])
            reward_desc = _format_rewards(reward_window, pad_mask[i])
        if returns is not None:
            return_window = _select_window(returns[i], timesteps[i])
            return_desc = _format_rewards(return_window, pad_mask[i])
        lines.append(
            f"[batch {batch_index} sample {i}] {description} | language='{lang}'"
            + (f" | rewards={reward_desc}" if reward_desc else "")
            + (f" | returns={return_desc}" if return_desc else "")
        )
    return lines


def inspect_windows(args: argparse.Namespace) -> None:
    tf.config.set_visible_devices([], "GPU")

    cfg = _apply_overrides(ttt_data_config.get_config(), args)
    cfg = _finalize_dataset_kwargs(cfg)

    dataset = make_interleaved_dataset(**cfg, train=True)
    batched = dataset.unbatch().repeat().batch(args.batch_size)
    iterator = batched.iterator()

    for idx in range(args.num_batches):
        batch = next(iterator)
        lines = summarize_windows(batch, idx)
        print("\n".join(lines))
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trajectory window coverage")
    parser.add_argument("--data-dir", required=True, help="Path or GCS URI containing RLDS datasets")
    parser.add_argument("--data-mix", default="fractal", help="OXE data_mix to sample")
    parser.add_argument("--window-size", type=int, default=120, help="Window size handed to traj transform")
    parser.add_argument("--subsample-length", type=int, default=None, help="Optional subsampling horizon")
    parser.add_argument(
        "--goal-relabeling-strategy",
        type=str,
        default="uniform",
        help="Goal relabeling strategy to forward to dataset config",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Number of windows reported per batch")
    parser.add_argument("--num-batches", type=int, default=3, help="How many batches to print")
    return parser.parse_args()


if __name__ == "__main__":
    inspect_windows(parse_args())
