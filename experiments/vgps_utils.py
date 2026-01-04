"""Helpers shared across V-GPS evaluation/collection scripts."""

from __future__ import annotations

import os
from typing import Tuple

import jax
import numpy as np
import wandb
import yaml
from flax.training import checkpoints

from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders


def load_vgps_checkpoint(path: str, wandb_run_name: str):
    """Load a V-GPS critic checkpoint and return callable value fn + text encoder."""
    assert os.path.exists(path), f"Checkpoint path {path} does not exist"

    if wandb_run_name == "":
        with open("experiments/configs/pretrained_checkpoint.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
    example_actions = np.zeros((1, 7), dtype=np.float32)
    example_obs = {
        "image": np.zeros((1, 256, 256, 3), dtype=np.uint8)
    }
    example_batch = {
        "observations": example_obs,
        "goals": {
            "language": np.zeros((1, 512), dtype=np.float32),
        },
        "actions": example_actions,
    }

    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        encoder_def=encoder_def,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        **config["agent_kwargs"],
    )
    critic_text_processor = text_processors[config["text_processor"]]()

    agent = checkpoints.restore_checkpoint(path, agent)

    def get_values(observations, goals, actions):
        return agent.get_q_values(observations, goals, actions)

    return get_values, critic_text_processor
