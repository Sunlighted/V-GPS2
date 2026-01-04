"""
Training script for unified CQL+SimSiam agent.

This is a simplified training script compared to the original because:
1. SimSiam is integrated into the agent (single update call)
2. No need for separate SimSiam state management
3. TTT adaptation uses the agent's built-in methods

Features:
- Unified RL + SimSiam training
- TTT adaptation at evaluation time
- Value function plots comparing TTT-adapted vs base
"""

import copy
import os
from typing import Any, Dict, Iterator, Optional
from functools import partial

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from jax.experimental.compilation_cache import compilation_cache
compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

import flax
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from ml_collections import config_flags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb

from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

# Import the unified agent
from jaxrl_m.agents import agents
from jaxrl_m.agents.continuous.cql_simsiam import CQLSimSiamAgent
from jaxrl_m.networks.simsiam_networks import cosine_similarity_loss

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "cql_simsiam", "WandB project name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "oxedata_config",
    None,
    "File path to the data configuration.",
    lock_config=False,
)


# =============================================================================
# Data Processing Utilities
# =============================================================================

def _maybe_text_processor():
    """Create text processor if configured."""
    if FLAGS.config.get("text_processor") is None:
        return None
    return text_processors[FLAGS.config.text_processor](
        **FLAGS.config.text_processor_kwargs
    )


def _process_text(batch, processor):
    """Process language instructions in batch."""
    if processor is None:
        return batch
    language = [s.decode("utf-8") for s in batch["goals"]["language"]]
    batch["goals"]["language"] = processor.encode(language)
    return batch


def _process_oxe_batch(raw_batch, processor):
    """Convert OXE batch format to agent format."""
    batch = dict(
        actions=raw_batch["action"].squeeze(),
        goals=dict(language=raw_batch["task"]["language_instruction"]),
        mc_returns=raw_batch["mc_return"].squeeze(-1),
        observations=dict(image=raw_batch["observation"]["image_primary"].squeeze()),
        next_observations=dict(
            image=raw_batch["next_observation"]["image_primary"].squeeze()
        ),
        rewards=raw_batch["reward"].squeeze(-1),
        masks=raw_batch["td_mask"].squeeze(-1),
    )
    return _process_text(batch, processor)


def _to_jax_array(x):
    """Convert to JAX array, handling special types."""
    if isinstance(x, (str, bytes)):
        return x
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, np.ndarray) and (x.dtype == object or x.dtype.kind in ('U', 'S', 'O')):
        return x
    return jnp.asarray(x)


def _safe_filter_datasets(dataset_kwargs_list, sample_weights, include_names):
    """Filter datasets with fallback to original list."""
    try:
        filtered = filter_eval_datasets(dataset_kwargs_list, sample_weights, include_names)
        if not filtered or len(filtered[0]) == 0:
            raise ValueError("empty filter result")
        return filtered
    except ValueError:
        logging.warning(
            "No datasets matched filter %s; using original dataset list.", include_names
        )
        return dataset_kwargs_list, sample_weights


# =============================================================================
# TTT Adaptation
# =============================================================================

def build_ttt_adaptation_fn(agent: CQLSimSiamAgent):
    """
    Build TTT adaptation function that updates BOTH the encoder AND SimSiam components.

    IMPORTANT: The encoder is now a standalone module at modules_encoder.
    It is shared by actor/critic/simsiam through the _encode() method.
    To affect Q-values during TTT, we update modules_encoder and modules_simsiam.

    Args:
        agent: The CQLSimSiamAgent

    Returns:
        Adaptation function that returns ((adapted_encoder, adapted_simsiam), losses)
    """
    apply_fn = agent.state.apply_fn
    config = agent.config
    stop_grad_target = config.get("stop_grad_target", True)
    normalize_latents = config.get("normalize_latents", True)
    lambda_sim = config.get("lambda_sim", 2.0)
    goal_conditioned = config.get("goal_conditioned", False)
    uses_batch_norm = config.get("uses_batch_norm", False)

    # Get batch_stats if using batch norm (these are NOT adapted, just needed for forward pass)
    batch_stats = agent.state.batch_stats if uses_batch_norm else None

    @partial(jax.jit, static_argnames=("num_steps",))
    def _adapt_jit(full_params, batch, lr: float, num_steps: int):
        """
        JIT-compiled inner adaptation loop.

        We optimize over the FULL params dict, but only compute gradients
        w.r.t. encoder (modules_encoder) and simsiam components (modules_simsiam).

        Args:
            full_params: Complete agent params dict
            batch: Training batch
            lr: Learning rate
            num_steps: Number of adaptation steps
        """

        def _include_goals_in_obs(batch_dict, obs_key):
            """Include goals in observation if goal-conditioned."""
            obs = batch_dict[obs_key]
            if goal_conditioned and "goals" in batch_dict:
                return (obs, batch_dict["goals"])
            return obs

        def compute_loss(params):
            """Compute SimSiam loss given full params."""
            # Prepare inputs
            obs = _include_goals_in_obs(batch, "observations")
            next_obs = _include_goals_in_obs(batch, "next_observations")
            actions = batch["actions"]

            # Build variables dict - include batch_stats if using batch norm
            variables = {"params": params}
            if batch_stats is not None:
                variables["batch_stats"] = batch_stats

            # Step 1: Encode using standalone encoder module (modules_encoder)
            z_t = apply_fn(variables, obs, name="encoder")
            z_tp1 = apply_fn(variables, next_obs, name="encoder")

            # Step 2: Forward pass through SimSiam module (dynamics/projector/predictor)
            # train=False for TTT (use running stats, don't update them)
            simsiam_out = apply_fn(
                variables,
                z_t,
                z_tp1,
                actions,
                False,  # train=False
                name="simsiam",
            )

            proj_real = simsiam_out["proj_real"]
            prediction = simsiam_out["prediction"]

            # Stop gradient on target
            if stop_grad_target:
                proj_target = jax.lax.stop_gradient(proj_real)
            else:
                proj_target = proj_real

            # Cosine similarity loss
            loss = cosine_similarity_loss(
                prediction,
                proj_target,
                normalize=normalize_latents,
            )

            return lambda_sim * loss

        def adapt_step(i, carry):
            """Single adaptation step for fori_loop."""
            current_params, losses = carry

            loss, grads = jax.value_and_grad(compute_loss)(current_params)

            # Only update encoder (modules_encoder) and simsiam params (modules_simsiam)
            # Keep actor/critic/temperature frozen

            def make_update_mask(params):
                """Create a mask: 1.0 for params to update, 0.0 for frozen."""
                def _mask_fn(path, _):
                    # Convert path to string - path elements are DictKey/SequenceKey
                    path_keys = [p.key if hasattr(p, 'key') else str(p) for p in path]
                    path_str = "/".join(str(k) for k in path_keys)
                    # Update encoder (modules_encoder) and simsiam (modules_simsiam)
                    if "modules_encoder" in path_str:
                        return 1.0
                    if "modules_simsiam" in path_str:
                        return 1.0
                    return 0.0
                return jax.tree_util.tree_map_with_path(_mask_fn, params)

            mask = make_update_mask(current_params)

            # Apply masked SGD update: p_new = p - lr * mask * grad
            new_params = jax.tree_util.tree_map(
                lambda p, g, m: p - lr * m * g,
                current_params,
                grads,
                mask,
            )

            # Store loss
            losses = losses.at[i].set(loss)

            return new_params, losses

        # Initialize loss array
        losses = jnp.zeros(num_steps)

        # Run adaptation loop
        final_params, losses = jax.lax.fori_loop(
            0, num_steps, adapt_step, (full_params, losses)
        )

        return final_params, losses

    def adapt(simsiam_params, frozen_params, batch, lr: float, num_steps: int):
        """
        Run multiple TTT adaptation steps.

        Args:
            simsiam_params: Initial simsiam params (unused, kept for API compat)
            frozen_params: Full agent params dict
            batch: Training batch
            lr: Learning rate
            num_steps: Number of adaptation steps

        Returns:
            (adapted_encoder_params, adapted_simsiam_params), losses
        """
        # Run adaptation on full params
        adapted_full_params, losses = _adapt_jit(frozen_params, batch, lr, num_steps)

        # Extract the adapted encoder and simsiam params
        # Encoder is now at modules_encoder (not modules_actor/encoder)
        adapted_encoder = adapted_full_params.get("modules_encoder", {})
        adapted_simsiam = adapted_full_params.get("modules_simsiam", {})

        return (adapted_encoder, adapted_simsiam), losses

    return adapt


def _update_agent_simsiam(agent: CQLSimSiamAgent, adapted_params):
    """
    Update agent's encoder and simsiam parameters (for TTT).

    Args:
        agent: The agent to update
        adapted_params: Tuple of (adapted_encoder_params, adapted_simsiam_params)
                       OR just adapted_simsiam_params for backward compatibility
    """
    if adapted_params is None:
        logging.warning("TTT adaptation returned None params, skipping update")
        return agent

    # Handle both new format (encoder, simsiam) and old format (simsiam only)
    if isinstance(adapted_params, tuple) and len(adapted_params) == 2:
        adapted_encoder, adapted_simsiam = adapted_params
    else:
        # Backward compatibility - only simsiam params
        adapted_encoder = None
        adapted_simsiam = adapted_params

    # Ensure simsiam params are frozen
    if not isinstance(adapted_simsiam, flax.core.FrozenDict):
        adapted_simsiam = flax.core.freeze(dict(adapted_simsiam))

    # Build new params dict
    new_params_dict = dict(agent.state.params)
    new_params_dict["modules_simsiam"] = adapted_simsiam

    # Update encoder (now at modules_encoder, not modules_actor/encoder)
    if adapted_encoder is not None:
        if not isinstance(adapted_encoder, flax.core.FrozenDict):
            adapted_encoder = flax.core.freeze(dict(adapted_encoder))

        # Update standalone encoder module
        new_params_dict["modules_encoder"] = adapted_encoder

    new_params = flax.core.freeze(new_params_dict)

    # Same for target params
    new_target_dict = dict(agent.state.target_params)
    new_target_dict["modules_simsiam"] = adapted_simsiam

    if adapted_encoder is not None:
        new_target_dict["modules_encoder"] = adapted_encoder

    new_target_params = flax.core.freeze(new_target_dict)

    new_state = agent.state.replace(
        params=new_params,
        target_params=new_target_params
    )
    return agent.replace(state=new_state)


# =============================================================================
# Trajectory Processing for Value Plots
# =============================================================================

def process_trajectory_batch(traj_batch, text_processor=None):
    """
    Process a trajectory batch for evaluation/visualization.
    
    Args:
        traj_batch: Raw trajectory batch with (T, ...) or (B, T, ...) shaped arrays
        text_processor: Optional text processor for language
        
    Returns:
        Dict with trajectory data formatted for the agent
    """
    # Get images
    images = traj_batch["observation"]["image_primary"]
    next_images = traj_batch["next_observation"]["image_primary"]
    actions = traj_batch["action"]
    
    # Convert to JAX arrays
    images = jnp.asarray(images)
    next_images = jnp.asarray(next_images)
    actions = jnp.asarray(actions)
    
    # Handle various input shapes - squeeze to (T, H, W, C)
    while images.ndim > 4:
        if images.shape[0] == 1:
            images = images[0]
            next_images = next_images[0]
            actions = actions[0]
        elif images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
            next_images = next_images[:, 0]
        else:
            images = images[0]
            next_images = next_images[0]
            actions = actions[0]
    
    # Squeeze actions to (T, action_dim)
    while actions.ndim > 2:
        if actions.shape[1] == 1:
            actions = actions[:, 0]
        else:
            actions = actions.reshape(actions.shape[0], -1)
    
    T = images.shape[0]
    
    # Process language
    language = traj_batch["task"]["language_instruction"]
    if isinstance(language, (str, bytes)):
        lang_str = language.decode('utf-8') if isinstance(language, bytes) else language
    elif isinstance(language, (list, np.ndarray)):
        lang_entry = language[0] if len(language) > 0 else ""
        if isinstance(lang_entry, (list, np.ndarray)):
            lang_entry = lang_entry[0] if len(lang_entry) > 0 else ""
        lang_str = lang_entry.decode('utf-8') if isinstance(lang_entry, bytes) else str(lang_entry)
    else:
        lang_str = str(language)
    
    # Process language embeddings if processor available
    if text_processor is not None:
        language_emb = text_processor.encode([lang_str] * T)
    else:
        language_emb = None
    
    # Process rewards and masks
    rewards = jnp.asarray(traj_batch.get("reward", jnp.zeros(T)))
    masks = jnp.asarray(traj_batch.get("td_mask", jnp.ones(T)))
    mc_returns = jnp.asarray(traj_batch.get("mc_return", rewards))
    
    # Flatten to (T,)
    while rewards.ndim > 1:
        rewards = rewards.reshape(-1)
    while masks.ndim > 1:
        masks = masks.reshape(-1)
    while mc_returns.ndim > 1:
        mc_returns = mc_returns.reshape(-1)
    
    # Trim to T
    rewards = rewards[:T]
    masks = masks[:T]
    mc_returns = mc_returns[:T]
    
    result = {
        "observations": {"image": images},
        "next_observations": {"image": next_images},
        "actions": actions,
        "rewards": rewards,
        "masks": masks,
        "mc_returns": mc_returns,
        "language_str": lang_str,
        "raw_images": np.asarray(jax.device_get(images)),  # For visualization
    }
    
    if language_emb is not None:
        result["goals"] = {"language": language_emb}
    
    return result


def plot_value_overlay(
    agent_adapted: CQLSimSiamAgent,
    traj: Dict[str, Any],
    agent_base: CQLSimSiamAgent,
    seed: Optional[jax.Array] = None,
    title_suffix: str = "",
) -> np.ndarray:
    """
    Plot Q-values over time comparing TTT-adapted vs non-adapted agent.

    This function uses get_eval_values to compute Q-values at each timestep,
    showing how the value estimates evolve along the trajectory.

    Args:
        agent_adapted: TTT-adapted agent (or agent to show as "adapted" line)
        traj: Trajectory dict
        agent_base: Base agent (non-adapted) for comparison
        seed: Random seed
        title_suffix: Additional title text

    Returns:
        RGB image array of the plot
    """
    if seed is None:
        seed = jax.random.PRNGKey(0)

    goals = traj.get("goals", {})

    # Get Q-values and other metrics using get_eval_values from BOTH agents
    metrics_adapted = agent_adapted.get_eval_values(traj, seed, goals)
    metrics_base = agent_base.get_eval_values(traj, seed, goals)

    # Get images for visualization
    visuals = traj.get("raw_images", traj["observations"]["image"])
    images = np.asarray(visuals)
    if images.ndim == 3:
        images = images[None, ...]
    T = images.shape[0]

    # Metrics to plot (from get_eval_values: q, target_q, mse, rewards, masks)
    metric_keys = list(metrics_adapted.keys())
    num_rows = len(metric_keys) + 1  # +1 for image row

    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows))
    if num_rows == 1:
        axs = [axs]
    canvas = FigureCanvas(fig)

    # Row 0: Sample trajectory frames
    interval = max(1, T // 8)
    sel_images = images[::interval]
    if sel_images.shape[0] > 0:
        flattened = np.concatenate([np.asarray(frame).squeeze() for frame in sel_images], axis=1)
        if flattened.max() > 1:
            flattened = flattened.astype(np.uint8)
        axs[0].imshow(flattened)
        axs[0].set_title(f'Trajectory frames (T={T}){title_suffix}')
    axs[0].axis('off')

    # Remaining rows: Value metrics over time
    for idx, key in enumerate(metric_keys, start=1):
        series_adapted = np.asarray(metrics_adapted[key])
        series_base = np.asarray(metrics_base.get(key, series_adapted))

        # Trim to common length
        plot_len = min(len(series_adapted), len(series_base), T)
        if plot_len == 0:
            axs[idx].set_ylabel(key)
            axs[idx].set_title('No data')
            continue

        steps = np.arange(plot_len)

        if key in ('rewards', 'masks'):
            # Non-TTT-dependent quantities - just plot once
            axs[idx].plot(steps, series_adapted[:plot_len],
                         label=key, linestyle='-', marker='o', markersize=3)
        else:
            # Plot both TTT-adapted and base
            axs[idx].plot(steps, series_adapted[:plot_len],
                         label='TTT-adapted', linestyle='-', marker='o', markersize=3, color='blue')
            axs[idx].plot(steps, series_base[:plot_len],
                         label='Base (no TTT)', linestyle='--', marker='x', markersize=3, color='orange')
            axs[idx].legend(loc='best', fontsize=8)

        # Set y-axis limits with margin
        combined = np.concatenate([series_adapted[:plot_len], series_base[:plot_len]])
        if combined.size > 0 and not np.all(np.isnan(combined)):
            ymin, ymax = np.nanmin(combined), np.nanmax(combined)
            margin = max(0.1 * (ymax - ymin), 0.1)
            axs[idx].set_ylim([ymin - margin, ymax + margin])

        axs[idx].set_ylabel(key)
        axs[idx].grid(True, alpha=0.3)

    axs[-1].set_xlabel('Timestep')
    plt.tight_layout()

    # Convert to image
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return out_image


# Alias for backward compatibility
plot_value_comparison = plot_value_overlay


def run_ttt_adaptation_on_trajectory(
    agent: CQLSimSiamAgent,
    traj: Dict[str, Any],
    ttt_adapt_fn,
    ttt_lr: float,
    ttt_steps: int,
    mode: str = "online",  # "reset", "cumulative", or "online"
) -> CQLSimSiamAgent:
    """
    Run TTT adaptation on a trajectory.

    Now adapts BOTH the encoder (in modules_actor) AND simsiam components.
    This is necessary because Q-values depend on the encoder.

    Args:
        agent: Base agent
        traj: Trajectory dict
        ttt_adapt_fn: TTT adaptation function (returns (encoder, simsiam), losses)
        ttt_lr: Learning rate
        ttt_steps: Number of adaptation steps
        mode:
            - "reset": Single adaptation on whole trajectory from base params
            - "cumulative": Multiple passes on whole trajectory
            - "online": Step-by-step adaptation, keeping params within trajectory

    Returns:
        Agent with adapted encoder and simsiam module
    """
    if ttt_adapt_fn is None or ttt_steps <= 0:
        logging.warning("TTT adaptation skipped: adapt_fn=%s, ttt_steps=%d",
                       ttt_adapt_fn is not None, ttt_steps)
        return agent

    T = traj["observations"]["image"].shape[0]
    base_simsiam_params = agent.state.params.get("modules_simsiam")
    base_encoder_params = agent.state.params.get("modules_encoder")

    if base_simsiam_params is None:
        logging.error("No 'modules_simsiam' key in agent params for trajectory adaptation! Keys: %s",
                     list(agent.state.params.keys()))
        return agent

    if base_encoder_params is None:
        logging.error("No 'modules_encoder' key in agent params for trajectory adaptation! Keys: %s",
                     list(agent.state.params.keys()))
        return agent

    # Debug: compute param stats before adaptation
    def _param_stats(params, name):
        flat = jax.tree_util.tree_leaves(params)
        all_vals = jnp.concatenate([jnp.ravel(x) for x in flat])
        return f"{name}: mean={float(jnp.mean(all_vals)):.6f}, std={float(jnp.std(all_vals)):.6f}, norm={float(jnp.linalg.norm(all_vals)):.4f}"

    logging.info("TTT Debug [%s mode, T=%d, steps=%d, lr=%s]", mode, T, ttt_steps, ttt_lr)
    logging.info("  Before: %s", _param_stats(base_simsiam_params, "simsiam"))
    logging.info("  Before: %s", _param_stats(base_encoder_params, "encoder"))

    all_losses = []

    # Helper to rebuild full params with current encoder and simsiam for online mode
    def _rebuild_full_params(current_encoder, current_simsiam):
        """Rebuild agent params with updated encoder and simsiam for next iteration."""
        new_params = dict(agent.state.params)
        new_params["modules_encoder"] = current_encoder
        new_params["modules_simsiam"] = current_simsiam
        return flax.core.freeze(new_params)

    # Alias for backward compat
    def _rebuild_params_with_encoder(current_encoder):
        return _rebuild_full_params(current_encoder, base_simsiam_params)

    if mode == "online":
        # Online mode: adapt step-by-step, keeping params across timesteps
        # This simulates real online inference where we adapt as we go
        current_encoder = base_encoder_params
        current_simsiam = base_simsiam_params
        current_full_params = agent.state.params

        for t in range(T - 1):  # T-1 because we need (obs_t, obs_{t+1}, action_t)
            # Create single-step batch
            step_batch = {
                "observations": jax.tree_map(lambda x: x[t:t+1], traj["observations"]),
                "next_observations": jax.tree_map(lambda x: x[t:t+1], traj["next_observations"]),
                "actions": traj["actions"][t:t+1],
            }
            if "goals" in traj:
                step_batch["goals"] = jax.tree_map(lambda x: x[t:t+1], traj["goals"])

            # Adapt from current params (accumulate changes)
            (current_encoder, current_simsiam), losses = ttt_adapt_fn(
                current_simsiam,
                current_full_params,
                step_batch,
                ttt_lr,
                ttt_steps,
            )
            all_losses.append(losses)

            # Rebuild full params with updated encoder AND simsiam for next iteration
            current_full_params = _rebuild_full_params(current_encoder, current_simsiam)

        adapted_params = (current_encoder, current_simsiam)

    elif mode == "reset":
        # Single adaptation on whole trajectory from base params
        traj_batch = {
            "observations": traj["observations"],
            "next_observations": traj["next_observations"],
            "actions": traj["actions"],
        }
        if "goals" in traj:
            traj_batch["goals"] = traj["goals"]

        adapted_params, losses = ttt_adapt_fn(
            base_simsiam_params,
            agent.state.params,
            traj_batch,
            ttt_lr,
            ttt_steps,
        )
        all_losses.append(losses)

    else:  # cumulative - multiple passes on whole trajectory
        traj_batch = {
            "observations": traj["observations"],
            "next_observations": traj["next_observations"],
            "actions": traj["actions"],
        }
        if "goals" in traj:
            traj_batch["goals"] = traj["goals"]

        current_simsiam = base_simsiam_params
        current_encoder = base_encoder_params
        current_full_params = agent.state.params
        for _ in range(3):  # Multiple adaptation rounds
            (current_encoder, current_simsiam), losses = ttt_adapt_fn(
                current_simsiam,
                current_full_params,
                traj_batch,
                ttt_lr,
                ttt_steps,
            )
            all_losses.append(losses)
            # Update params for next round
            current_full_params = _rebuild_full_params(current_encoder, current_simsiam)

        adapted_params = (current_encoder, current_simsiam)

    # Debug: log adaptation results
    # adapted_params is now (encoder, simsiam) tuple
    adapted_encoder, adapted_simsiam = adapted_params
    logging.info("  After:  %s", _param_stats(adapted_simsiam, "simsiam"))
    logging.info("  After:  %s", _param_stats(adapted_encoder, "encoder"))

    # Compute param difference for simsiam
    def _param_diff(p1, p2):
        flat1 = jax.tree_util.tree_leaves(p1)
        flat2 = jax.tree_util.tree_leaves(p2)
        diff = jnp.concatenate([jnp.ravel(a - b) for a, b in zip(flat1, flat2)])
        return float(jnp.linalg.norm(diff)), float(jnp.max(jnp.abs(diff)))

    sim_diff_norm, sim_diff_max = _param_diff(adapted_simsiam, base_simsiam_params)
    enc_diff_norm, enc_diff_max = _param_diff(adapted_encoder, base_encoder_params)
    logging.info("  Simsiam param change: norm=%.6f, max=%.6f", sim_diff_norm, sim_diff_max)
    logging.info("  Encoder param change: norm=%.6f, max=%.6f", enc_diff_norm, enc_diff_max)

    # Log losses
    if all_losses:
        if mode == "online":
            # For online, log first and last step losses
            first_losses = np.asarray(all_losses[0])
            last_losses = np.asarray(all_losses[-1])
            logging.info("  Losses - first step: [%.4f -> %.4f], last step: [%.4f -> %.4f]",
                        first_losses[0], first_losses[-1], last_losses[0], last_losses[-1])
        else:
            stacked = np.asarray(all_losses)
            logging.info("  Losses: initial=%.4f, final=%.4f",
                        stacked[0, 0], stacked[-1, -1])

    return _update_agent_simsiam(agent, adapted_params)


# =============================================================================
# Main Training Loop
# =============================================================================

def main(_):
    # Disable GPU for TensorFlow (we use JAX)
    tf.config.set_visible_devices([], "GPU")

    # ==========================================================================
    # Multi-device sharding setup
    # ==========================================================================
    # We use data parallelism: batch is sharded across devices, model is replicated.
    # JAX's SPMD (via PositionalSharding) automatically handles gradient all-reduce.
    #
    # For batch normalization:
    # - Each device computes local batch statistics on its shard
    # - Since agent is replicated, batch_stats remain synchronized across devices
    # - For exact cross-device statistics, explicit jax.lax.pmean would be needed
    #   inside the forward pass (not implemented here for simplicity)
    # ==========================================================================
    devices = jax.local_devices()
    num_devices = len(devices)
    logging.info(f"Found {num_devices} devices: {devices}")

    # Validate batch size is divisible by number of devices
    batch_size = FLAGS.oxedata_config.batch_size
    assert batch_size % num_devices == 0, (
        f"Batch size {batch_size} must be divisible by number of devices {num_devices}"
    )
    logging.info(f"Batch size: {batch_size}, per device: {batch_size // num_devices}")

    # Create sharding for data parallelism
    # - Batches are sharded on first dimension across devices
    # - Agent (params, batch_stats) is replicated across all devices
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Setup text processor
    text_processor = _maybe_text_processor()
    
    # Setup WandB logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": FLAGS.project,
        "exp_descriptor": FLAGS.name
    })
    variant = FLAGS.config.to_dict()
    variant["oxe_config"] = FLAGS.oxedata_config.to_dict()
    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)
    
    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )
    
    # ==========================================================================
    # Setup Datasets
    # ==========================================================================
    
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
        del FLAGS.oxedata_config["oxe_kwargs"]
    
    train_datasets_kwargs_list, train_sample_weights = _safe_filter_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    
    train_data = make_interleaved_dataset(
        dataset_kwargs_list=train_datasets_kwargs_list,
        sample_weights=train_sample_weights,
        train=True,
        shuffle_buffer_size=FLAGS.oxedata_config["shuffle_buffer_size"],
        traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
        frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
        batch_size=FLAGS.oxedata_config["batch_size"],
        balance_weights=FLAGS.oxedata_config.get("balance_weights", False),
        traj_transform_threads=FLAGS.oxedata_config.get("traj_transform_threads", None),
        traj_read_threads=FLAGS.oxedata_config.get("traj_read_threads", None),
    )
    
    val_datasets_kwargs_list, _ = _safe_filter_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    val_data = create_validation_dataset(
        val_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
    )
    
    # Create trajectory dataset for value plots (full trajectories, not transitions)
    # Use larger window or no subsampling to get full trajectories
    viz_traj_transform_kwargs = dict(FLAGS.oxedata_config["traj_transform_kwargs"])
    viz_traj_transform_kwargs["subsample_length"] = None  # Don't subsample
    if "window_size" in viz_traj_transform_kwargs:
        viz_traj_transform_kwargs["window_size"] = 100  # Larger window for trajectories
    
    val_traj_data = create_validation_dataset(
        val_datasets_kwargs_list[0],
        viz_traj_transform_kwargs,
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
    )
    
    def make_iter(dataset, apply_sharding: bool = True) -> Iterator[Dict[str, Any]]:
        """Create iterator with optional sharding across devices."""
        for batch in dataset.iterator(prefetch=0):
            batch = jax.tree_map(_to_jax_array, batch)
            processed = _process_oxe_batch(batch, text_processor)
            if apply_sharding:
                processed = shard_fn(processed)
            yield processed

    # Training data iterator with sharding
    train_data_iter = make_iter(train_data, apply_sharding=True)

    # Validation data iterator with sharding
    val_data_iter = make_iter(
        val_data.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(FLAGS.oxedata_config["batch_size"]),
        apply_sharding=True,
    )
    
    # Trajectory iterator for value plots (returns full trajectories)
    val_traj_iter = val_traj_data.shuffle(100).repeat().iterator(prefetch=0)
    
    # ==========================================================================
    # Create Agent
    # ==========================================================================
    
    example_batch = next(train_data_iter)
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, agent_rng = jax.random.split(rng)
    
    # Create unified CQL+SimSiam agent
    agent = agents[FLAGS.config.agent].create(
        rng=agent_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )
    
    logging.info("Created CQLSimSiamAgent")
    logging.info(f"  lambda_sim: {agent.config.get('lambda_sim', 0.0)}")
    logging.info(f"  shared_encoder: {agent.config.get('shared_encoder', True)}")
    logging.info(f"  param keys: {list(agent.state.params.keys())}")

    # Resume from checkpoint if specified
    if FLAGS.config.resume_path:
        checkpoint = checkpoints.restore_checkpoint(
            FLAGS.config.resume_path,
            target={"agent": agent}
        )
        agent = checkpoint["agent"]
        logging.info("Restored agent from %s", FLAGS.config.resume_path)

    # Replicate agent across all devices for data parallelism
    # The agent params/state are replicated, while batches are sharded
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
    logging.info("Replicated agent across %d devices", num_devices)
    
    # ==========================================================================
    # Setup TTT Adaptation (if enabled)
    # ==========================================================================
    
    ttt_mode = str(FLAGS.config.get("ttt_mode", "off")).lower()
    ttt_enabled = (
        ttt_mode in ("standard", "online") and
        FLAGS.config.get("ttt_steps", 0) > 0 and
        agent.config.get("lambda_sim", 0.0) > 0
    )
    
    if ttt_enabled:
        ttt_adapt_fn = build_ttt_adaptation_fn(agent)
        logging.info(f"TTT enabled: mode={ttt_mode}, steps={FLAGS.config.ttt_steps}, lr={FLAGS.config.ttt_lr}")
    else:
        ttt_adapt_fn = None
        logging.info("TTT disabled")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================

    timer = Timer()

    # JIT-compile training step for efficiency (like train_embedding.py)
    @jax.jit
    def train_step(agent, batch):
        agent, update_info = agent.update(batch)
        # Aggregate metrics across devices (they're sharded like the batch)
        # With PositionalSharding, metrics are sharded - take mean to aggregate
        update_info = jax.tree_map(
            lambda x: jnp.mean(x) if isinstance(x, jnp.ndarray) else x,
            update_info
        )
        return agent, update_info

    for step in tqdm.tqdm(range(int(FLAGS.config.num_steps)), desc="Training"):
        timer.tick("total")

        try:
            # Get training batch
            batch = next(train_data_iter)

            # =======================================================================
            # Single unified update (handles both RL and SimSiam losses)
            # =======================================================================
            timer.tick("train")
            agent, update_info = train_step(agent, batch)
            timer.tock("train")

            # =======================================================================
            # Logging
            # =======================================================================
            if (step + 1) % FLAGS.config.log_interval == 0:
                # Log training metrics
                metrics = jax.device_get(update_info)

                # Debug: print metric shapes and sharding info
                if step < 2000:  # Only log for first few intervals
                    logging.info("=== Metrics Debug ===")
                    for k, v in metrics.items():
                        if hasattr(v, 'shape'):
                            logging.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, value={v}")
                        else:
                            logging.info(f"  {k}: type={type(v)}, value={v}")

                # Separate RL and SimSiam metrics for cleaner logging
                rl_metrics = {k: v for k, v in metrics.items() if not k.startswith("simsiam/")}
                simsiam_metrics = {k: v for k, v in metrics.items() if k.startswith("simsiam/")}

                wandb_logger.log({"training/rl": rl_metrics}, step=step)
                if simsiam_metrics:
                    wandb_logger.log({"training/simsiam": simsiam_metrics}, step=step)

                wandb_logger.log({"timer": timer.get_average_times(reset=False)}, step=step)

            # =======================================================================
            # Evaluation (with optional TTT)
            # =======================================================================
            if (step + 1) % FLAGS.config.eval_interval == 0:
                logging.info("Evaluating...")

                num_eval_batches = int(FLAGS.config.get("eval_batches", 4))
                eval_metrics = []
                ttt_loss_logs = []

                # Store base simsiam params for TTT
                base_simsiam_params = agent.state.params.get("modules_simsiam")
                if base_simsiam_params is None:
                    logging.error("No 'modules_simsiam' key in agent params! Keys: %s", list(agent.state.params.keys()))
                online_simsiam_params = base_simsiam_params

                for eval_idx in range(num_eval_batches):
                    val_batch = next(val_data_iter)
                    rng, val_rng = jax.random.split(rng)

                    eval_agent = agent
                    adapt_losses = None

                    # TTT adaptation if enabled
                    if ttt_enabled and ttt_adapt_fn is not None:
                        # Choose starting params based on mode
                        start_params = (
                            base_simsiam_params if ttt_mode == "standard"
                            else online_simsiam_params
                        )

                        # Get frozen params (everything except simsiam)
                        frozen_params = agent.state.params

                        # Adapt
                        adapted_params, adapt_losses = ttt_adapt_fn(
                            start_params,
                            frozen_params,
                            val_batch,
                            FLAGS.config.ttt_lr,
                            FLAGS.config.ttt_steps,
                        )

                        # Update online params for next batch
                        online_simsiam_params = adapted_params

                        # Create eval agent with adapted simsiam
                        eval_agent = _update_agent_simsiam(agent, adapted_params)

                    # Get evaluation metrics
                    batch_metrics = eval_agent.get_debug_metrics(val_batch, seed=val_rng)
                    eval_metrics.append(batch_metrics)

                    if adapt_losses is not None and len(adapt_losses) > 0:
                        ttt_loss_logs.append(adapt_losses)

                # Aggregate evaluation metrics
                aggregated_metrics = jax.tree_map(
                    lambda *xs: np.mean([np.asarray(x) for x in xs]),
                    *eval_metrics
                )
                wandb_logger.log({"validation": jax.device_get(aggregated_metrics)}, step=step)

                # Log TTT metrics
                if ttt_loss_logs:
                    stacked = jnp.stack(ttt_loss_logs)
                    stacked_np = np.asarray(jax.device_get(stacked))
                    wandb_logger.log({
                        "validation_ttt/loss_initial": float(stacked_np[:, 0].mean()),
                        "validation_ttt/loss_final": float(stacked_np[:, -1].mean()),
                        "validation_ttt/loss_reduction": float(
                            (stacked_np[:, 0] - stacked_np[:, -1]).mean()
                        ),
                    }, step=step)

                # ===================================================================
                # Value Function Plots
                # ===================================================================
                num_value_plots = int(FLAGS.config.get("num_value_plots", 2))
                if num_value_plots > 0:
                    logging.info(f"Generating {num_value_plots} value plots...")

                    prev_traj_language = None

                    for plot_idx in range(num_value_plots):
                        try:
                            # Get a trajectory for visualization
                            traj_raw = next(val_traj_iter)
                            traj_raw = jax.tree_map(_to_jax_array, traj_raw)

                            # Process trajectory
                            traj_processed = process_trajectory_batch(traj_raw, text_processor)
                            T = traj_processed["observations"]["image"].shape[0]
                            lang_str = traj_processed.get("language_str", "unknown")

                            logging.info(f"  Plot {plot_idx}: T={T}, language='{lang_str[:50]}...'")

                            rng, plot_rng = jax.random.split(rng)

                            # ---------------------------------------------------------
                            # Plot 1: TTT-adapted (online mode) - step-by-step within traj
                            # ---------------------------------------------------------
                            if ttt_enabled and ttt_adapt_fn is not None:
                                agent_adapted_online = run_ttt_adaptation_on_trajectory(
                                    agent,
                                    traj_processed,
                                    ttt_adapt_fn,
                                    FLAGS.config.ttt_lr,
                                    FLAGS.config.ttt_steps,
                                    mode="online",
                                )

                                # =======================================================
                                # DEBUG: Compare Q-values and encoder outputs
                                # =======================================================
                                goals = traj_processed.get("goals", {})

                                # Get Q-values from both agents
                                q_base = agent.get_eval_values(traj_processed, plot_rng, goals)
                                q_adapted = agent_adapted_online.get_eval_values(traj_processed, plot_rng, goals)

                                q_base_arr = np.asarray(q_base.get("q", []))
                                q_adapted_arr = np.asarray(q_adapted.get("q", []))

                                if len(q_base_arr) > 0 and len(q_adapted_arr) > 0:
                                    q_diff = q_adapted_arr - q_base_arr
                                    logging.info("  Q-value comparison (online TTT vs base):")
                                    logging.info("    Base Q:    mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                                               np.mean(q_base_arr), np.std(q_base_arr),
                                               np.min(q_base_arr), np.max(q_base_arr))
                                    logging.info("    Adapted Q: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                                               np.mean(q_adapted_arr), np.std(q_adapted_arr),
                                               np.min(q_adapted_arr), np.max(q_adapted_arr))
                                    logging.info("    Q diff:    mean=%.6f, std=%.6f, max_abs=%.6f",
                                               np.mean(q_diff), np.std(q_diff), np.max(np.abs(q_diff)))

                                # Compare encoder outputs (simsiam module)
                                # Get a sample observation for encoder comparison
                                sample_obs = jax.tree_map(lambda x: x[:1], traj_processed["observations"])
                                sample_next_obs = jax.tree_map(lambda x: x[:1], traj_processed["next_observations"])
                                sample_actions = traj_processed["actions"][:1]

                                # Forward through simsiam for base agent
                                try:
                                    simsiam_base = agent.state.apply_fn(
                                        {"params": agent.state.params},
                                        sample_obs,
                                        sample_next_obs,
                                        sample_actions,
                                        False,
                                        name="simsiam",
                                    )
                                    simsiam_adapted = agent_adapted_online.state.apply_fn(
                                        {"params": agent_adapted_online.state.params},
                                        sample_obs,
                                        sample_next_obs,
                                        sample_actions,
                                        False,
                                        name="simsiam",
                                    )

                                    proj_base = np.asarray(simsiam_base["proj_real"])
                                    proj_adapted = np.asarray(simsiam_adapted["proj_real"])
                                    proj_diff = proj_adapted - proj_base

                                    logging.info("  SimSiam projection comparison:")
                                    logging.info("    Base proj:    mean=%.4f, std=%.4f, norm=%.4f",
                                               np.mean(proj_base), np.std(proj_base), np.linalg.norm(proj_base))
                                    logging.info("    Adapted proj: mean=%.4f, std=%.4f, norm=%.4f",
                                               np.mean(proj_adapted), np.std(proj_adapted), np.linalg.norm(proj_adapted))
                                    logging.info("    Proj diff:    mean=%.6f, std=%.6f, max_abs=%.6f, norm=%.6f",
                                               np.mean(proj_diff), np.std(proj_diff),
                                               np.max(np.abs(proj_diff)), np.linalg.norm(proj_diff))

                                    # Also check prediction output
                                    pred_base = np.asarray(simsiam_base["prediction"])
                                    pred_adapted = np.asarray(simsiam_adapted["prediction"])
                                    pred_diff = pred_adapted - pred_base
                                    logging.info("  SimSiam prediction comparison:")
                                    logging.info("    Pred diff:    mean=%.6f, std=%.6f, max_abs=%.6f, norm=%.6f",
                                               np.mean(pred_diff), np.std(pred_diff),
                                               np.max(np.abs(pred_diff)), np.linalg.norm(pred_diff))
                                except Exception as e:
                                    logging.warning("  Could not compare simsiam outputs: %s", e)

                                # Compare simsiam params directly
                                base_simsiam = agent.state.params.get("modules_simsiam", {})
                                adapted_simsiam = agent_adapted_online.state.params.get("modules_simsiam", {})
                                if base_simsiam and adapted_simsiam:
                                    base_flat = jax.tree_util.tree_leaves(base_simsiam)
                                    adapted_flat = jax.tree_util.tree_leaves(adapted_simsiam)
                                    param_diff = sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(adapted_flat, base_flat))
                                    logging.info("  Param diff (sum sq): %.8f", param_diff)

                                # DEBUG: Show param structure to understand what we're updating
                                logging.info("  Agent param keys: %s", list(agent.state.params.keys()))
                                if base_simsiam:
                                    logging.info("  modules_simsiam keys: %s", list(base_simsiam.keys()))
                                    for k, v in base_simsiam.items():
                                        if hasattr(v, 'keys'):
                                            logging.info("    %s subkeys: %s", k, list(v.keys()))

                                # Check modules_actor structure too
                                base_actor = agent.state.params.get("modules_actor", {})
                                if base_actor:
                                    logging.info("  modules_actor keys: %s", list(base_actor.keys()))

                                # Check modules_critic structure
                                base_critic = agent.state.params.get("modules_critic", {})
                                if base_critic:
                                    logging.info("  modules_critic keys: %s", list(base_critic.keys()))
                                # =======================================================
                                # END DEBUG
                                # =======================================================

                                plot_img_online = plot_value_comparison(
                                    agent_adapted_online,  # adapted agent
                                    traj_processed,        # trajectory
                                    agent,                 # base agent for comparison
                                    seed=plot_rng,
                                    title_suffix=" [TTT-online]",
                                )
                                wandb_logger.log({
                                    f"value_plots/traj_{plot_idx}_ttt_online": wandb.Image(plot_img_online)
                                }, step=step)

                                # ---------------------------------------------------------
                                # Plot 2: TTT-adapted (reset mode) - batch adaptation
                                # ---------------------------------------------------------
                                agent_adapted_reset = run_ttt_adaptation_on_trajectory(
                                    agent,
                                    traj_processed,
                                    ttt_adapt_fn,
                                    FLAGS.config.ttt_lr,
                                    FLAGS.config.ttt_steps,
                                    mode="reset",
                                )

                                plot_img_reset = plot_value_comparison(
                                    agent_adapted_reset,   # adapted agent
                                    traj_processed,        # trajectory
                                    agent,                 # base agent for comparison
                                    seed=plot_rng,
                                    title_suffix=" [TTT-reset]",
                                )
                                wandb_logger.log({
                                    f"value_plots/traj_{plot_idx}_ttt_reset": wandb.Image(plot_img_reset)
                                }, step=step)

                            # ---------------------------------------------------------
                            # Plot 3: Base (no TTT) - always generate this
                            # For base plot, both agents are the same (no adaptation)
                            # ---------------------------------------------------------
                            plot_img_base = plot_value_comparison(
                                agent,            # same agent for both
                                traj_processed,   # trajectory
                                agent,            # same agent (no difference expected)
                                seed=plot_rng,
                                title_suffix=" [Base - No TTT]",
                            )
                            wandb_logger.log({
                                f"value_plots/traj_{plot_idx}_base": wandb.Image(plot_img_base)
                            }, step=step)

                            # ---------------------------------------------------------
                            # Plot 4: Random language comparison (if we have prev trajectory)
                            # ---------------------------------------------------------
                            if prev_traj_language is not None and text_processor is not None:
                                # Re-encode with different (wrong) language
                                traj_wrong_lang = copy.deepcopy(traj_processed)
                                wrong_lang_emb = text_processor.encode(
                                    [prev_traj_language] * T
                                )
                                traj_wrong_lang["goals"] = {"language": wrong_lang_emb}

                                if ttt_enabled and ttt_adapt_fn is not None:
                                    agent_wrong_adapted = run_ttt_adaptation_on_trajectory(
                                        agent,
                                        traj_wrong_lang,
                                        ttt_adapt_fn,
                                        FLAGS.config.ttt_lr,
                                        FLAGS.config.ttt_steps,
                                        mode="online",
                                    )

                                    plot_img_wrong = plot_value_comparison(
                                        agent_wrong_adapted,  # adapted agent (wrong lang)
                                        traj_wrong_lang,      # trajectory with wrong lang
                                        agent,                # base agent for comparison
                                        seed=plot_rng,
                                        title_suffix=" [Wrong Language + TTT-online]",
                                    )
                                else:
                                    plot_img_wrong = plot_value_comparison(
                                        agent,            # same agent for both
                                        traj_wrong_lang,  # trajectory with wrong lang
                                        agent,            # same agent (no TTT)
                                        seed=plot_rng,
                                        title_suffix=" [Wrong Language]",
                                    )

                                wandb_logger.log({
                                    f"value_plots/traj_{plot_idx}_wrong_lang": wandb.Image(plot_img_wrong)
                                }, step=step)

                            # Store language for next iteration
                            prev_traj_language = lang_str

                        except Exception as e:
                            logging.warning(f"Failed to generate value plot {plot_idx}: {e}")
                            continue

            # =======================================================================
            # Checkpointing
            # =======================================================================
            if (step + 1) % FLAGS.config.save_interval == 0:
                logging.info("Saving checkpoint at step %d...", step + 1)
                checkpoints.save_checkpoint(
                    save_dir,
                    {"agent": agent, "step": step + 1},
                    step=step + 1,
                    keep=10,
                )

        except Exception as e:
            logging.error(f"Error at step {step}: {e}")
            raise
        finally:
            timer.tock("total")
    
    # Final save
    logging.info("Training complete. Saving final checkpoint...")
    checkpoints.save_checkpoint(
        save_dir,
        {"agent": agent, "step": FLAGS.config.num_steps},
        step=FLAGS.config.num_steps,
        keep=10,
    )
    
    wandb_logger.close()


if __name__ == "__main__":
    app.run(main)