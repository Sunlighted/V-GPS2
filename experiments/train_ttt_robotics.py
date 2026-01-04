"""
Training script for TTT Robotics - Action-Conditioned Version

Data flow:
    Image + Language → OCTO → obs_embedding (384/512)
                                    ↓
                         ┌──────────────────────┐
                         │   TTT Layer          │
                         │                      │
                         │ SSL: f(θ_K @ [obs, action]; W) → θ_V @ next_obs
                         │                      │
                         │ Output: f(θ_Q @ obs; W*) → features
                         └──────────┬───────────┘
                                    ↓
                              features (256)
                                    ↓
                         ┌──────────────────────┐
                         │   RL Agent (CQL)     │
                         │                      │
                         │ Q(features, action)  │
                         │ π(action | features) │
                         └──────────────────────┘

SSL Modes:
- "reconstruction": f(θ_K @ obs; W) → θ_V @ obs (paper-style, ignores actions)
- "prediction": f(θ_K @ [obs, action]; W) → θ_V @ next_obs (better for robotics)
"""

import copy
import os
from functools import partial
from pathlib import Path
import atexit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags, ConfigDict
import optax
import wandb

tf.get_logger().setLevel('ERROR')
from jax.experimental.compilation_cache import compilation_cache

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.model.octo_model import OctoModel

from ttt_robotics_layer import (
    TTTRoboticsAdapter,
    create_ttt_train_step,
    process_trajectory_with_ttt,
    process_trajectory_no_adapt,
    debug_ttt_state,
    print_diagnostics,
)

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

try:
    from jax_smi import initialise_tracking
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "ttt_robotics", "WandB project name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "oxedata_config",
    None,
    "File path to the OXE data configuration.",
    lock_config=False,
)


# ============================================================================
# OCTO Embedding Extraction
# ============================================================================

def get_octo_embeddings_batch(octo_model, images, language_list):
    """Extract fused vision⊗language embeddings from OCTO."""
    B = images.shape[0]
    images = jnp.asarray(images, dtype=jnp.float32)
    height, width = images.shape[1:3]
    channels = images.shape[-1]
    
    if (height, width) != (256, 256):
        from jax.image import resize
        images = resize(images, (B, 256, 256, channels), method='bilinear')
    
    images_octo = images[:, None, :, :, :]
    tasks = octo_model.create_tasks(texts=language_list)
    timestep_pad_mask = jnp.ones((B, 1), dtype=jnp.bool_)
    
    obs = {'image_primary': images_octo}
    output = octo_model.run_transformer(obs, tasks, timestep_pad_mask, train=False)
    
    readout_tokens = output['readout_action'].tokens
    embeddings = readout_tokens[:, 0].mean(axis=1)
    
    return embeddings


def process_batch_with_octo(batch, octo_model):
    """
    Process batch and extract OCTO embeddings.
    
    Returns dict with:
    - obs_embeddings: (B, obs_dim) current observation
    - next_obs_embeddings: (B, obs_dim) next observation
    - actions: (B, action_dim) actions taken
    - rewards, masks, mc_returns, etc.
    """
    obs_images = batch["observation"]["image_primary"]
    next_obs_images = batch["next_observation"]["image_primary"]
    actions = batch["action"]
    
    # Handle various image shapes
    # Could be (B, H, W, C), (B, T, H, W, C), (B, T, 1, H, W, C), etc.
    obs_images = jnp.asarray(obs_images)
    next_obs_images = jnp.asarray(next_obs_images)
    
    # Squeeze down to (B, H, W, C)
    while obs_images.ndim > 4:
        if obs_images.shape[1] == 1:
            obs_images = obs_images[:, 0]
        else:
            # Take first timestep if multiple
            obs_images = obs_images[:, 0]
    
    while next_obs_images.ndim > 4:
        if next_obs_images.shape[1] == 1:
            next_obs_images = next_obs_images[:, 0]
        else:
            next_obs_images = next_obs_images[:, 0]
    
    # Handle actions - could be (B, T, 1, action_dim), (B, T, action_dim), etc.
    # Target shape: (B, action_dim)
    actions = jnp.asarray(actions)
    
    # Remove singleton dimensions and take first timestep if needed
    # Shape could be: (256, 120, 1, 7) -> want (256, 7)
    while actions.ndim > 2:
        # Find a dimension that's either 1 (squeeze it) or take first element
        if actions.shape[-2] == 1:
            # Squeeze second-to-last dim: (B, ..., 1, action_dim) -> (B, ..., action_dim)
            actions = actions.squeeze(axis=-2)
        elif actions.shape[1] == 1:
            # Squeeze dim 1: (B, 1, ...) -> (B, ...)
            actions = actions[:, 0]
        else:
            # Take first timestep: (B, T, ...) -> (B, ...)
            actions = actions[:, 0]
    
    B = obs_images.shape[0]
    action_dim = actions.shape[-1]
    actions = actions.reshape(B, action_dim)
    
    # Process language
    language = batch["task"]["language_instruction"]
    if isinstance(language, (str, bytes)):
        language_list = [language.decode('utf-8') if isinstance(language, bytes) else language] * B
    elif isinstance(language, (np.ndarray, jnp.ndarray)):
        language_list = [
            lang.decode('utf-8') if isinstance(lang, bytes) else str(lang)
            for lang in language
        ]
    else:
        language_list = [str(l) if not isinstance(l, bytes) else l.decode('utf-8') for l in language]
    
    if len(language_list) < B:
        language_list = (language_list * ((B // len(language_list)) + 1))[:B]
    
    # Get OCTO embeddings for current and next observations
    obs_embeddings = get_octo_embeddings_batch(octo_model, obs_images, language_list)
    next_obs_embeddings = get_octo_embeddings_batch(octo_model, next_obs_images, language_list)
    
    # Process rewards and masks
    rewards = batch.get("reward", jnp.zeros(B))
    masks = batch.get("td_mask", jnp.ones(B, dtype=jnp.bool_))
    mc_returns = batch.get("mc_return", rewards)
    
    def squeeze_to_1d(arr, size):
        arr = jnp.asarray(arr)
        while arr.ndim > 1:
            arr = arr.squeeze(axis=-1) if arr.shape[-1] == 1 else arr.reshape(size)
        return arr.reshape(size)
    
    rewards = squeeze_to_1d(rewards, B)
    masks = squeeze_to_1d(masks, B)
    mc_returns = squeeze_to_1d(mc_returns, B)
    
    return {
        'obs_embeddings': obs_embeddings,           # (B, obs_dim)
        'next_obs_embeddings': next_obs_embeddings, # (B, obs_dim)
        'actions': actions,                          # (B, action_dim)
        'rewards': rewards,                          # (B,)
        'masks': masks,                              # (B,)
        'mc_returns': mc_returns,                    # (B,)
        'language': language_list,
        'obs_images': obs_images,
        'next_obs_images': next_obs_images,
    }


def process_trajectory_for_eval(traj_batch, octo_model):
    """Process full trajectory for evaluation."""
    images = traj_batch["observation"]["image_primary"]
    actions = traj_batch["action"]
    
    images = jnp.asarray(images)
    while images.ndim > 4:
        if images.shape[0] == 1:
            images = images[0]
        elif images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        else:
            images = images[0]
    
    T = images.shape[0]
    
    # Handle actions - could be (T, window, 1, action_dim), etc.
    # Target: (T, action_dim)
    actions = jnp.asarray(actions)
    while actions.ndim > 2:
        if actions.shape[-2] == 1:
            # Squeeze second-to-last: (..., 1, action_dim) -> (..., action_dim)
            actions = actions.squeeze(axis=-2)
        elif actions.shape[0] == 1:
            actions = actions[0]
        elif actions.shape[1] == 1:
            actions = actions[:, 0]
        else:
            # Take first element of extra dimension
            actions = actions[:, 0]
    
    if actions.shape[0] != T:
        actions = actions[:T]
    
    # Language
    language = traj_batch["task"]["language_instruction"]
    if isinstance(language, (str, bytes)):
        lang_str = language.decode('utf-8') if isinstance(language, bytes) else language
    elif isinstance(language, (list, np.ndarray)):
        lang_entry = language[0] if len(language) > 0 else ""
        lang_str = lang_entry.decode('utf-8') if isinstance(lang_entry, bytes) else str(lang_entry)
    else:
        lang_str = str(language)
    
    language_list = [lang_str] * T
    obs_embeddings = get_octo_embeddings_batch(octo_model, images, language_list)
    
    # Rewards and masks
    rewards = jnp.asarray(traj_batch.get("reward", jnp.zeros(T))).reshape(-1)[:T]
    masks = jnp.asarray(traj_batch.get("td_mask", jnp.ones(T))).reshape(-1)[:T]
    mc_returns = jnp.asarray(traj_batch.get("mc_return", rewards)).reshape(-1)[:T]
    
    return {
        'obs_embeddings': obs_embeddings,  # (T, obs_dim)
        'actions': actions,                 # (T, action_dim)
        'rewards': rewards,                 # (T,)
        'masks': masks,                     # (T,)
        'mc_returns': mc_returns,           # (T,)
        'images': images,                   # (T, H, W, C)
        'language': lang_str,
    }


# ============================================================================
# Evaluation Utilities
# ============================================================================

def prepare_trajectory_features(
    ttt_adapter,
    ttt_params,
    processed_traj,
    adapt=True,
    reset_each_step=True,
):
    """Prepare trajectory features for RL agent."""
    obs_embeddings = processed_traj['obs_embeddings']
    actions = processed_traj['actions']
    
    if adapt:
        features, ssl_losses, _ = process_trajectory_with_ttt(
            ttt_adapter,
            ttt_params,
            obs_embeddings,
            actions,
            reset_each_step=reset_each_step,
        )
    else:
        features = process_trajectory_no_adapt(
            ttt_adapter,
            ttt_params,
            obs_embeddings,
        )
        ssl_losses = jnp.zeros(obs_embeddings.shape[0])
    
    return {
        'observations': {'image': features},
        'actions': processed_traj['actions'],
        'rewards': processed_traj['rewards'],
        'masks': processed_traj['masks'],
        'mc_returns': processed_traj['mc_returns'],
        'goals': {},
        'visuals': np.asarray(jax.device_get(processed_traj['images'])),
        'ssl_losses': ssl_losses,
    }


def plot_value_overlay(rl_agent, traj_adapted, traj_base, seed=None):
    """Plot value functions comparing TTT-adapted vs baseline."""
    goals = traj_adapted.get('goals', {})
    
    metrics_adapted = rl_agent.get_eval_values(traj_adapted, seed, goals)
    metrics_base = rl_agent.get_eval_values(traj_base, seed, goals)
    
    visuals = traj_adapted.get('visuals')
    images = np.asarray(visuals)
    if images.ndim == 3:
        images = images[None, ...]
    T = images.shape[0]
    
    # Add SSL loss
    if 'ssl_losses' in traj_adapted:
        metrics_adapted['ssl_loss'] = np.asarray(traj_adapted['ssl_losses'])
    
    metric_keys = list(metrics_adapted.keys())
    num_rows = len(metric_keys) + 1
    fig, axs = plt.subplots(num_rows, 1, figsize=(12, 3 * num_rows))
    canvas = FigureCanvas(fig)
    
    # Row 0: frames
    interval = max(1, T // 8)
    sel_images = images[::interval]
    if sel_images.shape[0] > 0:
        flattened = np.concatenate([np.asarray(f).squeeze() for f in sel_images], axis=1)
        axs[0].imshow(flattened)
        axs[0].set_title('Trajectory frames')
    axs[0].axis('off')
    
    # Plot metrics
    for idx, key in enumerate(metric_keys, start=1):
        series_adapted = np.asarray(metrics_adapted[key])
        series_base = np.asarray(metrics_base.get(key, series_adapted))
        
        plot_len = min(len(series_adapted), len(series_base), T)
        if plot_len == 0:
            continue
        
        steps = np.arange(plot_len)
        
        if key in ('rewards', 'masks', 'ssl_loss'):
            axs[idx].plot(steps, series_adapted[:plot_len], 'b-o', markersize=2, label=key)
        else:
            axs[idx].plot(steps, series_adapted[:plot_len], 'b-o', markersize=2, label='TTT-adapted')
            axs[idx].plot(steps, series_base[:plot_len], 'r--x', markersize=2, label='No-TTT')
            axs[idx].legend(loc='best', fontsize=8)
        
        combined = np.concatenate([series_adapted[:plot_len], series_base[:plot_len]])
        if combined.size > 0:
            ymin, ymax = combined.min(), combined.max()
            margin = max(0.1 * (ymax - ymin), 0.1)
            axs[idx].set_ylim([ymin - margin, ymax + margin])
        
        axs[idx].set_ylabel(key)
        axs[idx].grid(True, alpha=0.3)
    
    axs[-1].set_xlabel('Timestep')
    plt.tight_layout()
    
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return out_image


def _to_jax_array(x):
    if isinstance(x, (str, bytes)):
        return x
    if hasattr(x, 'numpy'):
        arr = x.numpy()
    else:
        arr = x
    if hasattr(arr, 'dtype') and arr.dtype is not None and arr.dtype.kind in ('O', 'U', 'S'):
        return arr
    return jnp.array(arr)


# ============================================================================
# Main
# ============================================================================

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)
    
    tf.config.set_visible_devices([], "GPU")
    
    # WandB
    wandb.init(
        project=FLAGS.project,
        name=FLAGS.name,
        config={**FLAGS.config.to_dict(), **FLAGS.oxedata_config.to_dict()}
    )
    
    save_dir = os.path.join(
        FLAGS.config.save_dir,
        FLAGS.project,
        f"{FLAGS.name}_{wandb.run.id}",
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # ========================================================================
    # Load OCTO
    # ========================================================================
    model_type = f"hf://rail-berkeley/{FLAGS.config.encoder}"
    octo_model = OctoModel.load_pretrained(model_type)
    logging.info(f"Loaded OCTO: {model_type}")
    
    octo_feature_dim = 384 if "small" in model_type else 512
    logging.info(f"OCTO feature dim: {octo_feature_dim}")
    
    # ========================================================================
    # Data Pipeline
    # ========================================================================
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
        del FLAGS.oxedata_config["oxe_kwargs"]
    
    batch_size = FLAGS.config.batch_size
    assert batch_size % num_devices == 0
    
    train_data = make_interleaved_dataset(
        dataset_kwargs_list=FLAGS.oxedata_config["dataset_kwargs_list"],
        sample_weights=FLAGS.oxedata_config["sample_weights"],
        train=True,
        shuffle_buffer_size=FLAGS.oxedata_config["shuffle_buffer_size"],
        traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
        frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
        batch_size=batch_size,
        balance_weights=FLAGS.oxedata_config.get("balance_weights", False),
    )
    
    val_data = create_validation_dataset(
        FLAGS.oxedata_config["dataset_kwargs_list"][0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    # Trajectory data for eval
    viz_kwargs = dict(FLAGS.oxedata_config["traj_transform_kwargs"])
    viz_kwargs.pop("window_size", None)
    viz_kwargs["subsample_length"] = None
    val_traj_data = create_validation_dataset(
        FLAGS.oxedata_config["dataset_kwargs_list"][0],
        viz_kwargs,
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    train_data_iter = train_data.iterator(prefetch=0)
    val_data_iter = val_data.unbatch().shuffle(1000).repeat().batch(batch_size).iterator(prefetch=0)
    val_traj_iter = val_traj_data.shuffle(100).repeat().iterator()
    
    # ========================================================================
    # Example Batch
    # ========================================================================
    example_raw = next(train_data_iter)
    example_raw = jax.tree_map(_to_jax_array, example_raw)
    example_batch = process_batch_with_octo(example_raw, octo_model)
    
    action_dim = example_batch['actions'].shape[-1]
    logging.info(f"obs_dim: {octo_feature_dim}, action_dim: {action_dim}")
    
    # ========================================================================
    # Initialize TTT Adapter
    # ========================================================================
    ttt_config = FLAGS.config.get('ttt', {})
    
    ttt_adapter = TTTRoboticsAdapter(
        obs_dim=octo_feature_dim,
        action_dim=action_dim,
        bottleneck_dim=ttt_config.get('bottleneck_dim', 64),
        output_dim=ttt_config.get('output_dim', 256),
        ssl_mode=ttt_config.get('ssl_mode', 'prediction'),  # "prediction" uses actions!
        eta_base=ttt_config.get('eta_base', 1.0),
        use_input_dependent_lr=ttt_config.get('use_input_dependent_lr', True),
    )
    
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    
    ttt_params = ttt_adapter.init(
        init_rng,
        example_batch['obs_embeddings'],
        example_batch['actions'],
        example_batch['next_obs_embeddings'],
    )['params']
    
    logging.info(f"TTT params: {sum(x.size for x in jax.tree_util.tree_leaves(ttt_params)):,}")
    logging.info(f"SSL mode: {ttt_config.get('ssl_mode', 'prediction')}")
    
    # ========================================================================
    # Initialize RL Agent
    # ========================================================================
    # Get example features
    example_features, _, _, _ = ttt_adapter.apply(
        {'params': ttt_params},
        example_batch['obs_embeddings'],
        example_batch['actions'],
        example_batch['next_obs_embeddings'],
    )
    
    rng, agent_rng = jax.random.split(rng)
    rl_agent = agents[FLAGS.config.agent].create(
        rng=agent_rng,
        observations={'image': example_features},
        goals={},
        actions=example_batch['actions'],
        octo_model=octo_model,
        **FLAGS.config.agent_kwargs,
    )
    logging.info("RL Agent initialized")
    
    # ========================================================================
    # Optimizer
    # ========================================================================
    base_lr = FLAGS.config.get('learning_rate', 1e-4)
    warmup_steps = FLAGS.config.get('warmup_steps', 2000)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=FLAGS.config.num_steps,
        end_value=0.0,
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.config.get('grad_clip', 1.0)),
        optax.adamw(lr_schedule, weight_decay=FLAGS.config.get('weight_decay', 0.01))
    )
    
    opt_state = tx.init(ttt_params)
    
    # ========================================================================
    # Training Step
    # ========================================================================
    lambda_ssl = FLAGS.config.get('lambda_ssl', 1.0)
    ttt_train_step = create_ttt_train_step(ttt_adapter, lambda_ssl=lambda_ssl)
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    logging.info("Starting training...")
    
    ttt_params = jax.device_put(jax.tree_map(jnp.array, ttt_params), sharding.replicate())
    opt_state = jax.device_put(jax.tree_map(jnp.array, opt_state), sharding.replicate())
    rl_agent = jax.device_put(jax.tree_map(jnp.array, rl_agent), sharding.replicate())
    
    prev_traj_language = None
    
    for step in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        rng, step_rng = jax.random.split(rng)
        
        # Get batch
        try:
            batch_raw = next(train_data_iter)
        except StopIteration:
            train_data_iter = train_data.iterator(prefetch=0)
            batch_raw = next(train_data_iter)
        
        batch_raw = jax.tree_map(_to_jax_array, batch_raw)
        batch_processed = process_batch_with_octo(batch_raw, octo_model)
        
        batch = {
            'obs_embeddings': shard_fn(batch_processed['obs_embeddings']),
            'next_obs_embeddings': shard_fn(batch_processed['next_obs_embeddings']),
            'actions': shard_fn(batch_processed['actions']),
            'rewards': shard_fn(batch_processed['rewards']),
            'masks': shard_fn(batch_processed['masks']),
            'mc_returns': shard_fn(batch_processed['mc_returns']),
        }
        
        # TTT update
        ttt_params, opt_state, ttt_metrics, features = ttt_train_step(
            ttt_params, opt_state, batch, tx
        )
        
        # RL update
        # Get next_obs features for TD target
        next_features, _, _, _ = ttt_adapter.apply(
            {'params': ttt_params},
            batch['next_obs_embeddings'],
            batch['actions'],  # dummy, not used for output
            batch['next_obs_embeddings'],  # dummy
            adapt=False,  # Don't adapt for next_obs
        )
        
        rl_batch = {
            'observations': {'image': features},
            'next_observations': {'image': next_features},
            'actions': batch['actions'],
            'rewards': batch['rewards'],
            'masks': batch['masks'],
            'mc_returns': batch['mc_returns'],
            'goals': {},
        }
        
        rl_agent, agent_metrics = rl_agent.update(rl_batch)
        
        # Logging
        if (step + 1) % FLAGS.config.log_interval == 0:
            def flatten(node, prefix=""):
                if isinstance(node, dict):
                    flat = {}
                    for k, v in node.items():
                        flat.update(flatten(v, f"{prefix}/{k}" if prefix else k))
                    return flat
                arr = jnp.asarray(node)
                return {prefix: float(arr) if arr.ndim == 0 else float(jnp.mean(arr))}
            
            metrics = flatten({'ttt': ttt_metrics, 'agent': agent_metrics})
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            
            ssl_before = metrics.get('ttt/ssl_loss_before', 0)
            ssl_after = metrics.get('ttt/ssl_loss_after', 0)
            logging.info(f"Step {step+1}: ssl_before={ssl_before:.4f}, ssl_after={ssl_after:.4f}")
        
        # Debug diagnostics
        if (step + 1) % (FLAGS.config.log_interval * 10) == 0:
            diag = debug_ttt_state(ttt_adapter, jax.device_get(ttt_params), batch, step)
            print_diagnostics(diag, step)
            wandb.log({f"debug/{k}": v for k, v in diag.items() if isinstance(v, (int, float))}, step=step)
        
        # Validation
        if (step + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Validation...")
            
            ttt_params_host = jax.device_get(ttt_params)
            
            val_metrics_list = []
            for _ in range(8):
                try:
                    val_raw = next(val_data_iter)
                except StopIteration:
                    val_data_iter = val_data.unbatch().shuffle(1000).repeat().batch(batch_size).iterator(prefetch=0)
                    val_raw = next(val_data_iter)
                
                val_raw = jax.tree_map(_to_jax_array, val_raw)
                val_processed = process_batch_with_octo(val_raw, octo_model)
                
                _, _, ssl_before, ssl_after = ttt_adapter.apply(
                    {'params': ttt_params_host},
                    val_processed['obs_embeddings'],
                    val_processed['actions'],
                    val_processed['next_obs_embeddings'],
                )
                
                val_metrics_list.append({
                    'ssl_before': float(ssl_before),
                    'ssl_after': float(ssl_after),
                })
            
            val_metrics = jax.tree_map(lambda *xs: np.mean(xs), *val_metrics_list)
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            logging.info(f"Val: {val_metrics}")
            
            # Value plots
            logging.info("Generating value plots...")
            for num in range(3):
                try:
                    traj_raw = next(val_traj_iter)
                except StopIteration:
                    val_traj_iter = val_traj_data.shuffle(100).repeat().iterator()
                    traj_raw = next(val_traj_iter)
                
                traj_raw = jax.tree_map(_to_jax_array, traj_raw)
                processed_traj = process_trajectory_for_eval(traj_raw, octo_model)
                
                T = processed_traj['obs_embeddings'].shape[0]
                logging.info(f"  Traj {num}: T={T}, '{processed_traj['language'][:40]}...'")
                
                traj_adapted = prepare_trajectory_features(
                    ttt_adapter, ttt_params_host, processed_traj,
                    adapt=True, reset_each_step=True
                )
                traj_base = prepare_trajectory_features(
                    ttt_adapter, ttt_params_host, processed_traj,
                    adapt=False
                )
                
                rng, plot_rng = jax.random.split(rng)
                plot_img = plot_value_overlay(rl_agent, traj_adapted, traj_base, seed=plot_rng)
                wandb.log({f"plots/traj_{num}_reset": wandb.Image(plot_img)}, step=step)
                
                # Cumulative adaptation
                traj_cumul = prepare_trajectory_features(
                    ttt_adapter, ttt_params_host, processed_traj,
                    adapt=True, reset_each_step=False
                )
                plot_cumul = plot_value_overlay(rl_agent, traj_cumul, traj_base, seed=plot_rng)
                wandb.log({f"plots/traj_{num}_cumul": wandb.Image(plot_cumul)}, step=step)
                
                prev_traj_language = processed_traj['language']
        
        # Checkpoint
        if (step + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            ckpt = {
                'ttt_params': jax.device_get(ttt_params),
                'opt_state': jax.device_get(opt_state),
                'rl_agent': jax.device_get(rl_agent),
                'step': step + 1,
            }
            checkpoints.save_checkpoint(save_dir, ckpt, step=step+1, keep=10)
    
    logging.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)