"""
Training script for TTT (2024 Paper Style): Test-Time Training with Multi-View Reconstruction.

Based on "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (Sun et al., 2024)

Key differences from train_ttt_predict.py:
- Multi-view reconstruction SSL: f(θ_K·x; W) → θ_V·x (instead of next-state prediction)
- Learned projections θ_K, θ_V, θ_Q
- Input-dependent learning rate
- Inner model f(x; W) = x + LayerNorm(W @ x)
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
import flax.linen as nn

tf.get_logger().setLevel('ERROR')
from jax.experimental.compilation_cache import compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.model.octo_model import OctoModel

from ttt_paper_layer import (
    TTTLinear,
    TTTMLP,
    TTTFeatureAdapter,
    process_sequence_with_ttt,
    process_sequence_no_adapt,
    compute_ttt_ssl_loss,
    compute_ttt_ssl_loss_with_adapt,
)

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.data.text_processing import text_processors

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

try:
    from jax_smi import initialise_tracking
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "ttt_paper", "WandB project name.")

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
    """
    Extract fused vision⊗language embeddings from OCTO model.
    """
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
    """Process a batch from OXE dataset and extract OCTO embeddings."""
    obs_images = batch["observation"]["image_primary"]
    next_obs_images = batch["next_observation"]["image_primary"]
    actions = batch["action"]
    
    # Handle various image shapes
    # Could be (B, H, W, C), (B, T, H, W, C), (B, T, 1, H, W, C), etc.
    obs_images = jnp.asarray(obs_images)
    next_obs_images = jnp.asarray(next_obs_images)
    
    # Squeeze down to (B, H, W, C) - take first timestep if needed
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
    
    # Remove extra dimensions until we get (B, action_dim)
    while actions.ndim > 2:
        # Find a dimension to reduce
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
    
    language = batch["task"]["language_instruction"]
    if isinstance(language, (str, bytes)):
        language_list = [language.decode('utf-8') if isinstance(language, bytes) else language] * B
    elif isinstance(language, (np.ndarray, jnp.ndarray)):
        language_list = []
        for lang in language:
            if isinstance(lang, bytes):
                language_list.append(lang.decode('utf-8'))
            else:
                language_list.append(str(lang))
    else:
        language_list = [str(l) if not isinstance(l, bytes) else l.decode('utf-8') for l in language]
    
    if len(language_list) < B:
        language_list = (language_list * ((B // len(language_list)) + 1))[:B]
    
    # Encode observations with OCTO
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
        'obs_embeddings': obs_embeddings,
        'next_obs_embeddings': next_obs_embeddings,
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'mc_returns': mc_returns,
        'language': language_list,
        'obs_images': obs_images,
        'next_obs_images': next_obs_images,
    }


def process_trajectory_for_eval(traj_batch, octo_model):
    """Process a full trajectory for evaluation."""
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
    
    rewards = traj_batch.get("reward", jnp.zeros(T))
    masks = traj_batch.get("td_mask", jnp.ones(T, dtype=jnp.bool_))
    mc_returns = traj_batch.get("mc_return", rewards)
    
    rewards = jnp.asarray(rewards).reshape(-1)[:T]
    masks = jnp.asarray(masks).reshape(-1)[:T]
    mc_returns = jnp.asarray(mc_returns).reshape(-1)[:T]
    
    return {
        'obs_embeddings': obs_embeddings,
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'mc_returns': mc_returns,
        'images': images,
        'language': lang_str,
    }


# ============================================================================
# Training Step
# ============================================================================

def create_train_step(
    ttt_adapter: TTTFeatureAdapter,
    lambda_ssl: float = 1.0,
    lambda_rl: float = 1.0,
    use_adaptation_in_training: bool = True,
):
    """
    Create the joint training step for TTT + RL.
    
    Args:
        ttt_adapter: TTT feature adapter module
        lambda_ssl: weight for SSL loss
        lambda_rl: weight for RL loss
        use_adaptation_in_training: whether to do one adaptation step during training
    """
    
    def loss_fn(ttt_params, batch, rng, rl_agent):
        """
        Compute joint loss: λ_ssl * L_ssl + λ_rl * L_rl
        """
        obs_embeddings = batch['obs_embeddings']
        
        # Compute TTT SSL loss and get features
        if use_adaptation_in_training:
            ssl_loss, ttt_info = compute_ttt_ssl_loss_with_adapt(
                ttt_params, ttt_adapter, obs_embeddings
            )
        else:
            ssl_loss, ttt_info = compute_ttt_ssl_loss(
                ttt_params, ttt_adapter, obs_embeddings
            )
        
        features = ttt_info['features']
        
        # Prepare batch for RL agent
        rl_batch = {
            'observations': {'image': features},
            'actions': batch['actions'],
            'rewards': batch['rewards'],
            'masks': batch['masks'],
            'mc_returns': batch['mc_returns'],
            'goals': {},
        }
        
        # Get next observation features (for TD learning)
        next_obs_embeddings = batch['next_obs_embeddings']
        if use_adaptation_in_training:
            next_features, _, _ = ttt_adapter.apply(
                {'params': ttt_params},
                next_obs_embeddings,
                method=ttt_adapter.forward_batch_with_single_step_adapt
            )
        else:
            next_features, _ = ttt_adapter.apply(
                {'params': ttt_params},
                next_obs_embeddings,
                train=True,
                method=ttt_adapter.forward_batch
            )
        rl_batch['next_observations'] = {'image': next_features}
        
        # Total loss
        total_loss = lambda_ssl * ssl_loss
        
        metrics = {
            'loss_total': total_loss,
            'loss_ssl': ssl_loss,
            **{f'ttt_{k}': v for k, v in ttt_info.items() if k != 'features'},
        }
        
        return total_loss, (metrics, rl_batch)
    
    @jax.jit
    def train_step(ttt_params, opt_state, batch, rng, tx, rl_agent):
        """Single training step."""
        (loss, (metrics, rl_batch)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(ttt_params, batch, rng, rl_agent)
        
        updates, opt_state = tx.update(grads, opt_state, ttt_params)
        ttt_params = optax.apply_updates(ttt_params, updates)
        
        # Add gradient norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = grad_norm
        
        return ttt_params, opt_state, metrics, rl_batch
    
    return train_step


# ============================================================================
# Evaluation Utilities
# ============================================================================

def prepare_trajectory_features(
    ttt_adapter: TTTFeatureAdapter,
    ttt_params: dict,
    processed_traj: dict,
    adapt: bool = True,
    reset_each_step: bool = True,
):
    """
    Prepare trajectory features for RL agent evaluation.
    
    Args:
        ttt_adapter: TTT feature adapter
        ttt_params: TTT parameters
        processed_traj: Dict from process_trajectory_for_eval
        adapt: Whether to run TTT adaptation
        reset_each_step: Reset W at each step (True) or cumulative (False)
    """
    obs_embeddings = processed_traj['obs_embeddings']
    
    if adapt:
        features, ssl_losses, _ = process_sequence_with_ttt(
            ttt_adapter,
            ttt_params,
            obs_embeddings,
            reset_each_step=reset_each_step,
        )
    else:
        features = process_sequence_no_adapt(
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
    """Plot value functions comparing TTT-adapted vs non-adapted features."""
    goals_adapted = traj_adapted.get('goals', {})
    goals_base = traj_base.get('goals', {})
    
    metrics_adapted = rl_agent.get_eval_values(traj_adapted, seed, goals_adapted)
    metrics_base = rl_agent.get_eval_values(traj_base, seed, goals_base)
    
    visuals = traj_adapted.get('visuals', traj_adapted['observations']['image'])
    images = np.asarray(visuals)
    if images.ndim == 3:
        images = images[None, ...]
    
    T = images.shape[0]
    
    # Add SSL loss to metrics if available
    if 'ssl_losses' in traj_adapted:
        metrics_adapted['ssl_loss'] = traj_adapted['ssl_losses']
    
    metric_keys = list(metrics_adapted.keys())
    num_rows = len(metric_keys) + 1
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows))
    canvas = FigureCanvas(fig)
    
    # Row 0: Sample frames
    interval = max(1, T // 8)
    sel_images = images[::interval]
    if sel_images.shape[0] > 0:
        flattened = np.concatenate([np.asarray(frame).squeeze() for frame in sel_images], axis=1)
        axs[0].imshow(flattened)
        axs[0].set_title('Trajectory frames')
    axs[0].axis('off')
    
    # Remaining rows: Value metrics
    for idx, key in enumerate(metric_keys, start=1):
        series_adapted = np.asarray(metrics_adapted[key])
        series_base = np.asarray(metrics_base.get(key, series_adapted))
        
        plot_len = min(len(series_adapted), len(series_base), T)
        if plot_len == 0:
            axs[idx].set_ylabel(key)
            axs[idx].set_title('No data')
            continue
        
        steps = np.arange(plot_len)
        
        if key in ('rewards', 'masks', 'ssl_loss'):
            axs[idx].plot(steps, series_adapted[:plot_len], label=key, linestyle='-', marker='o', markersize=3)
        else:
            axs[idx].plot(steps, series_adapted[:plot_len], label='TTT-adapted', linestyle='-', marker='o', markersize=3)
            axs[idx].plot(steps, series_base[:plot_len], label='No-TTT', linestyle='--', marker='x', markersize=3)
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
    """Convert various array types to JAX arrays."""
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
# Main Training Loop
# ============================================================================

def main(_):
    def _maybe_start_profiler():
        profile_dir = os.environ.get("JAX_PROFILE_DIR")
        if not profile_dir:
            return False
        profile_path = Path(profile_dir).expanduser()
        profile_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting JAX profiler trace at {profile_path}")
        jax.profiler.start_trace(str(profile_path))
        atexit.register(jax.profiler.stop_trace)
        return True

    _maybe_start_profiler()
    
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)
    
    tf.config.set_visible_devices([], "GPU")
    
    # Initialize WandB
    wandb.init(
        project=FLAGS.project,
        name=FLAGS.name,
        config={
            **FLAGS.config.to_dict(),
            **FLAGS.oxedata_config.to_dict()
        }
    )
    
    save_dir = os.path.join(
        FLAGS.config.save_dir,
        FLAGS.project,
        f"{FLAGS.name}_{wandb.run.id}",
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # ========================================================================
    # Load OCTO model
    # ========================================================================
    model_type = f"hf://rail-berkeley/{FLAGS.config.encoder}"
    octo_model = OctoModel.load_pretrained(model_type)
    logging.info(f"Loaded OCTO model: {model_type}")
    
    octo_feature_dim = 384 if "small" in model_type else 512
    logging.info(f"OCTO feature dimension: {octo_feature_dim}")
    
    # ========================================================================
    # Setup data pipeline
    # ========================================================================
    logging.info("Setting up data pipeline...")
    
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.oxedata_config["oxe_kwargs"]
        )
        del FLAGS.oxedata_config["oxe_kwargs"]
    
    batch_size = FLAGS.config.batch_size
    assert batch_size % num_devices == 0
    
    train_datasets_kwargs_list = FLAGS.oxedata_config["dataset_kwargs_list"]
    train_sample_weights = FLAGS.oxedata_config["sample_weights"]
    
    logging.info(f"Training on {len(train_datasets_kwargs_list)} dataset(s)")
    
    train_data = make_interleaved_dataset(
        dataset_kwargs_list=train_datasets_kwargs_list,
        sample_weights=train_sample_weights,
        train=True,
        shuffle_buffer_size=FLAGS.oxedata_config["shuffle_buffer_size"],
        traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
        frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
        batch_size=batch_size,
        balance_weights=FLAGS.oxedata_config.get("balance_weights", False),
        traj_transform_threads=FLAGS.oxedata_config.get("traj_transform_threads", None),
        traj_read_threads=FLAGS.oxedata_config.get("traj_read_threads", None),
    )
    
    val_data = create_validation_dataset(
        train_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    # Trajectory data for evaluation
    viz_traj_transform_kwargs = dict(FLAGS.oxedata_config["traj_transform_kwargs"])
    viz_traj_transform_kwargs.pop("window_size", None)
    viz_traj_transform_kwargs["subsample_length"] = None
    
    val_traj_data = create_validation_dataset(
        train_datasets_kwargs_list[0],
        viz_traj_transform_kwargs,
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    train_data_iter = train_data.iterator(prefetch=0)
    val_data_iter = (
        val_data.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(batch_size)
        .iterator(prefetch=0)
    )
    val_traj_iter = val_traj_data.shuffle(100).repeat().iterator()
    
    logging.info(f"Data pipeline ready. Batch size: {batch_size}")
    
    # ========================================================================
    # Get example batch
    # ========================================================================
    example_raw = next(train_data_iter)
    example_raw = jax.tree_map(_to_jax_array, example_raw)
    example_batch = process_batch_with_octo(example_raw, octo_model)
    
    logging.info(f"Example batch shapes:")
    logging.info(f"  obs_embeddings: {example_batch['obs_embeddings'].shape}")
    logging.info(f"  actions: {example_batch['actions'].shape}")
    
    action_dim = example_batch['actions'].shape[-1]
    
    # ========================================================================
    # Initialize TTT Adapter
    # ========================================================================
    ttt_config = FLAGS.config.get('ttt', {})
    bottleneck_dim = ttt_config.get('bottleneck_dim', 64)
    output_dim = ttt_config.get('output_dim', 256)
    ttt_type = ttt_config.get('type', 'linear')
    eta_base = ttt_config.get('eta_base', 1.0 if ttt_type == 'linear' else 0.1)
    use_input_dependent_lr = ttt_config.get('use_input_dependent_lr', True)
    
    logging.info(f"Initializing TTT Adapter:")
    logging.info(f"  type: {ttt_type}")
    logging.info(f"  input_dim: {octo_feature_dim}")
    logging.info(f"  bottleneck_dim: {bottleneck_dim}")
    logging.info(f"  output_dim: {output_dim}")
    logging.info(f"  eta_base: {eta_base}")
    
    ttt_adapter = TTTFeatureAdapter(
        input_dim=octo_feature_dim,
        bottleneck_dim=bottleneck_dim,
        output_dim=output_dim,
        ttt_type=ttt_type,
        eta_base=eta_base,
        use_input_dependent_lr=use_input_dependent_lr,
    )
    
    # Initialize TTT params
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    
    ttt_params = ttt_adapter.init(
        init_rng,
        example_batch['obs_embeddings'][0],  # Single sample for init
        method=ttt_adapter.__call__,
    )['params']
    
    ttt_param_count = sum(x.size for x in jax.tree_util.tree_leaves(ttt_params))
    logging.info(f"TTT parameters: {ttt_param_count:,}")
    
    # ========================================================================
    # Initialize RL Agent
    # ========================================================================
    logging.info(f"Initializing {FLAGS.config.agent} agent...")
    
    # Get example features for agent initialization
    example_features, _ = ttt_adapter.apply(
        {'params': ttt_params},
        example_batch['obs_embeddings'],
        train=False,
        method=ttt_adapter.forward_batch
    )
    
    agent_obs = {'image': example_features}
    agent_goals = {}
    
    rng, agent_rng = jax.random.split(rng)
    rl_agent = agents[FLAGS.config.agent].create(
        rng=agent_rng,
        observations=agent_obs,
        goals=agent_goals,
        actions=example_batch['actions'],
        octo_model=octo_model,
        **FLAGS.config.agent_kwargs,
    )
    
    logging.info("RL Agent initialized")
    
    # ========================================================================
    # Setup Optimizer
    # ========================================================================
    base_lr = FLAGS.config.get('learning_rate', 1e-4)
    warmup_steps = FLAGS.config.get('warmup_steps', 2000)
    
    learning_rate = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=FLAGS.config.num_steps - warmup_steps,
        alpha=0.0
    )
    
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0.0, base_lr, warmup_steps), learning_rate],
        [warmup_steps]
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.config.get('grad_clip', 1.0)),
        optax.adamw(learning_rate, weight_decay=FLAGS.config.get('weight_decay', 0.01))
    )
    
    opt_state = tx.init(ttt_params)
    logging.info("Optimizer initialized")
    
    # ========================================================================
    # Create Training Step
    # ========================================================================
    lambda_ssl = FLAGS.config.get('lambda_ssl', 1.0)
    lambda_rl = FLAGS.config.get('lambda_rl', 1.0)
    use_adaptation_in_training = FLAGS.config.get('use_adaptation_in_training', True)
    
    train_step_fn = create_train_step(
        ttt_adapter,
        lambda_ssl=lambda_ssl,
        lambda_rl=lambda_rl,
        use_adaptation_in_training=use_adaptation_in_training,
    )
    
    logging.info(f"Training step created (lambda_ssl={lambda_ssl}, lambda_rl={lambda_rl})")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    logging.info("Starting training...")
    
    # Replicate across devices
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
        
        # Shard batch
        batch = {
            'obs_embeddings': shard_fn(batch_processed['obs_embeddings']),
            'next_obs_embeddings': shard_fn(batch_processed['next_obs_embeddings']),
            'actions': shard_fn(batch_processed['actions']),
            'rewards': shard_fn(batch_processed['rewards']),
            'masks': shard_fn(batch_processed['masks']),
            'mc_returns': shard_fn(batch_processed['mc_returns']),
        }
        
        # TTT update
        ttt_params, opt_state, ttt_metrics, agent_batch = train_step_fn(
            ttt_params, opt_state, batch, step_rng, tx, rl_agent
        )
        
        # RL agent update
        rl_agent, agent_metrics = rl_agent.update(agent_batch)
        
        # Logging
        if (step + 1) % FLAGS.config.log_interval == 0:
            def _flatten_metrics(node, prefix=""):
                if isinstance(node, dict):
                    flat = {}
                    for key, val in node.items():
                        new_prefix = f"{prefix}/{key}" if prefix else key
                        flat.update(_flatten_metrics(val, new_prefix))
                    return flat
                arr = jnp.asarray(node)
                scalar = float(arr) if arr.ndim == 0 else float(jnp.mean(arr))
                return {prefix: scalar}
            
            combined = {'ttt': ttt_metrics, 'agent': agent_metrics}
            metrics_cpu = _flatten_metrics(combined)
            wandb.log({f"train/{k}": v for k, v in metrics_cpu.items()}, step=step)
            
            ssl_loss = metrics_cpu.get('ttt/loss_ssl', 0)
            total_loss = metrics_cpu.get('ttt/loss_total', 0)
            logging.info(f"Step {step + 1}: ssl_loss={ssl_loss:.4f}, total_loss={total_loss:.4f}")
        
        # Validation
        if (step + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Running validation...")
            
            def _to_host(x):
                return jax.device_get(x) if isinstance(x, jax.Array) else x
            ttt_params_host = jax.tree_map(_to_host, ttt_params)
            
            # Validation SSL loss
            val_metrics_list = []
            for _ in range(8):
                try:
                    val_raw = next(val_data_iter)
                except StopIteration:
                    val_data_iter = (
                        val_data.unbatch().shuffle(1000).repeat().batch(batch_size).iterator(prefetch=0)
                    )
                    val_raw = next(val_data_iter)
                
                val_raw = jax.tree_map(_to_jax_array, val_raw)
                val_processed = process_batch_with_octo(val_raw, octo_model)
                
                val_ssl_loss, val_info = compute_ttt_ssl_loss(
                    ttt_params_host, ttt_adapter, val_processed['obs_embeddings']
                )
                
                val_metrics_list.append({'val_ssl_loss': float(val_ssl_loss)})
            
            val_metrics = jax.tree_map(lambda *xs: np.mean(xs), *val_metrics_list)
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            logging.info(f"Validation: {val_metrics}")
            
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
                logging.info(f"  Trajectory {num}: T={T}, lang='{processed_traj['language'][:50]}...'")
                
                # With TTT adaptation (reset each step)
                traj_adapted = prepare_trajectory_features(
                    ttt_adapter, ttt_params_host, processed_traj,
                    adapt=True, reset_each_step=True
                )
                
                # Without TTT adaptation
                traj_base = prepare_trajectory_features(
                    ttt_adapter, ttt_params_host, processed_traj,
                    adapt=False
                )
                
                rng, plot_rng = jax.random.split(rng)
                plot_img = plot_value_overlay(rl_agent, traj_adapted, traj_base, seed=plot_rng)
                wandb.log({f"value_plots/traj_{num}_reset_true": wandb.Image(plot_img)}, step=step)
                
                # With cumulative adaptation
                traj_adapted_cumul = prepare_trajectory_features(
                    ttt_adapter, ttt_params_host, processed_traj,
                    adapt=True, reset_each_step=False
                )
                plot_img_cumul = plot_value_overlay(rl_agent, traj_adapted_cumul, traj_base, seed=plot_rng)
                wandb.log({f"value_plots/traj_{num}_reset_false": wandb.Image(plot_img_cumul)}, step=step)
                
                # Random language comparison
                if prev_traj_language is not None:
                    processed_traj_rand = copy.deepcopy(processed_traj)
                    rand_embeddings = get_octo_embeddings_batch(
                        octo_model,
                        processed_traj['images'],
                        [prev_traj_language] * processed_traj['images'].shape[0]
                    )
                    processed_traj_rand['obs_embeddings'] = rand_embeddings
                    
                    traj_rand_adapted = prepare_trajectory_features(
                        ttt_adapter, ttt_params_host, processed_traj_rand,
                        adapt=True, reset_each_step=True
                    )
                    traj_rand_base = prepare_trajectory_features(
                        ttt_adapter, ttt_params_host, processed_traj_rand,
                        adapt=False
                    )
                    
                    plot_img_rand = plot_value_overlay(rl_agent, traj_rand_adapted, traj_rand_base, seed=plot_rng)
                    wandb.log({f"value_plots/traj_{num}_random_lang": wandb.Image(plot_img_rand)}, step=step)
                
                prev_traj_language = processed_traj['language']
        
        # Save checkpoint
        if (step + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_data = {
                'ttt_params': jax.device_get(ttt_params),
                'opt_state': jax.device_get(opt_state),
                'rl_agent': jax.device_get(rl_agent),
                'step': step + 1,
            }
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, checkpoint_data, step=step + 1, keep=10
            )
            logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    logging.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)