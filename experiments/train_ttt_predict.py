"""
Training script for TTT-Predict: Test-Time Training with next-state prediction.

This script combines:
- train.py's simple transition-level data pipeline
- TTT-Predict's next-state prediction objective

Key differences from train_ttt.py:
- No window sampling - uses standard (obs, action, next_obs) transitions
- Self-supervised objective: f_adapt(P_K(obs, action)) → P_V(next_obs)
- Simpler data flow with transition-level batching
"""

import copy
import os
from functools import partial
from pathlib import Path
import atexit

# Suppress TensorFlow warnings
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
from ml_collections import config_flags
import optax
import wandb

tf.get_logger().setLevel('ERROR')
from jax.experimental.compilation_cache import compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.model.octo_model import OctoModel

from jaxrl_m.vision import encoders
from ttt_predict_agent import (
    TTTPredictFeatureExtractor,
    create_ttt_predict_agent,
    create_ttt_predict_update_step,
    get_adapted_features_for_trajectory,
    sequential_test_time_adapt,
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
flags.DEFINE_string("project", "ttt_predict", "WandB project name.")

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


def get_image_embeddings_with_encoder(encoder_def, encoder_params, images):
    """
    Extract image embeddings using a local encoder (jaxrl_m.vision encoder).

    Args:
        encoder_def: encoder module
        encoder_params: encoder parameters
        images: (B, H, W, C) or (T, H, W, C) or (H, W, C)

    Returns:
        embeddings: (B, hidden_dim) JAX array
    """
    imgs = jnp.asarray(images, dtype=jnp.float32)
    # Normalize batch dims: (H,W,C) -> (1,H,W,C), (T,H,W,C) -> (T,H,W,C)
    if imgs.ndim == 3:
        imgs = imgs[None, ...]
    # If input is (B, T, H, W, C) flatten to (B*T, H, W, C)
    if imgs.ndim == 5:
        imgs = imgs.reshape((-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]))

    B = imgs.shape[0]
    height, width = imgs.shape[1:3]
    channels = imgs.shape[-1]

    # Resize to expected shape if necessary
    if (height, width) != (256, 256):
        from jax.image import resize
        imgs = resize(imgs, (B, 256, 256, channels), method='bilinear')

    embeddings = encoder_def.apply({'params': encoder_params}, imgs, train=False)
    return embeddings


def process_batch_with_encoder(batch, encoder_def, encoder_params, text_processor=None):
    """
    Process a batch from OXE dataset and extract OCTO embeddings.
    
    Args:
        batch: Raw batch from dataset
        octo_model: OCTO model for embedding extraction
        text_processor: Optional text processor
        
    Returns:
        processed_batch: Dict with obs_embeddings, next_obs_embeddings, actions, etc.
    """
    # Extract images
    obs_images = batch["observation"]["image_primary"]
    next_obs_images = batch["next_observation"]["image_primary"]
    actions = batch["action"]
    
    # Handle squeeze if needed (remove time dimension if present)
    if obs_images.ndim == 5:  # (B, T, H, W, C)
        obs_images = obs_images[:, 0]
    if next_obs_images.ndim == 5:
        next_obs_images = next_obs_images[:, 0]
    
    # Squeeze actions to (B, action_dim) - may have extra dims like (B, 1, 1, A)
    actions = jnp.asarray(actions)
    while actions.ndim > 2:
        actions = actions.squeeze(axis=1)
    
    B = obs_images.shape[0]
    action_dim = actions.shape[-1]
    actions = actions.reshape(B, action_dim)  # Ensure (B, A)
    
    # Process language
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
    
    # Ensure we have B language strings
    if len(language_list) < B:
        language_list = (language_list * ((B // len(language_list)) + 1))[:B]
    
    # Stack images for efficient OCTO encoding (2B images in one pass)
    stacked_images = jnp.concatenate([obs_images, next_obs_images], axis=0)  # (2B, H, W, C)
    stacked_language = language_list + language_list  # 2B strings
    
    # Encode with OCTO
    all_embeddings = get_image_embeddings_with_encoder(encoder_def, encoder_params, stacked_images)
    
    # Split back
    obs_embeddings = all_embeddings[:B]
    next_obs_embeddings = all_embeddings[B:]
    
    # Process rewards and masks - may have extra dims like (B, 1, 1)
    rewards = batch.get("reward", jnp.zeros(B))
    masks = batch.get("td_mask", jnp.ones(B, dtype=jnp.bool_))
    mc_returns = batch.get("mc_return", rewards)
    
    # Ensure correct shapes (B,) - squeeze any extra dimensions
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
        # Keep raw images for visualization
        'obs_images': obs_images,
        'next_obs_images': next_obs_images,
    }


def process_trajectory_for_eval(traj_batch, octo_model):
    """
    Process a full trajectory for evaluation/visualization.
    
    Args:
        traj_batch: Trajectory batch with (T, ...) shaped arrays
        octo_model: OCTO model
        
    Returns:
        processed: Dict with trajectory data and embeddings
    """
    # Get images - handle various shapes
    images = traj_batch["observation"]["image_primary"]
    actions = traj_batch["action"]
    
    logging.info(f"  process_trajectory_for_eval: raw images shape={images.shape}, actions shape={actions.shape}")
    
    # Squeeze out singleton dimensions to get (T, H, W, C)
    # Possible input shapes: 
    #   (T, H, W, C) - already correct
    #   (B, T, H, W, C) - batched trajectories
    #   (T, 1, H, W, C) - extra camera dimension
    #   (B, T, 1, H, W, C) - batched with camera dim
    images = jnp.asarray(images)
    
    # Remove batch dimension if present (take first)
    while images.ndim > 4:
        if images.shape[0] == 1:
            images = images[0]
        elif images.ndim == 5 and images.shape[1] == 1:
            # (T, 1, H, W, C) -> (T, H, W, C)
            images = images[:, 0]
        else:
            # (B, T, H, W, C) -> (T, H, W, C)
            images = images[0]
    
    T = images.shape[0]
    logging.info(f"  process_trajectory_for_eval: processed T={T}, images shape={images.shape}")
    
    # Handle actions shape similarly
    actions = jnp.asarray(actions)
    # Remove batch/singleton dims to get (T, action_dim)
    while actions.ndim > 2:
        if actions.shape[0] == 1:
            actions = actions[0]
        elif actions.shape[1] == 1:
            actions = actions[:, 0]
        else:
            actions = actions[0]
    # Make sure first dim matches T
    if actions.shape[0] != T:
        actions = actions[:T]
    
    logging.info(f"  process_trajectory_for_eval: actions shape={actions.shape}")
    
    # Process language
    language = traj_batch["task"]["language_instruction"]
    if isinstance(language, (str, bytes)):
        lang_str = language.decode('utf-8') if isinstance(language, bytes) else language
    elif isinstance(language, (list, np.ndarray)):
        lang_entry = language[0] if len(language) > 0 else ""
        lang_str = lang_entry.decode('utf-8') if isinstance(lang_entry, bytes) else str(lang_entry)
    else:
        lang_str = str(language)
    
    language_list = [lang_str] * T
    
    # Encode all frames with OCTO
    obs_embeddings = get_image_embeddings_with_encoder(encoder_def, encoder_params, images)  # (T, D)
    
    # Process rewards and masks
    rewards = traj_batch.get("reward", jnp.zeros(T))
    masks = traj_batch.get("td_mask", jnp.ones(T, dtype=jnp.bool_))
    mc_returns = traj_batch.get("mc_return", rewards)
    
    # Ensure shapes for rewards/masks
    rewards = jnp.asarray(rewards)
    masks = jnp.asarray(masks)
    mc_returns = jnp.asarray(mc_returns)
    
    # Flatten and trim to T
    if rewards.ndim > 1:
        rewards = rewards.reshape(-1)[:T]
    if masks.ndim > 1:
        masks = masks.reshape(-1)[:T]
    if mc_returns.ndim > 1:
        mc_returns = mc_returns.reshape(-1)[:T]
    
    # Ensure length matches T
    if rewards.shape[0] != T:
        rewards = rewards[:T] if rewards.shape[0] > T else jnp.pad(rewards, (0, T - rewards.shape[0]))
    if masks.shape[0] != T:
        masks = masks[:T] if masks.shape[0] > T else jnp.pad(masks, (0, T - masks.shape[0]), constant_values=1)
    if mc_returns.shape[0] != T:
        mc_returns = mc_returns[:T] if mc_returns.shape[0] > T else jnp.pad(mc_returns, (0, T - mc_returns.shape[0]))
    
    return {
        'obs_embeddings': obs_embeddings,      # (T, octo_dim)
        'actions': actions,                     # (T, action_dim)
        'rewards': rewards,                     # (T,)
        'masks': masks,                         # (T,)
        'mc_returns': mc_returns,               # (T,)
        'images': images,                       # (T, H, W, C)
        'language': lang_str,
    }


def prepare_traj_for_rl_agent(
    processed_traj,
    ttt_params,
    ttt_extractor,
    adapt=True,
    reset=True,
    ttt_lr=1e-2,
    ttt_steps=5,
):
    """
    Prepare trajectory features for RL agent evaluation.
    
    Args:
        processed_traj: Dict from process_trajectory_for_eval
        ttt_params: TTT parameters
        ttt_extractor: TTT feature extractor
        adapt: Whether to run TTT adaptation
        reset: Reset f_adapt each step
        ttt_lr: Adaptation learning rate
        ttt_steps: Adaptation steps
        
    Returns:
        agent_traj: Dict formatted for RL agent
    """
    obs_embeddings = processed_traj['obs_embeddings']
    actions = processed_traj['actions']
    
    # Get features (with or without adaptation)
    features, _ = get_adapted_features_for_trajectory(
        ttt_params,
        ttt_extractor,
        obs_embeddings,
        actions,
        adapt=adapt,
        reset=reset,
        ttt_lr=ttt_lr,
        ttt_steps=ttt_steps,
    )
    
    return {
        'observations': {'image': features},
        'actions': processed_traj['actions'],
        'rewards': processed_traj['rewards'],
        'masks': processed_traj['masks'],
        'mc_returns': processed_traj['mc_returns'],
        'goals': {},
        'visuals': np.asarray(jax.device_get(processed_traj['images'])),
    }


def plot_value_overlay(
    rl_agent,
    traj_adapted,
    traj_base,
    seed=None,
):
    """
    Plot value functions comparing TTT-adapted vs non-adapted features.
    
    Args:
        rl_agent: RL agent with get_eval_values method
        traj_adapted: Trajectory with TTT-adapted features
        traj_base: Trajectory without TTT adaptation
        seed: Random seed for agent
        
    Returns:
        plot_image: RGB image array
    """
    goals_adapted = traj_adapted.get('goals', {})
    goals_base = traj_base.get('goals', {})
    
    metrics_adapted = rl_agent.get_eval_values(traj_adapted, seed, goals_adapted)
    metrics_base = rl_agent.get_eval_values(traj_base, seed, goals_base)
    
    # Get images for visualization
    visuals = traj_adapted.get('visuals', traj_adapted['observations']['image'])
    images = np.asarray(visuals)
    if images.ndim == 3:
        images = images[None, ...]
    
    T = images.shape[0]
    
    # Setup figure
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
        
        # Trim to common length
        plot_len = min(len(series_adapted), len(series_base), T)
        if plot_len == 0:
            axs[idx].set_ylabel(key)
            axs[idx].set_title('No data')
            continue
        
        steps = np.arange(plot_len)
        
        if key in ('rewards', 'masks'):
            # Just plot once for non-TTT-dependent quantities
            axs[idx].plot(steps, series_adapted[:plot_len], label=key, linestyle='-', marker='o', markersize=3)
        else:
            axs[idx].plot(steps, series_adapted[:plot_len], label='TTT-adapted', linestyle='-', marker='o', markersize=3)
            axs[idx].plot(steps, series_base[:plot_len], label='No-TTT', linestyle='--', marker='x', markersize=3)
            axs[idx].legend(loc='best', fontsize=8)
        
        # Set y-axis limits
        combined = np.concatenate([series_adapted[:plot_len], series_base[:plot_len]])
        if combined.size > 0:
            ymin, ymax = combined.min(), combined.max()
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
    # Setup data pipeline (transition-level, like train.py)
    # ========================================================================
    logging.info("Setting up transition-level data pipeline...")
    
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.oxedata_config["oxe_kwargs"]
        )
        oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
        del FLAGS.oxedata_config["oxe_kwargs"]
    
    batch_size = FLAGS.config.batch_size
    assert batch_size % num_devices == 0, (
        f"batch_size={batch_size} must be divisible by num_devices={num_devices}"
    )
    
    # Use full dataset list from oxe_kwargs
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
    
    # Validation data - use first dataset
    val_data = create_validation_dataset(
        train_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    # Create separate dataset for trajectory visualization with larger window
    # This gives us full trajectories for value plotting
    viz_traj_transform_kwargs = dict(FLAGS.oxedata_config["traj_transform_kwargs"])
    # Use natural trajectory lengths for visualization (no forced padding/truncation)
    viz_traj_transform_kwargs.pop("window_size", None)
    viz_traj_transform_kwargs["subsample_length"] = None  # Don't subsample
    
    val_traj_data = create_validation_dataset(
        train_datasets_kwargs_list[0],
        viz_traj_transform_kwargs,
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False
    )
    
    # Create iterators
    train_data_iter = train_data.iterator(prefetch=0)
    
    val_data_iter = (
        val_data.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(batch_size)
        .iterator(prefetch=0)
    )
    
    # Trajectory iterator for visualization (full trajectories, batch_size=1)
    val_traj_iter = val_traj_data.shuffle(100).repeat().iterator()
    
    logging.info(f"Data pipeline ready. Batch size: {batch_size}")
    
    # ========================================================================
    # Get example batch and initialize model
    # ========================================================================
    example_raw = next(train_data_iter)
    example_raw = jax.tree_map(_to_jax_array, example_raw)

    # Build local encoder (same pattern as train.py) and init params using an example image batch
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    enc_rng = jax.random.PRNGKey(int(FLAGS.config.seed) + 1)
    # pick representative images for init
    obs_images_ex = example_raw['observation']['image_primary']
    if obs_images_ex.ndim == 5:  # (B, T, H, W, C)
        obs_images_ex = obs_images_ex[:, 0]
    obs_images_ex = jnp.asarray(obs_images_ex, dtype=jnp.float32)
    try:
        encoder_params = encoder_def.init(enc_rng, obs_images_ex)
    except TypeError:
        # fallback if encoder init signature differs (single image)
        encoder_params = encoder_def.init(enc_rng, obs_images_ex[0])

    # Process example batch with local encoder
    example_batch = process_batch_with_encoder(example_raw, encoder_def, encoder_params)
    
    logging.info(f"Example batch shapes:")
    logging.info(f"  obs_embeddings: {example_batch['obs_embeddings'].shape}")
    logging.info(f"  next_obs_embeddings: {example_batch['next_obs_embeddings'].shape}")
    logging.info(f"  actions: {example_batch['actions'].shape}")
    
    action_dim = example_batch['actions'].shape[-1]
    projection_dim = FLAGS.config.agent_kwargs.get('projection_dim', 64)
    projection_hidden_dim = FLAGS.config.get('projection_hidden_dim', projection_dim)
    projection_num_layers = FLAGS.config.get('projection_num_layers', 1)
    
    # ========================================================================
    # Initialize TTT-Predict + RL Agent
    # ========================================================================
    logging.info(f"Initializing TTT-Predict + {FLAGS.config.agent} agent...")
    
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, agent_rng = jax.random.split(rng)

    ttt_extractor, rl_agent, ttt_params = create_ttt_predict_agent(
        rng=agent_rng,
        obs_example=example_batch['obs_embeddings'],
        next_obs_example=example_batch['next_obs_embeddings'],
        actions_example=example_batch['actions'],
        feature_dim=example_batch['obs_embeddings'].shape[-1],
        action_dim=action_dim,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_num_layers=projection_num_layers,
        agent_config=FLAGS.config.to_dict(),
        octo_model=None,
    )
    
    logging.info(f"Model initialized")
    logging.info(f"Total TTT parameters: {sum(x.size for x in jax.tree_util.tree_leaves(ttt_params)):,}")
    
    # ========================================================================
    # Setup optimizer
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
        optax.adamw(learning_rate)
    )
    
    opt_state = tx.init(ttt_params)
    logging.info("Optimizer initialized")
    
    # ========================================================================
    # Create training step
    # ========================================================================
    lambda_self = FLAGS.config.get('lambda_self', 0.5)
    lambda_rl = FLAGS.config.get('lambda_rl', 0.0)
    rl_loss_terms = tuple(FLAGS.config.get('rl_loss_terms', ("critic", "actor", "temperature")))
    ttt_adapt_lr = FLAGS.config.get('ttt_adapt_lr', 1e-2)
    ttt_adapt_steps = FLAGS.config.get('ttt_adapt_steps', 5)
    adapt_during_training=FLAGS.config.get('adapt_during_training', True)
    
    update_step_fn = create_ttt_predict_update_step(
        ttt_extractor,
        lambda_self=lambda_self,
        lambda_rl=lambda_rl,
        rl_loss_terms=rl_loss_terms,
        ttt_adapt_lr=ttt_adapt_lr,
        ttt_adapt_steps=ttt_adapt_steps,
        adapt_during_training=adapt_during_training,
    )
    
    @jax.jit
    def train_step(ttt_params, opt_state, batch, rng, rl_agent_state):
        return update_step_fn(ttt_params, opt_state, batch, rng, tx, rl_agent_state)
    
    logging.info(f"Training step created (lambda_self={lambda_self})")
    
    # ========================================================================
    # Training loop
    # ========================================================================
    logging.info("Starting training...")
    
    # Replicate params across devices
    ttt_params = jax.device_put(jax.tree_map(jnp.array, ttt_params), sharding.replicate())
    opt_state = jax.device_put(jax.tree_map(jnp.array, opt_state), sharding.replicate())
    rl_agent = jax.device_put(jax.tree_map(jnp.array, rl_agent), sharding.replicate())
    
    # For random language plot comparison
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

        # Process with local encoder
        batch_processed = process_batch_with_encoder(batch_raw, encoder_def, encoder_params)
        
        # Shard the batch
        batch = {
            'obs_embeddings': shard_fn(batch_processed['obs_embeddings']),
            'next_obs_embeddings': shard_fn(batch_processed['next_obs_embeddings']),
            'actions': shard_fn(batch_processed['actions']),
            'rewards': shard_fn(batch_processed['rewards']),
            'masks': shard_fn(batch_processed['masks']),
            'mc_returns': shard_fn(batch_processed['mc_returns']),
        }
        
        # TTT update
        ttt_params, opt_state, ttt_metrics, agent_batch = train_step(
            ttt_params,
            opt_state,
            batch,
            step_rng,
            rl_agent,
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
            
            loss_val = metrics_cpu.get('ttt/loss_total')
            ttt_loss = metrics_cpu.get('ttt/loss_ttt')
            if loss_val is not None:
                logging.info(f"Step {step + 1}: loss={loss_val:.4f}, ttt={ttt_loss:.4f}")
        
        # Validation
        if (step + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Running validation...")
            
            # Get TTT params on host
            def _to_host(x):
                return jax.device_get(x) if isinstance(x, jax.Array) else x
            ttt_params_host = jax.tree_map(_to_host, ttt_params)
            
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
                val_processed = process_batch_with_encoder(val_raw, encoder_def, encoder_params)
                
                # Compute TTT loss
                val_loss = ttt_extractor.apply(
                    {'params': ttt_params_host},
                    val_processed['obs_embeddings'],
                    val_processed['actions'],
                    val_processed['next_obs_embeddings'],
                    method=ttt_extractor.compute_self_supervised_loss,
                    train=False
                )
                
                val_metrics_list.append({'val_loss_ttt': float(val_loss)})
            
            val_metrics = jax.tree_map(
                lambda *xs: np.mean(xs),
                *val_metrics_list
            )
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            logging.info(f"Validation: {val_metrics}")
            
            # Value function plots
            logging.info("Generating value plots...")
            for num in range(3):
                # Get trajectory
                try:
                    traj_raw = next(val_traj_iter)
                except StopIteration:
                    val_traj_iter = val_traj_data.shuffle(100).repeat().iterator()
                    traj_raw = next(val_traj_iter)
                
                traj_raw = jax.tree_map(_to_jax_array, traj_raw)
                processed_traj = process_trajectory_for_eval(traj_raw, encoder_def, encoder_params)
                
                T = processed_traj['obs_embeddings'].shape[0]
                logging.info(f"  Trajectory {num}: T={T} frames, language='{processed_traj['language'][:50]}...'")
                
                # Prepare with TTT adaptation (reset=True)
                traj_adapted = prepare_traj_for_rl_agent(
                    processed_traj,
                    ttt_params_host,
                    ttt_extractor,
                    adapt=True,
                    reset=True,
                    ttt_lr=ttt_adapt_lr,
                    ttt_steps=ttt_adapt_steps,
                )
                
                # Prepare without TTT adaptation
                traj_base = prepare_traj_for_rl_agent(
                    processed_traj,
                    ttt_params_host,
                    ttt_extractor,
                    adapt=False,
                )
                
                # Plot
                rng, plot_rng = jax.random.split(rng)
                plot_img = plot_value_overlay(rl_agent, traj_adapted, traj_base, seed=plot_rng)
                wandb.log({f"value_plots/traj_{num}_reset_true": wandb.Image(plot_img)}, step=step)
                
                # Also plot with reset=False (cumulative adaptation)
                traj_adapted_cumul = prepare_traj_for_rl_agent(
                    processed_traj,
                    ttt_params_host,
                    ttt_extractor,
                    adapt=True,
                    reset=False,  # Cumulative
                    ttt_lr=ttt_adapt_lr,
                    ttt_steps=ttt_adapt_steps,
                )
                plot_img_cumul = plot_value_overlay(rl_agent, traj_adapted_cumul, traj_base, seed=plot_rng)
                wandb.log({f"value_plots/traj_{num}_reset_false": wandb.Image(plot_img_cumul)}, step=step)
                
                # Random language comparison
                if prev_traj_language is not None:
                    # Re-encode with different language
                    processed_traj_rand = copy.deepcopy(processed_traj)
                    # Re-compute embeddings with previous trajectory's language
                    # Note: local encoder currently does not accept language conditioning.
                    rand_embeddings = get_image_embeddings_with_encoder(
                        encoder_def,
                        encoder_params,
                        processed_traj['images']
                    )
                    processed_traj_rand['obs_embeddings'] = rand_embeddings
                    
                    traj_rand_adapted = prepare_traj_for_rl_agent(
                        processed_traj_rand,
                        ttt_params_host,
                        ttt_extractor,
                        adapt=True,
                        reset=True,
                    )
                    traj_rand_base = prepare_traj_for_rl_agent(
                        processed_traj_rand,
                        ttt_params_host,
                        ttt_extractor,
                        adapt=False,
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