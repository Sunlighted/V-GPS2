"""
Training script for VLM with Test-Time Training on trajectory data.
Based on train_embedding.py but modified for trajectory-level training.
"""

import copy
import itertools
import os
import atexit
from functools import partial
from pathlib import Path

# Suppress TensorFlow warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import jax
import jax.numpy as jnp
import flax.linen as nn
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
import dlimp as dl

# Further suppress TF logging
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)
from jax.experimental.compilation_cache import compilation_cache

from octo.data.dataset import make_interleaved_dataset
from octo.data.sim_dataset import make_simulated_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets
from octo.model.octo_model import OctoModel

from ttt_agent import (
    TTTFeatureExtractor,
    create_ttt_agent,
    create_ttt_agent_update_step,
    test_time_adapt_ttt,
    windowed_test_time_adapt_ttt,
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
flags.DEFINE_string("project", "vlm_ttt", "WandB project name.")

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


def get_octo_fused_embeddings(octo_model, images, language_instructions, max_frames_per_batch=None):
    """
    Extract fused vision⊗language embeddings from OCTO model.
    Batches all frames for efficient processing (B*T batch, T=1 per item).
    
    Args:
        octo_model: Loaded OctoModel
        images: Images in JAX format (B, T, H, W, C)
        language_instructions: List of strings, length B
        
    Returns:
        fused_embeddings: (B, T, hidden_dim) JAX array
    """
    import jax.numpy as jnp
    
    B, T = images.shape[:2]

    images = jnp.asarray(images, dtype=jnp.float32)
    height, width = images.shape[2:4]
    channels = images.shape[-1]

    if (height, width) != (256, 256):
        from jax.image import resize
        flat = images.reshape(B * T, height, width, channels)
        flat = resize(flat, (flat.shape[0], 256, 256, channels), method='bilinear')
        images = flat.reshape(B, T, 256, 256, channels)
        height, width = 256, 256

    images_flat = images.reshape(B * T, 1, height, width, channels)
    language_repeated = [lang for lang in language_instructions for _ in range(T)]
    
    # Create tasks (B*T tasks, one per frame)
    tasks = octo_model.create_tasks(texts=language_repeated)
    
    # Create mask (B*T, 1) - each item is a single frame
    timestep_pad_mask = jnp.ones((B * T, 1), dtype=jnp.bool_)
    
    # Process all frames in one batched call
    def slice_tree(tree, start, end):
        def maybe_slice(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)) and x.ndim >= 1:
                # Only slice along batch dimension if compatible
                if x.shape[0] >= end:
                    return x[start:end]
            return x
        return jax.tree_map(maybe_slice, tree)

    total_frames = images_flat.shape[0]
    chunk_size = max_frames_per_batch or total_frames
    if chunk_size >= total_frames:
        obs = {'image_primary': images_flat}
        output = octo_model.run_transformer(obs, tasks, timestep_pad_mask, train=False)
        readout_tokens = output['readout_action'].tokens
    else:
        readout_chunks = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            obs_chunk = {'image_primary': images_flat[start:end]}
            tasks_chunk = slice_tree(tasks, start, end)
            mask_chunk = timestep_pad_mask[start:end]
            output_chunk = octo_model.run_transformer(obs_chunk, tasks_chunk, mask_chunk, train=False)
            readout_chunks.append(output_chunk['readout_action'].tokens)
        readout_tokens = jnp.concatenate(readout_chunks, axis=0)
    embeddings_flat = readout_tokens[:, 0].mean(axis=1)  # (B*T, num_tokens, D) → (B*T, D)
    
    # Reshape back to trajectory format: (B*T, D) → (B, T, D)
    fused_embeddings = embeddings_flat.reshape(B, T, -1)
    
    return fused_embeddings


def process_trajectory_batch(batch, octo_model, text_processor=None, max_frames_per_octo_batch=None):
    # Extract data
    images = batch["observation"]["image_primary"]  # (B, T, H, W, C)
    actions = batch["action"]  # (B, T, action_dim)

    # Handle unbatched data (add batch dimension if needed)
    if images.ndim == 4:  # (T, H, W, C)
        images = images[None, ...]
    if actions.ndim == 2:  # (T, D)
        actions = actions[None, ...]

    B, T = images.shape[:2]
    
    # Language processing
    language = batch["task"]["language_instruction"]
    if isinstance(language, (str, bytes)):
        language = [language]
    
    # FIX: Handle rewards/masks that might be (B,) or (B, T)
    mc_returns = batch.get("mc_return", batch.get("reward"))
    masks = batch.get("td_mask", None)
    reward_data = batch.get("reward", mc_returns)
    
    # Helper to ensure (B, T) shape
    def ensure_bt_shape(arr, name):
        if arr is None:
            return None
        
        arr = jnp.asarray(arr)
        
        if arr.ndim == 0:  # Scalar
            return jnp.full((B, T), arr)
        elif arr.ndim == 1:
            if arr.shape[0] == B:
                # (B,) → broadcast to (B, T)
                return jnp.broadcast_to(arr[:, None], (B, T))
            elif arr.shape[0] == T:
                # (T,) → broadcast to (B, T)
                return jnp.broadcast_to(arr[None, :], (B, T))
            else:
                logging.warning(f"{name} has unexpected 1D shape {arr.shape}, broadcasting to ({B}, {T})")
                return jnp.broadcast_to(arr[None, :], (B, T))
        elif arr.ndim == 2:
            if arr.shape == (B, T):
                return arr
            else:
                logging.warning(f"{name} has unexpected 2D shape {arr.shape}, reshaping to ({B}, {T})")
                return arr.reshape(B, T)
        else:
            # Higher dims - squeeze extras
            while arr.ndim > 2:
                arr = arr.squeeze()
            return ensure_bt_shape(arr, name)
    
    mc_returns = ensure_bt_shape(mc_returns, 'mc_returns')
    if mc_returns is None:
        mc_returns = jnp.zeros((B, T))
    
    masks = ensure_bt_shape(masks, 'masks')
    if masks is None:
        masks = jnp.ones((B, T), dtype=jnp.bool_)
    
    reward_data = ensure_bt_shape(reward_data, 'rewards')
    if reward_data is None:
        reward_data = mc_returns

    # Language list conversion
    if isinstance(language, jnp.ndarray):
        language_list = []
        for lang in language:
            if isinstance(lang, bytes):
                language_list.append(lang.decode('utf-8'))
            else:
                language_list.append(str(lang))
    else:
        language_list = [str(l) if not isinstance(l, bytes) else l.decode('utf-8') for l in language]

    fused_embeddings = get_octo_fused_embeddings(
        octo_model,
        images,
        language_list,
        max_frames_per_batch=max_frames_per_octo_batch,
    )

    return {
        'fused_embeddings': fused_embeddings,
        'actions': actions,
        'mc_returns': mc_returns,
        'masks': masks,
        'rewards': reward_data,
    }

def adapt_trajectory_features(ttt_params, fused, ttt_extractor, config):
    """Apply either full-trajectory or windowed TTT adaptation for evaluation.

    Args:
        ttt_params: Parameters of the ``TTTFeatureExtractor``.
        fused: (B, T, D) fused OCTO embeddings.
        ttt_extractor: Module used to decode adapted features.
        config: Training config (provides adaptation hyperparameters).

    Returns:
        adapted_features: (B, T, projection_dim) tensor post-adaptation.
        adapted_params: Updated parameter tree containing new ``f_adapt`` weights.
        losses: Array of adaptation losses per inner step (shape depends on mode).
    """
    mode = config.get('ttt_adapt_mode', 'full')
    lr = config.get('ttt_adapt_lr', 1e-2)
    steps = config.get('ttt_adapt_steps', 5)

    if mode == 'windowed':
        window_size = config.get('ttt_adapt_window', 8)
        reset = config.get('ttt_adapt_reset', True)
        adapted_features, adapted_params, losses = windowed_test_time_adapt_ttt(
            ttt_params,
            fused,
            window_size=window_size,
            ttt_lr=lr,
            ttt_steps=steps,
            reset=reset,
        )
    else:
        adapted_params, losses = test_time_adapt_ttt(
            ttt_params,
            ttt_extractor,
            fused,
            ttt_lr=lr,
            ttt_steps=steps,
        )
        adapted_features = ttt_extractor.apply(
            {'params': adapted_params},
            fused,
            train=False,
        )

    return adapted_features, adapted_params, losses


def create_sharded_octo_encoder(octo_model, sharding, num_devices, max_frames_per_batch=None):
    """
    Create a sharded OCTO encoder that processes data across devices.
    
    Input: (B, T, H, W, C) - sharded on batch dimension
    Output: (B, T, D) - sharded on batch dimension
    """
    
    def encode_batch(images, language_list):
        """
        Encode images using OCTO.
        
        Args:
            images: (B, T, 256, 256, 3) - already sharded across devices
            language_list: list of B strings
        
        Returns:
            fused_embeddings: (B, T, D) - sharded on batch dim
        """
        B, T = images.shape[:2]
        
        # Flatten for OCTO: (B, T, ...) → (B*T, 1, ...)
        images_flat = images.reshape(B * T, 1, 256, 256, 3)
        language_repeated = [lang for lang in language_list for _ in range(T)]
        
        # Create tasks (B*T tasks, one per frame)
        tasks = octo_model.create_tasks(texts=language_repeated)
        timestep_pad_mask = jnp.ones((B * T, 1), dtype=jnp.bool_)
        
        # Helper to slice pytrees
        def slice_tree(tree, start, end):
            def maybe_slice(x):
                if isinstance(x, (np.ndarray, jnp.ndarray)) and x.ndim >= 1:
                    if x.shape[0] >= end:
                        return x[start:end]
                return x
            return jax.tree_map(maybe_slice, tree)
        
        # Process frames (with optional chunking)
        total_frames = images_flat.shape[0]
        chunk_size = max_frames_per_batch or total_frames
        
        if chunk_size >= total_frames:
            obs = {'image_primary': images_flat}
            output = octo_model.run_transformer(obs, tasks, timestep_pad_mask, train=False)
            readout_tokens = output['readout_action'].tokens
        else:
            readout_chunks = []
            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                obs_chunk = {'image_primary': images_flat[start:end]}
                tasks_chunk = slice_tree(tasks, start, end)
                mask_chunk = timestep_pad_mask[start:end]
                output_chunk = octo_model.run_transformer(obs_chunk, tasks_chunk, mask_chunk, train=False)
                readout_chunks.append(output_chunk['readout_action'].tokens)
            readout_tokens = jnp.concatenate(readout_chunks, axis=0)
        
        # Extract embeddings: (B*T, num_tokens, D) → (B*T, D) → (B, T, D)
        embeddings_flat = readout_tokens[:, 0].mean(axis=1)
        fused_embeddings = embeddings_flat.reshape(B, T, -1)
        
        return fused_embeddings
    
    return encode_batch

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


def _language_list_from_field(language_value, batch_size):
    def _to_text(entry):
        if isinstance(entry, bytes):
            return entry.decode('utf-8')
        return str(entry)

    if isinstance(language_value, (str, bytes)):
        entries = [language_value]
    elif isinstance(language_value, (list, tuple)):
        entries = list(language_value)
    elif isinstance(language_value, (np.ndarray, jnp.ndarray)):
        entries = list(np.asarray(language_value).reshape(-1))
    else:
        entries = [language_value]

    if not entries:
        entries = [""]

    if len(entries) < batch_size:
        repeats = (batch_size + len(entries) - 1) // len(entries)
        entries = (entries * repeats)[:batch_size]
    else:
        entries = entries[:batch_size]

    return [_to_text(entry) for entry in entries]


def _collapse_traj_axes(value, target_rank, traj_len, dtype=None):
    if value is None:
        return None

    arr = jnp.asarray(value, dtype=dtype) if dtype is not None else jnp.asarray(value)

    if arr.ndim == 0:
        shape = (traj_len,) + (1,) * (max(target_rank - 1, 0))
        arr = jnp.broadcast_to(arr, shape)

    while arr.ndim > target_rank:
        arr = arr[:, 0]

    if arr.shape[0] != traj_len:
        new_shape = (traj_len, *arr.shape[1:])
        arr = jnp.reshape(arr, new_shape)

    return arr

def create_trajectory_iterator(
    dataset,
    octo_model=None,  # Not used
    text_processor=None,
    max_frames_per_octo_batch=None,
):
    """Iterator that returns raw images (no OCTO encoding)."""
    iterator = dataset.iterator()
    
    for batch in iterator:
        batch_jax = jax.tree_map(_to_jax_array, batch)
        
        images = batch_jax["observation"]["image_primary"]
        actions = batch_jax["action"]
        language = batch_jax["task"]["language_instruction"]
        
        if images.ndim == 4:
            images = images[None, ...]
        if actions.ndim == 2:
            actions = actions[None, ...]
        
        B, T = images.shape[:2]
        
        # Process language
        language_list = _language_list_from_field(language, B)
        
        # Process rewards/masks
        def ensure_bt_shape(arr, name):
            if arr is None:
                return None
            arr = jnp.asarray(arr)
            if arr.ndim == 0:
                return jnp.full((B, T), arr)
            elif arr.ndim == 1:
                if arr.shape[0] == B:
                    return jnp.broadcast_to(arr[:, None], (B, T))
                elif arr.shape[0] == T:
                    return jnp.broadcast_to(arr[None, :], (B, T))
                else:
                    return jnp.broadcast_to(arr[None, :], (B, T))
            elif arr.ndim == 2:
                if arr.shape == (B, T):
                    return arr
                else:
                    return arr.reshape(B, T)
            else:
                while arr.ndim > 2:
                    arr = arr.squeeze()
                return ensure_bt_shape(arr, name)
        
        mc_returns = ensure_bt_shape(batch_jax.get("mc_return", batch_jax.get("reward")), 'mc_returns')
        if mc_returns is None:
            mc_returns = jnp.zeros((B, T))
        
        masks = ensure_bt_shape(batch_jax.get("td_mask"), 'masks')
        if masks is None:
            masks = jnp.ones((B, T), dtype=jnp.bool_)
        
        rewards = ensure_bt_shape(batch_jax.get("reward", mc_returns), 'rewards')
        if rewards is None:
            rewards = mc_returns

        yield {
            'images': images,
            'language': language_list,
            'actions': actions,
            'mc_returns': mc_returns,
            'masks': masks,
            'rewards': rewards,
        }


def create_full_trajectory_iterator(dataset):
    """Iterator that returns entire trajectories (variable T per episode)."""
    iterator = dataset.iterator()

    for traj in iterator:
        traj_jax = jax.tree_map(_to_jax_array, traj)

        images = traj_jax["observation"]["image_primary"]
        images = jnp.asarray(images)

        if images.ndim == 6:
            images = images[0]
        if images.ndim == 5:
            traj_len = images.shape[0]
            images = images[:, 0]
        elif images.ndim == 4:
            traj_len = images.shape[0]
        else:
            raise ValueError(f"Unexpected image shape for trajectory iterator: {images.shape}")

        images = images[None, ...]

        actions = _collapse_traj_axes(traj_jax["action"], target_rank=2, traj_len=traj_len)
        actions = actions[None, ...]

        rewards = _collapse_traj_axes(traj_jax.get("reward"), target_rank=1, traj_len=traj_len)
        mc_returns = _collapse_traj_axes(traj_jax.get("mc_return"), target_rank=1, traj_len=traj_len)
        masks = _collapse_traj_axes(
            traj_jax.get("td_mask"),
            target_rank=1,
            traj_len=traj_len,
            dtype=jnp.bool_,
        )

        if mc_returns is None and rewards is None:
            mc_returns = jnp.zeros((traj_len,))
        if rewards is None:
            rewards = mc_returns
        if mc_returns is None:
            mc_returns = rewards
        if masks is None:
            masks = jnp.ones((traj_len,), dtype=jnp.bool_)

        rewards = rewards[None, ...]
        mc_returns = mc_returns[None, ...]
        masks = masks[None, ...]

        language_list = _language_list_from_field(
            traj_jax["task"]["language_instruction"],
            batch_size=1,
        )

        yield {
            'images': images,
            'language': language_list,
            'actions': actions,
            'mc_returns': mc_returns,
            'masks': masks,
            'rewards': rewards,
        }

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
    
    # CREATE SHARDED OCTO ENCODER
    max_frames_per_octo_batch = FLAGS.config.get('octo_max_frames_per_batch')
    encode_sharded = create_sharded_octo_encoder(
        octo_model,
        sharding,
        num_devices,
        max_frames_per_batch=max_frames_per_octo_batch
    )
    logging.info(f"Created sharded OCTO encoder across {num_devices} devices")
    
    # ========================================================================
    # Setup data pipeline
    # ========================================================================
    logging.info("Setting up trajectory data pipeline...")
    
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.oxedata_config["oxe_kwargs"]
        )
        oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
        del FLAGS.oxedata_config["oxe_kwargs"]
    
    traj_config = FLAGS.oxedata_config.to_dict()
    
    if "traj_transform_kwargs" not in traj_config:
        traj_config["traj_transform_kwargs"] = {}
    
    window_size = traj_config.get("traj_transform_kwargs", {}).get("window_size", 120)
    traj_config["traj_transform_kwargs"]["goal_relabeling_strategy"] = "uniform"
    
    configured_batch = int(traj_config.get("batch_size", 1))
    
    assert configured_batch % num_devices == 0, (
        f"batch_size={configured_batch} must be divisible by num_devices={num_devices}"
    )
    
    logging.info(f"Using trajectory window_size: {window_size}")
    logging.info(f"Using batch_size: {configured_batch} (divisible by {num_devices} devices)")
    if max_frames_per_octo_batch is not None:
        logging.info(f"Limiting OCTO encoder batches to {max_frames_per_octo_batch} frames")
    
    # Create datasets
    train_data = make_interleaved_dataset(**traj_config, train=True)

    sim_cfg = traj_config.get("sim_data")
    if sim_cfg and sim_cfg.get("enable", False):
        if not sim_cfg.get("tfrecord_dir"):
            raise ValueError("sim_data.enable=True but no tfrecord_dir was provided")
        logging.info(
            "Mixing simulated dataset from %s with weight %.2f",
            sim_cfg.get("tfrecord_dir"),
            sim_cfg.get("sample_weight", 1.0),
        )
        sim_dataset = make_simulated_dataset(
            sim_cfg["tfrecord_dir"],
            train=True,
            traj_transform_kwargs=traj_config["traj_transform_kwargs"],
            frame_transform_kwargs=traj_config["frame_transform_kwargs"],
            shuffle_buffer_size=traj_config["shuffle_buffer_size"],
            batch_size=traj_config.get("batch_size"),
            num_parallel_calls=sim_cfg.get("num_parallel_calls", tf.data.AUTOTUNE),
            num_parallel_reads=sim_cfg.get("num_parallel_reads", tf.data.AUTOTUNE),
        )
        base_weight = 1.0
        sim_weight = sim_cfg.get("sample_weight", 1.0)
        weights = np.array([base_weight, sim_weight], dtype=np.float32)
        weights /= np.sum(weights)
        train_data = dl.DLataset.sample_from_datasets(
            [train_data, sim_dataset], weights=weights
        ).shuffle(traj_config["shuffle_buffer_size"])
    
    val_datasets_kwargs_list = traj_config["dataset_kwargs_list"]
    val_data = create_validation_dataset(
        val_datasets_kwargs_list[0],
        traj_config["traj_transform_kwargs"],
        traj_config["frame_transform_kwargs"],
        train=False
    )
    
    # Manual batching
    train_data_batched = (
        train_data
        .unbatch()
        .repeat()
        .batch(configured_batch)
    )
    
    val_data_batched = (
        val_data
        .unbatch()
        .shuffle(32)
        .repeat()
        .batch(configured_batch)
    )
    # Smaller batches (one trajectory) for qualitative value plotting
    val_traj_data = (
        val_data
        .shuffle(32)
        .repeat()
    )
    
    # Create iterators (return raw images, will shard them)
    train_iter_unsharded = create_trajectory_iterator(
        train_data_batched,
        octo_model,
        max_frames_per_octo_batch=max_frames_per_octo_batch,
    )
    
    val_iter_unsharded = create_trajectory_iterator(
        val_data_batched,
        octo_model,
        max_frames_per_octo_batch=max_frames_per_octo_batch,
    )
    val_traj_iter_unsharded = create_full_trajectory_iterator(
        val_traj_data,
    )
    
    # Shard the data (NOT pmap reshape - use shard_fn)
    train_iter = map(shard_fn, train_iter_unsharded)
    val_iter = map(shard_fn, val_iter_unsharded)
    val_traj_iter = iter(val_traj_iter_unsharded)
    prev_traj_language = None

    # Prime previous-language cache so random-lang plots exist from the first eval
    try:
        _seed_traj = next(val_traj_iter)
    except StopIteration:
        _seed_traj = None
    else:
        prev_traj_language = copy.deepcopy([str(lang) for lang in _seed_traj['language']])
        val_traj_iter = itertools.chain([_seed_traj], val_traj_iter)

    def _next_visual_trajectory():
        nonlocal val_traj_iter
        while True:
            try:
                return next(val_traj_iter)
            except StopIteration:
                val_traj_iter = iter(create_full_trajectory_iterator(
                    val_traj_data,
                ))

    def _prepare_value_traj(raw_traj, ttt_host_params, language_override=None, adapt=True):
        """Build a trajectory dict for value plotting.

        Args:
            raw_traj: Batch of raw images/actions/rewards (B=1).
            ttt_host_params: Parameter tree on host.
            language_override: Optional list of language strings to force.
            adapt: Whether to run TTT adaptation before extracting features.
        """
        images = raw_traj['images']
        actions = raw_traj['actions']
        rewards = raw_traj['rewards']
        masks = raw_traj['masks']
        mc_returns = raw_traj.get('mc_returns')
        language_list = language_override if language_override is not None else raw_traj['language']
        language_list = [str(lang) for lang in language_list]
        if len(language_list) != images.shape[0]:
            raise ValueError(
                f"Language count {len(language_list)} does not match batch size {images.shape[0]}"
            )

        fused = get_octo_fused_embeddings(
            octo_model,
            images,
            language_list,
            max_frames_per_batch=max_frames_per_octo_batch,
        )

        params_for_features = ttt_host_params
        if adapt:
            params_for_features = copy.deepcopy(ttt_host_params)
            base_params_snapshot = copy.deepcopy(params_for_features)

            feature_tensor, adapted_params, _ = adapt_trajectory_features(
                params_for_features,
                fused,
                ttt_extractor,
                FLAGS.config,
            )

            def _tree_max_abs_diff(tree_a, tree_b):
                diffs = []
                for a_leaf, b_leaf in zip(jax.tree_util.tree_leaves(tree_a), jax.tree_util.tree_leaves(tree_b)):
                    if isinstance(a_leaf, (str, bytes)) or isinstance(b_leaf, (str, bytes)):
                        continue
                    a_arr = np.asarray(a_leaf)
                    b_arr = np.asarray(b_leaf)
                    if a_arr.shape != b_arr.shape:
                        continue
                    diffs.append(float(np.max(np.abs(a_arr - b_arr))))
                return max(diffs) if diffs else 0.0

            f_adapt_delta = _tree_max_abs_diff(base_params_snapshot.get('f_adapt'), adapted_params.get('f_adapt'))

            base_feature_tensor = ttt_extractor.apply(
                {'params': base_params_snapshot},
                fused,
                train=False,
            )
            feature_delta = float(np.max(np.abs(np.asarray(feature_tensor) - np.asarray(base_feature_tensor))))
            logging.info(
                "TTT debug: max |f_adapt_delta|=%.6e, max |feature_delta|=%.6e",
                f_adapt_delta,
                feature_delta,
            )
        else:
            feature_tensor = ttt_extractor.apply(
                {'params': params_for_features},
                fused,
                train=False,
            )

        traj_obs = feature_tensor[0]
        traj = {
            'observations': {'image': traj_obs},
            'actions': actions[0],
            'goals': {},
            'rewards': rewards[0],
            'masks': masks[0],
        }
        if mc_returns is not None:
            traj['mc_returns'] = mc_returns[0]
        traj['visuals'] = np.asarray(jax.device_get(images[0]))
        return traj, language_list

    def _plot_value_overlay(rl_agent_obj, traj_ttt, traj_base, seed=None):
        """Render value plots comparing adapted vs. non-adapted features."""

        goals_ttt = traj_ttt.get('goals', {})
        goals_base = traj_base.get('goals', {})

        metrics_ttt = rl_agent_obj.get_eval_values(traj_ttt, seed, goals_ttt)
        metrics_base = rl_agent_obj.get_eval_values(traj_base, seed, goals_base)

        visuals = traj_ttt.get('visuals', traj_ttt['observations']['image'])
        images = np.asarray(visuals)
        if images.ndim == 3:
            images = images[None, ...]

        total_steps = images.shape[0]
        clip_len = max(1, total_steps)
        images_to_show = images[:clip_len]

        def _trim_series(series):
            arr = np.asarray(series)
            clip = min(clip_len, arr.shape[0])
            return arr[:clip]

        metric_keys = list(metrics_ttt.keys())
        num_rows = len(metric_keys) + 1
        fig, axs = plt.subplots(num_rows, 1, figsize=(8, 16))
        canvas = FigureCanvas(fig)

        interval = images_to_show.shape[0] // 8 if images_to_show.shape[0] >= 8 else 1
        interval = max(1, interval)
        sel_images = images_to_show[::interval]
        if sel_images.shape[0] > 0:
            flattened = np.concatenate([np.asarray(frame).squeeze() for frame in sel_images], axis=1)
            axs[0].imshow(flattened)
            axs[0].axis('off')
        else:
            axs[0].axis('off')

        for idx, key in enumerate(metric_keys, start=1):
            series_ttt = _trim_series(metrics_ttt[key])
            series_base = _trim_series(metrics_base.get(key, series_ttt))
            plot_len = min(series_ttt.shape[0], series_base.shape[0])
            if plot_len == 0:
                axs[idx].set_ylabel(key)
                axs[idx].set_title('no valid timesteps')
                continue
            steps = np.arange(plot_len)
            if key in ('rewards', 'masks'):
                axs[idx].plot(steps, series_ttt[:plot_len], label=key, linestyle='-', marker='o')
            else:
                axs[idx].plot(steps, series_ttt[:plot_len], label='TTT-adapted', linestyle='-', marker='o')
                axs[idx].plot(steps, series_base[:plot_len], label='No-TTT', linestyle='--', marker='x')
            combined = np.concatenate([
                series_ttt[:plot_len].flatten(),
                series_base[:plot_len].flatten(),
            ])
            if combined.size > 0:
                ymin, ymax = combined.min(), combined.max()
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                axs[idx].set_ylim([ymin, ymax])
            axs[idx].set_ylabel(key)
            if idx == 1:
                axs[idx].legend(loc='best', fontsize=8)

        axs[-1].set_xlabel('Timestep')
        plt.tight_layout()
        canvas.draw()
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image
    
    logging.info("Data pipeline ready")
    
    # Get example batch (need to encode it to get fused_embeddings for init)

    example_batch_raw = next(train_iter_unsharded)

    # Shard it using existing shard_fn
    example_batch_raw_sharded = shard_fn(example_batch_raw)

    # Encode (images are now sharded)
    example_fused = encode_sharded(
        example_batch_raw_sharded['images'],
        example_batch_raw['language']  # List, not array - don't shard
    )

    example_batch = {
        'fused_embeddings': example_fused,
        'actions': example_batch_raw_sharded['actions'],
        'mc_returns': example_batch_raw_sharded['mc_returns'],
    }

    logging.info(f"Example batch shapes:")
    logging.info(f"  fused_embeddings: {example_batch['fused_embeddings'].shape}")
    logging.info(f"  actions: {example_batch['actions'].shape}")
    
    # ========================================================================
    # Initialize TTT + RL Agent
    # ========================================================================
    agent_type = FLAGS.config.agent
    logging.info(f"Initializing TTT + {agent_type} agent...")
    
    agent_kwargs = FLAGS.config.agent_kwargs
    
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, agent_rng = jax.random.split(rng)
    
    ttt_extractor, rl_agent, ttt_params = create_ttt_agent(
        rng=agent_rng,
        fused_example=example_batch['fused_embeddings'],
        actions_example=example_batch['actions'],
        octo_feature_dim=agent_kwargs.get('octo_feature_dim', octo_feature_dim),
        projection_dim=agent_kwargs.get('projection_dim', 64),
        agent_config=FLAGS.config.to_dict(),
        octo_model=octo_model,
    )
    
    logging.info(f"Model initialized")
    logging.info(f"Total TTT parameters: {sum(x.size for x in jax.tree_util.tree_leaves(ttt_params)):,}")
    
    # ========================================================================
    # Setup optimizer
    # ========================================================================
    base_lr = FLAGS.config.get('base_lr', FLAGS.config.get('learning_rate', 1e-4))
    warmup_steps = FLAGS.config.get('warmup_steps', 2000)
    # weight_decay = FLAGS.config.get('weight_decay', 1e-4)
    
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
        optax.adamw(learning_rate)#, weight_decay=weight_decay)
    )
    
    opt_state = tx.init(ttt_params)
    
    logging.info("Optimizer initialized")
    
    # ========================================================================
    # Create training step (NO pmap, just jit)
    # ========================================================================
    lambda_self = FLAGS.config.get('lambda_self', 0.5)
    lambda_rl = FLAGS.config.get('lambda_rl', 1.0)
    rl_loss_terms = tuple(FLAGS.config.get('rl_loss_terms', ("critic","actor","temperature")))
    update_step_fn = create_ttt_agent_update_step(
        ttt_extractor,
        lambda_self=lambda_self,
        train_config=FLAGS.config.to_dict(),
        lambda_rl=lambda_rl,
        rl_loss_terms=rl_loss_terms,
    )
    
    @jax.jit
    def train_step(ttt_params, opt_state, batch, rng, rl_agent_state):
        return update_step_fn(ttt_params, opt_state, batch, rng, tx, rl_agent_state)
    
    logging.info(f"Training step created with lambda_self={lambda_self} (JIT compiled)")
    
    # ========================================================================
    # Training loop
    # ========================================================================
    logging.info("Starting training...")
    
    # Replicate params, opt_state, and agent state across devices
    ttt_params = jax.device_put(jax.tree_map(jnp.array, ttt_params), sharding.replicate())
    opt_state = jax.device_put(jax.tree_map(jnp.array, opt_state), sharding.replicate())
    rl_agent = jax.device_put(jax.tree_map(jnp.array, rl_agent), sharding.replicate())
    
    for step in tqdm.tqdm(range(FLAGS.config.num_steps)):
        rng, step_rng = jax.random.split(rng)
        
        # Get batch (raw images, sharded)
        try:
            batch_raw = next(train_iter)
        except StopIteration:
            train_data_batched = (train_data.unbatch().repeat().batch(configured_batch))
            train_iter_unsharded = create_trajectory_iterator(
                train_data_batched,
                octo_model,
                max_frames_per_octo_batch=max_frames_per_octo_batch,
            )
            train_iter = map(shard_fn, train_iter_unsharded)
            batch_raw = next(train_iter)
        
        # OCTO encoding (sharded across devices)
        # Images are already sharded by shard_fn on batch dimension
        fused_embeddings = encode_sharded(batch_raw['images'], batch_raw['language'])
        
        # Create batch for training
        batch = {
            'fused_embeddings': fused_embeddings,  # Sharded
            'actions': batch_raw['actions'],  # Sharded
            'rewards': batch_raw['rewards'],  # Sharded
            'masks': batch_raw['masks'],  # Sharded
            'mc_returns': batch_raw['mc_returns'],  # Sharded
        }
        
        # TTT update + feature construction (automatic gradient sync via sharding)
        ttt_params, opt_state, ttt_metrics, agent_batch = train_step(
            ttt_params,
            opt_state,
            batch,
            step_rng,
            rl_agent,
        )

        # RL agent update uses the adapted batch returned above
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
            
            combined_metrics = {
                'ttt': ttt_metrics,
                'agent': agent_metrics,
            }
            metrics_cpu = _flatten_metrics(combined_metrics)
            wandb.log({f"train/{k}": v for k, v in metrics_cpu.items()}, step=step)
            loss_value = metrics_cpu.get('ttt/loss_total')
            if loss_value is not None:
                logging.info(f"Step {step + 1}: loss={loss_value:.4f}")
        
        # Validation (same pattern)
        if (step + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Running validation...")
            val_metrics_list = []
            
            for _ in range(8):
                try:
                    val_batch_raw = next(val_iter)
                except StopIteration:
                    val_data_batched = (val_data.unbatch().shuffle(32).repeat().batch(configured_batch))
                    val_iter_unsharded = create_trajectory_iterator(
                        val_data_batched,
                        octo_model,
                        max_frames_per_octo_batch=max_frames_per_octo_batch,
                    )
                    val_iter = map(shard_fn, val_iter_unsharded)
                    val_batch_raw = next(val_iter)
                
                # Encode
                val_fused = encode_sharded(val_batch_raw['images'], val_batch_raw['language'])
                
                # Get TTT params on host for validation
                def _to_host(x):
                    return jax.device_get(x) if isinstance(x, jax.Array) else x

                ttt_params_host = jax.tree_map(_to_host, ttt_params)
                
                # Compute TTT loss
                val_loss_ttt = ttt_extractor.apply(
                    {'params': ttt_params_host},
                    val_fused,
                    method=ttt_extractor.compute_self_supervised_loss,
                    train=False
                )
                
                _, _, adapt_losses = adapt_trajectory_features(
                    ttt_params_host,
                    val_fused,
                    ttt_extractor,
                    FLAGS.config,
                )
                
                val_metrics_list.append({
                    'val_loss_ttt': val_loss_ttt,
                    'val_ttt_adapt_loss': jnp.mean(adapt_losses),
                })
            
            val_metrics = jax.tree_map(
                lambda *xs: float(jnp.mean(jnp.array(xs))),
                *val_metrics_list
            )
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            logging.info(f"Validation: {val_metrics}")

            logging.info("Plotting value functions...")
            for num in range(3):
                raw_traj = _next_visual_trajectory()
                plot_traj, traj_language = _prepare_value_traj(raw_traj, ttt_params_host, adapt=True)
                baseline_traj, _ = _prepare_value_traj(
                    raw_traj,
                    ttt_params_host,
                    language_override=traj_language,
                    adapt=False,
                )
                rng, val_rng = jax.random.split(rng)
                metrics_ttt = rl_agent.get_eval_values(plot_traj, val_rng, plot_traj.get('goals', {}))
                metrics_base = rl_agent.get_eval_values(baseline_traj, val_rng, baseline_traj.get('goals', {}))

                def _max_value_gap(m1, m2):
                    drop_keys = {'rewards', 'masks'}
                    diffs = []
                    for key in set(m1.keys()).intersection(m2.keys()):
                        if key in drop_keys:
                            continue
                        arr1 = np.asarray(m1[key])
                        arr2 = np.asarray(m2[key])
                        if arr1.shape != arr2.shape:
                            continue
                        if arr1.dtype == bool:
                            diffs.append(float(np.max(arr1 ^ arr2)))
                        else:
                            diffs.append(float(np.max(np.abs(arr1 - arr2))))
                    return max(diffs) if diffs else 0.0

                value_gap = _max_value_gap(metrics_ttt, metrics_base)
                logging.info(
                    "TTT debug: max |value_gap|=%.6e",
                    value_gap,
                )

                plot_image = _plot_value_overlay(rl_agent, plot_traj, baseline_traj, seed=val_rng)
                wandb.log({f"value_plots/traj_{num}": wandb.Image(plot_image)}, step=step)

                if prev_traj_language is not None:
                    rng, rand_rng = jax.random.split(rng)
                    random_lang_override = copy.deepcopy(prev_traj_language)
                    random_traj, _ = _prepare_value_traj(
                        raw_traj,
                        ttt_params_host,
                        language_override=random_lang_override,
                        adapt=True,
                    )
                    baseline_random, _ = _prepare_value_traj(
                        raw_traj,
                        ttt_params_host,
                        language_override=random_lang_override,
                        adapt=False,
                    )
                    random_image = _plot_value_overlay(
                        rl_agent,
                        random_traj,
                        baseline_random,
                        seed=rand_rng,
                    )
                    wandb.log({f"value_plots/traj_random_lang_{num}": wandb.Image(random_image)}, step=step)

                prev_traj_language = copy.deepcopy(traj_language)
        
        # Save checkpoint
        if (step + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_data = {
                'ttt_params': ttt_params,
                'opt_state': opt_state,
                'rl_agent': rl_agent,
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