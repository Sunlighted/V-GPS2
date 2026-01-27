import os
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
import wandb
from jax.experimental.compilation_cache import compilation_cache

from octo.data.embedding_dataset import (
    make_embedding_dataset,
    make_embedding_dataset_mix,
    is_embedding_dataset_mix,
)
from octo.data.oxe.oxe_dataset_configs import OXE_EMBEDDING_CONFIGS

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_bridgedata_embedding", "WandB project name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "embedding_data_config",
    None,
    "File path to the embedding dataset configuration.",
    lock_config=False,
)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({"project": FLAGS.project, "exp_descriptor": FLAGS.name})
    variant = FLAGS.config.to_dict()
    variant["embedding_data_config"] = FLAGS.embedding_data_config.to_dict()
    wandb_logger = WandBLogger(
        wandb_config=wandb_config, variant=variant,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    def process_embedding_batch(batch):
        """
        Process a batch from the embedding dataset to be compatible with jaxrl_minimal.
        Includes dimension checks.
        """

        # For language embedding, we use the pre-computed T5 embedding
        # Shape is (batch, num_tokens, 768), we might need to pool or reshape
        language_embedding = batch["task"]["language_embedding"]

        # If language_embedding has shape (batch, num_tokens, 768),
        # we can either use mean pooling or keep as is depending on agent architecture
        if len(language_embedding.shape) == 3:
            # Mean pool over tokens: (batch, num_tokens, 768) -> (batch, 768)
            language_embedding = np.mean(language_embedding, axis=1)
            print(f"{language_embedding.shape}")

        result = dict(
            actions=batch["action"].squeeze(),
            goals=dict(language=language_embedding),
            mc_returns=batch["mc_return"],
            observations=dict(image=batch["observation"]["embedding"]),
            next_observations=dict(image=batch["next_observation"]["embedding"]),
            rewards=batch["reward"],
            masks=batch["td_mask"],
        )

        return result

    # Load embedding dataset configuration
    embedding_config = FLAGS.embedding_data_config

    # Create training dataset
    # Check if dataset_name is a mix or a single dataset
    dataset_name = embedding_config.dataset_name
    is_mix = is_embedding_dataset_mix(dataset_name)

    if is_mix:
        logging.info(f"Loading training embedding dataset mix: {dataset_name}")
        train_dataset, train_stats = make_embedding_dataset_mix(
            name=dataset_name,
            data_dir=embedding_config.data_dir,
            train=True,
            shuffle=True,
            skip_unlabeled=embedding_config.get("skip_unlabeled", True),
            skip_norm=embedding_config.get("skip_norm", False),
        )
    else:
        logging.info(f"Loading training embedding dataset: {dataset_name}")
        train_dataset, train_stats = make_embedding_dataset(
            name=dataset_name,
            data_dir=embedding_config.data_dir,
            train=True,
            shuffle=True,
            skip_unlabeled=embedding_config.get("skip_unlabeled", True),
            skip_norm=embedding_config.get("skip_norm", False),
        )

    # Unbatch trajectories into individual transitions, then shuffle and batch
    train_data = (
        train_dataset
        .unbatch()
        .shuffle(embedding_config.get("shuffle_buffer_size", 100000))
        .repeat()
        .batch(FLAGS.config.batch_size)
    )

    train_data_iter = map(
        shard_fn, map(process_embedding_batch, train_data.prefetch(0).as_numpy_iterator())
    )

    # Create validation dataset
    logging.info("Loading validation embedding dataset...")
    if is_mix:
        val_dataset, _ = make_embedding_dataset_mix(
            name=dataset_name,
            data_dir=embedding_config.data_dir,
            train=False,
            shuffle=False,
            skip_unlabeled=embedding_config.get("skip_unlabeled", False),
            skip_norm=embedding_config.get("skip_norm", False),
            dataset_statistics=train_stats,  # Use training statistics for normalization
        )
    else:
        val_dataset, _ = make_embedding_dataset(
            name=dataset_name,
            data_dir=embedding_config.data_dir,
            train=False,
            shuffle=False,
            skip_unlabeled=embedding_config.get("skip_unlabeled", False),
            skip_norm=embedding_config.get("skip_norm", False),
            dataset_statistics=train_stats,  # Use training statistics for normalization
        )

    val_data = (
        val_dataset
        .unbatch()
        .shuffle(1000)
        .repeat()
        .batch(FLAGS.config.batch_size)
    )

    val_data_iter = map(
        shard_fn, map(process_embedding_batch, val_data.prefetch(0).as_numpy_iterator())
    )

    # Get example batch for initialization
    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Embedding dim: {example_batch['observations']['image'].shape[-1]}")
    logging.info(f"Language embedding dim: {example_batch['goals']['language'].shape[-1]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # For embedding-based training, we don't need an image encoder or Octo model
    # The agent should be configured with use_precomputed_embeddings=True

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)

    # Make sure agent config enables precomputed embeddings
    agent_kwargs = dict(FLAGS.config.agent_kwargs)
    agent_kwargs["use_precomputed_embeddings"] = True

    # Pass octo_model=None since we're using precomputed embeddings
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        octo_model=None,  # Not needed with precomputed embeddings
        **agent_kwargs,
    )

    if FLAGS.config.resume_path:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
        logging.info("Restored agent from %s", FLAGS.config.resume_path)

    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = next(train_data_iter)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            metrics = []
            for _ in range(8):
                batch = next(val_data_iter)
                rng, val_rng = jax.random.split(rng)
                metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i + 1, keep=100
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
