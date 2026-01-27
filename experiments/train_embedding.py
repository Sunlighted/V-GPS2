import os
from functools import partial
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
import dlimp as dl

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
import wandb
from jax.experimental.compilation_cache import compilation_cache
import random

from octo.data.dataset import make_interleaved_dataset
from octo.data.sim_dataset import make_simulated_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets
from octo.model.octo_model import OctoModel

from simpler_env.policies.octo.octo_model import FixedOctoEmbedding

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_bridgedata", "WandB project name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "oxedata_config",
    None,
    "File path to the autonomous data configuration.",
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
    variant["oxe_config"] = FLAGS.oxedata_config.to_dict()
    wandb_logger = WandBLogger(
        wandb_config=wandb_config, variant=variant,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )
    
    if FLAGS.config.get("text_processor") is None:
        text_processor = None
    else:
        text_processor = text_processors[FLAGS.config.text_processor](
            **FLAGS.config.text_processor_kwargs
        )

    def process_text(batch):
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(
                [s.decode("utf-8") for s in batch["goals"]["language"]]
            )
        return batch

    def process_oxe_batch(batch):
        """
        Process a batch from the oxe dataset to be compatible with jaxrl_minimal
        """
        return process_text(
            dict(
                actions=batch["action"].squeeze(),
                goals=dict(language=batch["task"]["language_instruction"]),
                mc_returns=batch["mc_return"],
                observations=dict(image=batch["observation"]["image_primary"].squeeze()),
                next_observations=dict(image=batch["next_observation"]["image_primary"].squeeze()),
                rewards=batch["reward"],
                masks=batch["td_mask"],
            )
        )
    
    print(FLAGS.oxedata_config)
    if "oxe_kwargs" in FLAGS.oxedata_config:
        # create dataset_kwargs_list from oxe_kwargs
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.oxedata_config["oxe_kwargs"]
        )
        oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
        del FLAGS.oxedata_config["oxe_kwargs"]
    
    train_data = make_interleaved_dataset(
        **FLAGS.oxedata_config, train=True
    )
    # train_datasets_kwargs_list, train_sample_weights = filter_eval_datasets(
    #     FLAGS.oxedata_config["dataset_kwargs_list"],
    #     FLAGS.oxedata_config["sample_weights"],
    #     ["bridge_dataset"],
    # )

    # train_data = make_interleaved_dataset(
    #     dataset_kwargs_list=train_datasets_kwargs_list,
    #     sample_weights=train_sample_weights,
    #     train=True,
    #     shuffle_buffer_size=FLAGS.oxedata_config["shuffle_buffer_size"],
    #     traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
    #     frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
    #     batch_size=FLAGS.oxedata_config["batch_size"],
    #     balance_weights=FLAGS.oxedata_config.get("balance_weights", False),
    #     traj_transform_threads=FLAGS.oxedata_config.get("traj_transform_threads", None),
    #     traj_read_threads=FLAGS.oxedata_config.get("traj_read_threads", None),
    # )

    sim_cfg = FLAGS.oxedata_config.get("sim_data")
    if sim_cfg and sim_cfg.get("enable", False):
        if not sim_cfg.get("tfrecord_dir"):
            raise ValueError("sim_data.enable=True but no tfrecord_dir was provided")
        logging.info(
            "Adding simulated dataset from %s with weight %.2f",
            sim_cfg.get("tfrecord_dir"),
            sim_cfg.get("sample_weight", 1.0),
        )
        sim_dataset = make_simulated_dataset(
            sim_cfg["tfrecord_dir"],
            train=True,
            traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
            frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
            shuffle_buffer_size=FLAGS.oxedata_config["shuffle_buffer_size"],
            batch_size=FLAGS.oxedata_config.get("batch_size"),
            num_parallel_calls=sim_cfg.get("num_parallel_calls", tf.data.AUTOTUNE),
            num_parallel_reads=sim_cfg.get("num_parallel_reads", tf.data.AUTOTUNE),
        )
        base_weight = 1.0
        sim_weight = sim_cfg.get("sample_weight", 1.0)
        weight_vec = np.array([base_weight, sim_weight], dtype=np.float32)
        weight_vec /= np.sum(weight_vec)
        train_data = dl.DLataset.sample_from_datasets(
            [train_data, sim_dataset], weights=weight_vec
        ).shuffle(FLAGS.oxedata_config["shuffle_buffer_size"])

    if "fractal" in oxe_kwargs.data_mix or "oxe" in oxe_kwargs.data_mix or "rtx" in oxe_kwargs.data_mix:
        # Try filtering - this excludes fractal from validation datasets
        filtered_result = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["fractal20220817_data"],
        )
        
        # If filtering returned datasets, use them; otherwise use all datasets
        if filtered_result and len(filtered_result) == 2 and len(filtered_result[0]) > 0:
            val_datasets_kwargs_list, _ = filtered_result
        else:
            # No datasets after filtering (fractal-only case), so use fractal itself
            val_datasets_kwargs_list = FLAGS.oxedata_config["dataset_kwargs_list"]
        
        val_data_fractal = create_validation_dataset(
            val_datasets_kwargs_list[0], 
            FLAGS.oxedata_config["traj_transform_kwargs"],
            FLAGS.oxedata_config["frame_transform_kwargs"],
            train=False
        )
        val_traj_data_fractal_iter = map(process_oxe_batch, val_data_fractal.shuffle(1000).repeat().iterator())

        val_data_fractal_iter = map(
            shard_fn, map(process_oxe_batch, val_data_fractal.iterator())
        )

        val_data_fractal_iter = (
                val_data_fractal.unbatch()
                .shuffle(1000)
                .repeat()
                .batch(FLAGS.oxedata_config.batch_size)
                .iterator(prefetch=0)
        )

        val_data_fractal_iter = map(shard_fn, map(process_oxe_batch, val_data_fractal_iter))
        prev_val_traj_fractal = next(val_traj_data_fractal_iter)

    else:
        val_data_fractal_iter = None


    # Try filtering - this excludes bridge from validation datasets
    filtered_result = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["bridge_dataset"],
    )

    # If filtering returned datasets, use them; otherwise use all datasets
    if filtered_result and len(filtered_result) == 2 and len(filtered_result[0]) > 0:
        val_datasets_kwargs_list, _ = filtered_result
    else:
        val_datasets_kwargs_list = FLAGS.oxedata_config["dataset_kwargs_list"]

    val_data = create_validation_dataset(
            val_datasets_kwargs_list[0],
            FLAGS.oxedata_config["traj_transform_kwargs"],
            FLAGS.oxedata_config["frame_transform_kwargs"],
            train=False
        )
    
    val_traj_data_iter = map(process_oxe_batch, val_data.shuffle(1000).repeat().iterator())

    val_data_iter = map(
        shard_fn, map(process_oxe_batch, val_data.iterator())
    )

    val_data_iter = (
            val_data.unbatch()
            .shuffle(1000)
            .repeat()
            .batch(FLAGS.oxedata_config.batch_size)
            .iterator(prefetch=0)
    )

    val_data_iter = map(shard_fn, map(process_oxe_batch, val_data_iter))
    prev_val_traj = next(val_traj_data_iter)

    train_data_iter = map(
        shard_fn, map(process_oxe_batch, train_data.iterator(prefetch=0))
    )

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # define encoder
    # encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    model_type = f"hf://rail-berkeley/{FLAGS.config.encoder}"
    octo_model = OctoModel.load_pretrained(model_type)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        octo_model=octo_model,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.config.resume_path:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
        logging.info("Restored agent from %s", FLAGS.config.resume_path)

    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    
    @jax.jit
    def train_step(agent, batch):
        return agent.update(batch)
    
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = next(train_data_iter)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = train_step(agent, batch)
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

            if val_data_fractal_iter is not None:
                metrics = []
                for _ in range(8):
                    batch = next(val_data_fractal_iter)
                    rng, val_rng = jax.random.split(rng)
                    metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
                metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                wandb_logger.log({"validation/fractal": metrics}, step=i)

            if "iql" in FLAGS.config.agent or "mc_regress" in FLAGS.config.agent or "cql" in FLAGS.config.agent:
                logging.info("Plotting value functions..")
                for num in range(3):
                    traj = next(val_traj_data_iter)
                    rng, val_rng = jax.random.split(rng)
                    plot = agent.plot_values(traj, seed=val_rng)
                    plot = wandb.Image(plot)
                    wandb_logger.log({f"value_plots/traj_{num}": plot}, step=i)

                    plot = agent.plot_values(traj, seed=val_rng, goals=prev_val_traj["goals"])
                    plot = wandb.Image(plot)
                    wandb_logger.log({f"value_plots/traj_random_lang_{num}": plot}, step=i)
                    
                    prev_val_traj = traj

                if val_data_fractal_iter is not None:
                    logging.info("Plotting value functions..")
                    for num in range(3):
                        traj = next(val_traj_data_fractal_iter)
                        rng, val_rng = jax.random.split(rng)
                        plot = agent.plot_values(traj, seed=val_rng)
                        plot = wandb.Image(plot)
                        wandb_logger.log({f"value_plots/fractal/traj_{num}": plot}, step=i)

                        plot = agent.plot_values(traj, seed=val_rng, goals=prev_val_traj_fractal["goals"])
                        plot = wandb.Image(plot)
                        wandb_logger.log({f"value_plots/fractal/traj_random_lang_{num}": plot}, step=i)
                        
                        prev_val_traj_fractal = traj

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