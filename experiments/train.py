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
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
import wandb
from jax.experimental.compilation_cache import compilation_cache
import random

from octo.data.dataset import make_interleaved_dataset
from octo.data.dataset import make_dataset_from_rlds, apply_trajectory_transforms
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

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
        
    # import copy

    # # ================= ⚡️ 极速部分抽样检查 (Partial Blind Scan) =================
    # CHECK_LIMIT = 10000  # 👈 你想检查多少个 Batch/Sample
    # print(f"\n" + "="*60)
    # print(f"⚡️ 正在构建'影子数据集'并快速抽样检查前 {CHECK_LIMIT} 条数据...")

    # # 1. 复制并修改配置：移除图片读取，极大加速 IO
    # check_config = copy.deepcopy(FLAGS.oxedata_config)
    
    # if "dataset_kwargs_list" in check_config:
    #     for ds_kwargs in check_config["dataset_kwargs_list"]:
    #         ds_kwargs["image_obs_keys"] = {}  # ❌ 不读 RGB
    #         ds_kwargs["depth_obs_keys"] = {}  # ❌ 不读 Depth
    
    # # 禁用帧级变换 (因为没有图片了)
    # check_config["frame_transform_kwargs"] = {}

    # # 2. 构建影子数据集
    # fast_check_data = make_interleaved_dataset(**check_config, train=True)

    # # 3. 开始扫描 (仅扫描前 CHECK_LIMIT 个)
    # empty_count = 0
    # scanned_samples = 0
    
    # # 使用 .take(CHECK_LIMIT) 限制读取数量
    # for i, sample in tqdm.tqdm(enumerate(fast_check_data.take(CHECK_LIMIT).as_numpy_iterator()), total=CHECK_LIMIT):
    #     # 提取指令
    #     raw_langs = sample["task"]["language_instruction"]
    #     raw_langs = np.ravel(raw_langs) # 展平以处理 Batch
        
    #     for lang in raw_langs:
    #         # 处理 numpy/bytes 类型
    #         if isinstance(lang, (np.ndarray, np.generic)):
    #             lang = lang.item()
    #         if isinstance(lang, bytes):
    #             lang = lang.decode("utf-8", errors='ignore')
            
    #         # 检查空字符串
    #         if not str(lang).strip():
    #             empty_count += 1
    #             if empty_count <= 3: # 只打印前 3 个错误示例
    #                  print(f"   ⚠️  [Index {i}] 发现空指令!")
        
    #     scanned_samples += 1

    # # 4. 输出结果
    # print(f"📊 抽样检查结束: 在 {scanned_samples} 个样本中，发现 {empty_count} 条空指令。")
    # if empty_count > 0:
    #     print(f"❌ 警告: 你的训练数据中包含空指令！请检查 'skip_unlabeled' 设置。")
    # else:
    #     print(f"✅ 通过: 前 {CHECK_LIMIT} 条数据均有语言标签。")
    
    # print("="*60 + "\n")

    # # 5. 清理内存
    # del fast_check_data, check_config
    # # =========================================================================

    # assert 0
    
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

    if "fractal" in oxe_kwargs.data_mix or "oxe" in oxe_kwargs.data_mix or "rtx" in oxe_kwargs.data_mix:
        val_datasets_kwargs_list, _ = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["fractal20220817_data"],
        )

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


    val_datasets_kwargs_list, _ = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["bridge_dataset"],
    )
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
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
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