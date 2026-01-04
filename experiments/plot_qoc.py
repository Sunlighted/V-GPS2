import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

import wandb

from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger

from octo.model.octo_model import OctoModel
from simpler_env.policies.octo.octo_model import OctoInference

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

import imageio

os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"

FLAGS = flags.FLAGS

# ==== 来自第一个脚本的 FLAGS ====
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_boolean("use_vgps", None, "Use V-GPS or not", required=True)
flags.DEFINE_string("vgps_checkpoint", "", "vgps checkpoint name.")
flags.DEFINE_string("vgps_wandb", "", "vgps wandb run name.")
flags.DEFINE_integer("num_samples", 50, "Number of action samples.")
flags.DEFINE_float("action_temp", 1.0, "action softmax temperature.")
flags.DEFINE_string("pretrain_method_name", "vgpsfix", "Model name for pretrained method.")

# ==== 来自第二个脚本的数据集 / wandb FLAGS ====
flags.DEFINE_string("name", "eval_vgps_on_dataset", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_bridgedata_eval", "WandB project name.")
flags.DEFINE_integer("num_eval_trajs", 200, "Number of trajectories to evaluate / plot.")
flags.DEFINE_string("save_dir", "./logs_dataset_eval", "Directory to save plots and logs.")
flags.DEFINE_boolean("use_wandb", False, "Whether to log plots to wandb.")

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


def load_vgps_checkpoint(path, wandb_run_name):
    # check path
    assert os.path.exists(path), f"Checkpoint path {path} does not exist"

    """
    You can either specify wandb_run_name to load the exact configuration from Weights & Biases
    or use the pretrained_checkpoint.yaml file if you are using the provided pre-trained checkpoints.
    """

    if wandb_run_name == "":
        import yaml
        if FLAGS.pretrain_method_name == 'vgps':
            with open("experiments/configs/pretrained_checkpoint.yaml", "r") as f:
                config = yaml.safe_load(f)
        else:
            with open("experiments/configs/pretrained_cqlfix_checkpoint.yaml", "r") as f:
                config = yaml.safe_load(f)
    else:
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    # create encoder or octo model from config
    if FLAGS.pretrain_method_name == 'vgps':
        encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
        octo_model = None
    else:
        encoder_def = None
        model_type = f"hf://rail-berkeley/{config['encoder']}"
        octo_model = OctoModel.load_pretrained(model_type)

    example_actions = np.zeros((1, 7), dtype=np.float32)
    example_obs = {
        "image": np.zeros((1, 256, 256, 3), dtype=np.uint8)
    }
    example_batch = {
        "observations": example_obs,
        "goals": {
            "language": np.zeros(
                (
                    1,
                    512,
                ),
                dtype=np.float32,
            ),
        },
        "actions": example_actions,
    }

    # create agent from wandb config
    if FLAGS.pretrain_method_name == 'vgps':
        agent = agents[config["agent"]].create(
            rng=jax.random.PRNGKey(0),
            encoder_def=encoder_def,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            **config["agent_kwargs"],
        )
    else:
        agent = agents[config["agent"]].create(
            rng=jax.random.PRNGKey(0),
            octo_model=octo_model,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            **config["agent_kwargs"],
        )

    # load text processor
    critic_text_processor = text_processors[config["text_processor"]]()

    agent = checkpoints.restore_checkpoint(path, agent)

    def get_values(observations, goals, actions):
        values = agent.get_q_values(observations, goals, actions)
        return values

    return get_values, critic_text_processor, agent


def build_validation_iters(shard_fn):
    
    print(FLAGS.oxedata_config)

    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.oxedata_config["oxe_kwargs"]
        )
        oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
        del FLAGS.oxedata_config["oxe_kwargs"]
    else:
        oxe_kwargs = None

    # text processor
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
        # 保存原始的 language_instruction 文本
        lang_text = [s.decode("utf-8") for s in batch["task"]["language_instruction"]]

        out = dict(
            actions=batch["action"].squeeze(),
            goals=dict(
                language=batch["task"]["language_instruction"],  # 这里还是 byte，后面 process_text 会 encode
                language_text=np.array(lang_text, dtype=object),  # 新增：原始文本
            ),
            mc_returns=batch["mc_return"],
            observations=dict(image=batch["observation"]["image_primary"].squeeze()),
            next_observations=dict(image=batch["next_observation"]["image_primary"].squeeze()),
            rewards=batch["reward"],
            masks=batch["td_mask"],
        )

        # 这里仍然用 text_processor 对 goals["language"] 做 encode
        out = process_text(out)
        return out
    
    # bridge_dataset validation
    val_datasets_kwargs_list, _ = filter_eval_datasets(
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

    val_traj_data_iter = map(
        process_oxe_batch, val_data.shuffle(1000).repeat().iterator()
    )

    # 如果需要 sharded batch 的 val_data_iter，也可以像原脚本一样建，这里主要用轨迹 iter
    # ============ fractal 验证集（如果存在） ============
    if oxe_kwargs is not None and (
        "fractal" in oxe_kwargs.data_mix or
        "oxe" in oxe_kwargs.data_mix or
        "rtx" in oxe_kwargs.data_mix
    ):
        val_datasets_kwargs_list_f, _ = filter_eval_datasets(
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
            ["fractal20220817_data"],
        )
        val_data_fractal = create_validation_dataset(
            val_datasets_kwargs_list_f[0],
            FLAGS.oxedata_config["traj_transform_kwargs"],
            FLAGS.oxedata_config["frame_transform_kwargs"],
            train=False,
        )

        val_traj_data_fractal_iter = map(
            process_oxe_batch, val_data_fractal.shuffle(1000).repeat().iterator()
        )
    else:
        val_traj_data_fractal_iter = None

    return val_traj_data_iter, val_traj_data_fractal_iter


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.config.set_visible_devices([], "GPU")
    print("FLAGS:", FLAGS.flag_values_dict())

    # ==== 设备 / sharding ====
    devices = jax.local_devices()
    print("Devices:", devices)
    num_devices = len(devices)
    # 有可能你这里只做 eval，不需要严格要求 batch_size % num_devices == 0，
    # 如果你后面不用 shard_batch，可以注释掉。
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # ==== wandb logger（可选） ====
    if FLAGS.use_wandb:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({"project": FLAGS.project, "exp_descriptor": FLAGS.name})
        variant = FLAGS.config.to_dict()
        variant["oxe_config"] = FLAGS.oxedata_config.to_dict()
        wandb_logger = WandBLogger(
            wandb_config=wandb_config, variant=variant,
        )
    else:
        wandb_logger = None

    # ==== 创建保存路径 ====
    base_folder = os.path.join(
        FLAGS.save_dir,
        FLAGS.model_name,
        f"VGPS_{FLAGS.use_vgps}_cosine_original_onlybridge",
        FLAGS.task_name,
    )
    if FLAGS.vgps_checkpoint == "/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQLFIX_bridge_fractal_b256_only-bridge_20251125_210541/checkpoint_500000":
        base_folder = os.path.join(
            FLAGS.save_dir,
            FLAGS.model_name,
            f"VGPS_{FLAGS.use_vgps}_cosine_vgpsfix_onlybridge",
            FLAGS.task_name,
        )
        
    if FLAGS.vgps_checkpoint == "/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_20251115_054407/checkpoint_500000":
        base_folder = os.path.join(
            FLAGS.save_dir,
            FLAGS.model_name,
            f"VGPS_{FLAGS.use_vgps}_cosine_vgps_both",
            FLAGS.task_name,
        )
        
    if FLAGS.vgps_checkpoint == "/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQLFIX_bridge_fractal_b256_octo-small_20251121_151543/checkpoint_500000":
        base_folder = os.path.join(
            FLAGS.save_dir,
            FLAGS.model_name,
            f"VGPS_{FLAGS.use_vgps}_cosine_vgpsfix_both",
            FLAGS.task_name,
        )
        
    if FLAGS.vgps_checkpoint == "/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_only-fractal_20251119_194638/checkpoint_500000":
        base_folder = os.path.join(
            FLAGS.save_dir,
            FLAGS.model_name,
            f"VGPS_{FLAGS.use_vgps}_cosine_vgps_onlyfractal",
            FLAGS.task_name,
        )
    os.makedirs(base_folder, exist_ok=True)
    print("Saving plots to:", base_folder)

    # ==== 加载 V-GPS / critic agent ====
    if FLAGS.use_vgps:
        assert FLAGS.vgps_checkpoint != "", "Must provide --vgps_checkpoint when use_vgps=True"
        # 1）先创建 policy_setup（和你 env evaluation 里一致）
        if "google" in FLAGS.task_name:
            policy_setup = "google_robot"
            STICKY_GRIPPER_NUM_STEPS = 15
        else:
            policy_setup = "widowx_bridge"
            STICKY_GRIPPER_NUM_STEPS = 3

        # 2）加载 Octo policy，用来 sample actions
        #    这里的 model 你后面可以传给 vgps_agent 或者在 get_eval_values 里用
        tf.config.set_visible_devices([], "GPU")  # 避免 TF 占 GPU
        model = OctoInference(
            model_type=FLAGS.model_name,
            policy_setup=policy_setup,
            init_rng=FLAGS.seed,
            sticky_step=STICKY_GRIPPER_NUM_STEPS,
        )

        # 3）加载 V-GPS critic agent
        get_values, critic_text_processor, vgps_agent = load_vgps_checkpoint(
            FLAGS.vgps_checkpoint, FLAGS.vgps_wandb
        )

    else:
        raise ValueError("This script is for V-GPS evaluation, please set --use_vgps=True")

    # ==== 构建 validation 迭代器（数据集轨迹） ====
    val_traj_data_iter, val_traj_data_fractal_iter = build_validation_iters(
        shard_fn=shard_fn
    )

    rng = jax.random.PRNGKey(FLAGS.seed)

    # 记录 VOC
    voc_values = []

    # ==== 在 validation data 上 evaluation 并画图 ====
    if "widowx" in FLAGS.task_name:
        for i in range(FLAGS.num_eval_trajs):
            traj = next(val_traj_data_iter)
            
            lang_text_arr = traj["goals"]["language_text"]
            if isinstance(lang_text_arr, str):
                task_description = lang_text_arr

            # 情况 2：是 list / tuple / ndarray，取第一个元素
            else:
                first = lang_text_arr[0]
                # 如果第一个元素还是 ndarray 标量，比如 dtype=object
                if isinstance(first, np.ndarray):
                    task_description = first.item()
                else:
                    task_description = first
                        
                    model.reset(task_description)

            rng, val_rng = jax.random.split(rng)
            value_plot_img, qoc = vgps_agent.plot_qoc(
                traj, seed=val_rng, model=model, num_samples=FLAGS.num_samples
            )

            if qoc is not None:
                voc_values.append(float(qoc))

            # 保存图片
            plot_path = os.path.join(base_folder, f"val_traj_{i}.png")
            imageio.imwrite(plot_path, value_plot_img)
            print(f"Saved value plot for traj {i} to {plot_path}, VOC={qoc}")

            # 可选：同时传到 wandb
            if wandb_logger is not None:
                wandb_plot = wandb.Image(value_plot_img)
                wandb_logger.log(
                    {f"value_plots_dataset/traj_{i}": wandb_plot,
                    f"value_plots_dataset/traj_{i}_voc": qoc},
                    step=i,
                )
    else:
    # fractal 验证集（如果有的话）也可以做同样的 evaluation
        if val_traj_data_fractal_iter is not None:
            for i in range(FLAGS.num_eval_trajs):
                traj = next(val_traj_data_fractal_iter)
                lang_text_arr = traj["goals"]["language_text"]
                if isinstance(lang_text_arr, str):
                    task_description = lang_text_arr

                # 情况 2：是 list / tuple / ndarray，取第一个元素
                else:
                    first = lang_text_arr[0]
                    # 如果第一个元素还是 ndarray 标量，比如 dtype=object
                    if isinstance(first, np.ndarray):
                        task_description = first.item()
                    else:
                        task_description = first
                    
                model.reset(task_description)
                rng, val_rng = jax.random.split(rng)
                value_plot_img, qoc = vgps_agent.plot_qoc(
                    traj, seed=val_rng, model=model, num_samples=FLAGS.num_samples
                )
                if qoc is not None:
                    voc_values.append(float(qoc))

                plot_path = os.path.join(base_folder, f"val_traj_fractal_{i}.png")
                imageio.imwrite(plot_path, value_plot_img)
                print(f"Saved fractal value plot for traj {i} to {plot_path}, VOC={qoc}")

                if wandb_logger is not None:
                    wandb_plot = wandb.Image(value_plot_img)
                    wandb_logger.log(
                        {f"value_plots_dataset/fractal_traj_{i}": wandb_plot,
                        f"value_plots_dataset/fractal_traj_{i}_voc": qoc},
                        step=i,
                    )

    # ====== 汇总 VOC ======
    if len(voc_values) > 0:
        voc_mean = float(np.nanmean(voc_values))
    else:
        voc_mean = float("nan")

    print(f"VOC_mean_over_dataset: {voc_mean}")

    # 写入 log 文件
    log_file = os.path.join(base_folder, "log_dataset_eval.txt")
    with open(log_file, "w") as f:
        f.write(
            f"model: {FLAGS.model_name}\n"
            f"use_vgps: {FLAGS.use_vgps}\n"
            f"task_name: {FLAGS.task_name}\n"
            f"seed: {FLAGS.seed}\n"
            f"num_eval_trajs: {FLAGS.num_eval_trajs}\n"
            f"VOC_mean_over_dataset: {voc_mean}\n"
        )

    if wandb_logger is not None:
        wandb_logger.log({"dataset_eval/VOC_mean": voc_mean}, step=FLAGS.num_eval_trajs)


if __name__ == "__main__":
    app.run(main)