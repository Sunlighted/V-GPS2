import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import os
import numpy as np
from absl import app, flags, logging
from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders
import wandb
import jax
import jax.numpy as jnp
import imageio
from simpler_env.policies.octo.octo_model import OctoInference
from octo.model.octo_model import OctoModel
from flax.training import checkpoints
import tensorflow as tf
from tqdm import tqdm

os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"

FLAGS = flags.FLAGS

# 基础实验参数
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_integer("num_eval_episodes", 20, "Number of evaluation episodes.")

# CREPE 核心参数
flags.DEFINE_boolean("use_crepe", True, "Use CREPE steering or not.")
flags.DEFINE_integer("num_replicas", 10, "Number of parallel replicas (M in paper).")
flags.DEFINE_integer("num_crepe_iterations", 3, "Iterations per control step (Online Refinement).")
flags.DEFINE_integer("num_burnin_iterations", 30, "Initial iterations at the first step.")
flags.DEFINE_float("action_temp", 1.0, "Temperature for reward tilting (beta).")

# Critic 相关参数 (沿用 V-GPS 的加载逻辑)
flags.DEFINE_string("vgps_checkpoint", "", "Path to the critic/Q-function checkpoint.")
flags.DEFINE_string("vgps_wandb", "", "Critic wandb run name.")
flags.DEFINE_string("pretrain_method_name", "vgpsfix", "Method name for loading.")
flags.DEFINE_string("override_instruction", "", "Instruction override.")

def load_critic(path, wandb_run_name):
    """加载 Q 函数作为 CREPE 的奖励函数"""
    assert os.path.exists(path), f"Checkpoint path {path} does not exist"
    
    if wandb_run_name == "":
        import yaml
        config_file = "experiments/configs/pretrained_checkpoint.yaml" if FLAGS.pretrain_method_name == 'vgps' else "experiments/configs/pretrained_cqlfix_checkpoint.yaml"
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    # 初始化环境模拟数据以构建 Agent
    if FLAGS.pretrain_method_name == 'vgps':
        encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
        octo_model = None
    else:
        model_type = f"hf://rail-berkeley/{config['encoder']}"
        octo_model = OctoModel.load_pretrained(model_type)
        encoder_def = None

    example_batch = {
        "observations": {"image": np.zeros((1, 256, 256, 3), dtype=np.uint8)},
        "goals": {"language": np.zeros((1, 512), dtype=np.float32)},
        "actions": np.zeros((1, 7), dtype=np.float32),
    }

    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        encoder_def=encoder_def,
        octo_model=octo_model,
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

def main(_):
    logging.set_verbosity(logging.ERROR)
    tf.config.set_visible_devices([], 'GPU')
    
    env = simpler_env.make(FLAGS.task_name)
    
    # 策略设置
    policy_setup = "google_robot" if "google" in FLAGS.task_name else "widowx_bridge"
    sticky_step = 15 if policy_setup == "google_robot" else 3

    # 初始化 Octo 推理类 (假设你已经按照之前的建议在 OctoInference 中集成了 get_crepe_action)
    model = OctoInference(
        model_type=FLAGS.model_name, 
        policy_setup=policy_setup, 
        init_rng=FLAGS.seed, 
        sticky_step=sticky_step
    )

    # 初始化 CREPE 引导
    if FLAGS.use_crepe:
        get_values, critic_text_processor = load_critic(FLAGS.vgps_checkpoint, FLAGS.vgps_wandb)
        # 在 model 中注入 CREPE 必要的组件
        model.init_fk_steering(FLAGS.num_replicas, get_values, critic_text_processor, FLAGS.action_temp, env._max_episode_steps)

    successes = []
    pbar = tqdm(total=FLAGS.num_eval_episodes)

    for i in range(FLAGS.num_eval_episodes):
        obs, reset_info = env.reset()
        instruction = FLAGS.override_instruction if FLAGS.override_instruction else env.get_language_instruction()
        model.reset(instruction)
        
        # CREPE 关键：在 Episode 开始时进行 Burn-in 迭代
        # 此时 model 内部应该初始化 replicas 并运行多次交换以达到平稳分布
        is_first_step = True
        
        images = []
        success, truncated = False, False

        while not (success or truncated):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)

            # 调用带 CREPE 引导的 step
            # 在内部，如果是第一步，运行 num_burnin_iterations；后续运行 num_crepe_iterations
            iters = FLAGS.num_burnin_iterations if is_first_step else FLAGS.num_crepe_iterations
            
            # 这里的 model.step 应该根据是否 use_crepe 内部映射到 get_crepe_action
            raw_action, action = model.step(image)
            is_first_step = False

            act_vec = np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]], axis=-1
            )
            obs, reward, success, truncated, info = env.step(act_vec)

        successes.append(1 if success else 0)
        pbar.set_description(f"SR: {sum(successes)/(i+1):.2f}")
        pbar.update(1)

        # 保存视频
        base_folder = f"logs/{FLAGS.model_name}_CREPE_{FLAGS.use_crepe}"
        video_folder = os.path.join(base_folder, f"seed_{FLAGS.seed}/{FLAGS.task_name}")
        os.makedirs(video_folder, exist_ok=True)
        imageio.mimsave(os.path.join(video_folder, f"{i}_success{success}.mp4"), images, fps=10)

    # 最终统计
    log_message = f"Model: {FLAGS.model_name}\nMethod: CREPE\nTask: {FLAGS.task_name}\n"
    log_message += f"Success Rate: {sum(successes)/len(successes)}\n"
    print(log_message)
    
    with open(os.path.join(video_folder, "log.txt"), "w") as f:
        f.write(log_message)

if __name__ == "__main__":
    app.run(main)