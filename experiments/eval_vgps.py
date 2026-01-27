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
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_integer("num_eval_episodes", 20, "Number of evaluation episodes.")
flags.DEFINE_boolean("use_vgps", None, "Use V-GPS or not", required=True)
flags.DEFINE_string("vgps_checkpoint", "", "vgps checkpoint name.")
flags.DEFINE_string("vgps_wandb", "", "vgps wandb run name.")
flags.DEFINE_integer("num_samples", 10, "Number of action samples.")
flags.DEFINE_float("action_temp", 1.0, "action softmax temperature. The beta value in the paper.")
flags.DEFINE_boolean("add_actions", None, "Add random actions during evaluation")
flags.DEFINE_string("pretrain_method_name", "vgpsfix", "Model name.")

# [修改 1] 添加 override_instruction flag
flags.DEFINE_string("override_instruction", "", "If provided, this instruction will overwrite the environment's ground truth instruction.")

def load_vgps_checkpoint(path, wandb_run_name):
    # check path
    assert os.path.exists(path), f"Checkpoint path {path} does not exist"

    """
    You can either specify wandb_run_name to load the exact configuration from Weights & Biases or use the pretrained_checkpoint.yaml file if you are using the provided pre-trained checkpoints.
    """

    if wandb_run_name == "":
        # load from experiments/configs/pretrained_checkpoints.yaml
        import yaml
        if FLAGS.pretrain_method_name == 'vgps':
            with open("experiments/configs/pretrained_checkpoint.yaml", "r") as f:
                config = yaml.safe_load(f)
        if FLAGS.pretrain_method_name == 'vgpsfix_ca':
            with open("experiments/configs/pretrained_cqlfix_ca_checkpoint.yaml", "r") as f:
                config = yaml.safe_load(f)
        else:
            with open("experiments/configs/pretrained_cqlfix_checkpoint.yaml", "r") as f:
                config = yaml.safe_load(f)
    else:
        # load information from wandb
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    # create encoder from wandb config
    if FLAGS.pretrain_method_name == 'vgps':
        encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
    else:
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

def main(_):
    logging.set_verbosity(logging.ERROR)
    tf.config.set_visible_devices([], 'GPU')
    print(FLAGS.flag_values_dict())
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    
    if 'env' in locals():
        print("Closing existing env")
        env.close()
        del env
    
    env = simpler_env.make(FLAGS.task_name)
    obs, reset_info = env.reset()
    
    real_instruction = env.get_language_instruction()
    print("Reset info", reset_info)
    print(f"Original Environment Instruction: {real_instruction}")
    if FLAGS.override_instruction:
        print(f"!!! WILL OVERRIDE WITH: {FLAGS.override_instruction}")

    if "google" in FLAGS.task_name:
        policy_setup = "google_robot"
        STICKY_GRIPPER_NUM_STEPS = 15
    else:
        policy_setup = "widowx_bridge"
        STICKY_GRIPPER_NUM_STEPS = 3

    # @title Select your model and environment
    tf.config.set_visible_devices([], 'GPU')
    model = OctoInference(model_type=FLAGS.model_name, policy_setup=policy_setup, init_rng=FLAGS.seed, sticky_step=STICKY_GRIPPER_NUM_STEPS)
    if FLAGS.use_vgps:
        assert FLAGS.vgps_checkpoint != ""
        get_values, critic_text_processor, vgps_agent = load_vgps_checkpoint(FLAGS.vgps_checkpoint, FLAGS.vgps_wandb)
        model.init_vgps(FLAGS.num_samples, get_values, critic_text_processor, action_temp=FLAGS.action_temp, max_episode_steps=env._max_episode_steps, add_actions=FLAGS.add_actions)
 

    #@title Run inference
    successes = []
    episode_stats_dict = None
    # store VOC values per episode (all) and VOC when final reward==1
    voc_values = []
    voc_values_success_final = []

    for i in range(FLAGS.num_eval_episodes):
        obs, reset_info = env.reset()
        
        real_instruction = env.get_language_instruction()
        if FLAGS.override_instruction:
            instruction = FLAGS.override_instruction
            print(f"Episode {i}: Overriding '{real_instruction}' -> '{instruction}'")
        else:
            instruction = real_instruction
            # print(f"Episode {i}: Instruction: {instruction}")
            
        model.reset(instruction)

        images = []

        obs_image_list = []
        goals_lang_list = []
        actions_list = []
        rewards_list = []
        masks_list = []

        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)

        if FLAGS.use_vgps:
            lang_goal = critic_text_processor.encode(instruction)   # (D,)
        else:
            lang_goal = None

        predicted_terminated, success, truncated = False, False, False
        timestep = 0

        while not (success or truncated):
            obs_image_list.append(image.copy())

            if lang_goal is not None:
                goals_lang_list.append(lang_goal)

            raw_action, action = model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)

            act_vec = np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]],
                axis=-1,
            )
            actions_list.append(act_vec.copy())
            obs, reward, success, truncated, info = env.step(act_vec)
            
            rewards_list.append(reward)
            
            done = bool(success or truncated)
            masks_list.append(0.0 if done else 1.0)

            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        episode_stats = info.get("episode_stats", {})
        if episode_stats_dict is None:
            episode_stats_dict = {}
            for key, value in episode_stats.items():
                episode_stats_dict[key] = [value]
        else:
            for key, value in episode_stats.items():
                episode_stats_dict[key].append(value)

        if success:
            successes.append(1)
        else:
            successes.append(0)
        print(f"Episode {i}, success: {success}")
        if "consecutive_grasp" in episode_stats_dict:
            print(f"Success Rate: grasp -- {sum(episode_stats_dict['consecutive_grasp'])} / {i+1} | success -- {sum(successes)} / {i + 1}")
        else:
            print(f"Success Rate: success -- {sum(successes)} / {i + 1}")

        base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_octo_ac1"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQL_bridge_fractal_b256_20251115_054407/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_vgps_both-1"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/tine-encoder/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_tine-encoder-5"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/tine-encoder/checkpoint_300000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_tine-encoder-3"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/tine-encoder/checkpoint_100000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_tine-encoder-1"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/tpu-fractal5":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_baseline-fractal-5"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQLFIX_bridge_fractal_b256_octo-small_20251121_151543/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPSFIX_{FLAGS.use_vgps}-5"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQLFIX_bridge_fractal_b256_only-bridge_20251125_210541/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPSFIX_{FLAGS.use_vgps}_only-bridge-5"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/VGPS/VGPS_CalQLFIX_bridge_fractal_b256_only-fractal_20251121_220603/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPSFIX_{FLAGS.use_vgps}_only-fractal-5"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/tine-saencoder/checkpoint_100000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_tine-saencoder-1"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/tine-saencoder/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_tine-saencoder-5"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/dyn_loss_srd/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_dyn_srd"
        if FLAGS.vgps_checkpoint=="/data/Chenyang/value_learning/V-GPS/save/skip-unlabel/checkpoint_500000":
            base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}_skip-unlabelled"

        if FLAGS.override_instruction:
            base_folder += "_WRONG_INSTR"
        if FLAGS.action_temp != 1.0:
            base_folder += f"_temp{FLAGS.action_temp}"
        if FLAGS.add_actions:
            base_folder += f"_add_manual_actions"

        video_folder = os.path.join(base_folder, f"seed_{FLAGS.seed}/{FLAGS.task_name}")
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
        video_path = os.path.join(video_folder, f"{i}_success{success}.mp4")
        imageio.mimsave(video_path, images, fps=10)

        # if FLAGS.use_vgps:

        #     obs_images = np.stack(obs_image_list, axis=0)   # (T, H, W, 3)

        #     if len(goals_lang_list) > 0:
        #         goals_language = np.stack(goals_lang_list, axis=0)  # (T, D)
        #         traj_goals = {"language": goals_language}
        #     else:
        #         traj_goals = {"language": np.zeros((obs_images.shape[0], 1), dtype=np.float32)}

        #     actions_arr = np.stack(actions_list, axis=0)    # (T, action_dim)
        #     rewards_arr = np.array(rewards_list, dtype=np.float32)  # (T,)
        #     masks_arr = np.array(masks_list, dtype=np.float32)      # (T,)

        #     traj = {
        #         "observations": {"image": obs_images},
        #         "goals": traj_goals,
        #         "actions": actions_arr,
        #         "rewards": rewards_arr,
        #         "masks": masks_arr,
        #     }

        #     N = 1 
        #     if (i + 1) % N == 0:
        #         rng, val_rng = jax.random.split(rng)
        #         # request VOC from plot_values_eval
        #         value_plot_img, q_voc = vgps_agent.plot_values_eval(traj, seed=val_rng, return_voc=True)

        #         # record VOC only when last reward == 1.0
        #         try:
        #             last_reward = float(rewards_arr[-1])
        #         except Exception:
        #             last_reward = None
        #         if last_reward == 1.0:
        #             if q_voc != None:
        #                 voc_values_success_final.append(float(q_voc))
        #         else:
        #             if q_voc != None:
        #                 voc_values.append(float(q_voc))
                        

        #         value_plot_folder = os.path.join(video_folder, "value_plots")
        #         os.makedirs(value_plot_folder, exist_ok=True)
        #         value_plot_path = os.path.join(value_plot_folder, f"episode_{i}.png")
        #         imageio.imwrite(value_plot_path, value_plot_img)


    log_message = f"model: {FLAGS.model_name}\nuse_vgps: {FLAGS.use_vgps}\ntask_name: {FLAGS.task_name}\nseed: {FLAGS.seed}\nsuccess_rate: {sum(successes) / len(successes)}"
    
    if FLAGS.override_instruction:
        log_message += f"\nOverride Instruction: {FLAGS.override_instruction}"

    if "consecutive_grasp" in episode_stats_dict:
        log_message += f"\nSuccess Rate: grasp -- {sum(episode_stats_dict['consecutive_grasp'])} / {len(successes)} | success -- {sum(successes)} / {len(successes)}"
    else:
        log_message += f"\nSuccess Rate: success -- {sum(successes)} / {len(successes)}"

    print(log_message)
    # append VOC summaries to the log message
    # try:
    #     voc_mean = float(np.nanmean(voc_values)) if len(voc_values) > 0 else float('nan')
    # except Exception:
    #     voc_mean = float('nan')
    # try:
    #     voc_success_mean = float(np.nanmean(voc_values_success_final)) if len(voc_values_success_final) > 0 else float('nan')
    # except Exception:
    #     voc_success_mean = float('nan')

    # voc_line = f"\nVOC_mean_final_reward_fail: {voc_mean} | VOC_mean_final_reward_success: {voc_success_mean}"
    # print(voc_line)
    # log_message += voc_line
    log_file = os.path.join(video_folder, "log.txt")

    with open(log_file, "w") as f:
        f.write(log_message)
    
    log_file_all = os.path.join(base_folder, f"log_{FLAGS.task_name}.txt")
    with open(log_file_all, "a") as f:
        f.write(f"seed: {FLAGS.seed}, success -- {sum(successes)} / {len(successes)}, success rate: {sum(successes) / len(successes)} -- {FLAGS.pretrain_method_name}\n") # VOC_mean_final_reward0: {voc_mean} | VOC_mean_final_reward1: {voc_success_mean}\n")
        
if __name__ == "__main__":
    app.run(main)