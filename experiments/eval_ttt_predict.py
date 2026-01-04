"""
Evaluation script for TTT-Predict: Test-Time Training with next-state prediction.

This script evaluates trained TTT-Predict models in SimplerEnv simulation.
Key features:
- Loads TTT-Predict checkpoint (ttt_params + rl_agent)
- Runs TTT adaptation at test time using (obs, action) -> next_obs prediction
- Uses adapted features for value-guided action selection (like V-GPS)
"""

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import os
import numpy as np
from absl import app, flags, logging
import jax
import jax.numpy as jnp
import imageio
from flax.training import checkpoints
import tensorflow as tf
import yaml

os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"

from octo.model.octo_model import OctoModel
from simpler_env.policies.octo.octo_model import OctoInference, rescale_actions, unnormalize_action
from ttt_predict_agent import (
    TTTPredictFeatureExtractor,
    create_ttt_predict_agent,
    sequential_test_time_adapt,
    _dense_project,
)
from ttt_module import TTTModule
from jaxrl_m.agents import agents

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Base Octo model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_integer("num_eval_episodes", 20, "Number of evaluation episodes.")
flags.DEFINE_boolean("use_ttt", True, "Use TTT adaptation or not.")
flags.DEFINE_string("checkpoint_path", "", "Path to TTT-Predict checkpoint.")
flags.DEFINE_string("config_path", "", "Path to config YAML (optional, if not using wandb).")
flags.DEFINE_string("wandb_run", "", "WandB run path to load config from.")
flags.DEFINE_integer("num_samples", 10, "Number of action samples for value-guided selection.")
flags.DEFINE_float("action_temp", 1.0, "Action softmax temperature (beta).")
flags.DEFINE_integer("ttt_steps", 5, "Number of TTT adaptation steps per transition.")
flags.DEFINE_float("ttt_lr", 1e-2, "TTT adaptation learning rate.")
flags.DEFINE_boolean("ttt_reset", True, "Reset TTT params each step (True) or accumulate (False).")


def get_octo_embedding(octo_model, image, language):
    """
    Extract fused vision-language embedding from OCTO model for a single image.

    Args:
        octo_model: Loaded OctoModel
        image: (H, W, C) or (1, H, W, C) image array
        language: Task description string

    Returns:
        embedding: (octo_dim,) JAX array
    """
    # Ensure correct shape
    if image.ndim == 3:
        image = image[None, :]  # (1, H, W, C)

    image = jnp.asarray(image, dtype=jnp.float32)
    B = image.shape[0]

    # Resize if needed
    if image.shape[1:3] != (256, 256):
        from jax.image import resize
        image = resize(image, (B, 256, 256, image.shape[-1]), method='bilinear')

    # OCTO expects (B, T, H, W, C) with T=1
    images_octo = image[:, None, :, :, :]

    # Create task
    tasks = octo_model.create_tasks(texts=[language] * B)

    # Create mask
    timestep_pad_mask = jnp.ones((B, 1), dtype=jnp.bool_)

    # Forward pass
    obs = {'image_primary': images_octo}
    output = octo_model.run_transformer(obs, tasks, timestep_pad_mask, train=False)

    # Extract embeddings: (B, 1, num_tokens, D) -> (B, D)
    readout_tokens = output['readout_action'].tokens
    embeddings = readout_tokens[:, 0].mean(axis=1)

    return embeddings[0] if B == 1 else embeddings


def get_octo_embeddings_batch(octo_model, images, language_list):
    """
    Extract fused vision-language embeddings from OCTO for a batch of images.

    Args:
        octo_model: Loaded OctoModel
        images: (B, H, W, C) images
        language_list: List of B strings

    Returns:
        embeddings: (B, octo_dim) JAX array
    """
    B = images.shape[0]
    images = jnp.asarray(images, dtype=jnp.float32)

    # Resize if needed
    if images.shape[1:3] != (256, 256):
        from jax.image import resize
        images = resize(images, (B, 256, 256, images.shape[-1]), method='bilinear')

    # OCTO expects (B, T, H, W, C) with T=1
    images_octo = images[:, None, :, :, :]

    # Create tasks
    tasks = octo_model.create_tasks(texts=language_list)

    # Create mask
    timestep_pad_mask = jnp.ones((B, 1), dtype=jnp.bool_)

    # Forward pass
    obs = {'image_primary': images_octo}
    output = octo_model.run_transformer(obs, tasks, timestep_pad_mask, train=False)

    # Extract embeddings
    readout_tokens = output['readout_action'].tokens
    embeddings = readout_tokens[:, 0].mean(axis=1)

    return embeddings


def load_ttt_predict_checkpoint(checkpoint_path, config_path="", wandb_run=""):
    """
    Load TTT-Predict checkpoint and reconstruct the model.

    Args:
        checkpoint_path: Path to the checkpoint directory
        config_path: Optional path to config YAML file
        wandb_run: Optional WandB run path to load config from

    Returns:
        ttt_extractor: TTTPredictFeatureExtractor module
        ttt_params: TTT parameters dict
        rl_agent: RL agent with Q-function
        octo_model: OCTO model for embedding extraction
        config: Training configuration dict
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"

    # Load config
    if wandb_run:
        import wandb
        api = wandb.Api()
        run = api.run(wandb_run)
        config = dict(run.config)
    elif config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Try to load from checkpoint directory
        config_file = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        else:
            # Use default ttt_predict_calql config
            from configs.ttt_predict_config import get_config
            config = get_config("ttt_predict_calql").to_dict()

    # Load OCTO model
    encoder_name = config.get("encoder", "octo-small")
    model_type = f"hf://rail-berkeley/{encoder_name}"
    octo_model = OctoModel.load_pretrained(model_type)
    logging.info(f"Loaded OCTO model: {model_type}")

    octo_feature_dim = 384 if "small" in encoder_name else 512

    # Get agent config
    agent_kwargs = config.get("agent_kwargs", {})
    projection_dim = agent_kwargs.get("projection_dim", 256)
    action_dim = agent_kwargs.get("action_dim", 7)
    projection_hidden_dim = config.get("projection_hidden_dim", 128)
    projection_num_layers = config.get("projection_num_layers", 2)
    share_pk_pq = config.get("share_pk_pq", False)

    # Create example batch for initialization
    example_obs = jnp.zeros((1, octo_feature_dim), dtype=jnp.float32)
    example_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)

    # Create TTT extractor and agent
    rng = jax.random.PRNGKey(0)
    ttt_extractor, rl_agent, ttt_params = create_ttt_predict_agent(
        rng=rng,
        obs_example=example_obs,
        next_obs_example=example_obs,
        actions_example=example_actions,
        octo_feature_dim=octo_feature_dim,
        action_dim=action_dim,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_num_layers=projection_num_layers,
        share_pk_pq=share_pk_pq,
        agent_config=config,
        octo_model=octo_model,
    )

    # Load checkpoint
    checkpoint_data = checkpoints.restore_checkpoint(checkpoint_path, target=None)

    # Extract params from checkpoint
    if 'ttt_params' in checkpoint_data:
        ttt_params = checkpoint_data['ttt_params']
    else:
        ttt_params = checkpoint_data

    if 'rl_agent' in checkpoint_data:
        rl_agent = checkpoint_data['rl_agent']

    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    logging.info(f"  TTT params keys: {list(ttt_params.keys())}")

    return ttt_extractor, ttt_params, rl_agent, octo_model, config


class TTTPredictInference:
    """
    Inference wrapper for TTT-Predict models in SimplerEnv.

    Extends OctoInference with TTT adaptation and value-guided action selection.
    """

    def __init__(
        self,
        octo_model,
        ttt_extractor,
        ttt_params,
        rl_agent,
        policy_setup: str = "widowx_bridge",
        num_samples: int = 10,
        action_temp: float = 1.0,
        ttt_lr: float = 1e-2,
        ttt_steps: int = 5,
        ttt_reset: bool = True,
        use_ttt: bool = True,
        horizon: int = 2,
        pred_action_horizon: int = 4,
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        sticky_step: int = 1,
    ):
        self.octo_model = octo_model
        self.ttt_extractor = ttt_extractor
        self.ttt_params = ttt_params
        self.rl_agent = rl_agent

        # TTT settings
        self.use_ttt = use_ttt
        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps
        self.ttt_reset = ttt_reset
        self.projection_dim = ttt_extractor.projection_dim
        self.share_pk_pq = ttt_extractor.share_pk_pq

        # Action selection settings
        self.num_samples = num_samples
        self.action_temp = action_temp

        # Policy setup
        if policy_setup == "widowx_bridge":
            self.dataset_id = "bridge_dataset"
        elif policy_setup == "google_robot":
            self.dataset_id = "fractal20220817_data"
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported")

        self.policy_setup = policy_setup
        self.action_statistics = octo_model.dataset_statistics[self.dataset_id]["action"]

        # Image/action settings
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.sticky_gripper_num_repeat = sticky_step

        # RNG
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            self.rng, _ = jax.random.split(self.rng)

        # State tracking
        self.task = None
        self.task_description = None
        self.current_ttt_params = None  # Will be reset each episode

        # Trajectory history for TTT adaptation
        self.obs_embedding_history = []
        self.action_history = []

        # Gripper state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.is_gripper_closed = False

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def reset(self, task_description: str) -> None:
        """Reset for a new episode."""
        self.task = self.octo_model.create_tasks(texts=[task_description])
        self.task_description = task_description

        # Reset TTT params to base
        self.current_ttt_params = jax.tree_map(lambda x: x, self.ttt_params)

        # Clear history
        self.obs_embedding_history = []
        self.action_history = []

        # Reset gripper state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.is_gripper_closed = False

    def _get_adapted_features(self, obs_embedding: jnp.ndarray) -> jnp.ndarray:
        """
        Get adapted features for the current observation.

        If use_ttt=True and we have history, performs TTT adaptation.
        Otherwise returns base features.
        """
        ttt_module = TTTModule(input_dim=self.projection_dim)

        # Project observation to query space
        if self.share_pk_pq:
            query = _dense_project(obs_embedding[None, :], self.current_ttt_params['P_K'])
        else:
            query = _dense_project(obs_embedding[None, :], self.current_ttt_params['P_Q'])

        # Apply f_adapt
        adapted_features = ttt_module.apply(
            {'params': self.current_ttt_params['f_adapt']},
            query
        )

        return adapted_features[0]  # Remove batch dim

    def _run_ttt_adaptation(self, prev_obs_emb: jnp.ndarray, action: jnp.ndarray, curr_obs_emb: jnp.ndarray):
        """
        Run TTT adaptation using the transition (prev_obs, action) -> curr_obs.

        Updates self.current_ttt_params['f_adapt'].
        """
        from ttt_predict_agent import ttt_adaptation_cosine

        # Get projections
        z_obs = _dense_project(prev_obs_emb[None, :], self.current_ttt_params['P_K'])
        z_action = _dense_project(action[None, :], self.current_ttt_params['P_action'])
        z_input = z_obs + z_action
        z_target = _dense_project(curr_obs_emb[None, :], self.current_ttt_params['P_K'])

        # Stop gradients for test-time adaptation
        z_input = jax.lax.stop_gradient(z_input)
        z_target = jax.lax.stop_gradient(z_target)

        # Get starting params (base or current depending on reset setting)
        if self.ttt_reset:
            start_f_adapt = self.ttt_params['f_adapt']
        else:
            start_f_adapt = self.current_ttt_params['f_adapt']

        # Adapt
        adapted_f_adapt, losses = ttt_adaptation_cosine(
            start_f_adapt,
            z_input,
            z_target,
            self.projection_dim,
            self.ttt_lr,
            self.ttt_steps
        )

        # Update current params
        self.current_ttt_params = dict(self.current_ttt_params)
        self.current_ttt_params['f_adapt'] = adapted_f_adapt

        return losses

    def _get_values_for_actions(self, obs_features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Get Q-values for a batch of actions given observation features.

        Args:
            obs_features: (projection_dim,) adapted features
            actions: (num_samples, action_dim) action candidates

        Returns:
            values: (num_samples,) Q-values
        """
        num_samples = actions.shape[0]

        # Repeat obs features for each action sample
        obs_batch = jnp.repeat(obs_features[None, :], num_samples, axis=0)

        # Create observation dict for agent
        observations = {'image': obs_batch}
        goals = {}

        # Get Q-values from RL agent
        # The agent should have a method to get Q-values
        if hasattr(self.rl_agent, 'get_q_values'):
            values = self.rl_agent.get_q_values(observations, goals, actions)
        elif hasattr(self.rl_agent, 'forward_critic'):
            # Try forward_critic if available
            values = self.rl_agent.forward_critic(observations, goals, actions)
        else:
            # Fallback: access critic network directly
            # This depends on agent implementation
            self.rng, critic_rng = jax.random.split(self.rng)
            values = self.rl_agent.state.apply_fn(
                {'params': self.rl_agent.state.params},
                observations,
                goals,
                actions,
                name='critic',
                rngs={'dropout': critic_rng},
            )
            if values.ndim > 1:
                values = values.mean(axis=-1)  # Average over ensemble

        return values

    def step(self, image: np.ndarray) -> tuple[dict, dict]:
        """
        Take a step: get action using value-guided selection with TTT adaptation.

        Args:
            image: (H, W, 3) uint8 image

        Returns:
            raw_action: dict with world_vector, rotation_delta, open_gripper
            action: dict with processed action for environment
        """
        assert image.dtype == np.uint8
        image = self._resize_image(image)

        # Get OCTO embedding for current observation
        curr_obs_emb = get_octo_embedding(self.octo_model, image, self.task_description)

        # Run TTT adaptation if we have previous observation
        if self.use_ttt and len(self.obs_embedding_history) > 0:
            prev_obs_emb = self.obs_embedding_history[-1]
            prev_action = self.action_history[-1]
            self._run_ttt_adaptation(prev_obs_emb, prev_action, curr_obs_emb)

        # Get adapted features
        adapted_features = self._get_adapted_features(curr_obs_emb)

        # Sample actions from Octo policy
        self.rng, sample_key = jax.random.split(self.rng)

        # Prepare input for Octo
        images_input = image[None, None, :, :, :]  # (1, 1, H, W, C)
        pad_mask = jnp.ones((1, 1), dtype=jnp.float32)

        input_observation = {"image_primary": images_input, "timestep_pad_mask": pad_mask}

        # Sample multiple actions
        norm_raw_actions = self.octo_model.sample_actions(
            input_observation,
            self.task,
            timestep_pad_mask=pad_mask,
            rng=sample_key,
            sample_shape=(self.num_samples,)
        )

        # Shape: (num_samples, 1, pred_action_horizon, 7)
        # Take first timestep action
        norm_action_candidates = norm_raw_actions[:, 0, 0, :]  # (num_samples, 7)

        # Unnormalize for critic evaluation
        critic_actions = unnormalize_action(np.array(norm_action_candidates), self.action_statistics)
        critic_actions = rescale_actions(critic_actions, dataset_id=self.dataset_id, dataset_statistics=self.action_statistics)
        critic_actions = jnp.array(critic_actions)

        # Get values for each action candidate
        values = self._get_values_for_actions(adapted_features, critic_actions)

        # Select action based on values
        if self.action_temp > 0:
            self.rng, select_key = jax.random.split(self.rng)
            action_index = jax.random.categorical(select_key, values / self.action_temp)
        else:
            action_index = jnp.argmax(values)

        selected_norm_action = norm_raw_actions[action_index, 0, 0, :]  # (7,)

        # Store for next TTT adaptation
        self.obs_embedding_history.append(curr_obs_emb)
        # Store the action we're about to take (unnormalized, rescaled)
        selected_critic_action = critic_actions[action_index]
        self.action_history.append(selected_critic_action)

        # Unnormalize selected action for environment
        raw_actions = unnormalize_action(np.array(selected_norm_action)[None, :], self.action_statistics)

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        # Process action for environment (same as OctoInference)
        from transforms3d.euler import euler2axangle

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        # Gripper processing
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            if (raw_action["open_gripper"].item() < 0.5) != self.is_gripper_closed:
                self.gripper_action_repeat += 1
            else:
                self.gripper_action_repeat = 0

            if self.gripper_action_repeat >= self.sticky_gripper_num_repeat:
                self.is_gripper_closed = not self.is_gripper_closed
                self.gripper_action_repeat = 0

            gripper_action = -1.0 if self.is_gripper_closed else 1.0
            action["gripper"] = np.array([gripper_action])

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.config.set_visible_devices([], 'GPU')

    print("=" * 60)
    print("TTT-Predict Evaluation")
    print("=" * 60)
    print(f"Task: {FLAGS.task_name}")
    print(f"Checkpoint: {FLAGS.checkpoint_path}")
    print(f"Use TTT: {FLAGS.use_ttt}")
    print(f"TTT steps: {FLAGS.ttt_steps}, lr: {FLAGS.ttt_lr}, reset: {FLAGS.ttt_reset}")
    print(f"Num samples: {FLAGS.num_samples}, action_temp: {FLAGS.action_temp}")
    print("=" * 60)

    rng = jax.random.PRNGKey(FLAGS.seed)

    # Setup environment
    env = simpler_env.make(FLAGS.task_name)
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print(f"Instruction: {instruction}")

    # Determine policy setup
    if "google" in FLAGS.task_name:
        policy_setup = "google_robot"
        sticky_step = 15
    else:
        policy_setup = "widowx_bridge"
        sticky_step = 3

    # Load TTT-Predict model
    ttt_extractor, ttt_params, rl_agent, octo_model, config = load_ttt_predict_checkpoint(
        FLAGS.checkpoint_path,
        config_path=FLAGS.config_path,
        wandb_run=FLAGS.wandb_run,
    )

    # Create inference wrapper
    model = TTTPredictInference(
        octo_model=octo_model,
        ttt_extractor=ttt_extractor,
        ttt_params=ttt_params,
        rl_agent=rl_agent,
        policy_setup=policy_setup,
        num_samples=FLAGS.num_samples,
        action_temp=FLAGS.action_temp,
        ttt_lr=FLAGS.ttt_lr,
        ttt_steps=FLAGS.ttt_steps,
        ttt_reset=FLAGS.ttt_reset,
        use_ttt=FLAGS.use_ttt,
        init_rng=FLAGS.seed,
        sticky_step=sticky_step,
    )

    # Evaluation loop
    successes = []
    episode_stats_dict = None

    for i in range(FLAGS.num_eval_episodes):
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        model.reset(instruction)

        images = []
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)

        success, truncated = False, False
        timestep = 0

        while not (success or truncated):
            raw_action, action = model.step(image)

            # Assemble action vector
            act_vec = np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]],
                axis=-1,
            )

            # Step environment
            obs, reward, success, truncated, info = env.step(act_vec)

            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        # Record results
        episode_stats = info.get("episode_stats", {})
        if episode_stats_dict is None:
            episode_stats_dict = {k: [v] for k, v in episode_stats.items()}
        else:
            for k, v in episode_stats.items():
                episode_stats_dict[k].append(v)

        successes.append(1 if success else 0)
        print(f"Episode {i}: success={success}, timesteps={timestep}")

        if "consecutive_grasp" in episode_stats_dict:
            print(f"  Grasp: {sum(episode_stats_dict['consecutive_grasp'])}/{i+1} | Success: {sum(successes)}/{i+1}")
        else:
            print(f"  Success: {sum(successes)}/{i+1}")

        # Save video
        ttt_str = f"ttt{FLAGS.ttt_steps}lr{FLAGS.ttt_lr}" if FLAGS.use_ttt else "nottt"
        base_folder = f"logs/ttt_predict_{FLAGS.model_name}_{ttt_str}"
        video_folder = os.path.join(base_folder, f"seed_{FLAGS.seed}/{FLAGS.task_name}")
        os.makedirs(video_folder, exist_ok=True)
        video_path = os.path.join(video_folder, f"{i}_success{success}.mp4")
        imageio.mimsave(video_path, images, fps=10)

    # Final summary
    success_rate = sum(successes) / len(successes)
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"Task: {FLAGS.task_name}")
    print(f"Success Rate: {sum(successes)}/{len(successes)} = {success_rate:.2%}")
    if "consecutive_grasp" in episode_stats_dict:
        grasp_rate = sum(episode_stats_dict['consecutive_grasp']) / len(successes)
        print(f"Grasp Rate: {sum(episode_stats_dict['consecutive_grasp'])}/{len(successes)} = {grasp_rate:.2%}")
    print("=" * 60)

    # Save log
    log_message = f"""TTT-Predict Evaluation
Task: {FLAGS.task_name}
Checkpoint: {FLAGS.checkpoint_path}
Use TTT: {FLAGS.use_ttt}
TTT steps: {FLAGS.ttt_steps}, lr: {FLAGS.ttt_lr}, reset: {FLAGS.ttt_reset}
Num samples: {FLAGS.num_samples}, action_temp: {FLAGS.action_temp}
Seed: {FLAGS.seed}
Success Rate: {sum(successes)}/{len(successes)} = {success_rate:.2%}
"""

    log_file = os.path.join(video_folder, "log.txt")
    with open(log_file, "w") as f:
        f.write(log_message)

    # Append to aggregate log
    log_file_all = os.path.join(base_folder, f"log_{FLAGS.task_name}.txt")
    with open(log_file_all, "a") as f:
        f.write(f"seed: {FLAGS.seed}, success: {sum(successes)}/{len(successes)}, rate: {success_rate:.4f}\n")


if __name__ == "__main__":
    app.run(main)
