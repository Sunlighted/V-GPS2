from collections import deque
from typing import Optional, Sequence
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from octo.model.octo_model import OctoModel
from octo.model.components.action_heads import ActionHead
import tensorflow as tf
from transforms3d.euler import euler2axangle
from tqdm import tqdm
from simpler_env.utils.action.action_ensemble import ActionEnsembler
from einops import rearrange
import jax.numpy as jnp
import imageio

def rescale_actions(actions, dataset_id, safety_margin=1e-5, dataset_statistics=None):
    """
    rescale xyz, and rotation actions to be within -1 and 1, then clip actions to stay within safety margin
    """
    if "bridge" in dataset_id:
        ACT_MIN = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.])
        ACT_MAX = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.])
    elif "fractal" in dataset_id:
        ACT_MIN = np.array([-2.02045202, -5.49789953, -2.03166342, -1.56991792, -1.56989217, -1.57041943,  0.        ])
        ACT_MAX = np.array([ 2.99845934, 22.09052849,  2.75075245,  1.57063651,  1.53210866, 1.56915224,  1.        ])
    else:
        assert dataset_statistics is not None
        ACT_MIN = dataset_statistics["min"]
        ACT_MAX = dataset_statistics["max"]
        
    mask = np.array([True, True, True, True, True, True, True])
    actions = np.where(
        mask,
        np.clip((actions - ACT_MIN) / (ACT_MAX - ACT_MIN) * 2 - 1, -1 + safety_margin, 1 - safety_margin),
        np.clip(actions, -1 + safety_margin, 1 - safety_margin),
    )
    return np.array(actions)

def unnormalize_action(action, unnormalization_statistics):
    mask = unnormalization_statistics.get(
        "mask", np.ones_like(unnormalization_statistics["mean"], dtype=bool)
    )
    action = action[..., : len(mask)]
    action = np.where(
        mask,
        (action * unnormalization_statistics["std"])
        + unnormalization_statistics["mean"],
        action,
    )
    return action

class OctoInference:
    def __init__(
        self,
        model_type: str = "octo-base",
        policy_setup: str = "widowx_bridge",
        horizon: int = 2,
        pred_action_horizon: int = 4,
        exec_horizon: int = 1,
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        sticky_step: int = 1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            dataset_id = "bridge_dataset"
            action_ensemble = True
            action_ensemble_temp = 0.0
        elif policy_setup == "google_robot":
            dataset_id = "fractal20220817_data"
            action_ensemble = True
            action_ensemble_temp = 0.0
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for octo models.")
        self.policy_setup = policy_setup
        self.sticky_gripper_num_repeat = sticky_step

        if model_type in ["octo-base", "octo-small", "octo-base-1.5", "octo-small-1.5"]:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_statistics = self.model.dataset_statistics[dataset_id]["action"]
        else:
            raise NotImplementedError(f"{model_type} not supported yet.")

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # the purpose of this for loop is just to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng)  # each shape [2,]

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.is_gripper_closed = False

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.num_samples = None
        self.use_vgps = False
        self.dataset_id = dataset_id

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        return images, pad_mask

    def reset(self, task_description: str) -> None:
        self.task = self.model.create_tasks(texts=[task_description])
        if self.use_vgps:
            self.task_value = self.model.create_tasks(texts=[task_description for _ in range(self.num_samples)])
            self.pbar = tqdm(total=self.max_episode_steps)
        
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.is_gripper_closed = False

    def get_action(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
        """
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        self.rng, key = jax.random.split(self.rng)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, timestep_pad_mask=pad_mask, rng=key)
            
        norm_raw_actions = norm_raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)
        assert norm_raw_actions.shape == (self.pred_action_horizon, 7)

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None]  # [1, 7]

        raw_actions = unnormalize_action(norm_raw_actions, self.action_statistics)

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        return raw_action
    
    def get_crepe_action(self, image: np.ndarray, num_iterations: int = 15) -> dict:
        """
        使用 CREPE 算法进行 Q 函数引导的动作采样
        """
        # 1. 基础准备
        prompt_embed_critic = self.critic_text_processor.encode(self.task_description)
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        self.rng, key = jax.random.split(self.rng)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        
        # 预先提取 transformer 编码，避免重复计算
        transformer_outputs = self.model.run_transformer(input_observation, self.task, timestep_pad_mask=pad_mask, train=False)

        # 2. 初始化副本池 (50个副本分布在 50个扩散步上)
        M = self.num_samples  # 50
        time_indices = jnp.arange(M - 1, -1, -1).reshape(-1, 1, 1)
        self.rng, key = jax.random.split(self.rng)
        
        # 初始状态：全部设为随机噪声
        window_size = images.shape[1] 
        replicas = jax.random.normal(
            key, (M, 1, window_size, self.pred_action_horizon * 7)
        )

        # 3. CREPE 主循环
        for _ in range(num_iterations):
            self.rng, key = jax.random.split(self.rng)
            
            # --- A. 并行去噪步 (JIT 加速) ---
            action_head = self.model.module.bind({"params": self.model.params}).heads[
            "action"
            ]

            pred_a0, next_replicas = action_head.predict_step_crepe(
                transformer_outputs, replicas, time_indices, key
            )
            pred_a0_flat = pred_a0[:, 0, -1, :]
            
            # --- B. 计算奖励 (Reward Calculation) ---
            # 将 pred_a0 转回原始动作空间用于 Q 函数评估
            pred_a0_actions = rearrange(pred_a0_flat, "m (h d) -> m h d", h=self.pred_action_horizon)
            critic_actions = unnormalize_action(pred_a0_actions[:, 0, :], self.action_statistics)
            critic_actions = rescale_actions(critic_actions, self.dataset_id, dataset_statistics=self.action_statistics)
            
            # 获取 Q 值
            values = self.get_values(
                observations={"image": np.repeat(images[-1][-1][None], M, axis=0)},
                goals={"language": prompt_embed_critic},
                actions=critic_actions
            ) # shape: (M,)

            # --- C. 副本交换 (Replica Exchange) ---
            # 我们尝试交换相邻的副本。这里使用简单的 Metropolis-Hastings 接受率
            # 核心思想：如果高噪声层的动作预测出的 Q 值更高，就让它“交换”到低噪声层
            for i in range(M - 1):
                # 计算接受概率 alpha
                # 在 Reward-tilting 中，简化接受率为奖励的指数差
                # (注意：实际公式还包含噪声分布的比值 RNE，但对于标准扩散 proposal，这往往简化)
                delta_reward = (values[i+1] - values[i]) / self.action_temp
                
                self.rng, key = jax.random.split(self.rng)
                if jax.random.uniform(key) < jnp.exp(delta_reward):
                    # 交换副本状态
                    next_replicas = next_replicas.at[i].set(next_replicas[i+1])
                    next_replicas = next_replicas.at[i+1].set(next_replicas[i])
                    # 同时交换对应的 Q 值，保证逻辑一致
                    v_i = values[i]
                    v_ip1 = values[i+1]
                    values = values.at[i].set(v_ip1).at[i+1].set(v_i)
            replicas = next_replicas

        # 4. 输出最底层的动作 (time=0 的副本)
        final_action_norm = rearrange(replicas[0, 0, -1], "(h d) -> h d", h=self.pred_action_horizon)
        
        # 后处理与之前一致
        if self.action_ensemble:
            final_action_norm = self.action_ensembler.ensemble_action(final_action_norm)
            final_action_norm = final_action_norm[None]

        raw_actions = unnormalize_action(final_action_norm, self.action_statistics)
        return {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

    def get_fk_steering_action(self, image: np.ndarray) -> dict:
        # 1. 准备输入 (编码文本指令)
        prompt_embed_critic = self.critic_text_processor.encode(self.task_description)
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        self.rng, key = jax.random.split(self.rng)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        
        # 预先提取 transformer 编码，只跑一次特征提取
        transformer_outputs = self.model.run_transformer(input_observation, self.task, timestep_pad_mask=pad_mask, train=False)

        # 2. 初始化 50 个噪声粒子 (M, Batch, Window, D)
        M = self.num_samples
        window_size = transformer_outputs["readout_action"].tokens.shape[1]
        self.rng, key = jax.random.split(self.rng)
        # 形状对齐 Octo 的 4D 期望，避免 safe_zip 错误
        current_particles = jax.random.normal(
            key, (M, 1, window_size, self.pred_action_horizon * 7)
        )
        
        action_head_module = self.model.module.heads["action"]
        
        action_head_params = {"params": self.model.params["heads_action"]}

        @jax.jit
        def one_step_denoise(particles, t, rng_key, transformer_outputs):
            # 将 t 广播为 (M, 1, 1)，以匹配 4D 粒子形状
            M = particles.shape[0]
            t_batch = jnp.full((M,), t).reshape(-1, 1, 1)
            
            # 使用函数式的 apply 接口
            # 第一个参数是变量字典（params），之后是 predict_step_crepe 接收的参数
            return action_head_module.apply(
                action_head_params,    # 显式传递参数数据
                transformer_outputs,   # 以下是 predict_step_crepe 的输入
                particles,
                t_batch,
                rng_key,
                method=action_head_module.predict_step_crepe  # 指定调用的方法名
            )

        # 3. 开始扩散去噪循环 (从 T-1 到 0)
        
        total_steps = action_head_module.diffusion_steps
        resample_steps = [total_steps // 4, 3 * total_steps // 4]
        for i in range(total_steps):
            t = total_steps - 1 - i
            self.rng, key = jax.random.split(self.rng)
            
            # --- A. 并行去噪一步 ---
            # t_batch 为 (M,) 形状，表示所有粒子处于同一时间层
            pred_a0_flat, next_particles = one_step_denoise(
                current_particles, t, key, transformer_outputs
            )
            
            current_particles = next_particles

            # --- B. 简单的重采样逻辑 (FK Steering 3.2节) ---
            # 如果到达重采样间隔且不是最后一步
            if i in resample_steps and t > 0:
                # 1. 提取预测的 a0 用于评估 Q 值 (取最后一帧)
                # pred_a0_flat 形状是 (M, 1, W, 28)
                current_a0 = pred_a0_flat[:, 0, -1, :] 
                pred_a0_actions = rearrange(current_a0, "m (h d) -> m h d", h=self.pred_action_horizon)
                
                # 2. 计算 Q 值 (注意这里要反归一化)
                critic_actions = unnormalize_action(pred_a0_actions[:, 0, :], self.action_statistics)
                critic_actions = rescale_actions(critic_actions, self.dataset_id, dataset_statistics=self.action_statistics)
                
                current_frame = images[0, -1] # 获取当前时刻的 (H, W, 3)
                obs_m = {"image": jnp.repeat(current_frame[None], M, axis=0)} # (M, H, W, 3)

                # --- B. 准备语言输入 (只取一个向量，广播到 M) ---
                # prompt_embed_critic 可能会随 window 增长变为 (T, D)
                lang_goal = prompt_embed_critic
                if lang_goal.ndim > 1:
                    lang_goal = lang_goal[-1] # 强制取最后一个时间步的嵌入 (D,)

                # 广播到 M 个粒子，确保形状为 (M, D)
                goals_m = {"language": jnp.tile(lang_goal[None], (M, 1))}

                # --- C. 调用 get_values ---
                values = self.get_values(
                    observations=obs_m,
                    goals=goals_m,
                    actions=critic_actions # 这里的 critic_actions 已经是 (M, 7)
                )
                
                # 3. 根据 Q 值执行重采样 (SMC 逻辑)
                # 计算权重并归一化
                weights = jax.nn.softmax(values / self.action_temp)
                
                # 按照权重有放回地抽取新的索引
                self.rng, key = jax.random.split(self.rng)
                resample_indices = jax.random.categorical(key, jnp.log(weights + 1e-9), shape=(M,))
                
                # 更新粒子群：优秀的动作被复制，差的动作被替换
                current_particles = current_particles[resample_indices]

        # 4. 循环结束，从最后的 50 个动作中挑一个 Q 值最高的输出
        # (此时 t=0，再次计算一次 values 来做最终选择)
        final_a0 = current_particles[:, 0, -1, :]
        final_a0_actions = rearrange(final_a0, "m (h d) -> m h d", h=self.pred_action_horizon)
        # # ... 同样计算一次 values ...
        # best_idx = jnp.argmax(values)
        # # 1. 选出最优秀的粒子 (4, 7)
        # best_action_chunk = final_a0_actions[best_idx] 

        # # 2. 只取该粒子的第一步动作 (7,)
        # best_action_current = best_action_chunk[0] 

        # # 3. 反归一化 (1, 7)
        # raw_actions = unnormalize_action(best_action_current[None], self.action_statistics)
        # print(raw_actions.shape) # 这时打印出来就是 (1, 7)

        # return {
        #     "world_vector": np.array(raw_actions[0, :3]),
        #     "rotation_delta": np.array(raw_actions[0, 3:6]),
        #     "open_gripper": np.array(raw_actions[0, 6:7]),
        # }
        # --- 为 Critic 准备输入 ---
        # 1. 取每个 Chunk 的第一步动作进行反归一化
        critic_actions_unnorm = unnormalize_action(final_a0_actions[:, 0, :], self.action_statistics)
        # 2. 缩放到 Critic 训练时的范围 (rescale)
        critic_actions_rescaled = rescale_actions(
            critic_actions_unnorm, 
            dataset_id=self.dataset_id, 
            dataset_statistics=self.action_statistics
        )

        current_frame = images[0, -1] # 获取当前时刻的 (H, W, 3)
        obs_m = {"image": jnp.repeat(current_frame[None], M, axis=0)} # (M, H, W, 3)

        # --- B. 准备语言输入 (只取一个向量，广播到 M) ---
        # prompt_embed_critic 可能会随 window 增长变为 (T, D)
        lang_goal = prompt_embed_critic
        if lang_goal.ndim > 1:
            lang_goal = lang_goal[-1] # 强制取最后一个时间步的嵌入 (D,)

        # 广播到 M 个粒子，确保形状为 (M, D)
        goals_m = {"language": jnp.tile(lang_goal[None], (M, 1))}

        # --- C. 调用 get_values ---
        values = self.get_values(
            observations=obs_m,
            goals=goals_m,
            actions=critic_actions_rescaled # 这里的 critic_actions 已经是 (M, 7)
        )

        # 更新进度条信息
        self.pbar.set_description(f"Final Values: max={values.max():.2f}, min={values.min():.2f}")
        self.pbar.update(1)

        # --- 选择最终索引 ---
        if self.action_temp > 0:
            self.rng, key = jax.random.split(self.rng)
            action_index = jax.random.categorical(key, values / self.action_temp)
        else:
            action_index = jnp.argmax(values)

        # 选出最终的归一化动作块 (Horizon, 7)
        selected_norm_chunk = final_a0_actions[action_index]

        # --- 动作集成与反归一化 (Action Ensemble) ---
        if self.action_ensemble:
            # ensemble_action 会将 (Horizon, 7) 处理为 (7,)
            selected_norm_action = self.action_ensembler.ensemble_action(selected_norm_chunk)
            selected_norm_action = selected_norm_action[None] # 补回维度变成 (1, 7)
        else:
            # 如果不使用 ensemble，直接取 Horizon 的第一步
            selected_norm_action = selected_norm_chunk[0:1] # (1, 7)

        # 最终反归一化，得到原始动作空间的值
        raw_actions = unnormalize_action(selected_norm_action, self.action_statistics)

        # 5. 返回结果字典
        return {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

    def init_vgps(self, num_samples, get_values, critic_text_processor, action_temp, max_episode_steps, add_actions=False): # <--- Updated Defaults based on your data
        self.num_samples = num_samples
        self.get_values = get_values
        self.critic_text_processor = critic_text_processor
        self.action_temp = action_temp
        self.add_actions = add_actions
        self.use_vgps = True
        self.max_episode_steps = max_episode_steps
        self.pbar = tqdm(total=self.max_episode_steps)

        # --- Generate 26 Manual Action Primitives ---
        if self.add_actions:
            grid_pos_scale = 0.02  # 2 cm per step
            grid_rot_scale = 0.05   # ~11.5 degrees per step
        
            # 1. Directions (7 types: +x, +y, +z, -x, -y, -z, zero)
            eye = np.eye(3)
            directions = np.concatenate([eye, -eye, np.zeros((1, 3))], axis=0) # (7, 3)
            
            # 2. Translation Actions (7 actions)
            # [pos_step, 0, 0, 0]
            pos_deltas = directions * grid_pos_scale
            rot_deltas_zero = np.zeros_like(pos_deltas)
            trans_actions = np.concatenate([pos_deltas, rot_deltas_zero], axis=1) # (7, 6)
            
            # 3. Rotation Actions (6 actions)
            # [0, 0, 0, rot_step]
            # Exclude the zero vector here to avoid duplicate "No-Op"
            rot_deltas = directions[:6] * grid_rot_scale 
            pos_deltas_zero = np.zeros_like(rot_deltas)
            rot_actions = np.concatenate([pos_deltas_zero, rot_deltas], axis=1) # (6, 6)
            
            # 4. Combine Motion Primitives (13 actions)
            motion_primitives = np.concatenate([trans_actions, rot_actions], axis=0) # (13, 6)

            # 5. Expand with Gripper States (Open=1.0, Closed=0.0)
            # Total = 13 * 2 = 26 actions
            
            # Actions with Open Gripper (1.0)
            actions_open = np.concatenate([motion_primitives, np.ones((len(motion_primitives), 1))], axis=1)
            
            # Actions with Closed Gripper (0.0)
            actions_closed = np.concatenate([
                motion_primitives, 
                np.full((len(motion_primitives), 1), -1.0)
            ], axis=1)
            
            self.manual_actions = np.concatenate([actions_open, actions_closed], axis=0)
            
            print(f"VGPS Initialized with {len(self.manual_actions)} manual actions.")
            print(f"Scale: Pos={grid_pos_scale}, Rot={grid_rot_scale}")
            
    def init_crepe(self, num_samples, get_values, critic_text_processor, action_temp, max_episode_steps):
        """
        初始化 CREPE 引导参数
        num_samples: 副本数量 (M)，通常设为 10-50
        get_values: Q 函数 / 奖励函数接口
        critic_text_processor: 文本编码器
        action_temp: 奖励温度 (beta)
        max_episode_steps: 用于进度条显示
        """
        self.num_samples = num_samples
        self.get_values = get_values
        self.critic_text_processor = critic_text_processor
        self.action_temp = action_temp
        
        # 核心标志位切换
        self.use_crepe = True
        self.use_vgps = False 
        
        # 用于进度条
        self.max_episode_steps = max_episode_steps
        self.pbar = tqdm(total=self.max_episode_steps)

        # 【重要】初始化副本状态为 None
        # 在 reset() 或 get_crepe_action() 第一帧时会将其初始化为随机噪声
        # 之后每一帧会复用这个 self.replicas 实现 Online Refinement
        self.replicas = None

    def init_fk_steering(self, num_samples, get_values, critic_text_processor, action_temp, max_episode_steps, resample_interval=5):
        self.num_samples = num_samples # 比如 50
        self.get_values = get_values
        self.critic_text_processor = critic_text_processor
        self.action_temp = action_temp
        self.max_episode_steps = max_episode_steps
        self.pbar = tqdm(total=self.max_episode_steps)
        self.resample_interval = resample_interval # 每隔多少步重采样一次
        self.use_fk_steering = True
        self.use_vgps = False
        self.use_crepe = False

    def get_vgps_action(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
        """
        prompt_embed_critic = self.critic_text_processor.encode(self.task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, timestep_pad_mask=pad_mask, rng=key, sample_shape=(self.num_samples,))
        
        assert norm_raw_actions.shape == (self.num_samples, 1, self.pred_action_horizon, 7)
        # we first unnormalize the actions with mean/std used for training Octo policy
        critic_actions = unnormalize_action(norm_raw_actions[:, 0, 0, :], self.action_statistics)
        if self.add_actions:
            critic_actions = np.concatenate([critic_actions, self.manual_actions], axis=0)
            # then normalize it for the critic with min/max used for training the critic
            critic_actions = rescale_actions(critic_actions, dataset_id = self.dataset_id, dataset_statistics = self.action_statistics)
            assert critic_actions.shape == (self.num_samples + len(self.manual_actions), 7)
        else:
            critic_actions = rescale_actions(critic_actions, dataset_id=self.dataset_id, dataset_statistics=self.action_statistics)
            assert critic_actions.shape == (self.num_samples, 7)
        total_samples = critic_actions.shape[0]
       
        values = self.get_values(
            observations = {"image": np.repeat(images[-1][-1][None], total_samples, axis=0)},
            goals = {"language": prompt_embed_critic},
            actions = critic_actions
        )
        assert values.shape == (total_samples,)

        self.pbar.set_description(f"Values: max={values.max():.2f}, min={values.min():.2f}, mean={values.mean():.2f}")
        self.pbar.update(1)

        if self.action_temp > 0:
            self.rng, key = jax.random.split(self.rng)
            action_index = jax.random.categorical(key, values / self.action_temp)
            # norm_raw_actions = norm_raw_actions[action_index]
        else:
            action_index = np.argmax(values)
            # norm_raw_actions = norm_raw_actions[action_index]

        if action_index < self.num_samples:
            selected_norm_action_chunk = norm_raw_actions[action_index, 0] # (Horizon, 7)
            
            if self.action_ensemble:
                selected_norm_action_chunk = self.action_ensembler.ensemble_action(selected_norm_action_chunk)
                selected_norm_action_chunk = selected_norm_action_chunk[None] 
            
            raw_actions_chunk = unnormalize_action(selected_norm_action_chunk, self.action_statistics)
            
            final_action = {
                "world_vector": np.array(raw_actions_chunk[0, :3]),
                "rotation_delta": np.array(raw_actions_chunk[0, 3:6]),
                "open_gripper": np.array(raw_actions_chunk[0, 6:7]),
            }
        else:
            manual_idx = action_index - self.num_samples
            selected_manual_action = self.manual_actions[manual_idx] # (7,)
            
            final_action = {
                "world_vector": np.array(selected_manual_action[:3]),
                "rotation_delta": np.array(selected_manual_action[3:6]),
                "open_gripper": np.array(selected_manual_action[6:7]),
            }

        return final_action


    def step(self, image: np.ndarray, num_iterations: Optional[int] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if hasattr(self, "use_fk_steering") and self.use_fk_steering:
            raw_action = self.get_fk_steering_action(image)
        elif self.use_vgps:
        # if self.use_vgps:
            raw_action = self.get_vgps_action(image)
        else:
            raw_action = self.get_action(image)

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
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
        # sticky gripper logic
        elif self.policy_setup == "widowx_bridge":
            
            if (raw_action["open_gripper"].item() < 0.5) != self.is_gripper_closed:
                self.gripper_action_repeat += 1
            else:
                self.gripper_action_repeat = 0

            if self.gripper_action_repeat >= self.sticky_gripper_num_repeat:
                self.is_gripper_closed = not self.is_gripper_closed
                self.gripper_action_repeat = 0


            gripper_action = -1.0 if self.is_gripper_closed else 1.0
            action["gripper"] = (
                np.array([gripper_action])
            )  # binarize gripper action to 1 (open) and -1 (close)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def step_with_samples(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
        """
        Modified step function that also returns all sampled actions for visualization.
        Only works when use_vgps=True.
        
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output (chosen action)
            action: dict; processed action to be sent to the maniskill2 environment
            samples_info: dict containing:
                - 'action_deltas': np.ndarray of shape (num_samples, 3), position deltas for all samples
                - 'q_values': np.ndarray of shape (num_samples,), Q-values for all samples
                - 'gripper_states': np.ndarray of shape (num_samples,), gripper states for all samples
                - 'rotation_deltas': np.ndarray of shape (num_samples, 3), rotation deltas for all samples
                - 'chosen_idx': int, index of chosen action
        """
        assert self.use_vgps, "step_with_samples only works when use_vgps=True"
        
        # Get all sampled actions and Q-values
        raw_action, samples_info = self.get_vgps_action_with_samples(image)
        
        # Process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
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

        return raw_action, action, samples_info

    def get_vgps_action_with_samples(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        """
        Modified version of get_vgps_action that also returns all sampled actions.
        
        Returns:
            raw_action: dict; the chosen action
            samples_info: dict containing all samples and their Q-values
        """
        prompt_embed_critic = self.critic_text_processor.encode(self.task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]

        # we need use a different rng key for each model forward step
        self.rng, key = jax.random.split(self.rng)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, timestep_pad_mask=pad_mask, rng=key, sample_shape=(self.num_samples,))
        
        assert norm_raw_actions.shape == (self.num_samples, 1, self.pred_action_horizon, 7)
        
        # Unnormalize all actions
        all_raw_actions = unnormalize_action(norm_raw_actions[:, 0, 0, :], self.action_statistics)
        
        # Normalize for critic
        critic_actions = rescale_actions(all_raw_actions, dataset_id=self.dataset_id, dataset_statistics=self.action_statistics)
        assert critic_actions.shape == (self.num_samples, 7)
       
        # Get Q-values
        values = self.get_values(
            observations={"image": np.repeat(images[-1][-1][None], self.num_samples, axis=0)},
            goals={"language": prompt_embed_critic},
            actions=critic_actions
        )
        assert values.shape == (self.num_samples,)

        self.pbar.set_description(f"Values: max={values.max():.2f}, min={values.min():.2f}, mean={values.mean():.2f}")
        self.pbar.update(1)

        # Select action
        if self.action_temp > 0:
            self.rng, key = jax.random.split(self.rng)
            action_index = jax.random.categorical(key, values / self.action_temp)
            norm_raw_actions = norm_raw_actions[action_index]
        else:
            action_index = np.argmax(values)
            norm_raw_actions = norm_raw_actions[action_index]

        norm_raw_actions = norm_raw_actions[0]

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None]

        raw_actions = unnormalize_action(norm_raw_actions, self.action_statistics)
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        
        # Prepare samples info for visualization
        samples_info = {
            'action_deltas': all_raw_actions[:, :3],  # (num_samples, 3)
            'rotation_deltas': all_raw_actions[:, 3:6],  # (num_samples, 3)
            'gripper_states': all_raw_actions[:, 6],  # (num_samples,)
            'q_values': np.array(values),  # (num_samples,)
            'chosen_idx': int(action_index),  # scalar
        }

        return raw_action, samples_info
    
      
    def step_with_samples(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
        """
        Modified step function that also returns all sampled actions for visualization.
        Only works when use_vgps=True.
        
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output (chosen action)
            action: dict; processed action to be sent to the maniskill2 environment
            samples_info: dict containing:
                - 'action_deltas': np.ndarray of shape (num_samples, 3), position deltas for all samples
                - 'q_values': np.ndarray of shape (num_samples,), Q-values for all samples
                - 'gripper_states': np.ndarray of shape (num_samples,), gripper states for all samples
                - 'rotation_deltas': np.ndarray of shape (num_samples, 3), rotation deltas for all samples
                - 'chosen_idx': int, index of chosen action
        """
        assert self.use_vgps, "step_with_samples only works when use_vgps=True"
        
        # Get all sampled actions and Q-values
        raw_action, samples_info = self.get_vgps_action_with_samples(image)
        
        # Process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
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

        return raw_action, action, samples_info

    def get_vgps_action_with_samples(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        """
        Modified version of get_vgps_action that also returns all sampled actions.
        
        Returns:
            raw_action: dict; the chosen action
            samples_info: dict containing all samples and their Q-values
        """
        prompt_embed_critic = self.critic_text_processor.encode(self.task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]

        # we need use a different rng key for each model forward step
        self.rng, key = jax.random.split(self.rng)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, timestep_pad_mask=pad_mask, rng=key, sample_shape=(self.num_samples,))
        
        assert norm_raw_actions.shape == (self.num_samples, 1, self.pred_action_horizon, 7)
        
        # Unnormalize all actions
        all_raw_actions = unnormalize_action(norm_raw_actions[:, 0, 0, :], self.action_statistics)
        
        # Normalize for critic
        critic_actions = rescale_actions(all_raw_actions, dataset_id=self.dataset_id, dataset_statistics=self.action_statistics)
        assert critic_actions.shape == (self.num_samples, 7)
       
        # Get Q-values
        values = self.get_values(
            observations={"image": np.repeat(images[-1][-1][None], self.num_samples, axis=0)},
            goals={"language": prompt_embed_critic},
            actions=critic_actions
        )
        assert values.shape == (self.num_samples,)

        self.pbar.set_description(f"Values: max={values.max():.2f}, min={values.min():.2f}, mean={values.mean():.2f}")
        self.pbar.update(1)

        # Select action
        if self.action_temp > 0:
            self.rng, key = jax.random.split(self.rng)
            action_index = jax.random.categorical(key, values / self.action_temp)
            norm_raw_actions = norm_raw_actions[action_index]
        else:
            action_index = np.argmax(values)
            norm_raw_actions = norm_raw_actions[action_index]

        norm_raw_actions = norm_raw_actions[0]

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None]

        raw_actions = unnormalize_action(norm_raw_actions, self.action_statistics)
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        
        # Prepare samples info for visualization
        samples_info = {
            'action_deltas': all_raw_actions[:, :3],  # (num_samples, 3)
            'rotation_deltas': all_raw_actions[:, 3:6],  # (num_samples, 3)
            'gripper_states': all_raw_actions[:, 6],  # (num_samples,)
            'q_values': np.array(values),  # (num_samples,)
            'chosen_idx': int(action_index),  # scalar
        }

        return raw_action, samples_info
    
    def sample_actions_for_image(self, image: np.ndarray, rng: jax.Array, num_samples: int):
        """
        输入:
            image: (H, W, 3), uint8
            rng: jax.random.PRNGKey
            num_samples: 采样个数
        输出:
            norm_actions: (num_samples, 1, pred_action_horizon, 7)  # 和原始 Octo 一致
        """
        assert image.dtype == np.uint8
        image = self._resize_image(image)

        tmp_images = np.stack([image for _ in range(self.horizon)], axis=0)   # (T, H, W, 3)
        tmp_pad_mask = np.ones(self.horizon, dtype=np.float64)               # (T,)

        tmp_images = tmp_images[None]    # (1, T, H, W, 3)
        tmp_pad_mask = tmp_pad_mask[None]  # (1, T)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {
            "image_primary": tmp_images,
            pad_key: tmp_pad_mask,
        }

        norm_raw_actions = self.model.sample_actions(
            input_observation,
            self.task,
            timestep_pad_mask=tmp_pad_mask,
            rng=rng,
            sample_shape=(num_samples,),   # (num_samples, 1, pred_action_horizon, 7)
        )
        return norm_raw_actions


class FixedOctoEmbedding:
    def __init__(
        self,
        model_type: str = "octo-base",
        horizon: int = 2,
        image_size: int = 256,
        init_rng: int = 0,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if model_type in ["octo-base", "octo-small", "octo-base-1.5", "octo-small-1.5"]:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.model = OctoModel.load_pretrained(self.model_type)
        else:
            raise NotImplementedError(f"{model_type} not supported yet.")

        self.image_size = image_size
        self.horizon = horizon
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            self.rng, _ = jax.random.split(self.rng)
        
        # 每个样本的 image history 队列
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True
        )
        return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()

    def _add_image_to_history(self, image: np.ndarray):
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _get_history_and_mask(self):
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        return images, pad_mask

    def reset_history(self):
        self.image_history.clear()
        self.num_image_history = 0

    def get_batch_embeddings(self, images_batch: list[np.ndarray], task_descriptions: list[str]):
        """
        images_batch: list of np.ndarray (H, W, 3)
        task_descriptions: list[str] of same length as images_batch
        Returns:
            embeddings: jnp.ndarray of shape (batch_size, horizon, embedding_dim)
        """
        if len(images_batch) == 1:
            images_batch = [images_batch[0][i] for i in range(images_batch[0].shape[0])]
            
        batch_size = len(images_batch)
        assert len(task_descriptions) == batch_size

        # resize images
        resized_images = np.stack([self._resize_image(img) for img in images_batch], axis=0)
        
        tasks = self.model.create_tasks(texts=task_descriptions, already_encoded=True)

        images_input = resized_images[:, None]  # (batch, horizon=1, H, W, 3)
        pad_mask = np.ones((batch_size, 1), dtype=np.float64)

        self.rng, key = jax.random.split(self.rng)
        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images_input, pad_key: pad_mask}

        transformer_outputs = jax.lax.stop_gradient(
            self.model.run_transformer(input_observation, tasks, timestep_pad_mask=pad_mask, train=False)
        )
        embeddings = transformer_outputs['readout_action'].tokens
        embeddings = embeddings.mean(axis=-2)
        
        # action_head: ActionHead = self.model.module.bind({"params": self.model.params}).heads[
        #     "action"
        # ]
        
        # embedding = action_head.get_embedding(transformer_outputs)

        return embeddings  # shape: (batch_size, horizon, embedding_dim)
