import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

# === 1. 强制无头模式设置 ===
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import imageio
import jax
import numpy as np
import simpler_env
import sapien.core as sapien  # 必须导入 sapien
from PIL import Image, ImageDraw, ImageFont
from absl import app, flags, logging
from simpler_env.policies.octo.octo_model import OctoInference, unnormalize_action
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle, euler2mat
from transforms3d.quaternions import mat2quat

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 5, "Random seed for the Octo policy.")
flags.DEFINE_string("model_name", "octo-small", "Octo model variant to load.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "SimplerEnv task name.")
flags.DEFINE_integer("num_samples", 8, "Number of candidate actions per timestep.")
flags.DEFINE_integer("max_timesteps", 80, "Maximum timesteps per episode.")
flags.DEFINE_integer("num_eval_episodes", 1, "Number of manual-control episodes to run.")
flags.DEFINE_boolean(
    "display",
    False,
    "Whether to open a matplotlib window for each timestep (annotated frames are always written to disk).",
)
flags.DEFINE_string(
    "output_dir",
    "logs/manual_action_custom",
    "Directory to store annotated dual-camera frames for each timestep.",
)

# -----------------------------------------------------------------------------
# 1. Custom Camera Injection Logic (Added Feature)
# -----------------------------------------------------------------------------

def setup_new_camera(env, name="top_down_camera", pos=[0.3, 0.0, 0.6], target=[0.3, 0.0, 0.0]):
    """创建新相机 (自动匹配分辨率 + 带光源)"""
    env = env.unwrapped
    scene = env._scene
    
    # 自动匹配分辨率
    ref_cam = None
    if "3rd_view_camera" in env._cameras: ref_cam = env._cameras["3rd_view_camera"]
    elif "overhead_camera" in env._cameras: ref_cam = env._cameras["overhead_camera"]
    
    width, height = (640, 480)
    print(f"[Setup] Creating camera '{name}' with resolution {width}x{height}")

    camera = scene.add_camera(name, width=width, height=height, fovy=1.57, near=0.1, far=10.0)
    
    # Pose 计算
    cam_pos = np.array(pos)
    target_pos = np.array(target)
    f = target_pos - cam_pos
    f /= np.linalg.norm(f) 
    
    world_up = np.array([0, 0, 1])
    if np.abs(np.dot(f, world_up)) > 0.99: world_up = np.array([1, 0, 0])
    
    s = np.cross(f, world_up); s /= np.linalg.norm(s)
    u = np.cross(s, f); u /= np.linalg.norm(u)
    
    R = np.vstack([s, u, -f]).T
    quat = mat2quat(R)
    camera.set_local_pose(sapien.Pose(p=pos, q=quat))
    
    # 补光灯
    scene.add_point_light(pos, color=[1.0, 1.0, 1.0], shadow=True)
    return camera

def capture_and_inject_obs(custom_camera, obs, camera_name="top_down_camera"):
    """
    [关键] 拍照并转换坐标系 SAPIEN(GL) -> OpenCV
    """
    custom_camera.take_picture()
    
    # 1. 获取 RGB
    rgba = custom_camera.get_float_texture('Color') 
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    
    # 2. 坐标系转换
    model_matrix_gl = custom_camera.get_model_matrix()
    T_gl_cv = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    model_matrix_cv = model_matrix_gl @ T_gl_cv
    extrinsic_cv = np.linalg.inv(model_matrix_cv)
    
    intrinsic = custom_camera.get_intrinsic_matrix()
    
    # 3. 注入
    if "image" not in obs: obs["image"] = {}
    if "camera_param" not in obs: obs["camera_param"] = {}
    
    obs["image"][camera_name] = {"rgb": rgb, "depth": None} # 强制透视
    obs["camera_param"][camera_name] = {
        "intrinsic_cv": intrinsic,
        "extrinsic_cv": extrinsic_cv
    }
    return obs

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def project_3d_to_2d(points_3d: np.ndarray, camera_intrinsic: np.ndarray, camera_extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points_3d.shape[0]
    points_h = np.concatenate([points_3d, np.ones((num_points, 1))], axis=1)
    points_cam = (camera_extrinsic @ points_h.T).T
    points_cam_3d = points_cam[:, :3]
    depths = points_cam_3d[:, 2]
    pixels_h = (camera_intrinsic @ points_cam_3d.T).T
    pixels = pixels_h[:, :2] / np.clip(pixels_h[:, 2:3], a_min=1e-6, a_max=None)
    return pixels, depths

def check_occlusion(pixel_coords: np.ndarray, predicted_depths: np.ndarray, depth_image: Optional[np.ndarray], threshold: float = 0.02) -> np.ndarray:
    if depth_image is None:
        return np.ones(len(pixel_coords), dtype=bool)
    depth = depth_image.squeeze()
    h, w = depth.shape
    visible = np.zeros(len(pixel_coords), dtype=bool)
    for i, (u, v) in enumerate(pixel_coords):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            observed = depth[vi, ui]
            visible[i] = predicted_depths[i] <= observed + threshold
    return visible

def get_ee_pose_world(env) -> Tuple[np.ndarray, np.ndarray]:
    tcp_pose = env.tcp.pose
    return tcp_pose.p, tcp_pose.to_transformation_matrix()[:3, :3]

# -----------------------------------------------------------------------------
# Visualization utilities (Original Style)
# -----------------------------------------------------------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

OPEN_COLOR = np.array([80, 220, 140])
CLOSED_COLOR = np.array([220, 90, 180])

class DisplayManager:
    def __init__(self) -> None:
        plt.ion()
        self.fig = None
        self.axes: List[plt.Axes] = []
        self.num_axes = 0

    def show(self, frames: Sequence[np.ndarray], titles: Sequence[str], timestep: int) -> None:
        num_frames = len(frames)
        if self.fig is None or self.num_axes != num_frames:
            self.close()
            self.fig, axes = plt.subplots(1, num_frames, figsize=(6 * num_frames, 5))
            if num_frames == 1:
                axes = [axes]
            self.axes = list(axes)
            self.num_axes = num_frames
        for ax, frame, title in zip(self.axes, frames, titles):
            ax.clear()
            ax.imshow(frame)
            ax.set_title(f"{title} | t={timestep}")
            ax.axis("off")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = []
            self.num_axes = 0

def annotate_actions_on_view(rgb, action_deltas, rotation_deltas, gripper_states, 
                             ee_pos, ee_rot, K, E, depth, font):
    annotated = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(annotated, "RGBA")
    
    # 投影
    scaled = action_deltas * 3.0
    world_pos = ee_pos[None, :] + (ee_rot @ scaled.T).T
    pixels, depths = project_3d_to_2d(world_pos, K, E)
    
    # Stick (计算方向棍)
    probe = np.array([0, 0, 0.05])
    tips = []
    for i in range(len(action_deltas)):
        # 计算旋转后的棍子尖端位置
        tips.append(world_pos[i] + (ee_rot @ euler2mat(*rotation_deltas[i]) @ probe))
    pixels_tips, _ = project_3d_to_2d(np.array(tips), K, E)

    h, w = rgb.shape[:2]
    # 遮挡检测
    if depth is not None:
        d_map = depth.squeeze()
        visible = []
        for i, (u, v) in enumerate(pixels):
            ui, vi = int(round(u)), int(round(v))
            if 0<=ui<w and 0<=vi<h: visible.append(depths[i] <= d_map[vi, ui] + 0.05)
            else: visible.append(False)
    else:
        visible = [True] * len(pixels)

    OPEN = np.array([80, 220, 140])
    CLOSED = np.array([220, 90, 180])

    for i, (u, v) in enumerate(pixels):
        u_tip, v_tip = pixels_tips[i]
        if not (0<=u<w and 0<=v<h): continue
        color = OPEN if gripper_states[i]>0.5 else CLOSED
        alpha = 255 if visible[i] else 100
        
        # 1. 画蓝色的棍子 (Stick)
        draw.line([(u, v), (u_tip, v_tip)], fill=(0, 0, 255, alpha), width=3)
        # 2. 画圆点
        draw.ellipse([u-6, v-6, u+6, v+6], fill=tuple(color.tolist()+[alpha]), outline=(0,0,0,255))
        # 3. 画数字
        draw.text((u-5, v-20), str(i), font=font, fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0))
    return np.array(annotated)

_DISPLAY_MANAGER: Optional[DisplayManager] = None

def save_and_optionally_display(frames: Sequence[np.ndarray], titles: Sequence[str], output_path: str, timestep: int) -> None:
    max_height = max(frame.shape[0] for frame in frames)
    normalized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h == max_height:
            normalized_frames.append(frame)
            continue
        pad_total = max_height - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = np.pad(frame, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)
        normalized_frames.append(padded)

    stacked = np.concatenate(normalized_frames, axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, stacked)
    logging.info("Saved annotated views to %s", output_path)

    if FLAGS.display:
        global _DISPLAY_MANAGER
        if _DISPLAY_MANAGER is None:
            _DISPLAY_MANAGER = DisplayManager()
        _DISPLAY_MANAGER.show(frames, titles, timestep)

# -----------------------------------------------------------------------------
# Sampling and action utilities
# -----------------------------------------------------------------------------

def sample_action_batch(model: OctoInference, image: np.ndarray, num_samples: int) -> Dict[str, np.ndarray]:
    """Robust version: Uses list instead of deque to prevent errors."""
    assert image.dtype == np.uint8, "Expected uint8 image input"
    resized = model._resize_image(image)
    
    # 强制初始化
    if not hasattr(model, "window_size") or model.window_size is None or model.window_size < 1:
        model.window_size = 2
    if not hasattr(model, "image_history") or model.image_history is None:
        model.image_history = []
    if not isinstance(model.image_history, list):
        model.image_history = list(model.image_history)

    # Append
    model.image_history.append(resized)
    if len(model.image_history) > model.window_size:
        model.image_history = model.image_history[-model.window_size:]

    if len(model.image_history) == 0:
        raise ValueError("History is empty.")

    # Stack & Pad
    images = np.stack(model.image_history, axis=0)
    current_len = len(model.image_history)
    target_len = model.window_size
    
    if current_len < target_len:
        pad_len = target_len - current_len
        padding = np.tile(images[0:1], (pad_len, 1, 1, 1))
        images = np.concatenate([padding, images], axis=0)
        pad_mask = np.array([True] * pad_len + [False] * current_len)
    else:
        pad_mask = np.array([False] * target_len)

    images = images[None] 
    pad_mask = pad_mask[None]

    model.rng, key = jax.random.split(model.rng)
    pad_key = "timestep_pad_mask" if "-1.5" in model.model_type else "pad_mask"
    input_observation = {"image_primary": images, pad_key: pad_mask}
    
    norm_actions = model.model.sample_actions(
        input_observation,
        model.task,
        timestep_pad_mask=pad_mask,
        rng=key,
        sample_shape=(num_samples,),
    )
    norm_actions = np.array(norm_actions)[:, 0]
    first_step = norm_actions[:, 0, :]
    raw_first_step = unnormalize_action(first_step, model.action_statistics)
    
    return {
        "norm_actions": norm_actions,
        "action_deltas": raw_first_step[:, :3],
        "rotation_deltas": raw_first_step[:, 3:6],
        "gripper_states": raw_first_step[:, 6],
    }

def build_raw_action(model: OctoInference, batch: Dict[str, np.ndarray], choice_idx: int) -> dict:
    norm_action = np.array(batch["norm_actions"][choice_idx])
    if model.action_ensemble:
        norm_action = model.action_ensembler.ensemble_action(norm_action)
    if norm_action.ndim == 1:
        norm_action = norm_action[None, :]
    raw_actions = unnormalize_action(norm_action, model.action_statistics)
    if raw_actions.ndim == 1:
        raw_actions = raw_actions[None, :]
    raw_action = {
        "world_vector": np.array(raw_actions[0, :3]),
        "rotation_delta": np.array(raw_actions[0, 3:6]),
        "open_gripper": np.array(raw_actions[0, 6:7]),
    }
    return raw_action

def format_action_for_env(model: OctoInference, raw_action: dict) -> dict:
    action = {}
    action["world_vector"] = raw_action["world_vector"] * model.action_scale
    roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action["rot_axangle"] = action_rotation_ax * action_rotation_angle * model.action_scale

    if (raw_action["open_gripper"].item() < 0.5) != model.is_gripper_closed:
        model.gripper_action_repeat += 1
    else:
        model.gripper_action_repeat = 0

    if model.gripper_action_repeat >= model.sticky_gripper_num_repeat:
        model.is_gripper_closed = not model.is_gripper_closed
        model.gripper_action_repeat = 0

    gripper_action = -1.0 if model.is_gripper_closed else 1.0
    action["gripper"] = np.array([gripper_action])
    action["terminate_episode"] = np.array([0.0])
    return action

# -----------------------------------------------------------------------------
# Camera helpers and prompting
# -----------------------------------------------------------------------------

def infer_policy_camera(env) -> str:
    if "google_robot" in env.robot_uid:
        return "overhead_camera"
    if "widowx" in env.robot_uid:
        return "3rd_view_camera"
    raise NotImplementedError(f"Unknown robot uid: {env.robot_uid}")

def prompt_for_action(num_samples: int) -> Optional[int]:
    prompt = f"Select action index [0-{num_samples - 1}] (or 'q' to quit): "
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in {"q", "quit", "exit"}:
            return None
        if user_input.isdigit():
            choice = int(user_input)
            if 0 <= choice < num_samples:
                return choice
        print(f"Invalid input '{user_input}'. Please enter a value between 0 and {num_samples - 1} or 'q'.")

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main(_):
    logging.set_verbosity(logging.INFO)
    tf_visible = False
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf_visible = True
    except Exception:
        pass

    env = simpler_env.make(FLAGS.task_name)
    policy_camera = infer_policy_camera(env)

    if "google" in FLAGS.task_name:
        policy_setup = "google_robot"
        sticky_steps = 15
    else:
        policy_setup = "widowx_bridge"
        sticky_steps = 3

    model = OctoInference(
        model_type=FLAGS.model_name,
        policy_setup=policy_setup,
        init_rng=FLAGS.seed,
        sticky_step=sticky_steps,
    )

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selection_font = load_font(18)

    # === 1. 初始化自定义相机 ===
    env.reset()
    custom_cam_name = "top_down_custom"
    custom_cam = setup_new_camera(
        env, custom_cam_name,
        pos=[-0.1, 0.4, 1.15], # 使用你指定的最终参数
        target=[0, 0.6, -0.5]
    )

    try:
        for episode_idx in range(FLAGS.num_eval_episodes):
            obs, _ = env.reset()
            instruction = env.get_language_instruction()
            model.reset(instruction)
            logging.info("Episode %d instruction: %s", episode_idx, instruction)

            episode_dir = os.path.join(
                FLAGS.output_dir,
                f"{FLAGS.model_name}_{FLAGS.task_name}",
                f"seed_{FLAGS.seed}",
                f"episode_{episode_idx}_{run_stamp}",
            )
            os.makedirs(episode_dir, exist_ok=True)

            success = False
            truncated = False
            timestep = 0

            while not (success or truncated) and timestep < FLAGS.max_timesteps:
                # === 2. 注入相机并修复 RGB ===
                env.unwrapped._scene.update_render()
                raw_obs = env.unwrapped.get_obs()
                for k, v in raw_obs["image"].items(): 
                    if "Color" in v: v["rgb"] = (v["Color"][..., :3] * 255).astype(np.uint8)
                obs = raw_obs
                
                # 注入自定义相机数据
                obs = capture_and_inject_obs(custom_cam, obs, custom_cam_name)

                # 获取策略图像
                policy_image = obs["image"][policy_camera]["rgb"]
                ee_pos_world, ee_rot_world = get_ee_pose_world(env)

                batch = sample_action_batch(model, policy_image, FLAGS.num_samples)

                annotated_frames = []
                titles = []
                
                # === 3. 三视口循环 [主相机, 自定义相机, 腕部相机] ===
                view_list = [policy_camera, custom_cam_name, "wrist_camera"]
                
                for cam_name in view_list:
                    if cam_name not in obs["image"]: continue
                    
                    rgb = obs["image"][cam_name]["rgb"]
                    # 只有主相机用深度遮挡，其他强制透视
                    depth = obs["image"][cam_name].get("depth") if cam_name == policy_camera else None

                    camera_intrinsic = obs["camera_param"][cam_name]["intrinsic_cv"]
                    camera_extrinsic = obs["camera_param"][cam_name]["extrinsic_cv"]
                    annotated = annotate_actions_on_view(
                        rgb,
                        batch["action_deltas"],
                        batch["rotation_deltas"], # 传入 rotation
                        batch["gripper_states"],
                        ee_pos_world,
                        ee_rot_world,
                        camera_intrinsic,
                        camera_extrinsic,
                        depth,
                        selection_font,
                    )
                    annotated_frames.append(annotated)
                    titles.append(cam_name)

                frame_path = os.path.join(episode_dir, f"timestep_{timestep:03d}.png")
                save_and_optionally_display(annotated_frames, titles, frame_path, timestep)

                choice = prompt_for_action(FLAGS.num_samples)
                if choice is None:
                    logging.info("User requested exit. Stopping simulation.")
                    return

                raw_action = build_raw_action(model, batch, choice)
                action = format_action_for_env(model, raw_action)
                env_step_action = np.concatenate([
                    action["world_vector"],
                    action["rot_axangle"],
                    action["gripper"],
                ])

                obs, reward, success, truncated, _ = env.step(env_step_action)
                logging.info(
                    "t=%d | choice=%d | success=%s | truncated=%s",
                    timestep,
                    choice,
                    success,
                    truncated,
                )
                timestep += 1

            outcome = "SUCCESS" if success else ("TRUNCATED" if truncated else "MAX_STEPS")
            logging.info("Episode %d finished with status %s", episode_idx, outcome)
    finally:
        env.close()
        if _DISPLAY_MANAGER is not None:
            _DISPLAY_MANAGER.close()
        plt.close("all")

    logging.info("Manual action selection run complete. Annotated frames saved under %s", FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)