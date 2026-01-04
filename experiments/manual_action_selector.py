import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import imageio
import jax
import matplotlib.pyplot as plt
import numpy as np
import simpler_env
from PIL import Image, ImageDraw, ImageFont
from absl import app, flags, logging
from simpler_env.policies.octo.octo_model import OctoInference, unnormalize_action
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 5, "Random seed for the Octo policy.")
flags.DEFINE_string("model_name", "octo-small", "Octo model variant to load.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "SimplerEnv task name.")
flags.DEFINE_integer("num_samples", 8, "Number of candidate actions per timestep.")
flags.DEFINE_integer("max_timesteps", 60, "Maximum timesteps per episode.")
flags.DEFINE_integer("num_eval_episodes", 1, "Number of manual-control episodes to run.")
flags.DEFINE_string(
    "camera_pair",
    "",
    "Optional comma-separated camera names to annotate (defaults to policy camera + a secondary view).",
)
flags.DEFINE_boolean(
    "display",
    True,
    "Whether to open a matplotlib window for each timestep (annotated frames are always written to disk).",
)
flags.DEFINE_string(
    "output_dir",
    "logs/manual_action_picker",
    "Directory to store annotated dual-camera frames for each timestep.",
)

# -----------------------------------------------------------------------------
# Geometry helpers (duplicated from visualize_vgps_actions.py to avoid flag conflicts)
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
# Visualization utilities
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


def annotate_actions_on_view(
    rgb_image: np.ndarray,
    action_deltas: np.ndarray,
    gripper_states: np.ndarray,
    ee_pos_world: np.ndarray,
    ee_rot_world: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    depth_image: Optional[np.ndarray],
    font: ImageFont.ImageFont,
) -> np.ndarray:
    annotated = Image.fromarray(rgb_image.copy())
    draw = ImageDraw.Draw(annotated, "RGBA")

    scaled_deltas = action_deltas * 3.0
    action_deltas_world = (ee_rot_world @ scaled_deltas.T).T
    predicted_positions = ee_pos_world[None, :] + action_deltas_world
    pixels, depths = project_3d_to_2d(predicted_positions, camera_intrinsic, camera_extrinsic)
    visible = check_occlusion(pixels, depths, depth_image, threshold=0.05)

    height, width = rgb_image.shape[:2]
    for idx, (pixel, is_visible) in enumerate(zip(pixels, visible)):
        u, v = pixel
        if not (0 <= u < width and 0 <= v < height):
            continue
        color = OPEN_COLOR if gripper_states[idx] > 0.5 else CLOSED_COLOR
        alpha = 220 if is_visible else 120
        radius = 8
        bbox = [u - radius, v - radius, u + radius, v + radius]
        draw.ellipse(bbox, fill=tuple(color.tolist() + [alpha]), outline=(0, 0, 0, 255), width=2)
        label = str(idx)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_pos = (u - text_w / 2, v - text_h / 2)
        text_fill = (20, 20, 20, 255) if gripper_states[idx] > 0.5 else (255, 255, 255, 255)
        draw.text(text_pos, label, font=font, fill=text_fill)

    instructions = "Color: green=open, magenta=closed"
    draw.rectangle([10, 10, 10 + font.size * 12, 10 + font.size + 8], fill=(0, 0, 0, 160))
    draw.text((14, 12), instructions, font=font, fill=(255, 255, 255, 255))

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
    assert image.dtype == np.uint8, "Expected uint8 image input"
    resized = model._resize_image(image)
    model._add_image_to_history(resized)
    images, pad_mask = model._obtain_image_history_and_mask()
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
    norm_actions = np.array(norm_actions)  # (num_samples, 1, pred_horizon, 7)
    norm_actions = norm_actions[:, 0]
    first_step = norm_actions[:, 0, :]
    raw_first_step = unnormalize_action(first_step, model.action_statistics)
    batch = {
        "norm_actions": norm_actions,
        "action_deltas": raw_first_step[:, :3],
        "rotation_deltas": raw_first_step[:, 3:6],
        "gripper_states": raw_first_step[:, 6],
    }
    return batch

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

    if model.policy_setup == "google_robot":
        current = raw_action["open_gripper"]
        if model.previous_gripper_action is None:
            relative = np.array([0])
        else:
            relative = model.previous_gripper_action - current
        model.previous_gripper_action = current

        if np.abs(relative) > 0.5 and not model.sticky_action_is_on:
            model.sticky_action_is_on = True
            model.sticky_gripper_action = relative

        if model.sticky_action_is_on:
            model.gripper_action_repeat += 1
            relative = model.sticky_gripper_action

        if model.gripper_action_repeat == model.sticky_gripper_num_repeat:
            model.sticky_action_is_on = False
            model.gripper_action_repeat = 0
            model.sticky_gripper_action = 0.0

        action["gripper"] = relative
    else:  # widowx_bridge
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

def select_camera_pair(obs: dict, policy_camera: str) -> Tuple[str, str]:
    available = list(obs["image"].keys())
    if FLAGS.camera_pair:
        names = [name.strip() for name in FLAGS.camera_pair.split(",") if name.strip()]
        if len(names) != 2:
            raise ValueError("camera_pair must contain exactly two camera names")
        for name in names:
            if name not in available:
                raise ValueError(f"Camera '{name}' not found in observation keys: {available}")
        return names[0], names[1]

    if policy_camera not in available:
        raise ValueError(f"Policy camera '{policy_camera}' not found. Available: {available}")

    secondary = None
    if policy_camera != "overhead_camera" and "overhead_camera" in available:
        secondary = "overhead_camera"
    preferred = ["3rd_view_camera", "2nd_view_camera", "hand_camera", "wrist_camera"]
    for candidate in preferred:
        if candidate != policy_camera and candidate in available:
            secondary = candidate
            break
    if secondary is None:
        for candidate in available:
            if candidate != policy_camera:
                secondary = candidate
                break
    if secondary is None:
        raise ValueError("Need at least two cameras in the observation to enable dual view mode.")
    return policy_camera, secondary

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
    if tf_visible:
        logging.info("Disabled TF GPU visibility for lighter-weight inference.")

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

    try:
        for episode_idx in range(FLAGS.num_eval_episodes):
            obs, _ = env.reset()
            instruction = env.get_language_instruction()
            model.reset(instruction)
            logging.info("Episode %d instruction: %s", episode_idx, instruction)

            camera_primary, camera_secondary = select_camera_pair(obs, policy_camera)
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
                policy_image = get_image_from_maniskill2_obs_dict(env, obs, camera_primary)
                ee_pos_world, ee_rot_world = get_ee_pose_world(env)

                batch = sample_action_batch(model, policy_image, FLAGS.num_samples)

                annotated_frames = []
                titles = []
                for cam_name in [camera_primary, camera_secondary]:
                    rgb = obs["image"][cam_name]["rgb"]
                    depth = obs["image"][cam_name].get("depth")
                    camera_intrinsic = obs["camera_param"][cam_name]["intrinsic_cv"]
                    camera_extrinsic = obs["camera_param"][cam_name]["extrinsic_cv"]
                    annotated = annotate_actions_on_view(
                        rgb,
                        batch["action_deltas"],
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
