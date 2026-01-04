import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
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
from flax.training import checkpoints
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import cv2
from PIL import Image, ImageDraw, ImageFont
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_integer("num_eval_episodes", 20, "Number of evaluation episodes.")
flags.DEFINE_string("vgps_checkpoint", "", "vgps checkpoint name.")
flags.DEFINE_string("vgps_wandb", "", "vgps wandb run name.")
flags.DEFINE_integer("num_samples", 50, "Number of action samples.")
flags.DEFINE_float("action_temp", 1.0, "action softmax temperature. The beta value in the paper.")
flags.DEFINE_integer("max_timesteps", 20, "Maximum timesteps to visualize per episode.")
flags.DEFINE_boolean("show_rotation", False, "Whether to visualize rotation arrows.")
flags.DEFINE_boolean("show_gripper", True, "Whether to show gripper state with colors.")

def load_vgps_checkpoint(path, wandb_run_name):
    # check path
    assert os.path.exists(path), f"Checkpoint path {path} does not exist"

    """
    You can either specify wandb_run_name to load the exact configuration from Weights & Biases or use the pretrained_checkpoint.yaml file if you are using the provided pre-trained checkpoints.
    """

    if wandb_run_name == "":
        # load from experiments/configs/pretrained_checkpoints.yaml
        import yaml
        with open("experiments/configs/pretrained_checkpoint.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        # load information from wandb
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    # create encoder from wandb config
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
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
    agent = agents[config["agent"]].create(
            rng=jax.random.PRNGKey(0),
            encoder_def=encoder_def,
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

    return get_values, critic_text_processor


def project_3d_to_2d(points_3d, camera_intrinsic, camera_extrinsic):
    """
    Project 3D points in world coordinates to 2D pixel coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        camera_intrinsic: (3, 3) camera intrinsic matrix
        camera_extrinsic: (4, 4) camera extrinsic matrix (world to camera)
    
    Returns:
        pixels: (N, 2) array of pixel coordinates (u, v)
        depths: (N,) array of depths in camera frame
    """
    N = points_3d.shape[0]
    
    # Convert to homogeneous coordinates
    points_homogeneous = np.concatenate([points_3d, np.ones((N, 1))], axis=1)  # (N, 4)
    
    # Transform to camera coordinates
    points_camera = (camera_extrinsic @ points_homogeneous.T).T  # (N, 4)
    points_camera_3d = points_camera[:, :3]  # (N, 3)
    
    # Get depths (z coordinate in camera frame)
    depths = points_camera_3d[:, 2]
    
    # Project to image plane
    points_2d_homogeneous = (camera_intrinsic @ points_camera_3d.T).T  # (N, 3)
    
    # Normalize by depth to get pixel coordinates
    pixels = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # (N, 2)
    
    return pixels, depths


def check_occlusion(pixel_coords, predicted_depths, depth_image, threshold=0.02):
    """
    Check if projected points are occluded using depth buffer.
    
    Args:
        pixel_coords: (N, 2) array of pixel coordinates (u, v)
        predicted_depths: (N,) array of predicted depths
        depth_image: (H, W) or (H, W, 1) depth image
        threshold: depth threshold in meters for occlusion check
    
    Returns:
        is_visible: (N,) boolean array, True if point is visible
    """
    N = pixel_coords.shape[0]
    H, W = depth_image.shape[:2]
    
    # Squeeze depth image if needed
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]
    
    is_visible = np.zeros(N, dtype=bool)
    
    for i in range(N):
        u, v = pixel_coords[i]
        
        # Check if pixel is within image bounds
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
        
        # Get depth at pixel
        u_int, v_int = int(round(u)), int(round(v))
        if u_int < 0 or u_int >= W or v_int < 0 or v_int >= H:
            continue
            
        observed_depth = depth_image[v_int, u_int]
        
        # Point is visible if its depth is close to or less than observed depth
        if predicted_depths[i] <= observed_depth + threshold:
            is_visible[i] = True
    
    return is_visible


def overlay_action_proposals_on_frame(rgb_image, action_deltas, q_values, gripper_states, 
                                      chosen_idx, ee_pos_world, ee_rot_world, camera_intrinsic, 
                                      camera_extrinsic, depth_image, timestep):
    """
    Overlay action proposal markers on the RGB frame.
    
    Args:
        rgb_image: (H, W, 3) RGB image
        action_deltas: (N, 3) action position deltas in end-effector frame
        q_values: (N,) Q-values for each action
        gripper_states: (N,) gripper states (0-1)
        chosen_idx: index of chosen action
        ee_pos_world: (3,) current EE position in world coordinates
        ee_rot_world: (3, 3) current EE rotation matrix in world coordinates
        camera_intrinsic: (3, 3) camera intrinsic matrix
        camera_extrinsic: (4, 4) camera extrinsic matrix
        depth_image: (H, W, 1) depth image
        timestep: current timestep number
    
    Returns:
        annotated_image: RGB image with overlays
    """
    # Create a copy to draw on
    annotated = rgb_image.copy()
    H, W = annotated.shape[:2]
    
    # Scale all action deltas by 3x for better visibility
    scaled_action_deltas = action_deltas * 3.0
    
    # Transform action deltas from EE frame to world frame
    # Action deltas are in EE frame, need to rotate them to world frame
    action_deltas_world = (ee_rot_world @ scaled_action_deltas.T).T  # (N, 3)
    
    # Compute predicted EE positions in world frame (using scaled deltas in world frame)
    predicted_positions = ee_pos_world[None, :] + action_deltas_world  # (N, 3)
    
    # Project to pixel coordinates
    pixels, depths = project_3d_to_2d(predicted_positions, camera_intrinsic, camera_extrinsic)
    
    # Check occlusion
    is_visible = check_occlusion(pixels, depths, depth_image, threshold=0.05)
    
    # Normalize Q-values for color mapping (per sample batch)
    q_values = np.asarray(q_values)
    q_range = q_values.max() - q_values.min()
    q_norm = (q_values - q_values.min()) / (q_range + 1e-8)
    cmap = cm.get_cmap('rainbow')
    low_q_cmap_offset = 0.12  # pull minimum away from purple into dark blue

    def map_to_cmap(val: float) -> float:
        return float(np.clip(low_q_cmap_offset + (1.0 - low_q_cmap_offset) * val, 0.0, 1.0))
    
    # Use PIL for better drawing quality
    pil_image = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    try:
        stats_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        legend_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        stats_font = legend_font = ImageFont.load_default()
    
    # Legend placement parameters (right-hand side strip)
    legend_width = 150
    legend_margin = 15
    legend_height = 150
    legend_x0 = max(W - legend_width - legend_margin, legend_margin)
    legend_y0 = legend_margin
    legend_x1 = W - legend_margin
    legend_y1 = legend_y0 + legend_height

    # Draw markers for each proposal
    for i in range(len(action_deltas)):
        u, v = pixels[i]
        
        # Skip if out of bounds
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
        
        # Determine color based on Q-value (full rainbow scale per sample batch)
        q_val = q_norm[i]
        rgb = np.array(cmap(map_to_cmap(q_val))[:3]) * 255
        color = rgb.astype(int)

        # Determine gripper outline color (white when gripping/closed, black when open)
        is_gripping = gripper_states[i] < 0.5
        outline_color = (255, 255, 255, 255) if is_gripping else (0, 0, 0, 255)
        
        # Draw marker (3 px radius for chosen, 2 px otherwise) with slightly smaller outline
        radius = 3 if i == chosen_idx else 2
        radius_offset = 1 if i == chosen_idx else 0
        bbox = [u - radius - radius_offset, v - radius - radius_offset, u + radius + radius_offset, v + radius + radius_offset]
        draw.ellipse(bbox, fill=tuple(list(color) + [255]))
        draw.ellipse(bbox, outline=outline_color, width=1)
    
    # Draw arrow from current EE to chosen action
    current_ee_pixels, _ = project_3d_to_2d(ee_pos_world[None, :], camera_intrinsic, camera_extrinsic)
    chosen_pixels = pixels[chosen_idx]
    
    if (0 <= current_ee_pixels[0, 0] < W and 0 <= current_ee_pixels[0, 1] < H and
        0 <= chosen_pixels[0] < W and 0 <= chosen_pixels[1] < H):
        # Arrow already uses scaled positions (deltas were scaled 3x above)
        dx = chosen_pixels[0] - current_ee_pixels[0, 0]
        dy = chosen_pixels[1] - current_ee_pixels[0, 1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length
            arrow_shorten_px = 6.0
            shorten = min(arrow_shorten_px, length * 0.8)
            line_end = current_ee_pixels[0] + (length - shorten) * np.array([dx, dy])
            draw.line([tuple(current_ee_pixels[0]), tuple(line_end)], 
                     fill=(255, 0, 0, 200), width=2)
            # Perpendicular direction
            px, py = -dy, dx
            arrow_tip = chosen_pixels
            arrow_size = 9
            arrow_left = arrow_tip - arrow_size * np.array([dx, dy]) + (arrow_size / 2) * np.array([px, py])
            arrow_right = arrow_tip - arrow_size * np.array([dx, dy]) - (arrow_size / 2) * np.array([px, py])
            draw.polygon([tuple(arrow_tip), tuple(arrow_left), tuple(arrow_right)], 
                        fill=(255, 0, 0, 200))

    # Draw legend on the right-hand side
    legend_bg = [legend_x0, legend_y0, legend_x1, legend_y1]
    draw.rectangle(legend_bg, fill=(0, 0, 0, 180), outline=(255, 255, 255, 220), width=2)

    gradient_padding = 12
    gradient_height = legend_height - 2 * gradient_padding - 50
    gradient_height = max(gradient_height, 20)
    grad_x0 = legend_x0 + gradient_padding
    grad_x1 = grad_x0 + 25
    grad_y0 = legend_y0 + gradient_padding
    grad_y1 = grad_y0 + gradient_height

    for y in range(gradient_height):
        t = 1.0 - y / max(gradient_height - 1, 1)
        grad_color = tuple(int(c * 255) for c in cmap(map_to_cmap(t))[:3]) + (255,)
        draw.rectangle([grad_x0, grad_y0 + y, grad_x1, grad_y0 + y + 1], fill=grad_color)

    high_label = "High Q"
    low_label = "Low Q"
    text_x = grad_x1 + 8
    high_bbox = draw.textbbox((0, 0), high_label, font=legend_font)
    high_height = high_bbox[3] - high_bbox[1]
    low_bbox = draw.textbbox((0, 0), low_label, font=legend_font)
    low_height = low_bbox[3] - low_bbox[1]
    draw.text((text_x, grad_y0), high_label, fill=(255, 255, 255, 255), font=legend_font)
    draw.text((text_x, grad_y1 - low_height), low_label, fill=(255, 255, 255, 255), font=legend_font)

    outline_y = grad_y1 + 15
    sample_radius = 6
    # Closed/gripping sample (white outline)
    sample_bbox = [grad_x0, outline_y, grad_x0 + 2 * sample_radius, outline_y + 2 * sample_radius]
    draw.ellipse(sample_bbox, fill=(200, 200, 200, 255))
    draw.ellipse(sample_bbox, outline=(255, 255, 255, 255), width=1)
    draw.text((grad_x0 + 2 * sample_radius + 6, outline_y - 2), "Closed gripper", fill=(255, 255, 255, 255), font=legend_font)

    # Open gripper sample (black outline)
    outline_y += sample_radius * 2 + 8
    sample_bbox = [grad_x0, outline_y, grad_x0 + 2 * sample_radius, outline_y + 2 * sample_radius]
    draw.ellipse(sample_bbox, fill=(200, 200, 200, 255))
    draw.ellipse(sample_bbox, outline=(0, 0, 0, 255), width=1)
    draw.text((grad_x0 + 2 * sample_radius + 6, outline_y - 2), "Open gripper", fill=(255, 255, 255, 255), font=legend_font)
    
    # Add text overlay with statistics
    stats_text = f"t={timestep} | Q: [{q_values.min():.1f}, {q_values.max():.1f}] | Chosen: {q_values[chosen_idx]:.1f}"
    # Draw text with background
    bbox = draw.textbbox((10, 10), stats_text, font=stats_font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text((10, 10), stats_text, fill=(255, 255, 255, 255), font=stats_font)
    
    # Convert back to numpy
    annotated = np.array(pil_image)
    
    return annotated


def get_ee_position_from_env(env):
    """
    Get the current end-effector position from the environment.
    Returns xyz position in the robot base frame.
    """
    # Access the robot's TCP (tool center point) pose
    tcp_pose = env.tcp.pose
    robot_pose = env.agent.robot.pose
    
    # Get TCP position relative to robot base
    tcp_pose_at_base = robot_pose.inv() * tcp_pose
    ee_pos = tcp_pose_at_base.p  # xyz position
    
    return ee_pos


def get_ee_pose_world(env):
    """
    Get the current end-effector pose in world coordinates.
    Returns position and rotation matrix.
    """
    tcp_pose = env.tcp.pose
    return tcp_pose.p, tcp_pose.to_transformation_matrix()[:3, :3]


def get_ee_position_world(env):
    """
    Get the current end-effector position in world coordinates.
    """
    tcp_pose = env.tcp.pose
    return tcp_pose.p


def visualize_action_samples_3d(ee_pos, action_deltas, q_values, gripper_states, chosen_idx, timestep, save_path):
    """
    Create a 3D visualization of the sampled actions.
    
    Args:
        ee_pos: Current end-effector position (3,)
        action_deltas: Sampled position deltas (num_samples, 3)
        q_values: Q-values for each action (num_samples,)
        gripper_states: Gripper values for each action (num_samples,)
        chosen_idx: Index of the chosen action
        timestep: Current timestep number
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Normalize Q-values for color mapping
    q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)
    
    # Create colormap
    cmap = cm.get_cmap('RdYlGn')  # Red (low Q) -> Yellow -> Green (high Q)
    
    # Compute predicted positions
    predicted_positions = ee_pos[None, :] + action_deltas  # (num_samples, 3)
    
    # --- Plot 1: 3D Scatter ---
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot current EE position
    ax1.scatter(*ee_pos, c='blue', s=200, marker='o', label='Current EE', edgecolors='black', linewidths=2)
    
    # Plot predicted positions with Q-value coloring
    scatter = ax1.scatter(
        predicted_positions[:, 0],
        predicted_positions[:, 1],
        predicted_positions[:, 2],
        c=q_norm,
        cmap=cmap,
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Highlight the chosen action
    chosen_pos = predicted_positions[chosen_idx]
    ax1.scatter(*chosen_pos, c='red', s=300, marker='*', 
                label='Chosen Action', edgecolors='black', linewidths=2)
    
    # Draw arrow from current to chosen position
    ax1.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
               action_deltas[chosen_idx, 0], action_deltas[chosen_idx, 1], action_deltas[chosen_idx, 2],
               color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Action Proposals (t={timestep})')
    ax1.legend()
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
    cbar1.set_label('Normalized Q-value')
    
    # --- Plot 2: XY Projection ---
    ax2 = fig.add_subplot(132)
    
    # Plot current EE
    ax2.scatter(ee_pos[0], ee_pos[1], c='blue', s=200, marker='o', 
                label='Current EE', edgecolors='black', linewidths=2, zorder=3)
    
    # Plot predicted positions
    if FLAGS.show_gripper:
        # Color by gripper state
        gripper_colors = ['green' if g > 0.5 else 'purple' for g in gripper_states]
        for i, (pos, color) in enumerate(zip(predicted_positions, gripper_colors)):
            alpha = 0.3 + 0.7 * q_norm[i]  # Vary alpha by Q-value
            ax2.scatter(pos[0], pos[1], c=color, s=50, alpha=alpha, 
                       edgecolors='black', linewidths=0.5, zorder=2)
    else:
        scatter2 = ax2.scatter(predicted_positions[:, 0], predicted_positions[:, 1],
                              c=q_norm, cmap=cmap, s=50, alpha=0.6,
                              edgecolors='black', linewidths=0.5, zorder=2)
        plt.colorbar(scatter2, ax=ax2, label='Normalized Q-value')
    
    # Highlight chosen action
    ax2.scatter(chosen_pos[0], chosen_pos[1], c='red', s=300, marker='*',
                label='Chosen Action', edgecolors='black', linewidths=2, zorder=4)
    
    # Draw arrow
    ax2.arrow(ee_pos[0], ee_pos[1], 
              action_deltas[chosen_idx, 0], action_deltas[chosen_idx, 1],
              head_width=0.01, head_length=0.01, fc='red', ec='red', 
              linewidth=2, alpha=0.7, zorder=3)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'XY Projection (t={timestep})')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    if FLAGS.show_gripper:
        # Add legend for gripper states
        open_patch = mpatches.Patch(color='green', label='Gripper Open')
        closed_patch = mpatches.Patch(color='purple', label='Gripper Closed')
        ax2.legend(handles=[open_patch, closed_patch], loc='upper right')
    else:
        ax2.legend()
    
    # --- Plot 3: Q-value Distribution ---
    ax3 = fig.add_subplot(133)
    
    # Sort by Q-value for better visualization
    sorted_indices = np.argsort(q_values)
    sorted_q_values = q_values[sorted_indices]
    
    # Plot all Q-values
    ax3.bar(range(len(sorted_q_values)), sorted_q_values, color='lightblue', edgecolor='black')
    
    # Highlight the chosen action
    chosen_rank = np.where(sorted_indices == chosen_idx)[0][0]
    ax3.bar(chosen_rank, sorted_q_values[chosen_rank], color='red', edgecolor='black', linewidth=2)
    
    ax3.set_xlabel('Action (sorted by Q-value)')
    ax3.set_ylabel('Q-value')
    ax3.set_title(f'Q-value Distribution (t={timestep})')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Max Q: {q_values.max():.2f}\nMin Q: {q_values.min():.2f}\n"
    stats_text += f"Mean Q: {q_values.mean():.2f}\nChosen Q: {q_values[chosen_idx]:.2f}\n"
    stats_text += f"Chosen Rank: {chosen_rank + 1}/{len(q_values)}"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(_):
    logging.set_verbosity(logging.ERROR)
    tf.config.set_visible_devices([], 'GPU')
    print(FLAGS.flag_values_dict())
    
    if 'env' in locals():
        print("Closing existing env")
        env.close()
        del env
    
    env = simpler_env.make(FLAGS.task_name)
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print("Reset info", reset_info)
    print("Instruction", instruction)

    if "google" in FLAGS.task_name:
        policy_setup = "google_robot"
        STICKY_GRIPPER_NUM_STEPS = 15
    else:
        policy_setup = "widowx_bridge"
        STICKY_GRIPPER_NUM_STEPS = 3

    # @title Select your model and environment
    tf.config.set_visible_devices([], 'GPU')
    model = OctoInference(model_type=FLAGS.model_name, policy_setup=policy_setup, 
                         init_rng=FLAGS.seed, sticky_step=STICKY_GRIPPER_NUM_STEPS)
    
    assert FLAGS.vgps_checkpoint != ""
    get_values, critic_text_processor = load_vgps_checkpoint(FLAGS.vgps_checkpoint, FLAGS.vgps_wandb)
    model.init_vgps(FLAGS.num_samples, get_values, critic_text_processor, 
                   action_temp=FLAGS.action_temp, max_episode_steps=env._max_episode_steps)
 
    # Create output directory with timestamp to avoid overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"logs/visualizations/{FLAGS.model_name}_VGPS"
    vis_folder = os.path.join(base_folder, f"seed_{FLAGS.seed}/{FLAGS.task_name}/{timestamp}")
    os.makedirs(vis_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting V-GPS Action Visualization")
    print(f"Task: {FLAGS.task_name}")
    print(f"Number of action samples: {FLAGS.num_samples}")
    print(f"Action temperature: {FLAGS.action_temp}")
    print(f"Output directory: {vis_folder}")
    print(f"{'='*60}\n")

    #@title Run inference with visualization
    successes = []
    
    for episode_idx in range(FLAGS.num_eval_episodes):
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        model.reset(instruction)
        print(f"\nEpisode {episode_idx}: {instruction}")

        image = get_image_from_maniskill2_obs_dict(env, obs)
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        
        # Create episode-specific directory
        episode_folder = os.path.join(vis_folder, f"episode_{episode_idx}")
        os.makedirs(episode_folder, exist_ok=True)
        
        # Determine camera name
        if "google" in FLAGS.task_name:
            camera_name = "overhead_camera"
        else:
            camera_name = "3rd_view_camera"
        
        # Storage for annotated frames
        annotated_images = []
        
        while not (success or truncated) and timestep < FLAGS.max_timesteps:
            # Get current end-effector position (for 3D plots)
            ee_pos = get_ee_position_from_env(env)
            
            # Get EE pose in world coordinates (for projection)
            ee_pos_world, ee_rot_world = get_ee_pose_world(env)
            
            # Get camera parameters from observation
            camera_intrinsic = obs["camera_param"][camera_name]["intrinsic_cv"]
            camera_extrinsic = obs["camera_param"][camera_name]["extrinsic_cv"]
            depth_image = obs["image"][camera_name]["depth"]
            
            # Get all sampled actions and their Q-values
            raw_action, action, action_samples_info = model.step_with_samples(image)
            
            # Extract information
            action_deltas = action_samples_info['action_deltas']  # (num_samples, 3)
            q_values = action_samples_info['q_values']  # (num_samples,)
            gripper_states = action_samples_info['gripper_states']  # (num_samples,)
            chosen_idx = action_samples_info['chosen_idx']  # scalar
            
            # Create 3D visualization (existing plots)
            vis_path_3d = os.path.join(episode_folder, f"timestep_{timestep:03d}_3d.png")
            visualize_action_samples_3d(ee_pos, action_deltas, q_values, 
                                       gripper_states, chosen_idx, timestep, vis_path_3d)
            
            # Create frame overlay visualization (NEW!)
            annotated_frame = overlay_action_proposals_on_frame(
                image, action_deltas, q_values, gripper_states, chosen_idx,
                ee_pos_world, ee_rot_world, camera_intrinsic, camera_extrinsic, depth_image, timestep
            )
            annotated_images.append(annotated_frame)
            
            # Save annotated frame
            frame_path = os.path.join(episode_folder, f"timestep_{timestep:03d}_overlay.png")
            imageio.imwrite(frame_path, annotated_frame)
            
            print(f"  t={timestep}: Q-values [{q_values.min():.2f}, {q_values.max():.2f}], "
                  f"Chosen Q={q_values[chosen_idx]:.2f}")
            # print(f"    Saved 3D plot: {vis_path_3d}")
            # print(f"    Saved overlay: {frame_path}")
            
            # Execute the chosen action
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            
            # Update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        # Rename episode folder to include success status
        success_status = "SUCCESS" if success else "FAILURE"
        new_episode_folder = os.path.join(vis_folder, f"episode_{episode_idx}_{success_status}")
        os.rename(episode_folder, new_episode_folder)
        episode_folder = new_episode_folder
        
        # Save the original episode video
        video_path = os.path.join(episode_folder, f"episode_{episode_idx}_original.mp4")
        imageio.mimsave(video_path, images, fps=10)
        
        # Save the annotated episode video
        video_path_annotated = os.path.join(episode_folder, f"episode_{episode_idx}_annotated.mp4")
        imageio.mimsave(video_path_annotated, annotated_images, fps=10)
        
        successes.append(1 if success else 0)
        print(f"\nEpisode {episode_idx} completed: {success_status}, timesteps={timestep}")
        print(f"Original video: {video_path}")
        print(f"Annotated video: {video_path_annotated}")

    # Save summary
    summary_path = os.path.join(vis_folder, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"V-GPS Action Visualization Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {FLAGS.model_name}\n")
        f.write(f"Task: {FLAGS.task_name}\n")
        f.write(f"Seed: {FLAGS.seed}\n")
        f.write(f"Num samples: {FLAGS.num_samples}\n")
        f.write(f"Action temp: {FLAGS.action_temp}\n")
        f.write(f"Episodes: {FLAGS.num_eval_episodes}\n")
        f.write(f"Success rate: {sum(successes)}/{len(successes)} = {sum(successes)/len(successes):.2%}\n")
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Success rate: {sum(successes)}/{len(successes)} = {sum(successes)/len(successes):.2%}")
    print(f"Results saved to: {vis_folder}")
    print(f"{'='*60}\n")

        
if __name__ == "__main__":
    app.run(main)
