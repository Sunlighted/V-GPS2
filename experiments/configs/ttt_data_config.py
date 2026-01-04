"""
Data configuration for TTT trajectory-level training.
Separate from the original data_config.py for clean separation.
"""

from copy import deepcopy
from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from octo.data.utils.data_utils import NormalizationType


def get_config():
    """
    Default TTT data configuration.
    Configured for trajectory-level training (batch_size=1, window_size=120).
    """
    action_dim = FieldReference(7)
    
    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    
    config = ConfigDict({
        # OXE dataset configuration
        "oxe_kwargs": dict(
            data_dir=placeholder(str),
            data_mix="fractal",  # Default dataset
            load_camera_views=("primary",),
            load_depth=False,
            force_recompute_dataset_statistics=False,
            discount=0.98,
            num_final_repeat=3,
            action_proprio_normalization_type=NormalizationType.BOUNDS,
        ),
        
        # Trajectory transform settings
        "traj_transform_kwargs": dict(
            window_size=120,  # TRAJECTORY LENGTH for TTT
            action_horizon=1,
            max_action_dim=action_dim,
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=0.0,
            ),
            goal_relabeling_strategy="uniform",
            subsample_length=None,  # Use full episodes
        ),
        
        # Frame transform settings
        "frame_transform_kwargs": dict(
            resize_size={"primary": (256, 256)},
            image_dropout_prob=0.0,
            image_augment_kwargs={"primary": primary_augment_kwargs},
            num_parallel_calls=200,
        ),
        
        # Dataset threading and batching
        "traj_transform_threads": 48,
        "traj_read_threads": 48,
        # Smaller buffer keeps RAM usage manageable for 120-frame trajectories
        "shuffle_buffer_size": 1024,
        "batch_size": 1,  # Default: 1 trajectory per batch (override via --oxedata_config.batch_size)
        "balance_weights": True,
    })

    config["sim_data"] = ConfigDict(
        dict(
            enable=False,
            tfrecord_dir="",
            sample_weight=1.0,
            num_parallel_calls=200,
            num_parallel_reads=8,
        )
    )
    
    return config