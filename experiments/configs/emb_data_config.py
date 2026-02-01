from copy import deepcopy
import imp
import os
from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from octo.data.utils.data_utils import NormalizationType

def get_dataset_config(window_size=1):
    task_augmentation = dict(
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.5,
        ),
    )

    return {
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(str),
            data_dir=placeholder(str),
            load_camera_views=("primary", "wrist"),
            load_depth=False,
        ),
        "traj_transform_kwargs": dict(
            window_size=window_size,
            action_horizon=1,
            goal_relabeling_strategy="uniform",
            subsample_length=100,
            **task_augmentation,
        ),
        "frame_transform_kwargs": dict(),
        "traj_transform_threads": 48,  # shared between all datasets
        "traj_read_threads": 48,  # shared between all datasets
        "shuffle_buffer_size": 100000,  # shared between all datasets
        "batch_size": 1024,
        "balance_weights": True,
    }


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config():

    config = get_dataset_config(window_size=1)
    action_dim = FieldReference(7)


    config = update_config(
        config,
        oxe_kwargs=dict(
            data_dir=placeholder(str),
            data_mix="bridge_fractal_embedding",
            load_camera_views=("primary", ),
            load_depth=False,
            force_recompute_dataset_statistics=False,
            discount=0.98,
            num_final_repeat=3,
            action_proprio_normalization_type=NormalizationType.BOUNDS, # we normalize actions to [-1, 1] with min max bounds,
        ),
        traj_transform_kwargs=dict(
            action_horizon=1,
            max_action_dim=action_dim,
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=0.0,
            ),
            goal_relabeling_strategy=None,
        ),
        frame_transform_kwargs=dict(),
        batch_size=4096,
        shuffle_buffer_size=50000,
        balance_weights=True,
    )

    return ConfigDict(config)