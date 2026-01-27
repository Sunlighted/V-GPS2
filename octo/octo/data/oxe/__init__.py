import copy
import logging
from typing import Any, Dict, List, Sequence, Tuple, Union

from octo.data.oxe.oxe_dataset_configs import ActionEncoding, OXE_DATASET_CONFIGS
from octo.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from octo.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from octo.data.oxe.oxe_dataset_configs import (
    OXE_EMBEDDING_CONFIGS,
    OXE_EMBEDDING_TRANSFORMS,
)
from octo.data.utils.data_utils import NormalizationType
from octo.utils.spec import ModuleSpec


def make_oxe_dataset_kwargs(
    name: str,
    data_dir: str,
    load_camera_views: Sequence[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = False,
    load_language: bool = True,
    force_recompute_dataset_statistics: bool = False,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
    discount=0.98,
    num_final_repeat=3,
) -> Dict[str, Any]:
    """Generates dataset kwargs for a given dataset from Open X-Embodiment. The returned kwargs can be passed
    directly into `octo.data.dataset.make_dataset_from_rlds`.

    Args:
        name: Name of the dataset to load. See `oxe_dataset_configs.py` for available datasets.
        data_dir: Base data directory that contains the dataset.
        load_camera_views: Which views to load. See `oxe_dataset_configs.py` for available views.
        load_depth: If True, loads corresponding depth channels for each RGB channel.
        load_proprio: If True, loads proprioceptive information.
        load_language: If True, loads language instructions.
        force_recompute_dataset_statistics: If True, recompute dataset statistics.
        action_proprio_normalization_type: Normalization type to use for proprioceptive actions.
    """
    dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[name])

    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        # with EEF_POS actions, the last action dimension is gripper
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS:
        # with JOINT_POS actions, last dimension is gripper
        dataset_kwargs["action_normalization_mask"] = [True] * 7 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL:
        # with JOINT_POS_BIMANUAL actions, 7th and 14th dimension are gripper
        dataset_kwargs["action_normalization_mask"] = (
            [True] * 6 + [False] + [True] * 6 + [False]
        )
    elif dataset_kwargs["action_encoding"] is ActionEncoding.NAV_2D:
        # with NAV_2D actions, all dimensions are deltas
        dataset_kwargs["action_normalization_mask"] = [True] * 2
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL_NAV:
        # with JOINT_POS_BIMANUAL_NAV actions, 7th and 14th dimension are gripper
        dataset_kwargs["action_normalization_mask"] = (
            [True] * 6 + [False] + [True] * 6 + [False] + [True] * 2
        )
    else:
        raise ValueError(
            f"Cannot load {name} with unsupported action encoding {dataset_kwargs['action_encoding']}."
        )

    # adjust loaded camera views
    if missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"])):
        raise ValueError(
            f"Cannot load {name} with views {missing_keys} since they are not available."
        )
    dataset_kwargs["image_obs_keys"] = {
        k: v
        for k, v in dataset_kwargs["image_obs_keys"].items()
        if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v
        for k, v in dataset_kwargs["depth_obs_keys"].items()
        if k in load_camera_views
    }

    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if load_proprio:
        dataset_kwargs["proprio_obs_key"] = "proprio"
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    dataset_kwargs[
        "action_proprio_normalization_type"
    ] = action_proprio_normalization_type

    del dataset_kwargs["proprio_encoding"]
    del dataset_kwargs["action_encoding"]

    dataset_kwargs["standardize_fn"] = ModuleSpec.create(
        OXE_STANDARDIZATION_TRANSFORMS[name]
    )

    if force_recompute_dataset_statistics:
        dataset_kwargs["force_recompute_dataset_statistics"] = True

    dataset_kwargs["discount"] = discount
    dataset_kwargs["num_final_repeat"] = num_final_repeat

    return {"name": name, "data_dir": data_dir, **dataset_kwargs}


def make_oxe_dataset_kwargs_and_weights(
    data_mix: Union[str, Sequence[Tuple[str, float]]],
    data_dir: str,
    load_camera_views: Sequence[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = False,
    load_language: bool = True,
    force_recompute_dataset_statistics: bool = False,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    discount=0.98,
    num_final_repeat=3,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    and weights can be passed directly into `octo.data.dataset.make_interleaved_dataset`.

    Args:
        data_mix: List of (dataset name, sampling weight) tuples, or a string specifying a pre-defined mix to
            load from `OXE_NAMED_MIXES`.
        data_dir: Base data directory that contains the datasets.
        load_camera_views: Which views to load. See `oxe_dataset_configs.py` for available views.
        load_depth: If True, loads corresponding depth channels for each RGB channel.
        load_proprio: If True, loads proprioceptive information.
        load_language: If True, loads language instructions.
        force_recompute_dataset_statistics: If True, recompute dataset statistics.
        action_proprio_normalization_type: Normalization type to use for proprioceptive actions.
    Returns:
        Tuple of (dataset_kwargs_list, sampling weights).
    """
    if isinstance(data_mix, str):
        data_mix = OXE_NAMED_MIXES[data_mix]

    filtered_datasets, included_dataset_names = [], []
    for name, weight in data_mix:
        if name not in included_dataset_names:
            filtered_datasets.append((name, weight))
            included_dataset_names.append(name)
        else:
            logging.warning(f"Skipping duplicate: {(name, weight)}.")
    data_mix = filtered_datasets

    data_kwargs_list, weights = [], []
    for name, weight in data_mix:
        try:
            data_kwargs_list.append(
                make_oxe_dataset_kwargs(
                    name,
                    data_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    force_recompute_dataset_statistics,
                    action_proprio_normalization_type,
                    discount,
                    num_final_repeat,
                )
            )
            weights.append(weight)
        except ValueError as e:
            logging.warning(f"Skipping {name} due to error: {e}")

    return data_kwargs_list, weights


def make_embedding_dataset_kwargs(
    name: str,
    data_dir: str,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
) -> Dict[str, Any]:
    """Generates dataset kwargs for an embedding dataset.

    This function is analogous to make_oxe_dataset_kwargs but for pre-computed
    embedding datasets created by Octo_embedding_save.py.

    Args:
        name: Name of the embedding dataset (e.g., 'bridge_dataset_embedding').
        data_dir: Base data directory containing the embedding datasets.
        action_proprio_normalization_type: Normalization type for actions.

    Returns:
        Dictionary of kwargs for make_embedding_dataset.
    """
    if name not in OXE_EMBEDDING_CONFIGS:
        raise ValueError(
            f"Unknown embedding dataset: {name}. "
            f"Available: {list(OXE_EMBEDDING_CONFIGS.keys())}"
        )

    config = OXE_EMBEDDING_CONFIGS[name]

    # For embedding datasets, normalize all action dimensions except gripper
    # (gripper is last dimension, following OXE convention)
    action_normalization_mask = [True] * (config["action_dim"] - 1) + [False]

    return {
        "name": name,
        "data_dir": data_dir,
        "action_proprio_normalization_type": action_proprio_normalization_type,
        "action_normalization_mask": action_normalization_mask,
    }


def make_embedding_dataset_kwargs_and_weights(
    data_mix: Union[str, Sequence[Tuple[str, float]]],
    data_dir: str,
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Generates dataset kwargs and weights for embedding dataset mixes.

    This function is analogous to make_oxe_dataset_kwargs_and_weights but for
    pre-computed embedding datasets.

    Args:
        data_mix: List of (dataset name, sampling weight) tuples, or a string
            specifying a pre-defined mix (e.g., 'bridge_embedding').
        data_dir: Base data directory containing the embedding datasets.
        action_proprio_normalization_type: Normalization type for actions.

    Returns:
        Tuple of (dataset_kwargs_list, sampling weights).
    """
    if isinstance(data_mix, str):
        if data_mix not in OXE_NAMED_MIXES:
            raise ValueError(
                f"Unknown data mix: {data_mix}. "
                f"Available embedding mixes: bridge_embedding, fractal_embedding, "
                f"bridge_fractal_embedding"
            )
        data_mix = OXE_NAMED_MIXES[data_mix]

    # Filter duplicates
    filtered_datasets, included_dataset_names = [], []
    for name, weight in data_mix:
        if name not in included_dataset_names:
            filtered_datasets.append((name, weight))
            included_dataset_names.append(name)
        else:
            logging.warning(f"Skipping duplicate: {(name, weight)}.")
    data_mix = filtered_datasets

    data_kwargs_list, weights = [], []
    for name, weight in data_mix:
        try:
            data_kwargs_list.append(
                make_embedding_dataset_kwargs(
                    name,
                    data_dir,
                    action_proprio_normalization_type,
                )
            )
            weights.append(weight)
        except ValueError as e:
            logging.warning(f"Skipping {name} due to error: {e}")

    return data_kwargs_list, weights
