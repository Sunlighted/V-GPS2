"""Configuration for pre-computed embedding datasets.

This config is used with train_from_embedding.py for training with
pre-computed Octo embeddings instead of raw images.
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config():
    config = ConfigDict()

    # Dataset name (must be in OXE_EMBEDDING_CONFIGS)
    config.dataset_name = placeholder(str)

    # Base data directory containing the embedding datasets
    config.data_dir = placeholder(str)

    # Whether to skip trajectories with no language labels
    config.skip_unlabeled = True

    # Whether to skip action normalization
    config.skip_norm = False

    # Shuffle buffer size for training
    config.shuffle_buffer_size = 50000

    return config
