"""Configuration for CQL agent training with pre-computed embeddings.

This config is designed for use with train_from_embedding.py and
EmbeddingCQLAgent with use_precomputed_embeddings=True.
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config():
    config = ConfigDict()

    # Agent type
    config.agent = "cqlfix"

    # Training parameters
    config.batch_size = 512
    config.num_steps = 500000
    config.log_interval = 100
    config.eval_interval = 5000
    config.save_interval = 25000
    config.seed = 42

    # Save directory
    config.save_dir = placeholder(str)

    # Resume from checkpoint (optional)
    config.resume_path = ""

    # Agent kwargs - configured for embedding-based training
    config.agent_kwargs = ConfigDict()
    config.agent_kwargs.use_precomputed_embeddings = True
    config.agent_kwargs.goal_conditioned = True
    config.agent_kwargs.language_conditioned = True
    config.agent_kwargs.use_calql = True
    config.agent_kwargs.cql_alpha = 5.0
    config.agent_kwargs.discount = 0.98
    config.agent_kwargs.target_update_rate = 0.005
    config.agent_kwargs.warmup_steps = 2000
    config.agent_kwargs.use_proprio = False
    config.agent_kwargs.shared_encoder = False

    # Policy kwargs
    config.agent_kwargs.policy_kwargs = ConfigDict()
    config.agent_kwargs.policy_kwargs.tanh_squash_distribution = True
    config.agent_kwargs.policy_kwargs.std_parameterization = "exp"

    # Goal-conditioning kwargs
    config.agent_kwargs.gc_kwargs = ConfigDict()
    config.agent_kwargs.gc_kwargs.negative_proportion = 0.0

    # Network architecture - adjusted for embeddings
    # Input embedding dim is 512 (Octo) + 768 (T5 language) = 1280
    config.agent_kwargs.critic_network_kwargs = ConfigDict()
    config.agent_kwargs.critic_network_kwargs.hidden_dims = [512, 512, 256]
    config.agent_kwargs.critic_network_kwargs.activate_final = True
    config.agent_kwargs.critic_network_kwargs.use_layer_norm = True

    config.agent_kwargs.policy_network_kwargs = ConfigDict()
    config.agent_kwargs.policy_network_kwargs.hidden_dims = [512, 512, 256]
    config.agent_kwargs.policy_network_kwargs.activate_final = True
    config.agent_kwargs.policy_network_kwargs.use_layer_norm = True

    # No encoder needed for pre-computed embeddings
    config.encoder = None
    config.encoder_kwargs = ConfigDict()

    # No text processor needed - language embeddings are pre-computed
    config.text_processor = None
    config.text_processor_kwargs = ConfigDict()

    return config
