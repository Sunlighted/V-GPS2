"""
Example configuration for train_ttt_paper.py

Usage:
    python train_ttt_paper.py \
        --config=config_ttt_paper.py \
        --oxedata_config=path/to/oxe_config.py \
        --name="ttt_paper_exp" \
        --project="ttt_paper"
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    
    # General
    config.seed = 42
    config.save_dir = "./checkpoints"
    
    # Training
    config.num_steps = 100000
    config.batch_size = 256
    config.learning_rate = 1e-4
    config.warmup_steps = 2000
    config.grad_clip = 1.0
    config.weight_decay = 0.01
    
    # Logging
    config.log_interval = 100
    config.eval_interval = 2000
    config.save_interval = 10000
    
    # OCTO encoder
    config.encoder = "octo-small"  # or "octo-base"
    
    # TTT Configuration (2024 paper style)
    config.ttt = ConfigDict()
    config.ttt.type = "linear"  # "linear" or "mlp"
    config.ttt.bottleneck_dim = 64  # IMPORTANT: Much smaller than input_dim to prevent collapse
    config.ttt.output_dim = 256  # Feature dim for RL agent
    config.ttt.eta_base = 1.0  # Inner loop learning rate (use 0.1 for MLP)
    config.ttt.use_input_dependent_lr = True  # Learn per-sample learning rate
    
    # Loss weights
    config.lambda_ssl = 1.0  # SSL loss weight
    config.lambda_rl = 1.0   # RL loss weight
    config.use_adaptation_in_training = True  # Do one adaptation step during training
    
    # RL Agent (CQL)
    config.agent = "cql"
    config.agent_kwargs = ConfigDict()
    config.agent_kwargs.discount = 0.98
    config.agent_kwargs.use_calql = True
    config.agent_kwargs.cql_alpha = 5.0
    config.agent_kwargs.cql_n_actions = 4
    config.agent_kwargs.use_td_loss = True
    config.agent_kwargs.goal_conditioned = False
    config.agent_kwargs.language_conditioned = False
    config.agent_kwargs.use_precomputed_embeddings = True  # We feed TTT features directly
    
    config.agent_kwargs.critic_network_kwargs = ConfigDict({
        "hidden_dims": [256, 256],
        "activate_final": True,
        "use_layer_norm": False,
    })
    config.agent_kwargs.policy_network_kwargs = ConfigDict({
        "hidden_dims": [256, 256],
        "activate_final": True,
        "use_layer_norm": False,
    })
    
    return config


# ============================================================================
# Alternative configs for different setups
# ============================================================================

def get_ttt_mlp_config():
    """Config using TTT-MLP (more expressive, better for longer sequences)."""
    config = get_config()
    config.ttt.type = "mlp"
    config.ttt.eta_base = 0.1  # Lower LR for MLP
    config.ttt.bottleneck_dim = 64
    return config


def get_ttt_large_bottleneck_config():
    """
    Config with larger bottleneck.
    WARNING: May be more prone to collapse - monitor ssl_loss!
    """
    config = get_config()
    config.ttt.bottleneck_dim = 128
    return config


def get_minimal_ttt_config():
    """Minimal TTT config for debugging."""
    config = get_config()
    config.num_steps = 1000
    config.eval_interval = 200
    config.log_interval = 10
    config.ttt.bottleneck_dim = 32
    config.ttt.output_dim = 128
    return config