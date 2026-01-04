"""
Configuration for TTT Robotics training.

SSL Modes:
- "reconstruction": Paper-style, f(θ_K @ obs; W) → θ_V @ obs
- "prediction": Robotics-style, f(θ_K @ [obs, action]; W) → θ_V @ next_obs

The "prediction" mode is recommended for robotics because:
1. It uses temporal structure (action causes state transition)
2. Non-trivial task: predicting next state requires understanding dynamics
3. Aligns with what RL needs: features that capture transition dynamics
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
    config.encoder = "octo-small"  # "octo-small" (384) or "octo-base" (512)
    
    # =========================================================================
    # TTT Configuration
    # =========================================================================
    config.ttt = ConfigDict()
    
    # SSL mode: "prediction" (recommended) or "reconstruction"
    # - prediction: f(θ_K @ [obs, action]; W) → θ_V @ next_obs
    # - reconstruction: f(θ_K @ obs; W) → θ_V @ obs (paper-style, ignores actions)
    config.ttt.ssl_mode = "prediction"
    
    # Bottleneck dimension - CRITICAL for preventing collapse
    # Must be << obs_dim (384 or 512)
    # Start with 32-64, increase only if SSL loss is high
    config.ttt.bottleneck_dim = 64
    
    # Output feature dimension for RL agent
    config.ttt.output_dim = 256
    
    # Inner loop learning rate
    # Paper uses 1.0 for linear, 0.1 for MLP
    config.ttt.eta_base = 1.0
    
    # Learn per-input learning rate: η(x) = η_base * σ(θ_lr @ x)
    config.ttt.use_input_dependent_lr = True
    
    # =========================================================================
    # Loss Weights
    # =========================================================================
    # SSL loss weight - drives the TTT learning
    config.lambda_ssl = 1.0
    
    # RL loss weight (if training RL jointly)
    config.lambda_rl = 1.0
    
    # =========================================================================
    # RL Agent (CQL)
    # =========================================================================
    config.agent = "cql"
    config.agent_kwargs = ConfigDict()
    
    config.agent_kwargs.discount = 0.98
    config.agent_kwargs.use_calql = True  # Use Cal-QL for better offline RL
    config.agent_kwargs.cql_alpha = 5.0
    config.agent_kwargs.cql_n_actions = 4
    config.agent_kwargs.use_td_loss = True
    
    config.agent_kwargs.goal_conditioned = False
    config.agent_kwargs.language_conditioned = False
    config.agent_kwargs.use_precomputed_embeddings = True  # TTT features are precomputed
    
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
# Alternative Configurations
# ============================================================================

def get_reconstruction_config():
    """
    Paper-style SSL (ignores actions).
    Use this to compare against prediction mode.
    """
    config = get_config()
    config.ttt.ssl_mode = "reconstruction"
    return config


def get_small_bottleneck_config():
    """
    Very small bottleneck for debugging collapse.
    If this still collapses, the problem is elsewhere.
    """
    config = get_config()
    config.ttt.bottleneck_dim = 32
    config.ttt.output_dim = 128
    return config


def get_debug_config():
    """Fast iteration for debugging."""
    config = get_config()
    config.num_steps = 2000
    config.log_interval = 10
    config.eval_interval = 500
    config.save_interval = 1000
    config.batch_size = 64
    config.ttt.bottleneck_dim = 32
    return config