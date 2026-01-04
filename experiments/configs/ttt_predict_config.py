"""
Training configuration for TTT-Predict v2 agents.

Changes from v1:
- P_K now takes obs only (not concat(obs, action))
- P_action separately encodes action
- P_V removed - target is stop_grad(P_K(next_obs))
- Cosine similarity loss (not MSE)
- Optional P_K/P_Q weight sharing via share_pk_pq
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string="ttt_predict_calql"):
    """
    Get training configuration for TTT-Predict v2 variants.
    
    Args:
        config_string: Configuration variant
            ("ttt_predict_cql", "ttt_predict_calql", "ttt_predict_iql",
             "ttt_predict_calql_shared", "ttt_predict_calql_nottt")
    """
    base_config = dict(
        # Training params
        batch_size=256,  # Transition-level batching
        num_steps=int(1e6),
        log_interval=1000,
        eval_interval=20000,
        save_interval=100000,
        save_dir=placeholder(str),
        resume_path="",
        seed=42,
        
        # Optimizer
        learning_rate=1e-4,
        warmup_steps=2000,
        grad_clip=1.0,
        
        # TTT-Predict v2 architecture
        share_pk_pq=False,          # Share P_K and P_Q weights
        projection_hidden_dim=128, # Hidden dim for projection heads (None = projection_dim)
        projection_num_layers=2,    # Number of layers in projection heads
        
        # TTT-Predict adaptation params
        lambda_self=0.5,       # Weight for next-state prediction loss (cosine)
        ttt_adapt_lr=1e-1,     # Inner-loop learning rate
        ttt_adapt_steps=1,     # Inner-loop steps per transition
        ttt_adapt_reset=False, # Reset f_adapt each step (True) or cumulative (False)
        adapt_during_training=True,

    # Text / encoder config helpers (present so CLI overrides work cleanly)
    # text_processor is name of text processing backend (or None)
    text_processor=None,
    text_processor_kwargs=dict(),
    # encoder kwargs are forwarded to local encoder constructors when used
    encoder_kwargs=dict(),
        
        # RL gradient control
        lambda_rl=1.0,         # Weight on RL loss backpropagating into TTT params
        rl_loss_terms=("critic", "actor", "temperature"),
    )

    possible_structures = {
        # ====================================================================
        # TTT-Predict v2 + CQL
        # ====================================================================
        "ttt_predict_cql": ConfigDict(
            dict(
                agent="cql",
                agent_kwargs=dict(
                    language_conditioned=False,
                    goal_conditioned=False,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=False,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                    # TTT-Predict dimensions
                    octo_feature_dim=384,   # 384 for octo-small, 512 for octo-base
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                encoder="octo-small",
                **base_config,
            )
        ),
        
        # ====================================================================
        # TTT-Predict v2 + CalQL (Calibrated Q-Learning) - DEFAULT
        # ====================================================================
        "ttt_predict_calql": ConfigDict(
            dict(
                agent="cql",
                agent_kwargs=dict(
                    language_conditioned=False,
                    goal_conditioned=False,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=True,
                    cql_autotune_alpha=True,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                    feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                **base_config,
            )
        ),
        
        # ====================================================================
        # TTT-Predict v2 + CalQL with shared P_K/P_Q
        # ====================================================================
        "ttt_predict_calql_shared": ConfigDict(
            dict(
                agent="cqlfix",
                agent_kwargs=dict(
                    language_conditioned=False,
                    goal_conditioned=False,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=True,
                    cql_autotune_alpha=True,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                    octo_feature_dim=384,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                encoder="octo-small",
                share_pk_pq=True,  # Share P_K and P_Q
                **{k: v for k, v in base_config.items() if k != 'share_pk_pq'},
            )
        ),
        
        # ====================================================================
        # TTT-Predict v2 + IQL
        # ====================================================================
        "ttt_predict_iql": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=dict(
                    language_conditioned=False,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    expectile=0.7,
                    temperature=1.0,
                    target_update_rate=0.002,
                    negative_proportion=0.1,
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        std_parameterization="exp",
                    ),
                    network_kwargs=dict(
                        hidden_dims=(256, 256),
                        dropout_rate=0.0,
                    ),
                    octo_feature_dim=384,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                encoder="octo-small",
                **base_config,
            )
        ),
        
        # ====================================================================
        # Ablation: No TTT (lambda_self=0)
        # ====================================================================
        "ttt_predict_calql_nottt": ConfigDict(
            dict(
                agent="cqlfix",
                agent_kwargs=dict(
                    language_conditioned=False,
                    goal_conditioned=False,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=True,
                    cql_autotune_alpha=True,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                    octo_feature_dim=384,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                encoder="octo-small",
                lambda_self=0.0,  # Disable TTT loss
                **{k: v for k, v in base_config.items() if k != 'lambda_self'},
            )
        ),
        
        # ====================================================================
        # Ablation: Deeper projection heads (2-layer MLPs)
        # ====================================================================
        "ttt_predict_calql_deep": ConfigDict(
            dict(
                agent="cqlfix",
                agent_kwargs=dict(
                    language_conditioned=False,
                    goal_conditioned=False,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=True,
                    cql_autotune_alpha=True,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                    octo_feature_dim=384,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                encoder="octo-small",
                projection_num_layers=2,
                projection_hidden_dim=256,
                **{k: v for k, v in base_config.items() 
                   if k not in ('projection_num_layers', 'projection_hidden_dim')},
            )
        ),
    }

    return possible_structures[config_string]