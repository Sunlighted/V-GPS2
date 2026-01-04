"""
Training configuration for TTT + RL agents.
Separate from the original train_config.py for clean separation.
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string="ttt_calql"):
    """
    Get training configuration for TTT variants.
    
    Args:
        config_string: Configuration variant 
            ("ttt_cql", "ttt_calql", "ttt_iql", or their _ft variants)
    """
    base_config = dict(
        batch_size=1,  # 1 trajectory at a time for TTT
        num_steps=int(1e6),
        log_interval=1000,
        eval_interval=20000,
        save_interval=100000,
        save_dir=placeholder(str),
        resume_path="",
        seed=42,
        base_lr=1e-4,  # Base learning rate
        #weight_decay=1e-4,
        warmup_steps=2000,
        grad_clip=1.0,
        octo_max_frames_per_batch=16,  # Limit OCTO encoder batch size (None = no chunking)
        # Test-time adaptation defaults
        ttt_adapt_mode="windowed",  # "full" or "windowed"
        ttt_adapt_lr=1e-2,
        ttt_adapt_steps=2,
        ttt_adapt_window=8,
        ttt_adapt_reset=True,
        window_sampling_mode="dissimilarity",  # "full", "random", "dissimilarity"
    )

    possible_structures = {
        # ====================================================================
        # TTT-CQL: Test-Time Training + CQL
        # ====================================================================
        "ttt_cql": ConfigDict(
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
                    gc_kwargs=dict(
                        negative_proportion=0.0,
                    ),
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
                    # TTT-specific
                    octo_feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="t5",
                text_processor_kwargs=dict(),
                encoder="octo-small",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                # TTT parameters
                lambda_self=0.5,
                lambda_rl=1.0,
                rl_loss_terms=["critic","actor","temperature"],
                window_size=8,
                num_windows=8,
                **base_config,
            )
        ),
        
        "ttt_cql_ft": ConfigDict(
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
                    octo_feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="t5",
                text_processor_kwargs=dict(),
                encoder="octo-base",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                # Disable TTT
                lambda_self=0.0,
                lambda_rl=1.0,
                rl_loss_terms=["critic","actor","temperature"],
                window_size=8,
                num_windows=1,
                **base_config,
            )
        ),
        
        # ====================================================================
        # TTT-CalQL: Test-Time Training + Calibrated Q-Learning
        # ====================================================================
        "ttt_calql": ConfigDict(
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
                    use_calql=True,  # Enable CalQL
                    cql_autotune_alpha = True,
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
                    octo_feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="t5",
                text_processor_kwargs=dict(),
                encoder="octo-base",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                lambda_self=0.5,
                lambda_rl=1.0,
                rl_loss_terms=["critic","actor","temperature"],
                window_size=8,
                num_windows=8,
                **base_config,
            )
        ),
        
        "ttt_calql_ft": ConfigDict(
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
                    octo_feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="t5",
                text_processor_kwargs=dict(),
                encoder="octo-base",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                lambda_self=0.0,
                lambda_rl=1.0,
                rl_loss_terms=["critic","actor","temperature"],
                window_size=8,
                num_windows=1,
                **base_config,
            )
        ),
        
        # ====================================================================
        # TTT-IQL: Test-Time Training + IQL
        # ====================================================================
        "ttt_iql": ConfigDict(
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
                    lambda_rl=1.0,
                    rl_loss_terms=["critic","actor","temperature"],
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
                    octo_feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="t5",
                text_processor_kwargs=dict(),
                encoder="octo-base",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                lambda_self=0.5,
                window_size=8,
                num_windows=8,
                **base_config,
            )
        ),
        
        "ttt_iql_ft": ConfigDict(
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
                    octo_feature_dim=512,
                    projection_dim=256,
                    action_dim=7,
                    use_precomputed_embeddings=True,
                ),
                text_processor="t5",
                text_processor_kwargs=dict(),
                encoder="octo-base",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                lambda_self=0.0,
                window_size=8,
                num_windows=1,
                **base_config,
            )
        ),
    }

    return possible_structures[config_string]