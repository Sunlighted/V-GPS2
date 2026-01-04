"""
Configuration for unified CQL+SimSiam training.

This config is simplified compared to the original since SimSiam parameters
are now part of the agent config rather than separate.
"""

from typing import Sequence
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string: str = "lc_cql_simsiam"):
    """Get training configuration."""
    
    base_training = {
        # Training schedule
        "batch_size": 256,
        "num_steps": int(1e6),
        "log_interval": 1000,
        "eval_interval": 20000,
        "save_interval": 100000,
        
        # Checkpointing
        "save_dir": placeholder(str),
        "resume_path": "",
        
        # Reproducibility
        "seed": 42,
        
        # Evaluation
        "eval_batches": 4,
        "num_value_plots": 2,
        
        # TTT (Test-Time Training)
        "ttt_steps": 1,
        "ttt_lr": 1e-3,
        "ttt_mode": "online",  # "standard", "online", or "off"
    }

    # SimSiam network architecture (EfficientZero-style)
    ttt_architecture = {
        "feature_dim": 512,             # 编码器输出的维度
        "action_dim": 7,                # 机械臂动作维度 (如 Franka 为 7)
        "projection_hidden_dim": 512,    # P_K 和 P_Q 的隐藏层维度
        "projection_num_layers": 2,     # 投影头的层数
    }
    
    # TTT 训练损失设置
    ttt_training = {
        "lambda_ttt": 0.5,           # TTT 自监督损失权重
        "stop_grad_target": True,    # 对下一帧编码应用 stop_gradient
        "normalize_latents": True,   # 计算余弦相似度前是否进行 L2 归一化
    }

    configs = {
        # ============================================================
        # Language-conditioned CQL + TTT (recommended)
        # ============================================================
        "lc_cql_ttt": ConfigDict({
            "agent": "cql_ttt",
            "agent_kwargs": {
                # Goal/language conditioning
                "language_conditioned": True,
                "goal_conditioned": True,
                "early_goal_concat": None,
                "shared_goal_encoder": None,
                "shared_encoder": True,  # Required for SimSiam!
                
                # RL hyperparameters
                "discount": 0.98,
                "cql_alpha": 5.0,
                "target_update_rate": 5e-3,
                "use_calql": True,
                
                # CQL-specific
                "gc_kwargs": {
                    "negative_proportion": 0.0,
                },
                
                # Network architectures
                "critic_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_kwargs": {
                    "tanh_squash_distribution": True,
                    "std_parameterization": "exp",
                },
                
                # Optimizers
                "actor_optimizer_kwargs": {
                    "learning_rate": 1e-4,
                    "warmup_steps": 2000,
                },
                "critic_optimizer_kwargs": {
                    "learning_rate": 3e-4,
                    "warmup_steps": 2000,
                },
                
                # SimSiam optimizer (for projector, predictor, dynamics)
                "ttt_module_optimizer_kwargs": {
                    "learning_rate": 3e-4,
                    "warmup_steps": 2000,
                },
                
                # TTT 损失与训练参数
                "lambda_ttt": ttt_training["lambda_ttt"],
                "stop_grad_target": ttt_training["stop_grad_target"],
                "normalize_latents": ttt_training["normalize_latents"],
                
                # TTT 架构参数 (传给 agent.create)
                "ttt_kwargs": {
                    "feature_dim": ttt_architecture["feature_dim"],
                    "action_dim": ttt_architecture["action_dim"],
                    "projection_hidden_dim": ttt_architecture["projection_hidden_dim"],
                    "projection_num_layers": ttt_architecture["projection_num_layers"],
                },
            },
            
            # Text/language processing
            "text_processor": "muse_embedding",
            "text_processor_kwargs": {},
            
            # Vision encoder
            "encoder": "resnetv1-34-bridge-film",
            "encoder_kwargs": {
                "pooling_method": "avg",
                "add_spatial_coordinates": True,
                "act": "swish",
            },
            
            # Training settings
            **base_training,
        }),
        
        # ============================================================
        # CQL + SimSiam with convolutional dynamics (for spatial features)
        # ============================================================
        "lc_cql_simsiam_conv": ConfigDict({
            "agent": "cql_simsiam",
            "agent_kwargs": {
                "language_conditioned": True,
                "goal_conditioned": True,
                "early_goal_concat": None,
                "shared_goal_encoder": None,
                "shared_encoder": True,
                
                "discount": 0.98,
                "cql_alpha": 5.0,
                "target_update_rate": 5e-3,
                "use_calql": True,
                
                "gc_kwargs": {"negative_proportion": 0.0},
                
                "critic_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_kwargs": {
                    "tanh_squash_distribution": True,
                    "std_parameterization": "exp",
                },
                
                "actor_optimizer_kwargs": {"learning_rate": 1e-4, "warmup_steps": 2000},
                "critic_optimizer_kwargs": {"learning_rate": 3e-4, "warmup_steps": 2000},
                "simsiam_optimizer_kwargs": {"learning_rate": 3e-4, "warmup_steps": 2000},
                
                "lambda_sim": 2.0,
                "stop_grad_target": True,
                "normalize_latents": True,
                
                "simsiam_kwargs": {
                    "projector_hidden_dims": (512, 512, 512),
                    "projector_output_dim": 512,
                    "projector_norm": "layer",
                    "predictor_hidden_dims": (256,),
                    "predictor_output_dim": 512,
                    "predictor_norm": "layer",
                    "dynamics_hidden_dim": 512,
                    "dynamics_num_residual_blocks": 1,
                    "dynamics_norm": "layer",
                    "use_conv_dynamics": True,  # Use convolutional dynamics!
                    "dynamics_latent_channels": 64,
                },
            },
            
            "text_processor": "muse_embedding",
            "text_processor_kwargs": {},
            
            # Use encoder WITHOUT pooling for conv dynamics
            "encoder": "resnetv1-34-bridge-film",
            "encoder_kwargs": {
                "pooling_method": "none",  # No pooling - keep spatial dims
                "add_spatial_coordinates": True,
                "act": "swish",
            },
            
            **base_training,
        }),
        
        # ============================================================
        # CQL + SimSiam with SGD optimizer (closer to EfficientZero)
        # ============================================================
        "lc_cql_simsiam_sgd": ConfigDict({
            "agent": "cql_simsiam",
            "agent_kwargs": {
                "language_conditioned": True,
                "goal_conditioned": True,
                "early_goal_concat": None,
                "shared_goal_encoder": None,
                "shared_encoder": True,
                
                "discount": 0.98,
                "cql_alpha": 5.0,
                "target_update_rate": 5e-3,
                "use_calql": True,
                
                "gc_kwargs": {"negative_proportion": 0.0},
                
                "critic_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_kwargs": {
                    "tanh_squash_distribution": True,
                    "std_parameterization": "exp",
                },
                
                # Use SGD with momentum for all components (EfficientZero style)
                # LR schedule: 0.2 -> 0.02 at 100k steps (10x decay)
                "actor_optimizer_kwargs": {
                    "learning_rate": 0.2,
                    "warmup_steps": 0,
                    "optimizer_type": "sgd",
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "lr_decay_steps": 100000,
                    "end_learning_rate": 0.02,
                    "clip_grad_norm": 5.0,
                },
                "critic_optimizer_kwargs": {
                    "learning_rate": 0.2,
                    "warmup_steps": 0,
                    "optimizer_type": "sgd",
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "lr_decay_steps": 100000,
                    "end_learning_rate": 0.02,
                    "clip_grad_norm": 5.0,
                },
                "simsiam_optimizer_kwargs": {
                    "learning_rate": 0.2,
                    "warmup_steps": 0,
                    "optimizer_type": "sgd",
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "lr_decay_steps": 100000,
                    "end_learning_rate": 0.02,
                    "clip_grad_norm": 5.0,
                },
                
                "lambda_sim": 2.0,
                "stop_grad_target": True,
                "normalize_latents": True,
                
                "simsiam_kwargs": {
                    "projector_hidden_dims": (512, 512, 512),
                    "projector_output_dim": 512,
                    "projector_norm": "layer",
                    "predictor_hidden_dims": (256,),
                    "predictor_output_dim": 512,
                    "predictor_norm": "layer",
                    "dynamics_hidden_dim": 512,
                    "dynamics_num_residual_blocks": 1,
                    "dynamics_norm": "layer",
                    "use_conv_dynamics": False,
                    "dynamics_latent_channels": 64,
                },
            },
            
            "text_processor": "muse_embedding",
            "text_processor_kwargs": {},
            
            "encoder": "resnetv1-34-bridge-film",
            "encoder_kwargs": {
                "pooling_method": "avg",
                "add_spatial_coordinates": True,
                "act": "swish",
            },
            
            **base_training,
        }),

        # ============================================================
        # CQL + SimSiam with BatchNorm (instead of LayerNorm)
        # ============================================================
        "lc_cql_simsiam_batchnorm": ConfigDict({
            "agent": "cql_simsiam",
            "agent_kwargs": {
                "language_conditioned": True,
                "goal_conditioned": True,
                "early_goal_concat": None,
                "shared_goal_encoder": None,
                "shared_encoder": True,

                "discount": 0.98,
                "cql_alpha": 5.0,
                "target_update_rate": 5e-3,
                "use_calql": True,

                "gc_kwargs": {"negative_proportion": 0.0},

                "critic_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_kwargs": {
                    "tanh_squash_distribution": True,
                    "std_parameterization": "exp",
                },

                "actor_optimizer_kwargs": {"learning_rate": 1e-4, "warmup_steps": 2000},
                "critic_optimizer_kwargs": {"learning_rate": 3e-4, "warmup_steps": 2000},
                "simsiam_optimizer_kwargs": {"learning_rate": 3e-4, "warmup_steps": 2000},

                "lambda_sim": 2.0,
                "stop_grad_target": True,
                "normalize_latents": True,

                # SimSiam architecture with BatchNorm
                "simsiam_kwargs": {
                    "projector_hidden_dims": (512, 512, 512),
                    "projector_output_dim": 512,
                    "projector_norm": "batch",
                    "predictor_hidden_dims": (256,),
                    "predictor_output_dim": 512,
                    "predictor_norm": "batch",
                    "dynamics_hidden_dim": 512,
                    "dynamics_num_residual_blocks": 1,
                    "dynamics_norm": "batch",
                    "use_conv_dynamics": False,
                    "dynamics_latent_channels": 64,
                },
            },

            "text_processor": "muse_embedding",
            "text_processor_kwargs": {},

            "encoder": "resnetv1-34-bridge-film",
            "encoder_kwargs": {
                "pooling_method": "avg",
                "add_spatial_coordinates": True,
                "act": "swish",
            },

            **base_training,
        }),

        # ============================================================
        # Baseline: CQL only (no SimSiam) for comparison
        # ============================================================
        "lc_cql_baseline": ConfigDict({
            "agent": "cql_simsiam",
            "agent_kwargs": {
                "language_conditioned": True,
                "goal_conditioned": True,
                "early_goal_concat": None,
                "shared_goal_encoder": None,
                "shared_encoder": True,
                
                "discount": 0.98,
                "cql_alpha": 5.0,
                "target_update_rate": 5e-3,
                "use_calql": True,
                
                "gc_kwargs": {"negative_proportion": 0.0},
                
                "critic_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_network_kwargs": {
                    "hidden_dims": [256, 256],
                    "activate_final": True,
                    "use_layer_norm": False,
                },
                "policy_kwargs": {
                    "tanh_squash_distribution": True,
                    "std_parameterization": "exp",
                },
                
                "actor_optimizer_kwargs": {"learning_rate": 1e-4, "warmup_steps": 2000},
                "critic_optimizer_kwargs": {"learning_rate": 3e-4, "warmup_steps": 2000},
                "simsiam_optimizer_kwargs": {"learning_rate": 3e-4, "warmup_steps": 2000},
                
                # Disable SimSiam by setting lambda_sim=0
                "lambda_sim": 0.0,
                "stop_grad_target": True,
                "normalize_latents": True,
                
                "simsiam_kwargs": {},  # Not used when lambda_sim=0
            },
            
            "text_processor": "muse_embedding",
            "text_processor_kwargs": {},
            
            "encoder": "resnetv1-34-bridge-film",
            "encoder_kwargs": {
                "pooling_method": "avg",
                "add_spatial_coordinates": True,
                "act": "swish",
            },
            
            # Include base_training but override ttt_mode
            **base_training,
            "ttt_mode": "off",  # No TTT without SimSiam
        }),
    }

    if config_string not in configs:
        raise ValueError(
            f"Unknown config '{config_string}'. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[config_string]