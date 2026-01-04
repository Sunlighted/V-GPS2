"""
TTT-Predict Agent v3: Test-Time Training with next-state prediction objective.

Architecture:
    Encoder → Projections → TTT Module (f_adapt) → RL Agent

Self-supervised objective (cosine similarity):
    f_adapt(P_K(obs, action)) → stop_grad(next_obs)
    
Feature extraction for RL:
    f_adapt(P_Q(obs, action)) → Q-network
    
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Any, Sequence, Tuple
import optax
from functools import partial
from absl import logging

from ttt_module import TTTModule
from jaxrl_m.agents import agents


class ProjectionHead(nn.Module):
    """Linear projection head with optional hidden layers and LayerNorm on every layer."""
    output_dim: int
    hidden_dim: Optional[int] = None
    num_layers: int = 1
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x):
        hidden_dim = self.hidden_dim or self.output_dim
        if self.num_layers <= 1:
            x = nn.Dense(self.output_dim)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name="ln_out")(x)
            return x
        
        num_hidden = self.num_layers - 1
        for idx in range(num_hidden):
            x = nn.Dense(hidden_dim, name=f"hidden_{idx}")(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"ln_{idx}")(x)
            x = nn.relu(x)
        
        x = nn.Dense(self.output_dim, name="Dense_0")(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="ln_out")(x)
        return x

def cosine_similarity_loss(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Compute cosine similarity loss: 1 - cos_sim(pred, target).
    
    Args:
        pred: (B, D) predictions
        target: (B, D) targets (should have stop_gradient applied)
        eps: Small constant for numerical stability
        
    Returns:
        loss: Scalar loss value
    """
    pred_norm = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + eps)
    target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + eps)
    cos_sim = jnp.sum(pred_norm * target_norm, axis=-1)  # (B,)
    loss = jnp.mean(1 - cos_sim)
    return loss


class TTTPredictFeatureExtractor(nn.Module):
    """
    TTT-Predict v3:
    - 适配输入: P_K(concat(obs, action))
    - 适配目标: next_obs (原始 OCTO 特征)
    """
    feature_dim: int      # OCTO 原始维度 (如 384 或 512)
    action_dim: int            # 动作维度
    projection_dim: int = 256   # 最终输出给 RL 的维度
    projection_hidden_dim: Optional[int] = None  
    projection_num_layers: int = 2              

    def setup(self):
        head_hidden_dim = self.projection_hidden_dim or self.feature_dim

        # P_K: 将 (obs + action) 投影到原始特征维度，以便与 next_obs 对齐
        self.P_K = ProjectionHead(
            output_dim=self.feature_dim,
            hidden_dim=head_hidden_dim,
            num_layers=self.projection_num_layers,
            name='P_K',
        )
        
        # TTT 适配模块，输入和输出都是 feature_dim
        self.f_adapt = TTTModule(
            input_dim=self.feature_dim,
            name='f_adapt'
        )

        # P_Q: 将适配后的特征映射到 RL 代理所需的维度
        self.P_Q = ProjectionHead(
            output_dim=self.feature_dim,
            hidden_dim=head_hidden_dim,
            num_layers=self.projection_num_layers,
            name='P_Q',
        )

    def __call__(self, obs_embeddings: jnp.ndarray, action: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        提取用于 RL 的特征。
        """
        # 1. 拼接 obs 和零动作并投影
        z_input = jnp.concatenate([obs_embeddings, action], axis=-1)
        z_proj = self.P_Q(z_input)
        
        # 2. 经过 TTT 适配层
        z_adapted = self.f_adapt(z_proj)
        
        # 3. 投影到 RL 维度
        return z_adapted
    
    def compute_self_supervised_loss(
        self,
        obs_embeddings: jnp.ndarray,
        actions: jnp.ndarray,
        next_obs_embeddings: jnp.ndarray,
        train: bool = True
    ) -> jnp.ndarray:
        """
        Loss: 1 - cos_sim(f_adapt(P_K(obs, action)), stop_grad(next_obs))
        """
        # 1. 拼接并投影
        combined = jnp.concatenate([obs_embeddings, actions], axis=-1)
        z_input = self.P_K(combined)
        
        if not train:
            # 测试时只更新 f_adapt 的权重，P_K 视为固定特征提取器
            z_input = jax.lax.stop_gradient(z_input)
        
        # 2. 预测下一帧
        pred = self.f_adapt(z_input)
        
        # 3. 目标是原始 next_obs (加 stop_gradient)
        target = jax.lax.stop_gradient(next_obs_embeddings)
        
        return cosine_similarity_loss(pred, target)

    def get_adaptation_inputs(
        self,
        obs_embeddings: jnp.ndarray,
        actions: jnp.ndarray,
        next_obs_embeddings: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """获取适配过程的直接输入/输出对"""
        combined = jnp.concatenate([obs_embeddings, actions], axis=-1)
        z_input = self.P_K(combined)
        return z_input, next_obs_embeddings

def _dense_project(x: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Apply a projection head using raw params."""
    # Handle single-layer case
    if 'kernel' in params:
        y = x @ params['kernel'] + params['bias']
        # Check for output LayerNorm
        if 'ln_out' in params:
            ln = params['ln_out']
            y = (y - jnp.mean(y, axis=-1, keepdims=True)) / (jnp.std(y, axis=-1, keepdims=True) + 1e-6)
            y = y * ln['scale'] + ln['bias']
        return y
    
    # Multi-layer case
    y = x
    layer_names = list(params.keys())
    
    # Count hidden layers
    hidden_layers = sorted([n for n in layer_names if n.startswith('hidden_')])
    num_hidden = len(hidden_layers)
    
    for idx in range(num_hidden):
        # Dense
        h_params = params[f'hidden_{idx}']
        y = y @ h_params['kernel'] + h_params['bias']
        # LayerNorm (if present)
        ln_key = f'ln_{idx}'
        if ln_key in params:
            ln_params = params[ln_key]
            y = (y - jnp.mean(y, axis=-1, keepdims=True)) / (jnp.std(y, axis=-1, keepdims=True) + 1e-6)
            y = y * ln_params['scale'] + ln_params['bias']
        # ReLU
        y = nn.relu(y)
    
    # Final layer
    final_params = params['Dense_0']
    y = y @ final_params['kernel'] + final_params['bias']
    
    # Output LayerNorm (if present)
    if 'ln_out' in params:
        ln_params = params['ln_out']
        y = (y - jnp.mean(y, axis=-1, keepdims=True)) / (jnp.std(y, axis=-1, keepdims=True) + 1e-6)
        y = y * ln_params['scale'] + ln_params['bias']
    
    return y


def ttt_adaptation_step_cosine(
    f_adapt_params: Dict,
    z_input: jnp.ndarray,
    z_target: jnp.ndarray,
    projection_dim: int,
    lr: float,
    eps: float = 1e-8
) -> Tuple[Dict, jnp.ndarray]:
    """
    Single gradient step for TTT adaptation with cosine similarity loss.
    
    Args:
        f_adapt_params: Parameters of f_adapt module
        z_input: (B, proj_dim) - P_K(obs) + P_action(action)
        z_target: (B, proj_dim) - stop_grad(P_K(next_obs))
        projection_dim: Dimension of projections
        lr: Learning rate
        eps: Small constant for numerical stability
        
    Returns:
        updated_params: New f_adapt parameters
        loss: Scalar loss value
    """
    ttt_module = TTTModule(input_dim=projection_dim)
    
    def loss_fn(params):
        pred = ttt_module.apply({'params': params}, z_input)
        return cosine_similarity_loss(pred, z_target, eps)
    
    loss, grads = jax.value_and_grad(loss_fn)(f_adapt_params)
    updated_params = jax.tree_map(lambda p, g: p - lr * g, f_adapt_params, grads)
    
    return updated_params, loss


def ttt_adaptation_cosine(
    f_adapt_params: Dict,
    z_input: jnp.ndarray,
    z_target: jnp.ndarray,
    projection_dim: int,
    lr: float,
    steps: int
) -> Tuple[Dict, jnp.ndarray]:
    """
    Run multiple TTT adaptation steps with cosine similarity loss.
    
    Args:
        f_adapt_params: Initial parameters of f_adapt
        z_input: (B, proj_dim) - input (P_K(obs) + P_action(action))
        z_target: (B, proj_dim) - target (P_K(next_obs))
        projection_dim: Dimension of projections
        lr: Learning rate
        steps: Number of adaptation steps
        
    Returns:
        adapted_params: Updated f_adapt parameters
        losses: (steps,) array of losses
    """
    def step_fn(params, _):
        new_params, loss = ttt_adaptation_step_cosine(
            params, z_input, z_target, projection_dim, lr
        )
        return new_params, loss
    
    adapted_params, losses = jax.lax.scan(step_fn, f_adapt_params, None, length=steps)
    return adapted_params, losses


def create_ttt_predict_agent(
    rng: jnp.ndarray,
    obs_example: jnp.ndarray,          # (B, octo_dim)
    next_obs_example: jnp.ndarray,     # (B, octo_dim)
    actions_example: jnp.ndarray,      # (B, action_dim)
    feature_dim: int,
    action_dim: int,
    projection_dim: int,
    agent_config: Dict,
    projection_hidden_dim: Optional[int] = None,
    projection_num_layers: int = 2,
    octo_model=None,
) -> Tuple[TTTPredictFeatureExtractor, Any, Dict]:
    """
    Create a combined TTT-Predict + RL agent.
    
    Args:
        rng: Random key
        obs_example: Example observation embeddings
        next_obs_example: Example next observation embeddings
        actions_example: Example actions
        feature_dim: OCTO feature dimension
        action_dim: Action dimension
        projection_dim: TTT projection dimension
        agent_config: Full config dict with agent type and parameters
        projection_hidden_dim: Hidden dimension for intermediate projection layers
        projection_num_layers: Number of Dense layers per projection head
        share_pk_pq: Whether to share P_K and P_Q weights
        octo_model: OCTO model (for EmbeddingCQLAgent if needed)
        
    Returns:
        ttt_extractor: TTT feature extractor module
        rl_agent: RL agent (CQL/IQL/etc.)
        ttt_params: TTT parameters dict
    """
    # Initialize TTT feature extractor
    ttt_extractor = TTTPredictFeatureExtractor(
        feature_dim=feature_dim,
        action_dim=action_dim,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_num_layers=projection_num_layers,
    )
    
    rng, ttt_rng = jax.random.split(rng)
    
    # Initialize all parameters by calling both methods
    def init_all_params(module, obs, actions, next_obs, train):
        # Initialize P_Q (or P_K if shared) and f_adapt via forward pass
        _ = module(obs, train=train)
        # Initialize P_K and P_action via self-supervised loss
        _ = module.compute_self_supervised_loss(obs, actions, next_obs, train=train)
        return None
    
    ttt_vars = ttt_extractor.init(
        ttt_rng,
        obs_example,
        actions_example,
        next_obs_example,
        True,  # train
        method=init_all_params
    )
    ttt_params = ttt_vars['params']
    
    # Get adapted features for agent initialization
    adapted_features = ttt_extractor.apply(
        {'params': ttt_params},
        obs_example,
        train=True
    )
    
    # Create observation dict with adapted features
    obs_for_agent = {'image': adapted_features}
    goals_for_agent = {}
    
    # Initialize RL agent
    agent_type = agent_config['agent']
    agent_kwargs = dict(agent_config['agent_kwargs'])
    
    logging.info(f"Initializing {agent_type} agent with TTT-Predict features (projection_dim={projection_dim})")
    
    rng, agent_rng = jax.random.split(rng)
    
    if octo_model is not None:
        agent_kwargs['octo_model'] = octo_model
    
    rl_agent = agents[agent_type].create(
        rng=agent_rng,
        observations=obs_for_agent,
        goals=goals_for_agent,
        actions=actions_example,
        **agent_kwargs,
    )
    
    logging.info(f"Created TTT-Predict + {agent_type} agent successfully")
    logging.info(f"  P_K input dim: {feature_dim+action_dim}")
    logging.info(f"  P_Q: {f'input dim {feature_dim}'}")
    logging.info(f"  Output dim: {projection_dim}")
    logging.info(f"  Loss: cosine similarity")
    
    return ttt_extractor, rl_agent, ttt_params


def create_ttt_predict_update_step(
    ttt_extractor: TTTPredictFeatureExtractor,
    lambda_self: float = 0.5,
    lambda_rl: float = 0.0,
    rl_loss_terms: Sequence[str] = ("critic", "actor", "temperature"),
    ttt_adapt_lr: float = 1e-2,
    ttt_adapt_steps: int = 5,
    adapt_during_training: bool = True, 
    axis_name: str | None = None,
):
    """
    Create the training update step for TTT-Predict.
    
    This function builds a pure-JAX update that:
    1. Computes the self-supervised next-state prediction loss (cosine similarity)
    2. Runs differentiable TTT adaptation
    3. Extracts adapted features for RL
    4. Returns the RL batch for the agent's own update
    
    Args:
        ttt_extractor: TTT feature extractor module
        lambda_self: Weight for self-supervised loss
        lambda_rl: Weight for RL loss (if computed here)
        rl_loss_terms: Sequence of RL loss components to include
        ttt_adapt_lr: Learning rate for inner-loop adaptation
        ttt_adapt_steps: Number of inner-loop steps
        axis_name: Axis name for pmap (None for single-device)
        
    Returns:
        update_step: Function (params, opt_state, batch, rng, tx, rl_agent) → (params, opt_state, metrics, agent_batch)
    """
    projection_dim = ttt_extractor.projection_dim
    rl_loss_terms = tuple(rl_loss_terms or ())

    def _compute_rl_loss(rl_agent, batch, rng):
        if rl_agent is None or lambda_rl <= 0.0 or not rl_loss_terms:
            return 0.0, {}

        rl_loss = 0.0
        rl_metrics = {}
        params = rl_agent.state.params

        for term in rl_loss_terms:
            rng, term_rng = jax.random.split(rng)
            if term == 'critic':
                term_loss, term_info = rl_agent.critic_loss_fn(
                    batch, params, term_rng, train=True,
                )
            elif term == 'actor':
                term_loss, term_info = rl_agent.policy_loss_fn(
                    batch, params, term_rng,
                )
            elif term == 'temperature':
                term_loss, term_info = rl_agent.temperature_loss_fn(
                    batch, params, term_rng,
                )
            else:
                raise ValueError(f"Unknown rl_loss_terms entry '{term}'")

            rl_loss = rl_loss + term_loss
            rl_metrics[f"{term}_loss"] = term_loss
            rl_metrics[f"{term}_info"] = term_info

        return rl_loss, rl_metrics
    
    def loss_fn(ttt_params, batch, rng, rl_agent):
        """
        Compute TTT loss and construct RL batch.
        
        batch contains:
            - obs_embeddings: (B, octo_dim)
            - next_obs_embeddings: (B, octo_dim)
            - actions: (B, action_dim)
            - rewards: (B,)
            - masks: (B,)
            - mc_returns: (B,) [optional]
        """
        obs = batch['obs_embeddings']
        next_obs = batch['next_obs_embeddings']
        actions = batch['actions']
        rewards = batch['rewards']
        masks = batch['masks']
        mc_returns = batch.get('mc_returns')
        
        # 1) Compute projections
        combined = jnp.concatenate([obs, actions], axis=-1)
        z_combined = _dense_project(combined, ttt_params['P_K'])
        z_input = z_combined
        
        # Target with stop_gradient
        z_target = jax.lax.stop_gradient(next_obs)
        
        # 2) TTT self-supervised loss: cosine similarity
        ttt_module = TTTModule(input_dim=projection_dim)
        pred = ttt_module.apply({'params': ttt_params['f_adapt']}, z_input)
        loss_ttt = cosine_similarity_loss(pred, z_target)
        
        # 3) Differentiable TTT adaptation for feature extraction
        if adapt_during_training:
            # Run differentiable adaptation
            adapted_f_adapt, inner_losses = _run_differentiable_adaptation(
                ttt_params, obs, actions, next_obs,
                projection_dim, ttt_adapt_lr, ttt_adapt_steps,
            )
            inner_loss_mean = jnp.mean(inner_losses)
            inner_loss_last = inner_losses[-1] if inner_losses.size > 0 else 0.0
        else:
            # Use base f_adapt (no adaptation during training)
            adapted_f_adapt = ttt_params['f_adapt']
            inner_loss_mean = 0.0
            inner_loss_last = 0.0
        
        # 4) Extract features with adapted f_adapt
        query = _dense_project(obs, ttt_params['P_Q'])
        query_next = _dense_project(next_obs, ttt_params['P_Q'])
        
        adapted_features = ttt_module.apply({'params': adapted_f_adapt}, query)
        adapted_next_features = ttt_module.apply({'params': adapted_f_adapt}, query_next)
        
        # Track feature statistics
        feat_mean = jnp.mean(adapted_features)
        feat_std = jnp.std(adapted_features)
        feat_batch_std = jnp.mean(jnp.std(adapted_features, axis=0))
        
        # Effective rank (measure of representation quality)
        U, S, Vt = jnp.linalg.svd(adapted_features, full_matrices=False)
        normalized_S = S / (jnp.sum(S) + 1e-8)
        effective_rank = jnp.exp(-jnp.sum(normalized_S * jnp.log(normalized_S + 1e-8)))
        
        # Cosine similarity between pred and target (for monitoring)
        pred_norm = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        target_norm = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
        cos_sim_mean = jnp.mean(jnp.sum(pred_norm * target_norm, axis=-1))
        
        # 5) Construct RL batch
        agent_batch = {
            'observations': {'image': adapted_features},
            'next_observations': {'image': adapted_next_features},
            'actions': actions,
            'rewards': rewards,
            'masks': masks,
            'goals': {},
        }
        
        if mc_returns is not None:
            agent_batch['mc_returns'] = mc_returns
        
        # Total loss
        loss_total = lambda_self * loss_ttt
        
        metrics = {
            'loss_total': loss_total,
            'loss_ttt': loss_ttt,
            'cos_sim': cos_sim_mean,
            'ttt_inner_loss_mean': inner_loss_mean,
            'ttt_inner_loss_last': inner_loss_last,
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'feat_batch_std': feat_batch_std,
            'feat_eff_rank': effective_rank,
            'z_obs_action_mean': jnp.mean(z_combined),
            'z_obs_action_std': jnp.std(z_combined),
            'z_target_mean': jnp.mean(z_target),
            'z_target_std': jnp.std(z_target),
        }

        if lambda_rl > 0.0 and rl_loss_terms:
            rng, rl_rng = jax.random.split(rng)
            rl_loss, rl_info = _compute_rl_loss(rl_agent, agent_batch, rl_rng)
            loss_total = loss_total + lambda_rl * rl_loss
            metrics['loss_total'] = loss_total
            metrics['loss_rl'] = rl_loss
            for key, value in rl_info.items():
                metrics[f"rl/{key}"] = value
        
        return loss_total, (metrics, agent_batch)
    
    def _maybe_pmean(value):
        if axis_name is None:
            return value
        if isinstance(value, jax.Array):
            return jax.lax.pmean(value, axis_name=axis_name)
        return value

    def update_step(ttt_params, opt_state, batch, rng, tx, rl_agent):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, agent_batch)), grads = grad_fn(ttt_params, batch, rng, rl_agent)

        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name=axis_name)
            metrics = jax.tree_map(_maybe_pmean, metrics)
            loss = jax.lax.pmean(loss, axis_name=axis_name)
        
        updates, new_opt_state = tx.update(grads, opt_state, ttt_params)
        new_params = optax.apply_updates(ttt_params, updates)
        
        return new_params, new_opt_state, metrics, agent_batch
    
    return update_step


def _run_differentiable_adaptation(
    ttt_params: Dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    next_obs: jnp.ndarray,
    projection_dim: int,
    adapt_lr: float,
    adapt_steps: int,
) -> Tuple[Dict, jnp.ndarray]:
    """
    Run differentiable TTT adaptation so gradients flow through projection heads.
    
    Args:
        ttt_params: Full TTT parameter tree
        obs: (B, octo_dim) current observations
        actions: (B, action_dim) actions
        next_obs: (B, octo_dim) next observations
        projection_dim: Projection dimension
        adapt_lr: Adaptation learning rate
        adapt_steps: Number of adaptation steps
        
    Returns:
        adapted_f_adapt: Updated f_adapt parameters
        losses: (adapt_steps,) array of adaptation losses
    """
    ttt_module = TTTModule(input_dim=projection_dim)
    
    # Project inputs (these stay connected to the graph)
    combined = jnp.concatenate([obs, actions], axis=-1)
    z_combined = _dense_project(combined, ttt_params['P_K'])
    z_input = z_combined

    # Target with stop_gradient
    z_target = jax.lax.stop_gradient(next_obs)
    
    def cosine_loss(f_params):
        pred = ttt_module.apply({'params': f_params}, z_input)
        return cosine_similarity_loss(pred, z_target)
    
    def step_fn(f_params, _):
        loss, grads = jax.value_and_grad(cosine_loss)(f_params)
        new_params = jax.tree_map(lambda p, g: p - adapt_lr * g, f_params, grads)
        return new_params, loss
    
    adapted_f_adapt, losses = jax.lax.scan(
        step_fn,
        ttt_params['f_adapt'],
        None,
        length=adapt_steps
    )
    
    return adapted_f_adapt, losses


def test_time_adapt(
    ttt_params: Dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    next_obs: jnp.ndarray,
    projection_dim: int,
    ttt_lr: float = 1e-2,
    ttt_steps: int = 5,
) -> Tuple[Dict, jnp.ndarray]:
    """
    Adapt TTT module at test time using next-state prediction.
    
    This adapts only f_adapt while keeping P_K, P_action, P_Q frozen.
    
    Args:
        ttt_params: TTT parameter tree
        obs: (B, octo_dim) current observations
        actions: (B, action_dim) actions taken
        next_obs: (B, octo_dim) next observations
        projection_dim: Projection dimension
        ttt_lr: Adaptation learning rate
        ttt_steps: Number of adaptation steps
        
    Returns:
        adapted_params: TTT params with updated f_adapt
        losses: (ttt_steps,) adaptation losses
    """
    # Project inputs (frozen via stop_gradient)
    combined = jnp.concatenate([obs, actions], axis=-1)
    z_combined = _dense_project(combined, ttt_params['P_K'])
    
    # Stop gradients - only adapt f_adapt
    z_input = jax.lax.stop_gradient(z_combined)
    z_target = jax.lax.stop_gradient(next_obs)
    
    # Adapt f_adapt with cosine similarity loss
    adapted_f_adapt, losses = ttt_adaptation_cosine(
        ttt_params['f_adapt'],
        z_input,
        z_target,
        projection_dim,
        ttt_lr,
        ttt_steps
    )
    
    # Return full params with updated f_adapt
    adapted_params = dict(ttt_params)
    adapted_params['f_adapt'] = adapted_f_adapt
    
    return adapted_params, losses


def sequential_test_time_adapt(
    ttt_params: Dict,
    obs_sequence: jnp.ndarray,      # (T, octo_dim)
    action_sequence: jnp.ndarray,   # (T, action_dim)
    projection_dim: int,
    ttt_lr: float = 1e-2,
    ttt_steps: int = 5,
    reset: bool = True,
) -> Tuple[jnp.ndarray, Dict, jnp.ndarray]:
    """
    Sequential TTT adaptation over a trajectory.
    
    For each timestep t:
    1. Use transition (obs[t-1], action[t-1]) → obs[t] to adapt f_adapt
    2. Extract features for obs[t] using (possibly adapted) f_adapt
    
    Args:
        ttt_params: TTT parameter tree
        obs_sequence: (T, octo_dim) observation sequence
        action_sequence: (T, action_dim) action sequence
        projection_dim: Projection dimension
        ttt_lr: Adaptation learning rate
        ttt_steps: Steps per adaptation
        reset: If True, reset to base f_adapt each step; if False, accumulate
        share_pk_pq: Whether P_Q shares weights with P_K
        
    Returns:
        adapted_features: (T, projection_dim) adapted features
        final_params: TTT params with final f_adapt state
        all_losses: (T, ttt_steps) adaptation losses
    """
    T = obs_sequence.shape[0]
    ttt_module = TTTModule(input_dim=projection_dim)
    
    base_f_adapt = ttt_params['f_adapt']
    
    # Precompute projections for efficiency
    # P_Q (or P_K if shared) projections for all obs (used for feature extraction)
    combined = jnp.concatenate([obs_sequence, action_sequence], axis=-1)
    query_all = _dense_project(combined, ttt_params['P_Q'])  # (T, proj_dim)

    # P_K projections for all obs (used for adaptation)
    z_obs_all = _dense_project(combined, ttt_params['P_K'])  # (T, proj_dim)
    
    # For adaptation: need (obs[t-1], action[t-1]) → obs[t]
    # z_input[t-1] = z_obs[t-1] + z_action[t-1], target[t-1] = z_obs[t]
    obs_prev = obs_sequence[:-1] if T > 1 else obs_sequence[:1]
    z_input_all = z_obs_all[:-1] if T > 1 else z_obs_all[:1]
    z_target_all = obs_sequence[1:] if T > 1 else obs_sequence[:1]
    
    def body_fn(carry, t):
        f_adapt_params = carry
        
        # For t=0: no previous transition, use base f_adapt
        # For t>0: adapt using transition (t-1) → t
        
        safe_idx = jnp.maximum(t - 1, 0)
        
        def adapt_and_extract(f_params):
            # Get input/target for this transition
            z_input_t = jax.lax.dynamic_index_in_dim(z_input_all, safe_idx, axis=0, keepdims=False)
            z_target_t = jax.lax.dynamic_index_in_dim(z_target_all, safe_idx, axis=0, keepdims=False)
            
            # Add batch dim for adaptation
            z_input_t = z_input_t[None, :]
            z_target_t = z_target_t[None, :]
            
            # Stop gradients for test-time adaptation
            z_input_t = jax.lax.stop_gradient(z_input_t)
            z_target_t = jax.lax.stop_gradient(z_target_t)
            
            # Adapt with cosine loss
            adapted, losses = ttt_adaptation_cosine(
                f_params,
                z_input_t,
                z_target_t,
                projection_dim,
                ttt_lr,
                ttt_steps
            )
            return adapted, losses
        
        def no_adapt(f_params):
            return f_params, jnp.zeros(ttt_steps)
        
        # Use base params if reset=True, otherwise use carried params
        start_params = base_f_adapt if reset else f_adapt_params
        
        # Adapt only if t > 0
        adapted_params, losses = jax.lax.cond(
            t > 0,
            lambda p: adapt_and_extract(p),
            lambda p: no_adapt(p),
            start_params
        )
        
        # Extract features for timestep t
        query_t = jax.lax.dynamic_index_in_dim(query_all, t, axis=0, keepdims=False)
        feature_t = ttt_module.apply({'params': adapted_params}, query_t[None, :])[0]
        
        # Carry forward (only matters if reset=False)
        next_carry = adapted_params
        
        return next_carry, (feature_t, losses)
    
    time_indices = jnp.arange(T)
    final_f_adapt, (features, all_losses) = jax.lax.scan(
        body_fn,
        base_f_adapt,
        time_indices
    )
    
    # Construct final params
    final_params = dict(ttt_params)
    final_params['f_adapt'] = final_f_adapt
    
    return features, final_params, all_losses


def get_adapted_features_for_trajectory(
    ttt_params: Dict,
    ttt_extractor: TTTPredictFeatureExtractor,
    obs_sequence: jnp.ndarray,      # (T, octo_dim)
    action_sequence: jnp.ndarray,   # (T, action_dim) 
    adapt: bool = True,
    reset: bool = True,
    ttt_lr: float = 1e-2,
    ttt_steps: int = 5,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Get adapted features for a trajectory, with optional TTT adaptation.
    
    Args:
        ttt_params: TTT parameter tree
        ttt_extractor: TTT module (for projection_dim and share_pk_pq)
        obs_sequence: (T, octo_dim) observations
        action_sequence: (T, action_dim) actions
        adapt: Whether to run TTT adaptation
        reset: Reset f_adapt each step (only matters if adapt=True)
        ttt_lr: Adaptation learning rate
        ttt_steps: Adaptation steps per transition
        
    Returns:
        features: (T, projection_dim) features
        params: (Possibly adapted) TTT params
    """
    projection_dim = ttt_extractor.projection_dim
    
    if adapt:
        features, adapted_params, _ = sequential_test_time_adapt(
            ttt_params,
            obs_sequence,
            action_sequence,
            projection_dim,
            ttt_lr=ttt_lr,
            ttt_steps=ttt_steps,
            reset=reset,
        )
        return features, adapted_params
    else:
        # No adaptation - just apply base f_adapt
        features = ttt_extractor.apply(
            {'params': ttt_params},
            obs_sequence,
            train=False
        )
        return features, ttt_params