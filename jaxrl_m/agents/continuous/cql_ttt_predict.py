"""
Unified CQL + SimSiam Agent.

This agent combines Conservative Q-Learning with EfficientZero-style
self-supervised learning in a single, unified training loop.

Key design:
- Encoder is a standalone module with its own optimizer
- All losses (actor, critic, simsiam) share the encoder and update it
- Combined loss: L_total = L_actor + L_critic + λ_sim * L_simsiam

Architecture:
                                    ┌──────────────┐
    observations ──► [Encoder] ──┬──► Actor Head   │  RL Branch
                                 │  ► Critic Head  │
                                 │                 │
                                 └──► Dynamics ────┤  SimSiam Branch
                                     Projector     │
                                     Predictor     │
                                    └──────────────┘

The encoder is a first-class module with its own optimizer, ensuring that
gradients from all losses (actor, critic, simsiam) properly update it.
"""

from functools import partial
from typing import Optional, Tuple, Dict, Any

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from overrides import overrides

from jaxrl_m.agents.continuous.cql import ContinuousCQLAgent, get_default_config as get_cql_default_config
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import *
from jaxrl_m.networks.actor_critic_nets import Critic_no_action, Policy, ensemblize
from jaxrl_m.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from jaxrl_m.networks.mlp import MLP

from jaxrl_m.networks.simsiam_networks import (
    SimSiamProjector,
    SimSiamPredictor,
    DynamicsNetworkFlat,
    DynamicsNetworkConv,
    cosine_similarity_loss,
)


class EncoderModule(nn.Module):
    """
    Standalone encoder wrapper module.

    This exists as a first-class module in the ModuleDict so it can have
    its own optimizer and receive gradients from all losses (actor, critic, simsiam).
    """
    encoder: nn.Module

    @nn.compact
    def __call__(self, observations):
        """Encode observations."""
        z = self.encoder(observations)
        # Handle dict outputs (e.g., from image encoders)
        if isinstance(z, dict):
            z_values = list(z.values())
            if len(z_values) == 1:
                z = z_values[0]
            else:
                z = jnp.concatenate(z_values, axis=-1)
        return z

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

def cosine_similarity_loss(
    pred: jnp.ndarray, 
    target: jnp.ndarray, 
    *, 
    normalize: bool = True,
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    Compute cosine similarity loss for SimSiam.
    
    Args:
        pred: Predicted embeddings (batch, dim)
        target: Target embeddings - should be stop_gradient (batch, dim)
        normalize: Whether to L2-normalize before computing similarity
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss (mean of 1 - cosine_similarity)
    """
    # Flatten if needed
    if pred.ndim > 2:
        pred = pred.reshape(pred.shape[0], -1)
    if target.ndim > 2:
        target = target.reshape(target.shape[0], -1)
    
    if normalize:
        pred = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + eps)
        target = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + eps)
    
    # Cosine similarity
    cos_sim = jnp.sum(pred * target, axis=-1)
    
    # Loss: 1 - cos_sim (range [0, 2])
    loss = jnp.mean(1.0 - cos_sim)
    
    return loss

class TTTModule(nn.Module):
    """
    Test-Time Training module.
    
    Architecture:
        Linear(input_dim -> 4*input_dim) -> LayerNorm -> GELU 
        -> Linear(4*input_dim -> input_dim) -> Residual (no final LayerNorm)
    """
    input_dim: int
    hidden_mult: float = 0.5
    
    @nn.compact
    def __call__(self, x):
        residual = x
        hidden_dim = int(self.hidden_mult * self.input_dim)
        
        # First linear layer
        z1 = nn.Dense(hidden_dim, name='linear1')(x)
        
        # LayerNorm on hidden
        z1 = nn.LayerNorm(name='ln_hidden')(z1)
        
        # GELU activation
        z1_gelu = nn.gelu(z1, approximate=True)
        
        # Second linear layer
        z2 = nn.Dense(self.input_dim, name='linear2')(z1_gelu)
        
        # Residual connection (no final LayerNorm)
        out = residual + z2
        
        return out


def gelu_bwd(x):
    """
    Backward pass of GELU activation (tanh approximation).
    
    Args:
        x: Input tensor
        
    Returns:
        Gradient of GELU
    """
    tanh_out = jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (1 - tanh_out ** 2) * (0.79788456 + 0.1070322243 * x ** 2) + 0.5 * (1 + tanh_out)

class TTTPredictModule(nn.Module):
    """
    TTT Dynamics Module: Predicts next state features.
    Following Section 2.2, it uses a projection theta_K to process input tokens[cite: 116].
    The hidden state W is updated via gradient descent on the prediction loss[cite: 98].
    """
    feature_dim: int
    action_dim: int 
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

        _ = self.P_K(z_input)

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
    
    def predict_next(self, z, action):
        # 将 P_K 和 f_adapt 封装在一起
        combined = jnp.concatenate([z, action], axis=-1)
        proj = self.P_K(combined)
        return self.f_adapt(proj)

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

class CQLTTTPredictAgent(ContinuousCQLAgent):
    """
    CQL agent with integrated EfficientZero-style SimSiam self-supervised learning.
    
    The key insight from EfficientZero is that self-supervised losses help the
    representation learning, especially in limited data regimes. This agent
    computes both RL losses (actor, critic, CQL) and SimSiam losses in a
    unified manner.
    
    Training Flow:
    1. Encode observations using shared encoder
    2. Compute RL losses (actor, critic, CQL)
    3. Compute SimSiam loss using dynamics prediction
    4. Combine losses: L = L_rl + λ_sim * L_simsiam
    5. Single gradient update for all parameters
    """
    
    def _encode(self, obs, params):
        """
        Encode observations using the standalone encoder module.

        All losses (actor, critic, simsiam) should use this method to ensure
        gradients flow back through the shared encoder (modules_encoder).
        """
        variables = {"params": params}
        if self.state.batch_stats is not None:
            variables["batch_stats"] = self.state.batch_stats

        z = self.state.apply_fn(variables, obs, name="encoder")
        return z

    def encoder_loss_fn(
        self,
        batch: Batch,
        params: Params,
        rng: PRNGKey,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Placeholder encoder loss function.

        Returns 0 - the encoder receives gradients from actor/critic/simsiam losses
        which all encode using _encode() and compute gradients w.r.t. modules_encoder.
        This loss function exists to satisfy the optimizer structure.
        """
        return jnp.array(0.0), {"encoder/loss": jnp.array(0.0)}

    def ttt_loss_fn(
        self,
        batch: Batch,
        params: Params,
        rng: PRNGKey,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute EfficientZero-style SimSiam loss.

        Flow:
        1. Encode observations using standalone encoder (modules_encoder)
        2. Pass latents to SimSiam module (dynamics/projector/predictor)
        3. Compute cosine similarity loss

        Gradients flow back through modules_encoder, which is updated by
        the encoder optimizer with accumulated gradients from all losses.
        """
        obs = self._include_goals_in_obs(batch, "observations")
        next_obs = self._include_goals_in_obs(batch, "next_observations")
        actions = batch["actions"]

        # Build variables dict for apply
        variables = {"params": params}
        uses_batch_norm = self.config.get("uses_batch_norm", False)
        if uses_batch_norm and self.state.batch_stats is not None:
            variables["batch_stats"] = self.state.batch_stats

        # Step 1: Encode using standalone encoder module
        # Gradients flow back through modules_encoder
        z_t = self._encode(obs, params)
        z_tp1 = self._encode(next_obs, params)
        
        # ttt_params = params["ttt_module"]

        # combined = jnp.concatenate([z_t, actions], axis=-1)
        # z_input = _dense_project(combined, ttt_params['P_K'])

        # Step 2: Forward pass through TTT module (dynamics/projector/predictor)
        pred = self.state.apply_fn(
                variables,
                z_t,
                actions,
                method=lambda m, z, a: m.modules["ttt_module"].predict_next(z, a)
            )

        # Stop gradient on target (standard SimSiam)
        if self.config.get("stop_grad_target", True):
            target = jax.lax.stop_gradient(z_tp1)
        else:
            target = z_tp1

        # Cosine similarity loss
        loss = cosine_similarity_loss(pred, target)

        # Scale by lambda_ttt
        lambda_ttt = self.config.get("lambda_ttt", 2.0)
        scaled_loss = lambda_ttt * loss

        # Compute auxiliary metrics
        eps = 1e-8
        pred_norm_vec = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + eps)
        target_norm_vec = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + eps)
        cosine_sim = jnp.sum(pred_norm_vec * target_norm_vec, axis=-1).mean()

        # 2. 构建新的 info 字典
        info = {
            "ttt/loss": loss,                 # 原始 Loss (1 - cos_sim)
            "ttt/scaled_loss": scaled_loss,   # 乘以 lambda 后的 Loss
            "ttt/cosine_sim": cosine_sim,     # 实际相似度 (越接近 1 预测越准)
            "ttt/z_t_norm": jnp.linalg.norm(z_t, axis=-1).mean(),
            "ttt/z_tp1_norm": jnp.linalg.norm(z_tp1, axis=-1).mean(),
            "ttt/pred_norm": jnp.linalg.norm(pred, axis=-1).mean(),
        }

        return scaled_loss, info
    
    def inner_loop_fn(self, params, observations, actions, next_observations):
        """
        针对单个 Transition (或单个小 Batch) 的内循环适配
        """
        z_t = self._encode(observations, params)
        z_tp1 = self._encode(next_observations, params)
        
        def loss_fn(f_adapt_params):
            temp_ttt_params = params['ttt_module'].copy({'f_adapt': f_adapt_params})
            
            # P_K -> f_adapt -> prediction
            combined = jnp.concatenate([z_t, actions], axis=-1)
            z_input = _dense_project(combined, temp_ttt_params['P_K'])
            
            # 注意：这里需要确保 f_adapt 的调用是纯函数
            pred = self.state.apply_fn(
                {'params': params.copy({'ttt_module': temp_ttt_params})},
                z_input,
                method=lambda m, x: m.modules['ttt_module'].f_adapt(x)
            )
            
            target = jax.lax.stop_gradient(z_tp1)
            return cosine_similarity_loss(pred, target)

        grads = jax.grad(loss_fn)(params['ttt_module']['f_adapt'])
        
        inner_lr = self.config.get("ttt_inner_lr", 1e-3)
        new_f_params = jax.tree_util.tree_map(
            lambda p, g: p - inner_lr * g, 
            params['ttt_module']['f_adapt'], 
            grads
        )
        
        return new_f_params

    @overrides
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey, train=True):
        """
        CQL + TTT Critic Loss:
        """
        batch_size = batch["rewards"].shape[0]
        
        rng, target_rng = jax.random.split(rng)
        next_obs = self._include_goals_in_obs(batch, "next_observations")
        rng, next_action_key = jax.random.split(target_rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(batch, next_action_key)
        
        target_next_qs = self.forward_target_critic(next_obs, next_actions, rng=rng)
        target_next_min_q = target_next_qs.min(axis=0)
        target_next_min_q = self._process_target_next_qs(target_next_min_q, next_actions_log_probs)
        
        target_q_batch = batch["rewards"] + self.config["discount"] * batch["masks"] * target_next_min_q
        # (batch_size, ) -> (critic_ensemble_size, batch_size)
        target_qs_batch = target_q_batch[None].repeat(self.config.critic_ensemble_size, axis=0)

        def single_sample_loss_fn(obs, action, next_obs, target_qs, mc_return, sample_rng):
            """
            针对单个 Transition 运行：适配 -> 计算 Q -> 计算 CQL 差异
            """
            # inner update
            adapted_f_params = self.inner_loop_fn(params, obs, action, next_obs)
            
            sample_ttt_params = params['ttt_module'].copy({'f_adapt': adapted_f_params})
            sample_params = params.copy({'ttt_module': sample_ttt_params})
            
            current_q = self.forward_critic(
                obs, action, rng=sample_rng, 
                grad_params=sample_params, train=train
            )
            
            cql_q_diff, cql_extras = self._get_single_cql_q_diff(obs, next_obs, action, mc_return, sample_rng, sample_params)
            
            td_loss = jnp.mean((current_q - target_qs) ** 2)
            
            return td_loss, cql_q_diff, current_q, cql_extras

        rng, vmap_rng = jax.random.split(rng)
        vmap_keys = jax.random.split(vmap_rng, batch_size)

        vmapped_fn = jax.vmap(
            single_sample_loss_fn,
            in_axes=(0, 0, 0, 1, 0) 
        )
        
        obs_batch = self._include_goals_in_obs(batch, "observations")
        td_losses, cql_q_diffs, current_qs, cql_extras_batch = vmapped_fn(
            obs_batch, 
            batch["actions"], 
            next_obs,
            target_qs_batch,
            batch["mc_returns"],
            vmap_keys
        )

        mean_td_loss = td_losses.mean()
        mean_cql_q_diff = cql_q_diffs.mean()

        if self.config["cql_autotune_alpha"]:
            alpha = self.forward_cql_alpha_lagrange()
            cql_loss = (mean_cql_q_diff - self.config["cql_target_action_gap"])
        else:
            alpha = self.config["cql_alpha"]
            cql_loss = jnp.clip(
                mean_cql_q_diff,
                self.config["cql_clip_diff_min"],
                self.config["cql_clip_diff_max"],
            )

        total_critic_loss = mean_td_loss + alpha * cql_loss

        info = {
            "critic_loss": total_critic_loss,
            "td_err": mean_td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": alpha,
            "cql_diff": mean_cql_q_diff,
            "online_q": current_qs.mean(),
            "target_q": target_q_batch.mean(),
        }
        
        return total_critic_loss, info
    
    def _get_single_cql_q_diff(
        self, obs: Data, next_obs: Data, action: jnp.ndarray, mc_return: jnp.ndarray, rng: PRNGKey, grad_params: Params
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

        q_pred = self.forward_critic(
            obs, 
            action,
            rng, 
            grad_params=grad_params,
            train=True
        ) 

        action_dim = self.config["action_dim"] # 假设配置中有
        rng, action_rng = jax.random.split(rng)
        num_samples = self.config["cql_n_actions"]

        cql_random_actions = jax.random.uniform(
            action_rng, shape=(num_samples, action_dim), minval=-1.0, maxval=1.0
        )

        rng, cur_rng, next_rng = jax.random.split(rng, 3)
        
        cql_current_actions, cql_current_log_pis = self.forward_policy_and_sample(
            obs, cur_rng, repeat=num_samples, grad_params=grad_params
        )
        cql_next_actions, cql_next_log_pis = self.forward_policy_and_sample(
            next_obs, next_rng, repeat=num_samples, grad_params=grad_params
        )

        all_sampled_actions = jnp.concatenate([
            cql_random_actions,
            cql_current_actions,
            cql_next_actions,
        ], axis=0)

        rng, q_rng = jax.random.split(rng)
        
        cql_q_samples = self.forward_critic(
            obs, 
            all_sampled_actions, 
            q_rng, 
            grad_params=grad_params,
            train=True
        )

        if self.config["use_calql"]:
            cql_q_pi = cql_q_samples[:, num_samples:]
            
            calql_bound_rate = jnp.mean(cql_q_pi < mc_return)
            cql_q_pi_calibrated = jnp.maximum(cql_q_pi, mc_return)

            cql_q_samples = jnp.concatenate([
                cql_q_samples[:, :num_samples],
                cql_q_pi_calibrated
            ], axis=-1)

        if self.config["cql_importance_sample"]:
            random_density = jnp.log(0.5**action_dim)
            importance_prob = jnp.concatenate([
                jnp.full((num_samples,), random_density),
                cql_current_log_pis,
                cql_next_log_pis
            ], axis=0)
            cql_q_samples = cql_q_samples - importance_prob
        else:
            cql_q_samples = jnp.concatenate([
                cql_q_samples, 
                jnp.expand_dims(q_pred, -1)
            ], axis=-1)
            cql_q_samples -= jnp.log(cql_q_samples.shape[-1]) * self.config["cql_temp"]

        cql_ood_values = (
            jax.scipy.special.logsumexp(
                cql_q_samples / self.config["cql_temp"], axis=-1
            ) * self.config["cql_temp"]
        )

        cql_q_diff = cql_ood_values - q_pred
        
        info = {"ood_val": cql_ood_values.mean()}
        if self.config["use_calql"]:
            info["calql_bound_rate"] = calql_bound_rate
            
        return cql_q_diff, info
        
    @overrides
    def forward_policy(
        self,
        observations,
        rng=None,
        *,
        grad_params=None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy: encode first, then call actor head.
        """
        params = grad_params or self.state.params
        # Encode observations using standalone encoder
        z = self._encode(observations, params)
        # Forward through actor head (which expects latents)
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": params},
            z,  # Pass latents, not observations
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    @overrides
    def forward_policy_and_sample(
        self,
        obs: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        repeat=None,
    ):
        """
        Override to ensure encoding happens before actor forward pass.

        CQL's forward_policy_and_sample calls super().forward_policy() which
        bypasses our forward_policy override. We need to call self.forward_policy
        instead to ensure observations are encoded first.
        """
        rng, sample_rng = jax.random.split(rng)
        # Use self.forward_policy (not super) to ensure encoding happens
        action_dist = self.forward_policy(obs, rng, grad_params=grad_params)
        if repeat:
            new_actions, log_pi = action_dist.sample_and_log_prob(
                seed=sample_rng, sample_shape=repeat
            )
            new_actions = jnp.transpose(
                new_actions, (1, 0, 2)
            )  # (batch, repeat, action_dim)
            log_pi = jnp.transpose(log_pi, (1, 0))  # (batch, repeat)
        else:
            new_actions, log_pi = action_dist.sample_and_log_prob(seed=sample_rng)
        return new_actions, log_pi

    @overrides
    def forward_critic(
        self,
        observations,
        actions,
        rng=None,
        *,
        grad_params=None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic: encode first, then call critic head.
        """
        params = grad_params or self.state.params
        # Encode observations using standalone encoder
        z = self._encode(observations, params)
        # Forward through critic head (which expects latents)
        if actions.ndim == 3:
            def single_action_q_pass(one_sample_action):
                # one_sample_action 维度是 (B, action_dim)
                
                z_adapted = self.state.apply_fn(
                    {"params": params},
                    z, 
                    one_sample_action,
                    name="ttt_module"
                )
                
                return self.state.apply_fn(
                    {"params": params},
                    z_adapted,
                    one_sample_action,
                    name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                )

            return jax.vmap(
                single_action_q_pass,
                in_axes=1,
                out_axes=-1,
            )(actions)
        
        else:
            z_adapted = self.state.apply_fn(
                {"params": params},
                z, 
                actions,
                name="ttt_module"
            )
            return self.state.apply_fn(
                {"params": params},
                z_adapted,
                actions,
                name="critic",
                rngs={"dropout": rng} if train else {},
                train=train,
            )

    @overrides
    def loss_fns(self, batch):
        """
        Return all loss functions including encoder and SimSiam.

        The encoder loss returns 0 - it receives gradients from actor/critic/simsiam
        losses which all use _encode() and compute gradients w.r.t. modules_encoder.
        """
        losses = super().loss_fns(batch)

        # Add encoder loss (placeholder - receives gradients from other losses)
        losses["encoder"] = partial(self.encoder_loss_fn, batch)

        # Add TTT loss if enabled
        if self.config.get("lambda_ttt", 0.0) > 0:
            losses["ttt_module"] = partial(self.ttt_loss_fn, batch)

        return losses

    @overrides
    def update(
        self,
        batch: Batch,
        pmap_axis: str = None,
        networks_to_update: set = None,
    ):
        """
        Update all networks including encoder and SimSiam components.

        Also handles batch normalization statistics updates if batch norm is used.
        """
        if networks_to_update is None:
            networks_to_update = {"encoder", "actor", "critic"}
        else:
            networks_to_update = set(networks_to_update)
            networks_to_update.add("encoder")  # Always update encoder

        # Add TTT if loss is enabled
        if self.config.get("lambda_ttt", 0.0) > 0:
            networks_to_update.add("ttt_module")

        # Pass as regular set - parent class will add temperature/cql_alpha as needed
        new_agent, info = super().update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=networks_to_update,
        )

        return new_agent, info
    
    def get_encoder_features(
        self,
        observations: Data,
        goals: Optional[Data] = None,
        *,
        params: Optional[Params] = None,
    ) -> jnp.ndarray:
        """Get encoder features for observations (for TTT adaptation)."""
        if params is None:
            params = self.state.params

        obs_input = (observations, goals) if goals is not None else observations

        # Use the standalone encoder module
        return self._encode(obs_input, params)
    
    def compute_ttt_loss(
        self,
        batch: Batch,
        f_adapt_params: Params,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute TTT adaptation loss (TTT) with only f adapt gradients.

        Returns:
            loss: Scalar TTT loss
            grads: Gradients w.r.t. ttt parameters only
        """
        def loss_fn(f_params):
            current_ttt_params = self.state.params["ttt_module"].copy({
                "f_adapt": f_params
            })
            full_params = self.state.params.copy({
                "ttt_module": current_ttt_params
            })

            loss, _ = self.ttt_loss_fn(
                batch,
                full_params,
                jax.random.PRNGKey(0),
                train=False, 
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(f_adapt_params)
        return loss, grads
    
    @jax.jit
    @overrides
    def get_debug_metrics(self, batch, **kwargs):
        """Extended debug metrics including SimSiam info."""
        # Encode observations first (don't call super() which bypasses encoding)
        obs = self._include_goals_in_obs(batch, "observations")
        z = self._encode(obs, self.state.params)

        # Get actor distribution using encoded latents
        dist = self.state.apply_fn(
            {"params": self.state.params},
            z,
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        # Get critic metrics
        _, critic_metrics = self.critic_loss_fn(
            batch, self.state.params, rng=jax.random.PRNGKey(0), train=False
        )

        metrics = {
            "log_probs": log_probs,
            "mse": mse,
            "pi_actions": pi_actions,
            **critic_metrics,
        }

        # Add TTT metrics if enabled
        if self.config.get("lambda_ttt", 0.0) > 0:
            _, ttt_info = self.ttt_loss_fn(
                batch,
                self.state.params,
                jax.random.PRNGKey(0),
                train=False,
            )
            metrics.update(ttt_info)

        return metrics
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,  # Should be True for SimSiam to work
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        # Goals
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        # SimSiam config
        ttt_kwargs: dict = {},
        **kwargs,
    ):
        """
        Create a CQL+SimSiam agent.
        
        Args:
            rng: Random key
            observations: Example observations
            actions: Example actions
            encoder_def: Encoder module
            shared_encoder: Must be True for SimSiam (encoder shared across all)
            critic_network_kwargs: Critic MLP config
            policy_network_kwargs: Policy MLP config
            policy_kwargs: Policy distribution config
            goals: Example goals
            early_goal_concat: Concat goals early in encoder
            shared_goal_encoder: Share goal encoder
            simsiam_kwargs: SimSiam config:
                - projector_hidden_dims: tuple (default: (512, 512, 512))
                - projector_output_dim: int (default: 512)
                - projector_norm: str (default: "layer")
                - predictor_hidden_dims: tuple (default: (256,))
                - predictor_output_dim: int (default: 512)
                - predictor_norm: str (default: "layer")
                - dynamics_hidden_dim: int (default: 512)
                - dynamics_num_residual_blocks: int (default: 1)
                - dynamics_norm: str (default: "layer")
                - use_conv_dynamics: bool (default: False)
            **kwargs: CQL config overrides
        """
        assert shared_encoder, "CQLSimSiam requires shared_encoder=True"

        config = get_default_config(updates=kwargs)

        if config.language_conditioned:
            assert config.goal_conditioned

        # Create wrapped encoder
        encoder_def_wrapped = cls._create_encoder_def(
            encoder_def,
            use_proprio=False,
            enable_stacking=False,
            goal_conditioned=config.goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
            language_conditioned=config.language_conditioned,
        )

        # Probe to get latent dimension
        rng, probe_rng = jax.random.split(rng)
        network_input = (observations, goals) if config.goal_conditioned else observations
        dummy_params = encoder_def_wrapped.init(probe_rng, network_input)
        latent_example = encoder_def_wrapped.apply(dummy_params, network_input)
        # Handle dict outputs from encoder
        if isinstance(latent_example, dict):
            latent_values = list(latent_example.values())
            if len(latent_values) == 1:
                latent_example = latent_values[0]
            else:
                latent_example = jnp.concatenate(latent_values, axis=-1)
        latent_dim = latent_example.shape[-1]
        action_dim = actions.shape[-1]

        # =======================================================================
        # Create standalone encoder module (first-class with its own optimizer)
        # =======================================================================
        encoder_module = EncoderModule(
            encoder=encoder_def_wrapped,
            name="encoder",
        )

        # =======================================================================
        # Create actor/critic heads WITHOUT encoders (they receive latents)
        # =======================================================================
        policy_def = Policy(
            encoder=None,  # No encoder - receives latents
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic_no_action, encoder=None, network=critic_backbone  # No encoder - receives latents
        )(name="critic")

        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )

        # =======================================================================
        # Create TTT sub-networks (receives latents from encoder)
        # =======================================================================
        ttt_cfg = {
            "feature_dim": 512,
            "action_dim": 7,
            "projection_hidden_dim": 512,
            "projection_num_layers": 2,
        }
        ttt_cfg.update(ttt_kwargs)

        # projector_def = SimSiamProjector(
        #     hidden_dims=tuple(sim_cfg["projector_hidden_dims"]),
        #     output_dim=sim_cfg["projector_output_dim"],
        #     norm_type=sim_cfg["projector_norm"],
        # )

        # predictor_def = SimSiamPredictor(
        #     hidden_dims=tuple(sim_cfg["predictor_hidden_dims"]),
        #     output_dim=sim_cfg["predictor_output_dim"],
        #     norm_type=sim_cfg["predictor_norm"],
        # )

        # if sim_cfg["use_conv_dynamics"]:
        #     dynamics_def = DynamicsNetworkConv(
        #         latent_channels=sim_cfg["dynamics_latent_channels"],
        #         action_dim=action_dim,
        #         num_residual_blocks=sim_cfg["dynamics_num_residual_blocks"],
        #         norm_type=sim_cfg["dynamics_norm"],
        #     )
        # else:
        #     dynamics_def = DynamicsNetworkFlat(
        #         latent_dim=latent_dim,
        #         action_dim=action_dim,
        #         hidden_dim=sim_cfg["dynamics_hidden_dim"],
        #         num_residual_blocks=sim_cfg["dynamics_num_residual_blocks"],
        #         norm_type=sim_cfg["dynamics_norm"],
        #     )

        ttt_module = TTTPredictModule(
            feature_dim=ttt_cfg["feature_dim"],
            action_dim=ttt_cfg["action_dim"],
            projection_hidden_dim=ttt_cfg["projection_hidden_dim"],
            projection_num_layers=ttt_cfg["projection_num_layers"],
            name="ttt_module",
        )

        # =======================================================================
        # Build complete network dict with encoder as first-class module
        # =======================================================================
        networks = {
            "encoder": encoder_module,  # Standalone encoder with its own optimizer
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
            "ttt_module": ttt_module,
        }

        if config["cql_autotune_alpha"]:
            networks["cql_alpha_lagrange"] = LeqLagrangeMultiplier(
                init_value=config.cql_alpha_lagrange_init,
                constraint_shape=(),
                name="cql_alpha_lagrange",
            )

        model_def = ModuleDict(networks)

        # =======================================================================
        # Optimizers - encoder has its own optimizer!
        # =======================================================================
        txs = {
            "encoder": make_optimizer(**config.encoder_optimizer_kwargs),
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
            "ttt_module": make_optimizer(**config.ttt_module_optimizer_kwargs),
        }

        if config["cql_autotune_alpha"]:
            txs["cql_alpha_lagrange"] = make_optimizer(
                **config.cql_alpha_lagrange_otpimizer_kwargs
            )

        # Initialize parameters
        rng, init_rng = jax.random.split(rng)

        # Build initialization inputs
        # - encoder takes observations
        # - actor/critic take latents
        # - simsiam takes (z_t, z_tp1, actions, train)
        dummy_latent = latent_example
        init_kwargs = {
            "encoder": [network_input],
            "actor": [dummy_latent],  # Latent input, not observations
            "critic": [dummy_latent, actions],  # Latent input, not observations
            "temperature": [],
            "ttt_module": [dummy_latent, actions, True],
        }

        if config["cql_autotune_alpha"]:
            init_kwargs["cql_alpha_lagrange"] = []

        params = model_def.init(init_rng, **init_kwargs)["params"]
        batch_stats = None

        # Create train state
        rng, state_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=state_rng,
            batch_stats=batch_stats,
        )
        
        # Finalize config
        if config.target_entropy >= 0.0:
            config.target_entropy = -actions.shape[-1]
        config = flax.core.FrozenDict(config)
        
        return cls(state, config)


def get_default_config(updates=None):
    """Get default config for CQL+SimSiam agent."""
    # Start with CQL defaults
    config = get_cql_default_config()

    # Add SimSiam-specific defaults
    config.lambda_ttt = 2.0  # EfficientZero uses λ_3 = 2
    config.stop_grad_target = True
    config.normalize_latents = True

    # Encoder optimizer (shared encoder updated by all losses)
    config.encoder_optimizer_kwargs = ConfigDict({
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
    })

    # SimSiam optimizer (for dynamics/projector/predictor only)
    config.ttt_module_optimizer_kwargs = ConfigDict({
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
    })

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config