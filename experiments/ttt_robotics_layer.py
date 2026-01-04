"""
TTT Layer for Robotics - Action-Conditioned Version

This version adapts the 2024 TTT paper for robotics by:
1. Conditioning the SSL task on (obs, action) to predict next_obs
2. Keeping the learned projections and input-dependent LR from the paper
3. Maintaining the residual + LayerNorm inner model for stability

Two modes:
- "reconstruction": Paper-style multi-view reconstruction of obs (no actions)
- "prediction": Predict next_obs from (obs, action) - better for robotics
"""

from typing import Optional, Tuple, Literal
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class TTTRoboticsLayer(nn.Module):
    """
    TTT Layer adapted for robotics with (obs, action, next_obs) transitions.
    
    SSL Modes:
    - "reconstruction": f(θ_K @ obs; W) → θ_V @ obs (paper-style)
    - "prediction": f(θ_K @ [obs, action]; W) → θ_V @ next_obs (robotics)
    
    The key insight: the SSL task should be non-trivial but learnable.
    For robotics, predicting next_obs from (obs, action) is a natural task.
    """
    obs_dim: int                    # OCTO embedding dimension (384 or 512)
    action_dim: int                 # Robot action dimension (e.g., 7)
    bottleneck_dim: int = 64        # MUST be << obs_dim to prevent collapse
    output_dim: int = 256           # Feature dim for RL agent
    
    ssl_mode: Literal["reconstruction", "prediction"] = "prediction"
    eta_base: float = 1.0           # Inner loop learning rate
    use_input_dependent_lr: bool = True
    
    def setup(self):
        # Input dimension depends on SSL mode
        if self.ssl_mode == "reconstruction":
            ssl_input_dim = self.obs_dim
            ssl_target_dim = self.obs_dim
        else:  # prediction
            ssl_input_dim = self.obs_dim + self.action_dim
            ssl_target_dim = self.obs_dim  # predicting next_obs
        
        # Projections for SSL task
        # θ_K: projects SSL input to bottleneck (training view)
        self.proj_K = nn.Dense(self.bottleneck_dim, name="proj_K")
        # θ_V: projects SSL target to bottleneck (label view)
        self.proj_V = nn.Dense(self.bottleneck_dim, name="proj_V")
        
        # θ_Q: projects obs to bottleneck for output features
        # Note: output always depends on obs only (not action)
        self.proj_Q = nn.Dense(self.bottleneck_dim, name="proj_Q")
        
        # Output projection
        self.proj_out = nn.Dense(self.output_dim, name="proj_out")
        
        # Inner model components
        self.inner_ln = nn.LayerNorm(name="inner_ln")
        
        # Input-dependent learning rate
        if self.use_input_dependent_lr:
            # LR depends on SSL input (obs or obs+action)
            self.lr_dense = nn.Dense(1, name="lr_dense")
    
    @nn.compact
    def get_initial_W(self):
        """Learnable initial hidden state W₀."""
        W_init = self.param(
            "W_init",
            nn.initializers.normal(stddev=0.01),
            (self.bottleneck_dim, self.bottleneck_dim)
        )
        return W_init
    
    def inner_model(self, x: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
        """
        f(x; W) = x + LayerNorm(W @ x)
        
        Residual + LayerNorm is ESSENTIAL for stability.
        """
        Wx = x @ W.T
        return x + self.inner_ln(Wx)
    
    def get_ssl_input_and_target(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get SSL input and target based on mode.
        
        Args:
            obs: (batch, obs_dim) current observation embedding
            action: (batch, action_dim) action taken
            next_obs: (batch, obs_dim) next observation embedding
            
        Returns:
            ssl_input: input to θ_K projection
            ssl_target: target for θ_V projection
        """
        if self.ssl_mode == "reconstruction":
            # Paper-style: reconstruct obs from obs
            ssl_input = obs
            ssl_target = obs
        else:  # prediction
            # Robotics: predict next_obs from (obs, action)
            ssl_input = jnp.concatenate([obs, action], axis=-1)
            ssl_target = next_obs
        
        return ssl_input, ssl_target
    
    def compute_ssl_loss(
        self,
        ssl_input: jnp.ndarray,
        ssl_target: jnp.ndarray,
        W: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        SSL loss: ||f(θ_K @ input; W) - θ_V @ target||²
        """
        k = self.proj_K(ssl_input)   # (batch, bottleneck_dim)
        v = self.proj_V(ssl_target)  # (batch, bottleneck_dim)
        pred = self.inner_model(k, W)
        return jnp.mean((pred - v) ** 2)
    
    def compute_ssl_loss_per_sample(
        self,
        ssl_input: jnp.ndarray,
        ssl_target: jnp.ndarray,
        W: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-sample SSL loss for monitoring."""
        k = self.proj_K(ssl_input)
        v = self.proj_V(ssl_target)
        pred = self.inner_model(k, W)
        return jnp.mean((pred - v) ** 2, axis=-1)  # (batch,)
    
    def get_learning_rate(self, ssl_input: jnp.ndarray) -> jnp.ndarray:
        """Input-dependent learning rate: η(x) = η_base * σ(θ_lr @ x)"""
        if self.use_input_dependent_lr:
            lr_logit = self.lr_dense(ssl_input)
            lr_scale = jax.nn.sigmoid(lr_logit.squeeze(-1))
            # For batch input, take mean across batch
            if lr_scale.ndim > 0:
                lr_scale = jnp.mean(lr_scale)
            return self.eta_base * lr_scale
        return self.eta_base
    
    def forward_batch(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        adapt: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for a batch of transitions.
        
        Args:
            obs: (batch, obs_dim) current observation embeddings
            action: (batch, action_dim) actions
            next_obs: (batch, obs_dim) next observation embeddings
            adapt: whether to perform TTT adaptation
            
        Returns:
            features: (batch, output_dim) adapted features for RL
            W_adapted: adapted hidden state
            ssl_loss_before: SSL loss before adaptation
            ssl_loss_after: SSL loss after adaptation
        """
        W0 = self.get_initial_W()
        
        # Get SSL input and target based on mode
        ssl_input, ssl_target = self.get_ssl_input_and_target(obs, action, next_obs)
        
        # SSL loss before adaptation
        ssl_loss_before = self.compute_ssl_loss(ssl_input, ssl_target, W0)
        
        if adapt:
            # Compute gradient of SSL loss w.r.t. W
            def loss_fn(W):
                return self.compute_ssl_loss(ssl_input, ssl_target, W)
            
            grad_W = jax.grad(loss_fn)(W0)
            
            # Input-dependent learning rate
            eta = self.get_learning_rate(ssl_input)
            
            # Update W
            W_adapted = W0 - eta * grad_W
        else:
            W_adapted = W0
        
        # SSL loss after adaptation
        ssl_loss_after = self.compute_ssl_loss(ssl_input, ssl_target, W_adapted)
        
        # Output features (always based on obs, not obs+action)
        q = self.proj_Q(obs)
        h = self.inner_model(q, W_adapted)
        features = self.proj_out(h)
        
        return features, W_adapted, ssl_loss_before, ssl_loss_after
    
    def forward_features_only(
        self,
        obs: jnp.ndarray,
        W: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Get features without computing SSL loss.
        For inference when we already have W.
        """
        if W is None:
            W = self.get_initial_W()
        
        q = self.proj_Q(obs)
        h = self.inner_model(q, W)
        return self.proj_out(h)


class TTTRoboticsAdapter(nn.Module):
    """
    High-level adapter wrapping TTTRoboticsLayer.
    
    Handles:
    - Initialization
    - Training vs inference modes
    - Sequential processing for trajectories
    """
    obs_dim: int
    action_dim: int
    bottleneck_dim: int = 64
    output_dim: int = 256
    ssl_mode: Literal["reconstruction", "prediction"] = "prediction"
    eta_base: float = 1.0
    use_input_dependent_lr: bool = True
    
    def setup(self):
        self.ttt_layer = TTTRoboticsLayer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            bottleneck_dim=self.bottleneck_dim,
            output_dim=self.output_dim,
            ssl_mode=self.ssl_mode,
            eta_base=self.eta_base,
            use_input_dependent_lr=self.use_input_dependent_lr,
            name="ttt_layer"
        )
    
    def get_initial_W(self):
        return self.ttt_layer.get_initial_W()
    
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        adapt: bool = True,
        train: bool = True,
    ):
        """
        Main forward pass for training.
        
        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)  
            next_obs: (batch, obs_dim)
            adapt: whether to do TTT adaptation
            train: training mode flag
        """
        return self.ttt_layer.forward_batch(obs, action, next_obs, adapt=adapt)
    
    def get_features(
        self,
        obs: jnp.ndarray,
        W: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Get features for inference (no SSL computation)."""
        return self.ttt_layer.forward_features_only(obs, W)


# ============================================================================
# Sequential Processing for Trajectories
# ============================================================================

def process_trajectory_with_ttt(
    ttt_adapter: TTTRoboticsAdapter,
    params: dict,
    obs_sequence: jnp.ndarray,      # (T, obs_dim)
    action_sequence: jnp.ndarray,   # (T, action_dim) 
    reset_each_step: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Process a trajectory with TTT adaptation.
    
    For robotics trajectories, we adapt W sequentially using each
    (obs_t, action_t, obs_{t+1}) transition.
    
    Args:
        ttt_adapter: TTT adapter module
        params: module parameters
        obs_sequence: (T, obs_dim) observation embeddings
        action_sequence: (T, action_dim) actions
        reset_each_step: reset W to W₀ at each step (True) or cumulative (False)
        
    Returns:
        features: (T, output_dim) adapted features
        ssl_losses: (T,) SSL loss at each step
        W_final: final hidden state
    """
    T = obs_sequence.shape[0]
    
    def step_fn(carry, inputs):
        W, t = carry
        obs_t = inputs['obs']
        action_t = inputs['action']
        
        # Next observation (for SSL target)
        # At last step, use same obs as target
        next_obs_t = jnp.where(
            t < T - 1,
            obs_sequence[t + 1],
            obs_t
        )
        
        if reset_each_step:
            W = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_W)
        
        # Get SSL input/target
        ssl_input, ssl_target = ttt_adapter.apply(
            {'params': params},
            obs_t[None], action_t[None], next_obs_t[None],
            method=lambda m, o, a, n: m.ttt_layer.get_ssl_input_and_target(o, a, n)
        )
        ssl_input = ssl_input[0]
        ssl_target = ssl_target[0]
        
        # Compute SSL loss and adapt
        def loss_fn(W_):
            k = ttt_adapter.apply(
                {'params': params},
                ssl_input[None],
                method=lambda m, x: m.ttt_layer.proj_K(x)
            )[0]
            v = ttt_adapter.apply(
                {'params': params},
                ssl_target[None],
                method=lambda m, x: m.ttt_layer.proj_V(x)
            )[0]
            pred = ttt_adapter.apply(
                {'params': params},
                k, W_,
                method=lambda m, x, w: m.ttt_layer.inner_model(x, w)
            )
            return jnp.mean((pred - v) ** 2)
        
        ssl_loss = loss_fn(W)
        grad_W = jax.grad(loss_fn)(W)
        
        # Get learning rate
        eta = ttt_adapter.apply(
            {'params': params},
            ssl_input[None],
            method=lambda m, x: m.ttt_layer.get_learning_rate(x)
        )
        
        W_new = W - eta * grad_W
        
        # Get output features
        features_t = ttt_adapter.apply(
            {'params': params},
            obs_t[None], W_new,
            method=ttt_adapter.get_features
        )[0]
        
        return (W_new, t + 1), (features_t, ssl_loss)
    
    # Initial state
    W0 = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_W)
    
    # Prepare inputs for scan
    inputs = {
        'obs': obs_sequence,
        'action': action_sequence,
    }
    
    # Run sequential processing
    (W_final, _), (features, ssl_losses) = jax.lax.scan(
        step_fn,
        (W0, 0),
        inputs
    )
    
    return features, ssl_losses, W_final


def process_trajectory_no_adapt(
    ttt_adapter: TTTRoboticsAdapter,
    params: dict,
    obs_sequence: jnp.ndarray,
) -> jnp.ndarray:
    """Process trajectory without adaptation (baseline)."""
    W0 = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_W)
    features = ttt_adapter.apply(
        {'params': params},
        obs_sequence, W0,
        method=ttt_adapter.get_features
    )
    return features


# ============================================================================
# Training Utilities
# ============================================================================

def create_ttt_train_step(
    ttt_adapter: TTTRoboticsAdapter,
    lambda_ssl: float = 1.0,
):
    """
    Create training step function.
    
    The outer loop optimizes θ = {W₀, θ_K, θ_V, θ_Q, θ_lr, proj_out}
    to minimize SSL loss after one adaptation step.
    """
    
    def loss_fn(params, batch):
        """
        Compute SSL loss for outer loop optimization.
        
        The loss is computed AFTER one inner loop adaptation step,
        which encourages learning:
        - W₀ that adapts well
        - Projections that create a useful (non-trivial) task
        """
        obs = batch['obs_embeddings']
        action = batch['actions']
        next_obs = batch['next_obs_embeddings']
        
        # Forward pass with adaptation
        features, W_adapted, ssl_before, ssl_after = ttt_adapter.apply(
            {'params': params},
            obs, action, next_obs,
            adapt=True,
            train=True,
        )
        
        # Outer loop optimizes post-adaptation loss
        loss = lambda_ssl * ssl_after
        
        metrics = {
            'ssl_loss_before': ssl_before,
            'ssl_loss_after': ssl_after,
            'ssl_improvement': ssl_before - ssl_after,
            'ssl_improvement_pct': (ssl_before - ssl_after) / (ssl_before + 1e-8) * 100,
        }
        
        return loss, (metrics, features)
    
    @jax.jit
    def train_step(params, opt_state, batch, tx):
        (loss, (metrics, features)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, batch)
        
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        metrics['grad_norm'] = optax.global_norm(grads)
        metrics['loss'] = loss
        
        return params, opt_state, metrics, features
    
    return train_step


# ============================================================================
# Debugging Utilities
# ============================================================================

def debug_ttt_state(ttt_adapter, params, batch, step=0):
    """
    Diagnostic function to detect collapse.
    """
    diagnostics = {}
    
    obs = batch['obs_embeddings']
    action = batch['actions']
    next_obs = batch['next_obs_embeddings']
    
    # Forward pass
    features, W_adapted, ssl_before, ssl_after = ttt_adapter.apply(
        {'params': params},
        obs, action, next_obs,
        adapt=True,
    )
    
    diagnostics['ssl_loss_before'] = float(ssl_before)
    diagnostics['ssl_loss_after'] = float(ssl_after)
    diagnostics['ssl_improvement'] = float(ssl_before - ssl_after)
    
    # Check projection similarity
    ttt_params = params.get('ttt_layer', params)
    if 'proj_K' in ttt_params and 'proj_V' in ttt_params:
        K = ttt_params['proj_K']['kernel']
        V = ttt_params['proj_V']['kernel']
        
        # For prediction mode, K and V have different input dims
        # So we can't directly compare them
        diagnostics['proj_K_norm'] = float(jnp.linalg.norm(K))
        diagnostics['proj_V_norm'] = float(jnp.linalg.norm(V))
    
    # Check W₀
    if 'W_init' in ttt_params:
        W0 = ttt_params['W_init']
        diagnostics['W_init_norm'] = float(jnp.linalg.norm(W0))
        diagnostics['W_init_mean_abs'] = float(jnp.mean(jnp.abs(W0)))
    
    # Check adaptation magnitude
    W0 = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_W)
    W_change = jnp.linalg.norm(W_adapted - W0) / (jnp.linalg.norm(W0) + 1e-8)
    diagnostics['W_adaptation_magnitude'] = float(W_change)
    
    # Feature statistics
    diagnostics['feature_mean'] = float(jnp.mean(features))
    diagnostics['feature_std'] = float(jnp.std(features))
    
    # Warnings
    if diagnostics['ssl_loss_before'] < 0.1:
        diagnostics['WARNING'] = "SSL loss too low - possible collapse!"
    if diagnostics['ssl_improvement'] < 0:
        diagnostics['WARNING'] = "SSL loss increased after adaptation!"
    
    return diagnostics


def print_diagnostics(diagnostics, step=0):
    """Pretty print diagnostics."""
    print(f"\n{'='*60}")
    print(f"TTT Diagnostics @ Step {step}")
    print('='*60)
    
    warnings = []
    for key, val in diagnostics.items():
        if key == 'WARNING':
            warnings.append(val)
        elif isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")
    
    for w in warnings:
        print(f"\n⚠️  {w}")
    
    print('='*60 + '\n')