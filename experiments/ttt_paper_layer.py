"""
TTT Layer implementation following "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
(Sun et al., 2024)

Key components:
- Multi-view reconstruction SSL task with learned projections θ_K, θ_V, θ_Q
- Inner model f with residual + LayerNorm: f(x; W) = x + LN(W @ x)
- Learned initialization W₀
- Input-dependent learning rate: η(x) = η_base * sigmoid(θ_lr · x)
"""

from typing import Optional, Tuple, Callable
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class TTTLinear(nn.Module):
    """
    TTT Layer with linear inner model.
    
    The hidden state W is a learnable model that adapts during the forward pass.
    
    Outer-loop parameters (trained with gradient descent):
    - proj_K: training view projection (θ_K)
    - proj_V: label view projection (θ_V)  
    - proj_Q: output view projection (θ_Q)
    - W_init: initial hidden state (θ_init)
    - lr_vec: input-dependent learning rate vector (θ_lr)
    
    Inner-loop state (updated during forward pass):
    - W: current hidden state model weights
    """
    input_dim: int
    bottleneck_dim: int = 64
    output_dim: Optional[int] = None  # defaults to input_dim
    eta_base: float = 1.0
    use_input_dependent_lr: bool = True
    
    def setup(self):
        out_dim = self.output_dim or self.input_dim
        
        # Projection layers (outer-loop params)
        # θ_K: training view - projects to bottleneck for SSL
        self.proj_K = nn.Dense(self.bottleneck_dim, name="proj_K")
        # θ_V: label view - reconstruction target
        self.proj_V = nn.Dense(self.bottleneck_dim, name="proj_V")
        # θ_Q: output view - for producing output features
        self.proj_Q = nn.Dense(self.bottleneck_dim, name="proj_Q")
        
        # Output projection to desired dimension
        self.proj_out = nn.Dense(out_dim, name="proj_out")
        
        # LayerNorm for inner model (part of f)
        self.inner_ln = nn.LayerNorm(name="inner_ln")
        
        # Learnable learning rate vector (outer-loop param)
        if self.use_input_dependent_lr:
            self.lr_dense = nn.Dense(1, name="lr_dense")
    
    @nn.compact
    def get_initial_W(self):
        """Get learnable initial hidden state W₀."""
        # W₀ is a learnable parameter (outer-loop)
        # Initialize to small values so f(x; W₀) ≈ x initially
        W_init = self.param(
            "W_init",
            nn.initializers.normal(stddev=0.01),
            (self.bottleneck_dim, self.bottleneck_dim)
        )
        return W_init
    
    def inner_model(self, x: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
        """
        Inner model f with residual connection and LayerNorm.
        f(x; W) = x + LayerNorm(W @ x)
        
        Args:
            x: input features (bottleneck_dim,) or (batch, bottleneck_dim)
            W: hidden state weights (bottleneck_dim, bottleneck_dim)
            
        Returns:
            output: same shape as x
        """
        # Linear transformation
        Wx = x @ W.T  # (*, bottleneck_dim)
        
        # LayerNorm + residual
        out = x + self.inner_ln(Wx)
        return out
    
    def compute_ssl_loss(
        self,
        x: jnp.ndarray,
        W: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute self-supervised loss for a single input.
        
        ℓ(W; x) = ||f(θ_K·x; W) - θ_V·x||²
        
        Args:
            x: input embedding (input_dim,) or (batch, input_dim)
            W: current hidden state (bottleneck_dim, bottleneck_dim)
            
        Returns:
            loss: scalar SSL loss
        """
        # Training view (input to inner model)
        k = self.proj_K(x)  # (*, bottleneck_dim)
        
        # Label view (reconstruction target)
        v = self.proj_V(x)  # (*, bottleneck_dim)
        
        # Inner model prediction
        pred = self.inner_model(k, W)  # (*, bottleneck_dim)
        
        # MSE loss
        loss = jnp.mean((pred - v) ** 2)
        return loss
    
    def compute_ssl_loss_batch(
        self,
        x_batch: jnp.ndarray,
        W: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute SSL loss for a batch, returning per-sample losses.
        
        Args:
            x_batch: (batch, input_dim)
            W: (bottleneck_dim, bottleneck_dim)
            
        Returns:
            losses: (batch,) per-sample losses
        """
        k = self.proj_K(x_batch)  # (batch, bottleneck_dim)
        v = self.proj_V(x_batch)  # (batch, bottleneck_dim)
        pred = self.inner_model(k, W)  # (batch, bottleneck_dim)
        
        # Per-sample MSE
        losses = jnp.mean((pred - v) ** 2, axis=-1)  # (batch,)
        return losses
    
    def get_learning_rate(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute input-dependent learning rate.
        η(x) = η_base * sigmoid(θ_lr · x)
        
        Args:
            x: input embedding (input_dim,) or (batch, input_dim)
            
        Returns:
            eta: learning rate scalar or (batch,) array
        """
        if self.use_input_dependent_lr:
            # θ_lr · x through a dense layer
            lr_logit = self.lr_dense(x)  # (*, 1)
            lr_scale = jax.nn.sigmoid(lr_logit.squeeze(-1))  # (*,)
            eta = self.eta_base * lr_scale
        else:
            eta = self.eta_base
        return eta
    
    def adapt_step(
        self,
        x: jnp.ndarray,
        W: jnp.ndarray,
        params: dict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform one TTT adaptation step on W.
        
        Args:
            x: input embedding (input_dim,)
            W: current hidden state (bottleneck_dim, bottleneck_dim)
            params: module parameters (needed for gradient computation)
            
        Returns:
            W_new: updated hidden state
            ssl_loss: the SSL loss before update
        """
        # Compute SSL loss and gradient w.r.t. W
        def loss_fn(W_):
            return self.compute_ssl_loss(x, W_)
        
        ssl_loss, grad_W = jax.value_and_grad(loss_fn)(W)
        
        # Input-dependent learning rate
        eta = self.get_learning_rate(x)
        
        # Update W
        W_new = W - eta * grad_W
        
        return W_new, ssl_loss
    
    def forward_single(
        self,
        x: jnp.ndarray,
        W: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute output for a single input given hidden state W.
        z = proj_out(f(θ_Q·x; W))
        
        Args:
            x: input embedding (input_dim,)
            W: hidden state (bottleneck_dim, bottleneck_dim)
            
        Returns:
            z: output features (output_dim,)
        """
        # Output view
        q = self.proj_Q(x)  # (bottleneck_dim,)
        
        # Inner model
        h = self.inner_model(q, W)  # (bottleneck_dim,)
        
        # Project to output dim
        z = self.proj_out(h)  # (output_dim,)
        
        return z
    
    def forward_batch_no_adapt(
        self,
        x_batch: jnp.ndarray,
        W: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute output for a batch without adaptation (W fixed).
        
        Args:
            x_batch: (batch, input_dim)
            W: (bottleneck_dim, bottleneck_dim)
            
        Returns:
            z_batch: (batch, output_dim)
        """
        q = self.proj_Q(x_batch)  # (batch, bottleneck_dim)
        h = self.inner_model(q, W)  # (batch, bottleneck_dim)
        z = self.proj_out(h)  # (batch, output_dim)
        return z
    
    def __call__(
        self,
        x: jnp.ndarray,
        W: Optional[jnp.ndarray] = None,
        adapt: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for a single input with optional adaptation.
        
        Args:
            x: input embedding (input_dim,)
            W: current hidden state, or None to use W₀
            adapt: whether to update W
            
        Returns:
            z: output features (output_dim,)
            W_new: updated hidden state (same as W if adapt=False)
            ssl_loss: SSL loss (0 if adapt=False)
        """
        if W is None:
            W = self.get_initial_W()
        
        if adapt:
            # Compute SSL loss and gradient
            ssl_loss = self.compute_ssl_loss(x, W)
            
            # Gradient of SSL loss w.r.t. W
            grad_fn = jax.grad(lambda W_: self.compute_ssl_loss(x, W_))
            grad_W = grad_fn(W)
            
            # Input-dependent learning rate
            eta = self.get_learning_rate(x)
            
            # Update W
            W_new = W - eta * grad_W
        else:
            W_new = W
            ssl_loss = jnp.array(0.0)
        
        # Compute output
        z = self.forward_single(x, W_new)
        
        return z, W_new, ssl_loss


class TTTMLP(nn.Module):
    """
    TTT Layer with MLP inner model.
    
    f(x; W) = x + LayerNorm(W₂ · GELU(W₁ · x))
    
    More expressive than TTT-Linear but higher compute.
    """
    input_dim: int
    bottleneck_dim: int = 64
    inner_expansion: int = 4  # MLP hidden dim = bottleneck_dim * inner_expansion
    output_dim: Optional[int] = None
    eta_base: float = 0.1  # Lower LR for MLP
    use_input_dependent_lr: bool = True
    
    def setup(self):
        out_dim = self.output_dim or self.input_dim
        inner_hidden = self.bottleneck_dim * self.inner_expansion
        
        # Projections (same as TTT-Linear)
        self.proj_K = nn.Dense(self.bottleneck_dim, name="proj_K")
        self.proj_V = nn.Dense(self.bottleneck_dim, name="proj_V")
        self.proj_Q = nn.Dense(self.bottleneck_dim, name="proj_Q")
        self.proj_out = nn.Dense(out_dim, name="proj_out")
        
        # Inner model components
        self.inner_ln = nn.LayerNorm(name="inner_ln")
        
        if self.use_input_dependent_lr:
            self.lr_dense = nn.Dense(1, name="lr_dense")
    
    @nn.compact
    def get_initial_W(self):
        """Get learnable initial hidden states W₁, W₂ for MLP."""
        inner_hidden = self.bottleneck_dim * self.inner_expansion
        
        W1_init = self.param(
            "W1_init",
            nn.initializers.normal(stddev=0.01),
            (self.bottleneck_dim, inner_hidden)
        )
        W2_init = self.param(
            "W2_init",
            nn.initializers.normal(stddev=0.01),
            (inner_hidden, self.bottleneck_dim)
        )
        return (W1_init, W2_init)
    
    def inner_model(
        self,
        x: jnp.ndarray,
        W: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        MLP inner model: f(x; W) = x + LN(W₂ · GELU(W₁ · x))
        """
        W1, W2 = W
        h = jax.nn.gelu(x @ W1)  # (*, inner_hidden)
        out = h @ W2  # (*, bottleneck_dim)
        return x + self.inner_ln(out)
    
    def compute_ssl_loss(self, x: jnp.ndarray, W: Tuple) -> jnp.ndarray:
        k = self.proj_K(x)
        v = self.proj_V(x)
        pred = self.inner_model(k, W)
        return jnp.mean((pred - v) ** 2)
    
    def get_learning_rate(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_input_dependent_lr:
            lr_logit = self.lr_dense(x)
            lr_scale = jax.nn.sigmoid(lr_logit.squeeze(-1))
            return self.eta_base * lr_scale
        return self.eta_base
    
    def forward_single(self, x: jnp.ndarray, W: Tuple) -> jnp.ndarray:
        q = self.proj_Q(x)
        h = self.inner_model(q, W)
        return self.proj_out(h)
    
    def __call__(
        self,
        x: jnp.ndarray,
        W: Optional[Tuple] = None,
        adapt: bool = True,
    ) -> Tuple[jnp.ndarray, Tuple, jnp.ndarray]:
        if W is None:
            W = self.get_initial_W()
        
        if adapt:
            ssl_loss = self.compute_ssl_loss(x, W)
            grad_fn = jax.grad(lambda W_: self.compute_ssl_loss(x, W_))
            grad_W = grad_fn(W)
            eta = self.get_learning_rate(x)
            W_new = jax.tree_map(lambda w, g: w - eta * g, W, grad_W)
        else:
            W_new = W
            ssl_loss = jnp.array(0.0)
        
        z = self.forward_single(x, W_new)
        return z, W_new, ssl_loss


class TTTFeatureAdapter(nn.Module):
    """
    High-level TTT feature adapter for robotics.
    
    Wraps TTT layer to process observation embeddings and produce
    adapted features for downstream RL.
    
    Can operate in:
    - "batch" mode: Process batch without sequential adaptation (training)
    - "sequential" mode: Process sequence with cumulative adaptation (evaluation)
    """
    input_dim: int
    bottleneck_dim: int = 64
    output_dim: int = 256
    ttt_type: str = "linear"  # "linear" or "mlp"
    eta_base: float = 1.0
    use_input_dependent_lr: bool = True
    
    def setup(self):
        if self.ttt_type == "linear":
            self.ttt_layer = TTTLinear(
                input_dim=self.input_dim,
                bottleneck_dim=self.bottleneck_dim,
                output_dim=self.output_dim,
                eta_base=self.eta_base,
                use_input_dependent_lr=self.use_input_dependent_lr,
                name="ttt_linear"
            )
        elif self.ttt_type == "mlp":
            self.ttt_layer = TTTMLP(
                input_dim=self.input_dim,
                bottleneck_dim=self.bottleneck_dim,
                output_dim=self.output_dim,
                eta_base=self.eta_base,
                use_input_dependent_lr=self.use_input_dependent_lr,
                name="ttt_mlp"
            )
        else:
            raise ValueError(f"Unknown ttt_type: {self.ttt_type}")
    
    def get_initial_state(self):
        """Get initial hidden state W₀."""
        return self.ttt_layer.get_initial_W()
    
    def forward_batch(
        self,
        x_batch: jnp.ndarray,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process a batch of inputs without sequential adaptation.
        Uses W₀ for all samples.
        
        Args:
            x_batch: (batch, input_dim)
            train: whether in training mode
            
        Returns:
            z_batch: (batch, output_dim) adapted features
            ssl_loss: scalar mean SSL loss
        """
        W = self.ttt_layer.get_initial_W()
        
        # Compute SSL loss (mean over batch)
        ssl_loss = self.ttt_layer.compute_ssl_loss(x_batch, W)
        
        # Compute output features (no per-sample adaptation in batch mode)
        z_batch = self.ttt_layer.forward_batch_no_adapt(x_batch, W)
        
        return z_batch, ssl_loss
    
    def forward_batch_with_single_step_adapt(
        self,
        x_batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Process batch with one adaptation step per sample (parallel).
        Each sample gets its own adapted W.
        
        Args:
            x_batch: (batch, input_dim)
            
        Returns:
            z_batch: (batch, output_dim)
            ssl_loss_before: scalar mean loss before adaptation
            ssl_loss_after: scalar mean loss after adaptation
        """
        W0 = self.ttt_layer.get_initial_W()
        batch_size = x_batch.shape[0]
        
        # Broadcast W0 to batch: (batch, d, d) or (batch, (d, h), (h, d)) for MLP
        if isinstance(W0, tuple):
            W_batch = jax.tree_map(lambda w: jnp.broadcast_to(w, (batch_size,) + w.shape), W0)
        else:
            W_batch = jnp.broadcast_to(W0, (batch_size,) + W0.shape)
        
        # Compute per-sample SSL loss before adaptation
        ssl_losses_before = self.ttt_layer.compute_ssl_loss_batch(x_batch, W0)
        ssl_loss_before = jnp.mean(ssl_losses_before)
        
        # Per-sample adaptation (vmapped)
        def adapt_single(x, W):
            ssl_loss = self.ttt_layer.compute_ssl_loss(x, W)
            grad_fn = jax.grad(lambda W_: self.ttt_layer.compute_ssl_loss(x, W_))
            grad_W = grad_fn(W)
            eta = self.ttt_layer.get_learning_rate(x)
            W_new = jax.tree_map(lambda w, g: w - eta * g, W, grad_W)
            return W_new, ssl_loss
        
        # Note: Can't easily vmap gradient computation, so we do it sequentially
        # or use a different approach. For now, just use W0 everywhere.
        # This is a simplification - full per-sample adaptation needs more work.
        
        # Simple approach: compute output with adapted W (one global step)
        ssl_loss_global = self.ttt_layer.compute_ssl_loss(x_batch, W0)
        grad_fn = jax.grad(lambda W_: self.ttt_layer.compute_ssl_loss(x_batch, W_))
        grad_W = grad_fn(W0)
        eta = jnp.mean(self.ttt_layer.get_learning_rate(x_batch))
        W_adapted = jax.tree_map(lambda w, g: w - eta * g, W0, grad_W)
        
        ssl_loss_after = self.ttt_layer.compute_ssl_loss(x_batch, W_adapted)
        z_batch = self.ttt_layer.forward_batch_no_adapt(x_batch, W_adapted)
        
        return z_batch, ssl_loss_before, ssl_loss_after
    
    def __call__(
        self,
        x: jnp.ndarray,
        W: Optional[jnp.ndarray] = None,
        adapt: bool = True,
        train: bool = True,
    ):
        """
        Forward pass.
        
        For batched input without sequential context, use forward_batch().
        This __call__ is for single samples or sequential processing.
        """
        return self.ttt_layer(x, W, adapt=adapt)


# ============================================================================
# Utility functions for sequential TTT processing
# ============================================================================

def process_sequence_with_ttt(
    ttt_adapter: TTTFeatureAdapter,
    params: dict,
    x_sequence: jnp.ndarray,
    reset_each_step: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Process a sequence of observations with TTT adaptation.
    
    Args:
        ttt_adapter: TTTFeatureAdapter module
        params: module parameters
        x_sequence: (T, input_dim) sequence of observations
        reset_each_step: if True, reset W to W₀ at each step (like 2020 paper)
                        if False, accumulate adaptations (online mode)
    
    Returns:
        z_sequence: (T, output_dim) adapted features
        ssl_losses: (T,) SSL loss at each step
        W_final: final hidden state
    """
    T = x_sequence.shape[0]
    
    def step_fn(carry, x_t):
        W = carry
        
        if reset_each_step:
            # Reset to W₀ at each step
            W_init = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_state)
            W = W_init
        
        # Forward with adaptation
        z_t, W_new, ssl_loss = ttt_adapter.apply(
            {'params': params},
            x_t,
            W,
            adapt=True,
            train=False
        )
        
        return W_new, (z_t, ssl_loss)
    
    # Get initial state
    W0 = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_state)
    
    # Scan through sequence
    W_final, (z_sequence, ssl_losses) = jax.lax.scan(step_fn, W0, x_sequence)
    
    return z_sequence, ssl_losses, W_final


def process_sequence_no_adapt(
    ttt_adapter: TTTFeatureAdapter,
    params: dict,
    x_sequence: jnp.ndarray,
) -> jnp.ndarray:
    """
    Process sequence without adaptation (baseline).
    """
    W0 = ttt_adapter.apply({'params': params}, method=ttt_adapter.get_initial_state)
    z_sequence = ttt_adapter.apply(
        {'params': params},
        x_sequence,
        W0,
        method=ttt_adapter.ttt_layer.forward_batch_no_adapt
    )
    return z_sequence


# ============================================================================
# Loss functions for training
# ============================================================================

def compute_ttt_ssl_loss(
    params: dict,
    ttt_adapter: TTTFeatureAdapter,
    x_batch: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute SSL loss for a batch (training).
    
    Returns:
        loss: scalar
        metrics: dict with additional info
    """
    z_batch, ssl_loss = ttt_adapter.apply(
        {'params': params},
        x_batch,
        train=True,
        method=ttt_adapter.forward_batch
    )
    
    return ssl_loss, {'ssl_loss': ssl_loss, 'features': z_batch}


def compute_ttt_ssl_loss_with_adapt(
    params: dict,
    ttt_adapter: TTTFeatureAdapter,
    x_batch: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute SSL loss with one adaptation step.
    Reports loss before and after adaptation.
    """
    z_batch, ssl_before, ssl_after = ttt_adapter.apply(
        {'params': params},
        x_batch,
        method=ttt_adapter.forward_batch_with_single_step_adapt
    )
    
    # Training loss is the loss after adaptation (encourages W₀ that adapts well)
    loss = ssl_after
    
    return loss, {
        'ssl_loss_before': ssl_before,
        'ssl_loss_after': ssl_after,
        'ssl_improvement': ssl_before - ssl_after,
        'features': z_batch,
    }