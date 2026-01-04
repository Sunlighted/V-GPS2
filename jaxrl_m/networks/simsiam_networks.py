"""
EfficientZero-style SimSiam and Dynamics networks.

Key differences from the flat version:
1. Dynamics network is convolutional, operating on spatial features (H×W×C)
2. Action is broadcast to spatial dimensions and concatenated along channels
3. Residual connection from input state after first conv
"""

from typing import Sequence, Tuple, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp


class SimSiamProjector(nn.Module):
    """
    Projector MLP for SimSiam.
    
    Takes encoder output (can be spatial or flat) and projects to embedding space.
    If input is spatial (H, W, C), it's flattened first.
    
    Architecture: input → hidden → hidden → ... → output
    All layers except final have Norm + ReLU.
    """
    hidden_dims: Sequence[int] = (512, 512, 512)
    output_dim: int = 512
    norm_type: str = "layer"  # "batch", "layer", or "none"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Flatten if spatial
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            
            x = nn.relu(x)
        
        # Final layer: no norm, no activation
        x = nn.Dense(self.output_dim, name="output")(x)
        return x


class SimSiamPredictor(nn.Module):
    """
    Predictor MLP for SimSiam.
    
    Architecture: input → hidden → ... → output
    All layers except final have Norm + ReLU.
    """
    hidden_dims: Sequence[int] = (256,)
    output_dim: int = 512
    norm_type: str = "layer"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            
            x = nn.relu(x)
        
        x = nn.Dense(self.output_dim, name="output")(x)
        return x


class ResidualBlockConv(nn.Module):
    """
    Single residual block with two convolutions.
    
    Architecture:
        x → Conv → Norm → ReLU → Conv → Norm → (+x) → ReLU
    """
    channels: int
    kernel_size: Tuple[int, int] = (3, 3)
    norm_type: str = "layer"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        
        # First conv
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            padding="SAME",
            name="conv_0"
        )(x)
        
        if self.norm_type == "batch":
            x = nn.BatchNorm(use_running_average=not train, name="bn_0")(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm(name="ln_0")(x)
        
        x = nn.relu(x)
        
        # Second conv
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            padding="SAME",
            name="conv_1"
        )(x)
        
        if self.norm_type == "batch":
            x = nn.BatchNorm(use_running_average=not train, name="bn_1")(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm(name="ln_1")(x)
        
        # Residual connection
        x = x + residual
        x = nn.relu(x)
        
        return x


class DynamicsNetworkConv(nn.Module):
    """
    EfficientZero-style convolutional dynamics network.
    
    Predicts next hidden state from current state and action.
    
    Architecture (from EfficientZero Appendix A.1):
        Input: Hidden state s_t (H×W×C), Action a_t (action_dim,)
        
        1. Broadcast action to spatial dims: (H, W, action_dim)
        2. Concatenate: (H, W, C + action_dim)
        3. Conv (64 channels, 3×3, stride 1) → Norm
        4. Residual add: + s_t → ReLU
        5. 1 Residual Block (64 channels)
        
        Output: Next hidden state ŝ_{t+1} (H×W×C)
    
    Note: EfficientZero uses stride=1 (not stride=2 as stated in paper, 
    which would break the residual connection dimensions).
    """
    latent_channels: int = 64
    action_dim: int = 7
    num_residual_blocks: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    norm_type: str = "layer"
    
    @nn.compact
    def __call__(
        self, 
        z_t: jnp.ndarray, 
        action: jnp.ndarray, 
        train: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            z_t: Current state encoding, shape (batch, H, W, C)
            action: Action taken, shape (batch, action_dim)
            train: Whether in training mode
        
        Returns:
            Predicted next state encoding, shape (batch, H, W, C)
        """
        batch_size = z_t.shape[0]
        H, W, C = z_t.shape[1], z_t.shape[2], z_t.shape[3]
        
        # Broadcast action to spatial dimensions: (batch, action_dim) → (batch, H, W, action_dim)
        action_broadcast = action[:, None, None, :]  # (batch, 1, 1, action_dim)
        action_broadcast = jnp.broadcast_to(
            action_broadcast, 
            (batch_size, H, W, self.action_dim)
        )
        
        # Concatenate state and action along channel dimension
        x = jnp.concatenate([z_t, action_broadcast], axis=-1)  # (batch, H, W, C + action_dim)
        
        # First convolution (maintains spatial dims with stride=1)
        x = nn.Conv(
            features=self.latent_channels,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding="SAME",
            name="conv_init"
        )(x)
        
        if self.norm_type == "batch":
            x = nn.BatchNorm(use_running_average=not train, name="bn_init")(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm(name="ln_init")(x)
        
        # Residual connection from input state (key EfficientZero design)
        # This helps preserve information during recurrent rollouts
        # Need to match channels if they differ
        if C != self.latent_channels:
            z_t_proj = nn.Conv(
                features=self.latent_channels,
                kernel_size=(1, 1),
                name="input_proj"
            )(z_t)
        else:
            z_t_proj = z_t
        
        x = x + z_t_proj
        x = nn.relu(x)
        
        # Residual blocks (EfficientZero uses 1, MuZero uses 16)
        for i in range(self.num_residual_blocks):
            x = ResidualBlockConv(
                channels=self.latent_channels,
                kernel_size=self.kernel_size,
                norm_type=self.norm_type,
                name=f"res_block_{i}"
            )(x, train=train)
        
        return x


class DynamicsNetworkFlat(nn.Module):
    """
    Flat dynamics network for when encoder outputs pooled features.
    
    Similar structure to EfficientZero but operates on flat vectors.
    Useful when using encoders with global average pooling.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_residual_blocks: int = 1
    norm_type: str = "layer"
    
    @nn.compact
    def __call__(
        self, 
        z_t: jnp.ndarray, 
        action: jnp.ndarray, 
        train: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            z_t: Current state encoding, shape (batch, latent_dim)
            action: Action taken, shape (batch, action_dim)
            train: Whether in training mode
        
        Returns:
            Predicted next state encoding, shape (batch, latent_dim)
        """
        # Concatenate state and action
        x = jnp.concatenate([z_t, action], axis=-1)
        
        # First layer
        x = nn.Dense(self.latent_dim, name="dense_init")(x)
        
        if self.norm_type == "batch":
            x = nn.BatchNorm(use_running_average=not train, name="bn_init")(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm(name="ln_init")(x)
        
        # Residual from input state
        x = x + z_t
        x = nn.relu(x)
        
        # Residual blocks
        for i in range(self.num_residual_blocks):
            residual = x
            
            x = nn.Dense(self.hidden_dim, name=f"res_{i}_dense_0")(x)
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"res_{i}_bn_0")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"res_{i}_ln_0")(x)
            x = nn.relu(x)
            
            x = nn.Dense(self.latent_dim, name=f"res_{i}_dense_1")(x)
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"res_{i}_bn_1")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"res_{i}_ln_1")(x)
            
            x = x + residual
            x = nn.relu(x)
        
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


class SimSiamModule(nn.Module):
    """
    Combined SimSiam module containing projector, predictor, and dynamics.
    
    This makes it easier to manage as a single Flax module in the agent.
    """
    # Projector config
    projector_hidden_dims: Sequence[int] = (512, 512, 512)
    projector_output_dim: int = 512
    projector_norm: str = "layer"
    
    # Predictor config
    predictor_hidden_dims: Sequence[int] = (256,)
    predictor_output_dim: int = 512
    predictor_norm: str = "layer"
    
    # Dynamics config
    latent_dim: int = 512  # For flat dynamics
    action_dim: int = 7
    dynamics_hidden_dim: int = 512
    dynamics_num_residual_blocks: int = 1
    dynamics_norm: str = "layer"
    use_conv_dynamics: bool = False  # Whether to use conv or flat dynamics
    dynamics_latent_channels: int = 64  # For conv dynamics
    
    def setup(self):
        self.projector = SimSiamProjector(
            hidden_dims=self.projector_hidden_dims,
            output_dim=self.projector_output_dim,
            norm_type=self.projector_norm,
        )
        
        self.predictor = SimSiamPredictor(
            hidden_dims=self.predictor_hidden_dims,
            output_dim=self.predictor_output_dim,
            norm_type=self.predictor_norm,
        )
        
        if self.use_conv_dynamics:
            self.dynamics = DynamicsNetworkConv(
                latent_channels=self.dynamics_latent_channels,
                action_dim=self.action_dim,
                num_residual_blocks=self.dynamics_num_residual_blocks,
                norm_type=self.dynamics_norm,
            )
        else:
            self.dynamics = DynamicsNetworkFlat(
                latent_dim=self.latent_dim,
                action_dim=self.action_dim,
                hidden_dim=self.dynamics_hidden_dim,
                num_residual_blocks=self.dynamics_num_residual_blocks,
                norm_type=self.dynamics_norm,
            )
    
    def __call__(
        self,
        z_t: jnp.ndarray,
        z_tp1: jnp.ndarray,
        action: jnp.ndarray,
        train: bool = True,
        stop_grad_target: bool = True,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute SimSiam loss and auxiliary metrics.
        
        Args:
            z_t: Current state encoding
            z_tp1: Next state encoding (ground truth)
            action: Action taken
            train: Training mode
            stop_grad_target: Whether to stop gradient on target branch
        
        Returns:
            loss: SimSiam cosine similarity loss
            metrics: Dictionary of auxiliary metrics
        """
        # Predict next state
        z_tp1_pred = self.dynamics(z_t, action, train=train)
        
        # Project both real and predicted next states
        proj_real = self.projector(z_tp1, train=train)
        proj_pred = self.projector(z_tp1_pred, train=train)
        
        # Target (with optional stop gradient)
        if stop_grad_target:
            proj_target = jax.lax.stop_gradient(proj_real)
        else:
            proj_target = proj_real
        
        # Predict from predicted projection
        prediction = self.predictor(proj_pred, train=train)
        
        # Compute loss
        loss = cosine_similarity_loss(prediction, proj_target, normalize=True)
        
        metrics = {
            "simsiam/z_norm_real": jnp.linalg.norm(z_tp1.reshape(z_tp1.shape[0], -1), axis=-1).mean(),
            "simsiam/z_norm_pred": jnp.linalg.norm(z_tp1_pred.reshape(z_tp1_pred.shape[0], -1), axis=-1).mean(),
            "simsiam/proj_norm_real": jnp.linalg.norm(proj_real, axis=-1).mean(),
            "simsiam/proj_norm_pred": jnp.linalg.norm(proj_pred, axis=-1).mean(),
            "simsiam/pred_norm": jnp.linalg.norm(prediction, axis=-1).mean(),
            "simsiam/cosine_sim": jnp.mean(
                jnp.sum(
                    (prediction / (jnp.linalg.norm(prediction, axis=-1, keepdims=True) + 1e-8)) *
                    (proj_target / (jnp.linalg.norm(proj_target, axis=-1, keepdims=True) + 1e-8)),
                    axis=-1
                )
            ),
        }
        
        return loss, metrics
    
    def predict_next_state(
        self,
        z_t: jnp.ndarray,
        action: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        """Predict next state encoding (for visualization/debugging)."""
        return self.dynamics(z_t, action, train=train)