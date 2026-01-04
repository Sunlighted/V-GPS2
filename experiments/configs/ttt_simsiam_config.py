"""
SimSiam and Dynamics networks for EfficientZero-style self-supervised learning.
"""

from typing import Sequence
import flax.linen as nn
import jax.numpy as jnp


class SimSiamProjector(nn.Module):
    """
    Projector MLP for SimSiam.
    
    Takes encoder output and projects to a higher-dimensional space.
    Architecture: input → hidden → hidden → ... → output
    All layers except final have Norm + ReLU.
    Final layer has no normalization or activation.
    
    Note: Default uses LayerNorm instead of BatchNorm for simplicity in RL settings.
    """
    hidden_dims: Sequence[int]
    output_dim: int
    norm_type: str = "layer"  # "batch", "layer", or "none" (default: layer for simplicity)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Hidden layers with norm + activation
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            
            x = nn.relu(x)
        
        # Final layer: no norm, no activation (following SimSiam paper)
        x = nn.Dense(self.output_dim, name="output")(x)
        return x


class SimSiamPredictor(nn.Module):
    """
    Predictor MLP for SimSiam.
    
    Takes projection and predicts the target projection.
    Architecture: input → hidden → hidden → ... → output
    All layers except final have Norm + ReLU.
    Final layer has no normalization or activation.
    
    Note: Default uses LayerNorm instead of BatchNorm for simplicity in RL settings.
    """
    hidden_dims: Sequence[int]
    output_dim: int
    norm_type: str = "layer"  # default: layer for simplicity
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Hidden layers with norm + activation
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


class DynamicsNetworkFlat(nn.Module):
    """
    Dynamics network for flat latent representations.
    
    Adapted from EfficientZero's conv-based dynamics network to work with
    flat (pooled) encoder outputs. Predicts next state encoding from
    current state encoding and action.
    
    Architecture:
    - Concatenate state + action
    - Dense layer with residual connection from input state
    - Optional residual blocks
    
    Note: Default uses LayerNorm instead of BatchNorm for simplicity in RL settings.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_residual_blocks: int = 1
    norm_type: str = "layer"  # default: layer for simplicity
    
    @nn.compact
    def __call__(self, z_t: jnp.ndarray, action: jnp.ndarray, train: bool = True) -> jnp.ndarray:
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
        
        # First layer with residual connection from input state
        x = nn.Dense(self.latent_dim, name="conv_0")(x)
        
        if self.norm_type == "batch":
            x = nn.BatchNorm(use_running_average=not train, name="bn_0")(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm(name="ln_0")(x)
        
        # Add residual from input state (analogous to EfficientZero's skip connection)
        x = x + z_t
        x = nn.relu(x)
        
        # Residual blocks
        for i in range(self.num_residual_blocks):
            residual = x
            
            # First conv in residual block
            x = nn.Dense(self.hidden_dim, name=f"res_{i}_dense_0")(x)
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"res_{i}_bn_0")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"res_{i}_ln_0")(x)
            x = nn.relu(x)
            
            # Second conv in residual block
            x = nn.Dense(self.latent_dim, name=f"res_{i}_dense_1")(x)
            if self.norm_type == "batch":
                x = nn.BatchNorm(use_running_average=not train, name=f"res_{i}_bn_1")(x)
            elif self.norm_type == "layer":
                x = nn.LayerNorm(name=f"res_{i}_ln_1")(x)
            
            # Residual connection
            x = x + residual
            x = nn.relu(x)
        
        return x


def cosine_similarity_loss(
    pred: jnp.ndarray, 
    target: jnp.ndarray, 
    *, 
    normalize: bool = True
) -> jnp.ndarray:
    """
    Compute cosine similarity loss for SimSiam.
    
    Args:
        pred: Predicted embeddings
        target: Target embeddings (should be stop_gradient)
        normalize: Whether to L2-normalize before computing cosine similarity
    
    Returns:
        Scalar loss (mean negative cosine similarity)
    """
    eps = 1e-8
    
    if normalize:
        pred = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + eps)
        target = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + eps)
    
    # Cosine similarity: dot product of normalized vectors
    cos_sim = jnp.sum(pred * target, axis=-1)
    
    # Loss is negative cosine similarity (or equivalently, 1 - cos_sim)
    # We use 1 - cos_sim to get a loss in [0, 2]
    loss = jnp.mean(1.0 - cos_sim)
    
    return loss