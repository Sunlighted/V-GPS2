"""
TTT Module implementation in JAX/Flax.
Matches the PyTorch implementation with manual gradient updates for test-time adaptation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Callable
from functools import partial


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


# def manual_update_step(params, x, target, lr, mask=None):
#     """
#     Manually compute gradients and update TTTModule parameters.
#     This is used for efficient test-time adaptation.
    
#     Args:
#         params: Dictionary of TTTModule parameters
#         x: Input tensor (B, T, D)
#         target: Target tensor (B, T, D)
#         lr: Learning rate
        
#     Returns:
#         updated_params: Updated parameters
#         loss: Scalar loss value
#     """
#     eps = 1e-6
    
#     # Extract parameters
#     W1 = params['linear1']['kernel']  # (D, 4D)
#     b1 = params['linear1']['bias']    # (4D,)
#     W2 = params['linear2']['kernel']  # (4D, D)
#     b2 = params['linear2']['bias']    # (D,)
#     gamma = params['layer_norm']['scale']  # (D,)
#     beta = params['layer_norm']['bias']    # (D,)
    
#     if mask is None:
#         mask = jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)
#     mask = mask.astype(x.dtype)
#     valid_elems = jnp.maximum(jnp.sum(mask) * x.shape[-1], 1.0)

#     # Forward pass
#     residual = x
#     z1 = x @ W1 + b1  # (B, T, 4D)
#     z1_gelu = nn.gelu(z1, approximate=True)
#     z2 = z1_gelu @ W2 + b2  # (B, T, D)
#     output = residual + z2
    
#     # LayerNorm
#     mean = jnp.mean(output, axis=-1, keepdims=True)
#     var = jnp.var(output, axis=-1, keepdims=True)
#     std = jnp.sqrt(var + eps)
#     x_hat = (output - mean) / std
#     y_norm = gamma * x_hat + beta
    
#     # Loss (MSE)
#     diff = y_norm - target
#     loss = jnp.sum((diff ** 2) * mask) / valid_elems
    
#     # Gradient of loss w.r.t. output
#     grad_y = 2 * diff * mask / valid_elems
    
#     # Backprop through LayerNorm
#     D = x.shape[-1]
#     grad_x_hat = grad_y * gamma
#     grad_out = (
#         1.0 / D * 
#         (D * grad_x_hat - jnp.sum(grad_x_hat, axis=-1, keepdims=True) - 
#          x_hat * jnp.sum(grad_x_hat * x_hat, axis=-1, keepdims=True))
#         / std
#     )
#     grad_gamma = jnp.sum(grad_y * x_hat, axis=(0, 1))
#     grad_beta = jnp.sum(grad_y, axis=(0, 1))
    
#     # Backprop through second linear layer
#     grad_z2 = grad_out
#     grad_W2 = jnp.reshape(grad_z2, (-1, grad_z2.shape[-1])).T @ jnp.reshape(z1_gelu, (-1, z1_gelu.shape[-1]))
#     grad_b2 = jnp.sum(grad_z2, axis=(0, 1))
    
#     # Backprop through GELU
#     grad_z1_gelu = grad_z2 @ W2.T
#     grad_z1 = grad_z1_gelu * gelu_bwd(z1)
    
#     # Backprop through first linear layer
#     grad_W1 = jnp.reshape(grad_z1, (-1, grad_z1.shape[-1])).T @ jnp.reshape(x, (-1, x.shape[-1]))
#     grad_b1 = jnp.sum(grad_z1, axis=(0, 1))
    
#     # Update parameters
#     updated_params = {
#         'linear1': {
#             'kernel': W1 - lr * grad_W1.T,
#             'bias': b1 - lr * grad_b1
#         },
#         'linear2': {
#             'kernel': W2 - lr * grad_W2.T,
#             'bias': b2 - lr * grad_b2
#         },
#         'layer_norm': {
#             'scale': gamma - lr * grad_gamma,
#             'bias': beta - lr * grad_beta
#         }
#     }
    
#     return updated_params, loss


# def ttt_adaptation(params, x, target, lr, num_steps, mask=None):
#     """
#     Run multiple steps of test-time adaptation.
    
#     Args:
#         params: TTTModule parameters
#         x: Input tensor (B, T, D)
#         target: Target tensor (B, T, D)
#         lr: Learning rate
#         num_steps: Number of adaptation steps
        
#     Returns:
#         adapted_params: Adapted parameters
#         losses: List of losses for each step
#     """
#     if mask is None:
#         mask = jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)

#     def body_fn(curr_params, _):
#         return manual_update_step(curr_params, x, target, lr, mask)

#     adapted_params, losses = lax.scan(body_fn, params, xs=None, length=num_steps)
#     return adapted_params, losses