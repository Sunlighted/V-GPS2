"""
TTT-CQL Agent: Combines Test-Time Training feature adaptation with CQL Q-learning.

Architecture:
    OCTO (frozen) → Projections (P_K, P_V, P_Q) → TTT Module (f_adapt) → CQL Agent

Training:
    loss = loss_cql + lambda_self * loss_ttt
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from typing import Optional, Dict, Any, Sequence
import optax
from functools import partial
from absl import logging

from ttt_module import TTTModule, ttt_adaptation
from jaxrl_m.agents import agents  # Your existing CQL agent


def _extract_sliding_windows(arr, window_length):
    """Extract all contiguous windows of length ``window_length`` along axis 1."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    
    B, T = arr.shape[:2]
    num_possible = T - window_length + 1
    if num_possible <= 0:
        raise ValueError(
            f"window_length={window_length} exceeds sequence length {T}"
        )
    
    # Static slicing - TPU friendly, compiles once then fast
    windows = [arr[:, i:i+window_length] for i in range(num_possible)]
    return jnp.stack(windows, axis=1)


def _gather_window_slices(arr, indices, window_length):
    """Gather specific windows from ``arr`` using ``indices`` per batch element."""
    # Handle different array dimensions:
    # - 2D: (B, T) - batched 1D sequences (rewards, masks)
    # - 3D: (B, T, D) - batched 2D sequences (embeddings)
    # - 4D: (B, T, ...) - batched higher-dim sequences (actions with extra dims)
    
    original_ndim = arr.ndim
    
    # For 2D arrays (B, T), add a feature dimension to make them (B, T, 1)
    if arr.ndim == 2:
        arr = arr[..., None]  # (B, T) -> (B, T, 1)
    
    all_windows = _extract_sliding_windows(arr, window_length)

    def gather_single(windows, idx):
        return windows[idx]

    selected = jax.vmap(gather_single)(all_windows, indices)
    
    # Remove the added dimension if we added it
    if original_ndim == 2:
        selected = selected.squeeze(-1)  # (B, K, W, 1) -> (B, K, W)
    
    return selected


def _select_random_indices(batch_size, num_possible, num_windows, rng):
    replace = num_windows > num_possible
    rngs = jax.random.split(rng, batch_size)

    def choose(key):
        return jax.random.choice(
            key,
            num_possible,
            shape=(num_windows,),
            replace=replace,
        )

    return jax.vmap(choose)(rngs)


def _select_dissimilar_indices(window_reps, num_windows):
    """Select windows whose representations maximize total dissimilarity."""
    # window_reps should be (B, N, D) where:
    #   B = batch size per device
    #   N = number of possible windows
    #   D = feature dimension
    
    # Handle edge cases where batch dimension might be missing/squeezed
    original_shape = window_reps.shape
    
    if window_reps.ndim == 2:
        # (N, D) - missing batch dim, add it back
        window_reps = window_reps[None, :, :]  # (1, N, D)
        needs_unsqueeze = True
    elif window_reps.ndim == 3:
        needs_unsqueeze = False
    else:
        raise ValueError(f"Expected window_reps to be 2D or 3D, got shape {original_shape}")
    
    # Now window_reps is guaranteed to be (B, N, D)
    B, N, D = window_reps.shape
    
    # Compute pairwise distances: (B, N, N)
    # Expand dims: (B, N, 1, D) - (B, 1, N, D) = (B, N, N, D)
    diffs = window_reps[:, :, None, :] - window_reps[:, None, :, :]
    pairwise = jnp.linalg.norm(diffs, axis=-1)  # (B, N, N)
    
    # Sum distances to get diversity score for each window
    diversity_scores = pairwise.sum(axis=-1)  # (B, N)
    
    def select_topk(scores):
        # Select top-k most diverse windows
        values, idxs = jax.lax.top_k(scores, num_windows)
        return idxs
    
    # Select for each batch element
    selected = jax.vmap(select_topk)(diversity_scores)  # (B, num_windows)
    
    # If we added batch dim, remove it
    if needs_unsqueeze:
        selected = selected[0]  # (num_windows,)
    
    return selected


def _maybe_sample_windows(batch_arrays, sampling_cfg, rng):
    """Slice trajectories into windows according to the sampling configuration."""
    mode = sampling_cfg.get('window_sampling_mode', 'full')
    if mode == 'full':
        return batch_arrays

    window_length = int(sampling_cfg.get('window_size', 8))
    # ADD 1 for next_obs!
    actual_window_length = window_length + 1  # e.g., 9 instead of 8
    
    num_windows = int(sampling_cfg.get('num_windows', 1))
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")

    fused = batch_arrays['fused_embeddings']
    logging.info(f"[window_sampling] mode={mode}, fused.shape={fused.shape}")
    B, T = fused.shape[:2]
    
    # Use actual_window_length for extraction
    num_possible = T - actual_window_length + 1
    if num_possible <= 0:
        raise ValueError(f"window_size+1={actual_window_length} exceeds trajectory length; lower it or adjust data chunks.")

    if num_windows > num_possible:
        raise ValueError(
            f"num_windows={num_windows} exceeds the {num_possible} possible windows"
        )

    if mode == 'random':
        indices = _select_random_indices(B, num_possible, num_windows, rng)
    elif mode == 'dissimilarity':
        # Extract windows of size 9
        fused_windows = _extract_sliding_windows(fused, actual_window_length)
        # Use only first 8 frames for diversity calculation
        window_reps = fused_windows[:, :, :window_length, :].mean(axis=2)
        indices = _select_dissimilar_indices(window_reps, num_windows)
    else:
        raise ValueError(f"Unknown window_sampling_mode '{mode}'")

    def slice_array(arr):
        return _gather_window_slices(arr, indices, actual_window_length)

    sampled = {
        'fused_embeddings': slice_array(fused),  # (B, K, 9, D)
    }

    for key in ('actions', 'rewards', 'masks', 'mc_returns'):
        if key not in batch_arrays or batch_arrays[key] is None:
            continue
        sampled[key] = slice_array(batch_arrays[key])

    def flatten_windows(x):
        b, k = x.shape[:2]
        return x.reshape(b * k, *x.shape[2:])

    sampled = {k: flatten_windows(v) for k, v in sampled.items()}
    return sampled


def _dense_project(seq, params):
    """Applies a Dense layer defined by ``params`` to a batched sequence."""
    return jnp.einsum('btd,df->btf', seq, params['kernel']) + params['bias']


def _run_differentiable_ttt_adaptation(
    fused_seq,
    ttt_params,
    window_size,
    adapt_lr,
    adapt_steps,
    reset=True,
):
    """Unroll the TTT inner loop so gradients flow through projection heads."""
    if fused_seq.ndim != 3:
        raise ValueError(f"Expected fused_seq to have rank 3, got {fused_seq.shape}")

    B, T, _ = fused_seq.shape
    if T == 0:
        proj_dim = ttt_params['P_Q']['kernel'].shape[-1]
        return jnp.empty((B, 0, proj_dim)), jnp.empty((0, max(1, adapt_steps)))

    projection_dim = ttt_params['P_Q']['kernel'].shape[-1]
    ttt_module = TTTModule(input_dim=projection_dim)

    corrupted_seq = _dense_project(fused_seq, ttt_params['P_K'])
    target_seq = _dense_project(fused_seq, ttt_params['P_V'])
    query_seq = _dense_project(fused_seq, ttt_params['P_Q'])

    effective_window = max(1, min(int(window_size), T))
    inner_steps = max(1, int(adapt_steps))
    base_f_adapt = ttt_params['f_adapt']

    def reconstruction_loss(params, corrupted_window, target_window, mask):
        recon = ttt_module.apply({'params': params}, corrupted_window)
        diff = (recon - target_window) * mask
        denom = jnp.maximum(jnp.sum(mask) * diff.shape[-1], 1.0)
        return jnp.sum(diff ** 2) / denom

    time_indices = jnp.arange(T, dtype=jnp.int32)

    def body_fn(carry_params, t_idx):
        start_params = base_f_adapt if reset else carry_params
        slice_start = jnp.maximum(0, t_idx - effective_window + 1)
        corrupted_window = jax.lax.dynamic_slice_in_dim(
            corrupted_seq, slice_start, effective_window, axis=1
        )
        target_window = jax.lax.dynamic_slice_in_dim(
            target_seq, slice_start, effective_window, axis=1
        )

        valid_len = jnp.minimum(t_idx - slice_start + 1, effective_window)
        mask_1d = (jnp.arange(effective_window) < valid_len).astype(fused_seq.dtype)
        mask = jnp.broadcast_to(mask_1d[None, :, None], corrupted_window.shape)

        def gd_step(params, _):
            loss, grads = jax.value_and_grad(reconstruction_loss)(
                params, corrupted_window, target_window, mask
            )
            params = jax.tree_map(lambda p, g: p - adapt_lr * g, params, grads)
            return params, loss

        adapted_params, losses = jax.lax.scan(
            gd_step,
            start_params,
            xs=None,
            length=inner_steps,
        )

        query_proj = jax.lax.dynamic_slice_in_dim(query_seq, t_idx, 1, axis=1)
        adapted_feat = ttt_module.apply({'params': adapted_params}, query_proj)
        next_carry = base_f_adapt if reset else adapted_params
        return next_carry, (adapted_feat, losses)

    _, (adapted_seq, loss_seq) = jax.lax.scan(
        body_fn,
        base_f_adapt,
        time_indices,
    )

    adapted_features = jnp.swapaxes(adapted_seq, 0, 1).reshape(B, T, projection_dim)
    return adapted_features, loss_seq


class TTTFeatureExtractor(nn.Module):
    """
    TTT-based feature extractor that preprocesses OCTO embeddings.
    
    This module learns to adapt fused OCTO features via self-supervision before
    feeding them to a downstream RL agent (like CQL).
    """
    octo_feature_dim: int  # 384 for octo-small, 512 for octo-base
    projection_dim: int = 64  # Dimension after projection heads
    
    def setup(self):
        # Projection heads
        self.P_K = nn.Dense(self.projection_dim, name='P_K')  # Corruption view
        self.P_V = nn.Dense(self.projection_dim, name='P_V')  # Target view
        self.P_Q = nn.Dense(self.projection_dim, name='P_Q')  # Query view
        
        # TTT adaptation module
        self.f_adapt = TTTModule(input_dim=self.projection_dim, name='f_adapt')
    
    def __call__(self, fused_embeddings, train=False):
        """
        Extract adapted features.
        
        Args:
            fused_embeddings: (B, T, octo_feature_dim) - OCTO fused features
            train: Whether in training mode
            
        Returns:
            adapted_features: (B, T, projection_dim)
        """
        # Project to query space
        z = self.P_Q(fused_embeddings)
        
        # Adapt features
        z_adapted = self.f_adapt(z)
        
        return z_adapted
    
    def compute_self_supervised_loss(self, fused_embeddings, train=True):
        """
        Compute self-supervised reconstruction loss for TTT.
        
        Args:
            fused_embeddings: (B, T, octo_feature_dim)
            train: If True, P_K and P_V are trainable; if False, frozen
            
        Returns:
            loss: Scalar MSE reconstruction loss
        """
        # Project to corruption and target views
        corrupted = self.P_K(fused_embeddings)
        target = self.P_V(fused_embeddings)
        
        if not train:
            # During test-time adaptation: P_K, P_V outputs are frozen
            corrupted = jax.lax.stop_gradient(corrupted)
            target = jax.lax.stop_gradient(target)
        
        # Reconstruct target from corrupted input
        recon = self.f_adapt(corrupted)
        loss = jnp.mean((recon - target) ** 2)
        
        return loss


def create_ttt_agent(
    rng,
    fused_example,  # (B, T, octo_dim)
    actions_example,  # (B, T, action_dim)
    octo_feature_dim,
    projection_dim,
    agent_config,
    octo_model=None,
):
    """
    Create a combined TTT + RL agent (works with any agent type: CQL, IQL, CalQL, etc.).
    
    Args:
        rng: Random key
        fused_example: Example fused embeddings
        actions_example: Example actions
        octo_feature_dim: OCTO feature dimension
        projection_dim: TTT projection dimension
        agent_config: Full config dict with agent type and parameters
        octo_model: OCTO model (required for EmbeddingCQLAgent)
        
    Returns:
        ttt_extractor: TTT feature extractor
        rl_agent: RL agent (CQL/IQL/CalQL/etc.)
        params: Combined parameters dict
    """
    # Initialize TTT feature extractor
    ttt_extractor = TTTFeatureExtractor(
        octo_feature_dim=octo_feature_dim,
        projection_dim=projection_dim,
    )
    
    rng, ttt_rng = jax.random.split(rng)
    
    # Initialize with a dummy forward pass that uses ALL parameters
    # We need to initialize both P_Q (used in __call__) and P_K, P_V (used in compute_self_supervised_loss)
    def init_all_params(module, x, train):
        """Initialize all parameters by calling both methods."""
        # This will initialize P_Q and f_adapt
        _ = module(x, train=train)
        # This will initialize P_K and P_V (and re-use f_adapt)
        _ = module.compute_self_supervised_loss(x, train=train)
        return None
    
    ttt_vars = ttt_extractor.init(
        ttt_rng,
        fused_example,
        train=True,
        method=init_all_params
    )
    ttt_params = ttt_vars['params']
    
    # Get adapted features for agent initialization
    adapted_features = ttt_extractor.apply(
        {'params': ttt_params},
        fused_example,
        train=True
    )
    
    # Reshape for agent: (B, T, projection_dim) → flatten to (B*T, projection_dim)
    B, T = adapted_features.shape[:2]
    adapted_flat = adapted_features.reshape(-1, projection_dim)
    actions_flat = actions_example.reshape(-1, actions_example.shape[-1])
    
    # Create observation dict with adapted features as "image"
    # The agent will see these 64-dim adapted features as the "image" observation
    obs_for_agent = {
        'image': adapted_flat,
    }
    
    # Create goals dict (empty for now, but needed by encoder)
    goals_for_agent = {}
    
    # Initialize RL agent with adapted features
    # Works with any agent type: cql, cqlfix, gc_iql, gc_ddpm_bc, etc.
    agent_type = agent_config['agent']
    agent_kwargs = dict(agent_config['agent_kwargs'])  # Convert to mutable dict
    
    logging.info(f"Initializing {agent_type} agent with TTT features (projection_dim={projection_dim})")
    
    rng, agent_rng = jax.random.split(rng)
    
    # Add octo_model to agent_kwargs if needed (for EmbeddingCQLAgent)
    if octo_model is not None:
        agent_kwargs['octo_model'] = octo_model
    
    rl_agent = agents[agent_type].create(
        rng=agent_rng,
        observations=obs_for_agent,
        goals=goals_for_agent,
        actions=actions_flat,
        **agent_kwargs,
    )
    
    # Combine parameters
    combined_params = {
        'ttt': ttt_params,
        'agent': rl_agent.state.params,  # Get params from state
    }
    
    logging.info(f"Created TTT + {agent_type} agent successfully")
    
    return ttt_extractor, rl_agent, ttt_params


def create_ttt_agent_update_step(
    ttt_extractor,
    lambda_self=0.5,
    train_config=None,
    lambda_rl=1.0,
    rl_loss_terms: Sequence[str] = ("critic","actor","temperature"),
):
    """Builds the pure-JAX update used to train the TTT module.

    The returned function optimizes only the TTT parameters. It slices windows,
    computes the self-supervised loss, generates adapted features, and returns
    the constructed RL batch so the caller can run the RL agent's internal
    update separately (preserving the agent's own optimizers and learning rates).
    """

    sampling_cfg = train_config or {}
    _ = ttt_extractor  # Preserved for API compatibility with previous callsites.

    adapt_window = int(sampling_cfg.get('ttt_adapt_window', sampling_cfg.get('window_size', 8)))
    adapt_steps = int(sampling_cfg.get('ttt_adapt_steps', 5))
    adapt_lr = float(sampling_cfg.get('ttt_adapt_lr', 1e-2))
    adapt_reset = bool(sampling_cfg.get('ttt_adapt_reset', True))

    rl_loss_terms = tuple(rl_loss_terms or ())

    def _compute_rl_loss(rl_agent, batch, rng):
        if rl_agent is None or not rl_loss_terms:
            return 0.0, {}

        rl_loss = 0.0
        rl_metrics = {}
        params = rl_agent.state.params

        for term in rl_loss_terms:
            rng, term_rng = jax.random.split(rng)
            if term == 'critic':
                term_loss, term_info = rl_agent.critic_loss_fn(
                    batch,
                    params,
                    term_rng,
                    train=True,
                )
            elif term == 'actor':
                term_loss, term_info = rl_agent.policy_loss_fn(
                    batch,
                    params,
                    term_rng,
                )
            elif term == 'temperature':
                term_loss, term_info = rl_agent.temperature_loss_fn(
                    batch,
                    params,
                    term_rng,
                )
            else:
                raise ValueError(f"Unknown rl_loss_terms entry '{term}'")

            rl_loss = rl_loss + term_loss
            rl_metrics[f"{term}_loss"] = term_loss
            rl_metrics[f"{term}_info"] = term_info

        return rl_loss, rl_metrics

    def loss_fn(ttt_params, batch, rng, rl_agent):
        fused = batch['fused_embeddings']
        B, T = fused.shape[:2]

        def ensure_bt_dim(x, name):
            if x is None:
                return None
            if x.ndim == 1:
                if x.shape[0] == B * T:
                    return x.reshape(B, T)
                if x.shape[0] == B:
                    return jnp.broadcast_to(x[:, None], (B, T))
            if x.ndim == 2:
                if x.shape == (B, T):
                    return x
                if x.shape[0] == B and x.shape[1] == 1:
                    return jnp.broadcast_to(x, (B, T))
            if x.ndim == 3 and x.shape[-1] == 1:
                return ensure_bt_dim(x.squeeze(-1), name)
            logging.warning(
                "Unexpected shape %s for %s, reshaping to (%d, %d)",
                x.shape,
                name,
                B,
                T,
            )
            return x.reshape(B, T)

        rewards_bt = jnp.asarray(ensure_bt_dim(batch['rewards'], 'rewards'))
        masks_bt = jnp.asarray(ensure_bt_dim(batch['masks'], 'masks'))
        mc_returns_bt = (
            jnp.asarray(ensure_bt_dim(batch.get('mc_returns'), 'mc_returns'))
            if 'mc_returns' in batch and batch.get('mc_returns') is not None
            else None
        )

        # 1) TTT self-supervised reconstruction loss over full trajectory
        corrupted = fused @ ttt_params['P_K']['kernel'] + ttt_params['P_K']['bias']
        target = fused @ ttt_params['P_V']['kernel'] + ttt_params['P_V']['bias']
        ttt_module = TTTModule(input_dim=corrupted.shape[-1])
        recon = ttt_module.apply({'params': ttt_params['f_adapt']}, corrupted)
        loss_ttt = jnp.mean((recon - target) ** 2)

        # 2) Window sampling for RL transitions
        rng, sampling_rng = jax.random.split(rng)
        window_batch = _maybe_sample_windows(
            {
                'fused_embeddings': fused,
                'actions': jnp.asarray(batch['actions']),
                'rewards': rewards_bt,
                'masks': masks_bt,
                'mc_returns': mc_returns_bt,
            },
            sampling_cfg,
            sampling_rng,
        )

        fused_rl = window_batch['fused_embeddings']
        actions_rl = window_batch['actions']
        rewards_rl = window_batch['rewards']
        masks_rl = window_batch['masks']
        mc_returns_rl = window_batch.get('mc_returns')

        # 3) Differentiable inner-loop adaptation so RL loss influences P_K/P_V
        adapted_full, inner_losses = _run_differentiable_ttt_adaptation(
            fused_rl,
            ttt_params,
            window_size=adapt_window,
            adapt_lr=adapt_lr,
            adapt_steps=adapt_steps,
            reset=adapt_reset,
        )

        adapted_features = adapted_full[:, :-1, :]
        adapted_next_features = adapted_full[:, 1:, :]

        # Track feature statistics to monitor representation collapse
        critic_feat_mean = jnp.mean(adapted_features)
        critic_feat_std = jnp.std(adapted_features)
        critic_feat_batch_std = jnp.mean(jnp.std(adapted_features, axis=(0, 1)))
        pk_mean = jnp.mean(corrupted)
        pk_std = jnp.std(corrupted)
        pk_batch_std = jnp.mean(jnp.std(corrupted, axis=(0, 1)))

        flat_size = adapted_features.shape[0] * adapted_features.shape[1]
        adapted_flat = adapted_features.reshape(flat_size, -1)
        adapted_next_flat = adapted_next_features.reshape(flat_size, -1)

        actions_flat = actions_rl[:, :-1, :].reshape(flat_size, -1)
        rewards_flat = rewards_rl[:, :-1].reshape(flat_size)
        masks_flat = masks_rl[:, :-1].reshape(flat_size)

        agent_batch = {
            'observations': {
                'image': adapted_flat,
            },
            'next_observations': {
                'image': adapted_next_flat,
            },
            'actions': actions_flat,
            'rewards': rewards_flat,
            'masks': masks_flat,
            'goals': {},
        }

        if mc_returns_rl is not None:
            mc_returns_flat = mc_returns_rl[:, :-1].reshape(flat_size)
            agent_batch['mc_returns'] = mc_returns_flat

        loss_total = lambda_self * loss_ttt
        metrics = {
            'loss_total': loss_total,
            'loss_ttt': loss_ttt,
            'critic_feat_mean': critic_feat_mean,
            'critic_feat_std': critic_feat_std,
            'critic_feat_batch_std': critic_feat_batch_std,
            'pk_mean': pk_mean,
            'pk_std': pk_std,
            'pk_batch_std': pk_batch_std,
            'ttt_inner_loss_variance': jnp.std(inner_losses), 
        }
        
        pv_mean = jnp.mean(target)
        pv_std = jnp.std(target)
        pv_batch_std = jnp.mean(jnp.std(target, axis=(0, 1)))

        # P_K vs P_V similarity (should be high = collapse)
        pk_pv_cosine = jnp.mean(
            jnp.sum(corrupted * target, axis=-1) / 
            (jnp.linalg.norm(corrupted, axis=-1) * jnp.linalg.norm(target, axis=-1) + 1e-8)
        )

        # Adapted features: per-dimension variance
        adapted_per_dim_var = jnp.var(adapted_features.reshape(-1, adapted_features.shape[-1]), axis=0)
        metrics['critic_feat_min_var'] = jnp.min(adapted_per_dim_var)
        metrics['critic_feat_max_var'] = jnp.max(adapted_per_dim_var)
        metrics['ttt_inner_loss_mean'] = jnp.mean(inner_losses)
        metrics['ttt_inner_loss_last'] = jnp.mean(inner_losses[:, -1])

        # Effective rank (expensive but informative)
        U, S, Vt = jnp.linalg.svd(adapted_flat, full_matrices=False)
        normalized_S = S / (jnp.sum(S) + 1e-8)
        effective_rank = jnp.exp(-jnp.sum(normalized_S * jnp.log(normalized_S + 1e-8)))
        metrics['critic_feat_eff_rank'] = effective_rank

        if lambda_rl > 0.0 and rl_loss_terms:
            rng, rl_rng = jax.random.split(rng)
            rl_loss, rl_info = _compute_rl_loss(rl_agent, agent_batch, rl_rng)
            loss_total = loss_total + lambda_rl * rl_loss
            metrics['loss_total'] = loss_total
        #     metrics['loss_rl'] = rl_loss
        #     for key, value in rl_info.items():
        #         metrics[f"rl/{key}"] = value

        return loss_total, (metrics, agent_batch)

    def update_step(ttt_params, opt_state, batch, rng, tx, rl_agent):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, agent_batch)), grads = grad_fn(ttt_params, batch, rng, rl_agent)

        updates, new_opt_state = tx.update(grads, opt_state, ttt_params)
        new_params = optax.apply_updates(ttt_params, updates)

        return new_params, new_opt_state, metrics, agent_batch

    return update_step


def test_time_adapt_ttt(
    ttt_params,
    ttt_extractor,
    fused,
    ttt_lr=1e-2,
    ttt_steps=5
):
    """
    Adapt TTT module at test time using self-supervised learning.
    
    Args:
        ttt_params: TTT module parameters
        ttt_extractor: TTT feature extractor module
        fused: (B, T, octo_dim) test trajectory
        ttt_lr: Adaptation learning rate
        ttt_steps: Number of adaptation steps
        
    Returns:
        adapted_params: Adapted TTT parameters
        losses: Adaptation losses
    """
    # Extract only f_adapt parameters for adaptation
    f_adapt_params = ttt_params['f_adapt']
    
    # Get frozen projections
    P_K_params = ttt_params['P_K']
    P_V_params = ttt_params['P_V']
    
    # Manually apply projections
    corrupted = fused @ P_K_params['kernel'] + P_K_params['bias']
    target = fused @ P_V_params['kernel'] + P_V_params['bias']
    
    # Run adaptation on f_adapt only
    adapted_f_adapt_params, losses = ttt_adaptation(
        f_adapt_params, corrupted, target, ttt_lr, ttt_steps
    )
    
    # Return full params with updated f_adapt
    adapted_params = ttt_params.copy()
    adapted_params['f_adapt'] = adapted_f_adapt_params
    
    return adapted_params, losses


def windowed_test_time_adapt_ttt(
    ttt_params,
    fused,
    window_size=8,
    ttt_lr=1e-2,
    ttt_steps=5,
    reset=True,
):
    """Sliding-window TTT adaptation akin to the PyTorch implementation.

    For each timestep we:
        1) Select the most recent ``window_size`` fused embeddings.
        2) Adapt ``f_adapt`` via ``ttt_adaptation`` on the corruption/target projections.
        3) Produce adapted features for the current timestep using the updated ``f_adapt``.

    Args:
        ttt_params: Full parameter tree of ``TTTFeatureExtractor``.
        fused: (B, T, octo_dim) fused OCTO embeddings.
        window_size: Number of timesteps per adaptation window.
        ttt_lr: Learning rate for each adaptation step.
        ttt_steps: Number of inner-loop steps per window.
        reset: If True, restart from the base ``f_adapt`` for every timestep (local
            adaptation). If False, carry parameters forward, matching the "online"
            recursive variant.

    Returns:
        adapted_features: (B, T, projection_dim) features produced after sliding-window
            adaptation.
        adapted_params: Parameter tree with ``f_adapt`` set to the final value
            (useful when ``reset=False``).
        window_losses: (T, ttt_steps) array of adaptation losses for diagnostics.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    P_K_params = ttt_params['P_K']
    P_V_params = ttt_params['P_V']
    P_Q_params = ttt_params['P_Q']
    projection_dim = P_Q_params['kernel'].shape[-1]
    ttt_module = TTTModule(input_dim=projection_dim)

    base_f_adapt = ttt_params['f_adapt']
    if hasattr(base_f_adapt, 'unfreeze'):
        base_f_adapt = base_f_adapt.unfreeze()
    else:
        base_f_adapt = unfreeze(base_f_adapt)

    B, T = fused.shape[:2]
            
    effective_window = min(window_size, T)

    def project(seq, params):
        return jnp.einsum('btd,df->btf', seq, params['kernel']) + params['bias']

    corrupted_seq = project(fused, P_K_params)
    target_seq = project(fused, P_V_params)
    query_seq = project(fused, P_Q_params)

    time_indices = jnp.arange(T, dtype=jnp.int32)

    def body_fn(carry, t_idx):
        carry_params, _ = carry
        start_params = base_f_adapt if reset else carry_params  # ← CRITICAL LINE
        
        start_idx = jnp.maximum(0, t_idx - window_size + 1)
        slice_start = start_idx
        corrupted_window = jax.lax.dynamic_slice_in_dim(
            corrupted_seq, slice_start, effective_window, axis=1
        )
        target_window = jax.lax.dynamic_slice_in_dim(
            target_seq, slice_start, effective_window, axis=1
        )

        valid_len = jnp.minimum(t_idx - slice_start + 1, effective_window)
        mask_1d = (jnp.arange(effective_window) < valid_len).astype(fused.dtype)
        mask = jnp.broadcast_to(mask_1d[None, :, None], (B, effective_window, 1))

        adapted_params, losses = ttt_adaptation(
            start_params,  # ← Now this will work
            corrupted_window,
            target_window,
            ttt_lr,
            ttt_steps,
            mask,
        )

        query_proj = jax.lax.dynamic_slice_in_dim(query_seq, t_idx, 1, axis=1)
        adapted_feat = ttt_module.apply({'params': freeze(adapted_params)}, query_proj)

        next_carry_params = (
            base_f_adapt if reset else adapted_params
        )
        next_carry = (next_carry_params, adapted_params)
        return next_carry, (adapted_feat, losses)

    initial_carry = (base_f_adapt, base_f_adapt)
    final_carry, (adapted_seq, loss_seq) = jax.lax.scan(
        body_fn,
        initial_carry,
        time_indices,
    )

    adapted_features = (
        jnp.swapaxes(adapted_seq, 0, 1).reshape(B, T, projection_dim)
        if T > 0
        else jnp.empty((B, 0, projection_dim))
    )
    window_losses = loss_seq if T > 0 else jnp.empty((0, ttt_steps))

    adapted_params = ttt_params.copy()
    adapted_params['f_adapt'] = freeze(final_carry[1])

    return adapted_features, adapted_params, window_losses