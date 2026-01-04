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
from jaxrl_m.networks.actor_critic_nets import Critic, Policy, ensemblize
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


class SimSiamModule(nn.Module):
    """
    SimSiam module containing projector, predictor, and dynamics.

    NOTE: This module does NOT contain its own encoder. It receives
    pre-encoded latent representations as input. The encoder is owned by
    the actor (shared with critic), and SimSiam gradients flow back through
    it when simsiam_loss_fn calls the actor's encoder.

    This design ensures true parameter sharing (EfficientZero style):
    - Actor owns the encoder (modules_actor/encoder)
    - Critic shares it (same instance, Flax deduplicates)
    - SimSiam operates on latents, gradients flow back through actor's encoder
    """
    projector: nn.Module
    predictor: nn.Module
    dynamics: nn.Module

    @nn.compact
    def __call__(self, z_t, z_tp1, actions, train: bool = True):
        """
        Forward pass for SimSiam loss computation on pre-encoded latents.

        Args:
            z_t: Encoded current observations (from actor's encoder)
            z_tp1: Encoded next observations (from actor's encoder)
            actions: Actions taken
            train: Whether in training mode

        Returns:
            Dict with all intermediate representations needed for loss
        """
        # Predict next state using dynamics: g(z_t, a_t) → ẑ_{t+1}
        z_tp1_pred = self.dynamics(z_t, actions, train)

        # Project both real and predicted next states
        proj_real = self.projector(z_tp1, train)
        proj_pred = self.projector(z_tp1_pred, train)

        # Apply predictor to predicted projection (asymmetric architecture)
        prediction = self.predictor(proj_pred, train)

        return {
            "z_t": z_t,
            "z_tp1": z_tp1,
            "z_tp1_pred": z_tp1_pred,
            "proj_real": proj_real,
            "proj_pred": proj_pred,
            "prediction": prediction,
        }


class CQLSimSiamAgent(ContinuousCQLAgent):
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

    def simsiam_loss_fn(
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

        # Step 2: Forward pass through SimSiam module (dynamics/projector/predictor)
        if uses_batch_norm and train:
            # During training with batch norm, we need mutable batch_stats
            simsiam_out, updates = self.state.apply_fn(
                variables,
                z_t,
                z_tp1,
                actions,
                train,
                name="simsiam",
                mutable=["batch_stats"],
            )
            # Store updated batch stats in info for later retrieval
            new_batch_stats = updates.get("batch_stats", None)
        else:
            simsiam_out = self.state.apply_fn(
                variables,
                z_t,
                z_tp1,
                actions,
                train,
                name="simsiam",
            )
            new_batch_stats = None

        z_tp1_pred = simsiam_out["z_tp1_pred"]
        proj_real = simsiam_out["proj_real"]
        proj_pred = simsiam_out["proj_pred"]
        prediction = simsiam_out["prediction"]

        # Stop gradient on target (standard SimSiam)
        if self.config.get("stop_grad_target", True):
            proj_target = jax.lax.stop_gradient(proj_real)
        else:
            proj_target = proj_real

        # Cosine similarity loss
        loss = cosine_similarity_loss(
            prediction,
            proj_target,
            normalize=self.config.get("normalize_latents", True),
        )

        # Scale by lambda_sim
        lambda_sim = self.config.get("lambda_sim", 2.0)
        scaled_loss = lambda_sim * loss

        # Compute auxiliary metrics
        eps = 1e-8
        pred_normalized = prediction / (jnp.linalg.norm(prediction, axis=-1, keepdims=True) + eps)
        tgt_normalized = proj_target / (jnp.linalg.norm(proj_target, axis=-1, keepdims=True) + eps)
        cosine_sim = jnp.sum(pred_normalized * tgt_normalized, axis=-1).mean()

        info = {
            "simsiam/loss": loss,
            "simsiam/scaled_loss": scaled_loss,
            "simsiam/z_norm_real": jnp.linalg.norm(z_tp1, axis=-1).mean(),
            "simsiam/z_norm_pred": jnp.linalg.norm(z_tp1_pred, axis=-1).mean(),
            "simsiam/proj_norm_real": jnp.linalg.norm(proj_real, axis=-1).mean(),
            "simsiam/proj_norm_pred": jnp.linalg.norm(proj_pred, axis=-1).mean(),
            "simsiam/pred_norm": jnp.linalg.norm(prediction, axis=-1).mean(),
            "simsiam/cosine_sim": cosine_sim,
        }

        # Include new batch stats if available (for batch norm update)
        if new_batch_stats is not None:
            info["_new_batch_stats"] = new_batch_stats

        return scaled_loss, info
    
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
            # For CQL with multiple sampled actions
            return jax.vmap(
                lambda a: self.state.apply_fn(
                    {"params": params},
                    z,  # Pass latents, not observations
                    a,
                    name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                ),
                in_axes=1,
                out_axes=-1,
            )(actions)
        else:
            return self.state.apply_fn(
                {"params": params},
                z,  # Pass latents, not observations
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

        # Add SimSiam loss if enabled
        if self.config.get("lambda_sim", 0.0) > 0:
            losses["simsiam"] = partial(self.simsiam_loss_fn, batch)

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

        # Add SimSiam if loss is enabled
        if self.config.get("lambda_sim", 0.0) > 0:
            networks_to_update.add("simsiam")

        # Pass as regular set - parent class will add temperature/cql_alpha as needed
        new_agent, info = super().update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=networks_to_update,
        )

        # Update batch stats if batch norm is used
        uses_batch_norm = self.config.get("uses_batch_norm", False)
        if uses_batch_norm and "simsiam" in networks_to_update:
            # Do a forward pass to get updated batch stats
            # First encode, then pass to simsiam
            obs = self._include_goals_in_obs(batch, "observations")
            next_obs = self._include_goals_in_obs(batch, "next_observations")
            actions = batch["actions"]

            z_t = self._encode(obs, new_agent.state.params)
            z_tp1 = self._encode(next_obs, new_agent.state.params)

            variables = {"params": new_agent.state.params}
            if new_agent.state.batch_stats is not None:
                variables["batch_stats"] = new_agent.state.batch_stats

            _, updates = new_agent.state.apply_fn(
                variables,
                z_t,
                z_tp1,
                actions,
                True,  # train=True
                name="simsiam",
                mutable=["batch_stats"],
            )

            new_batch_stats = updates.get("batch_stats", None)
            if new_batch_stats is not None:
                new_state = new_agent.state.update_batch_stats(new_batch_stats)
                new_agent = new_agent.replace(state=new_state)

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
        simsiam_params: Params,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute TTT adaptation loss (SimSiam) with only simsiam module gradients.
        
        Returns:
            loss: Scalar SimSiam loss
            grads: Gradients w.r.t. simsiam parameters only
        """
        def loss_fn(sim_params):
            params = self.state.params.copy({"simsiam": sim_params})
            loss, _ = self.simsiam_loss_fn(
                batch,
                params,
                jax.random.PRNGKey(0),
                train=False,
            )
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(simsiam_params)
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

        # Add SimSiam metrics if enabled
        if self.config.get("lambda_sim", 0.0) > 0:
            _, simsiam_info = self.simsiam_loss_fn(
                batch,
                self.state.params,
                jax.random.PRNGKey(0),
                train=False,
            )
            metrics.update(simsiam_info)

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
        simsiam_kwargs: dict = {},
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
            Critic, encoder=None, network=critic_backbone  # No encoder - receives latents
        )(name="critic")

        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )

        # =======================================================================
        # Create SimSiam sub-networks (receives latents from encoder)
        # =======================================================================
        sim_cfg = {
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
        }
        sim_cfg.update(simsiam_kwargs)

        projector_def = SimSiamProjector(
            hidden_dims=tuple(sim_cfg["projector_hidden_dims"]),
            output_dim=sim_cfg["projector_output_dim"],
            norm_type=sim_cfg["projector_norm"],
        )

        predictor_def = SimSiamPredictor(
            hidden_dims=tuple(sim_cfg["predictor_hidden_dims"]),
            output_dim=sim_cfg["predictor_output_dim"],
            norm_type=sim_cfg["predictor_norm"],
        )

        if sim_cfg["use_conv_dynamics"]:
            dynamics_def = DynamicsNetworkConv(
                latent_channels=sim_cfg["dynamics_latent_channels"],
                action_dim=action_dim,
                num_residual_blocks=sim_cfg["dynamics_num_residual_blocks"],
                norm_type=sim_cfg["dynamics_norm"],
            )
        else:
            dynamics_def = DynamicsNetworkFlat(
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=sim_cfg["dynamics_hidden_dim"],
                num_residual_blocks=sim_cfg["dynamics_num_residual_blocks"],
                norm_type=sim_cfg["dynamics_norm"],
            )

        simsiam_module = SimSiamModule(
            projector=projector_def,
            predictor=predictor_def,
            dynamics=dynamics_def,
            name="simsiam",
        )

        # =======================================================================
        # Build complete network dict with encoder as first-class module
        # =======================================================================
        networks = {
            "encoder": encoder_module,  # Standalone encoder with its own optimizer
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
            "simsiam": simsiam_module,
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
            "simsiam": make_optimizer(**config.simsiam_optimizer_kwargs),
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
            "simsiam": [dummy_latent, dummy_latent, actions, True],
        }

        if config["cql_autotune_alpha"]:
            init_kwargs["cql_alpha_lagrange"] = []

        # Check if batch norm is used in simsiam
        uses_batch_norm = (
            sim_cfg.get("projector_norm") == "batch"
            or sim_cfg.get("predictor_norm") == "batch"
            or sim_cfg.get("dynamics_norm") == "batch"
        )

        if uses_batch_norm:
            # Initialize with mutable batch_stats
            variables = model_def.init(init_rng, **init_kwargs)
            params = variables.get("params", variables)
            batch_stats = variables.get("batch_stats", None)
        else:
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
        # Store whether batch norm is used for simsiam
        config["uses_batch_norm"] = uses_batch_norm
        config = flax.core.FrozenDict(config)
        
        return cls(state, config)


def get_default_config(updates=None):
    """Get default config for CQL+SimSiam agent."""
    # Start with CQL defaults
    config = get_cql_default_config()

    # Add SimSiam-specific defaults
    config.lambda_sim = 2.0  # EfficientZero uses λ_3 = 2
    config.stop_grad_target = True
    config.normalize_latents = True

    # Encoder optimizer (shared encoder updated by all losses)
    config.encoder_optimizer_kwargs = ConfigDict({
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
    })

    # SimSiam optimizer (for dynamics/projector/predictor only)
    config.simsiam_optimizer_kwargs = ConfigDict({
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
    })

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config