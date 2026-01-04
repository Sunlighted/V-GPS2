from __future__ import annotations

from functools import partial
from typing import Optional, Dict, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from overrides import overrides

from experiments.ttt_module import TTTModule  # pylint: disable=import-error
from jaxrl_m.agents.continuous.cqlfix import EmbeddingCQLAgent, get_default_config as get_cql_config
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, Policy, ensemblize
from jaxrl_m.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from jaxrl_m.networks.mlp import MLP


class TTTFeatureExtractor(nn.Module):
    """TTT-based projection + adaptation block."""

    octo_feature_dim: int
    projection_dim: int

    def setup(self):
        self.P_K = nn.Dense(self.projection_dim, name="P_K")
        self.P_V = nn.Dense(self.projection_dim, name="P_V")
        self.P_Q = nn.Dense(self.projection_dim, name="P_Q")
        self.f_adapt = TTTModule(input_dim=self.projection_dim, name="f_adapt")

    def __call__(self, fused_embeddings: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        z = self.P_Q(fused_embeddings)
        return self.f_adapt(z)

    def compute_self_supervised_loss(
        self, fused_embeddings: jnp.ndarray, train: bool = True
    ) -> jnp.ndarray:
        corrupted = self.P_K(fused_embeddings)
        target = self.P_V(fused_embeddings)
        recon = self.f_adapt(corrupted)
        return jnp.mean((recon - target) ** 2)


def _ensure_bt_shape(fused: jnp.ndarray) -> jnp.ndarray:
    if fused.ndim == 2:
        return fused[:, None, :]
    if fused.ndim == 3:
        return fused
    raise ValueError(
        f"Expected fused embeddings to have shape (B, D) or (B, T, D), got {fused.shape}"
    )


class TTTEncoderModule(nn.Module):
    """Wraps ``TTTFeatureExtractor`` and flattens the (B, T, D) output to (B, T*D)."""

    octo_feature_dim: int
    projection_dim: int

    def setup(self):
        self.extractor = TTTFeatureExtractor(
            octo_feature_dim=self.octo_feature_dim,
            projection_dim=self.projection_dim,
        )

    def __call__(self, observations: Dict[str, jnp.ndarray], train: bool = False) -> jnp.ndarray:
        fused = observations.get("fused_embeddings", observations.get("image"))
        if fused is None:
            raise ValueError("Observations must contain 'fused_embeddings' or 'image'")
        fused = _ensure_bt_shape(jnp.asarray(fused))
        adapted = self.extractor(fused, train=train)
        batch, time = adapted.shape[:2]
        return adapted.reshape(batch, time * adapted.shape[-1])

    def compute_self_supervised_loss(
        self, observations: Dict[str, jnp.ndarray], train: bool = True
    ) -> jnp.ndarray:
        fused = observations.get("fused_embeddings", observations.get("image"))
        if fused is None:
            raise ValueError("Observations must contain 'fused_embeddings' or 'image'")
        fused = _ensure_bt_shape(jnp.asarray(fused))
        return self.extractor.compute_self_supervised_loss(fused, train=train)


class CQLTTTAgent(EmbeddingCQLAgent):
    """CQL agent whose encoder is a shared TTT module."""

    ttt_encoder_def: nn.Module = nonpytree_field()
    lambda_self: float = nonpytree_field()

    def _encode_observation_structure(
        self,
        obs_struct: Any,
        *,
        params: Optional[Params] = None,
        train: bool = False,
    ) -> Any:
        if isinstance(obs_struct, tuple):
            obs, goals = obs_struct
            return (self._encode_single(obs, params=params, train=train), goals)
        return self._encode_single(obs_struct, params=params, train=train)

    def _encode_single(
        self,
        obs: Dict[str, jnp.ndarray],
        *,
        params: Optional[Params] = None,
        train: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        encoder_input = {"fused_embeddings": obs.get("fused_embeddings", obs.get("image"))}
        if encoder_input["fused_embeddings"] is None:
            raise ValueError("Observation dict must include 'fused_embeddings' or 'image'")
        encoded = self.state.apply_fn(
            {"params": params or self.state.params},
            encoder_input,
            name="ttt_encoder",
            train=train,
        )
        new_obs = dict(obs)
        new_obs["image"] = encoded
        return new_obs

    @overrides
    def _include_goals_in_obs(self, batch, which_obs: str):
        obs_struct = super()._include_goals_in_obs(batch, which_obs)
        train = which_obs == "observations"
        return self._encode_observation_structure(obs_struct, train=train)

    def _ttt_self_supervised_loss(self, batch, params: Params) -> jnp.ndarray:
        obs = batch["observations"]
        if "fused_embeddings" not in obs:
            raise ValueError("Batch observations must contain 'fused_embeddings' for TTT loss")
        encoder_params = params["ttt_encoder"]
        return self.ttt_encoder_def.apply(
            {"params": encoder_params},
            {"fused_embeddings": obs["fused_embeddings"]},
            method=TTTEncoderModule.compute_self_supervised_loss,
            train=True,
        )

    @overrides
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey, train: bool = True):
        critic_loss, info = super().critic_loss_fn(batch, params, rng, train=train)
        if self.lambda_self > 0.0:
            loss_ttt = self._ttt_self_supervised_loss(batch, params)
            critic_loss = critic_loss + self.lambda_self * loss_ttt
            info = dict(info)
            info["ttt_loss"] = loss_ttt
        return critic_loss, info

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        fused_example: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        octo_feature_dim: int,
        projection_dim: int = 64,
        lambda_self: float = 0.5,
        critic_network_kwargs: Optional[Dict[str, Any]] = None,
        policy_network_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        ttt_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **agent_kwargs,
    ) -> "CQLTTTAgent":
        critic_network_kwargs = critic_network_kwargs or {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        }
        policy_network_kwargs = policy_network_kwargs or {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        }
        policy_kwargs = policy_kwargs or {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        }
        ttt_optimizer_kwargs = ttt_optimizer_kwargs or {
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
        }

        config = get_cql_config(updates=agent_kwargs)
        config = config.copy_and_resolve_references()
        config["lambda_self"] = lambda_self
        config["ttt_optimizer_kwargs"] = flax.core.FrozenDict(ttt_optimizer_kwargs)

        ttt_encoder_def = TTTEncoderModule(
            octo_feature_dim=octo_feature_dim,
            projection_dim=projection_dim,
        )

        encoded_dim = projection_dim * fused_example.shape[1]
        obs_example = jnp.zeros((fused_example.shape[0], encoded_dim))

        policy_def = Policy(
            encoder=None,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )
        critic_backbone = ensemblize(
            partial(MLP, **critic_network_kwargs), config.critic_ensemble_size
        )(name="critic_ensemble")
        critic_def = partial(Critic, encoder=None, network=critic_backbone)(name="critic")
        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )
        networks = {
            "ttt_encoder": ttt_encoder_def,
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        if config["cql_autotune_alpha"]:
            cql_alpha_lagrange_def = LeqLagrangeMultiplier(
                init_value=config.cql_alpha_lagrange_init,
                constraint_shape=(),
                name="cql_alpha_lagrange",
            )
            networks["cql_alpha_lagrange"] = cql_alpha_lagrange_def

        model_def = ModuleDict(networks)

        txs = {
            "ttt_encoder": make_optimizer(**ttt_optimizer_kwargs),
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
        }
        if config["cql_autotune_alpha"]:
            txs["cql_alpha_lagrange"] = make_optimizer(
                **config.cql_alpha_lagrange_otpimizer_kwargs
            )

        rng, init_rng = jax.random.split(rng)
        encoder_input = {"fused_embeddings": fused_example}
        extra_kwargs = {}
        if config["cql_autotune_alpha"]:
            extra_kwargs["cql_alpha_lagrange"] = []
        params = model_def.init(
            init_rng,
            ttt_encoder=[encoder_input],
            actor=[obs_example],
            critic=[obs_example, actions],
            temperature=[],
            **extra_kwargs,
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        if config.target_entropy >= 0.0:
            config.target_entropy = -actions.shape[-1]

        config = flax.core.FrozenDict(config)

        return cls(
            state=state,
            config=config,
            ttt_encoder_def=ttt_encoder_def,
            lambda_self=lambda_self,
        )