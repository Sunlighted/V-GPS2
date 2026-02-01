from functools import partial
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax
import copy

import chex
import distrax
import flax.linen as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper, GCEncodingWrapper, LCEncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import CriticVectorField, ActorVectorField, ensemblize
from jaxrl_m.networks.lagrange import GeqLagrangeMultiplier
from jaxrl_m.networks.mlp import MLP

class ValueFlowsAgent(flax.struct.PyTreeNode):
    
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    
    def _include_goals_in_obs(self, batch, which_obs: str):
        assert which_obs in ("observations", "next_observations")
        obs = batch[which_obs]
        if self.config["goal_conditioned"]:
            obs = (obs, batch["goals"])
        return obs
    
    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: Optional[Data],
        times: Optional[Data],
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            times,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )
        
    def forward_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        returns: Optional[jax.Array],
        times: Optional[jax.Array],
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        if jnp.ndim(actions) == 3:
            # forward the q function with multiple actions on each state
            return jax.vmap(
                lambda a: self.state.apply_fn(
                    {"params": grad_params or self.state.params},
                    returns,
                    times,
                    observations,
                    a,
                    name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                ),
                in_axes=1,
                out_axes=-1,
            )(actions)
        else:
            # forward the q function on 1 action on each state
            return self.state.apply_fn(
                {"params": grad_params or self.state.params},
                returns,
                times,
                observations,
                actions,
                name="critic",
                rngs={"dropout": rng} if train else {},
                train=train,
            )
            
    def forward_target_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        returns: Optional[jax.Array],
        times: Optional[jax.Array],
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, returns, times, rng=rng, grad_params=self.state.target_params, train=False
        )
    
    def critic_loss_fn(self, batch, grad_params: Params, rng: PRNGKey):
        """Compute the flow distributional critic loss."""
        batch_size = batch['actions'].shape[0]
        rng, actor_rng, noise_rng, time_rng, q_rng, ret_rng = jax.random.split(rng, 6)

        # Sample next actions using rejection sampling
        next_actions = self.sample_actions(self._include_goals_in_obs(batch, "next_observations"), actor_rng)

        # Using target networks to compute the confidence weights.
        ret_noises = jax.random.normal(ret_rng, (batch_size, 1))
        _, ret_jac_eps_prods1 = self.compute_flow_returns(
            ret_noises, self._include_goals_in_obs(batch, "observations"), batch['actions'],
            target=True, critic_index=0, return_jac_eps_prod=True)
        _, ret_jac_eps_prods2 = self.compute_flow_returns(
            ret_noises, self._include_goals_in_obs(batch, "observations"), batch['actions'],
            target=True, critic_index=1, return_jac_eps_prod=True)
        ret_stds1 = jnp.sqrt(ret_jac_eps_prods1.squeeze(-1) ** 2)
        ret_stds2 = jnp.sqrt(ret_jac_eps_prods2.squeeze(-1) ** 2)
        if self.config['q_agg'] == 'min':
            ret_stds = jnp.minimum(ret_stds1, ret_stds2)
        else:
            ret_stds = (ret_stds1 + ret_stds2) / 2
        weights = jax.nn.sigmoid(-self.config['confidence_weight_temp'] / ret_stds) + 0.5
        weights = jax.lax.stop_gradient(weights)

        # BCFM  regularization loss
        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        next_returns1 = self.compute_flow_returns(
            noises, self._include_goals_in_obs(batch, "next_observations"), next_actions,
            target=True, critic_index=0)
        next_returns2 = self.compute_flow_returns(
            noises, self._include_goals_in_obs(batch, "next_observations"), next_actions,
            target=True, critic_index=1)
        if self.config['ret_agg'] == 'min':
            next_returns = jnp.minimum(next_returns1, next_returns2)
        else:
            next_returns = (next_returns1 + next_returns2) / 2

        # The following returns will be bounded automatically
        returns = (jnp.expand_dims(batch['rewards'], axis=-1) +
                   self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * next_returns)
        noisy_returns = times * returns + (1 - times) * noises
        target_vector_field = returns - noises

        vector_fields = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch['actions'], 
            noisy_returns, 
            times, 
            rng=noise_rng, # dropout rng if needed
            grad_params=grad_params,
            train=True
        )
        vector_fields = vector_fields.reshape(2, batch_size, 1)
        bcfm_loss = ((vector_fields[0] - target_vector_field) ** 2 +
                     (vector_fields[1] - target_vector_field) ** 2).mean(axis=-1)

        # DCFM loss
        noisy_next_returns1 = self.compute_flow_returns(
            noises, self._include_goals_in_obs(batch, "next_observations"), next_actions, end_times=times,
            target=True, critic_index=0)
        noisy_next_returns2 = self.compute_flow_returns(
            noises, self._include_goals_in_obs(batch, "next_observations"), next_actions, end_times=times,
            target=True, critic_index=1)
        if self.config['ret_agg'] == 'min':
            noisy_next_returns = jnp.minimum(noisy_next_returns1, noisy_next_returns2)
        else:
            noisy_next_returns = (noisy_next_returns1 + noisy_next_returns2) / 2
        noisy_returns = (
            jnp.expand_dims(batch['rewards'], axis=-1) +
            self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * noisy_next_returns
        )
        vector_fields_dcfm = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch['actions'], 
            noisy_returns, 
            times, 
            rng=noise_rng,
            grad_params=grad_params,
            train=True
        )
        vector_fields_dcfm = vector_fields_dcfm.reshape(2, batch_size, 1)
        target_vector_fields = self.forward_target_critic(
            self._include_goals_in_obs(batch, "next_observations"),
            next_actions,
            noisy_next_returns,
            times,
            rng=noise_rng
        )
        target_vector_fields = target_vector_fields.reshape(2, batch_size, 1)

        if self.config['ret_agg'] == 'min':
            target_vector_field = jnp.minimum(target_vector_fields[0], target_vector_fields[1])
        else:
            target_vector_field = (target_vector_fields[0] + target_vector_fields[1]) / 2
        dcfm_loss = ((vector_fields_dcfm[0] - target_vector_field) ** 2 +
                     (vector_fields_dcfm[1] - target_vector_field) ** 2).mean(axis=-1)

        critic_loss = (self.config['bcfm_lambda'] * bcfm_loss + self.config['dcfm_lambda'] * dcfm_loss)
        critic_loss = (weights * critic_loss).mean()

        # For logging and confidence weights.
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q_vector_fields = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch['actions'],
            q_noises,
            jnp.zeros_like(q_noises), # t=0
            rng=q_rng,
            grad_params=None, # Logging 不需要梯度，可以使用 default params 或 grad_params
            train=False
        )
        q_vector_fields = q_vector_fields.reshape(2, batch_size, 1)
        qs = q_noises + q_vector_fields 
        q1, q2 = qs[0].squeeze(-1), qs[1].squeeze(-1)
        
        if self.config['clip_flow_returns']:
            limit_min = self.config['min_reward'] / (1 - self.config['discount'])
            limit_max = self.config['max_reward'] / (1 - self.config['discount'])
            q1 = jnp.clip(q1, limit_min, limit_max)
            q2 = jnp.clip(q2, limit_min, limit_max)
        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2
        q_stds = ret_stds

        return critic_loss, {
            'critic_loss': critic_loss,
            'bcfm_loss': bcfm_loss,
            'dcfm_loss': dcfm_loss,
            'q_mean': q.mean(),
            'q_std': q_stds.mean(),
            'q_std_max': q_stds.max(),
            'q_std_min': q_stds.min(),
            'q_max': q.max(),
            'q_min': q.min(),
            'weight': weights.mean(),
        }

    def actor_loss_fn(self, batch, grad_params: Params, rng: PRNGKey):
        """Compute the BC flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.forward_policy(self._include_goals_in_obs(batch, "observations"), x_t, t, rng=rng, grad_params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, info
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss_fn(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss_fn(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info
    
    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.actor_loss_fn, batch)
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic"}
        ),
    ) -> Tuple["ValueFlowsAgent", dict]:
        """Update the agent and return a new agent with information dictionary."""
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, goal_rng = jax.random.split(self.state.rng)
        # if self.config["goal_conditioned"] and self.config["gc_kwargs"]["negative_proportion"] > 0:
        #     new_stats, _neg_goal_masks = self._sample_negative_goals(batch, goal_rng)
        #     # save the new goals and rewards
        #     for k, v in new_stats.items():
        #         batch[k] = v

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info
    
    @partial(jax.jit, static_argnames=('target', 'critic_index', 'return_jac_eps_prod'))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        target=True,
        critic_index=0,
        return_jac_eps_prod=False,
    ):
        """Compute returns from the return flow model using the Euler method."""
        noisy_returns = noises
        if noisy_returns.ndim == 1:
            noisy_returns = noisy_returns.reshape(-1, 1)
            
        noisy_jac_eps_prod = jnp.ones_like(noisy_returns)

        if init_times is None:
            init_times = jnp.zeros(noisy_returns.shape, dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones(noisy_returns.shape, dtype=noisy_returns.dtype)

        init_times = init_times.reshape(-1, 1)
        end_times = end_times.reshape(-1, 1)
        
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def flow_step_fn(current_returns, current_times):
            rng = jax.random.PRNGKey(0) 
            if target:
                vector_fields = self.forward_target_critic(
                    observations, actions, current_returns, current_times, rng
                )
            else:
                vector_fields = self.forward_critic(
                    observations, actions, current_returns, current_times, 
                    rng=rng, train=False
                )

            vf = vector_fields[critic_index]

            return vf.reshape(current_returns.shape)

        def func(carry, i):
            (curr_returns, curr_jac_prod) = carry

            current_step_size = step_size.reshape(curr_returns.shape)
            
            times = i * current_step_size + init_times

            vector_field, jac_eps_prod = jax.jvp(
                lambda r: flow_step_fn(r, times),
                (curr_returns, ),
                (curr_jac_prod, ),
            )

            vector_field = vector_field.reshape(curr_returns.shape)
            jac_eps_prod = jac_eps_prod.reshape(curr_returns.shape)

            new_returns = curr_returns + current_step_size * vector_field
            new_jac_prod = curr_jac_prod + current_step_size * jac_eps_prod
            
            if self.config['clip_flow_returns']:
                min_ret = self.config['min_reward'] / (1 - self.config['discount'])
                max_ret = self.config['max_reward'] / (1 - self.config['discount'])
                new_returns = jnp.clip(new_returns, min_ret, max_ret)

            return (new_returns, new_jac_prod), None

        (final_returns, final_jac_prods), _ = jax.lax.scan(
            func, 
            (noisy_returns, noisy_jac_eps_prod), 
            jnp.arange(self.config['num_flow_steps'])
        )

        if return_jac_eps_prod:
            return final_returns, final_jac_prods
        else:
            return final_returns
        
    @jax.jit
    def compute_flow_actions(
        self,
        noises,       # Shape: (Batch, Num_Samples, Action_Dim)
        observations, # Tuple[Dict, Dict]
        init_times=None,
        end_times=None,
    ):
        """
        Compute actions using jax.lax.map to save VRAM.
        Instead of flattening (B*N), we iterate over N.
        Effective VRAM usage = Batch Size (512), not 512 * 16.
        """
        def transpose_to_scan(x):
            return jnp.swapaxes(x, 0, 1)

        # scan_noises shape: (Num_Samples, Batch, Action_Dim)
        scan_noises = transpose_to_scan(noises)
        # scan_observations shape: Tuple[Dict[N, B, ...], Dict[N, B, ...]]
        scan_observations = jax.tree_util.tree_map(transpose_to_scan, observations)

        batch_size = noises.shape[0]
        dtype = noises.dtype

        def solve_single_sample_slice(slice_noises, slice_obs):
            """
            slice_noises: (Batch, Action_Dim)
            slice_obs: Tuple[Dict(Batch, ...), Dict(Batch, ...)]
            """

            def ode_step(carry, i):
                (curr_actions,) = carry

                t_scalar = i / self.config['num_flow_steps']
                times = jnp.full((batch_size, 1), t_scalar, dtype=dtype)
                
                # Forward Policy
                rng = jax.random.PRNGKey(0) 
                vector_field = self.forward_policy(
                    slice_obs, curr_actions, times, rng=rng, train=False
                )
                
                dt = 1.0 / self.config['num_flow_steps']
                next_actions = curr_actions + vector_field * dt
                
                if self.config['clip_flow_actions']:
                    next_actions = jnp.clip(next_actions, -1, 1)
                    
                return (next_actions,), None

            (final_slice_actions,), _ = jax.lax.scan(
                ode_step, 
                (slice_noises,), 
                jnp.arange(self.config['num_flow_steps'])
            )
            
            return final_slice_actions

        flow_actions_transposed = jax.lax.map(
            lambda args: solve_single_sample_slice(args[0], args[1]),
            (scan_noises, scan_observations)
        )

        if not self.config['clip_flow_actions']:
            flow_actions_transposed = jnp.clip(flow_actions_transposed, -1, 1)

        # (Num_Samples, Batch, Action_Dim) -> (Batch, Num_Samples, Action_Dim)
        flow_actions = jnp.swapaxes(flow_actions_transposed, 0, 1)

        return flow_actions
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions using rejection sampling."""
        obs_dict, goal_dict = observations
        batch_size = obs_dict["image"].shape[0]
        num_samples = self.config['num_samples']
        action_dim = self.config['action_dim']
        
        action_seed, q_seed, ret_seed = jax.random.split(seed, 3)
        actor_noises = jax.random.normal(
            action_seed,
            (batch_size, num_samples, action_dim)
        )
        def expand_fn(x):
            # x shape: (Batch, D1, D2...)
            # expanded: (Batch, 1, D1, D2...)
            # repeated: (Batch, N, D1, D2...)
            return jnp.repeat(
                jnp.expand_dims(x, axis=1),
                num_samples,
                axis=1
            )
        n_obs_dict = jax.tree_util.tree_map(expand_fn, obs_dict)
        n_goal_dict = jax.tree_util.tree_map(expand_fn, goal_dict)
        n_observations = (n_obs_dict, n_goal_dict)
        flow_actions = self.compute_flow_actions(actor_noises, n_observations)
        
        def flatten_fn(x):
            # x shape: (Batch, Num_Samples, D1, D2...)
            # flattened: (Batch * Num_Samples, D1, D2...)
            return x.reshape(-1, *x.shape[2:])

        flat_obs_dict = jax.tree_util.tree_map(flatten_fn, n_obs_dict)
        flat_goal_dict = jax.tree_util.tree_map(flatten_fn, n_goal_dict)
        
        flat_observations = (flat_obs_dict, flat_goal_dict)
        
        # Flatten actions
        flat_actions = flow_actions.reshape(-1, *flow_actions.shape[2:])

        q_noises = jax.random.normal(
            q_seed,
            (batch_size, num_samples, 1)
        )
        flat_q_noises = q_noises.reshape(-1, 1)
        flat_times = jnp.zeros_like(flat_q_noises)
        
        flat_vector_fields = self.forward_critic(
            flat_observations, 
            flat_actions, 
            flat_q_noises, 
            flat_times, 
            rng=q_seed, 
            train=False
        )
        
        vector_fields = flat_vector_fields.reshape(2, batch_size, num_samples)

        base_noise = q_noises.squeeze(-1)
        q1 = base_noise + vector_fields[0]
        q2 = base_noise + vector_fields[1]
        if self.config['clip_flow_returns']:
            q1 = jnp.clip(
                q1,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
            q2 = jnp.clip(
                q2,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2

        if len(q.shape) > 1:
            actions = flow_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = flow_actions[jnp.argmax(q, axis=-1)]

        return actions

    def get_debug_metrics(self, batch, seed=None, **kwargs):
        """Get debug metrics for logging."""
        batch_size = batch["actions"].shape[0]
        rng = seed if seed is not None else jax.random.PRNGKey(0)

        # Sample actions from policy
        pi_actions = self.sample_actions(
            observations=(batch["observations"], batch["goals"]),
            seed=rng
        )

        # Compute MSE between policy actions and ground truth actions
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        # Compute critic loss for logging
        _, critic_metrics = self.critic_loss_fn(batch, self.state.params, rng=rng)

        metrics = {
            "mse": mse,
            "pi_actions": pi_actions,
            **critic_metrics,
        }
        return metrics

    @jax.jit
    def get_q_values(self, observations, goals, actions):
        """Get Q values for given observations, goals, and actions."""
        batch_size = actions.shape[0]
        rng = jax.random.PRNGKey(0)

        # Sample noise for computing Q values
        noises = jax.random.normal(rng, (batch_size, 1))

        # Compute flow returns (Q values)
        q1 = self.compute_flow_returns(
            noises, (observations, goals), actions,
            target=True, critic_index=0
        )
        q2 = self.compute_flow_returns(
            noises, (observations, goals), actions,
            target=True, critic_index=1
        )

        if self.config['clip_flow_returns']:
            limit_min = self.config['min_reward'] / (1 - self.config['discount'])
            limit_max = self.config['max_reward'] / (1 - self.config['discount'])
            q1 = jnp.clip(q1, limit_min, limit_max)
            q2 = jnp.clip(q2, limit_min, limit_max)

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2

        return q.squeeze(-1)

    def get_eval_values(self, traj, seed, goals):
        """Get evaluation values for a trajectory."""
        # Sample actions from policy
        actions = self.sample_actions(
            observations=(traj["observations"], goals),
            seed=seed
        )

        # Compute MSE between policy actions and ground truth actions
        mse = ((actions - traj["actions"]) ** 2).sum(-1)

        batch_size = traj["actions"].shape[0]
        rng = jax.random.PRNGKey(0)

        # Sample noise for computing Q values
        noises = jax.random.normal(rng, (batch_size, 1))

        # Compute Q values using flow returns (with current params)
        q1 = self.compute_flow_returns(
            noises, (traj["observations"], goals), traj["actions"],
            target=False, critic_index=0
        )
        q2 = self.compute_flow_returns(
            noises, (traj["observations"], goals), traj["actions"],
            target=False, critic_index=1
        )

        # Compute target Q values
        target_q1 = self.compute_flow_returns(
            noises, (traj["observations"], goals), traj["actions"],
            target=True, critic_index=0
        )
        target_q2 = self.compute_flow_returns(
            noises, (traj["observations"], goals), traj["actions"],
            target=True, critic_index=1
        )

        if self.config['clip_flow_returns']:
            limit_min = self.config['min_reward'] / (1 - self.config['discount'])
            limit_max = self.config['max_reward'] / (1 - self.config['discount'])
            q1 = jnp.clip(q1, limit_min, limit_max)
            q2 = jnp.clip(q2, limit_min, limit_max)
            target_q1 = jnp.clip(target_q1, limit_min, limit_max)
            target_q2 = jnp.clip(target_q2, limit_min, limit_max)

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
            target_q = jnp.minimum(target_q1, target_q2)
        else:
            q = (q1 + q2) / 2
            target_q = (target_q1 + target_q2) / 2

        metrics = {
            "q": q.squeeze(-1),
            "target_q": target_q.squeeze(-1),
            "mse": mse,
            "rewards": traj["rewards"],
            "masks": traj["masks"],
        }
        return metrics

    def plot_values(self, traj, seed=None, goals=None):
        """Plot evaluation values for a trajectory."""
        if goals is None:
            goals = traj["goals"]
        else:
            traj_len = traj["observations"]["image"].shape[0]

            if goals["language"].shape[0] > traj_len:
                goals = {k: v[:traj_len] for k, v in goals.items()}
            elif goals["language"].shape[0] < traj_len:
                num_repeat = traj_len - goals["language"].shape[0]
                for k, v in goals.items():
                    rep = jnp.repeat(v[-1:], num_repeat, axis=0)
                    goals[k] = jnp.concatenate([v, rep], axis=0)

        goals = traj["goals"] if goals is None else goals
        metrics = self.get_eval_values(traj, seed, goals)
        images = traj["observations"]["image"].squeeze()  # (T, H, W, 3)

        num_rows = len(metrics.keys()) + 1

        fig, axs = plt.subplots(num_rows, 1, figsize=(8, 16))
        canvas = FigureCanvas(fig)
        plt.xlim(0, len(metrics["rewards"]))

        interval = images.shape[0] // 8
        interval = max(1, interval)
        sel_images = images[::interval]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1)  # (200, 8*200, 3)
        axs[0].imshow(sel_images)

        for i, (key, metric_val) in enumerate(metrics.items()):
            row = i + 1
            axs[row].plot(metric_val, linestyle='--', marker='o')
            axs[row].set_ylim([metric_val.min(), metric_val.max()])
            axs[row].set_ylabel(key)
        plt.tight_layout()
        canvas.draw()  # draw the canvas, cache the renderer
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        # goal conditioned
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        language_conditioned: bool = False,
        **kwargs,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        
        config = get_default_config(updates=kwargs)

        # Define encoders.
        if not language_conditioned:
            if shared_goal_encoder is None or early_goal_concat is None:
                raise ValueError(
                    "If not language conditioned, shared_goal_encoder and early_goal_concat must be set"
                )
            if early_goal_concat:
                # passing None as the goal encoder causes early goal concat
                goal_encoder_def = None
            else:
                if shared_goal_encoder:
                    goal_encoder_def = encoder_def
                else:
                    goal_encoder_def = copy.deepcopy(encoder_def)

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        else:
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        print("Encoder def:", encoder_def)
        
        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "critic": copy.deepcopy(encoder_def),
            }

        # Define networks.
        policy_def = ActorVectorField(
            encoder=encoders["actor"],
            network=MLP(**network_kwargs),
            action_dim=actions.shape[-1],
            name="actor",
        )
        critic_backbone = partial(MLP, **network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            CriticVectorField, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        networks = {
            "actor": policy_def,
            "critic": critic_def,
        }
        model_def = ModuleDict(networks)
        
        txs = {
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
        }
        
        rng, init_rng = jax.random.split(rng)
        extra_kwargs = {}
        network_input = (
            (observations, goals) if config.goal_conditioned else observations
        )

        # Create dummy times and returns for initialization
        batch_size = actions.shape[0]
        dummy_times = jnp.zeros((batch_size, 1))
        dummy_returns = jnp.zeros((batch_size, 1))

        params = model_def.init(
            init_rng,
            actor=[network_input, actions, dummy_times],
            critic=[dummy_returns, dummy_times, network_input, actions],
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

        # Set config values before converting to FrozenDict
        config['ob_dims'] = observations["image"].shape[1:]
        config['action_dim'] = actions.shape[-1]
        config['min_reward'] = -1
        config['max_reward'] = 0

        config = flax.core.FrozenDict(config)
        
        return cls(state, config)

def get_default_config(updates=None):
    config = ml_collections.ConfigDict()
    config.discount = 0.98
    config.soft_target_update_rate = 5e-3
    config.critic_ensemble_size = 2
    config.actor_optimizer_kwargs = ml_collections.ConfigDict(
        {
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
        }
    )
    config.critic_optimizer_kwargs = ml_collections.ConfigDict(
        {
            "learning_rate": 3e-4,
            "warmup_steps": 2000,
        }
    )
    
    config.ret_agg = 'mean'          # 'min' or 'mean'
    config.q_agg = 'mean'            # 'min' or 'mean'
    config.clip_flow_actions = True   # Whether to clip the intermediate flow actions.
    config.clip_flow_returns = True    # Whether to clip flow returns.
    config.confidence_weight_temp = 0.3  # Temperature for the confidence weights.
    config.dcfm_lambda = 1.0  # distributional conditional flow matching loss coefficient.
    config.bcfm_lambda = 1.0  # bootstrapped conditional flow matching loss coefficient.
    config.num_samples = 16  # Number of action samples for rejection sampling.
    config.num_flow_steps = 10  # Number of flow steps.

    # Goal-conditioning
    config.goal_conditioned = True
    config.gc_kwargs = ml_collections.ConfigDict(
        {
            "negative_proportion": 0.0,
        }
    )

    config.early_goal_concat = False
    config.language_conditioned = True

    if updates is not None:
        config.update(ml_collections.ConfigDict(updates).copy_and_resolve_references())
    return config