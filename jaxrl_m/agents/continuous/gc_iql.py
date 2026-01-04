import copy
from functools import partial
from typing import *

import flax
import flax.linen as nn
import distrax
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from jaxrl_m.agents.continuous.iql import (
    expectile_loss,
    iql_actor_loss,
    iql_critic_loss,
    iql_value_loss,
)
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import GCEncodingWrapper, LCEncodingWrapper
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, Policy, ValueCritic, ensemblize, DistributionalCritic
from jaxrl_m.networks.distributional import (
    cross_entropy_loss_on_scalar,
    hl_gauss_transform,
)
from jaxrl_m.networks.mlp import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



class GCIQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()


    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients)
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
        distributional_critic_return_logits: bool = False,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).

        Note that if the agent is using a distributional critic, the output will
        be (Q, probs)
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        qs = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

        if (
            self.config["distributional_critic"]
            and not distributional_critic_return_logits
        ):
            qs, _ = qs  # unpack

        return qs

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_value(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for value network.
        Pass grad_params
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="value",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_value(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target value network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_value(
            observations, rng=rng, grad_params=self.state.target_params
        )

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        batch_size = batch["rewards"].shape[0]
        neg_goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)
        rng, new_rng = jax.random.split(self.state.rng)

        # selects a portion of goals to make negative
        def get_goals_rewards(key):
            neg_goal_mask = (
                jax.random.uniform(key, (batch_size,))
                < self.config["negative_proportion"]
            )
            goal_indices = jnp.where(
                neg_goal_mask, neg_goal_indices, jnp.arange(batch_size)
            )
            new_goals = jax.tree_map(lambda x: x[goal_indices], batch["goals"])
            new_rewards = jnp.where(neg_goal_mask, -1, batch["rewards"])
            new_masks = jnp.where(neg_goal_mask, 1, batch["masks"])
            return new_goals, new_rewards, new_masks

        def critic_loss_fn(params, rng):
            # if not self.config["language_conditioned"]:
            rng, key = jax.random.split(rng)
            goals, rewards, masks = get_goals_rewards(key)
            # else:
            #     goals = batch["goals"]
            #     rewards = batch["rewards"]
            #     masks = batch["masks"]

            rng, key = jax.random.split(rng)
            next_v = self.forward_target_value((batch["next_observations"], goals), key)
            target_q = rewards + self.config["discount"] * next_v * masks

            rng, key = jax.random.split(rng)
            q = self.forward_critic(
                (batch["observations"], goals), batch["actions"], key, grad_params=params,
                distributional_critic_return_logits=self.config["distributional_critic"],
            )
            if self.config["distributional_critic"]:
                q, q_logits = q
            chex.assert_shape(q, (self.config["critic_ensemble_size"], batch_size))

            if self.config["distributional_critic"]:
                # cross entropy loss
                critic_loss = cross_entropy_loss_on_scalar(
                    q_logits,
                    target_q,
                    self.config["scalar_target_to_dist_fn"],
                )
                chex.assert_shape(critic_loss, (self.config["critic_ensemble_size"], batch_size))
                return critic_loss.mean(), {
                    "cross_entropy_loss": critic_loss.mean(),
                    "q": q.mean(),
                    "target_q": target_q.mean(),
                    "td_loss": jnp.square(q - target_q).mean(),
                }
            else:
                # normal IQL MSE loss
                return iql_critic_loss(q, target_q)

        def value_loss_fn(params, rng):
            # if not self.config["language_conditioned"]:
            rng, key = jax.random.split(rng)
            goals, _, _ = get_goals_rewards(key)
            # else:
            #     goals = batch["goals"]

            rng, key = jax.random.split(rng)
            q = self.forward_target_critic(
                (batch["observations"], goals), batch["actions"], key
            )  # no gradient
            q = jnp.min(q, axis=0)  # min over 2 Q functions

            rng, key = jax.random.split(rng)
            v = self.forward_value((batch["observations"], goals), key, grad_params=params)
            return iql_value_loss(q, v, self.config["expectile"])

        def actor_loss_fn(params, rng):
            rng, key = jax.random.split(rng)

            if self.config["update_actor_with_target_adv"]:
                critic_fn = self.forward_target_critic
            else:
                # Seohong: not using the target will make updates faster
                critic_fn = self.forward_critic
            q = critic_fn(
                    (batch["observations"], batch["goals"]),
                    batch["actions"], 
                    key
                )  # no gradient
            q = jnp.min(q, axis=0)  # min over 2 Q functions

            rng, key = jax.random.split(rng)
            v = self.forward_value((batch["observations"], batch["goals"]), key)  # no gradients

            rng, key = jax.random.split(rng)
            dist = self.forward_policy((batch["observations"], batch["goals"]), key, grad_params=params)
            mask = batch.get("actor_loss_mask", None)
            return iql_actor_loss(
                q,
                v,
                dist,
                batch["actions"],
                self.config["temperature"],
                mask=mask,
            )

        loss_fns = {
            "critic": critic_loss_fn,
            "value": value_loss_fn,
            "actor": actor_loss_fn,
        }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # update rng
        new_state = new_state.replace(rng=new_rng)

        # log learning rates
        if "actor" in self.lr_schedules.keys():
            info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.forward_policy((observations, goals), seed, train=False)
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        v = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            name="value",
        )
        next_v = self.state.apply_fn(
            {"params": self.state.target_params},
            (batch["next_observations"], batch["goals"]),
            name="value",
        )
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        # q = self.state.apply_fn(
        #     {"params": self.state.params},
        #     (batch["observations"], batch["goals"]),
        #     batch["actions"],
        #     name="critic",
        # )
        q = self.forward_critic(
            (batch["observations"], batch["goals"]),
            batch["actions"],
            None,
            train=False,
            distributional_critic_return_logits=self.config["distributional_critic"]
        )
        if self.config["distributional_critic"]:
            q, q_logits = q
            cross_entropy_loss = cross_entropy_loss_on_scalar(
                    q_logits,
                    target_q,
                    self.config["scalar_target_to_dist_fn"],
                )


        q = jnp.min(q, axis=0)  # min over 2 Q functions

        metrics = {
            "log_probs": log_probs,
            "mse": mse,
            "pi_actions": pi_actions,
            "online_v": v,
            "online_q": q,
            "target_q": target_q,
            "value_err": expectile_loss(target_q - v, self.config["expectile"]),
            "expectile_ratio": jnp.mean(jnp.where(target_q > v, 1, 0)),
            "td_err": jnp.square(target_q - q),
            "advantage": target_q - v,
            "qf_advantage": q - v,
        }

        if self.config["distributional_critic"]:
            metrics.update(
                {
                    "cross_entropy_loss": cross_entropy_loss.mean(),
                }
            )

        if gripper_close_val is not None:
            gripper_close_q = self.state.apply_fn(
                {"params": self.state.params},
                (batch["observations"], batch["goals"]),
                jnp.broadcast_to(gripper_close_val, batch["actions"].shape),
                name="critic",
            )
            metrics.update(
                {
                    "gripper_close_q": gripper_close_q,
                    "gripper_close_adv": gripper_close_q - v,
                }
            )

        return metrics

    @jax.jit
    def get_q_values(self, observations, goals, actions):
        q = self.state.apply_fn(
            {"params": self.state.target_params},
            (observations, goals),
            actions,
            name="critic",
        )
        if self.config["distributional_critic"]:
            q, _ = q
        
        q = jnp.min(q.squeeze(), axis=0)
        return q


    # @jax.jit
    def get_eval_values(self, traj, seed, goals):
        actions = self.sample_actions(
            observations=traj["observations"], goals=goals, argmax=True
        )
        mse = ((actions - traj["actions"]) ** 2).sum((-1))
        v = self.forward_value(
            (traj["observations"], goals),
            seed,
            train=False,
        )
        next_v = self.forward_target_value(
            (traj["next_observations"], goals),
            seed,
        )
        target_q = traj["rewards"] + self.config["discount"] * next_v * traj["masks"]
        q = self.forward_critic(
            (traj["observations"], goals),
            traj["actions"],
            seed,
            train=False,
            distributional_critic_return_logits=self.config["distributional_critic"]
        )
        if self.config["distributional_critic"]:
            q, q_logits = q
            cross_entropy_loss = cross_entropy_loss_on_scalar(
                    q_logits,
                    target_q,
                    self.config["scalar_target_to_dist_fn"],
                )
            cross_entropy_loss = cross_entropy_loss.mean(axis=0)

        q = jnp.min(q, axis=0) 

        
        metrics = {
            "values": v,
            "q": q,
            "target_q": target_q,
            "advantage": target_q - v,
            "advantage_data": q - v,
            "mse": mse,
            "td_err": jnp.square(target_q - q),
            "value_expectile_loss": expectile_loss(target_q - v, self.config["expectile"]),
            "rewards": traj["rewards"],
        }
        if self.config["distributional_critic"]:
            metrics.update(
                {
                    "cross_entropy_loss": cross_entropy_loss,
                }
            )
        return metrics

    def plot_values(self, traj, seed=None, goals=None):
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
        images = traj["observations"]["image"].squeeze() # (T, H, W, 3)

        num_rows = len(metrics.keys()) + 1

        fig, axs = plt.subplots(num_rows, 1, figsize=(8, 16))
        canvas = FigureCanvas(fig)
        plt.xlim(0, len(metrics["values"]))

        interval = images.shape[0] // 8
        interval = max(1, interval)
        sel_images = images[::interval]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1) # (200, 8*200, 3)
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
        # plt.show()
        return out_image

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        shared_goal_encoder: bool = False,
        early_goal_concat: bool = False,
        language_conditioned: bool = False,
        use_proprio: bool = False,
        critic_ensemble_size: int = 2,
        distributional_critic: bool = False,
        distributional_critic_kwargs: dict = {
            "q_min": -100.0,
            "q_max": 0.0,
            "num_bins": 128,
        },
        negative_proportion: float = 0.0,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "dropout_rate": 0.0,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # Algorithm config
        discount=0.98,
        expectile=0.7,
        temperature=1.0,
        target_update_rate=0.002,
        update_actor_with_target_adv=True,
    ):
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
            if shared_goal_encoder is not None or early_goal_concat is not None:
                raise ValueError(
                    "If language conditioned, shared_goal_encoder and early_goal_concat must not be set"
                )
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        print("Encoder def:", encoder_def)
            
        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "value": encoder_def,
                "critic": encoder_def,
            }
        else:
            # I (kvablack) don't think these deepcopies will break
            # shared_goal_encoder, but I haven't tested it.
            encoders = {
                "actor": encoder_def,
                "value": copy.deepcopy(encoder_def),
                "critic": copy.deepcopy(encoder_def),
            }

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoders["actor"],
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            ),
            "value": ValueCritic(encoders["value"], MLP(**network_kwargs)),
        }

        if distributional_critic:
            q_min = distributional_critic_kwargs["q_min"]
            q_max = distributional_critic_kwargs["q_max"]
            networks["critic"] = DistributionalCritic(
                encoders["critic"],
                network=ensemblize(
                    partial(MLP, **network_kwargs), critic_ensemble_size
                )(name="critic_ensemble"),
                q_low=q_min,
                q_high=q_max,
                num_bins=distributional_critic_kwargs["num_bins"],
            )
            scalar_target_to_dist_fn = hl_gauss_transform(
                min_value=q_min,
                max_value=q_max,
                num_bins=distributional_critic_kwargs["num_bins"],
            )[0]
        else:
            networks["critic"] =  Critic(
                encoders["critic"],
                network=ensemblize(partial(MLP, **network_kwargs), critic_ensemble_size)(
                    name="critic_ensemble"
                ),
            )
            scalar_target_to_dist_fn = None

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[(observations, goals)],
            value=[(observations, goals)],
            critic=[(observations, goals), actions],
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
            "value": lr_schedule,
            "critic": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )

        txs = {k: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(v)) for k, v in lr_schedules.items()}
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                expectile=expectile,
                negative_proportion=negative_proportion,
                language_conditioned=language_conditioned,
                update_actor_with_target_adv=update_actor_with_target_adv,
                critic_ensemble_size=critic_ensemble_size,
                distributional_critic=distributional_critic,
                scalar_target_to_dist_fn=scalar_target_to_dist_fn,
                
            )
        )
        return cls(state, config, lr_schedules)
