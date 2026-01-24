import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from jaxrl_m.common.typing import Params, PRNGKey

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

default_init = nn.initializers.xavier_uniform


def shard_batch(batch, sharding):
    """
    Shard batch arrays across devices while preserving non-array fields.
    
    Arrays are sharded on the first dimension, with remaining dimensions replicated.
    Non-array fields (strings, lists, None, etc.) are kept as-is.
    """
    def shard_if_array(x):
        # Only shard if it's a JAX/numpy array
        if isinstance(x, jnp.ndarray):
            # Shard first dimension, replicate rest
            arr_sharding = sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
            return jax.device_put(x, arr_sharding)
        else:
            # Keep strings, lists, None, etc. as-is
            return x
    
    return jax.tree_map(
        shard_if_array, 
        batch, 
        is_leaf=lambda x: not isinstance(x, (jnp.ndarray, dict))
    )


class ModuleDict(nn.Module):
    """
    Utility class for wrapping a dictionary of modules. This is useful when you have multiple modules that you want to
    initialize all at once (creating a single `params` dictionary), but you want to be able to call them separately
    later. As a bonus, the modules may have sub-modules nested inside them that share parameters (e.g. an image encoder)
    and Flax will automatically handle this without duplicating the parameters.

    To initialize the modules, call `init` with no `name` kwarg, and then pass the example arguments to each module as
    additional kwargs. To call the modules, pass the name of the module as the `name` kwarg, and then pass the arguments
    to the module as additional args or kwargs.

    Example usage:
    ```
    shared_encoder = Encoder()
    actor = Actor(encoder=shared_encoder)
    critic = Critic(encoder=shared_encoder)

    model_def = ModuleDict({"actor": actor, "critic": critic})
    params = model_def.init(rng_key, actor=example_obs, critic=(example_obs, example_action))

    actor_output = model_def.apply({"params": params}, example_obs, name="actor")
    critic_output = model_def.apply({"params": params}, example_obs, action=example_action, name="critic")
    ```
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class JaxRLTrainState(struct.PyTreeNode):
    """
    Custom TrainState class to replace `flax.training.train_state.TrainState`.

    Adds support for holding target params and updating them via polyak
    averaging. Adds the ability to hold an rng key for dropout.

    Also generalizes the TrainState to support an arbitrary pytree of
    optimizers, `txs`. When `apply_gradients()` is called, the `grads` argument
    must have `txs` as a prefix. This is backwards-compatible, meaning `txs` can
    be a single optimizer and `grads` can be a single tree with the same
    structure as `self.params`.

    Also adds a convenience method `apply_loss_fns` that takes a pytree of loss
    functions with the same structure as `txs`, computes gradients, and applies
    them using `apply_gradients`.

    Attributes:
        step: The current training step.
        apply_fn: The function used to apply the model.
        params: The model parameters.
        target_params: The target model parameters.
        txs: The optimizer or pytree of optimizers.
        opt_states: The optimizer state or pytree of optimizer states.
        rng: The internal rng state.
        batch_stats: Optional batch normalization statistics (running mean/var).
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Params
    target_params: Params
    txs: Any = struct.field(pytree_node=False)
    opt_states: Any
    rng: PRNGKey
    batch_stats: Any = None  # For batch normalization running statistics

    @staticmethod
    def _tx_tree_map(*args, **kwargs):
        return jax.tree_map(
            *args,
            is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
            **kwargs,
        )

    def target_update(self, tau: float) -> "JaxRLTrainState":
        """
        Performs an update of the target params via polyak averaging. The new
        target params are given by:

            new_target_params = tau * params + (1 - tau) * target_params
        """
        new_target_params = jax.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), self.params, self.target_params
        )
        return self.replace(target_params=new_target_params)

    def apply_gradients(self, *, grads: Any, pmap_axis: Optional[str] = None) -> "JaxRLTrainState":
        """
        Only difference from flax's TrainState is that `grads` must have
        `self.txs` as a tree prefix (i.e. where `self.txs` has a leaf, `grads`
        has a subtree with the same structure as `self.params`.)
        """
        if pmap_axis is not None:
            grads = jax.lax.pmean(grads, axis_name=pmap_axis)

        updates_and_new_states = self._tx_tree_map(
            lambda tx, opt_state, grad: tx.update(grad, opt_state, self.params),
            self.txs,
            self.opt_states,
            grads,
        )
        updates = self._tx_tree_map(lambda _, x: x[0], self.txs, updates_and_new_states)
        new_opt_states = self._tx_tree_map(
            lambda _, x: x[1], self.txs, updates_and_new_states
        )

        updates_flat = []
        self._tx_tree_map(
            lambda _, update: updates_flat.append(update), self.txs, updates
        )
        
        from functools import reduce
        updates_acc = jax.tree_map(
            lambda *xs: reduce(jnp.add, xs), *updates_flat
        )

        from flax.core import freeze
        new_params = optax.apply_updates(self.params, updates_acc)
        new_params = freeze(new_params) 

        return self.replace(
            step=self.step + 1, 
            params=new_params, 
            opt_states=new_opt_states
        )

    def apply_loss_fns(
        self, loss_fns: Any, pmap_axis: str = None, has_aux: bool = False
    ) -> Union["JaxRLTrainState", Tuple["JaxRLTrainState", Any]]:
        # 1. 计算梯度
        treedef = jax.tree_util.tree_structure(loss_fns)
        new_rng, *rngs = jax.random.split(self.rng, treedef.num_leaves + 1)
        rngs = jax.tree_util.tree_unflatten(treedef, rngs)

        grads_and_aux = jax.tree_map(
            lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(self.params, rng),
            loss_fns,
            rngs,
        )

        self = self.replace(rng=new_rng)

        if pmap_axis is not None:
            grads_and_aux = jax.lax.pmean(grads_and_aux, axis_name=pmap_axis)

        # 2. 分离 Grads 和 Aux
        if has_aux:
            grads = jax.tree_map(lambda _, x: x[0], loss_fns, grads_and_aux)
            aux = jax.tree_map(lambda _, x: x[1], loss_fns, grads_and_aux)

            # === 修改日志记录逻辑 ===
            def _compute_grad_norm(grad_tree):
                leaves = jax.tree_util.tree_leaves(grad_tree)
                if not leaves: 
                    return jnp.array(0.0)
                return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
            
            def _compute_grad_norm_from_list(grad_list):
                leaves = []
                for g in grad_list: 
                    leaves.extend(jax.tree_util.tree_leaves(g))
                if not leaves: 
                    return jnp.array(0.0)
                return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))

            # 检测并处理 'combined' wrapper
            grads_for_logging = grads
            if "combined" in grads and len(grads) == 1:
                # 解包用于日志记录
                from flax.core import FrozenDict, unfreeze
                inner_grads = grads["combined"]
                if isinstance(inner_grads, FrozenDict):
                    inner_grads = unfreeze(inner_grads)
                grads_for_logging = inner_grads

            # 记录梯度范数（使用解包后的结构）
            for name, grad in grads_for_logging.items():
                aux[f"grad_norm/{name}"] = _compute_grad_norm(grad)
                
                # Per-module gradient norms
                flat_grad = flax.traverse_util.flatten_dict(grad, sep="/")
                
                # 针对不同模块记录详细的梯度范数
                for module in ["actor", "critic", "temperature"]:
                    module_grads = [
                        v for k, v in flat_grad.items()
                        if f"modules_{module}" in k
                    ]
                    if module_grads:
                        aux[f"grad_norm/{name}.{module}"] = _compute_grad_norm_from_list(
                            module_grads
                        )
                
                # 如果有 encoder，也记录其梯度范数
                encoder_grads = [
                    v for k, v in flat_grad.items()
                    if "encoder" in k
                ]
                if encoder_grads:
                    aux[f"grad_norm/{name}.encoder"] = _compute_grad_norm_from_list(
                        encoder_grads
                    )
        else:
            grads = grads_and_aux
            aux = {}

        # === 关键：保持原始的 {'combined': ...} 结构传给 apply_gradients ===
        if has_aux:
            return self.apply_gradients(grads=grads), aux
        else:
            return self.apply_gradients(grads=grads)

    def update_batch_stats(self, new_batch_stats: Any) -> "JaxRLTrainState":
        """
        Update batch normalization statistics.

        Args:
            new_batch_stats: New batch statistics from a forward pass with
                             mutable=['batch_stats'].
        """
        return self.replace(batch_stats=new_batch_stats)

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        txs,
        target_params=None,
        rng=jax.random.PRNGKey(0),
        batch_stats=None,
    ):
        """
        Initializes a new train state.

        Args:
            apply_fn: The function used to apply the model, typically `model_def.apply`.
            params: The model parameters, typically from `model_def.init`.
            txs: The optimizer or pytree of optimizers.
            target_params: The target model parameters.
            rng: The rng key used to initialize the rng chain for `apply_loss_fns`.
            batch_stats: Optional batch normalization statistics.
        """
        def ensure_frozen(tree):
            if isinstance(tree, (dict, flax.core.FrozenDict)):
                return flax.core.FrozenDict({k: ensure_frozen(v) for k, v in tree.items()})
            return tree

        params = ensure_frozen(params)
        if target_params is not None:
            target_params = ensure_frozen(target_params)
        else:
            target_params = params
        if batch_stats is not None:
            batch_stats = ensure_frozen(batch_stats)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            txs=txs,
            opt_states=cls._tx_tree_map(lambda tx: tx.init(params), txs),
            rng=rng,
            batch_stats=batch_stats,
        )
