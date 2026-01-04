"""
Training script for EfficientZero-style TTT SimSiam experiments.

This script uses three separate networks following EfficientZero:
1. Projector MLP: encoder output → projection space
2. Predictor MLP: projection → prediction
3. Dynamics Network: (state, action) → next state prediction

Training flow:
- Encode: z_t, z_{t+1} = encoder(obs_t), encoder(obs_{t+1})
- Dynamics: ẑ_{t+1} = dynamics(z_t, action)
- Project: p_{t+1} = projector(z_{t+1}), p̂_{t+1} = projector(ẑ_{t+1})
- Predict: q̂_{t+1} = predictor(p̂_{t+1})
- Loss: cosine_sim(q̂_{t+1}, stop_grad(p_{t+1}))
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple
from functools import partial

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import flax
import flax.linen as nn
import flax.core
from flax.training import checkpoints, train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from ml_collections import config_flags
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb

from jaxrl_m.agents import agents
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets

# Import our new networks
from simsiam_networks import (
    SimSiamProjector,
    SimSiamPredictor,
    DynamicsNetworkFlat,
    cosine_similarity_loss,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "ttt_simsiam_efficientzero", "WandB project name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "oxedata_config",
    None,
    "File path to the autonomous data configuration.",
    lock_config=False,
)


def _maybe_text_processor():
    if FLAGS.config.get("text_processor") is None:
        return None
    return text_processors[FLAGS.config.text_processor](
        **FLAGS.config.text_processor_kwargs
    )


def _process_text(batch, processor):
    if processor is None:
        return batch
    language = [s.decode("utf-8") for s in batch["goals"]["language"]]
    batch["goals"]["language"] = processor.encode(language)
    return batch


def _process_oxe_batch(raw_batch, processor):
    batch = dict(
        actions=raw_batch["action"].squeeze(),
        goals=dict(language=raw_batch["task"]["language_instruction"]),
        mc_returns=raw_batch["mc_return"].squeeze(-1),
        observations=dict(image=raw_batch["observation"]["image_primary"].squeeze()),
        next_observations=dict(
            image=raw_batch["next_observation"]["image_primary"].squeeze()
        ),
        rewards=raw_batch["reward"].squeeze(-1),
        masks=raw_batch["td_mask"].squeeze(-1),
    )
    return _process_text(batch, processor)


def _to_jax_array(x):
    if isinstance(x, (str, bytes)):
        return x
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, np.ndarray) and (x.dtype == object or x.dtype.kind in ('U', 'S', 'O')):
        return x
    return jnp.asarray(x)


def _safe_filter_datasets(dataset_kwargs_list, sample_weights, include_names):
    try:
        filtered = filter_eval_datasets(dataset_kwargs_list, sample_weights, include_names)
        if not filtered or len(filtered[0]) == 0:
            raise ValueError("empty filter result")
        return filtered
    except ValueError:
        logging.warning(
            "No datasets matched filter %s; using original dataset list.", include_names
        )
        return dataset_kwargs_list, sample_weights


@flax.struct.dataclass
class SimSiamState:
    """State for SimSiam components (projector, predictor, dynamics)."""
    params: flax.core.FrozenDict
    opt_state: optax.OptState


def _create_optimizer(config):
    """Create optimizer based on config."""
    if config.optimizer_type == "sgd":
        base_optimizer = optax.sgd(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
        )
    elif config.optimizer_type == "adamw":
        base_optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
    
    # Add weight decay if using SGD (AdamW has it built-in)
    if config.optimizer_type == "sgd" and config.weight_decay > 0:
        base_optimizer = optax.chain(
            base_optimizer,
            optax.add_decayed_weights(config.weight_decay),
        )
    
    # Add learning rate schedule if requested
    if config.use_lr_schedule:
        schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_factor,
        )
        base_optimizer = optax.chain(
            optax.scale_by_schedule(schedule),
            base_optimizer,
        )
    
    return base_optimizer


def build_simsiam_update_fn(
    encoder_def: nn.Module,
    projector_def: SimSiamProjector,
    predictor_def: SimSiamPredictor,
    dynamics_def: DynamicsNetworkFlat,
    optimizer: optax.GradientTransformation,
    *,
    lambda_sim: float,
    stop_grad_target: bool,
    normalize_latents: bool,
):
    """
    Build update function for EfficientZero-style SimSiam training.
    
    Training flow:
    1. Encode current and next observations
    2. Use dynamics network to predict next state from current state + action
    3. Project both real and predicted next states
    4. Apply predictor to predicted projection
    5. Compute cosine similarity loss between prediction and target
    """
    
    def loss_fn(encoder_params, projector_params, predictor_params, dynamics_params, batch):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        goals = batch.get("goals", {})
        actions = batch["actions"]

        # 1. Encode current and next states
        z_t = encoder_def.apply({"params": encoder_params}, (obs, goals))
        z_tp1 = encoder_def.apply({"params": encoder_params}, (next_obs, goals))
        
        # 2. Predict next state using dynamics network
        z_tp1_pred = dynamics_def.apply(
            {"params": dynamics_params},
            z_t,
            actions,
            train=True
        )
        
        # 3. Project both real and predicted next states
        proj_real = projector_def.apply(
            {"params": projector_params},
            z_tp1,
            train=True
        )
        proj_pred = projector_def.apply(
            {"params": projector_params},
            z_tp1_pred,
            train=True
        )
        
        # Stop gradient on target if requested (SimSiam style)
        if stop_grad_target:
            proj_target = jax.lax.stop_gradient(proj_real)
        else:
            proj_target = proj_real
        
        # 4. Apply predictor to predicted projection
        prediction = predictor_def.apply(
            {"params": predictor_params},
            proj_pred,
            train=True
        )
        
        # 5. Compute loss
        loss = cosine_similarity_loss(prediction, proj_target, normalize=normalize_latents)
        
        return loss, {
            "z_norm_real": jnp.linalg.norm(z_tp1, axis=-1).mean(),
            "z_norm_pred": jnp.linalg.norm(z_tp1_pred, axis=-1).mean(),
            "proj_norm_real": jnp.linalg.norm(proj_real, axis=-1).mean(),
            "proj_norm_pred": jnp.linalg.norm(proj_pred, axis=-1).mean(),
        }

    # Create gradient function for all parameters
    grad_fn = jax.value_and_grad(
        lambda params, batch: loss_fn(
            params["encoder"],
            params["projector"],
            params["predictor"],
            params["dynamics"],
            batch,
        ),
        has_aux=True,
    )

    @jax.jit
    def update(encoder_params, simsiam_state: SimSiamState, batch):
        # Combine all params for gradient computation
        all_params = {
            "encoder": encoder_params,
            "projector": simsiam_state.params["projector"],
            "predictor": simsiam_state.params["predictor"],
            "dynamics": simsiam_state.params["dynamics"],
        }
        
        (loss, aux_metrics), grads = grad_fn(all_params, batch)
        
        # Scale encoder gradients by lambda_sim (loss weight)
        scaled_encoder_grads = jax.tree_util.tree_map(
            lambda g: lambda_sim * g,
            grads["encoder"]
        )
        
        # Apply encoder gradients directly (manual SGD-style update)
        new_encoder_params = optax.apply_updates(encoder_params, scaled_encoder_grads)
        
        # Update SimSiam components (projector, predictor, dynamics) with optimizer
        simsiam_grads = {
            "projector": grads["projector"],
            "predictor": grads["predictor"],
            "dynamics": grads["dynamics"],
        }
        
        updates, new_opt_state = optimizer.update(
            simsiam_grads,
            simsiam_state.opt_state,
            simsiam_state.params,
        )
        new_simsiam_params = optax.apply_updates(simsiam_state.params, updates)
        new_simsiam_state = SimSiamState(params=new_simsiam_params, opt_state=new_opt_state)
        
        metrics = {
            "loss_sim": loss,
            **aux_metrics,
        }
        
        return new_encoder_params, new_simsiam_state, metrics

    return update


def build_ttt_adaptation_fn(
    encoder_def: nn.Module,
    projector_def: SimSiamProjector,
    predictor_def: SimSiamPredictor,
    dynamics_def: DynamicsNetworkFlat,
    *,
    stop_grad_target: bool,
    normalize_latents: bool,
):
    """
    Build TTT adaptation function.
    
    Only adapts the encoder, keeping projector/predictor/dynamics frozen.
    """
    
    def loss_fn(encoder_params, projector_params, predictor_params, dynamics_params, batch):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        goals = batch.get("goals", {})
        actions = batch["actions"]

        # Encode states
        z_t = encoder_def.apply({"params": encoder_params}, (obs, goals))
        z_tp1 = encoder_def.apply({"params": encoder_params}, (next_obs, goals))
        
        # Predict next state
        z_tp1_pred = dynamics_def.apply(
            {"params": dynamics_params},
            z_t,
            actions,
            train=False  # Not training dynamics during TTT
        )
        
        # Project
        proj_real = projector_def.apply(
            {"params": projector_params},
            z_tp1,
            train=False  # Not training projector during TTT
        )
        proj_pred = projector_def.apply(
            {"params": projector_params},
            z_tp1_pred,
            train=False
        )
        
        if stop_grad_target:
            proj_target = jax.lax.stop_gradient(proj_real)
        else:
            proj_target = proj_real
        
        # Predict
        prediction = predictor_def.apply(
            {"params": predictor_params},
            proj_pred,
            train=False  # Not training predictor during TTT
        )
        
        loss = cosine_similarity_loss(prediction, proj_target, normalize=normalize_latents)
        return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    def adapt(encoder_params, projector_params, predictor_params, dynamics_params, 
              batch, lr: float, steps: int):
        if steps <= 0:
            return encoder_params, jnp.empty((0,), dtype=jnp.float32)

        def body(params, _):
            loss, grads = grad_fn(params, projector_params, predictor_params, dynamics_params, batch)
            updates = jax.tree_util.tree_map(lambda g: -lr * g, grads)
            new_params = optax.apply_updates(params, updates)
            return new_params, loss

        new_params, losses = jax.lax.scan(body, encoder_params, None, length=steps)
        return new_params, losses

    return adapt


def _recursive_freeze(d):
    """Recursively convert all dicts to FrozenDict."""
    if isinstance(d, (dict, flax.core.FrozenDict)):
        return flax.core.FrozenDict({k: _recursive_freeze(v) for k, v in d.items()})
    return d


def _update_agent_encoder(agent, new_encoder_params):
    """Update agent's encoder parameters."""
    new_encoder_params = _recursive_freeze(new_encoder_params)
    
    def build_new_params(params):
        result = {}
        for k, v in params.items():
            if k == "modules_actor":
                actor_dict = dict(v)
                actor_dict["encoder"] = new_encoder_params
                result[k] = flax.core.FrozenDict(actor_dict)
            elif k == "modules_critic":
                critic_dict = dict(v)
                critic_dict["encoder"] = new_encoder_params
                result[k] = flax.core.FrozenDict(critic_dict)
            else:
                result[k] = v
        return flax.core.FrozenDict(result)
    
    new_params = build_new_params(agent.state.params)
    new_target_params = build_new_params(agent.state.target_params)

    new_state = agent.state.replace(params=new_params, target_params=new_target_params)
    return agent.replace(state=new_state)


def _slice_tree(tree, start: int, end: int):
    return jax.tree_map(lambda x: x[start:end], tree)


def _make_transition_batch(traj: Dict[str, Any], idx: int) -> Dict[str, Any]:
    obs = _slice_tree(traj["observations"], idx, idx + 1)
    next_obs = _slice_tree(traj["next_observations"], idx, idx + 1)
    actions = traj["actions"][idx : idx + 1]
    batch = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
    }
    goals = traj.get("goals")
    if goals is not None:
        batch["goals"] = _slice_tree(goals, idx, idx + 1)
    return batch


def main(_):
    tf.config.set_visible_devices([], "GPU")

    text_processor = _maybe_text_processor()

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({"project": FLAGS.project, "exp_descriptor": FLAGS.name})
    variant = FLAGS.config.to_dict()
    variant["oxe_config"] = FLAGS.oxedata_config.to_dict()
    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # Setup datasets
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.oxedata_config["oxe_kwargs"]
        )
        del FLAGS.oxedata_config["oxe_kwargs"]

    train_datasets_kwargs_list, train_sample_weights = _safe_filter_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )

    train_data = make_interleaved_dataset(
        dataset_kwargs_list=train_datasets_kwargs_list,
        sample_weights=train_sample_weights,
        train=True,
        shuffle_buffer_size=FLAGS.oxedata_config["shuffle_buffer_size"],
        traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
        frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
        batch_size=FLAGS.oxedata_config["batch_size"],
        balance_weights=FLAGS.oxedata_config.get("balance_weights", False),
        traj_transform_threads=FLAGS.oxedata_config.get("traj_transform_threads", None),
        traj_read_threads=FLAGS.oxedata_config.get("traj_read_threads", None),
    )

    val_datasets_kwargs_list, _ = _safe_filter_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    val_data = create_validation_dataset(
        val_datasets_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
    )

    def make_iter(dataset) -> Iterator[Dict[str, Any]]:
        for batch in dataset.iterator(prefetch=0):
            batch = jax.tree_map(_to_jax_array, batch)
            yield _process_oxe_batch(batch, text_processor)

    train_data_iter = make_iter(train_data)
    val_data_iter = make_iter(
        val_data.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(FLAGS.oxedata_config["batch_size"])
    )

    # Create agent
    example_batch = next(train_data_iter)
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, agent_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=agent_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )

    if FLAGS.config.resume_path:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
        logging.info("Restored agent from %s", FLAGS.config.resume_path)

    # Setup SimSiam networks
    from jaxrl_m.common.encoding import LCEncodingWrapper

    wrapped_encoder_def = LCEncodingWrapper(
        encoder=encoder_def,
        use_proprio=False,
        stop_gradient=False,
    )

    encoder_params = agent.state.params["modules_actor"]["encoder"]

    # Probe latent dimension
    rng, probe_rng = jax.random.split(rng)
    latent_example = wrapped_encoder_def.apply(
        {"params": encoder_params},
        (example_batch["observations"], example_batch["goals"]),
    )
    latent_dim = latent_example.shape[-1]
    action_dim = example_batch["actions"].shape[-1]

    logging.info(f"Latent dim: {latent_dim}, Action dim: {action_dim}")

    # Create projector
    projector_def = SimSiamProjector(
        hidden_dims=tuple(FLAGS.config.projector_hidden_dims),
        output_dim=FLAGS.config.projector_output_dim,
        norm_type=FLAGS.config.projector_norm,
    )
    
    # Create predictor
    predictor_def = SimSiamPredictor(
        hidden_dims=tuple(FLAGS.config.predictor_hidden_dims),
        output_dim=FLAGS.config.predictor_output_dim,
        norm_type=FLAGS.config.predictor_norm,
    )
    
    # Create dynamics network
    dynamics_def = DynamicsNetworkFlat(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=FLAGS.config.dynamics_hidden_dim,
        num_residual_blocks=FLAGS.config.dynamics_num_residual_blocks,
        norm_type=FLAGS.config.dynamics_norm,
    )

    # Initialize all SimSiam networks
    rng, proj_rng, pred_rng, dyn_rng = jax.random.split(rng, 4)
    
    projector_params = projector_def.init(proj_rng, latent_example, train=True)["params"]
    
    proj_output = projector_def.apply(
        {"params": projector_params},
        latent_example,
        train=True
    )
    predictor_params = predictor_def.init(pred_rng, proj_output, train=True)["params"]
    
    dynamics_params = dynamics_def.init(
        dyn_rng,
        latent_example,
        example_batch["actions"],
        train=True
    )["params"]

    # Create unified SimSiam state with optimizer
    simsiam_params = {
        "projector": projector_params,
        "predictor": predictor_params,
        "dynamics": dynamics_params,
    }
    
    optimizer = _create_optimizer(FLAGS.config)
    simsiam_state = SimSiamState(
        params=flax.core.freeze(simsiam_params),
        opt_state=optimizer.init(simsiam_params),
    )

    # Build update and adaptation functions
    simsiam_update = build_simsiam_update_fn(
        encoder_def=wrapped_encoder_def,
        projector_def=projector_def,
        predictor_def=predictor_def,
        dynamics_def=dynamics_def,
        optimizer=optimizer,
        lambda_sim=FLAGS.config.lambda_sim,
        stop_grad_target=FLAGS.config.stop_grad_target,
        normalize_latents=FLAGS.config.normalize_latents,
    )

    ttt_adapt_fn = build_ttt_adaptation_fn(
        encoder_def=wrapped_encoder_def,
        projector_def=projector_def,
        predictor_def=predictor_def,
        dynamics_def=dynamics_def,
        stop_grad_target=FLAGS.config.stop_grad_target,
        normalize_latents=FLAGS.config.normalize_latents,
    )

    # Training loop
    timer = Timer()
    for step in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")
        batch = next(train_data_iter)
        
        # Update RL agent
        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        # Update SimSiam (encoder + projector/predictor/dynamics)
        encoder_params = agent.state.params["modules_actor"]["encoder"]
        new_encoder_params, simsiam_state, sim_metrics = simsiam_update(
            encoder_params,
            simsiam_state,
            batch,
        )
        agent = _update_agent_encoder(agent, new_encoder_params)

        # Logging
        if (step + 1) % FLAGS.config.log_interval == 0:
            wandb_logger.log({"training": jax.device_get(update_info)}, step=step)
            wandb_logger.log({"simsiam": jax.device_get(sim_metrics)}, step=step)

        # Evaluation
        if (step + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            num_eval_batches = int(FLAGS.config.get("eval_batches", 4))
            metrics = []
            ttt_loss_logs = []
            eval_mode = str(FLAGS.config.get("ttt_mode", "standard")).lower()
            
            base_encoder_params = agent.state.params["modules_actor"]["encoder"]
            online_encoder_params = base_encoder_params
            
            for _ in range(num_eval_batches):
                val_batch = next(val_data_iter)
                rng, val_rng = jax.random.split(rng)
                eval_agent = agent
                adapt_losses = None
                
                if (
                    FLAGS.config.ttt_steps > 0
                    and eval_mode in ("standard", "online")
                ):
                    start_params = (
                        base_encoder_params
                        if eval_mode == "standard"
                        else online_encoder_params
                    )
                    
                    # TTT adaptation (only encoder, others frozen)
                    adapted_params, adapt_losses = ttt_adapt_fn(
                        start_params,
                        simsiam_state.params["projector"],
                        simsiam_state.params["predictor"],
                        simsiam_state.params["dynamics"],
                        val_batch,
                        FLAGS.config.ttt_lr,
                        FLAGS.config.ttt_steps,
                    )
                    online_encoder_params = adapted_params
                    eval_agent = _update_agent_encoder(agent, adapted_params)
                
                metrics.append(eval_agent.get_debug_metrics(val_batch, seed=val_rng))
                if adapt_losses is not None:
                    ttt_loss_logs.append(adapt_losses)
            
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=step)
            
            if ttt_loss_logs:
                stacked = jnp.stack(ttt_loss_logs)
                stacked_np = np.asarray(jax.device_get(stacked))
                wandb_logger.log(
                    {
                        "validation_ttt/loss_last": float(stacked_np[:, -1].mean()),
                        "validation_ttt/loss_mean": float(stacked_np.mean()),
                    },
                    step=step,
                )

        # Checkpointing
        if (step + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoints.save_checkpoint(
                save_dir,
                dict(agent=agent, simsiam_state=simsiam_state),
                step=step + 1,
                keep=100,
            )

        timer.tock("total")
        if (step + 1) % FLAGS.config.log_interval == 0:
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

    wandb_logger.close()


if __name__ == "__main__":
    app.run(main)