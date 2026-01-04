from typing import Optional, Sequence, Tuple, Union

import optax


def make_optimizer(
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    cosine_decay_steps: Optional[int] = None,
    weight_decay: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    optimizer_type: str = "adam",
    momentum: float = 0.9,
    lr_decay_steps: Optional[Union[int, Sequence[int]]] = None,
    lr_decay_factor: float = 0.1,
    end_learning_rate: Optional[float] = None,
    return_lr_schedule: bool = False,
) -> optax.GradientTransformation:
    """
    Create an optimizer with optional learning rate schedule.

    Args:
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps (linear warmup from 0)
        cosine_decay_steps: If specified, use cosine decay schedule
        weight_decay: Weight decay coefficient (uses AdamW for adam, add_decayed_weights for sgd)
        clip_grad_norm: If specified, clip gradients by global norm
        optimizer_type: "adam" or "sgd"
        momentum: Momentum for SGD (ignored for adam)
        lr_decay_steps: Step(s) at which to decay learning rate (EfficientZero style).
            Can be a single int or a sequence of ints for multi-step decay.
            E.g., 100000 for single decay, or [100000, 200000] for multi-step.
        lr_decay_factor: Factor to multiply learning rate by at each decay step (default 0.1).
            Ignored if end_learning_rate is specified.
        end_learning_rate: Final learning rate after decay. If specified, overrides lr_decay_factor.
            E.g., learning_rate=0.2, end_learning_rate=0.02 gives 10x decay.
        return_lr_schedule: If True, return (optimizer, schedule) tuple

    Returns:
        optax.GradientTransformation or tuple of (optimizer, schedule)
    """
    # Build learning rate schedule
    if cosine_decay_steps is not None:
        # Cosine decay schedule
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=cosine_decay_steps,
            end_value=0.0,
        )
    elif lr_decay_steps is not None:
        # Step decay schedule (EfficientZero style: 0.2 -> 0.02 at 100k steps)
        # Normalize to list
        if isinstance(lr_decay_steps, int):
            decay_steps_list = [lr_decay_steps]
        else:
            decay_steps_list = list(lr_decay_steps)

        # Compute decay factors for each step
        if end_learning_rate is not None:
            # Single decay to end_learning_rate (for single step decay)
            # For multi-step, compute geometric decay
            num_decays = len(decay_steps_list)
            total_decay = end_learning_rate / learning_rate
            per_step_factor = total_decay ** (1.0 / num_decays)
            decay_factors = [per_step_factor] * num_decays
        else:
            decay_factors = [lr_decay_factor] * len(decay_steps_list)

        # Build piecewise constant schedule with warmup
        # Compute learning rates at each stage
        lrs = [learning_rate]
        for factor in decay_factors:
            lrs.append(lrs[-1] * factor)

        # Build boundaries (including warmup)
        if warmup_steps > 0:
            boundaries = [warmup_steps] + [warmup_steps + s for s in decay_steps_list]
            schedules = [
                optax.linear_schedule(0.0, learning_rate, warmup_steps),
                optax.constant_schedule(lrs[0]),
            ]
            for lr in lrs[1:]:
                schedules.append(optax.constant_schedule(lr))
        else:
            boundaries = decay_steps_list
            schedules = [optax.constant_schedule(lr) for lr in lrs]

        learning_rate_schedule = optax.join_schedules(schedules, boundaries)
    else:
        # Simple warmup then constant
        learning_rate_schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, learning_rate, warmup_steps),
                optax.constant_schedule(learning_rate),
            ],
            [warmup_steps],
        )

    # Define optimizers
    @optax.inject_hyperparams
    def optimizer(learning_rate: float, weight_decay: Optional[float]):
        optimizer_stages = []

        if clip_grad_norm is not None:
            optimizer_stages.append(optax.clip_by_global_norm(clip_grad_norm))

        if optimizer_type == "sgd":
            # SGD with momentum (EfficientZero style)
            optimizer_stages.append(optax.sgd(learning_rate=learning_rate, momentum=momentum))
            # Add weight decay separately for SGD
            if weight_decay is not None:
                optimizer_stages.append(optax.add_decayed_weights(weight_decay))
        elif weight_decay is not None:
            # Adam with weight decay (AdamW)
            optimizer_stages.append(
                optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
            )
        else:
            # Standard Adam
            optimizer_stages.append(optax.adam(learning_rate=learning_rate))

        return optax.chain(*optimizer_stages)

    if return_lr_schedule:
        return (
            optimizer(learning_rate=learning_rate_schedule, weight_decay=weight_decay),
            learning_rate_schedule,
        )
    else:
        return optimizer(
            learning_rate=learning_rate_schedule, weight_decay=weight_decay
        )
