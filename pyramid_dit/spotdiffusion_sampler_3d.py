"""Stage-consistent shifted-window inference for 3-D Fluid Pyramid Flow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .spotdiffusion_3d import (
    Spatial3D,
    cyclic_shift_3d,
    inverse_cyclic_shift_3d,
    sample_shift_schedule,
    split_nonoverlapping_3d,
    stitch_nonoverlapping_3d,
)


@dataclass(frozen=True)
class GlobalStep3D:
    stage: int
    step: int
    timestep: float | None
    global_shape: Spatial3D
    window_size: int
    shift: Spatial3D
    event: str = "post_step"


def predict_global_velocity_3d(
    model,
    volume: torch.Tensor,
    timestep: torch.Tensor | float,
    window_size: int,
    patch_batch_size: int = 1,
    shift: Sequence[int] = (0, 0, 0),
) -> torch.Tensor:
    """Predict local windows and assemble one global velocity tensor.

    The state is shifted before tiling, while the assembled velocity is shifted
    back before it is returned. No averaging or local scheduler update occurs.
    """
    if patch_batch_size <= 0:
        raise ValueError("patch_batch_size must be positive")
    shifted = cyclic_shift_3d(volume, shift)
    patches, spec = split_nonoverlapping_3d(shifted, window_size)
    predictions = []
    for start in range(0, patches.shape[0], patch_batch_size):
        patch_batch = patches[start : start + patch_batch_size]
        prediction = model.predict_velocity(patch_batch, timestep)
        if prediction.shape != patch_batch.shape:
            raise ValueError(
                "model prediction shape must match patch batch: "
                f"got {tuple(prediction.shape)}, expected {tuple(patch_batch.shape)}"
            )
        predictions.append(prediction)
    shifted_velocity = stitch_nonoverlapping_3d(
        torch.cat(predictions, dim=0), spec
    )
    return inverse_cyclic_shift_3d(shifted_velocity, shift)


@torch.no_grad()
def sample_global_spotdiffusion_3d(
    model,
    batch_size: int,
    global_size: int,
    steps: Sequence[int],
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    patch_batch_size: int = 1,
    shifted_windows: bool = True,
    shift_policy: str | None = None,
    shift_seed: int = 0,
    step_callback: Callable[[GlobalStep3D, torch.Tensor], None] | None = None,
    stage_start_callback: Callable[[GlobalStep3D, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Sample one global field with stage-consistent local model windows.

    For the default ``stages=(1,2,4)`` and ``global_size=384``, global states
    have sizes 96, 192, and 384 while local windows have sizes 16, 32, and 64.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if global_size <= 0 or global_size % model.stages[-1]:
        raise ValueError("global_size must be positive and divisible by the final stage")
    if len(steps) != len(model.stages) or any(count <= 0 for count in steps):
        raise ValueError("steps must contain one positive value per pyramid stage")
    if shift_policy is None:
        shift_policy = "all" if shifted_windows else "fixed"
    if shift_policy not in {"fixed", "all", "early"}:
        raise ValueError("shift_policy must be 'fixed', 'all', or 'early'")

    final_local_size = model.dit.config.sample_size
    if final_local_size % model.stages[-1]:
        raise ValueError("model sample size must be divisible by the final stage")
    low_global_size = global_size // model.stages[-1]
    low_local_size = final_local_size // model.stages[-1]
    if low_global_size % low_local_size:
        raise ValueError(
            "global and local stage sizes must form an integer patch grid"
        )
    volume = torch.randn(
        (batch_size, 3, low_global_size, low_global_size, low_global_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    for stage, stage_steps in enumerate(steps):
        window_size = low_local_size * model.stages[stage]
        expected_global_size = low_global_size * model.stages[stage]
        if stage:
            volume = F.interpolate(
                volume,
                size=(expected_global_size,) * 3,
                mode="nearest",
            )
            sigma = model.scheduler.start_sigmas[stage]
            noise = torch.randn(
                volume.shape,
                generator=generator,
                device=volume.device,
                dtype=volume.dtype,
            )
            volume = (1 - sigma) * volume + sigma * noise

        model.scheduler.set_timesteps(stage_steps, stage, device=device)
        if stage_start_callback is not None:
            stage_start_callback(
                GlobalStep3D(
                    stage=stage,
                    step=-1,
                    timestep=None,
                    global_shape=tuple(volume.shape[-3:]),
                    window_size=window_size,
                    shift=(0, 0, 0),
                    event="stage_start",
                ),
                volume,
            )
        if shift_policy == "fixed":
            shifts = [(0, 0, 0)] * stage_steps
        else:
            shifts = sample_shift_schedule(
                stage_steps, window_size, shift_seed + stage
            )
            if shift_policy == "early":
                shifts[-1] = (0, 0, 0)
        for step_index, (timestep, shift) in enumerate(
            zip(model.scheduler.timesteps, shifts)
        ):
            velocity = predict_global_velocity_3d(
                model=model,
                volume=volume,
                timestep=timestep,
                window_size=window_size,
                patch_batch_size=patch_batch_size,
                shift=shift,
            )
            t = timestep.expand(batch_size).to(dtype)
            volume = model.scheduler.step(
                velocity, t, volume, generator=generator
            ).prev_sample
            if not torch.isfinite(volume).all():
                raise FloatingPointError(
                    f"non-finite global state at stage {stage}, step {step_index}"
                )
            if step_callback is not None:
                step_callback(
                    GlobalStep3D(
                        stage=stage,
                        step=step_index,
                        timestep=float(timestep),
                        global_shape=tuple(volume.shape[-3:]),
                        window_size=window_size,
                        shift=shift,
                        event="post_step",
                    ),
                    volume,
                )
    return volume
