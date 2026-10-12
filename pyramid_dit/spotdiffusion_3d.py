"""Tensor-only building blocks for shifted, non-overlapping 3-D windows.

This module deliberately contains no model or scheduler logic.  It operates on
batched channel-first volumes shaped ``[B, C, D, H, W]`` and only shifts or
tiles the three spatial axes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import torch


Spatial3D = tuple[int, int, int]


def _spatial_tuple(value: int | Sequence[int], name: str) -> Spatial3D:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(int(part) for part in value)
        if len(result) != 3:
            raise ValueError(f"{name} must contain exactly three values")
    if any(part <= 0 for part in result):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True)
class TilingSpec3D:
    """Metadata needed to invert :func:`split_nonoverlapping_3d`."""

    batch_size: int
    channels: int
    spatial_shape: Spatial3D
    window_size: Spatial3D
    grid_shape: Spatial3D

    @property
    def patches_per_sample(self) -> int:
        grid_d, grid_h, grid_w = self.grid_shape
        return grid_d * grid_h * grid_w


def patch_origins_3d(
    spatial_shape: int | Sequence[int],
    window_size: int | Sequence[int],
) -> list[Spatial3D]:
    """Return non-overlapping patch origins in depth-major grid order."""
    spatial = _spatial_tuple(spatial_shape, "spatial_shape")
    window = _spatial_tuple(window_size, "window_size")
    if any(size % width for size, width in zip(spatial, window)):
        raise ValueError(
            f"spatial shape {spatial} must be divisible by window size {window}"
        )
    return list(
        product(
            range(0, spatial[0], window[0]),
            range(0, spatial[1], window[1]),
            range(0, spatial[2], window[2]),
        )
    )


def split_nonoverlapping_3d(
    volume: torch.Tensor,
    window_size: int | Sequence[int],
) -> tuple[torch.Tensor, TilingSpec3D]:
    """Split ``[B,C,D,H,W]`` into exact non-overlapping spatial windows.

    The result is ordered first by batch, then by depth, height, and width
    grid coordinates.  The channel axis is preserved inside every patch.
    """
    if volume.ndim != 5:
        raise ValueError(
            f"volume must have shape [B,C,D,H,W], got {tuple(volume.shape)}"
        )
    batch_size, channels, depth, height, width = volume.shape
    patch_d, patch_h, patch_w = _spatial_tuple(window_size, "window_size")
    spatial_shape = (depth, height, width)
    if any(
        size % patch
        for size, patch in zip(spatial_shape, (patch_d, patch_h, patch_w))
    ):
        raise ValueError(
            f"spatial shape {spatial_shape} must be divisible by window size "
            f"{(patch_d, patch_h, patch_w)}"
        )

    grid_d, grid_h, grid_w = (
        depth // patch_d,
        height // patch_h,
        width // patch_w,
    )
    patches = (
        volume.reshape(
            batch_size,
            channels,
            grid_d,
            patch_d,
            grid_h,
            patch_h,
            grid_w,
            patch_w,
        )
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(
            batch_size * grid_d * grid_h * grid_w,
            channels,
            patch_d,
            patch_h,
            patch_w,
        )
    )
    spec = TilingSpec3D(
        batch_size=batch_size,
        channels=channels,
        spatial_shape=spatial_shape,
        window_size=(patch_d, patch_h, patch_w),
        grid_shape=(grid_d, grid_h, grid_w),
    )
    return patches, spec


def stitch_nonoverlapping_3d(
    patches: torch.Tensor,
    spec: TilingSpec3D,
) -> torch.Tensor:
    """Exactly invert :func:`split_nonoverlapping_3d` without averaging."""
    grid_d, grid_h, grid_w = spec.grid_shape
    patch_d, patch_h, patch_w = spec.window_size
    expected_shape = (
        spec.batch_size * spec.patches_per_sample,
        spec.channels,
        patch_d,
        patch_h,
        patch_w,
    )
    if tuple(patches.shape) != expected_shape:
        raise ValueError(
            f"patches have shape {tuple(patches.shape)}, expected {expected_shape}"
        )

    depth, height, width = spec.spatial_shape
    return (
        patches.reshape(
            spec.batch_size,
            grid_d,
            grid_h,
            grid_w,
            spec.channels,
            patch_d,
            patch_h,
            patch_w,
        )
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(spec.batch_size, spec.channels, depth, height, width)
    )


def cyclic_shift_3d(volume: torch.Tensor, shift: Sequence[int]) -> torch.Tensor:
    """Apply a wrap-around translation to D, H, and W only."""
    if volume.ndim != 5:
        raise ValueError(
            f"volume must have shape [B,C,D,H,W], got {tuple(volume.shape)}"
        )
    spatial_shift = _spatial_tuple_allow_zero(shift, "shift")
    return torch.roll(volume, shifts=spatial_shift, dims=(-3, -2, -1))


def inverse_cyclic_shift_3d(
    volume: torch.Tensor, shift: Sequence[int]
) -> torch.Tensor:
    """Reverse :func:`cyclic_shift_3d`."""
    spatial_shift = _spatial_tuple_allow_zero(shift, "shift")
    return torch.roll(
        volume,
        shifts=tuple(-part for part in spatial_shift),
        dims=(-3, -2, -1),
    )


def _spatial_tuple_allow_zero(value: Sequence[int], name: str) -> Spatial3D:
    result = tuple(int(part) for part in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(part < 0 for part in result):
        raise ValueError(f"{name} values must be non-negative")
    return result


def sample_shift_schedule(
    num_steps: int,
    window_size: int | Sequence[int],
    seed: int,
) -> list[Spatial3D]:
    """Sample a reproducible uniform spatial shift independently per step.

    Each axis is sampled from ``{0, ..., window_axis - 1}``.  Repeated shifts
    remain possible, as required by independent uniform sampling.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    window = _spatial_tuple(window_size, "window_size")
    rng = random.Random(seed)
    return [
        tuple(rng.randrange(width) for width in window)
        for _ in range(num_steps)
    ]
