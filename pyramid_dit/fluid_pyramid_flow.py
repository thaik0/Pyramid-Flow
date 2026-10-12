"""Minimal isotropic 3-D adaptation of the official Pyramid Flow objective."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_schedulers import PyramidFlowMatchEulerDiscreteScheduler
from .mmdit_modules import PyramidDiffusionMMDiT


class FluidPyramidFlow(nn.Module):
    """Unconditional volume runner preserving the upstream flow schedule/loss."""

    def __init__(
        self,
        patch_size: int = 4,
        stages: Sequence[int] = (1, 2, 4),
        stage_range: Sequence[float] = (0, 1 / 3, 2 / 3, 1),
        sample_ratios: Sequence[int] = (1, 2, 1),
        num_layers: int = 8,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.stages = tuple(stages)
        self.sample_ratios = tuple(sample_ratios)
        self.patch_size = patch_size
        inner_dim = num_attention_heads * attention_head_dim
        self.condition_dim = inner_dim
        self.pooled_dim = inner_dim
        self.dit = PyramidDiffusionMMDiT(
            sample_size=64,
            patch_size=patch_size,
            in_channels=3,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_projection_dim=inner_dim,
            pooled_projection_dim=self.pooled_dim,
            joint_attention_dim=self.condition_dim,
            pos_embed_max_size=None,
            pos_embed_type="rope",
            temp_pos_embed_type="rope",
            use_temporal_causal=False,
            use_flash_attn=False,
            add_temp_pos_embed=False,
            volume_mode=True,
        )
        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            stages=len(self.stages), stage_range=list(stage_range), gamma=1 / 3
        )

    def pyramid(self, volume: torch.Tensor) -> list[torch.Tensor]:
        """Return low-to-high isotropic clean volumes."""
        levels = [volume]
        for _ in range(len(self.stages) - 1):
            levels.append(F.interpolate(levels[-1], scale_factor=0.5, mode="trilinear", align_corners=False))
        return list(reversed(levels))

    def _conditions(self, batch_size: int, device, dtype):
        context = torch.zeros(batch_size, 1, self.condition_dim, device=device, dtype=dtype)
        mask = torch.ones(batch_size, 1, device=device, dtype=torch.long)
        pooled = torch.zeros(batch_size, self.pooled_dim, device=device, dtype=dtype)
        return context, mask, pooled

    def training_loss(
        self,
        volume: torch.Tensor,
        generator: torch.Generator | None = None,
        stage: int | None = None,
    ):
        levels = self.pyramid(volume)
        if stage is None:
            weights = torch.tensor(self.sample_ratios, device=volume.device, dtype=torch.float32)
            stage = int(torch.multinomial(weights, 1, generator=generator).item())
        if not 0 <= stage < len(levels):
            raise ValueError(f"stage must be in [0, {len(levels) - 1}]")
        clean = levels[stage]
        noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
        start_sigma = self.scheduler.start_sigmas[stage]
        end_sigma = self.scheduler.end_sigmas[stage]
        if stage == 0:
            start = noise
        else:
            lifted = F.interpolate(levels[stage - 1], size=clean.shape[-3:], mode="nearest")
            start = start_sigma * noise + (1 - start_sigma) * lifted
        end = clean if stage == len(levels) - 1 else end_sigma * noise + (1 - end_sigma) * clean

        count = self.scheduler.config.num_train_timesteps
        index = torch.randint(count, (volume.shape[0],), generator=generator, device=volume.device)
        ratio = self.scheduler.sigmas_per_stage[stage][index.cpu()].to(volume.device, volume.dtype)
        timestep = self.scheduler.timesteps_per_stage[stage][index.cpu()].to(volume.device, volume.dtype)
        ratio_view = ratio.view(-1, 1, 1, 1, 1)
        noisy = ratio_view * start + (1 - ratio_view) * end
        target = start - end
        context, mask, pooled = self._conditions(volume.shape[0], volume.device, volume.dtype)
        prediction = self.dit(
            sample=[noisy], timestep_ratio=timestep,
            encoder_hidden_states=context, encoder_attention_mask=mask,
            pooled_projections=pooled,
        )[0]
        return F.mse_loss(prediction.float(), target.float()), {"stage": stage}

    @torch.no_grad()
    def sample(self, batch_size: int, size: int, steps: Sequence[int], device, dtype, generator=None):
        if len(steps) != len(self.stages):
            raise ValueError("steps must contain one value per pyramid stage")
        low_size = size // self.stages[-1]
        volume = torch.randn((batch_size, 3, low_size, low_size, low_size), generator=generator, device=device, dtype=dtype)
        context, mask, pooled = self._conditions(batch_size, device, dtype)
        for stage, stage_steps in enumerate(steps):
            self.scheduler.set_timesteps(stage_steps, stage, device=device)
            if stage:
                volume = F.interpolate(volume, scale_factor=2, mode="nearest")
                sigma = self.scheduler.start_sigmas[stage]
                volume = (1 - sigma) * volume + sigma * torch.randn(volume.shape, generator=generator, device=device, dtype=dtype)
            for timestep in self.scheduler.timesteps:
                t = timestep.expand(batch_size).to(dtype)
                velocity = self.dit(
                    sample=[volume], timestep_ratio=t,
                    encoder_hidden_states=context, encoder_attention_mask=mask,
                    pooled_projections=pooled,
                )[0]
                volume = self.scheduler.step(velocity, t, volume, generator=generator).prev_sample
        return volume
