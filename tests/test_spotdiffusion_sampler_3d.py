from types import SimpleNamespace

import torch

from pyramid_dit.spotdiffusion_sampler_3d import (
    predict_global_velocity_3d,
    sample_global_spotdiffusion_3d,
)


class FakeScheduler:
    def __init__(self):
        self.start_sigmas = {0: 1.0, 1: 0.5}
        self.timesteps = None

    def set_timesteps(self, count, stage, device=None):
        self.timesteps = torch.linspace(1.0, 0.5, count, device=device)

    def step(self, velocity, timestep, volume, generator=None):
        return SimpleNamespace(prev_sample=volume - 0.25 * velocity)


class FakeModel:
    stages = (1, 2)

    def __init__(self):
        self.dit = SimpleNamespace(config=SimpleNamespace(sample_size=4))
        self.scheduler = FakeScheduler()
        self.batch_sizes = []

    def predict_velocity(self, patches, timestep):
        self.batch_sizes.append(patches.shape[0])
        return patches + 2


def test_shifted_local_prediction_reconstructs_pointwise_global_velocity():
    model = FakeModel()
    volume = torch.arange(3 * 8**3, dtype=torch.float32).reshape(1, 3, 8, 8, 8)
    velocity = predict_global_velocity_3d(
        model, volume, timestep=torch.tensor(0.5), window_size=2,
        patch_batch_size=7, shift=(1, 0, 1),
    )
    assert torch.equal(velocity, volume + 2)
    assert max(model.batch_sizes) <= 7
    assert sum(model.batch_sizes) == 64


def test_sampler_uses_stage_consistent_global_and_window_sizes():
    model = FakeModel()
    observed = []
    result = sample_global_spotdiffusion_3d(
        model=model,
        batch_size=1,
        global_size=8,
        steps=(1, 1),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(3),
        patch_batch_size=5,
        shifted_windows=False,
        step_callback=lambda step, volume: observed.append(step),
    )
    assert result.shape == (1, 3, 8, 8, 8)
    assert [step.global_shape for step in observed] == [(4, 4, 4), (8, 8, 8)]
    assert [step.window_size for step in observed] == [2, 4]
    assert [step.shift for step in observed] == [(0, 0, 0), (0, 0, 0)]
    assert torch.isfinite(result).all()


def test_early_shift_policy_keeps_each_stage_final_step_fixed():
    model = FakeModel()
    observed = []
    sample_global_spotdiffusion_3d(
        model=model,
        batch_size=1,
        global_size=8,
        steps=(3, 3),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(3),
        patch_batch_size=5,
        shift_policy="early",
        shift_seed=19,
        step_callback=lambda step, volume: observed.append(step),
    )
    stage_shifts = {
        stage: [step.shift for step in observed if step.stage == stage]
        for stage in range(2)
    }
    assert all(shifts[-1] == (0, 0, 0) for shifts in stage_shifts.values())
    assert all(any(shift != (0, 0, 0) for shift in shifts[:-1]) for shifts in stage_shifts.values())


def test_stage_start_callback_is_separate_from_post_step_callback():
    model = FakeModel()
    starts = []
    steps = []
    sample_global_spotdiffusion_3d(
        model=model,
        batch_size=1,
        global_size=8,
        steps=(1, 1),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(3),
        patch_batch_size=5,
        shift_policy="fixed",
        stage_start_callback=lambda step, volume: starts.append(step),
        step_callback=lambda step, volume: steps.append(step),
    )
    assert [item.event for item in starts] == ["stage_start", "stage_start"]
    assert [item.step for item in starts] == [-1, -1]
    assert [item.event for item in steps] == ["post_step", "post_step"]
