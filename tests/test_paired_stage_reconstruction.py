import torch

from tools.evaluate_paired_stage_reconstruction import reconstruct


class _Scheduler:
    timesteps_per_stage = {2: torch.linspace(386, 1, 1000)}


class _ConstantVelocityModel:
    scheduler = _Scheduler()

    def __init__(self, velocity):
        self.velocity = velocity

    def predict_velocity(self, state, timestep):
        return self.velocity.expand_as(state)


def test_reconstruction_integrates_from_selected_ratio_to_zero():
    target = torch.randn(1, 3, 4, 4, 4)
    start = torch.randn_like(target)
    ratio = 0.75
    state = ratio * start + (1 - ratio) * target
    model = _ConstantVelocityModel(start - target)
    reconstructed = reconstruct(model, state, stage=2, initial_ratio=ratio, updates=8)
    torch.testing.assert_close(reconstructed, target)
