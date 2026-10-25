import torch

from tools.build_fluid_reference_baseline import field_metrics, periodic_divergence_rms


def test_constant_field_has_zero_fluctuation_energy_and_divergence():
    field = torch.stack([torch.full((8, 8, 8), value) for value in (1.0, 2.0, 3.0)])
    metrics, fluctuations = field_metrics(field, (1.0, 1.0, 1.0))
    assert torch.count_nonzero(fluctuations) == 0
    assert metrics["fluctuation_kinetic_energy"] == 0.0
    assert metrics["divergence_rms"] == 0.0


def test_periodic_divergence_free_field():
    coordinate = torch.arange(16, dtype=torch.float32) * (2 * torch.pi / 16)
    z, y, x = torch.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    field = torch.stack((torch.sin(y), torch.sin(x), torch.sin(z)))
    assert periodic_divergence_rms(field, (1.0, 1.0, 1.0)) < 1e-6


def test_field_metrics_separates_mean_flow_from_fluctuation_energy():
    field = torch.zeros(3, 4, 4, 4)
    field[0] = 2.0
    metrics, _ = field_metrics(field, (1.0, 1.0, 1.0))
    assert metrics["kinetic_energy"] == 2.0
    assert metrics["fluctuation_kinetic_energy"] == 0.0
