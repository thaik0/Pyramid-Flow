import torch

from tools.evaluate_hit_protocol import vorticity_squared


def test_constant_periodic_velocity_has_zero_vorticity():
    field = torch.ones(3, 8, 8, 8)
    assert torch.count_nonzero(vorticity_squared(field, (1.0, 1.0, 1.0))) == 0


def test_periodic_shear_vorticity_matches_centered_difference():
    count = 32
    spacing = 2 * torch.pi / count
    coordinate = torch.arange(count) * spacing
    x, y, z = torch.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    field = torch.stack((torch.sin(y), torch.zeros_like(x), torch.zeros_like(x)))
    measured = vorticity_squared(field, (spacing, spacing, spacing)).sqrt()
    expected = torch.abs(torch.cos(y)) * (torch.sin(spacing) / spacing)
    torch.testing.assert_close(measured, expected, atol=1e-5, rtol=1e-5)
