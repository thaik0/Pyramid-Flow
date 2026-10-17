import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "evaluate_spotdiffusion_ablation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_spotdiffusion_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_boundary_metric_detects_fixed_patch_plane_jump():
    field = torch.zeros(1, 3, 8, 8, 8)
    field[:, :, 4:] = 10
    result = MODULE.boundary_artifacts(field, window_size=4)
    assert result["axes"]["depth"]["boundary_to_interior_ratio"] > 1
    assert result["aggregate"]["boundary_to_interior_ratio"] > 1


def test_zero_field_has_zero_energy_spectrum():
    field = torch.zeros(1, 3, 8, 8, 8)
    spectrum, counts = MODULE.isotropic_energy_spectrum(field)
    assert spectrum.sum() == 0
    assert counts.sum() == 8**3
