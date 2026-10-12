import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "evaluate_fluid_fields.py"
SPEC = importlib.util.spec_from_file_location("evaluate_fluid_fields", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_constant_field_has_zero_divergence_and_dc_energy():
    fields = np.ones((2, 3, 8, 8, 8), dtype=np.float64)
    assert MODULE.divergence_rms(fields, (1, 1, 1)) == 0.0
    spectrum, _ = MODULE.energy_spectrum(fields, (1, 1, 1))
    assert spectrum[0] > 0
    np.testing.assert_allclose(spectrum[1:], 0, atol=1e-12)


def test_component_statistics():
    fields = np.zeros((1, 3, 4, 4, 4), dtype=np.float64)
    fields[:, 1] = 2
    stats = MODULE.component_statistics(fields)
    assert stats["u"]["mean"] == 0
    assert stats["v"]["mean"] == 2
