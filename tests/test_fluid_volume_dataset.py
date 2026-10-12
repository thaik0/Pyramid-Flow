import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "dataset" / "fluid_volume_dataset.py"
SPEC = importlib.util.spec_from_file_location("fluid_volume_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
FluidVolumeDataset = MODULE.FluidVolumeDataset


class _Variable:
    def __init__(self, values, dims=None):
        self.values = values
        self.dims = dims or (("time", "d", "h", "w") if values.ndim == 4 else ("d", "h", "w"))
        self.shape = values.shape

    def isel(self, indexers=None, time=None):
        if time is not None:
            return _Variable(self.values[time], self.dims[1:])
        slices = tuple(indexers.get(dim, slice(None)) for dim in self.dims)
        return _Variable(self.values[slices], self.dims)


class _Dataset:
    def __init__(self, offset):
        shape = (3, 2, 3, 4)
        self.variables = {}
        for channel, delta in zip(("u", "v", "w"), (0, 100, 200)):
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset + delta
            self.variables[channel] = _Variable(values)
        self.sizes = {"time": 3}

    def __getitem__(self, key):
        return self.variables[key]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _fake_files(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()


def _patch_reader(monkeypatch):
    def open_dataset(path):
        return _Dataset(int(path.stem.split("_")[-1]) * 10)
    monkeypatch.setattr(FluidVolumeDataset, "_open_dataset", staticmethod(open_dataset))


def test_shape_dtype_traceability_and_boundary_dedup(tmp_path, monkeypatch):
    _fake_files(tmp_path, ["fluid_100.nc", "fluid_101.nc"])
    _patch_reader(monkeypatch)
    dataset = FluidVolumeDataset(tmp_path, timesteps_per_file=3)
    assert len(dataset) == 5
    item = dataset[3]
    assert item["volume"].shape == (3, 2, 3, 4)
    assert item["volume"].numpy().dtype == np.float32
    assert item["filename"] == "fluid_101.nc"
    assert item["timestep"] == 1


def test_gap_does_not_drop_first_timestep(tmp_path, monkeypatch):
    _fake_files(tmp_path, ["fluid_100.nc", "fluid_102.nc"])
    _patch_reader(monkeypatch)
    assert len(FluidVolumeDataset(tmp_path, timesteps_per_file=3)) == 6


def test_explicit_normalization_is_invertible(tmp_path, monkeypatch):
    _fake_files(tmp_path, ["fluid_100.nc"])
    _patch_reader(monkeypatch)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"mean": [1, 2, 3], "std": [2, 4, 5]}))
    raw = FluidVolumeDataset(tmp_path, timesteps_per_file=3)[0]["volume"]
    normalized_dataset = FluidVolumeDataset(tmp_path, normalization_json=stats, timesteps_per_file=3)
    normalized = normalized_dataset[0]["volume"]
    np.testing.assert_allclose(normalized_dataset.denormalize(normalized), raw)
