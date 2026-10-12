"""Lazy NetCDF adapter for 3-D velocity fields.

No split, crop, or normalization policy is chosen here. Each item is one
physical timestep with channels ``u``, ``v``, and ``w``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_NUMBERED_FILE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


@dataclass(frozen=True)
class FluidSample:
    path: Path
    timestep: int


def _file_sort_key(path: Path) -> Tuple[str, int, str]:
    match = _NUMBERED_FILE.match(path.stem)
    if match is None:
        return path.stem, -1, path.name
    return match.group("prefix"), int(match.group("number")), path.name


def _sequence_number(path: Path) -> Optional[Tuple[str, int]]:
    match = _NUMBERED_FILE.match(path.stem)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("number"))


class FluidVolumeDataset(Dataset):
    """Return velocity timesteps as ``[C,D,H,W]`` float32 tensors.

    Timestep zero is omitted from a numbered file only when its predecessor is
    present and consecutively numbered, avoiding the documented shared boundary
    without treating gaps as overlaps.
    """

    def __init__(
        self,
        root: str | Path,
        pattern: str = "fluid_*.nc",
        variables: Sequence[str] = ("u", "v", "w"),
        normalization_json: Optional[str | Path] = None,
        crop_size: Optional[int] = None,
        crop_seed: int = 0,
        timesteps_per_file: int = 10,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.variables = tuple(variables)
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        if timesteps_per_file <= 0:
            raise ValueError("timesteps_per_file must be positive")
        self.timesteps_per_file = timesteps_per_file
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.files = sorted(self.root.glob(pattern), key=_file_sort_key)
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} in {self.root}")

        self.mean = None
        self.std = None
        if normalization_json is not None:
            payload = json.loads(Path(normalization_json).read_text())
            mean = torch.tensor(payload["mean"], dtype=torch.float32)
            std = torch.tensor(payload["std"], dtype=torch.float32)
            if mean.numel() != len(self.variables) or std.numel() != len(self.variables):
                raise ValueError("Statistics must contain one mean/std per channel")
            if torch.any(std <= 0):
                raise ValueError("Normalization standard deviations must be positive")
            self.mean = mean[:, None, None, None]
            self.std = std[:, None, None, None]
        self.samples = self._build_index()

    @staticmethod
    def _open_dataset(path: Path):
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError("Install requirements-fluid.txt to read NetCDF data") from exc
        return xr.open_dataset(path, cache=False)

    def _build_index(self) -> list[FluidSample]:
        samples = []
        previous_sequence = None
        for path in self.files:
            sequence = _sequence_number(path)
            overlaps = (
                previous_sequence is not None
                and sequence is not None
                and sequence[0] == previous_sequence[0]
                and sequence[1] == previous_sequence[1] + 1
            )
            samples.extend(FluidSample(path, t) for t in range(1 if overlaps else 0, self.timesteps_per_file))
            previous_sequence = sequence
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        sample = self.samples[index]
        arrays = []
        crop_origin = (0, 0, 0)
        with self._open_dataset(sample.path) as ds:
            missing = set(self.variables).difference(ds.variables)
            if missing:
                raise KeyError(f"{sample.path} is missing variables: {sorted(missing)}")
            if "time" not in ds.sizes:
                raise KeyError(f"{sample.path} has no 'time' dimension")
            if int(ds.sizes["time"]) != self.timesteps_per_file:
                raise ValueError(
                    f"{sample.path} has {ds.sizes['time']} timesteps; "
                    f"expected {self.timesteps_per_file}"
                )
            first = ds[self.variables[0]].isel(time=sample.timestep)
            spatial_shape = first.shape
            if len(spatial_shape) != 3:
                raise ValueError(f"{sample.path}:{self.variables[0]} produced {spatial_shape}, expected [D,H,W]")
            crop_indexers = {}
            if self.crop_size is not None:
                if any(size < self.crop_size for size in spatial_shape):
                    raise ValueError(f"Crop {self.crop_size} does not fit volume shape {tuple(spatial_shape)}")
                rng = np.random.default_rng(np.random.SeedSequence([self.crop_seed, index]))
                crop_origin = tuple(
                    int(rng.integers(0, size - self.crop_size + 1)) for size in spatial_shape
                )
                crop_indexers = {
                    dim: slice(origin, origin + self.crop_size)
                    for dim, origin in zip(first.dims, crop_origin)
                }
            for variable in self.variables:
                selected = ds[variable].isel(time=sample.timestep)
                if crop_indexers:
                    selected = selected.isel(crop_indexers)
                data = np.asarray(selected.values, dtype=np.float32)
                if data.ndim != 3:
                    raise ValueError(f"{sample.path}:{variable} produced {data.shape}, expected [D,H,W]")
                if arrays and data.shape != arrays[0].shape:
                    raise ValueError(f"Velocity components have inconsistent shapes in {sample.path}")
                arrays.append(data)
        volume = torch.from_numpy(np.stack(arrays, axis=0))
        if self.mean is not None:
            volume = (volume - self.mean) / self.std
        return {
            "volume": volume,
            "filename": sample.path.name,
            "path": str(sample.path),
            "timestep": sample.timestep,
            "channels": self.variables,
            "crop_origin": crop_origin,
        }

    def denormalize(self, volume: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return volume
        mean = self.mean.to(volume)
        std = self.std.to(volume)
        if volume.ndim == 5:
            mean, std = mean.unsqueeze(0), std.unsqueeze(0)
        return volume * std + mean

    def iter_sample_metadata(self) -> Iterable[Mapping[str, object]]:
        for sample in self.samples:
            yield {"filename": sample.path.name, "path": str(sample.path), "timestep": sample.timestep}
