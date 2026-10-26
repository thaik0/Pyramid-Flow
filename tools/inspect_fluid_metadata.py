"""Inspect NetCDF time/grid metadata without loading full velocity volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def coordinate_summary(coordinate):
    values = np.asarray(coordinate.values)
    result = {
        "dims": list(coordinate.dims),
        "size": int(values.size),
        "attributes": {key: json_value(value) for key, value in coordinate.attrs.items()},
    }
    if values.size:
        result["first"] = json_value(values.reshape(-1)[0])
        result["last"] = json_value(values.reshape(-1)[-1])
    if values.size > 1 and np.issubdtype(values.dtype, np.number):
        differences = np.diff(values.astype(np.float64).reshape(-1))
        result["spacing_mean"] = float(differences.mean())
        result["spacing_min"] = float(differences.min())
        result["spacing_max"] = float(differences.max())
        result["uniform_spacing"] = bool(np.allclose(differences, differences[0]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--files", nargs="+", default=("fluid_25.nc", "fluid_30.nc"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Install requirements-fluid.txt to inspect NetCDF data") from exc

    records = []
    for filename in args.files:
        path = args.data_root / filename
        with xr.open_dataset(path, cache=False, decode_times=False) as dataset:
            records.append({
                "filename": filename,
                "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
                "dataset_attributes": {
                    key: json_value(value) for key, value in dataset.attrs.items()
                },
                "coordinates": {
                    name: coordinate_summary(coordinate)
                    for name, coordinate in dataset.coords.items()
                },
                "velocity_variables": {
                    name: {
                        "dims": list(dataset[name].dims),
                        "shape": list(dataset[name].shape),
                        "attributes": {
                            key: json_value(value) for key, value in dataset[name].attrs.items()
                        },
                    }
                    for name in ("u", "v", "w")
                    if name in dataset
                },
            })
    payload = {
        "status": "metadata_inspection",
        "files": records,
        "questions": [
            "Are time coordinates continuous and ordered across files?",
            "What physical units and spacing belong to each spatial axis?",
            "Are u, v, and w collocated and expressed in the same units?",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
