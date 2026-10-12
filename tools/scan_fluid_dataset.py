"""Compute traceable per-channel statistics from a bounded sample."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--pattern", default="fluid_*.nc")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = FluidVolumeDataset(args.data_root, pattern=args.pattern)
    count = min(args.max_samples, len(dataset))
    indices = np.sort(np.random.default_rng(args.seed).choice(len(dataset), count, replace=False))
    channels = len(dataset.variables)
    n = np.zeros(channels, dtype=np.int64)
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    minima = np.full(channels, np.inf)
    maxima = np.full(channels, -np.inf)
    sampled = []
    for index in indices:
        item = dataset[int(index)]
        flat = item["volume"].numpy().reshape(channels, -1).astype(np.float64, copy=False)
        n += flat.shape[1]
        total += flat.sum(axis=1)
        total_sq += np.square(flat).sum(axis=1)
        minima = np.minimum(minima, flat.min(axis=1))
        maxima = np.maximum(maxima, flat.max(axis=1))
        sampled.append({"filename": item["filename"], "timestep": item["timestep"]})
    mean = total / n
    result = {
        "channels": list(dataset.variables),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(total_sq / n - np.square(mean), 0)).tolist(),
        "min": minima.tolist(), "max": maxima.tolist(),
        "voxel_count": n.tolist(), "dataset_sample_count": len(dataset),
        "sampled": sampled, "seed": args.seed,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
