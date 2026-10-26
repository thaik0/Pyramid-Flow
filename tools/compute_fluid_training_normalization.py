"""Compute exact channel normalization over the selected training timesteps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-start", type=int, default=200)
    parser.add_argument("--index-count", type=int, default=128)
    parser.add_argument("--pattern", default="fluid_*.nc")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = FluidVolumeDataset(args.data_root, pattern=args.pattern)
    indices = range(args.index_start, args.index_start + args.index_count)
    if args.index_count <= 0 or indices.start < 0 or indices.stop > len(dataset):
        raise ValueError(f"training interval is outside dataset indices [0,{len(dataset) - 1}]")

    device = torch.device(args.device)
    total = torch.zeros(3, dtype=torch.float64, device=device)
    total_square = torch.zeros(3, dtype=torch.float64, device=device)
    minimum = torch.full((3,), float("inf"), dtype=torch.float32, device=device)
    maximum = torch.full((3,), -float("inf"), dtype=torch.float32, device=device)
    voxel_count = 0
    manifest = []
    for position, index in enumerate(indices, start=1):
        item = dataset[index]
        field = item["volume"].to(device=device, dtype=torch.float32)
        flattened = field.flatten(1)
        total += flattened.sum(dim=1, dtype=torch.float64)
        total_square += flattened.square().sum(dim=1, dtype=torch.float64)
        minimum = torch.minimum(minimum, flattened.amin(dim=1))
        maximum = torch.maximum(maximum, flattened.amax(dim=1))
        voxel_count += flattened.shape[1]
        manifest.append({
            "dataset_index": index,
            "filename": item["filename"],
            "timestep": int(item["timestep"]),
        })
        print(f"[{position}/{args.index_count}] {item['filename']} t={item['timestep']}")

    mean = total / voxel_count
    variance = torch.clamp(total_square / voxel_count - mean.square(), min=0)
    payload = {
        "status": "exact_full_volume_training_statistics",
        "channels": ["u", "v", "w"],
        "mean": mean.cpu().tolist(),
        "std": variance.sqrt().cpu().tolist(),
        "min": minimum.cpu().tolist(),
        "max": maximum.cpu().tolist(),
        "voxel_count_per_channel": voxel_count,
        "training_index_range": [indices.start, indices.stop - 1],
        "sample_count": args.index_count,
        "manifest": manifest,
        "warning": (
            "Use these statistics for new training runs. Changing normalization at "
            "inference time is incompatible with checkpoints trained using other statistics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
