"""Build a traceable multi-timestep physical reference for fluid generation.

The selected samples must be one contiguous block and must not overlap the
training interval.  Fields are processed one at a time so a 384^3 baseline
does not require holding the full reference ensemble in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402


METRIC_NAMES = (
    "mean_u", "mean_v", "mean_w", "std_u", "std_v", "std_w",
    "kinetic_energy", "fluctuation_kinetic_energy", "divergence_rms",
)


def periodic_divergence_rms(field: torch.Tensor, spacing) -> float:
    """Centered-difference divergence for a periodic [3,D,H,W] field."""
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError("field must have shape [3,D,H,W]")
    derivatives = [
        (torch.roll(field[c], -1, dims=axis) - torch.roll(field[c], 1, dims=axis))
        / (2.0 * float(dx))
        for c, axis, dx in zip(range(3), range(3), spacing)
    ]
    return float(torch.sqrt(torch.mean(torch.square(sum(derivatives)))).cpu())


def isotropic_energy_spectrum(field: torch.Tensor, spacing):
    """Return radial fluctuation-energy spectrum using an rFFT."""
    spatial_shape = field.shape[-3:]
    n_voxels = int(np.prod(spatial_shape))
    frequencies = (
        torch.fft.fftfreq(spatial_shape[0], d=float(spacing[0]), device=field.device),
        torch.fft.fftfreq(spatial_shape[1], d=float(spacing[1]), device=field.device),
        torch.fft.rfftfreq(spatial_shape[2], d=float(spacing[2]), device=field.device),
    )
    fundamental = min(float(torch.abs(freq[1]).cpu()) for freq in frequencies)
    kz, ky, kx = torch.meshgrid(*frequencies, indexing="ij")
    shells = torch.round(torch.sqrt(kx.square() + ky.square() + kz.square()) / fundamental).long()
    weights = torch.ones(spatial_shape[2] // 2 + 1, device=field.device)
    if spatial_shape[2] % 2 == 0:
        weights[1:-1] = 2
    else:
        weights[1:] = 2
    energy = torch.zeros_like(kx, dtype=torch.float64)
    for component in field:
        transformed = torch.fft.rfftn(component)
        energy += 0.5 * transformed.abs().square().double() * weights / (n_voxels ** 2)
    spectrum = torch.bincount(shells.ravel(), weights=energy.ravel())
    mode_count = torch.bincount(shells.ravel(), weights=weights.expand_as(shells).ravel())
    return spectrum.cpu().numpy(), mode_count.cpu().numpy()


def field_metrics(field: torch.Tensor, spacing):
    means = field.mean(dim=(1, 2, 3))
    fluctuations = field - means[:, None, None, None]
    stds = fluctuations.square().mean(dim=(1, 2, 3)).sqrt()
    result = {
        **{f"mean_{name}": float(value.cpu()) for name, value in zip("uvw", means)},
        **{f"std_{name}": float(value.cpu()) for name, value in zip("uvw", stds)},
        "kinetic_energy": float(0.5 * field.square().sum(dim=0).mean().cpu()),
        "fluctuation_kinetic_energy": float(
            0.5 * fluctuations.square().sum(dim=0).mean().cpu()
        ),
        "divergence_rms": periodic_divergence_rms(field, spacing),
    }
    return result, fluctuations


def summarize(rows):
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std_across_timesteps": float(values.std()),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-count", type=int, default=16)
    parser.add_argument("--training-index-start", type=int, default=200)
    parser.add_argument("--training-index-end", type=int, default=327)
    parser.add_argument("--minimum-buffer", type=int, default=32)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--pattern", default="fluid_*.nc")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = FluidVolumeDataset(args.data_root, pattern=args.pattern)
    indices = list(range(args.index_start, args.index_start + args.index_count))
    if not indices or indices[0] < 0 or indices[-1] >= len(dataset):
        raise ValueError(f"reference block is outside dataset indices [0,{len(dataset) - 1}]")
    train = range(args.training_index_start, args.training_index_end + 1)
    if set(indices).intersection(train):
        raise ValueError("reference block overlaps the checkpoint training interval")
    distance = min(abs(i - j) for i in indices for j in (train.start, train.stop - 1))
    if distance < args.minimum_buffer:
        raise ValueError(
            f"reference block has only a {distance}-sample temporal buffer; "
            f"require at least {args.minimum_buffer}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    spectrum_sum = None
    mode_count = None
    device = torch.device(args.device)
    for index in indices:
        item = dataset[index]
        field = item["volume"].to(device=device, dtype=torch.float32)
        metrics, fluctuations = field_metrics(field, args.spacing)
        spectrum, counts = isotropic_energy_spectrum(fluctuations, args.spacing)
        spectrum_sum = spectrum if spectrum_sum is None else spectrum_sum + spectrum
        mode_count = counts
        rows.append({"dataset_index": index, **metrics})
        manifest.append({
            "dataset_index": index,
            "filename": item["filename"],
            "timestep": int(item["timestep"]),
        })
        print(f"[{len(rows)}/{len(indices)}] {item['filename']} t={item['timestep']}")

    summary = {
        "status": "provisional_single_trajectory_heldout_index_block",
        "periodic_boundaries": True,
        "assumed_single_trajectory": True,
        "reference_index_range": [indices[0], indices[-1]],
        "training_index_range": [args.training_index_start, args.training_index_end],
        "index_buffer_from_training": distance,
        "sample_count": len(rows),
        "spacing": list(args.spacing),
        "spacing_status": "physical" if tuple(args.spacing) != (1.0, 1.0, 1.0) else "index_units_only",
        "spectrum_uses_spatial_mean_subtracted_fields": True,
        "metrics": summarize(rows),
        "manifest": manifest,
        "caveat": (
            "Dataset-index adjacency does not guarantee temporal continuity when numbered "
            "files are missing. Inspect the stored time coordinates. If spacing remains 1, "
            "divergence is in index units and only like-for-like comparisons are valid."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_timestep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset_index",) + METRIC_NAMES)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "mean_energy_spectrum.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shell", "mean_energy", "mode_count"])
        writer.writerows(
            (shell, float(value / len(rows)), float(mode_count[shell]))
            for shell, value in enumerate(spectrum_sum)
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
