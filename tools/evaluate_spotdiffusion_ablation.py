"""Paired diagnostics for fixed versus shifted 3-D patch inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


COMPONENTS = ("u", "v", "w")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf") if numerator > 0 else 1.0
    return numerator / denominator


def component_statistics(field: torch.Tensor) -> dict:
    return {
        name: {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for name, values in zip(COMPONENTS, field[0])
    }


def divergence_rms(field: torch.Tensor, spacing: tuple[float, float, float]) -> float:
    divergence = torch.zeros_like(field[:, 0])
    for component, dimension, dx in zip(range(3), (-3, -2, -1), spacing):
        values = field[:, component]
        divergence.add_(
            (torch.roll(values, -1, dimension) - torch.roll(values, 1, dimension))
            / (2 * dx)
        )
    return float(divergence.square().mean().sqrt())


def boundary_artifacts(field: torch.Tensor, window_size: int) -> dict:
    """Compare adjacent-voxel jumps on patch planes with all other planes."""
    axes = {}
    aggregate_boundary_sum = 0.0
    aggregate_boundary_sq_sum = 0.0
    aggregate_boundary_count = 0
    aggregate_interior_sum = 0.0
    aggregate_interior_sq_sum = 0.0
    aggregate_interior_count = 0
    for label, dimension in zip(("depth", "height", "width"), (-3, -2, -1)):
        jumps = torch.diff(field, dim=dimension).abs()
        length = field.shape[dimension]
        indices = torch.arange(
            window_size - 1, length - 1, window_size, device=field.device
        )
        boundary = torch.index_select(jumps, dimension, indices)
        boundary_sum = float(boundary.sum())
        boundary_sq_sum = float(boundary.square().sum())
        boundary_count = boundary.numel()
        total_sum = float(jumps.sum())
        total_sq_sum = float(jumps.square().sum())
        interior_sum = total_sum - boundary_sum
        interior_sq_sum = total_sq_sum - boundary_sq_sum
        interior_count = jumps.numel() - boundary_count
        boundary_mean = boundary_sum / boundary_count
        interior_mean = interior_sum / interior_count
        axes[label] = {
            "boundary_mean_absolute_jump": boundary_mean,
            "interior_mean_absolute_jump": interior_mean,
            "boundary_to_interior_ratio": safe_ratio(boundary_mean, interior_mean),
            "boundary_rms_jump": (boundary_sq_sum / boundary_count) ** 0.5,
            "interior_rms_jump": (interior_sq_sum / interior_count) ** 0.5,
        }
        aggregate_boundary_sum += boundary_sum
        aggregate_boundary_sq_sum += boundary_sq_sum
        aggregate_boundary_count += boundary_count
        aggregate_interior_sum += interior_sum
        aggregate_interior_sq_sum += interior_sq_sum
        aggregate_interior_count += interior_count
    boundary_mean = aggregate_boundary_sum / aggregate_boundary_count
    interior_mean = aggregate_interior_sum / aggregate_interior_count
    return {
        "window_size": window_size,
        "axes": axes,
        "aggregate": {
            "boundary_mean_absolute_jump": boundary_mean,
            "interior_mean_absolute_jump": interior_mean,
            "boundary_to_interior_ratio": safe_ratio(boundary_mean, interior_mean),
            "boundary_rms_jump": (
                aggregate_boundary_sq_sum / aggregate_boundary_count
            ) ** 0.5,
            "interior_rms_jump": (
                aggregate_interior_sq_sum / aggregate_interior_count
            ) ** 0.5,
        },
    }


def isotropic_energy_spectrum(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return the full-mode-weighted spectrum from a real-valued 3-D FFT."""
    depth, height, width = field.shape[-3:]
    if not (depth == height == width):
        raise ValueError("energy spectrum currently requires a cubic field")
    transformed = torch.fft.rfftn(field, dim=(-3, -2, -1), norm="forward")
    energy = 0.5 * transformed.abs().square().sum(dim=1)[0]
    kz = torch.fft.fftfreq(depth, device=field.device) * depth
    ky = torch.fft.fftfreq(height, device=field.device) * height
    kx = torch.fft.rfftfreq(width, device=field.device) * width
    zz, yy, xx = torch.meshgrid(kz, ky, kx, indexing="ij")
    shells = torch.sqrt(xx.square() + yy.square() + zz.square()).round().long()
    weights = torch.full((kx.numel(),), 2.0, device=field.device)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    mode_weights = weights.view(1, 1, -1).expand_as(energy)
    spectrum = torch.bincount(
        shells.reshape(-1),
        weights=(energy * mode_weights).reshape(-1),
    )
    counts = torch.bincount(
        shells.reshape(-1), weights=mode_weights.reshape(-1)
    )
    return spectrum.cpu().numpy(), counts.cpu().numpy()


def evaluate_field(
    path: Path,
    device: torch.device,
    spacing: tuple[float, float, float],
    window_size: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 5 or array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(f"expected [1,3,D,H,W], got {array.shape} from {path}")
    field = torch.from_numpy(np.asarray(array)).to(device=device, dtype=torch.float32)
    spectrum, counts = isotropic_energy_spectrum(field)
    center = field.shape[-1] // 2
    slices = field[0, :, center].cpu().numpy()
    speed_slice = field[0, :, center].square().sum(0).sqrt().cpu().numpy()
    metrics = {
        "shape": list(field.shape),
        "finite": bool(torch.isfinite(field).all()),
        "component_statistics": component_statistics(field),
        "mean_kinetic_energy": float(0.5 * field.square().sum(1).mean()),
        "divergence_rms_periodic": divergence_rms(field, spacing),
        "boundary_artifacts": boundary_artifacts(field, window_size),
    }
    del field
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, spectrum, counts, np.concatenate((speed_slice[None], slices))


def write_visuals(slices: dict[str, np.ndarray], output_dir: Path) -> str:
    np.savez_compressed(output_dir / "central_slices.npz", **slices)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "central_slices.npz (matplotlib unavailable; PNG not created)"

    labels = ("speed", "u", "v", "w")
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for column, label in enumerate(labels):
        combined = np.concatenate(
            (slices["fixed"][column].ravel(), slices["shifted"][column].ravel())
        )
        low, high = np.percentile(combined, (1, 99))
        if label != "speed":
            limit = max(abs(low), abs(high))
            low, high = -limit, limit
        for row, mode in enumerate(("fixed", "shifted")):
            image = axes[row, column].imshow(
                slices[mode][column],
                origin="lower",
                cmap="viridis" if label == "speed" else "coolwarm",
                vmin=low,
                vmax=high,
            )
            axes[row, column].set_title(f"{mode}: {label}")
            axes[row, column].set_axis_off()
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.savefig(output_dir / "central_slices.png", dpi=160)
    plt.close(figure)

    difference = slices["shifted"][0] - slices["fixed"][0]
    limit = np.percentile(np.abs(difference), 99)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(
        difference, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axis.set_title("Shifted minus fixed: speed")
    axis.set_axis_off()
    figure.colorbar(image, ax=axis)
    figure.savefig(output_dir / "speed_difference.png", dpi=160)
    plt.close(figure)
    return "central_slices.png and speed_difference.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-field", type=Path, required=True)
    parser.add_argument("--shifted-field", type=Path, required=True)
    parser.add_argument("--fixed-metadata", type=Path, required=True)
    parser.add_argument("--shifted-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    spacing = tuple(args.spacing)

    metrics = {}
    spectra = {}
    counts = {}
    slices = {}
    for mode, path in (("fixed", args.fixed_field), ("shifted", args.shifted_field)):
        metrics[mode], spectra[mode], counts[mode], slices[mode] = evaluate_field(
            path, device, spacing, args.window_size
        )
    metadata = {
        "fixed": json.loads(args.fixed_metadata.read_text()),
        "shifted": json.loads(args.shifted_metadata.read_text()),
    }
    controlled_keys = (
        "checkpoint_sha256",
        "sampling_seed",
        "global_size",
        "steps_per_stage",
        "patch_batch_size",
    )
    controls_match = {
        key: metadata["fixed"][key] == metadata["shifted"][key]
        for key in controlled_keys
    }
    if not all(controls_match.values()):
        raise ValueError(f"ablation controls do not match: {controls_match}")
    if metadata["fixed"]["mode"] != "fixed" or metadata["shifted"]["mode"] != "shifted":
        raise ValueError("metadata modes do not describe fixed and shifted arms")

    fixed_seam = metrics["fixed"]["boundary_artifacts"]["aggregate"]
    shifted_seam = metrics["shifted"]["boundary_artifacts"]["aggregate"]
    seam_reduction = 100 * (
        fixed_seam["boundary_to_interior_ratio"]
        - shifted_seam["boundary_to_interior_ratio"]
    ) / fixed_seam["boundary_to_interior_ratio"]
    spectrum_length = min(len(spectra["fixed"]), len(spectra["shifted"]))
    fixed_spectrum = spectra["fixed"][:spectrum_length]
    shifted_spectrum = spectra["shifted"][:spectrum_length]
    spectrum_relative_l1 = float(
        np.abs(shifted_spectrum - fixed_spectrum).sum()
        / max(np.abs(fixed_spectrum).sum(), np.finfo(float).eps)
    )
    visual_outputs = write_visuals(slices, args.output_dir)
    report = {
        "status": "single_seed_provisional_ablation",
        "question": "Does shifting reduce seams without hurting the generated field?",
        "controls_match": controls_match,
        "metrics": metrics,
        "comparison": {
            "seam_ratio_reduction_percent": seam_reduction,
            "divergence_change_percent": 100
            * (
                metrics["shifted"]["divergence_rms_periodic"]
                - metrics["fixed"]["divergence_rms_periodic"]
            )
            / metrics["fixed"]["divergence_rms_periodic"],
            "kinetic_energy_change_percent": 100
            * (
                metrics["shifted"]["mean_kinetic_energy"]
                - metrics["fixed"]["mean_kinetic_energy"]
            )
            / metrics["fixed"]["mean_kinetic_energy"],
            "energy_spectrum_relative_l1": spectrum_relative_l1,
            "runtime_seconds": {
                mode: metadata[mode]["elapsed_seconds"] for mode in metadata
            },
            "peak_memory_bytes": {
                mode: metadata[mode]["peak_memory_bytes"] for mode in metadata
            },
        },
        "visual_outputs": visual_outputs,
        "spacing": list(spacing),
        "caveat": (
            "One seed can establish the paired mechanism but not statistical or "
            "scientific equivalence. Periodic divergence and unit spacing are provisional."
        ),
    }
    (args.output_dir / "ablation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "energy_spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shell", "fixed_energy", "shifted_energy", "mode_count"))
        for shell in range(spectrum_length):
            writer.writerow(
                (
                    shell,
                    float(fixed_spectrum[shell]),
                    float(shifted_spectrum[shell]),
                    float(counts["fixed"][shell]),
                )
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
