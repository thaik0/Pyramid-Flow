"""Lab-aligned physical diagnostics for periodic 3-D HIT velocity fields."""

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
from tools.evaluate_spotdiffusion_ablation import isotropic_energy_spectrum  # noqa: E402


def centered_derivative(values, axis, spacing):
    return (
        torch.roll(values, -1, dims=axis) - torch.roll(values, 1, dims=axis)
    ) / (2 * spacing)


def vorticity_squared(field, spacing):
    # field is [3,x,y,z], with u/v/w aligned to x/y/z.
    omega_x = centered_derivative(field[2], 1, spacing[1]) - centered_derivative(
        field[1], 2, spacing[2]
    )
    omega_y = centered_derivative(field[0], 2, spacing[2]) - centered_derivative(
        field[2], 0, spacing[0]
    )
    omega_z = centered_derivative(field[1], 0, spacing[0]) - centered_derivative(
        field[0], 1, spacing[1]
    )
    return omega_x.square() + omega_y.square() + omega_z.square()


def histogram_update(counts, values, low, high):
    inside = (values >= low) & (values <= high)
    counts += torch.histc(values, bins=counts.numel(), min=low, max=high).double()
    return int(values.numel()), int(inside.sum())


def spectrum_for_field(field):
    spectrum, counts = isotropic_energy_spectrum(field.unsqueeze(0))
    return spectrum.astype(np.float64), counts.astype(np.float64)


def load_generated(path, device):
    array = np.load(path, mmap_mode="r")
    if array.shape != (1, 3, 384, 384, 384):
        raise ValueError(f"expected [1,3,384,384,384], got {array.shape} from {path}")
    return torch.from_numpy(np.asarray(array[0])).to(device=device, dtype=torch.float32)


def accumulate_pdf_metrics(
    field,
    increment_counts,
    vorticity_counts,
    u_rms_reference,
    omega_rms_reference,
    increment_range,
    vorticity_range,
    spacing,
):
    increment_total = increment_inside = 0
    for component, axis in zip(range(3), range(3)):
        values = (
            torch.roll(field[component], -1, dims=axis) - field[component]
        ) / u_rms_reference
        total, inside = histogram_update(
            increment_counts, values, -increment_range, increment_range
        )
        increment_total += total
        increment_inside += inside
    omega = vorticity_squared(field, spacing).sqrt() / omega_rms_reference
    vorticity_total, vorticity_inside = histogram_update(
        vorticity_counts, omega, 0.0, vorticity_range
    )
    return {
        "increment_total": increment_total,
        "increment_inside": increment_inside,
        "vorticity_total": vorticity_total,
        "vorticity_inside": vorticity_inside,
    }


def density(counts, low, high):
    counts = counts.cpu().numpy()
    width = (high - low) / len(counts)
    centers = low + (np.arange(len(counts)) + 0.5) * width
    denominator = max(counts.sum() * width, np.finfo(float).eps)
    return centers, counts / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fixed-field", type=Path, required=True)
    parser.add_argument("--shifted-field", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-index-start", type=int, default=0)
    parser.add_argument("--reference-index-count", type=int, default=16)
    parser.add_argument("--spacing", type=float, nargs=3, required=True)
    parser.add_argument("--pdf-bins", type=int, default=201)
    parser.add_argument("--increment-range", type=float, default=8.0)
    parser.add_argument("--vorticity-range", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    spacing = tuple(args.spacing)
    dataset = FluidVolumeDataset(args.data_root)
    indices = range(
        args.reference_index_start,
        args.reference_index_start + args.reference_index_count,
    )
    if indices.start < 0 or indices.stop > len(dataset):
        raise ValueError("reference index range is outside the dataset")

    # First reference pass: fixed GT normalizers and mean spectra.
    velocity_square_sum = 0.0
    velocity_value_count = 0
    omega_square_sum = 0.0
    omega_voxel_count = 0
    reference_spectrum = None
    mode_counts = None
    manifest = []
    for position, index in enumerate(indices, start=1):
        item = dataset[index]
        field = item["volume"].to(device)
        fluctuations = field - field.mean(dim=(1, 2, 3), keepdim=True)
        velocity_square_sum += float(fluctuations.square().sum().cpu())
        velocity_value_count += fluctuations.numel()
        omega_square = vorticity_squared(field, spacing)
        omega_square_sum += float(omega_square.sum().cpu())
        omega_voxel_count += omega_square.numel()
        spectrum, counts = spectrum_for_field(field)
        reference_spectrum = spectrum if reference_spectrum is None else reference_spectrum + spectrum
        mode_counts = counts
        manifest.append({
            "dataset_index": index,
            "filename": item["filename"],
            "timestep": int(item["timestep"]),
        })
        print(f"Reference normalizers [{position}/{len(indices)}]")
    u_rms_reference = (velocity_square_sum / velocity_value_count) ** 0.5
    omega_rms_reference = (omega_square_sum / omega_voxel_count) ** 0.5
    reference_spectrum /= len(indices)

    modes = {
        "reference": {
            "increment": torch.zeros(args.pdf_bins, dtype=torch.float64, device=device),
            "vorticity": torch.zeros(args.pdf_bins, dtype=torch.float64, device=device),
            "coverage": {"increment_total": 0, "increment_inside": 0, "vorticity_total": 0, "vorticity_inside": 0},
            "spectrum": reference_spectrum,
        }
    }
    # Second reference pass: PDFs with reference-defined normalization.
    for position, index in enumerate(indices, start=1):
        field = dataset[index]["volume"].to(device)
        coverage = accumulate_pdf_metrics(
            field,
            modes["reference"]["increment"],
            modes["reference"]["vorticity"],
            u_rms_reference,
            omega_rms_reference,
            args.increment_range,
            args.vorticity_range,
            spacing,
        )
        for key, value in coverage.items():
            modes["reference"]["coverage"][key] += value
        print(f"Reference PDFs [{position}/{len(indices)}]")

    for name, path in (("fixed", args.fixed_field), ("shifted", args.shifted_field)):
        field = load_generated(path, device)
        increment_counts = torch.zeros(args.pdf_bins, dtype=torch.float64, device=device)
        vorticity_counts = torch.zeros(args.pdf_bins, dtype=torch.float64, device=device)
        coverage = accumulate_pdf_metrics(
            field,
            increment_counts,
            vorticity_counts,
            u_rms_reference,
            omega_rms_reference,
            args.increment_range,
            args.vorticity_range,
            spacing,
        )
        spectrum, generated_counts = spectrum_for_field(field)
        if not np.array_equal(generated_counts, mode_counts):
            raise ValueError("spectrum shell counts do not match")
        modes[name] = {
            "increment": increment_counts,
            "vorticity": vorticity_counts,
            "coverage": coverage,
            "spectrum": spectrum,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    increment_centers = vorticity_centers = None
    increment_densities = {}
    vorticity_densities = {}
    for name, values in modes.items():
        increment_centers, increment_densities[name] = density(
            values["increment"], -args.increment_range, args.increment_range
        )
        vorticity_centers, vorticity_densities[name] = density(
            values["vorticity"], 0, args.vorticity_range
        )

    with (args.output_dir / "longitudinal_increment_pdf.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("normalized_increment", "reference", "fixed", "shifted"))
        for index, center in enumerate(increment_centers):
            writer.writerow((center, *[increment_densities[name][index] for name in ("reference", "fixed", "shifted")]))
    with (args.output_dir / "vorticity_magnitude_pdf.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("normalized_vorticity", "reference", "fixed", "shifted"))
        for index, center in enumerate(vorticity_centers):
            writer.writerow((center, *[vorticity_densities[name][index] for name in ("reference", "fixed", "shifted")]))

    spectrum_length = min(len(values["spectrum"]) for values in modes.values())
    domain_length = spacing[0] * 384
    fundamental_wavenumber = 2 * np.pi / domain_length
    with (args.output_dir / "hit_energy_spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "shell", "wavenumber", "mode_count",
            "reference_integrated", "fixed_integrated", "shifted_integrated",
            "reference_mode_mean", "fixed_mode_mean", "shifted_mode_mean",
        ))
        for shell in range(spectrum_length):
            integrated = [modes[name]["spectrum"][shell] for name in ("reference", "fixed", "shifted")]
            count = mode_counts[shell]
            writer.writerow((
                shell, shell * fundamental_wavenumber, count, *integrated,
                *[value / count if count else 0 for value in integrated],
            ))

    visual_output = "matplotlib unavailable"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for name in ("reference", "fixed", "shifted"):
            spectrum = modes[name]["spectrum"][:spectrum_length]
            wavenumbers = np.arange(spectrum_length) * fundamental_wavenumber
            valid = (wavenumbers > 0) & (spectrum > 0)
            axes[0].loglog(wavenumbers[valid], spectrum[valid], label=name)
            axes[1].semilogy(increment_centers, np.maximum(increment_densities[name], 1e-16), label=name)
            axes[2].semilogy(vorticity_centers, np.maximum(vorticity_densities[name], 1e-16), label=name)
        axes[0].set(xlabel="wavenumber k", ylabel="shell-integrated E(k)", title="Kinetic-energy spectrum")
        axes[1].set(xlabel="longitudinal increment / GT u_rms", ylabel="PDF", title="One-grid velocity increments")
        axes[2].set(xlabel="|vorticity| / GT omega_rms", ylabel="PDF", title="Vorticity magnitude")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
        figure.savefig(args.output_dir / "hit_metrics.png", dpi=180)
        plt.close(figure)
        visual_output = "hit_metrics.png"
    except ImportError:
        pass

    coverage_report = {}
    for name, values in modes.items():
        coverage = values["coverage"]
        coverage_report[name] = {
            "increment_histogram_coverage": coverage["increment_inside"] / coverage["increment_total"],
            "vorticity_histogram_coverage": coverage["vorticity_inside"] / coverage["vorticity_total"],
        }
    report = {
        "status": "lab_aligned_hit_metrics_provisional_derivative_convention",
        "reference_manifest": manifest,
        "reference_u_rms_component_pooled": u_rms_reference,
        "reference_omega_rms": omega_rms_reference,
        "spacing": list(spacing),
        "increment_definition": "one-grid longitudinal increments pooled over x/u, y/v, z/w with periodic wrapping",
        "increment_normalization": "reference component-pooled u_rms",
        "vorticity_definition": "periodic second-order centered finite-difference curl magnitude",
        "vorticity_normalization": "reference omega_rms",
        "spectrum_outputs": "both shell-integrated energy and mean energy per shell mode",
        "histogram_coverage": coverage_report,
        "outputs": [
            "hit_energy_spectra.csv",
            "longitudinal_increment_pdf.csv",
            "vorticity_magnitude_pdf.csv",
            visual_output,
        ],
        "caveat": "Confirm shell integration/averaging and finite-difference versus spectral curl with the lab standard.",
    }
    (args.output_dir / "hit_metrics_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
