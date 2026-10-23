"""Compare three patch schedules with a full-resolution simulation reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_spotdiffusion_ablation import evaluate_field


MODES = ("fixed", "shifted", "shifted_early", "reference")
GENERATED_MODES = MODES[:-1]


def relative_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if value != 0 else 0.0
    return (value - baseline) / abs(baseline)


def spectrum_distance(spectrum: np.ndarray, reference: np.ndarray) -> float:
    length = min(len(spectrum), len(reference))
    denominator = max(
        float(np.abs(reference[:length]).sum()), np.finfo(float).eps
    )
    return float(np.abs(spectrum[:length] - reference[:length]).sum() / denominator)


def field_summary(metrics: dict, spectrum: np.ndarray, reference_metrics: dict, reference_spectrum: np.ndarray) -> dict:
    component_std_error = {
        component: relative_change(
            values["std"],
            reference_metrics["component_statistics"][component]["std"],
        )
        for component, values in metrics["component_statistics"].items()
    }
    return {
        "kinetic_energy_relative_error": relative_change(
            metrics["mean_kinetic_energy"],
            reference_metrics["mean_kinetic_energy"],
        ),
        "divergence_relative_error": relative_change(
            metrics["divergence_rms_periodic"],
            reference_metrics["divergence_rms_periodic"],
        ),
        "component_std_relative_error": component_std_error,
        "mean_absolute_component_std_relative_error": float(
            np.mean(np.abs(list(component_std_error.values())))
        ),
        "energy_spectrum_relative_l1_to_reference": spectrum_distance(
            spectrum, reference_spectrum
        ),
        "phase_invariant_seam_score": metrics["boundary_artifacts"]["aggregate"][
            "phase_invariant_seam_score"
        ],
    }


def write_visuals(slices: dict[str, np.ndarray], spectra: dict[str, np.ndarray], output_dir: Path) -> str:
    np.savez_compressed(output_dir / "central_slices.npz", **slices)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "central_slices.npz (matplotlib unavailable; PNGs not created)"

    labels = ("speed", "u", "v", "w")
    figure, axes = plt.subplots(4, 4, figsize=(16, 15), constrained_layout=True)
    for column, label in enumerate(labels):
        combined = np.concatenate([slices[mode][column].ravel() for mode in MODES])
        low, high = np.percentile(combined, (1, 99))
        if label != "speed":
            limit = max(abs(low), abs(high))
            low, high = -limit, limit
        for row, mode in enumerate(MODES):
            image = axes[row, column].imshow(
                slices[mode][column],
                origin="lower",
                cmap="viridis" if label == "speed" else "coolwarm",
                vmin=low,
                vmax=high,
            )
            axes[row, column].set_title(f"{mode}: {label}")
            axes[row, column].set_axis_off()
            figure.colorbar(image, ax=axes[row, column], shrink=0.7)
    figure.savefig(output_dir / "schedule_central_slices.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for mode in MODES:
        values = spectra[mode]
        shells = np.arange(len(values))
        positive = (shells > 0) & (values > 0)
        axis.loglog(shells[positive], values[positive], label=mode)
    axis.set_xlabel("Wavenumber shell")
    axis.set_ylabel("Kinetic energy")
    axis.set_title("Isotropic energy spectra")
    axis.legend()
    axis.grid(True, which="both", alpha=0.25)
    figure.savefig(output_dir / "energy_spectra.png", dpi=160)
    plt.close(figure)
    return "schedule_central_slices.png and energy_spectra.png"


def main():
    parser = argparse.ArgumentParser()
    for mode in MODES:
        parser.add_argument(f"--{mode.replace('_', '-')}-field", type=Path, required=True)
    for mode in GENERATED_MODES:
        parser.add_argument(
            f"--{mode.replace('_', '-')}-metadata", type=Path, required=True
        )
    parser.add_argument("--reference-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    spacing = tuple(args.spacing)

    field_paths = {
        "fixed": args.fixed_field,
        "shifted": args.shifted_field,
        "shifted_early": args.shifted_early_field,
        "reference": args.reference_field,
    }
    metrics = {}
    spectra = {}
    counts = {}
    slices = {}
    for mode in MODES:
        metrics[mode], spectra[mode], counts[mode], slices[mode] = evaluate_field(
            field_paths[mode], device, spacing, args.window_size
        )

    metadata = {
        "fixed": json.loads(args.fixed_metadata.read_text()),
        "shifted": json.loads(args.shifted_metadata.read_text()),
        "shifted_early": json.loads(args.shifted_early_metadata.read_text()),
    }
    controlled_keys = (
        "checkpoint_sha256",
        "sampling_seed",
        "global_size",
        "steps_per_stage",
        "patch_batch_size",
    )
    controls_match = {
        key: len({metadata[mode][key] for mode in GENERATED_MODES}) == 1
        for key in controlled_keys
    }
    expected_modes = {
        "fixed": "fixed",
        "shifted": "shifted",
        "shifted_early": "shifted-early",
    }
    if not all(controls_match.values()):
        raise ValueError(f"schedule controls do not match: {controls_match}")
    if any(metadata[mode]["mode"] != expected for mode, expected in expected_modes.items()):
        raise ValueError("generation metadata does not match schedule arms")

    summaries = {
        mode: field_summary(
            metrics[mode], spectra[mode], metrics["reference"], spectra["reference"]
        )
        for mode in GENERATED_MODES
    }
    fixed_seam = summaries["fixed"]["phase_invariant_seam_score"]
    for mode in GENERATED_MODES:
        summaries[mode]["seam_score_change_vs_fixed_percent"] = 100 * relative_change(
            summaries[mode]["phase_invariant_seam_score"], fixed_seam
        )

    visual_outputs = write_visuals(slices, spectra, args.output_dir)
    report = {
        "status": "single_seed_provisional_schedule_ablation_with_reference",
        "controls_match": controls_match,
        "reference": json.loads(args.reference_metadata.read_text()),
        "metrics": metrics,
        "comparison_to_reference": summaries,
        "runtime_seconds": {
            mode: metadata[mode]["elapsed_seconds"] for mode in GENERATED_MODES
        },
        "peak_memory_bytes": {
            mode: metadata[mode]["peak_memory_bytes"] for mode in GENERATED_MODES
        },
        "visual_outputs": visual_outputs,
        "spacing": list(spacing),
        "caveat": (
            "The reference is outside checkpoint training indices but is provisional "
            "until a trajectory-aware split and physical grid spacing are approved."
        ),
    }
    (args.output_dir / "schedule_ablation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    spectrum_length = min(len(spectra[mode]) for mode in MODES)
    with (args.output_dir / "energy_spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shell", *[f"{mode}_energy" for mode in MODES]))
        for shell in range(spectrum_length):
            writer.writerow((shell, *[float(spectra[mode][shell]) for mode in MODES]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
