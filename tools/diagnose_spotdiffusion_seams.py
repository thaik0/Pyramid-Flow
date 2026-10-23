"""Locate when patch artifacts appear during fixed and shifted sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_spotdiffusion_ablation import boundary_artifacts, divergence_rms


MODES = ("fixed", "shifted")


def load_state_metrics(record: dict, device: torch.device) -> tuple[dict, np.ndarray]:
    array = np.load(record["state_path"], mmap_mode="r")
    field = torch.from_numpy(np.asarray(array)).to(device=device, dtype=torch.float32)
    seams = boundary_artifacts(field, record["window_size"])
    center = field.shape[-3] // 2
    speed_slice = field[0, :, center].square().sum(0).sqrt().cpu().numpy()
    result = {
        "event": record["event"],
        "stage": record["stage"],
        "step": record["step"],
        "timestep": record["timestep"],
        "global_shape": record["global_shape"],
        "window_size": record["window_size"],
        "shift": record["shift"],
        "phase_invariant_seam_score": seams["aggregate"][
            "phase_invariant_seam_score"
        ],
        "fixed_grid_boundary_ratio": seams["aggregate"][
            "boundary_to_interior_ratio"
        ],
        "phase_peak_by_axis": {
            axis: values["phase_scan"]["peak_phase"]
            for axis, values in seams["axes"].items()
        },
        "mean_kinetic_energy": float(0.5 * field.square().sum(1).mean()),
        "divergence_rms_periodic": divergence_rms(field, (1.0, 1.0, 1.0)),
        "mean_component_std": float(
            torch.stack([component.std(unbiased=False) for component in field[0]]).mean()
        ),
    }
    del field
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, speed_slice


def boundary_lines(size: int, window: int, shift: int) -> list[float]:
    first = (-shift) % window - 0.5
    lines = []
    value = first
    while value < size:
        if value > 0:
            lines.append(value)
        value += window
    return lines


def write_plots(
    timeline: dict[str, list[dict]],
    slices: dict[tuple[str, int, int], np.ndarray],
    output_dir: Path,
) -> str:
    np.savez_compressed(
        output_dir / "step_slices.npz",
        **{
            f"{mode}_stage_{stage}_step_{step}": values
            for (mode, stage, step), values in slices.items()
        },
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "step_slices.npz (matplotlib unavailable; PNGs not created)"

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for stage, axis in enumerate(axes):
        for mode in MODES:
            records = [item for item in timeline[mode] if item["stage"] == stage]
            x = [item["step"] for item in records]
            y = [item["phase_invariant_seam_score"] for item in records]
            axis.plot(x, y, marker="o", label=mode)
        axis.axvline(-0.5, color="gray", linewidth=0.8, linestyle="--")
        axis.set_xticks((-1, 0, 1, 2, 3), ("start", "1", "2", "3", "4"))
        axis.set_title(f"Stage {stage}: seam score")
        axis.set_xlabel("Saved state / flow step")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Phase-invariant seam score")
    axes[-1].legend()
    figure.savefig(output_dir / "seam_timeline.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(6, 4, figsize=(14, 19), constrained_layout=True)
    for stage in range(3):
        stage_values = [
            slices[(mode, stage, step)]
            for mode in MODES
            for step in range(4)
        ]
        low, high = np.percentile(np.concatenate([x.ravel() for x in stage_values]), (1, 99))
        for mode_index, mode in enumerate(MODES):
            row = stage * 2 + mode_index
            records = {
                item["step"]: item
                for item in timeline[mode]
                if item["stage"] == stage and item["event"] == "post_step"
            }
            for step in range(4):
                axis = axes[row, step]
                image = axis.imshow(
                    slices[(mode, stage, step)],
                    origin="lower",
                    cmap="viridis",
                    vmin=low,
                    vmax=high,
                )
                record = records[step]
                size = record["global_shape"][0]
                window = record["window_size"]
                shift_h, shift_w = record["shift"][1:]
                for value in boundary_lines(size, window, shift_w):
                    axis.axvline(value, color="white", linewidth=0.35, alpha=0.7)
                for value in boundary_lines(size, window, shift_h):
                    axis.axhline(value, color="white", linewidth=0.35, alpha=0.7)
                axis.set_title(f"{mode}, stage {stage}, step {step + 1}")
                axis.set_axis_off()
                figure.colorbar(image, ax=axis, shrink=0.65)
    figure.savefig(output_dir / "central_slice_sequence.png", dpi=150)
    plt.close(figure)
    return "seam_timeline.png and central_slice_sequence.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-metadata", type=Path, required=True)
    parser.add_argument("--shifted-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
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
        raise ValueError(f"trajectory controls do not match: {controls_match}")
    if not all(metadata[mode].get("step_states_saved") for mode in MODES):
        raise ValueError("both runs must include saved step states")

    timeline = {mode: [] for mode in MODES}
    slices = {}
    for mode in MODES:
        for record in metadata[mode]["steps"]:
            metrics, speed_slice = load_state_metrics(record, device)
            timeline[mode].append(metrics)
            if record["event"] == "post_step":
                slices[(mode, record["stage"], record["step"])] = speed_slice
            print({"mode": mode, **metrics}, flush=True)
    visual_outputs = write_plots(timeline, slices, args.output_dir)
    report = {
        "status": "single_seed_stepwise_seam_diagnostic",
        "controls_match": controls_match,
        "timeline": timeline,
        "visual_outputs": visual_outputs,
        "interpretation_guide": {
            "stage_start_jump": "resolution transition or injected stage noise",
            "post_step_jump": "local prediction/update at that flow step",
            "late_reintroduction": "last-step schedule effect",
            "slow_decline": "local-context limitation or too few flow steps",
        },
    }
    (args.output_dir / "seam_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
