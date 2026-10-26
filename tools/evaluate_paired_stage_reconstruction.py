"""Paired final-stage reconstruction metrics for the 3-D fluid model.

This is a teacher-forced diagnostic: the stage-2 starting state uses the true
lower-resolution context from the paired target. It measures conditional
denoising/reconstruction, not unconditional generative fidelity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402
from pyramid_dit.fluid_pyramid_flow import FluidPyramidFlow  # noqa: E402
from tools.evaluate_paired_fluid_similarity import psnr_3d, ssim_3d  # noqa: E402


def reconstruct(model, state, stage, initial_ratio, updates):
    ratios = torch.linspace(
        initial_ratio, 0.0, updates + 1, device=state.device, dtype=state.dtype
    )
    stage_timesteps = model.scheduler.timesteps_per_stage[stage]
    timestep_max = float(stage_timesteps[0])
    timestep_min = float(stage_timesteps[-1])
    result = state
    for current, following in zip(ratios[:-1], ratios[1:]):
        timestep = timestep_min + (timestep_max - timestep_min) * current
        velocity = model.predict_velocity(result, timestep)
        result = result + (following - current) * velocity
    return result


def component_similarity(prediction, target, ranges, window_size):
    values = {}
    for channel, name in enumerate("uvw"):
        predicted = prediction[:, channel:channel + 1]
        expected = target[:, channel:channel + 1]
        values[name] = {
            "psnr": psnr_3d(predicted, expected, ranges[channel]),
            "ssim": ssim_3d(predicted, expected, ranges[channel], window_size),
            "rmse": float(torch.mean((predicted - expected).square()).sqrt().cpu()),
        }
    values["mean_psnr"] = float(np.mean([values[name]["psnr"] for name in "uvw"]))
    values["mean_ssim"] = float(np.mean([values[name]["ssim"] for name in "uvw"]))
    values["mean_rmse"] = float(np.mean([values[name]["rmse"] for name in "uvw"]))
    return values


def aggregate(rows, ratios):
    result = {}
    for ratio in ratios:
        selected = [row for row in rows if row["noise_ratio"] == ratio]
        ratio_result = {"sample_count": len(selected)}
        for phase in ("input", "reconstruction"):
            phase_result = {}
            for metric in ("mean_psnr", "mean_ssim", "mean_rmse"):
                values = np.asarray([row[phase][metric] for row in selected])
                phase_result[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            ratio_result[phase] = phase_result
        ratio_result["improvement"] = {
            "psnr_db": (
                ratio_result["reconstruction"]["mean_psnr"]["mean"]
                - ratio_result["input"]["mean_psnr"]["mean"]
            ),
            "ssim": (
                ratio_result["reconstruction"]["mean_ssim"]["mean"]
                - ratio_result["input"]["mean_ssim"]["mean"]
            ),
            "rmse_percent": 100 * (
                ratio_result["reconstruction"]["mean_rmse"]["mean"]
                / ratio_result["input"]["mean_rmse"]["mean"] - 1
            ),
        }
        result[str(ratio)] = ratio_result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-count", type=int, default=16)
    parser.add_argument("--crop-seed", type=int, default=20261026)
    parser.add_argument("--noise-seed", type=int, default=314159)
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    stats = json.loads(args.normalization_json.read_text())
    if "min" not in stats or "max" not in stats:
        raise ValueError("normalization JSON must include fixed training min/max values")
    ranges = [high - low for low, high in zip(stats["min"], stats["max"])]
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FluidPyramidFlow(
        patch_size=config["patch_size"],
        num_layers=config["model_layers"],
        num_attention_heads=config["model_heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = FluidVolumeDataset(
        args.data_root,
        normalization_json=args.normalization_json,
        crop_size=config["crop_size"],
        crop_seed=args.crop_seed,
    )
    indices = range(args.index_start, args.index_start + args.index_count)
    if indices.start < 0 or indices.stop > len(dataset):
        raise ValueError("held-out index range is outside the dataset")

    mean = torch.tensor(stats["mean"], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 3, 1, 1, 1)
    stage = len(model.stages) - 1
    start_sigma = model.scheduler.start_sigmas[stage]
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for position, index in enumerate(indices, start=1):
            item = dataset[index]
            target = item["volume"].unsqueeze(0).to(device)
            levels = model.pyramid(target)
            coarse = functional.interpolate(
                levels[stage - 1], size=target.shape[-3:], mode="nearest"
            )
            generator = torch.Generator(device=device).manual_seed(args.noise_seed + index)
            noise = torch.randn(target.shape, generator=generator, device=device)
            start = start_sigma * noise + (1 - start_sigma) * coarse
            physical_target = target * std + mean
            for ratio in args.noise_ratios:
                state = ratio * start + (1 - ratio) * target
                prediction = reconstruct(model, state, stage, ratio, args.updates)
                physical_state = state * std + mean
                physical_prediction = prediction * std + mean
                rows.append({
                    "dataset_index": index,
                    "filename": item["filename"],
                    "timestep": int(item["timestep"]),
                    "crop_origin": list(item["crop_origin"]),
                    "noise_ratio": ratio,
                    "input": component_similarity(
                        physical_state, physical_target, ranges, args.window_size
                    ),
                    "reconstruction": component_similarity(
                        physical_prediction, physical_target, ranges, args.window_size
                    ),
                })
            print(f"[{position}/{args.index_count}] {item['filename']} t={item['timestep']}")

    report = {
        "status": "paired_teacher_forced_final_stage_reconstruction",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_steps": int(checkpoint["steps"]),
        "heldout_index_range": [indices.start, indices.stop - 1],
        "sample_count": args.index_count,
        "crop_size": config["crop_size"],
        "crop_seed": args.crop_seed,
        "noise_seed": args.noise_seed,
        "noise_ratios": args.noise_ratios,
        "updates_per_reconstruction": args.updates,
        "psnr_data_ranges_from_training_extrema": dict(zip("uvw", ranges)),
        "summary": aggregate(rows, args.noise_ratios),
        "caveat": (
            "The true lower-resolution target supplies the final-stage context. "
            "These PSNR/SSIM values measure paired conditional reconstruction and "
            "must not be reported as unconditional generation metrics."
        ),
    }
    (args.output_dir / "paired_reconstruction_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "paired_reconstruction_samples.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with (args.output_dir / "paired_reconstruction_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("noise_ratio", "phase", "mean_psnr", "mean_ssim", "mean_rmse"))
        for ratio, values in report["summary"].items():
            for phase in ("input", "reconstruction"):
                writer.writerow((
                    ratio,
                    phase,
                    values[phase]["mean_psnr"]["mean"],
                    values[phase]["mean_ssim"]["mean"],
                    values[phase]["mean_rmse"]["mean"],
                ))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
