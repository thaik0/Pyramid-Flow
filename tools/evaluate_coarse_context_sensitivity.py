"""Measure paired final-stage reconstruction sensitivity to coarse-context error."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402
from pyramid_dit.fluid_pyramid_flow import FluidPyramidFlow  # noqa: E402
from tools.evaluate_paired_stage_reconstruction import component_similarity, reconstruct  # noqa: E402


def summarize(rows, levels):
    result = {}
    for level in levels:
        selected = [row for row in rows if row["context_noise_std"] == level]
        result[str(level)] = {
            "sample_count": len(selected),
            "coarse_context_physical_rmse_mean": float(np.mean([row["coarse_context_physical_rmse"] for row in selected])),
            "psnr_mean": float(np.mean([row["reconstruction"]["mean_psnr"] for row in selected])),
            "psnr_std": float(np.std([row["reconstruction"]["mean_psnr"] for row in selected])),
            "ssim_mean": float(np.mean([row["reconstruction"]["mean_ssim"] for row in selected])),
            "ssim_std": float(np.std([row["reconstruction"]["mean_ssim"] for row in selected])),
            "rmse_mean": float(np.mean([row["reconstruction"]["mean_rmse"] for row in selected])),
        }
    baseline = result[str(levels[0])]
    for level in levels:
        current = result[str(level)]
        current["change_from_clean_context"] = {
            "psnr_db": current["psnr_mean"] - baseline["psnr_mean"],
            "ssim": current["ssim_mean"] - baseline["ssim_mean"],
            "rmse_percent": 100 * (current["rmse_mean"] / baseline["rmse_mean"] - 1),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-count", type=int, default=16)
    parser.add_argument("--context-noise-levels", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--crop-seed", type=int, default=20261026)
    parser.add_argument("--noise-seed", type=int, default=271828)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    stats = json.loads(args.normalization_json.read_text())
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
    mean = torch.tensor(stats["mean"], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 3, 1, 1, 1)
    stage = len(model.stages) - 1
    start_sigma = model.scheduler.start_sigmas[stage]
    rows = []
    with torch.no_grad():
        for position, index in enumerate(indices, start=1):
            item = dataset[index]
            target = item["volume"].unsqueeze(0).to(device)
            coarse = model.pyramid(target)[stage - 1]
            stage_generator = torch.Generator(device=device).manual_seed(args.noise_seed + index)
            stage_noise = torch.randn(target.shape, generator=stage_generator, device=device)
            context_generator = torch.Generator(device=device).manual_seed(args.noise_seed + 100000 + index)
            context_noise = torch.randn(coarse.shape, generator=context_generator, device=device)
            physical_target = target * std + mean
            clean_context = functional.interpolate(coarse, size=target.shape[-3:], mode="nearest")
            for level in args.context_noise_levels:
                perturbed_coarse = coarse + level * context_noise
                context = functional.interpolate(perturbed_coarse, size=target.shape[-3:], mode="nearest")
                state = start_sigma * stage_noise + (1 - start_sigma) * context
                prediction = reconstruct(model, state, stage, 1.0, args.updates)
                physical_prediction = prediction * std + mean
                physical_context_error = (context - clean_context) * std
                rows.append({
                    "dataset_index": index,
                    "filename": item["filename"],
                    "timestep": int(item["timestep"]),
                    "context_noise_std": level,
                    "coarse_context_physical_rmse": float(physical_context_error.square().mean().sqrt().cpu()),
                    "reconstruction": component_similarity(
                        physical_prediction, physical_target, ranges, args.window_size
                    ),
                })
            print(f"[{position}/{args.index_count}] {item['filename']} t={item['timestep']}")

    report = {
        "status": "paired_final_stage_coarse_context_sensitivity",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_steps": int(checkpoint["steps"]),
        "heldout_index_range": [indices.start, indices.stop - 1],
        "context_noise_levels_normalized_units": args.context_noise_levels,
        "summary": summarize(rows, args.context_noise_levels),
        "caveat": (
            "This controlled Gaussian context corruption measures sensitivity, but does not yet "
            "replicate the structured errors produced by the lower pyramid stages."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coarse_context_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "coarse_context_sensitivity_samples.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
