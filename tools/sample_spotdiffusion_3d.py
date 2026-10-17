"""Run stage-consistent global 3-D inference from a fluid PFM checkpoint."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pyramid_dit.fluid_pyramid_flow import FluidPyramidFlow  # noqa: E402
from pyramid_dit.spotdiffusion_sampler_3d import (  # noqa: E402
    sample_global_spotdiffusion_3d,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--global-size", type=int, default=384)
    parser.add_argument("--steps-per-stage", type=int, default=1)
    parser.add_argument("--patch-batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("fixed", "shifted"), default="shifted")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--shift-seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-volume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = FluidPyramidFlow(
        patch_size=config["patch_size"],
        num_layers=config["model_layers"],
        num_attention_heads=config["model_heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []

    def record_step(step, volume):
        record = {
            "stage": step.stage,
            "step": step.step,
            "timestep": step.timestep,
            "global_shape": list(step.global_shape),
            "window_size": step.window_size,
            "shift": list(step.shift),
            "mean": float(volume.mean()),
            "std": float(volume.std()),
            "finite": bool(torch.isfinite(volume).all()),
        }
        records.append(record)
        print(record, flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    normalized = sample_global_spotdiffusion_3d(
        model=model,
        batch_size=1,
        global_size=args.global_size,
        steps=[args.steps_per_stage] * len(model.stages),
        device=device,
        dtype=torch.float32,
        generator=generator,
        patch_batch_size=args.patch_batch_size,
        shifted_windows=args.mode == "shifted",
        shift_seed=args.shift_seed,
        step_callback=record_step,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    stats = json.loads(args.normalization_json.read_text())
    mean = torch.tensor(stats["mean"], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 3, 1, 1, 1)
    physical = normalized * std + mean
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_volume:
        np.savez_compressed(
            args.output_dir / "generated_fields.npz",
            normalized=normalized.cpu().numpy(),
            physical=physical.cpu().numpy(),
        )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_steps": checkpoint["steps"],
        "training_seed": checkpoint["seed"],
        "sampling_seed": args.seed,
        "shift_seed": args.shift_seed,
        "mode": args.mode,
        "global_size": args.global_size,
        "steps_per_stage": args.steps_per_stage,
        "patch_batch_size": args.patch_batch_size,
        "shape": list(physical.shape),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "physical_mean": [float(value) for value in physical.mean((0, 2, 3, 4))],
        "physical_std": [float(value) for value in physical.std((0, 2, 3, 4))],
        "steps": records,
        "volume_saved": args.save_volume,
    }
    (args.output_dir / "generation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
