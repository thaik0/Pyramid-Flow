"""Generate traceable velocity fields from a fluid Pyramid Flow checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pyramid_dit.fluid_pyramid_flow import FluidPyramidFlow  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--steps-per-stage", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = FluidPyramidFlow(
        patch_size=config["patch_size"], num_layers=config["model_layers"],
        num_attention_heads=config["model_heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    normalized = model.sample(
        args.num_samples, config["crop_size"],
        [args.steps_per_stage] * len(model.stages), device, torch.float32, generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    stats = json.loads(args.normalization_json.read_text())
    mean = torch.tensor(stats["mean"], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(1, 3, 1, 1, 1)
    physical = normalized * std + mean
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "generated_fields.npz",
        normalized=normalized.cpu().numpy(), physical=physical.cpu().numpy(),
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "training_seed": checkpoint["seed"],
        "sampling_seed": args.seed, "steps_per_stage": args.steps_per_stage,
        "num_samples": args.num_samples, "shape": list(physical.shape),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    (args.output_dir / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
