"""Bounded trainer for the provisional 64^3 fluid-volume baseline."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402
from pyramid_dit.fluid_pyramid_flow import FluidPyramidFlow  # noqa: E402


class CachedDataset(Dataset):
    """Materialize a bounded crop dataset once to avoid repeated NetCDF reads."""

    def __init__(self, source):
        started = time.perf_counter()
        self.items = []
        for index in range(len(source)):
            item = dict(source[index])
            item["volume"] = item["volume"].clone()
            self.items.append(item)
            print(f"Cached training crop {index + 1}/{len(source)}")
        self.elapsed_seconds = time.perf_counter() - started

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path)
    parser.add_argument("--pattern", default="fluid_*.nc")
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-layers", type=int, default=8)
    parser.add_argument("--model-heads", type=int, default=8)
    parser.add_argument("--fixed-indices", default="")
    parser.add_argument("--index-start", type=int)
    parser.add_argument("--index-count", type=int)
    parser.add_argument("--stage-mode", choices=("sample", "cycle"), default="sample")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    base_dataset = FluidVolumeDataset(
        args.data_root, pattern=args.pattern,
        normalization_json=args.normalization_json,
        crop_size=args.crop_size, crop_seed=args.seed,
    )
    fixed_indices = [int(value) for value in args.fixed_indices.split(",") if value.strip()]
    if args.index_start is not None or args.index_count is not None:
        if fixed_indices or args.index_start is None or args.index_count is None:
            raise ValueError("Use either --fixed-indices or both --index-start/--index-count")
        fixed_indices = list(range(args.index_start, args.index_start + args.index_count))
    if fixed_indices:
        if min(fixed_indices) < 0 or max(fixed_indices) >= len(base_dataset):
            raise ValueError("fixed index is outside the dataset")
        dataset = Subset(base_dataset, fixed_indices)
    else:
        dataset = base_dataset
    cache_seconds = 0.0
    if args.cache_dataset:
        dataset = CachedDataset(dataset)
        cache_seconds = dataset.elapsed_seconds
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=loader_generator,
    )
    device = torch.device(args.device)
    model = FluidPyramidFlow(
        patch_size=args.patch_size,
        num_layers=args.model_layers,
        num_attention_heads=args.model_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint["config"]
        for name in ("crop_size", "patch_size", "model_layers", "model_heads"):
            if checkpoint_config[name] != getattr(args, name):
                raise ValueError(
                    f"resume mismatch for {name}: checkpoint={checkpoint_config[name]}, "
                    f"requested={getattr(args, name)}"
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["steps"])
        if start_step >= args.steps:
            raise ValueError(
                f"checkpoint is already at step {start_step}; --steps must be larger"
            )
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if "loader_rng_state" in checkpoint:
            loader_generator.set_state(checkpoint["loader_rng_state"])

    saved_config = vars(args).copy()
    saved_config["data_root"] = str(args.data_root.resolve())
    saved_config["output_dir"] = str(args.output_dir.resolve())
    saved_config["normalization_json"] = str(args.normalization_json.resolve()) if args.normalization_json else None
    saved_config["resume"] = str(args.resume.resolve()) if args.resume else None
    saved_config["dataset_cache_seconds"] = cache_seconds
    (args.output_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
    manifest = []
    manifest_indices = fixed_indices or list(range(len(base_dataset)))
    if not fixed_indices:
        manifest_indices = manifest_indices[: min(32, len(manifest_indices))]
    for index in manifest_indices:
        sample = base_dataset.samples[index]
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        origin = [int(rng.integers(0, 384 - args.crop_size + 1)) for _ in range(3)]
        manifest.append({
            "dataset_index": index, "filename": sample.path.name,
            "timestep": sample.timestep, "crop_origin": origin,
        })
    (args.output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    iterator = iter(loader)
    log_path = args.output_dir / "metrics.jsonl"
    run_start = time.perf_counter()
    for step in range(start_step, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        volume = batch["volume"].to(device)
        step_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        forced_stage = step % len(model.stages) if args.stage_mode == "cycle" else None
        loss, details = model.training_loss(volume, stage=forced_stage)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "step": step + 1, "loss": float(loss.detach()), "stage": details["stage"],
            "filename": list(batch["filename"]),
            "timestep": batch["timestep"].tolist(),
            "crop_origin": [part.tolist() for part in batch["crop_origin"]],
            "seed": args.seed,
            "step_seconds": time.perf_counter() - step_start,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        }
        with log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(record)

        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "steps": step + 1, "seed": args.seed, "config": saved_config,
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "loader_rng_state": loader_generator.get_state()},
                args.output_dir / f"checkpoint-{step + 1}.pt",
            )

    torch.save(
        {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "steps": args.steps, "seed": args.seed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_rng_state": loader_generator.get_state(),
            "config": saved_config,
            "elapsed_seconds": time.perf_counter() - run_start,
        },
        args.output_dir / f"checkpoint-{args.steps}.pt",
    )


if __name__ == "__main__":
    main()
