"""Export one traceable full-resolution simulation field for evaluation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.fluid_volume_dataset import FluidVolumeDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--excluded-index-start", type=int, default=200)
    parser.add_argument("--excluded-index-end", type=int, default=327)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="fluid_*.nc")
    args = parser.parse_args()

    dataset = FluidVolumeDataset(args.data_root, pattern=args.pattern)
    if args.excluded_index_start <= args.dataset_index <= args.excluded_index_end:
        raise ValueError(
            "reference index overlaps the checkpoint training interval "
            f"[{args.excluded_index_start}, {args.excluded_index_end}]"
        )
    if args.dataset_index < 0 or args.dataset_index >= len(dataset):
        raise ValueError(
            f"dataset index {args.dataset_index} is outside [0, {len(dataset) - 1}]"
        )
    item = dataset[args.dataset_index]
    field = item["volume"].numpy()[None]
    if field.shape != (1, 3, 384, 384, 384):
        raise ValueError(f"expected reference shape [1,3,384,384,384], got {field.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "physical_fields.npy", field)
    metadata = {
        "status": "provisional_reference_pending_trajectory_split",
        "dataset_index": args.dataset_index,
        "filename": item["filename"],
        "timestep": int(item["timestep"]),
        "shape": list(field.shape),
        "channels": list(item["channels"]),
        "finite": bool(np.isfinite(field).all()),
        "note": (
            "Selected outside checkpoint training indices 200-327, but the dataset "
            "does not yet have an approved trajectory-aware held-out split."
        ),
    }
    (args.output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
