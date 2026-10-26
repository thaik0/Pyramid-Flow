"""PSNR/SSIM for genuinely paired 3-D velocity fields.

These image-space metrics are not distributional metrics and must not be used
to match an unconditional sample to an arbitrary held-out turbulence state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def load_field(path: Path, key: str = "physical") -> torch.Tensor:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if key not in payload:
            raise KeyError(f"{path} has no array named {key!r}")
        payload = payload[key]
    field = torch.from_numpy(np.asarray(payload, dtype=np.float32))
    if field.ndim == 4:
        field = field.unsqueeze(0)
    if field.ndim != 5 or field.shape[1] != 3:
        raise ValueError("expected [N,3,D,H,W] or [3,D,H,W]")
    return field


def psnr_3d(prediction: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = torch.mean((prediction - target).square())
    if mse == 0:
        return float("inf")
    return float((10 * torch.log10(torch.as_tensor(data_range ** 2, device=mse.device) / mse)).cpu())


def ssim_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    window_size: int = 7,
) -> float:
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")
    padding = window_size // 2
    mean_x = functional.avg_pool3d(prediction, window_size, stride=1, padding=padding)
    mean_y = functional.avg_pool3d(target, window_size, stride=1, padding=padding)
    variance_x = functional.avg_pool3d(prediction.square(), window_size, 1, padding) - mean_x.square()
    variance_y = functional.avg_pool3d(target.square(), window_size, 1, padding) - mean_y.square()
    covariance = functional.avg_pool3d(prediction * target, window_size, 1, padding) - mean_x * mean_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return float(score.mean().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-key", default="physical")
    parser.add_argument("--target-key", default="physical")
    parser.add_argument("--data-range", type=float)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confirm-paired", action="store_true")
    args = parser.parse_args()
    if not args.confirm_paired:
        raise ValueError(
            "PSNR/SSIM require voxel-aligned paired fields. Pass --confirm-paired only "
            "when prediction and target represent the same physical state."
        )
    prediction = load_field(args.prediction, args.prediction_key).to(args.device)
    target = load_field(args.target, args.target_key).to(args.device)
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} versus {target.shape}")
    data_range = args.data_range
    if data_range is None:
        data_range = float((target.max() - target.min()).cpu())
    if data_range <= 0:
        raise ValueError("data range must be positive")
    per_component = {}
    for index, name in enumerate("uvw"):
        selected_prediction = prediction[:, index:index + 1]
        selected_target = target[:, index:index + 1]
        per_component[name] = {
            "psnr": psnr_3d(selected_prediction, selected_target, data_range),
            "ssim": ssim_3d(selected_prediction, selected_target, data_range, args.window_size),
        }
    payload = {
        "status": "paired_voxel_aligned_similarity",
        "data_range": data_range,
        "window_size": args.window_size,
        "aggregate": {
            "psnr": psnr_3d(prediction, target, data_range),
            "ssim": ssim_3d(prediction, target, data_range, args.window_size),
        },
        "per_component": per_component,
        "caveat": "PSNR/SSIM measure paired spatial reconstruction, not generative distribution quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
