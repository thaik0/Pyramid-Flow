"""Summarize stage-wise learning and throughput from a fluid training log."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def window(values, fraction, first):
    count = max(1, int(np.ceil(len(values) * fraction)))
    return values[:count] if first else values[-count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.window_fraction <= 0.5:
        raise ValueError("window fraction must be in (0, 0.5]")

    records = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("training log is empty")
    by_stage = defaultdict(list)
    for record in records:
        by_stage[int(record["stage"])].append(record)

    stages = {}
    for stage, stage_records in sorted(by_stage.items()):
        losses = np.asarray([record["loss"] for record in stage_records], dtype=np.float64)
        early = window(losses, args.window_fraction, True)
        late = window(losses, args.window_fraction, False)
        stages[str(stage)] = {
            "updates": len(losses),
            "early_loss_mean": float(early.mean()),
            "late_loss_mean": float(late.mean()),
            "loss_change_percent": float(100 * (late.mean() - early.mean()) / early.mean()),
            "late_loss_std": float(late.std()),
        }

    seconds = np.asarray([record["step_seconds"] for record in records], dtype=np.float64)
    payload = {
        "status": "training_log_summary",
        "steps": len(records),
        "stage_metrics": stages,
        "step_seconds": {
            "median": float(np.median(seconds)),
            "mean": float(seconds.mean()),
            "p95": float(np.percentile(seconds, 95)),
        },
        "interpretation_rule": (
            "Strongly negative late-versus-early loss changes support longer/capacity-scaled training. "
            "Flat low training loss with poor samples points instead to generalization, exposure bias, "
            "or objective/model mismatch."
        ),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
