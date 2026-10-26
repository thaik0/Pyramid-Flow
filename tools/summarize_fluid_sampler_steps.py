"""Summarize a matched sampler-step ablation against held-out ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE = {
    "kinetic_energy": {"mean": 2.726488396525383, "min": 2.692843437194824, "max": 2.748013734817505},
    "divergence_rms": {"mean_index_units": 0.03722295700572431},
    "component_std": {"u": 1.201975055038929, "v": 1.3703940659761429, "w": 1.457924336194992},
}


def relative_error(value, reference):
    return (value - reference) / abs(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--spacing", type=float, default=0.016362462547794956)
    args = parser.parse_args()

    reference_divergence = REFERENCE["divergence_rms"]["mean_index_units"] / args.spacing
    results = {}
    for steps in args.steps:
        path = args.run_dir / f"steps-{steps}" / "evaluation" / "metrics.json"
        metrics = json.loads(path.read_text())
        component_std = {
            name: metrics["component_statistics"][name]["std"] for name in "uvw"
        }
        results[str(steps)] = {
            "metrics_path": str(path),
            "kinetic_energy": metrics["mean_kinetic_energy"],
            "kinetic_energy_relative_error": relative_error(
                metrics["mean_kinetic_energy"], REFERENCE["kinetic_energy"]["mean"]
            ),
            "component_std": component_std,
            "component_std_relative_error": {
                name: relative_error(value, REFERENCE["component_std"][name])
                for name, value in component_std.items()
            },
            "divergence_rms": metrics["divergence_rms_periodic_crop_approximation"],
            "divergence_relative_error": relative_error(
                metrics["divergence_rms_periodic_crop_approximation"], reference_divergence
            ),
        }
    payload = {
        "status": "matched_sampler_step_ablation",
        "reference": {
            **REFERENCE,
            "divergence_rms_physical_spacing": reference_divergence,
            "spacing": args.spacing,
        },
        "results": results,
        "interpretation_rule": (
            "If velocity variance and kinetic energy approach the reference as steps increase, "
            "integration error is important. If they plateau far below it, improve training/model capacity."
        ),
    }
    output = args.run_dir / "sampler_step_ablation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
