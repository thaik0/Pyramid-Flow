"""Small provisional metric set for generated 3-D velocity fields.

These are research diagnostics, not the CoMSAIL standardized metrics. Spectrum
and divergence use a periodic-crop approximation that must be reviewed with the
group before official comparisons.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def component_statistics(fields):
    flat = fields.transpose(1, 0, 2, 3, 4).reshape(3, -1)
    return {
        name: {"mean": float(x.mean()), "std": float(x.std()),
               "min": float(x.min()), "max": float(x.max())}
        for name, x in zip(("u", "v", "w"), flat)
    }


def divergence_rms(fields, spacing):
    derivatives = []
    for component, axis, dx in zip(range(3), (1, 2, 3), spacing):
        value = fields[:, component]
        derivatives.append((np.roll(value, -1, axis=axis) - np.roll(value, 1, axis=axis)) / (2 * dx))
    divergence = sum(derivatives)
    return float(np.sqrt(np.mean(divergence ** 2)))


def energy_spectrum(fields, spacing):
    n = fields.shape[-1]
    transformed = np.fft.fftn(fields, axes=(-3, -2, -1))
    energy = 0.5 * np.sum(np.abs(transformed) ** 2, axis=1) / (n ** 6)
    frequencies = [np.fft.fftfreq(n, d=dx) for dx in spacing]
    kz, ky, kx = np.meshgrid(*frequencies, indexing="ij")
    shell = np.rint(np.sqrt(kx * kx + ky * ky + kz * kz) / min(abs(f[1]) for f in frequencies)).astype(int)
    spectrum = np.bincount(shell.ravel(), weights=energy.sum(axis=0).ravel())
    counts = np.bincount(shell.ravel())
    return spectrum, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()
    payload = np.load(args.fields)
    fields = np.asarray(payload["physical"], dtype=np.float64)
    if fields.ndim != 5 or fields.shape[1] != 3:
        raise ValueError("Expected physical fields shaped [N,3,D,H,W]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectrum, counts = energy_spectrum(fields, args.spacing)
    metrics = {
        "status": "provisional_pending_Delin_Meet_standardization",
        "component_statistics": component_statistics(fields),
        "mean_kinetic_energy": float(0.5 * np.mean(np.sum(fields ** 2, axis=1))),
        "divergence_rms_periodic_crop_approximation": divergence_rms(fields, args.spacing),
        "spacing": list(args.spacing), "sample_count": int(fields.shape[0]),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output_dir / "energy_spectrum.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shell", "energy", "mode_count"])
        writer.writerows((i, float(e), int(c)) for i, (e, c) in enumerate(zip(spectrum, counts)))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
