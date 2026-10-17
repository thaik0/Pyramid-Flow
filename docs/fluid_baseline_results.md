# Preliminary fluid Pyramid Flow baseline results

These results are engineering validation for the provisional 3-D fluid port.
They demonstrate that the implementation executes, optimizes, and samples;
they are not yet evidence of scientific fidelity or readiness for a
`384^3` comparison.

## Per-stage optimization smoke test

Each pyramid stage completed a forward pass, backward pass, and optimizer
step.

| Resolution | Tokens | Loss | Peak GPU memory |
|---|---:|---:|---:|
| `16^3` | 64 | 0.231 | approximately 0.08 GB |
| `32^3` | 512 | 0.576 | approximately 0.09 GB |
| `64^3` | 4,096 | 0.596 | approximately 0.66 GB |

The loss values from this smoke test are not intended as quality metrics.

## Eight-crop overfit test

The model was trained for 300 steps on eight fixed `64^3` crops. Loss fell at
all three stages:

- `16^3`: approximately 41 percent
- `32^3`: approximately 52 percent
- `64^3`: approximately 49 percent

Four new `64^3` velocity fields were sampled from the final checkpoint.

## Small training run

The larger smoke baseline used 128 crops and 600 optimizer steps with:

```yaml
seed: 42
batch_size: 1
learning_rate: 1.0e-4
model_layers: 1
attention_heads: 2
```

Loss fell at all three stages:

- `16^3`: approximately 47 percent
- `32^3`: approximately 58 percent
- `64^3`: approximately 52 percent

Eight samples were generated from the final checkpoint. The run was reported
as reproducible.

## Preliminary generated-field diagnostics

For the eight generated samples:

| Diagnostic | Value |
|---|---:|
| `u` standard deviation | 0.780 |
| `v` standard deviation | 0.851 |
| `w` standard deviation | 0.807 |
| Mean kinetic energy | 1.019 |
| Approximate divergence RMS | 0.815 |

The divergence implementation uses a periodic-crop approximation. These
diagnostics have not yet been compared with held-out real fields or an
approved laboratory metric implementation.

## Observed bottleneck

The 600-step job took approximately 44 minutes of wall time while recorded GPU
operations took approximately 9 seconds. This strongly suggests an input
pipeline bottleneck, but the cause has not yet been isolated. Candidate
follow-up measurements include NetCDF open time, crop extraction time, host to
device transfer time, data-loader wait time, and GPU utilization.

Potential optimizations to test only after profiling include keeping files
open, grouping crops from the same file, caching the bounded training subset,
and using multiple data-loader workers.

## Current interpretation

The 600-step checkpoint is suitable for implementation-level inference smoke
tests. It should not yet be treated as a scientifically validated generator.
The current dense-attention model also cannot be run naively at `384^3`; see
[`scaling_384.md`](scaling_384.md).

Before a formal experiment, record the checkpoint's durable location and
SHA-256 checksum, the exact job identifier, software environment, complete
metrics, and generated-sample paths.

## SpotDiffusion-style tiling correctness gate

The tensor-only 3-D shifted-window implementation was tested on NERSC before
any model inference changes. Slurm job `59716765` ran with the PyTorch 2.6.0
module and reported `5 passed in 8.35s`.

The passing tests verify:

- exact split and stitch of a `[1,3,384,384,384]` tensor into 216
  non-overlapping `[3,64,64,64]` patches;
- exact reconstruction after cyclic shift, split, stitch, and inverse shift;
- complete voxel coverage with no gaps or overlaps;
- reproducible seeded 3-D shift schedules, with different seeds producing
  different schedules; and
- preservation of the `u`, `v`, and `w` channel axis during shifting and
  patching.

This validates patch geometry and reproducibility only. It does not yet test a
checkpoint forward pass, scheduler integration, memory use during model
inference, seam reduction, or physical-field quality.

## Gate 2 implementation status

The checkpoint-integrated global sampler is implemented on the
`spotdiffusion-3d` branch. It keeps one global state at each pyramid
resolution (`96^3`, `192^3`, then `384^3`), predicts stage-consistent local
windows (`16^3`, `32^3`, then `64^3`) in bounded batches, stitches and
unshifts one global velocity field, and applies one scheduler update to the
global state.

The accompanying Slurm smoke job runs the tensor tests, sampler unit tests,
and one checkpoint-backed global step per stage.

Gate 2 passed on Perlmutter in Slurm job `59911248`:

- all seven tensor and sampler tests passed in 26.93 seconds;
- the 600-step checkpoint loaded with SHA-256
  `881aecf70a1e8220f2eaf9891d8bb627f2bd31d13eeface1b077558909bbe019`;
- the global stage shapes were `96^3`, `192^3`, and `384^3` with local window
  sizes `16^3`, `32^3`, and `64^3`;
- all three post-update global states contained only finite values;
- checkpoint-backed sampling took 3.51 seconds; and
- peak allocated GPU memory was 6,220,302,848 bytes (approximately 6.22 GB).

The job used one shifted-window flow step per stage and did not save the final
volume. This establishes checkpoint, local-prediction, stitching, shift
inversion, and global-scheduler integration. It does not establish seam
reduction or scientific quality; those require a matched fixed-versus-shifted
multi-step evaluation.
