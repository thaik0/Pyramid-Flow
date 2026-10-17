#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=spotdiffusion-gate2
#SBATCH --output=spotdiffusion-gate2-%j.out

set -euo pipefail

module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint path}"
NORMALIZATION_JSON="${NORMALIZATION_JSON:-configs/fluid_normalization_seed42_8crops.json}"
cd "$REPO_DIR"

srun "$PYTHON_BIN" -m pytest -q \
  tests/test_spotdiffusion_3d.py \
  tests/test_spotdiffusion_sampler_3d.py

srun "$PYTHON_BIN" tools/sample_spotdiffusion_3d.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT/spotdiffusion-gate2/shifted" \
  --normalization-json "$NORMALIZATION_JSON" \
  --global-size 384 \
  --steps-per-stage 1 \
  --patch-batch-size 1 \
  --mode shifted \
  --seed 4242 \
  --shift-seed 1729 \
  --device cuda
