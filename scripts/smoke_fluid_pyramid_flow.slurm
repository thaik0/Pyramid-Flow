#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=pyramid-fluid-smoke
#SBATCH --output=pyramid-flow-fluid-smoke-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
cd "$REPO_DIR"
OUT="$RUN_ROOT/fluid-smoke/run-seed-42-deterministic"
mkdir -p "$OUT"

srun "$PYTHON_BIN" train/train_fluid_pyramid_flow.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --normalization-json configs/fluid_normalization_seed42_8crops.json \
  --crop-size 64 \
  --patch-size 4 \
  --batch-size 1 \
  --steps 1 \
  --model-layers 1 \
  --model-heads 2 \
  --num-workers 0 \
  --seed 42 \
  --device cuda
