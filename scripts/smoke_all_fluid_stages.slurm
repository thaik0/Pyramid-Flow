#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=pyramid-fluid-stages
#SBATCH --output=pyramid-flow-stage-smoke-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
cd "$REPO_DIR"
OUT="$RUN_ROOT/stage-smoke/run-seed-42"
mkdir -p "$OUT"

srun "$PYTHON_BIN" train/train_fluid_pyramid_flow.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --normalization-json configs/fluid_normalization_seed42_8crops.json \
  --crop-size 64 --patch-size 4 --fixed-indices 200,201,202 \
  --batch-size 1 --steps 3 --model-layers 1 --model-heads 2 \
  --stage-mode cycle --seed 42 --device cuda
