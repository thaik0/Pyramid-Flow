#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=pyramid-fluid-overfit
#SBATCH --output=pyramid-flow-overfit-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
cd "$REPO_DIR"
OUT="$RUN_ROOT/overfit/run-seed-42"
NORM=configs/fluid_normalization_seed42_8crops.json
mkdir -p "$OUT"

srun "$PYTHON_BIN" train/train_fluid_pyramid_flow.py \
  --data-root "$DATA_ROOT" --output-dir "$OUT" \
  --normalization-json "$NORM" --crop-size 64 --patch-size 4 \
  --fixed-indices 200,201,202,203,204,205,206,207 \
  --batch-size 1 --steps 300 --model-layers 1 --model-heads 2 \
  --stage-mode cycle --checkpoint-every 100 --seed 42 --device cuda

srun "$PYTHON_BIN" tools/sample_fluid_pyramid_flow.py \
  --checkpoint "$OUT/checkpoint-300.pt" --output-dir "$OUT/generated" \
  --normalization-json "$NORM" --steps-per-stage 4 --num-samples 4 \
  --seed 4242 --device cuda

srun "$PYTHON_BIN" tools/evaluate_fluid_fields.py \
  "$OUT/generated/generated_fields.npz" --output-dir "$OUT/evaluation"
