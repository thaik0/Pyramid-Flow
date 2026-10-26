#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=fluid-paired-recon
#SBATCH --output=fluid-paired-recon-%j.out

set -euo pipefail
module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
TRAIN_DIR="${CORRECTED_RUN_DIR:-$RUN_ROOT/corrected-normalization/run-seed-42}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_DIR/checkpoint-20000.pt}"
OUTPUT_DIR="${PAIRED_RECON_DIR:-$TRAIN_DIR/paired-reconstruction-20000}"
NORM="configs/fluid_normalization_training_200_327.json"

cd "$REPO_DIR"
test -f "$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m pytest -q \
  tests/test_paired_fluid_similarity.py \
  tests/test_paired_stage_reconstruction.py

srun "$PYTHON_BIN" tools/evaluate_paired_stage_reconstruction.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --normalization-json "$NORM" \
  --output-dir "$OUTPUT_DIR" \
  --index-start 0 \
  --index-count 16 \
  --noise-ratios 0.25 0.5 0.75 1.0 \
  --updates 16 \
  --window-size 7 \
  --device cuda

echo "Paired metrics: $OUTPUT_DIR/paired_reconstruction_report.json"
