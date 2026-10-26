#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=fluid-hit-eval
#SBATCH --output=fluid-hit-eval-%j.out

set -euo pipefail
module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
TRAIN_DIR="${CORRECTED_RUN_DIR:-$RUN_ROOT/corrected-normalization/run-seed-42}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_DIR/checkpoint-20000.pt}"
OUTPUT_DIR="${HIT_EVAL_DIR:-$TRAIN_DIR/hit-and-context-evaluation}"
GLOBAL_DIR="$OUTPUT_DIR/global-384"
FIXED_FIELD="$GLOBAL_DIR/fixed/physical_fields.npy"
SHIFTED_FIELD="$GLOBAL_DIR/shifted/physical_fields.npy"
NORM="configs/fluid_normalization_training_200_327.json"
SPACING="0.016362462547794956"

cd "$REPO_DIR"
test -f "$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m pytest -q \
  tests/test_hit_protocol.py \
  tests/test_paired_stage_reconstruction.py

for MODE in fixed shifted; do
  srun "$PYTHON_BIN" tools/sample_spotdiffusion_3d.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$GLOBAL_DIR/$MODE" \
    --normalization-json "$NORM" \
    --global-size 384 \
    --steps-per-stage 4 \
    --patch-batch-size 1 \
    --mode "$MODE" \
    --seed 4242 \
    --shift-seed 1729 \
    --device cuda \
    --save-volume
done

srun "$PYTHON_BIN" tools/evaluate_hit_protocol.py \
  --data-root "$DATA_ROOT" \
  --fixed-field "$FIXED_FIELD" \
  --shifted-field "$SHIFTED_FIELD" \
  --output-dir "$OUTPUT_DIR/hit-metrics" \
  --reference-index-start 0 \
  --reference-index-count 16 \
  --spacing "$SPACING" "$SPACING" "$SPACING" \
  --device cuda

srun "$PYTHON_BIN" tools/evaluate_coarse_context_sensitivity.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --normalization-json "$NORM" \
  --output-dir "$OUTPUT_DIR/context-sensitivity" \
  --index-start 0 \
  --index-count 16 \
  --context-noise-levels 0 0.1 0.25 0.5 1.0 \
  --updates 16 \
  --device cuda

echo "HIT report: $OUTPUT_DIR/hit-metrics/hit_metrics_report.json"
echo "Context report: $OUTPUT_DIR/context-sensitivity/coarse_context_sensitivity.json"
