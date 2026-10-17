#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=spotdiff-ablation
#SBATCH --output=spotdiff-ablation-%j.out

set -euo pipefail

module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint path}"
NORMALIZATION_JSON="${NORMALIZATION_JSON:-configs/fluid_normalization_seed42_8crops.json}"
OUTPUT_DIR="$RUN_ROOT/spotdiffusion-ablation"
cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m pytest -q \
  tests/test_spotdiffusion_3d.py \
  tests/test_spotdiffusion_sampler_3d.py

for MODE in fixed shifted; do
  srun "$PYTHON_BIN" tools/sample_spotdiffusion_3d.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR/$MODE" \
    --normalization-json "$NORMALIZATION_JSON" \
    --global-size 384 \
    --steps-per-stage 4 \
    --patch-batch-size 1 \
    --mode "$MODE" \
    --seed 4242 \
    --shift-seed 1729 \
    --device cuda \
    --save-volume
done

srun "$PYTHON_BIN" tools/evaluate_spotdiffusion_ablation.py \
  --fixed-field "$OUTPUT_DIR/fixed/physical_fields.npy" \
  --shifted-field "$OUTPUT_DIR/shifted/physical_fields.npy" \
  --fixed-metadata "$OUTPUT_DIR/fixed/generation.json" \
  --shifted-metadata "$OUTPUT_DIR/shifted/generation.json" \
  --output-dir "$OUTPUT_DIR/evaluation" \
  --window-size 64 \
  --spacing 1 1 1 \
  --device cuda

echo "Ablation outputs: $OUTPUT_DIR"
