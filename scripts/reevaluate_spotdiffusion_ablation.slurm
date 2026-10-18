#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=spotdiff-reeval
#SBATCH --output=spotdiff-reeval-%j.out

set -euo pipefail

module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
OUTPUT_DIR="$RUN_ROOT/spotdiffusion-ablation"
cd "$REPO_DIR"

srun "$PYTHON_BIN" -m pytest -q tests/test_spotdiffusion_ablation.py

srun "$PYTHON_BIN" tools/evaluate_spotdiffusion_ablation.py \
  --fixed-field "$OUTPUT_DIR/fixed/physical_fields.npy" \
  --shifted-field "$OUTPUT_DIR/shifted/physical_fields.npy" \
  --fixed-metadata "$OUTPUT_DIR/fixed/generation.json" \
  --shifted-metadata "$OUTPUT_DIR/shifted/generation.json" \
  --output-dir "$OUTPUT_DIR/evaluation" \
  --window-size 64 \
  --spacing 1 1 1 \
  --device cuda

echo "Revised ablation report: $OUTPUT_DIR/evaluation/ablation_report.json"
