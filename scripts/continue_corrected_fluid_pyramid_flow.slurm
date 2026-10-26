#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=pyramid-fluid-20k
#SBATCH --output=pyramid-fluid-20k-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
OUT="${CORRECTED_RUN_DIR:-$RUN_ROOT/corrected-normalization/run-seed-42}"
RESUME_CHECKPOINT="${CHECKPOINT:-$OUT/checkpoint-5000.pt}"
NORM="configs/fluid_normalization_training_200_327.json"
SPACING="0.016362462547794956"

cd "$REPO_DIR"
test -f "$RESUME_CHECKPOINT"

srun "$PYTHON_BIN" train/train_fluid_pyramid_flow.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --normalization-json "$NORM" \
  --crop-size 64 \
  --patch-size 4 \
  --index-start 200 \
  --index-count 128 \
  --batch-size 1 \
  --steps 20000 \
  --model-layers 1 \
  --model-heads 2 \
  --stage-mode sample \
  --checkpoint-every 5000 \
  --cache-dataset \
  --resume "$RESUME_CHECKPOINT" \
  --seed 42 \
  --device cuda

srun "$PYTHON_BIN" tools/sample_fluid_pyramid_flow.py \
  --checkpoint "$OUT/checkpoint-20000.pt" \
  --output-dir "$OUT/generated-20000" \
  --normalization-json "$NORM" \
  --steps-per-stage 4 \
  --num-samples 8 \
  --seed 4242 \
  --device cuda

srun "$PYTHON_BIN" tools/evaluate_fluid_fields.py \
  "$OUT/generated-20000/generated_fields.npz" \
  --output-dir "$OUT/evaluation-20000" \
  --spacing "$SPACING" "$SPACING" "$SPACING"

srun "$PYTHON_BIN" tools/summarize_fluid_training.py \
  "$OUT/metrics.jsonl" \
  --output "$OUT/training_summary_20000.json"

echo "20k metrics: $OUT/evaluation-20000/metrics.json"
