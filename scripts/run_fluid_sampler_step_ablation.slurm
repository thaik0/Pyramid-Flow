#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=fluid-sampler-steps
#SBATCH --output=fluid-sampler-steps-%j.out

set -euo pipefail
module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
TRAIN_DIR="${CORRECTED_RUN_DIR:-$RUN_ROOT/corrected-normalization/run-seed-42}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_DIR/checkpoint-5000.pt}"
OUTPUT_DIR="${SAMPLER_ABLATION_DIR:-$TRAIN_DIR/sampler-step-ablation}"
NORM="configs/fluid_normalization_training_200_327.json"
SPACING="0.016362462547794956"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

for STEPS in 4 8 16 32; do
  srun "$PYTHON_BIN" tools/sample_fluid_pyramid_flow.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR/steps-$STEPS/generated" \
    --normalization-json "$NORM" \
    --steps-per-stage "$STEPS" \
    --num-samples 8 \
    --seed 4242 \
    --device cuda

  srun "$PYTHON_BIN" tools/evaluate_fluid_fields.py \
    "$OUTPUT_DIR/steps-$STEPS/generated/generated_fields.npz" \
    --output-dir "$OUTPUT_DIR/steps-$STEPS/evaluation" \
    --spacing "$SPACING" "$SPACING" "$SPACING"
done

srun "$PYTHON_BIN" tools/summarize_fluid_sampler_steps.py \
  --run-dir "$OUTPUT_DIR" \
  --steps 4 8 16 32 \
  --spacing "$SPACING"

echo "Sampler-step ablation: $OUTPUT_DIR/sampler_step_ablation.json"
