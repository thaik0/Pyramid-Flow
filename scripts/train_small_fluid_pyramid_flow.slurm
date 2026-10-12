#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=pyramid-fluid-small
#SBATCH --output=/pscratch/sd/t/tak86/pyramid-flow-small/slurm-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0
cd /pscratch/sd/t/tak86/Pyramid-Flow-fluid
PY=/pscratch/sd/t/tak86/venvs/pyramid-flow-fluid/bin/python
OUT=/pscratch/sd/t/tak86/pyramid-flow-small/run-seed-42
NORM=configs/fluid_normalization_seed42_8crops.json

srun "$PY" train/train_fluid_pyramid_flow.py \
  --data-root /global/cfs/cdirs/m5401/3D_HIT --output-dir "$OUT" \
  --normalization-json "$NORM" --crop-size 64 --patch-size 4 \
  --index-start 200 --index-count 128 \
  --batch-size 1 --steps 600 --model-layers 1 --model-heads 2 \
  --stage-mode sample --checkpoint-every 100 --seed 42 --device cuda

srun "$PY" tools/sample_fluid_pyramid_flow.py \
  --checkpoint "$OUT/checkpoint-600.pt" --output-dir "$OUT/generated" \
  --normalization-json "$NORM" --steps-per-stage 4 --num-samples 8 \
  --seed 4242 --device cuda

srun "$PY" tools/evaluate_fluid_fields.py \
  "$OUT/generated/generated_fields.npz" --output-dir "$OUT/evaluation"
