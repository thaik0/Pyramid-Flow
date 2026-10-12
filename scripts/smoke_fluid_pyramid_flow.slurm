#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=pyramid-fluid-smoke
#SBATCH --output=/pscratch/sd/t/tak86/pyramid-flow-fluid-smoke/slurm-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0

cd /pscratch/sd/t/tak86/Pyramid-Flow-fluid
PY=/pscratch/sd/t/tak86/venvs/pyramid-flow-fluid/bin/python

srun "$PY" train/train_fluid_pyramid_flow.py \
  --data-root /global/cfs/cdirs/m5401/3D_HIT \
  --output-dir /pscratch/sd/t/tak86/pyramid-flow-fluid-smoke/run-seed-42-deterministic \
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
