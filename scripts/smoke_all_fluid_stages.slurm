#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=pyramid-fluid-stages
#SBATCH --output=/pscratch/sd/t/tak86/pyramid-flow-stage-smoke/slurm-%j.out

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0
cd /pscratch/sd/t/tak86/Pyramid-Flow-fluid
PY=/pscratch/sd/t/tak86/venvs/pyramid-flow-fluid/bin/python

srun "$PY" train/train_fluid_pyramid_flow.py \
  --data-root /global/cfs/cdirs/m5401/3D_HIT \
  --output-dir /pscratch/sd/t/tak86/pyramid-flow-stage-smoke/run-seed-42 \
  --normalization-json configs/fluid_normalization_seed42_8crops.json \
  --crop-size 64 --patch-size 4 --fixed-indices 200,201,202 \
  --batch-size 1 --steps 3 --model-layers 1 --model-heads 2 \
  --stage-mode cycle --seed 42 --device cuda
