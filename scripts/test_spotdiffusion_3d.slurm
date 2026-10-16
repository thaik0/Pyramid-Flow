#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=spotdiffusion-tests
#SBATCH --output=spotdiffusion-tests-%j.out

set -euo pipefail

module load pytorch/2.6.0
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

srun "$PYTHON_BIN" -m pytest -q tests/test_spotdiffusion_3d.py
