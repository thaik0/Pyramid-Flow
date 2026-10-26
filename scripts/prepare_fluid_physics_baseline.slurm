#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --job-name=fluid-physics
#SBATCH --output=fluid-physics-%j.out

set -euo pipefail

module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
OUTPUT_DIR="${PHYSICS_OUTPUT_DIR:-$RUN_ROOT/fluid-physics-baseline}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m pytest -q \
  tests/test_paired_fluid_similarity.py \
  tests/test_fluid_reference_baseline.py \
  tests/test_fluid_volume_dataset.py

srun "$PYTHON_BIN" tools/inspect_fluid_metadata.py \
  --data-root "$DATA_ROOT" \
  --files fluid_25.nc fluid_30.nc \
  --output "$OUTPUT_DIR/dataset_metadata.json"

srun "$PYTHON_BIN" tools/compute_fluid_training_normalization.py \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_DIR/training_normalization.json" \
  --index-start 200 \
  --index-count 128 \
  --device cuda

echo "Physics baseline outputs: $OUTPUT_DIR"
