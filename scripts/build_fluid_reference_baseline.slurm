#!/bin/bash
#SBATCH --account=m5401_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=fluid-reference
#SBATCH --output=fluid-reference-%j.out

set -euo pipefail

module load pytorch/2.6.0
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${FLUID_DATA_ROOT:-/global/cfs/cdirs/m5401/3D_HIT}"
RUN_ROOT="${PYRAMID_FLOW_RUN_ROOT:-${PSCRATCH:-$REPO_DIR}/pyramid-flow-runs}"
OUTPUT_DIR="${REFERENCE_OUTPUT_DIR:-$RUN_ROOT/fluid-reference-baseline}"
REFERENCE_INDEX_START="${REFERENCE_INDEX_START:-0}"
REFERENCE_INDEX_COUNT="${REFERENCE_INDEX_COUNT:-16}"
GRID_SPACING="${GRID_SPACING:-1 1 1}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

srun "$PYTHON_BIN" -m pytest -q \
  tests/test_fluid_reference_baseline.py \
  tests/test_fluid_metrics.py \
  tests/test_fluid_volume_dataset.py

srun "$PYTHON_BIN" tools/build_fluid_reference_baseline.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --index-start "$REFERENCE_INDEX_START" \
  --index-count "$REFERENCE_INDEX_COUNT" \
  --training-index-start 200 \
  --training-index-end 327 \
  --minimum-buffer 32 \
  --spacing $GRID_SPACING \
  --device cuda

echo "Reference baseline outputs: $OUTPUT_DIR"
