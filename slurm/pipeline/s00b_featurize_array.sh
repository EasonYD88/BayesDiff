#!/bin/bash
#SBATCH --job-name=pdbbind_feat
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm/logs/%A_%a_pdbbind_feat.log
#SBATCH --error=slurm/logs/%A_%a_pdbbind_feat.err

# ─────────────────────────────────────────────────────────────
# Stage 0b: Featurize PDBbind complexes via SLURM array job
#
# Runs 50 parallel shards, each processing ~106 complexes with
# 16 CPUs per shard (multiprocessing within shard).
# Total: 50 shards × 16 CPUs = 800 CPU cores processing in parallel.
#
# ~5,316 complexes / 50 shards ≈ 106 per shard
# Each shard takes ~5-10 min with 16 workers → total wall time ~10 min.
#
# Prerequisites:
#   - labels.csv must exist (run s00a_parse.sh first)
#
# Submit:
#   sbatch slurm/pipeline/s00b_featurize_array.sh
#
# Or with custom settings:
#   PDBBIND_DIR=/path/to/pdbbind sbatch --array=0-99 slurm/pipeline/s00b_featurize_array.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PDBBIND_DIR="${PDBBIND_DIR:-data/pdbbind}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pdbbind_v2020}"

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-50}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p slurm/logs "${OUTPUT_DIR}/processed"

echo "=== PDBbind v2020: Featurize Shard ${SHARD_INDEX}/${NUM_SHARDS} ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "CPUs:        ${NUM_WORKERS}"
echo "PDBbind dir: ${PDBBIND_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

python scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --stage featurize \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}" \
    --num_workers "${NUM_WORKERS}"

echo ""
echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} complete."
