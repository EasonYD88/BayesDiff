#!/bin/bash
#SBATCH --job-name=bayesdiff_merge_shards
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_merge.log
#SBATCH --error=slurm/logs/%j_merge.err

# Merge sharded sampling outputs from slurm/sample_array_job.sh
#
# Required env vars:
#   RUN_DIR=results/generated_molecules_parallel/<run_tag>
# Optional:
#   EXPECTED_SHARDS=<N>

set -euo pipefail

RUN_DIR="${RUN_DIR:-}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR is required."
  echo "Example: sbatch --export=ALL,RUN_DIR=results/generated_molecules_parallel/<run_tag>,EXPECTED_SHARDS=4 slurm/merge_sample_shards_job.sh"
  exit 1
fi

SHARDS_DIR="${RUN_DIR}/shards"

echo "=== BayesDiff Merge Shards Job ==="
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "Run dir:         ${RUN_DIR}"
echo "Shards dir:      ${SHARDS_DIR}"
echo "Expected shards: ${EXPECTED_SHARDS:-<not-set>}"
echo ""

# Activate conda environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

if [[ -n "${EXPECTED_SHARDS}" ]]; then
  python scripts/07_merge_sampling_shards.py \
    --shards_dir "${SHARDS_DIR}" \
    --output_dir "${RUN_DIR}" \
    --expected_shards "${EXPECTED_SHARDS}"
else
  python scripts/07_merge_sampling_shards.py \
    --shards_dir "${SHARDS_DIR}" \
    --output_dir "${RUN_DIR}"
fi

echo ""
echo "Merge complete."
ls -lh "${RUN_DIR}/all_embeddings.npz" "${RUN_DIR}/sampling_summary.json"
