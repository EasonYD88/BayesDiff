#!/bin/bash
#SBATCH --job-name=ml_emb
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/ml_emb_%A_%a.out
#SBATCH --error=slurm/logs/ml_emb_%A_%a.err

set -euo pipefail

REPO="/scratch/yd2915/BayesDiff"
CONDA_ENV="/scratch/yd2915/conda_envs/bayesdiff"
export PATH="${CONDA_ENV}/bin:${PATH}"

mkdir -p "${REPO}/slurm/logs"

echo "=== P0++: Multi-layer embedding extraction (shard ${SLURM_ARRAY_TASK_ID}/4) ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

cd "${REPO}"

python scripts/22_extract_multilayer_embeddings.py \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --num-shards 4

echo "=== Done: $(date) ==="
