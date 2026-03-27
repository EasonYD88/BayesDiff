#!/bin/bash
#SBATCH --job-name=gp_agg
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/gp_agg_%j.out
#SBATCH --error=slurm/logs/gp_agg_%j.err

set -euo pipefail

REPO="/scratch/yd2915/BayesDiff"
CONDA_ENV="/scratch/yd2915/conda_envs/bayesdiff"

export PATH="${CONDA_ENV}/bin:${PATH}"

mkdir -p "${REPO}/slurm/logs"

echo "=== P0+: Aggregation Strategy Comparison ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

cd "${REPO}"

python scripts/21_train_gp_aggregation.py

echo "=== Done: $(date) ==="
