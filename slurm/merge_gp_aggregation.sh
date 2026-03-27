#!/bin/bash
#SBATCH --job-name=gp_merge
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/gp_merge_%j.out
#SBATCH --error=slurm/logs/gp_merge_%j.err

set -euo pipefail

REPO="/scratch/yd2915/BayesDiff"
CONDA_ENV="/scratch/yd2915/conda_envs/bayesdiff"

export PATH="${CONDA_ENV}/bin:${PATH}"

echo "=== P0+: Merge Aggregation Results ==="
echo "Start: $(date)"

cd "${REPO}"

python scripts/21_train_gp_aggregation.py --merge

echo "=== Done: $(date) ==="
