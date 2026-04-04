#!/bin/bash
#SBATCH --job-name=ml_merge
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/ml_merge_%j.out
#SBATCH --error=slurm/logs/ml_merge_%j.err

set -euo pipefail

REPO="/scratch/yd2915/BayesDiff"
CONDA_ENV="/scratch/yd2915/conda_envs/bayesdiff"
export PATH="${CONDA_ENV}/bin:${PATH}"

echo "=== P0++: Merge multi-layer results ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Start: $(date)"

cd "${REPO}"

python scripts/23_train_gp_multilayer.py --merge

echo "=== Done: $(date) ==="
