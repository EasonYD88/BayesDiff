#!/bin/bash
#SBATCH --job-name=gp_agg
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm/logs/gp_agg_%A_%a.out
#SBATCH --error=slurm/logs/gp_agg_%A_%a.err

set -euo pipefail

REPO="/scratch/yd2915/BayesDiff"
CONDA_ENV="/scratch/yd2915/conda_envs/bayesdiff"

export PATH="${CONDA_ENV}/bin:${PATH}"

mkdir -p "${REPO}/slurm/logs"

STRATEGIES=(mean max "mean+max" "mean+std" attn_norm attn_pca trimmed_mean)

echo "=== P0+: Aggregation Strategy ${STRATEGIES[$SLURM_ARRAY_TASK_ID]} ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

cd "${REPO}"

python scripts/studies/gp_aggregation.py --strategy-index "${SLURM_ARRAY_TASK_ID}"

echo "=== Done: $(date) ==="
