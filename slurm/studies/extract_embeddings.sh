#!/bin/bash
#SBATCH --job-name=encoder_emb
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/encoder_emb_%j.out
#SBATCH --error=slurm/logs/encoder_emb_%j.err

set -euo pipefail

REPO="/scratch/yd2915/BayesDiff"
CONDA_ENV="/scratch/yd2915/conda_envs/bayesdiff"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO}/external/targetdiff:${PYTHONPATH:-}"

mkdir -p "${REPO}/slurm/logs"

echo "=== Encoder Embedding Extraction ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

cd "${REPO}"

python scripts/studies/embedding_encoder_only.py \
    --pocket-list "${REPO}/data/tier3_pocket_list.json" \
    --sampling-dir "${REPO}/results/tier3_sampling" \
    --output-dir "${REPO}/results/tier3_gp" \
    --batch-size 8

echo "=== Done: $(date) ==="
