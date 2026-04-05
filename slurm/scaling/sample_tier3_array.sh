#!/bin/bash
#SBATCH --job-name=tier3_sample
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/tier3_sample_%A_%a.out
#SBATCH --error=slurm/logs/tier3_sample_%A_%a.err
# Submit with: sbatch --array=0-15 slurm/sample_tier3_array.sh

set -euo pipefail

# Activate conda environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

# Determine number of shards from array task range
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-16}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "=== Tier 3 Sampling Shard ${SHARD_INDEX}/${NUM_SHARDS} ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python scripts/scaling/s04_sample_tier3_shard.py \
    --pocket-list data/tier3_pocket_list.json \
    --output-dir results/tier3_sampling \
    --num-samples 64 \
    --num-steps 100 \
    --num-shards ${NUM_SHARDS} \
    --shard-index ${SHARD_INDEX} \
    --skip-existing \
    --device cuda

echo "=== Shard ${SHARD_INDEX} complete at $(date) ==="
