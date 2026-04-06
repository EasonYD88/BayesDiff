#!/bin/bash
#SBATCH --job-name=multilayer_extract
#SBATCH --partition=a100_chemistry
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm/logs/multilayer_extract_%A_%a.out
#SBATCH --error=slurm/logs/multilayer_extract_%A_%a.err

# Multi-layer embedding extraction for PDBbind v2020
# Extracts per-layer embeddings from all 10 encoder layers
# using crystal ligand poses (single forward pass per complex).

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s08b_extract_multilayer.py \
    --data_dir data/pdbbind_v2020 \
    --output_dir results/multilayer_embeddings \
    --targetdiff_dir external/targetdiff \
    --device cuda \
    --shard_index $SLURM_ARRAY_TASK_ID \
    --num_shards 50

echo "Shard $SLURM_ARRAY_TASK_ID complete at $(date)"
