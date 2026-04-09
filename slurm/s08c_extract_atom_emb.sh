#!/bin/bash
#SBATCH --job-name=atom_emb_extract
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm/logs/atom_emb_extract_%A_%a.out
#SBATCH --error=slurm/logs/atom_emb_extract_%A_%a.err

# Sub-Plan 2: Extract atom-level embeddings for attention pooling.
# Saves per-complex .pt files with (N_lig, 128) atom tensors per layer.
# Prerequisite: s08b (multilayer mean-pooled) already done — this is complementary.

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

python scripts/pipeline/s08c_extract_atom_embeddings.py \
    --data_dir data/pdbbind_v2020 \
    --output_dir results/atom_embeddings \
    --targetdiff_dir external/targetdiff \
    --device cuda \
    --shard_index $SLURM_ARRAY_TASK_ID \
    --num_shards 50

echo "Shard $SLURM_ARRAY_TASK_ID complete at $(date)"
