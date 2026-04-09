#!/bin/bash
#SBATCH --job-name=atom_emb_bulk
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/atom_emb_bulk_%j.out
#SBATCH --error=slurm/logs/atom_emb_bulk_%j.err

# Run ALL remaining shards sequentially in a single GPU job.
# Shard 0 already done; run shards 1-49.

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

for SHARD in $(seq 1 49); do
    echo "--- Shard $SHARD/49 ---"
    python scripts/pipeline/s08c_extract_atom_embeddings.py \
        --data_dir data/pdbbind_v2020 \
        --output_dir results/atom_embeddings \
        --targetdiff_dir external/targetdiff \
        --device cuda \
        --shard_index $SHARD \
        --num_shards 50
done

echo "=== All shards complete at $(date) ==="
echo "Total .pt files: $(ls results/atom_embeddings/*.pt | wc -l)"
