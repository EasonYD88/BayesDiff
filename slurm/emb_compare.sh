#!/bin/bash
#SBATCH --job-name=emb_compare
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/emb_compare_%j.out
#SBATCH --error=slurm/logs/emb_compare_%j.err

set -euo pipefail

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff.worktrees/copilot-worktree-2026-03-26T13-27-15

echo "========================================="
echo "Embedding Comparison (6 embeddings × ExactGP)"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "========================================="

python scripts/13_embedding_comparison.py \
    --sdf_dir results/embedding_1000step \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/embedding_comparison \
    --n_epochs 80 \
    --n_repeats 30 \
    --pca_dim 20

echo "Done! $(date)"
