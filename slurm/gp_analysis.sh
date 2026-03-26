#!/bin/bash
#SBATCH --job-name=gp_analysis
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/gp_analysis_%j.out
#SBATCH --error=slurm/logs/gp_analysis_%j.err

set -euo pipefail

# ── Environment ──
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff.worktrees/copilot-worktree-2026-03-26T13-27-15

echo "========================================="
echo "GP Training Analysis (Train/Val/Test)"
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

python scripts/11_gp_training_analysis.py \
    --embeddings results/embedding_rdkit/all_embeddings.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/embedding_rdkit/gp_analysis \
    --n_epochs 300 \
    --n_inducing 48 \
    --augment_to 200 \
    --val_frac 0.2 \
    --test_frac 0.2

echo "Done! $(date)"
