#!/bin/bash
#SBATCH --job-name=robust_eval
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/robust_eval_%j.out
#SBATCH --error=slurm/logs/robust_eval_%j.err

set -euo pipefail

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff.worktrees/copilot-worktree-2026-03-26T13-27-15

echo "========================================="
echo "Robust GP Evaluation (LOOCV + Splits + Bootstrap)"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "========================================="

python scripts/12_robust_evaluation.py \
    --embeddings results/embedding_rdkit/all_embeddings.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/embedding_rdkit/robust_eval \
    --n_epochs 80 \
    --n_repeats 50 \
    --n_bootstrap 200

echo "Done! $(date)"
