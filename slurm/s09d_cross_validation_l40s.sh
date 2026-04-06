#!/bin/bash
#SBATCH --job-name=cv_5fold
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/cv_5fold_%j.out
#SBATCH --error=slurm/logs/cv_5fold_%j.err

# Stage 2.5: 5-Fold Cross-Validation (4 models × 5 folds = 20 runs)

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s09d_cross_validation.py \
    --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \
    --labels data/pdbbind_v2020/labels.csv \
    --splits_5fold data/pdbbind_v2020/splits_5fold.json \
    --output results/stage2/cross_validation \
    --n_inducing 512 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --device cuda

echo "5-fold CV complete at $(date)"
