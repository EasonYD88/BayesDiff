#!/bin/bash
#SBATCH --job-name=concat_mlp
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/concat_mlp_%j.out
#SBATCH --error=slurm/logs/concat_mlp_%j.err

# Stage 4: Concat+MLP Fusion — 3 output dims × 5 folds = 15 runs

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s09e_concat_mlp_fusion.py \
    --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \
    --labels data/pdbbind_v2020/labels.csv \
    --splits_5fold data/pdbbind_v2020/splits_5fold.json \
    --cv_summary results/stage2/cross_validation/cv_summary.csv \
    --output results/stage2/concat_mlp \
    --n_inducing 512 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --fusion_lr 0.01 \
    --device cuda

echo "Stage 4 Concat+MLP complete at $(date)"
