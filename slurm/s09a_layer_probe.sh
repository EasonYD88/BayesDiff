#!/bin/bash
#SBATCH --job-name=layer_probe
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/layer_probe_%j.out
#SBATCH --error=slurm/logs/layer_probe_%j.err

# Stage 1: Single-layer probing — GP per encoder layer
# Prerequisite: s08b extraction + merge must be complete

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s09a_single_layer_probe.py \
    --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/layer_probing \
    --n_inducing 512 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --device cuda

echo "Layer probing complete at $(date)"
