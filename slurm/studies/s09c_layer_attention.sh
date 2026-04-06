#!/bin/bash
#SBATCH --job-name=attn_fusion
#SBATCH --partition=a100_chemistry
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/attn_fusion_%j.out
#SBATCH --error=slurm/logs/attn_fusion_%j.err

# Stage 3: Layer Attention Fusion — input-dependent layer weighting
# Prerequisite: s09a (layer probing) + s09b (weighted sum) complete

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s09c_layer_attention_fusion.py \
    --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --stage1_results results/stage2/layer_probing/layer_probing.csv \
    --stage2_results results/stage2/weighted_sum/gate2_decision.json \
    --stage2_csv results/stage2/weighted_sum/weighted_sum_results.csv \
    --output results/stage2/layer_attention \
    --n_inducing 512 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --fusion_lr 0.01 \
    --hidden_dim 64 \
    --device cuda

echo "Layer attention fusion complete at $(date)"
