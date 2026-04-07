#!/bin/bash
#SBATCH --job-name=ignn_node
#SBATCH --partition=a100_chemistry
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/ignn_node_%j.out
#SBATCH --error=slurm/logs/ignn_node_%j.err

# ─────────────────────────────────────────────────────────────────
# Retrain InteractionGNN with NODE readout (fix for edge-readout
# distance-homogeneity collapse). Real topology edges.
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== InteractionGNN NODE-READOUT Training (a100) ==="
echo "Date   : $(date)"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================================"

python scripts/pipeline/s10_train_interaction_gnn.py \
    --data_dir data/pdbbind_v2020 \
    --output results/stage2/interaction_gnn_node \
    --cutoff 4.5 \
    --hidden_dim 128 \
    --n_layers 2 \
    --output_dim 128 \
    --readout_mode node \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --n_epochs 100 \
    --batch_size 32 \
    --patience 15 \
    --num_workers 4 \
    --device cuda \
    --seed 42

echo ""
echo "=== Node-readout training complete $(date) ==="
