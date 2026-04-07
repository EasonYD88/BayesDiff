#!/bin/bash
#SBATCH --job-name=ignn_train
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/ignn_train_%j.out
#SBATCH --error=slurm/logs/ignn_train_%j.err

# ─────────────────────────────────────────────────────────────────
# Stage 2 Sub-Plan 1: Train InteractionGNN on PDBbind v2020
#
# Submit:   sbatch slurm/s10_train_ignn.sh
# Monitor:  squeue -u $USER && tail -f slurm/logs/ignn_train_<jobid>.out
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== InteractionGNN Training ==="
echo "Date   : $(date)"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python : $(which python)"
echo "================================"

python scripts/pipeline/s10_train_interaction_gnn.py \
    --data_dir data/pdbbind_v2020 \
    --output results/stage2/interaction_gnn \
    --cutoff 4.5 \
    --hidden_dim 128 \
    --n_layers 2 \
    --output_dim 128 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --n_epochs 100 \
    --batch_size 32 \
    --patience 15 \
    --num_workers 4 \
    --device cuda \
    --seed 42

echo ""
echo "=== Training Complete $(date) ==="
