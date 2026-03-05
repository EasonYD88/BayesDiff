#!/bin/bash
#SBATCH --job-name=bayesdiff_gp
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_gp_train.log
#SBATCH --error=slurm/logs/%j_gp_train.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Train SVGP oracle on GPU
#
# Submit:  sbatch slurm/train_gp.sh
# Monitor: squeue -u $USER && tail -f slurm/logs/<jobid>_gp_train.log
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Activate environment ─────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

echo "=== GP Training Job ==="
echo "Date      : $(date)"
echo "Host      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python    : $(which python)"
echo "========================"

# ── Run GP training on GPU ───────────────────────────────────
python scripts/04_train_gp.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --output results/gp_model \
    --n_inducing 48 \
    --n_epochs 200 \
    --augment_to 200 \
    --batch_size 64 \
    --device auto

echo ""
echo "=== GP Training Complete ==="
echo "Model: results/gp_model/gp_model.pt"
echo "Meta:  results/gp_model/train_meta.json"
ls -lh results/gp_model/
