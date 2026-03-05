#!/bin/bash
#SBATCH --job-name=bayesdiff_eval
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_eval.log
#SBATCH --error=slurm/logs/%j_eval.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: S6 Evaluation + S7 Ablation (GPU)
#
# Submit:  sbatch slurm/eval_ablation.sh
# Monitor: squeue -u $USER && tail -f slurm/logs/<jobid>_eval.log
# ─────────────────────────────────────────────────────────────

set -euo pipefail

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

echo "=== S6 + S7 Evaluation Job ==="
echo "Date : $(date)"
echo "Host : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "==============================="

# ── S6: Full Evaluation ──────────────────────────────────────
echo ""
echo ">>> S6: Running full evaluation pipeline..."
python scripts/05_evaluate.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --gp_model results/gp_model/gp_model.pt \
    --gp_train_data results/gp_model/train_data.npz \
    --output results/evaluation \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 500

echo ""
echo ">>> S6 complete. Results:"
ls -lh results/evaluation/

# ── S7: Ablation Study ──────────────────────────────────────
echo ""
echo ">>> S7: Running ablation study..."
python scripts/06_ablation.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --gp_model results/gp_model/gp_model.pt \
    --gp_train_data results/gp_model/train_data.npz \
    --output results/ablation \
    --y_target 7.0 \
    --bootstrap_n 500

echo ""
echo ">>> S7 complete. Results:"
ls -lh results/ablation/

echo ""
echo "=== All Evaluation & Ablation Complete ==="
echo "Evaluation: results/evaluation/"
echo "Ablation:   results/ablation/"
