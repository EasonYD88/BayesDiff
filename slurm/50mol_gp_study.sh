#!/bin/bash
#SBATCH --job-name=50mol_gp_study
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/50mol_gp_%j.out
#SBATCH --error=slurm/logs/50mol_gp_%j.err

set -e

cd /scratch/yd2915/BayesDiff
source /scratch/yd2915/miniconda3/bin/activate /scratch/yd2915/conda_envs/bayesdiff

# Install optuna if missing
pip install optuna --quiet 2>/dev/null || true

echo "=== Step 1: Extract 50mol encoder embeddings ==="
echo "Start: $(date)"
python scripts/29_extract_50mol_embeddings.py
echo "Done: $(date)"

echo ""
echo "=== Step 2: GP Study ==="
echo "Start: $(date)"
python scripts/28_50mol_gp_study.py --n_bo_trials 200
echo "Done: $(date)"

echo ""
echo "=== All done ==="
echo "Results: results/50mol_gp/"
echo "Figures: results/50mol_gp/figures/"
