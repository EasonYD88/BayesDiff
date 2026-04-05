#!/bin/bash
#SBATCH --job-name=tier3_analysis
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --output=/scratch/yd2915/BayesDiff/slurm/logs/tier3_analysis_%j.out
#SBATCH --error=/scratch/yd2915/BayesDiff/slurm/logs/tier3_analysis_%j.err
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/yd2915/BayesDiff

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Tier 3 Training Curves & Data Analysis ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo "Python: $(python --version)"
echo "Start: $(date)"

python scripts/studies/tier3_training_curves.py

echo ""
echo "=== Output figures ==="
ls -lh results/tier3_gp/figures/tier3_analysis/
echo ""
echo "Done: $(date)"
