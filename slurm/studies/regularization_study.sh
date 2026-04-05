#!/bin/bash
#SBATCH --job-name=reg_study
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/reg_study_%j.out
#SBATCH --error=slurm/logs/reg_study_%j.err

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Regularization Study ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python scripts/studies/regularization_study.py

echo "End: $(date)"
