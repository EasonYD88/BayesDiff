#!/bin/bash
#SBATCH --job-name=schnet_v2
#SBATCH --output=slurm/logs/schnet_v2_%j.out
#SBATCH --error=slurm/logs/schnet_v2_%j.out
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_281_chemistry

echo "=== SchNet Extraction + Comparison Update ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

echo "=== Phase 1: SchNet extraction ==="
python scripts/studies/embedding_schnet.py
echo "SchNet exit code: $?"

echo "=== Phase 2: Update comparison ==="
python scripts/studies/embedding_compare_all.py
echo "Comparison exit code: $?"

echo "=== DONE ==="
echo "Date: $(date)"
