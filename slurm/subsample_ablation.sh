#!/bin/bash
#SBATCH --job-name=a5_ablation
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_872_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/a5_ablation_%j.out
#SBATCH --error=slurm/logs/a5_ablation_%j.err

set -e

cd /scratch/yd2915/BayesDiff
source /scratch/yd2915/miniconda3/bin/activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== A5 Ablation: Subsample molecules per pocket ==="
echo "Start: $(date)"
python scripts/28c_subsample_ablation.py
echo "Done: $(date)"

echo ""
echo "Results: results/50mol_gp/ablation_results.json"
echo "Figure:  results/50mol_gp/figures/05_ablation.png"
