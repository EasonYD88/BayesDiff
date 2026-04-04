#!/bin/bash
#SBATCH --job-name=subsample_abl
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_872_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/subsample_ablation_%j.out
#SBATCH --error=slurm/logs/subsample_ablation_%j.err

echo "=== Subsample Ablation (CPU) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

source /scratch/yd2915/miniconda3/bin/activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

python scripts/28c_subsample_ablation.py 2>&1

echo "End: $(date)"
echo "Exit code: $?"
