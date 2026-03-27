#!/bin/bash
#SBATCH --job-name=bo_gp
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/bo_gp_%j.out
#SBATCH --error=slurm/logs/bo_gp_%j.err

set -euo pipefail

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

echo "========================================="
echo "Bayesian Optimization of GP Hyperparameters"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "========================================="

python scripts/14_bo_gp_hyperparams.py \
    --sdf_dir results/embedding_1000step \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/bo_gp \
    --n_trials 200 \
    --n_val_repeats 50 \
    --seed 42

echo "Done! $(date)"
