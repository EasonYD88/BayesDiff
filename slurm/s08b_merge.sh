#!/bin/bash
#SBATCH --job-name=multilayer_merge
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/multilayer_merge_%j.out
#SBATCH --error=slurm/logs/multilayer_merge_%j.err

# Merge multi-layer embedding shards after s08b array job completes.
# Submit with: sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/s08b_merge.sh

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

python scripts/pipeline/s08b_extract_multilayer.py \
    --output_dir results/multilayer_embeddings \
    --stage merge

echo "Merge complete at $(date)"
