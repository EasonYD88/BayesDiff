#!/bin/bash
#SBATCH --job-name=s19_oracle_diag
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/s19_diag_%j.out
#SBATCH --error=slurm/logs/s19_diag_%j.err

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s19_oracle_diagnostics.py \
    --results_dir results/stage2/oracle_heads \
    --frozen_embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
    --output_dir results/stage2/oracle_heads/figures \
    --format pdf \
    --device cuda

# Also generate PNG for quick viewing
python scripts/pipeline/s19_oracle_diagnostics.py \
    --results_dir results/stage2/oracle_heads \
    --frozen_embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
    --output_dir results/stage2/oracle_heads/figures \
    --format png \
    --device cuda

echo ""
echo "=== Done ==="
echo "Date: $(date)"
