#!/bin/bash
#SBATCH --job-name=s18_tier1b
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/s18_tier1b_%j.out
#SBATCH --error=slurm/logs/s18_tier1b_%j.err

# Sub-Plan 04 Phase C': SNGP + Evidential Tier 1b baselines
# Uses same frozen embeddings as Tier 1 (A3.6 Independent, z ∈ ℝ^128)

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

EMB="results/stage2/oracle_heads/frozen_embeddings.npz"
OUT="results/stage2/oracle_heads"

echo "=== Tier 1b: SNGP ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    --frozen_embeddings "$EMB" \
    --output "$OUT" \
    --heads sngp \
    --seed 42 \
    --device cuda

echo ""
echo "=== Tier 1b: Evidential ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    --frozen_embeddings "$EMB" \
    --output "$OUT" \
    --heads evidential \
    --seed 42 \
    --device cuda

echo ""
echo "=== Tier 1b complete ==="
