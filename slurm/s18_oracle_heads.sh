#!/bin/bash
#SBATCH --job-name=s18_oracle
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/s18_oracle_%j.out
#SBATCH --error=slurm/logs/s18_oracle_%j.err

# Sub-Plan 4: Oracle Head Family Comparison (Tier 1)
# Estimated wall time: ~2.5 hours on L40S (6h limit for margin)

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

# Phase A.3: Extract frozen embeddings (if not already cached)
FROZEN_EMB="results/stage2/frozen_embeddings.npz"
if [ ! -f "$FROZEN_EMB" ]; then
    echo "=== Phase A.3: Extracting frozen embeddings ==="
    python scripts/pipeline/s18_train_oracle_heads.py \
        --extract_embeddings \
        --schemeb_checkpoint results/stage2/ablation_viz/A36_independent_model.pt \
        --model_type independent \
        --atom_emb_dir results/atom_embeddings \
        --labels data/pdbbind_v2020/labels.csv \
        --splits data/pdbbind_v2020/splits.json \
        --output results/stage2/oracle_heads \
        --device cuda
    FROZEN_EMB="results/stage2/oracle_heads/frozen_embeddings.npz"
fi

# Phase 4.2: Train all Tier 1 oracle heads
echo "=== Phase 4.2: Training Tier 1 Oracle Heads ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    --frozen_embeddings "$FROZEN_EMB" \
    --output results/stage2/oracle_heads \
    --heads dkl,dkl_ensemble,nn_residual,svgp,pca_svgp \
    --n_inducing 512 \
    --ensemble_members 5 \
    --n_epochs 300 \
    --batch_size 256 \
    --device cuda \
    --seed 42

echo "=== Done ==="
echo "Date: $(date)"
echo "Results: results/stage2/oracle_heads/tier1_comparison.json"
