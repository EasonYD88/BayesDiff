#!/bin/bash
#SBATCH --job-name=s18_tier2
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/s18_tier2_%j.out
#SBATCH --error=slurm/logs/s18_tier2_%j.err

# Sub-Plan 4 Tier 2: DKL Ensemble Ablations (A4.8–A4.15)
# Winner from Tier 1: DKL Ensemble (ρ=0.781, ρ_err_σ=0.088–0.144)
# Estimated wall time: ~3 hours on L40S

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

FROZEN_EMB="results/stage2/oracle_heads/frozen_embeddings.npz"
TIER2_DIR="results/stage2/oracle_heads/tier2"
COMMON="--frozen_embeddings $FROZEN_EMB --n_epochs 300 --batch_size 256 --device cuda"

# ====================================================================
# Tier 1 winner baseline (DKL Ensemble M=5, d_u=32, residual=True, bootstrap=True)
# Already have this from Tier 1, but re-run with 3 seeds for variance estimation
# ====================================================================
echo "=== A4.0: Baseline (multi-seed) ==="
for SEED in 42 123 777; do
    echo "--- Seed $SEED ---"
    python scripts/pipeline/s18_train_oracle_heads.py \
        $COMMON \
        --output ${TIER2_DIR}/baseline_seed${SEED} \
        --heads dkl_ensemble \
        --ensemble_members 5 \
        --feature_dim 32 \
        --n_inducing 512 \
        --seed $SEED
done

# ====================================================================
# A4.8: DKL Ensemble residual=False
# ====================================================================
echo "=== A4.8: residual=False ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_8_no_residual \
    --heads dkl_ensemble \
    --ensemble_members 5 \
    --feature_dim 32 \
    --residual 0 \
    --seed 42

# ====================================================================
# A4.9: DKL Ensemble d_u=16
# ====================================================================
echo "=== A4.9: feature_dim=16 ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_9_fd16 \
    --heads dkl_ensemble \
    --ensemble_members 5 \
    --feature_dim 16 \
    --seed 42

# ====================================================================
# A4.10: DKL Ensemble d_u=64
# ====================================================================
echo "=== A4.10: feature_dim=64 ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_10_fd64 \
    --heads dkl_ensemble \
    --ensemble_members 5 \
    --feature_dim 64 \
    --seed 42

# ====================================================================
# A4.11: DKL Ensemble M=3
# ====================================================================
echo "=== A4.11: n_members=3 ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_11_M3 \
    --heads dkl_ensemble \
    --ensemble_members 3 \
    --feature_dim 32 \
    --seed 42

# ====================================================================
# A4.12: DKL Ensemble seed-only (no bootstrap)
# ====================================================================
echo "=== A4.12: bootstrap=False ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_12_no_bootstrap \
    --heads dkl_ensemble \
    --ensemble_members 5 \
    --feature_dim 32 \
    --bootstrap 0 \
    --seed 42

# ====================================================================
# A4.13: DKL Ensemble 3-layer FeatureExtractor
# ====================================================================
echo "=== A4.13: n_layers=3 ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_13_3layer \
    --heads dkl_ensemble \
    --ensemble_members 5 \
    --feature_dim 32 \
    --dkl_n_layers 3 \
    --seed 42

# ====================================================================
# A4.14: DKL Ensemble n_inducing=1024
# ====================================================================
echo "=== A4.14: n_inducing=1024 ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_14_ind1024 \
    --heads dkl_ensemble \
    --ensemble_members 5 \
    --feature_dim 32 \
    --n_inducing 1024 \
    --seed 42

# ====================================================================
# A4.15: NNResidual with MC Dropout (secondary candidate)
# ====================================================================
echo "=== A4.15: NNResidual MC Dropout ==="
python scripts/pipeline/s18_train_oracle_heads.py \
    $COMMON \
    --output ${TIER2_DIR}/A4_15_nn_mcdropout \
    --heads nn_residual \
    --mc_dropout 1 \
    --mc_samples 20 \
    --seed 42

# ====================================================================
# Aggregate all Tier 2 results
# ====================================================================
echo ""
echo "=== Tier 2 Summary ==="
echo "Results in: ${TIER2_DIR}/"
for d in ${TIER2_DIR}/*/; do
    name=$(basename "$d")
    json="${d}tier1_comparison.json"
    if [ -f "$json" ]; then
        echo "--- $name ---"
        python -c "
import json
with open('$json') as f:
    data = json.load(f)
for head, res in data.items():
    t = res['test']
    print(f\"  {head}: R2={t['R2']:.4f} rho={t['spearman_rho']:.4f} err_sigma_rho={t['err_sigma_rho']:.4f} NLL={t['nll']:.4f}\")
"
    fi
done

echo ""
echo "=== Done ==="
echo "Date: $(date)"
