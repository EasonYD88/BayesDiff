#!/bin/bash
#SBATCH --job-name=sp05-tier1
#SBATCH --output=slurm/logs/sp05_tier1_%j.out
#SBATCH --error=slurm/logs/sp05_tier1_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --account=torch_pr_281_general

set -euo pipefail

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

EMB="results/stage2/oracle_heads/frozen_embeddings_augmented.npz"
OUT="results/stage2/multitask_trunk"

mkdir -p slurm/logs "$OUT"

# ── Step 0: Augment NPZ if not already done ──────────────────────
if [ ! -f "$EMB" ]; then
    echo "=== Augmenting frozen_embeddings.npz ==="
    python scripts/pipeline/s20_augment_npz.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --splits data/pdbbind_v2020/splits.json \
        --clusters data/pdbbind_v2020/cluster_assignments.csv \
        --output "$EMB"
fi

# ── 3-seed protocol: 42, 123, 777 ────────────────────────────────
for SEED in 42 123 777; do
    echo ""
    echo "================================================================"
    echo "=== SEED=$SEED ==="
    echo "================================================================"

    echo "=== A5.0: True SP4 baseline (no trunk) ==="
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings "$EMB" --output "$OUT/A5.0_no_trunk" \
        --phase v1 --no_trunk --seed $SEED --device cuda

    echo "=== A5.1: Regression-only trunk ==="
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings "$EMB" --output "$OUT/A5.1_reg_only" \
        --phase v1 --lambda_cls 0.0 --lambda_rank 0.0 \
        --n_epochs 200 --seed $SEED --device cuda

    echo "=== A5.2: Reg + Cls trunk (primary hypothesis) ==="
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings "$EMB" --output "$OUT/A5.2_reg_cls" \
        --phase v1 --lambda_cls 0.5 --lambda_rank 0.0 \
        --n_epochs 200 --seed $SEED --device cuda
done

echo ""
echo "=== Tier 1 Evaluation ==="
python scripts/pipeline/s21_evaluate_multitask.py \
    --results_dir "$OUT" \
    --embeddings "$EMB" \
    --output "$OUT/figures"

echo "=== Tier 1 complete ==="
