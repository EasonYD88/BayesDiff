#!/bin/bash
#SBATCH --job-name=sp05-tier2
#SBATCH --output=slurm/logs/sp05_tier2_%j.out
#SBATCH --error=slurm/logs/sp05_tier2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_281_general

set -euo pipefail

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

EMB="results/stage2/oracle_heads/frozen_embeddings_augmented.npz"
OUT="results/stage2/multitask_trunk"

mkdir -p slurm/logs "$OUT"

# Augment NPZ if needed
if [ ! -f "$EMB" ]; then
    echo "=== Augmenting frozen_embeddings.npz ==="
    python scripts/pipeline/s20_augment_npz.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --splits data/pdbbind_v2020/splits.json \
        --clusters data/pdbbind_v2020/cluster_assignments.csv \
        --output "$EMB"
fi

# ── 3-seed protocol ──────────────────────────────────────────────
for SEED in 42 123 777; do
    echo ""
    echo "================================================================"
    echo "=== SEED=$SEED ==="
    echo "================================================================"

    echo "=== A5.3: Reg + Rank (no cls) ==="
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings "$EMB" --output "$OUT/A5.3_reg_rank" \
        --phase v2 --lambda_cls 0.0 --lambda_rank 0.3 \
        --n_epochs 200 --seed $SEED --device cuda

    echo "=== A5.4: Reg + Cls + Rank (full three-task) ==="
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings "$EMB" --output "$OUT/A5.4_reg_cls_rank" \
        --phase v2 --lambda_cls 0.5 --lambda_rank 0.3 \
        --n_epochs 200 --seed $SEED --device cuda

    echo "=== A5.5: Learned weights ==="
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings "$EMB" --output "$OUT/A5.5_learned_weights" \
        --phase v2 --lambda_cls 0.5 --lambda_rank 0.3 \
        --learned_weights --n_epochs 200 --seed $SEED --device cuda
done

echo ""
echo "=== Tier 2 Evaluation ==="
python scripts/pipeline/s21_evaluate_multitask.py \
    --results_dir "$OUT" \
    --embeddings "$EMB" \
    --output "$OUT/figures"

echo "=== Tier 2 complete ==="
