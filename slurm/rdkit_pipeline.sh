#!/bin/bash
#SBATCH --job-name=bayesdiff_rdkit_pipeline
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j_rdkit_pipeline.log
#SBATCH --error=slurm/logs/%j_rdkit_pipeline.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Full pipeline on ECFP4 fingerprint embeddings
#   Step 1: Train GP (SVGP, 200 epochs)
#   Step 2: Evaluate (ECE, AUROC, EF@1%, Spearman, NLL, RMSE)
#   Step 3: Ablation (7 variants: full, A1-A5, A7)
#   Step 4: Visualize (6 publication figures)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Activate environment ─────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

WORKDIR=/scratch/yd2915/BayesDiff
cd "${WORKDIR}"

EMB=results/embedding_rdkit/all_embeddings.npz
AFF=external/targetdiff/data/affinity_info.pkl
OUT=results/embedding_rdkit

echo "=== BayesDiff ECFP4 Pipeline ==="
echo "Date   : $(date)"
echo "Host   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python : $(which python)"
echo "Embeddings: ${EMB}"
echo "================================="

# ── Step 1: Train GP ─────────────────────────────────────────
echo ""
echo ">>> Step 1/4: Training GP..."
python scripts/04_train_gp.py \
    --embeddings "${EMB}" \
    --affinity_pkl "${AFF}" \
    --output "${OUT}/gp_model" \
    --n_inducing 48 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200 \
    --device auto \
    --seed 42

echo ">>> GP training complete."
ls -lh "${OUT}/gp_model/"

# ── Step 2: Evaluate ─────────────────────────────────────────
echo ""
echo ">>> Step 2/4: Evaluating..."
python scripts/05_evaluate.py \
    --embeddings "${EMB}" \
    --gp_model "${OUT}/gp_model/gp_model.pt" \
    --gp_train_data "${OUT}/gp_model/train_data.npz" \
    --affinity_pkl "${AFF}" \
    --output "${OUT}/evaluation" \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000

echo ">>> Evaluation complete."
cat "${OUT}/evaluation/eval_metrics.json" | python -m json.tool

# ── Step 3: Ablation ─────────────────────────────────────────
echo ""
echo ">>> Step 3/4: Running ablation study..."
python scripts/06_ablation.py \
    --embeddings "${EMB}" \
    --gp_model "${OUT}/gp_model/gp_model.pt" \
    --gp_train_data "${OUT}/gp_model/train_data.npz" \
    --affinity_pkl "${AFF}" \
    --output "${OUT}/ablation" \
    --y_target 7.0 \
    --bootstrap_n 1000

echo ">>> Ablation complete."
cat "${OUT}/ablation/ablation_summary.json" | python -m json.tool

# ── Step 4: Visualize ────────────────────────────────────────
echo ""
echo ">>> Step 4/4: Generating figures..."
python scripts/09_generate_figures.py \
    --eval_dir "${OUT}/evaluation" \
    --ablation_dir "${OUT}/ablation" \
    --embeddings "${EMB}" \
    --gp_meta "${OUT}/gp_model/train_meta.json" \
    --output "${OUT}/figures"

echo ">>> Figures generated."
ls -lh "${OUT}/figures/"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Pipeline complete! Results in: ${OUT}/"
echo "  gp_model/    → gp_model.pt, train_meta.json"
echo "  evaluation/  → eval_metrics.json"
echo "  ablation/    → ablation_summary.json"
echo "  figures/     → fig1-fig6 PNG files"
echo "=============================================="
echo "Finished at: $(date)"
