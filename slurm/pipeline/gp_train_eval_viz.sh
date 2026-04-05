#!/bin/bash
#SBATCH --job-name=bd_gp_eval
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%j_gp_eval_1k.log
#SBATCH --error=slurm/logs/%j_gp_eval_1k.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Merge 88-pocket embeddings → GP Train/Val → Eval → Ablation → Viz
#
# Runs on existing 1000-step embeddings from shards.
# 88/93 pockets completed (5 timed out — proceeding with available data).
#
# Submit:
#   sbatch slurm/gp_train_eval_viz.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

OUTPUT_DIR="results/embedding_1000step/merged"
AFFINITY_PKL="external/targetdiff/data/affinity_info.pkl"

mkdir -p slurm/logs "${OUTPUT_DIR}/gp_model" "${OUTPUT_DIR}/evaluation" \
         "${OUTPUT_DIR}/ablation" "${OUTPUT_DIR}/figures"

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: GP Training + Evaluation + Visualization"
echo " (1000-step, 88/93 pockets)"
echo "═══════════════════════════════════════════════════════════"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ── Environment ──────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Merge all shard embeddings ───────────────────────
echo ">>> [Step 1] Merging embeddings from 4 shard directories..."
python3 scripts/scaling/s08_merge_and_train_eval.py \
    --step merge \
    --output_dir "${OUTPUT_DIR}"
echo ""

# ── Step 2: GP Training with train/val split ─────────────────
echo ">>> [Step 2] Training GP oracle (with train/val split)..."
python3 scripts/scaling/s08_merge_and_train_eval.py \
    --step train \
    --output_dir "${OUTPUT_DIR}" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --n_epochs 300 \
    --n_inducing 128 \
    --val_fraction 0.2 \
    --device cuda
echo ""

# ── Step 3: Evaluation ──────────────────────────────────────
echo ">>> [Step 3] Running evaluation..."
python3 scripts/pipeline/s05_evaluate.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_model "${OUTPUT_DIR}/gp_model/gp_model.pt" \
    --gp_train_data "${OUTPUT_DIR}/gp_model/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${OUTPUT_DIR}/evaluation" \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000
echo ""

# ── Step 4: Ablation study ──────────────────────────────────
echo ">>> [Step 4] Running ablation study..."
python3 scripts/pipeline/s06_ablation.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_model "${OUTPUT_DIR}/gp_model/gp_model.pt" \
    --gp_train_data "${OUTPUT_DIR}/gp_model/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${OUTPUT_DIR}/ablation" \
    --y_target 7.0 \
    --bootstrap_n 1000
echo ""

# ── Step 5: Comprehensive visualization ─────────────────────
echo ">>> [Step 5] Generating all figures (including GP training curves)..."
python3 scripts/scaling/s08_merge_and_train_eval.py \
    --step visualize \
    --output_dir "${OUTPUT_DIR}"
echo ""

# Also run the standard figure generation
python3 scripts/pipeline/s07_generate_figures.py \
    --eval_dir "${OUTPUT_DIR}/evaluation" \
    --ablation_dir "${OUTPUT_DIR}/ablation" \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_meta "${OUTPUT_DIR}/gp_model/train_meta.json" \
    --output "${OUTPUT_DIR}/figures"
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Pipeline Complete at $(date)"
echo "═══════════════════════════════════════════════════════════"

python3 -c "
import json, numpy as np
data = np.load('${OUTPUT_DIR}/all_embeddings.npz')
print(f'Pockets: {len(data.files)}')
print(f'Total molecules: {sum(data[k].shape[0] for k in data.files)}')
print(f'Embedding dim: {data[list(data.keys())[0]].shape[1]}')
print()
try:
    with open('${OUTPUT_DIR}/evaluation/eval_metrics.json') as f:
        m = json.load(f)
    print('--- Evaluation Metrics ---')
    for k in ['ece','auroc','ef_1pct','hit_rate','spearman_rho','rmse','nll']:
        v = m.get(k, 'N/A')
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
except Exception as e: print(f'  (metrics: {e})')
print()
try:
    with open('${OUTPUT_DIR}/ablation/ablation_summary.json') as f:
        d = json.load(f)
    print('--- Ablation Summary ---')
    for aid, r in d.items():
        desc = r.get('description', aid)[:28]
        print(f'  {desc:<30} AUROC={r.get(\"auroc\",0):.4f}  ECE={r.get(\"ece\",0):.4f}')
except Exception as e: print(f'  (ablation: {e})')
"

echo ""
echo "Output: ${OUTPUT_DIR}/"
echo "Figures: ${OUTPUT_DIR}/figures/"
ls -la "${OUTPUT_DIR}/figures/"*.png 2>/dev/null
echo ""
echo "Done!"
