#!/bin/bash
#SBATCH --job-name=bd_merge_eval
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%j_merge_eval_1k.log
#SBATCH --error=slurm/logs/%j_merge_eval_1k.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Merge shard outputs + GP train + Evaluation
#
# Run after all array tasks from embedding_1000step_array.sh finish.
# Automatically reads the latest run directory, or override via env.
#
# Submit with dependency:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/merge_and_evaluate_1000step.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-results/embedding_1000step}"
AFFINITY_PKL="${AFFINITY_PKL:-external/targetdiff/data/affinity_info.pkl}"

# Auto-detect latest run if not specified
if [[ -z "${RUN_DIR:-}" ]]; then
    if [[ -f "${OUTPUT_ROOT}/latest_run_dir.txt" ]]; then
        RUN_DIR="$(cat ${OUTPUT_ROOT}/latest_run_dir.txt)"
    else
        echo "ERROR: RUN_DIR not set and no latest_run_dir.txt found."
        exit 1
    fi
fi

NUM_SHARDS="${NUM_SHARDS:-}"
if [[ -z "${NUM_SHARDS}" ]] && [[ -f "${OUTPUT_ROOT}/latest_num_shards.txt" ]]; then
    NUM_SHARDS="$(cat ${OUTPUT_ROOT}/latest_num_shards.txt)"
fi

SHARDS_DIR="${RUN_DIR}/shards"
GP_OUTPUT="${RUN_DIR}/gp_model"
EVAL_OUTPUT="${RUN_DIR}/evaluation"
ABLATION_OUTPUT="${RUN_DIR}/ablation"

mkdir -p slurm/logs "${GP_OUTPUT}" "${EVAL_OUTPUT}" "${ABLATION_OUTPUT}"

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: Merge + GP Training + Evaluation"
echo "═══════════════════════════════════════════════════════════"
echo "Date:            $(date)"
echo "Node:            $(hostname)"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Run dir:         ${RUN_DIR}"
echo "Shards dir:      ${SHARDS_DIR}"
echo "Expected shards: ${NUM_SHARDS:-auto}"
echo ""

# ── Environment ──────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Merge shards ────────────────────────────────────
echo ">>> [Step 1] Merging shard outputs..."
MERGE_CMD=(
    python scripts/scaling/s02_merge_shards.py
    --shards_dir "${SHARDS_DIR}"
    --output_dir "${RUN_DIR}"
)
if [[ -n "${NUM_SHARDS}" ]]; then
    MERGE_CMD+=(--expected_shards "${NUM_SHARDS}")
fi
"${MERGE_CMD[@]}"
echo "    Merge complete."
echo ""

# ── Step 2: Train GP Oracle ─────────────────────────────────
echo ">>> [Step 2] Training GP oracle..."
python scripts/pipeline/s04_train_gp.py \
    --embeddings "${RUN_DIR}/all_embeddings.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${GP_OUTPUT}" \
    --n_inducing 128 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200
echo "    GP training complete."
echo ""

# ── Step 3: Evaluation ──────────────────────────────────────
echo ">>> [Step 3] Running evaluation..."
python scripts/pipeline/s05_evaluate.py \
    --embeddings "${RUN_DIR}/all_embeddings.npz" \
    --gp_model "${GP_OUTPUT}/gp_model.pt" \
    --gp_train_data "${GP_OUTPUT}/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${EVAL_OUTPUT}" \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000
echo "    Evaluation complete."
echo ""

# ── Step 4: Ablation study ──────────────────────────────────
echo ">>> [Step 4] Running ablation study..."
python scripts/pipeline/s06_ablation.py \
    --embeddings "${RUN_DIR}/all_embeddings.npz" \
    --gp_model "${GP_OUTPUT}/gp_model.pt" \
    --gp_train_data "${GP_OUTPUT}/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${ABLATION_OUTPUT}" \
    --y_target 7.0 \
    --bootstrap_n 1000
echo "    Ablation complete."
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Pipeline Complete at $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "--- Embedding Summary ---"
python -c "
import numpy as np
data = np.load('${RUN_DIR}/all_embeddings.npz')
n_pockets = len(data.files)
shapes = [data[k].shape for k in data.files]
total_mols = sum(s[0] for s in shapes)
dim = shapes[0][1] if shapes else 0
print(f'  Pockets: {n_pockets}')
print(f'  Total molecules: {total_mols}')
print(f'  Embedding dim: {dim}')
print(f'  Samples per pocket: {shapes[0][0] if shapes else 0}')
" 2>/dev/null || echo "  (summary not available)"

echo ""
echo "--- Evaluation Metrics ---"
python -c "
import json
with open('${EVAL_OUTPUT}/eval_metrics.json') as f:
    m = json.load(f)
for k in ['ece','auroc','ef_1pct','hit_rate','spearman_rho','rmse','nll']:
    v = m.get(k, 'N/A')
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
" 2>/dev/null || echo "  (metrics not available)"

echo ""
echo "--- Ablation Summary ---"
python -c "
import json
with open('${ABLATION_OUTPUT}/ablation_summary.json') as f:
    d = json.load(f)
for aid, r in d.items():
    desc = r.get('description', aid)[:28]
    print(f'  {desc:<30} AUROC={r.get(\"auroc\",0):.4f}  ECE={r.get(\"ece\",0):.4f}')
" 2>/dev/null || echo "  (ablation not available)"

echo ""
echo "Output: ${RUN_DIR}/"
echo "Done!"
