#!/bin/bash
#SBATCH --job-name=bayesdiff_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%j_full.log
#SBATCH --error=slurm/logs/%j_full.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Full HPC Pipeline (S2–S7)
#
# Runs sampling → GP training → evaluation → ablation in one job.
#
# Submit:
#   sbatch slurm/full_pipeline_job.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm/logs/<jobid>_full.log
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda}"
PDBBIND_DIR="${PDBBIND_DIR:-external/targetdiff/data/test_set}"
TARGETDIFF_DIR="${TARGETDIFF_DIR:-external/targetdiff}"
AFFINITY_PKL="${AFFINITY_PKL:-external/targetdiff/data/affinity_info.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/generated_molecules}"
GP_OUTPUT="${GP_OUTPUT:-results/gp_model}"
EVAL_OUTPUT="${EVAL_OUTPUT:-results/evaluation}"
ABLATION_OUTPUT="${ABLATION_OUTPUT:-results/ablation}"

# ── Environment ──────────────────────────────────────────────
echo "=== BayesDiff Full Pipeline ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Samples/pkt: ${NUM_SAMPLES}"
echo "Steps:       ${NUM_STEPS}"
echo ""

# Activate conda environment (adjust for your HPC)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate bayesdiff 2>/dev/null || source activate bayesdiff 2>/dev/null || true

mkdir -p slurm/logs data/splits "${OUTPUT_DIR}" "${GP_OUTPUT}" "${EVAL_OUTPUT}" "${ABLATION_OUTPUT}"

# ── S2: Generate pocket list ────────────────────────────────
echo ">>> [S2] Preparing pocket list..."
ls "${PDBBIND_DIR}" > data/splits/test_pockets.txt
POCKET_COUNT=$(wc -l < data/splits/test_pockets.txt)
echo "    Pockets: ${POCKET_COUNT}"

# ── S3: Batch sampling ──────────────────────────────────────
echo ""
echo ">>> [S3] Sampling molecules (${POCKET_COUNT} pockets × ${NUM_SAMPLES} samples × ${NUM_STEPS} steps)..."
echo "    Start: $(date)"
python scripts/pipeline/s02_sample_molecules.py \
    --pocket_list data/splits/test_pockets.txt \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --targetdiff_dir "${TARGETDIFF_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_steps "${NUM_STEPS}" \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}"
echo "    End:   $(date)"

# ── S5: Train GP ────────────────────────────────────────────
echo ""
echo ">>> [S5] Training GP oracle..."
python scripts/pipeline/s04_train_gp.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${GP_OUTPUT}" \
    --n_inducing 128 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200

# ── S6: Evaluation ──────────────────────────────────────────
echo ""
echo ">>> [S6] Running evaluation..."
python scripts/pipeline/s05_evaluate.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_model "${GP_OUTPUT}/gp_model.pt" \
    --gp_train_data "${GP_OUTPUT}/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${EVAL_OUTPUT}" \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000

# ── S7: Ablation study ─────────────────────────────────────
echo ""
echo ">>> [S7] Running ablation study..."
python scripts/pipeline/s06_ablation.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_model "${GP_OUTPUT}/gp_model.pt" \
    --gp_train_data "${GP_OUTPUT}/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${ABLATION_OUTPUT}" \
    --y_target 7.0 \
    --bootstrap_n 1000

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "=== BayesDiff Full Pipeline Complete! ===="
echo "=========================================="
echo "End time: $(date)"
echo ""

echo "--- Sampling Summary ---"
python -c "
import json
with open('${OUTPUT_DIR}/sampling_summary.json') as f:
    s = json.load(f)
n_ok = sum(1 for t in s.get('timing', []) if 'error' not in t)
print(f'  Sampled: {n_ok}/{s.get(\"n_pockets\", \"?\")} pockets')
print(f'  Samples/pocket: ${NUM_SAMPLES}')
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
    desc = r.get('description', aid)[:25]
    print(f'  {desc:<28} AUROC={r.get(\"auroc\",0):.4f}  ECE={r.get(\"ece\",0):.4f}  RMSE={r.get(\"rmse\",0):.4f}')
" 2>/dev/null || echo "  (ablation not available)"

echo ""
echo "Output files:"
ls -la "${EVAL_OUTPUT}/" 2>/dev/null
ls -la "${ABLATION_OUTPUT}/" 2>/dev/null
echo ""
echo "Done!"
