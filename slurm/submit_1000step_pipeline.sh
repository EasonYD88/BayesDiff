#!/bin/bash
# ─────────────────────────────────────────────────────────────
# BayesDiff: Submit 1000-step multi-GPU embedding pipeline
#
# This script:
#   1. Submits a 4-GPU array job for sampling (1000 steps, 64 samples/pocket)
#   2. Submits a dependent merge+GP+evaluation job
#
# Usage:
#   cd /scratch/yd2915/BayesDiff
#   bash slurm/submit_1000step_pipeline.sh
#
# Optional overrides:
#   NUM_SAMPLES=128 NUM_STEPS=1000 bash slurm/submit_1000step_pipeline.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults (can be overridden via environment)
export NUM_SAMPLES="${NUM_SAMPLES:-64}"
export NUM_STEPS="${NUM_STEPS:-1000}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/embedding_1000step}"

mkdir -p slurm/logs "${OUTPUT_ROOT}"

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: 1000-step Multi-GPU Pipeline Submission"
echo "═══════════════════════════════════════════════════════════"
echo "  Pockets:        93 (data/splits/test_pockets.txt)"
echo "  Samples/pocket: ${NUM_SAMPLES}"
echo "  Diffusion steps:${NUM_STEPS}"
echo "  GPUs:           4 (array 0-3)"
echo "  Output:         ${OUTPUT_ROOT}"
echo ""

# Step 1: Submit array sampling job
echo ">>> Submitting 4-GPU sampling array job..."
ARRAY_JOB=$(sbatch \
    --export=ALL \
    --parsable \
    slurm/embedding_1000step_array.sh)

echo "    Array job submitted: ${ARRAY_JOB}"
echo ""

# Step 2: Submit merge+evaluate job (depends on array completion)
echo ">>> Submitting merge + evaluation job (dependency: afterok:${ARRAY_JOB})..."
EVAL_JOB=$(sbatch \
    --dependency=afterok:${ARRAY_JOB} \
    --export=ALL \
    --parsable \
    slurm/merge_and_evaluate_1000step.sh)

echo "    Eval job submitted: ${EVAL_JOB}"
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════"
echo " Jobs submitted successfully!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Sampling (4 GPUs):  ${ARRAY_JOB}_[0-3]"
echo "  Merge+Evaluate:     ${EVAL_JOB} (waits for ${ARRAY_JOB})"
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f slurm/logs/${ARRAY_JOB}_*_emb1k.log"
echo ""
echo "  Expected timeline:"
echo "    Sampling:  ~8-16 hours (93 pockets × 64 mols × 1000 steps / 4 GPUs)"
echo "    Merge+GP:  ~1-2 hours"
echo ""
echo "  Results will be in: ${OUTPUT_ROOT}/"
