#!/bin/bash
#SBATCH --job-name=bd_emb1k
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/%A_%a_emb1k.log
#SBATCH --error=slurm/logs/%A_%a_emb1k.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Multi-GPU 1000-step Sampling + Embedding Extraction
#
# 93 pockets split across 4 GPUs (array tasks).
# Each GPU samples 64 molecules × 1000 diffusion steps per pocket,
# extracts SE(3)-invariant embeddings, and saves per-shard results.
#
# Submit:
#   sbatch slurm/embedding_1000step_array.sh
#
# Then after all array tasks finish, run:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/merge_and_evaluate_1000step.sh
#
# Or use the wrapper:
#   bash slurm/submit_1000step_pipeline.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
POCKET_LIST="${POCKET_LIST:-data/splits/test_pockets.txt}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS="${NUM_STEPS:-1000}"
DEVICE="${DEVICE:-cuda}"
PDBBIND_DIR="${PDBBIND_DIR:-external/targetdiff/data/test_set}"
TARGETDIFF_DIR="${TARGETDIFF_DIR:-external/targetdiff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/embedding_1000step}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_j${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"

# Shard info from SLURM array
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-4}}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
SHARDS_DIR="${RUN_DIR}/shards"
SHARD_DIR="${SHARDS_DIR}/shard_${SHARD_INDEX}of${NUM_SHARDS}"

mkdir -p slurm/logs "${SHARD_DIR}"

# Save run tag for downstream jobs (only shard 0 writes this)
if [[ "${SHARD_INDEX}" == "0" ]]; then
    echo "${RUN_TAG}" > "${OUTPUT_ROOT}/latest_run_tag.txt"
    echo "${RUN_DIR}" > "${OUTPUT_ROOT}/latest_run_dir.txt"
    echo "${NUM_SHARDS}" > "${OUTPUT_ROOT}/latest_num_shards.txt"
fi

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: 1000-step Embedding Extraction (Shard ${SHARD_INDEX}/${NUM_SHARDS})"
echo "═══════════════════════════════════════════════════════════"
echo "Date:         $(date)"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array job id: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task id:      ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Run dir:      ${RUN_DIR}"
echo "Shard dir:    ${SHARD_DIR}"
echo "Pocket list:  ${POCKET_LIST}"
echo "Samples/pkt:  ${NUM_SAMPLES}"
echo "Steps:        ${NUM_STEPS}"
echo ""

# ── Environment ──────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run shard sampling ───────────────────────────────────────
CMD=(
  python scripts/08_sample_molecules_shard.py
  --pocket_list "${POCKET_LIST}"
  --num_shards "${NUM_SHARDS}"
  --shard_index "${SHARD_INDEX}"
  --pdbbind_dir "${PDBBIND_DIR}"
  --targetdiff_dir "${TARGETDIFF_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --num_steps "${NUM_STEPS}"
  --device "${DEVICE}"
  --output_dir "${SHARD_DIR}"
)

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Shard ${SHARD_INDEX}/${NUM_SHARDS} complete at $(date)"
echo " Output: ${SHARD_DIR}"
echo "═══════════════════════════════════════════════════════════"
