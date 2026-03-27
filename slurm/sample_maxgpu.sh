#!/bin/bash
#SBATCH --job-name=bd_emb50
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-30
#SBATCH --output=slurm/logs/%A_%a_emb50.log
#SBATCH --error=slurm/logs/%A_%a_emb50.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: 31-GPU array — 93 pockets × 50 molecules × 1000 steps
#
# Each array task handles ceil(93/31) = 3 pockets in parallel.
# Estimated time: ~3 × 12 min = ~36 min per task.
#
# Submit:
#   cd /scratch/yd2915/BayesDiff.worktrees/copilot-worktree-2026-03-27T12-00-01
#   sbatch slurm/sample_maxgpu.sh
#
# After all tasks finish, merge and evaluate:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/merge_maxgpu.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

WORKTREE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKTREE_DIR}"

# ── Configuration ─────────────────────────────────────────────
POCKET_LIST="${POCKET_LIST:-data/splits/test_pockets.txt}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_STEPS="${NUM_STEPS:-1000}"
DEVICE="${DEVICE:-cuda}"
PDBBIND_DIR="${PDBBIND_DIR:-external/targetdiff/data/test_set}"
TARGETDIFF_DIR="${TARGETDIFF_DIR:-external/targetdiff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/embedding_50mol}"

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-31}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_j${SLURM_ARRAY_JOB_ID:-local}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
SHARD_DIR="${RUN_DIR}/shards/shard_${SHARD_INDEX}of${NUM_SHARDS}"

mkdir -p slurm/logs "${SHARD_DIR}"

# Shard 0 records the run tag for downstream merge job
if [[ "${SHARD_INDEX}" == "0" ]]; then
    mkdir -p "${OUTPUT_ROOT}"
    echo "${RUN_TAG}"    > "${OUTPUT_ROOT}/latest_run_tag.txt"
    echo "${RUN_DIR}"    > "${OUTPUT_ROOT}/latest_run_dir.txt"
    echo "${NUM_SHARDS}" > "${OUTPUT_ROOT}/latest_num_shards.txt"
fi

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: 50-mol Embedding (Shard ${SHARD_INDEX}/${NUM_SHARDS})"
echo "═══════════════════════════════════════════════════════════"
echo "Date:         $(date)"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array job id: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task id:      ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Worktree:     ${WORKTREE_DIR}"
echo "Run dir:      ${RUN_DIR}"
echo "Shard dir:    ${SHARD_DIR}"
echo "Pocket list:  ${POCKET_LIST}"
echo "Samples/pkt:  ${NUM_SAMPLES}"
echo "Steps:        ${NUM_STEPS}"
echo ""

# ── Environment ───────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run shard ─────────────────────────────────────────────────
CMD=(
  python scripts/08_sample_molecules_shard.py
  --pocket_list  "${POCKET_LIST}"
  --num_shards   "${NUM_SHARDS}"
  --shard_index  "${SHARD_INDEX}"
  --pdbbind_dir  "${PDBBIND_DIR}"
  --targetdiff_dir "${TARGETDIFF_DIR}"
  --num_samples  "${NUM_SAMPLES}"
  --num_steps    "${NUM_STEPS}"
  --device       "${DEVICE}"
  --output_dir   "${SHARD_DIR}"
)

echo "Command: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Shard ${SHARD_INDEX}/${NUM_SHARDS} done at $(date)"
echo " Output: ${SHARD_DIR}"
echo "═══════════════════════════════════════════════════════════"
