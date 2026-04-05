#!/bin/bash
#SBATCH --job-name=bayesdiff_sample_array
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/%A_%a_sample.log
#SBATCH --error=slurm/logs/%A_%a_sample.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Multi-GPU batch sampling via Slurm job array
#
# Submit example (4 GPUs):
#   sbatch --array=0-3 slurm/sample_array_job.sh
#
# Key behavior:
#   - Each array task uses 1 GPU and handles a disjoint shard of pockets.
#   - Outputs are written to a unique run directory (timestamp + job id)
#     by default to avoid overwriting existing results.
#   - Original scripts/pipeline/s02_sample_molecules.py remains unchanged.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

POCKET_LIST="${POCKET_LIST:-data/splits/test_pockets.txt}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda}"
PDBBIND_DIR="${PDBBIND_DIR:-external/targetdiff/data/test_set}"
TARGETDIFF_DIR="${TARGETDIFF_DIR:-external/targetdiff}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/generated_molecules_parallel}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_j${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"

# Prefer Slurm-provided array metadata, fallback to env defaults.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
SHARDS_DIR="${RUN_DIR}/shards"
SHARD_DIR="${SHARDS_DIR}/shard_${SHARD_INDEX}of${NUM_SHARDS}"

mkdir -p slurm/logs "${SHARD_DIR}"

echo "=== BayesDiff Multi-GPU Sampling (Array Task) ==="
echo "Date:         $(date)"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Array job id: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task id:      ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Task count:   ${SLURM_ARRAY_TASK_COUNT:-N/A}"
echo "Run dir:      ${RUN_DIR}"
echo "Shard dir:    ${SHARD_DIR}"
echo "Shard:        ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Pocket list:  ${POCKET_LIST}"
echo "Samples/pkt:  ${NUM_SAMPLES}"
echo "Steps:        ${NUM_STEPS}"
echo ""

# Activate conda environment
# NOTE: keep this consistent with current NYU Torch setup.
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(
  python scripts/scaling/s01_sample_shard.py
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

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi

"${CMD[@]}"

echo ""
echo "Task complete: shard ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Run dir: ${RUN_DIR}"
echo "Next: submit merge job after all array tasks succeed."
