#!/bin/bash
#SBATCH --job-name=bayesdiff_sample
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j_sample.log
#SBATCH --error=slurm/logs/%j_sample.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Batch molecular sampling on HPC
#
# Submit:
#   sbatch slurm/sample_job.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm/logs/<jobid>_sample.log
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
POCKET_LIST="${POCKET_LIST:-data/splits/test_pockets.txt}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda}"
# PDBBIND_DIR supports both PDBbind layout (refined-set/{code}/{code}_pocket.pdb)
# and TargetDiff test_set layout ({target}/{*_rec.pdb}). For open data:
PDBBIND_DIR="${PDBBIND_DIR:-external/targetdiff/data/test_set}"
TARGETDIFF_DIR="${TARGETDIFF_DIR:-external/targetdiff}"
OUTPUT_DIR="${OUTPUT_DIR:-results/generated_molecules}"

# ── Environment ──────────────────────────────────────────────
echo "=== BayesDiff Sampling Job ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Pocket list: ${POCKET_LIST}"
echo "Samples/pkt: ${NUM_SAMPLES}"
echo "Output:      ${OUTPUT_DIR}"
echo ""

# Activate conda environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

mkdir -p slurm/logs "${OUTPUT_DIR}"

# ── Step 1: Sample molecules ────────────────────────────────
echo ">>> Step 1/2: Sampling molecules..."
python scripts/02_sample_molecules.py \
    --pocket_list "${POCKET_LIST}" \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --targetdiff_dir "${TARGETDIFF_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_steps "${NUM_STEPS}" \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}"

# ── Step 2: Extract embeddings (optional) ───────────────────
# Note: 02_sample_molecules.py already saves per-pocket and combined
# embeddings via sample_and_embed(). This step is useful when:
#   - Re-extracting embeddings from existing SDF files
#   - Using a different encoder/checkpoint
echo ""
echo ">>> Step 2/2: Re-extracting embeddings (from generated SDFs)..."
python scripts/03_extract_embeddings.py \
    --mode generated \
    --input_dir "${OUTPUT_DIR}" \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --targetdiff_dir "${TARGETDIFF_DIR}" \
    --device "${DEVICE}" \
    --output "${OUTPUT_DIR}/all_embeddings_reextracted.npz"

echo ""
echo "=== Done! ==="
echo "End time: $(date)"
