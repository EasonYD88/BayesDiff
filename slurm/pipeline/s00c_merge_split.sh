#!/bin/bash
#SBATCH --job-name=pdbbind_merge
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j_pdbbind_merge.log
#SBATCH --error=slurm/logs/%j_pdbbind_merge.err

# ─────────────────────────────────────────────────────────────
# Stage 0c: Merge shard results and create protein-family splits
#
# Run after all featurize array tasks complete.
#
# Submit (with dependency on featurize job):
#   FEAT_JOB_ID=12345  # from sbatch output of s00b
#   sbatch --dependency=afterok:${FEAT_JOB_ID} slurm/pipeline/s00c_merge_split.sh
#
# Or standalone:
#   sbatch slurm/pipeline/s00c_merge_split.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PDBBIND_DIR="${PDBBIND_DIR:-data/pdbbind}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pdbbind_v2020}"
SEED="${SEED:-42}"

mkdir -p slurm/logs

echo "=== PDBbind v2020: Merge + Split ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "PDBbind dir: ${PDBBIND_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

# Merge shard status files
python scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --stage merge

# Create protein-family splits (uses mmseqs2 if available)
python scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --stage split \
    --seed "${SEED}"

echo ""
echo "=== Merge + Split complete ==="
echo "Output files:"
ls -la "${OUTPUT_DIR}"/*.{csv,json,txt} 2>/dev/null || true
echo ""
echo "Processed .pt files:"
ls "${OUTPUT_DIR}/processed/" | wc -l
echo ""
echo "Next: run EDA with s00d_eda.sh"
