#!/bin/bash
#SBATCH --job-name=pdbbind_eda
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_pdbbind_eda.log
#SBATCH --error=slurm/logs/%j_pdbbind_eda.err

# ─────────────────────────────────────────────────────────────
# Stage 0d: EDA visualization and data quality analysis
#
# Run after merge+split is complete.
#
# Submit (with dependency):
#   MERGE_JOB_ID=12345
#   sbatch --dependency=afterok:${MERGE_JOB_ID} slurm/pipeline/s00d_eda.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PDBBIND_DIR="${PDBBIND_DIR:-data/pdbbind}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pdbbind_v2020}"
EDA_DIR="${EDA_DIR:-results/pdbbind_eda}"

mkdir -p slurm/logs "${EDA_DIR}"

echo "=== PDBbind v2020: EDA ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo ""

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

python scripts/pipeline/s00b_pdbbind_eda.py \
    --data_dir "${OUTPUT_DIR}" \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --output_dir "${EDA_DIR}"

echo ""
echo "EDA complete. Outputs in: ${EDA_DIR}"
ls -la "${EDA_DIR}/"
