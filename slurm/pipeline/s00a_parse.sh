#!/bin/bash
#SBATCH --job-name=pdbbind_parse
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j_pdbbind_parse.log
#SBATCH --error=slurm/logs/%j_pdbbind_parse.err

# ─────────────────────────────────────────────────────────────
# Stage 0a: Parse PDBbind INDEX file and create labels.csv
# This must run before the featurize array job.
#
# Submit:
#   sbatch slurm/pipeline/s00a_parse.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PDBBIND_DIR="${PDBBIND_DIR:-data/pdbbind}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pdbbind_v2020}"

mkdir -p slurm/logs "${OUTPUT_DIR}"

echo "=== PDBbind v2020: Parse INDEX ==="
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "PDBbind dir: ${PDBBIND_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

python scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --stage parse

echo "Parse complete. Submit featurize array job next."
