#!/bin/bash
# ─────────────────────────────────────────────────────────────
# PDBbind v2020 Full Preparation Pipeline
#
# One-command launch for the entire preparation pipeline:
#   1. Parse INDEX file
#   2. Featurize all complexes (50-way parallel array job)
#   3. Merge shards + create protein-family splits
#   4. Run EDA visualization
#
# Usage:
#   bash slurm/pipeline/s00_launch_all.sh
#
# Or with custom PDBbind path:
#   PDBBIND_DIR=/path/to/pdbbind bash slurm/pipeline/s00_launch_all.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PDBBIND_DIR="${PDBBIND_DIR:-data/pdbbind}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pdbbind_v2020}"

echo "=== PDBbind v2020 Full Preparation Pipeline ==="
echo "PDBbind dir: ${PDBBIND_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

export PDBBIND_DIR OUTPUT_DIR

# Step 1: Parse INDEX
echo "Submitting Step 1: Parse INDEX..."
PARSE_JOB=$(sbatch --parsable slurm/pipeline/s00a_parse.sh)
echo "  Job ID: ${PARSE_JOB}"

# Step 2: Featurize (50 parallel shards, depends on parse)
echo "Submitting Step 2: Featurize (50 shards)..."
FEAT_JOB=$(sbatch --parsable --dependency=afterok:${PARSE_JOB} slurm/pipeline/s00b_featurize_array.sh)
echo "  Job ID: ${FEAT_JOB}"

# Step 3: Merge + Split (depends on all featurize shards)
echo "Submitting Step 3: Merge + Split..."
MERGE_JOB=$(sbatch --parsable --dependency=afterok:${FEAT_JOB} slurm/pipeline/s00c_merge_split.sh)
echo "  Job ID: ${MERGE_JOB}"

# Step 4: EDA (depends on merge)
echo "Submitting Step 4: EDA..."
EDA_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_JOB} slurm/pipeline/s00d_eda.sh)
echo "  Job ID: ${EDA_JOB}"

echo ""
echo "=== All jobs submitted ==="
echo "  Parse:     ${PARSE_JOB}"
echo "  Featurize: ${FEAT_JOB} (array 0-49)"
echo "  Merge:     ${MERGE_JOB}"
echo "  EDA:       ${EDA_JOB}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel ${PARSE_JOB} ${FEAT_JOB} ${MERGE_JOB} ${EDA_JOB}"
