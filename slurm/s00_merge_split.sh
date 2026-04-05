#!/bin/bash
#SBATCH --job-name=s00_split
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/s00_split_%j.out
#SBATCH --error=slurm/logs/s00_split_%j.err

# PDBbind v2020 R1: merge shard results + cluster-stratified split

set -euo pipefail

cd /scratch/yd2915/BayesDiff
export PATH="/scratch/yd2915/BayesDiff/tools/mmseqs/bin:$PATH"

PYTHON=/scratch/yd2915/conda_envs/bayesdiff/bin/python
PDBBIND_DIR=data/PDBbind_data_set
OUTPUT_DIR=data/pdbbind_v2020

echo "=== Merge + Split on $(hostname) ==="
echo "Start: $(date)"

# Stage 3: Merge shard status files
$PYTHON scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir $PDBBIND_DIR \
    --output_dir $OUTPUT_DIR \
    --stage merge

# Stage 4: Cluster-stratified split (uses mmseqs2)
$PYTHON scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir $PDBBIND_DIR \
    --output_dir $OUTPUT_DIR \
    --stage split \
    --val_frac 0.12

echo "End: $(date)"
