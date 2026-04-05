#!/bin/bash
#SBATCH --job-name=s00_feat
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm/logs/s00_feat_%A_%a.out
#SBATCH --error=slurm/logs/s00_feat_%A_%a.err

# PDBbind v2020 R1 featurization — 100-shard array job
# Each shard processes ~190 complexes with 16 CPU workers

set -euo pipefail

cd /scratch/yd2915/BayesDiff
export PATH="/scratch/yd2915/BayesDiff/tools/mmseqs/bin:$PATH"

PYTHON=/scratch/yd2915/conda_envs/bayesdiff/bin/python
PDBBIND_DIR=data/PDBbind_data_set
OUTPUT_DIR=data/pdbbind_v2020

echo "=== Featurize shard ${SLURM_ARRAY_TASK_ID}/100 on $(hostname) ==="
echo "Start: $(date)"

$PYTHON scripts/pipeline/s00_prepare_pdbbind.py \
    --pdbbind_dir $PDBBIND_DIR \
    --output_dir $OUTPUT_DIR \
    --stage featurize \
    --shard_index ${SLURM_ARRAY_TASK_ID} \
    --num_shards 100 \
    --num_workers 16 \
    --radius 10.0

echo "End: $(date)"
