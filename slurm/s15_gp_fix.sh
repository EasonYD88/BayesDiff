#!/bin/bash
#SBATCH --job-name=gp_fix
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/gp_fix_%j.out
#SBATCH --error=slurm/logs/gp_fix_%j.err

# Sub-Plan 2 Phase 3 GP fix:
# A3.4b: PCA(128→32) → SVGP (constrained noise)
# A3.4c: DKL (128→32 MLP + SVGP, joint)
# A3.4d: PCA(128→16) → SVGP (constrained noise)

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

# Force unbuffered output for real-time monitoring
export PYTHONUNBUFFERED=1

python scripts/pipeline/s15_gp_fix.py \
    --atom_emb_dir results/atom_embeddings \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/gp_fix \
    --phase3_dir results/stage2/phase3_refinement \
    --n_inducing 256 \
    --gp_epochs 500 \
    --gp_batch_size 256 \
    --gp_lr 0.005 \
    --gp_patience 80 \
    --device cuda

echo "=== Done: $(date) ==="
