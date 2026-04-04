#!/bin/bash
#SBATCH --job-name=multilayer_extract
#SBATCH --output=slurm/logs/multilayer_extract_%j.out
#SBATCH --error=slurm/logs/multilayer_extract_%j.out
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_281_chemistry

echo "=== MULTILAYER EXTRACTION (FULL 942 POCKETS) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

echo ""
echo "=== Phase A: Multi-layer TargetDiff extraction ==="
python scripts/26a_extract_multilayer_full.py

echo ""
echo "=== Phase A Complete ==="
echo "Multi-layer embeddings extracted: $(find results/tier3_sampling/ -name 'multilayer_embeddings.npz' | wc -l)"
echo "Date: $(date)"
