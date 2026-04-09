#!/bin/bash
#SBATCH --job-name=ablation_viz
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/ablation_viz_%j.out
#SBATCH --error=slurm/logs/ablation_viz_%j.err

# Sub-Plan 2 Phase 3 supplementary:
# A3.5: SchemeB with Multi-Head AttnPool (H=4)
# A3.6: SchemeB with Independent per-layer AttnPool
# VIZ:  Molecular-level attention visualization

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

export PYTHONUNBUFFERED=1

python scripts/pipeline/s17_ablation_and_viz.py \
    --atom_emb_dir results/atom_embeddings \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/ablation_viz \
    --experiment A3.5 A3.6 VIZ \
    --embed_dim 128 \
    --attn_hidden_dim 64 \
    --entropy_weight 0.01 \
    --lr 1e-3 \
    --n_epochs 200 \
    --batch_size 64 \
    --patience 30 \
    --device cuda \
    --seed 42

echo "=== Done ==="
echo "Date: $(date)"
