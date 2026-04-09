#!/bin/bash
#SBATCH --job-name=scheme_compare
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/scheme_compare_%j.out
#SBATCH --error=slurm/logs/scheme_compare_%j.err

# Sub-Plan 2 Phase 2: Scheme A vs Scheme B comparison
# A2.1: MeanPool→AttnFusion→MLP
# A2.2: SchemeA TwoBranch→MLP
# A2.3: SchemeB SingleBranch→MLP

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s13_scheme_comparison.py \
    --atom_emb_dir results/atom_embeddings \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/scheme_comparison \
    --experiment A2.1 A2.2 A2.3 \
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
