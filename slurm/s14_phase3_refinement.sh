#!/bin/bash
#SBATCH --job-name=phase3_refine
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/phase3_refine_%j.out
#SBATCH --error=slurm/logs/phase3_refine_%j.err

# Sub-Plan 2 Phase 3: Scheme B refinement + GP integration
# A3.2: entropy reg λ=0.1
# A3.4: SchemeB → freeze → SVGP

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

python scripts/pipeline/s14_phase3_refinement.py \
    --atom_emb_dir results/atom_embeddings \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/phase3_refinement \
    --experiment A3.2 A3.4 \
    --embed_dim 128 \
    --attn_hidden_dim 64 \
    --entropy_weight 0.01 \
    --lr 1e-3 \
    --n_epochs 200 \
    --batch_size 64 \
    --patience 30 \
    --n_inducing 512 \
    --gp_epochs 200 \
    --gp_batch_size 256 \
    --gp_lr 0.01 \
    --device cuda \
    --seed 42

echo "=== Done ==="
echo "Date: $(date)"
