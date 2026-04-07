#!/bin/bash
#SBATCH --job-name=s11_mgp
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/s11_multigranularity_gp_%j.out
#SBATCH --error=slurm/logs/s11_multigranularity_gp_%j.err

# ─────────────────────────────────────────────────────────────────
# Stage 2 SP1: Multi-Granularity GP Evaluation
#
# Combines z_global (layer 8) + z_interaction (from InteractionGNN)
# and evaluates via GP. Optionally includes shuffled-edge ablation.
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Multi-Granularity GP Evaluation (s11) ==="
echo "Date   : $(date)"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================"

# Build command with optional shuffled-edge directory
CMD="python scripts/pipeline/s11_multigranularity_gp.py \
    --multilayer_emb results/multilayer_embeddings/all_multilayer_embeddings.npz \
    --z_interaction_dir results/stage2/interaction_gnn \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/multigranularity_gp \
    --layer_idx 8 \
    --n_inducing 512 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --device cuda"

# Add shuffled dir if it exists (from completed s10b job)
SHUFFLED_DIR="results/stage2/interaction_gnn_shuffled"
if [ -f "${SHUFFLED_DIR}/z_interaction_train.npz" ]; then
    echo "Found shuffled-edge embeddings — including ablation A1.10"
    CMD="${CMD} --z_shuffled_dir ${SHUFFLED_DIR}"
else
    echo "No shuffled-edge embeddings found — skipping ablation A1.10"
    echo "(Run s10b_train_ignn_shuffled.sh first, then re-run this script)"
fi

echo ""
echo "Running: ${CMD}"
echo ""
eval "${CMD}"

echo ""
echo "=== s11 complete $(date) ==="
