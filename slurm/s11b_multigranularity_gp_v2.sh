#!/bin/bash
#SBATCH --job-name=s11_mgp_v2
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/s11_mgp_v2_%j.out
#SBATCH --error=slurm/logs/s11_mgp_v2_%j.err

# ─────────────────────────────────────────────────────────────────
# Stage 2 SP1: Multi-Granularity GP Evaluation (v2: node readout)
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Multi-Granularity GP v2 (node readout) ==="
echo "Date   : $(date)"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================"

CMD="python scripts/pipeline/s11_multigranularity_gp.py \
    --multilayer_emb results/multilayer_embeddings/all_multilayer_embeddings.npz \
    --z_interaction_dir results/stage2/interaction_gnn_node \
    --labels data/pdbbind_v2020/labels.csv \
    --splits data/pdbbind_v2020/splits.json \
    --output results/stage2/multigranularity_gp_v2 \
    --layer_idx 8 \
    --n_inducing 512 \
    --n_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --device cuda"

SHUFFLED_DIR="results/stage2/interaction_gnn_shuffled_node"
if [ -f "${SHUFFLED_DIR}/z_interaction_train.npz" ]; then
    echo "Found shuffled-edge (node readout) embeddings — including ablation A1.10"
    CMD="${CMD} --z_shuffled_dir ${SHUFFLED_DIR}"
else
    echo "No shuffled-edge embeddings found — skipping ablation A1.10"
fi

echo ""
echo "Running: ${CMD}"
echo ""
eval "${CMD}"

echo ""
echo "=== s11 v2 complete $(date) ==="
