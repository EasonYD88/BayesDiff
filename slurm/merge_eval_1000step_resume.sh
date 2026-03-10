squeue | grep h100#!/bin/bash
#SBATCH --job-name=bd_merge1k
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%j_merge_1k_resume.log
#SBATCH --error=slurm/logs/%j_merge_1k_resume.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Merge all 4 shard dirs + GP + Evaluation
#
# Custom merge for the split-directory layout from job 3387783.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

AFFINITY_PKL="${AFFINITY_PKL:-external/targetdiff/data/affinity_info.pkl}"
OUTPUT_DIR="results/embedding_1000step/merged"

mkdir -p slurm/logs "${OUTPUT_DIR}/gp_model" "${OUTPUT_DIR}/evaluation" "${OUTPUT_DIR}/ablation"

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: Merge + GP + Eval (1000-step resumed)"
echo "═══════════════════════════════════════════════════════════"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# ── Environment ──────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Merge all shard embeddings ───────────────────────
echo ">>> [Step 1] Merging embeddings from 4 shard directories..."
python3 -c "
import numpy as np
from pathlib import Path
import json

shard_dirs = [
    'results/embedding_1000step/20260305_085825_j3387783/shards/shard_0of4',
    'results/embedding_1000step/20260305_085827_j3387783/shards/shard_1of4',
    'results/embedding_1000step/20260305_085834_j3387783/shards/shard_2of4',
    'results/embedding_1000step/20260305_085839_j3387783/shards/shard_3of4',
]
output_dir = Path('${OUTPUT_DIR}')

all_embeddings = {}
all_timing = []
n_pockets = 0

for sd in shard_dirs:
    sd = Path(sd)
    if not sd.exists():
        print(f'WARNING: {sd} not found, skipping')
        continue
    for pocket_dir in sorted(sd.iterdir()):
        if not pocket_dir.is_dir():
            continue
        pdb_code = pocket_dir.name
        emb_files = list(pocket_dir.glob('*_embeddings.npy'))
        if emb_files:
            emb = np.load(emb_files[0])
            all_embeddings[pdb_code] = emb
            n_pockets += 1
            print(f'  {pdb_code}: {emb.shape}')

# Save merged embeddings
out_path = output_dir / 'all_embeddings.npz'
np.savez(out_path, **all_embeddings)

# Save summary
summary = {
    'n_pockets': n_pockets,
    'n_sampled': len(all_embeddings),
    'pockets': sorted(all_embeddings.keys()),
    'shapes': {k: list(v.shape) for k, v in all_embeddings.items()},
}
with open(output_dir / 'sampling_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nMerged {n_pockets} pockets -> {out_path}')
total_mols = sum(v.shape[0] for v in all_embeddings.values())
print(f'Total molecules: {total_mols}')
print(f'Embedding dim: {list(all_embeddings.values())[0].shape[1] if all_embeddings else \"N/A\"}')
"
echo "    Merge complete."
echo ""

# ── Step 2: Train GP Oracle ─────────────────────────────────
echo ">>> [Step 2] Training GP oracle..."
python scripts/04_train_gp.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${OUTPUT_DIR}/gp_model" \
    --n_inducing 128 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200
echo "    GP training complete."
echo ""

# ── Step 3: Evaluation ──────────────────────────────────────
echo ">>> [Step 3] Running evaluation..."
python scripts/05_evaluate.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_model "${OUTPUT_DIR}/gp_model/gp_model.pt" \
    --gp_train_data "${OUTPUT_DIR}/gp_model/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${OUTPUT_DIR}/evaluation" \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000
echo "    Evaluation complete."
echo ""

# ── Step 4: Ablation ────────────────────────────────────────
echo ">>> [Step 4] Running ablation study..."
python scripts/06_ablation.py \
    --embeddings "${OUTPUT_DIR}/all_embeddings.npz" \
    --gp_model "${OUTPUT_DIR}/gp_model/gp_model.pt" \
    --gp_train_data "${OUTPUT_DIR}/gp_model/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${OUTPUT_DIR}/ablation" \
    --y_target 7.0 \
    --bootstrap_n 1000
echo "    Ablation complete."
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Complete at $(date)"
echo "═══════════════════════════════════════════════════════════"

python3 -c "
import json, numpy as np
data = np.load('${OUTPUT_DIR}/all_embeddings.npz')
print(f'Pockets: {len(data.files)}')
print(f'Total mols: {sum(data[k].shape[0] for k in data.files)}')
try:
    with open('${OUTPUT_DIR}/evaluation/eval_metrics.json') as f:
        m = json.load(f)
    for k in ['ece','auroc','ef_1pct','spearman_rho','rmse','nll']:
        v = m.get(k, 'N/A')
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
except: pass
"
echo ""
echo "Output: ${OUTPUT_DIR}/"
echo "Done!"
