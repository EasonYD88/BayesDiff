#!/bin/bash
#SBATCH --job-name=bd_1k_eval
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j_1000step_eval.log
#SBATCH --error=slurm/logs/%j_1000step_eval.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Merge 93 pockets + GP Train + Eval + Ablation
# (1000-step embeddings, all sampling already complete)
# ─────────────────────────────────────────────────────────────

set -euo pipefail
cd /scratch/yd2915/BayesDiff

AFFINITY_PKL="external/targetdiff/data/affinity_info.pkl"
MERGED_DIR="results/embedding_1000step/merged"

mkdir -p slurm/logs "${MERGED_DIR}/gp_model" "${MERGED_DIR}/evaluation" "${MERGED_DIR}/ablation"

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: 1000-step Merge + GP + Eval + Ablation"
echo "═══════════════════════════════════════════════════════════"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# ── Environment ──────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Verify all 93 pockets exist ─────────────────────
echo ">>> [Step 1] Verifying pocket completeness..."
python3 -c "
from pathlib import Path

all_pockets = Path('data/splits/test_pockets.txt').read_text().strip().splitlines()
done = set()
for sd in Path('results/embedding_1000step').glob('*/shards/shard_*'):
    for p in sd.iterdir():
        if p.is_dir() and list(p.glob('*_embeddings.npy')):
            done.add(p.name)

missing = [p for p in all_pockets if p.strip() and p.strip() not in done]
print(f'Expected: {len(all_pockets)}, Found: {len(done)}, Missing: {len(missing)}')
if missing:
    print(f'Missing pockets: {missing}')
    print('WARNING: Proceeding with available pockets')
else:
    print('All 93 pockets present!')
"
echo ""

# ── Step 2: Merge all shard embeddings ───────────────────────
echo ">>> [Step 2] Merging embeddings from all shard directories..."
python3 -c "
import numpy as np
from pathlib import Path
import json

all_embeddings = {}
for sd in sorted(Path('results/embedding_1000step').glob('*/shards/shard_*')):
    for pocket_dir in sorted(sd.iterdir()):
        if not pocket_dir.is_dir():
            continue
        pdb_code = pocket_dir.name
        emb_files = list(pocket_dir.glob('*_embeddings.npy'))
        if emb_files:
            emb = np.load(emb_files[0])
            all_embeddings[pdb_code] = emb
            print(f'  {pdb_code}: {emb.shape}')

output_dir = Path('${MERGED_DIR}')
out_path = output_dir / 'all_embeddings.npz'
np.savez(out_path, **all_embeddings)

summary = {
    'n_pockets': len(all_embeddings),
    'pockets': sorted(all_embeddings.keys()),
    'shapes': {k: list(v.shape) for k, v in all_embeddings.items()},
}
with open(output_dir / 'sampling_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

total_mols = sum(v.shape[0] for v in all_embeddings.values())
print(f'\nMerged {len(all_embeddings)} pockets -> {out_path}')
print(f'Total molecules: {total_mols}')
"
echo "    Merge complete."
echo ""

# ── Step 3: Train GP Oracle ─────────────────────────────────
echo ">>> [Step 3] Training GP oracle..."
python scripts/pipeline/s04_train_gp.py \
    --embeddings "${MERGED_DIR}/all_embeddings.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${MERGED_DIR}/gp_model" \
    --n_inducing 128 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200
echo "    GP training complete."
echo ""

# ── Step 4: Evaluation ──────────────────────────────────────
echo ">>> [Step 4] Running evaluation..."
python scripts/pipeline/s05_evaluate.py \
    --embeddings "${MERGED_DIR}/all_embeddings.npz" \
    --gp_model "${MERGED_DIR}/gp_model/gp_model.pt" \
    --gp_train_data "${MERGED_DIR}/gp_model/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${MERGED_DIR}/evaluation" \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000
echo "    Evaluation complete."
echo ""

# ── Step 5: Ablation ────────────────────────────────────────
echo ">>> [Step 5] Running ablation study..."
python scripts/pipeline/s06_ablation.py \
    --embeddings "${MERGED_DIR}/all_embeddings.npz" \
    --gp_model "${MERGED_DIR}/gp_model/gp_model.pt" \
    --gp_train_data "${MERGED_DIR}/gp_model/train_data.npz" \
    --affinity_pkl "${AFFINITY_PKL}" \
    --output "${MERGED_DIR}/ablation" \
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

data = np.load('${MERGED_DIR}/all_embeddings.npz')
print(f'Pockets: {len(data.files)}')
print(f'Total molecules: {sum(data[k].shape[0] for k in data.files)}')
print()

print('=== Evaluation Metrics (1000-step) ===')
with open('${MERGED_DIR}/evaluation/eval_metrics.json') as f:
    m = json.load(f)
for k in ['ece','auroc','ef_1pct','hit_rate','spearman_rho','rmse','nll']:
    v = m.get(k, 'N/A')
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
print()

print('=== Ablation Summary (1000-step) ===')
with open('${MERGED_DIR}/ablation/ablation_summary.json') as f:
    d = json.load(f)
for aid, r in d.items():
    desc = r.get('description', aid)
    print(f'  {desc}: ECE={r[\"ece\"]:.4f} AUROC={r[\"auroc\"]:.4f} EF@1%={r[\"ef_1pct\"]:.2f}')
"
echo ""
echo "Output: ${MERGED_DIR}/"
echo "Done!"
