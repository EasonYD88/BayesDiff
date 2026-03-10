#!/bin/bash
#SBATCH --job-name=bd_1k_fin
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%j_finish_1000step.log
#SBATCH --error=slurm/logs/%j_finish_1000step.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Finish last 5 pockets (1000-step) + Merge + GP + Eval + Ablation
#
# Remaining: RG1_RAUSE_1_513_0, SIR3_HUMAN_117_398_0,
#            TNKS1_HUMAN_1099_1319_0, UPPS_ECOLI_1_253_0,
#            VAOX_PENSI_1_560_0
# ─────────────────────────────────────────────────────────────

set -euo pipefail
cd /scratch/yd2915/BayesDiff

POCKET_LIST="data/splits/remaining_1000step_final.txt"
NUM_SAMPLES=64
NUM_STEPS=1000
DEVICE="cuda"
PDBBIND_DIR="external/targetdiff/data/test_set"
TARGETDIFF_DIR="external/targetdiff"
AFFINITY_PKL="external/targetdiff/data/affinity_info.pkl"

# Output the remaining samples into shard_0of4 (same as original job)
SAMPLE_OUT="results/embedding_1000step/20260305_085825_j3387783/shards/shard_0of4"
MERGED_DIR="results/embedding_1000step/merged"

mkdir -p slurm/logs "${SAMPLE_OUT}" "${MERGED_DIR}/gp_model" "${MERGED_DIR}/evaluation" "${MERGED_DIR}/ablation"

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: Finish 1000-step Pipeline (5 remaining pockets)"
echo "═══════════════════════════════════════════════════════════"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ── Environment ──────────────────────────────────────────────
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Sample remaining 5 pockets ──────────────────────
echo ">>> [Step 1] Sampling 5 remaining pockets (64 mols × 1000 steps each)..."
python scripts/02_sample_molecules.py \
    --pocket_list "${POCKET_LIST}" \
    --pdbbind_dir "${PDBBIND_DIR}" \
    --targetdiff_dir "${TARGETDIFF_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_steps "${NUM_STEPS}" \
    --device "${DEVICE}" \
    --output_dir "${SAMPLE_OUT}"
echo "    Sampling complete."
echo ""

# ── Step 2: Merge all shard embeddings ───────────────────────
echo ">>> [Step 2] Merging embeddings from all 4 shard directories..."
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
output_dir = Path('${MERGED_DIR}')

all_embeddings = {}
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
            print(f'  {pdb_code}: {emb.shape}')

out_path = output_dir / 'all_embeddings.npz'
np.savez(out_path, **all_embeddings)

summary = {
    'n_pockets': len(all_embeddings),
    'n_sampled': len(all_embeddings),
    'pockets': sorted(all_embeddings.keys()),
    'shapes': {k: list(v.shape) for k, v in all_embeddings.items()},
}
with open(output_dir / 'sampling_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nMerged {len(all_embeddings)} pockets -> {out_path}')
total_mols = sum(v.shape[0] for v in all_embeddings.values())
print(f'Total molecules: {total_mols}')
"
echo "    Merge complete."
echo ""

# ── Step 3: Train GP Oracle ─────────────────────────────────
echo ">>> [Step 3] Training GP oracle..."
python scripts/04_train_gp.py \
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
python scripts/05_evaluate.py \
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
python scripts/06_ablation.py \
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
echo ""

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
echo "Output dir: ${MERGED_DIR}/"
echo "Done!"
