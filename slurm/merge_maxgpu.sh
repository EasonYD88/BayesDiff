#!/bin/bash
#SBATCH --job-name=bd_merge50
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j_merge50.log
#SBATCH --error=slurm/logs/%j_merge50.err

# ─────────────────────────────────────────────────────────────
# BayesDiff: Merge 93-pocket shards → GP train → Evaluate
# Run after sample_maxgpu.sh array completes.
#
# Submit automatically:
#   ARRAY_JID=$(sbatch --parsable slurm/sample_maxgpu.sh)
#   sbatch --dependency=afterok:${ARRAY_JID} slurm/merge_maxgpu.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

WORKTREE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKTREE_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/embedding_50mol}"

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "═══════════════════════════════════════════════════════════"
echo " BayesDiff: Merge + GP + Evaluate (50-mol run)"
echo "═══════════════════════════════════════════════════════════"
echo "Date:     $(date)"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

# ── Read run metadata ─────────────────────────────────────────
RUN_TAG=$(cat "${OUTPUT_ROOT}/latest_run_tag.txt")
RUN_DIR=$(cat "${OUTPUT_ROOT}/latest_run_dir.txt")
NUM_SHARDS=$(cat "${OUTPUT_ROOT}/latest_num_shards.txt")
MERGED_DIR="${RUN_DIR}/merged"
mkdir -p "${MERGED_DIR}"

echo "Run tag:  ${RUN_TAG}"
echo "Run dir:  ${RUN_DIR}"
echo ""

# ── Step 1: Verify pocket coverage ───────────────────────────
echo ">>> [1/4] Verifying pocket coverage..."
python3 - <<'PYEOF'
import json
from pathlib import Path
import os

run_dir = Path(os.environ.get("RUN_DIR", ""))
pocket_list = Path("data/splits/test_pockets.txt").read_text().strip().splitlines()

found = set()
for shard_dir in sorted(run_dir.glob("shards/shard_*")):
    for pocket_dir in sorted(shard_dir.iterdir()):
        if pocket_dir.is_dir() and list(pocket_dir.glob("*_embeddings.npy")):
            found.add(pocket_dir.name)

missing = [p for p in pocket_list if p.strip() and p.strip() not in found]
print(f"Expected: {len(pocket_list)}, Found: {len(found)}, Missing: {len(missing)}")
if missing:
    print(f"Missing pockets: {missing}")
PYEOF
export RUN_DIR
echo ""

# ── Step 2: Merge shard embeddings ───────────────────────────
echo ">>> [2/4] Merging shard embeddings..."
python3 - <<'PYEOF'
import numpy as np, json
from pathlib import Path
import os

run_dir = Path(os.environ["RUN_DIR"])
merged_dir = run_dir / "merged"
merged_dir.mkdir(exist_ok=True)

all_emb = {}
for shard_dir in sorted(run_dir.glob("shards/shard_*")):
    for pocket_dir in sorted(shard_dir.iterdir()):
        if not pocket_dir.is_dir():
            continue
        code = pocket_dir.name
        files = sorted(pocket_dir.glob("*_embeddings.npy"))
        if files and code not in all_emb:
            all_emb[code] = np.load(files[0])
            print(f"  {code}: {all_emb[code].shape}")

out_path = merged_dir / "all_embeddings.npz"
np.savez(out_path, **all_emb)

total = sum(v.shape[0] for v in all_emb.values())
summary = {"n_pockets": len(all_emb), "total_molecules": total,
           "pockets": sorted(all_emb.keys()),
           "shapes": {k: list(v.shape) for k, v in all_emb.items()}}
with open(merged_dir / "sampling_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nMerged {len(all_emb)} pockets → {out_path}")
print(f"Total molecules: {total}")
PYEOF
echo ""

# ── Step 3: GP Training ───────────────────────────────────────
echo ">>> [3/4] Training GP on merged embeddings..."
python scripts/04_train_gp.py \
  --embeddings    "${MERGED_DIR}/all_embeddings.npz" \
  --splits        data/splits/splits.json \
  --labels        data/splits/labels.csv \
  --affinity_pkl  external/targetdiff/data/affinity_info.pkl \
  --output_dir    "${MERGED_DIR}/gp_model" \
  --device        cuda || echo "GP training script not available, skipping."
echo ""

# ── Step 4: Evaluate ─────────────────────────────────────────
echo ">>> [4/4] Evaluating..."
python scripts/05_evaluate.py \
  --embeddings  "${MERGED_DIR}/all_embeddings.npz" \
  --gp_model    "${MERGED_DIR}/gp_model" \
  --affinity_pkl external/targetdiff/data/affinity_info.pkl \
  --output_dir  "${MERGED_DIR}/evaluation" \
  --device      cuda || echo "Evaluate script not available, skipping."

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Merge + GP + Eval complete at $(date)"
echo " Results: ${MERGED_DIR}"
echo "═══════════════════════════════════════════════════════════"
