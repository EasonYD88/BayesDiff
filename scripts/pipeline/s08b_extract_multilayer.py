"""
scripts/pipeline/s08b_extract_multilayer.py
───────────────────────────────────────────
Extract multi-layer embeddings from TargetDiff encoder for all PDBbind complexes.

Uses crystal ligand poses (single forward pass, no diffusion sampling) to
extract per-layer mean-pooled embeddings from each encoder layer.

Architecture: UniTransformerO2TwoUpdateGeneral with 1 init + 9 base layers = 10 layers total.

Usage (single node):
    python scripts/pipeline/s08b_extract_multilayer.py \\
        --data_dir data/pdbbind_v2020 \\
        --output_dir results/multilayer_embeddings \\
        --device cuda

Usage (SLURM shard for HPC):
    python scripts/pipeline/s08b_extract_multilayer.py \\
        --data_dir data/pdbbind_v2020 \\
        --output_dir results/multilayer_embeddings \\
        --shard_index $SLURM_ARRAY_TASK_ID --num_shards 50 \\
        --device cuda

Usage (merge after all shards):
    python scripts/pipeline/s08b_extract_multilayer.py \\
        --data_dir data/pdbbind_v2020 \\
        --output_dir results/multilayer_embeddings \\
        --stage merge

Output:
    results/multilayer_embeddings/all_multilayer_embeddings.npz
        For each pdb_code:
            '{pdb_code}_layer_0' ... '{pdb_code}_layer_9': (d,) embeddings
            '{pdb_code}_z_global': (d,) last-layer embedding
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_checkpoint(targetdiff_dir: Path) -> Path:
    """Auto-detect pretrained TargetDiff checkpoint."""
    candidates = [
        targetdiff_dir / "pretrained_models" / "pretrained_diffusion.pt",
        targetdiff_dir / "checkpoints" / "pretrained_diffusion.pt",
    ]
    pm_dir = targetdiff_dir / "pretrained_models"
    if pm_dir.exists():
        for p in pm_dir.glob("*.pt"):
            candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No TargetDiff checkpoint found. Searched: {[str(c) for c in candidates[:3]]}"
    )


def extract_shard(args):
    """Extract multi-layer embeddings for a shard of PDBbind complexes."""
    from bayesdiff.sampler import TargetDiffSampler

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_dir / "processed"

    # Get all pdb codes
    codes = sorted([p.stem for p in processed_dir.glob("*.pt")])
    logger.info(f"Total complexes: {len(codes)}")

    # Shard if specified
    if args.num_shards > 1:
        shard_size = (len(codes) + args.num_shards - 1) // args.num_shards
        start = args.shard_index * shard_size
        end = min(start + shard_size, len(codes))
        codes = codes[start:end]
        logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(codes)} complexes ({start}–{end})")

    # Init sampler
    targetdiff_dir = Path(args.targetdiff_dir)
    ckpt_path = find_checkpoint(targetdiff_dir)
    sampler = TargetDiffSampler(
        targetdiff_dir=str(targetdiff_dir),
        checkpoint_path=str(ckpt_path),
        device=args.device,
    )

    n_layers = sampler.num_encoder_layers
    logger.info(f"Encoder layers: {n_layers}, hidden_dim: {sampler.hidden_dim}")

    # Extract embeddings
    results = {}
    failed = []
    t_start = time.time()

    for i, pdb_code in enumerate(codes):
        pt_path = processed_dir / f"{pdb_code}.pt"
        try:
            emb = sampler.extract_multilayer_embeddings(pt_path)
            for key, val in emb.items():
                if isinstance(val, np.ndarray):
                    results[f"{pdb_code}_{key}"] = val
            results[f"{pdb_code}_n_layers"] = np.array([emb["n_layers"]])

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                logger.info(
                    f"  [{i+1}/{len(codes)}] {rate:.1f} complexes/s "
                    f"(elapsed: {elapsed:.0f}s)"
                )
        except Exception as e:
            logger.warning(f"  Failed {pdb_code}: {e}")
            failed.append(pdb_code)

    # Save shard results
    if args.num_shards > 1:
        out_file = output_dir / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.npz"
    else:
        out_file = output_dir / "all_multilayer_embeddings.npz"

    np.savez_compressed(str(out_file), **results)
    logger.info(f"Saved {len(codes) - len(failed)} embeddings to {out_file}")

    # Save status
    status = {
        "n_total": len(codes),
        "n_success": len(codes) - len(failed),
        "n_failed": len(failed),
        "failed_codes": failed,
        "n_layers": n_layers,
        "hidden_dim": sampler.hidden_dim,
        "elapsed_seconds": time.time() - t_start,
    }
    if args.num_shards > 1:
        status_file = output_dir / f"status_shard_{args.shard_index:04d}.json"
    else:
        status_file = output_dir / "extraction_status.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    if failed:
        logger.warning(f"Failed: {len(failed)} complexes: {failed[:10]}...")


def merge_shards(args):
    """Merge shard .npz files into a single file."""
    output_dir = Path(args.output_dir)
    shard_files = sorted(output_dir.glob("shard_*_of_*.npz"))
    logger.info(f"Found {len(shard_files)} shard files")

    if not shard_files:
        logger.error("No shard files found!")
        return

    merged = {}
    for sf in shard_files:
        data = np.load(str(sf), allow_pickle=True)
        for key in data.files:
            merged[key] = data[key]
        logger.info(f"  Loaded {sf.name}: {len(data.files)} keys")

    out_file = output_dir / "all_multilayer_embeddings.npz"
    np.savez_compressed(str(out_file), **merged)

    # Count unique pdb codes
    pdb_codes = set()
    for key in merged:
        parts = key.rsplit("_", 1)
        if len(parts) == 2 and parts[1] == "global":
            # key like "1a30_z_global" -> pdb_code = "1a30_z" — not right
            pass
        if key.endswith("_n_layers"):
            pdb_codes.add(key.replace("_n_layers", ""))

    logger.info(f"Merged: {len(pdb_codes)} complexes → {out_file}")

    # Merge status files
    total_status = {"n_total": 0, "n_success": 0, "n_failed": 0, "failed_codes": []}
    for sf in sorted(output_dir.glob("status_shard_*.json")):
        with open(sf) as f:
            s = json.load(f)
        total_status["n_total"] += s["n_total"]
        total_status["n_success"] += s["n_success"]
        total_status["n_failed"] += s["n_failed"]
        total_status["failed_codes"].extend(s.get("failed_codes", []))

    with open(output_dir / "extraction_status.json", "w") as f:
        json.dump(total_status, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-layer embeddings from TargetDiff encoder"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/pdbbind_v2020",
        help="PDBbind data directory with processed/ subdir",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/multilayer_embeddings",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--targetdiff_dir", type=str, default="external/targetdiff",
        help="Path to TargetDiff repo",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu or cuda)",
    )
    parser.add_argument(
        "--shard_index", type=int, default=0,
        help="Shard index for HPC array jobs",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of shards (1 = no sharding)",
    )
    parser.add_argument(
        "--stage", type=str, default="extract",
        choices=["extract", "merge"],
        help="Pipeline stage: 'extract' or 'merge'",
    )
    args = parser.parse_args()

    if args.stage == "merge":
        merge_shards(args)
    else:
        extract_shard(args)


if __name__ == "__main__":
    main()
