"""
scripts/pipeline/s08c_extract_atom_embeddings.py
─────────────────────────────────────────────────
Extract per-layer ATOM-LEVEL embeddings from TargetDiff encoder.

Unlike s08b (which saves mean-pooled (d,) per layer), this produces
per-complex .pt files containing full (N_lig, d) atom tensors for each layer.

Usage (single node):
    python scripts/pipeline/s08c_extract_atom_embeddings.py \\
        --data_dir data/pdbbind_v2020 \\
        --output_dir results/atom_embeddings \\
        --device cuda

Usage (SLURM shard):
    python scripts/pipeline/s08c_extract_atom_embeddings.py \\
        --data_dir data/pdbbind_v2020 \\
        --output_dir results/atom_embeddings \\
        --shard_index $SLURM_ARRAY_TASK_ID --num_shards 50 \\
        --device cuda

Output:
    results/atom_embeddings/{pdb_code}.pt
        dict with keys:
            'layer_0' ... 'layer_9': (N_lig, d) numpy arrays
            'pocket_mean': (d,) numpy array
            'n_ligand_atoms': int
            'n_layers': int
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# PyG Data objects and torch.load can require deep recursion
sys.setrecursionlimit(10000)

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
    """Extract atom-level embeddings for a shard of PDBbind complexes."""
    from bayesdiff.sampler import TargetDiffSampler

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = data_dir / "processed"

    codes = sorted([p.stem for p in processed_dir.glob("*.pt")])
    logger.info(f"Total complexes: {len(codes)}")

    # Shard if specified
    if args.num_shards > 1:
        shard_size = (len(codes) + args.num_shards - 1) // args.num_shards
        start = args.shard_index * shard_size
        end = min(start + shard_size, len(codes))
        codes = codes[start:end]
        logger.info(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(codes)} complexes ({start}–{end})"
        )

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

    failed = []
    t_start = time.time()

    for i, pdb_code in enumerate(codes):
        pt_path = processed_dir / f"{pdb_code}.pt"
        out_path = output_dir / f"{pdb_code}.pt"

        if out_path.exists() and not args.overwrite:
            continue

        try:
            emb = sampler.extract_multilayer_atom_embeddings(pt_path)

            # Save as individual .pt file (numpy arrays inside dict)
            import torch as _torch
            _torch.save(emb, str(out_path))

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                logger.info(
                    f"  [{i+1}/{len(codes)}] {rate:.1f} complexes/s "
                    f"(elapsed: {elapsed:.0f}s)"
                )
        except Exception as e:
            if not failed:  # Full traceback for first failure only
                import traceback
                logger.error(f"  First failure {pdb_code}:\n{traceback.format_exc()}")
            else:
                logger.warning(f"  Failed {pdb_code}: {e}")
            failed.append(pdb_code)

    elapsed = time.time() - t_start
    logger.info(
        f"Done: {len(codes) - len(failed)}/{len(codes)} extracted "
        f"in {elapsed:.0f}s"
    )

    # Save status
    status = {
        "n_total": len(codes),
        "n_success": len(codes) - len(failed),
        "n_failed": len(failed),
        "failed_codes": failed,
        "n_layers": n_layers,
        "hidden_dim": sampler.hidden_dim,
        "elapsed_seconds": elapsed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    if args.num_shards > 1:
        status_file = output_dir / f"status_shard_{args.shard_index:04d}.json"
    else:
        status_file = output_dir / "extraction_status.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    if failed:
        logger.warning(f"Failed: {len(failed)} complexes: {failed[:10]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract atom-level multi-layer embeddings from TargetDiff encoder"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/pdbbind_v2020",
        help="Path to PDBbind v2020 data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/atom_embeddings",
        help="Output directory for per-complex .pt files",
    )
    parser.add_argument(
        "--targetdiff_dir", type=str, default="external/targetdiff",
        help="Path to TargetDiff repo root",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    extract_shard(args)


if __name__ == "__main__":
    main()
