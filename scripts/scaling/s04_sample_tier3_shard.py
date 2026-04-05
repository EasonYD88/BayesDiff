#!/usr/bin/env python
"""
scripts/scaling/s04_sample_tier3_shard.py
─────────────────────────────────
Array-job worker for Tier 3 dataset generation.

Reads the pocket list JSON, shards by SLURM_ARRAY_TASK_ID, and generates
molecules for each pocket in the shard using TargetDiff.

Supports two pocket sources:
  - test_set: loads PDB file directly
  - lmdb: loads pre-processed .pt file (from scripts/scaling/s03_prepare_tier3.py)

Usage:
  python scripts/scaling/s04_sample_tier3_shard.py \\
      --pocket-list data/tier3_pocket_list.json \\
      --output-dir results/tier3_sampling \\
      --num-samples 64 --num-steps 100 \\
      --num-shards 16 --shard-index $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shard-%(shard)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "external" / "targetdiff" / "pretrained_models" / "pretrained_diffusion.pt"
TD_DIR = PROJECT_ROOT / "external" / "targetdiff"

sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Tier 3 sampling shard worker")
    parser.add_argument("--pocket-list", type=str, required=True,
                        help="Path to tier3_pocket_list.json")
    parser.add_argument("--output-dir", type=str, default="results/tier3_sampling")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Set up shard-aware logging
    shard_id = args.shard_index
    for handler in logging.root.handlers:
        if hasattr(handler, 'formatter') and handler.formatter:
            handler.formatter = logging.Formatter(
                f"%(asctime)s [shard-{shard_id}] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
    logger = logging.getLogger(__name__)

    # Load pocket list
    with open(args.pocket_list) as f:
        pocket_data = json.load(f)
    all_pockets = pocket_data["pockets"]

    # Shard: round-robin assignment
    shard_pockets = [
        p for i, p in enumerate(all_pockets)
        if i % args.num_shards == shard_id
    ]
    logger.info(
        f"Shard {shard_id}/{args.num_shards}: "
        f"{len(shard_pockets)}/{len(all_pockets)} pockets"
    )

    if not shard_pockets:
        logger.info("No pockets assigned to this shard. Exiting.")
        return

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize sampler
    from bayesdiff.sampler import TargetDiffSampler
    sampler = TargetDiffSampler(
        targetdiff_dir=str(TD_DIR),
        checkpoint_path=str(CHECKPOINT),
        device=args.device,
        num_steps=args.num_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each pocket
    results_summary = []
    for idx, pocket in enumerate(shard_pockets):
        family = pocket["family"]
        source = pocket["source"]
        t0 = time.time()

        # Output paths
        pocket_out = output_dir / family
        sdf_path = pocket_out / "molecules.sdf"
        emb_path = pocket_out / "embeddings.npy"

        # Skip if already done
        if args.skip_existing and sdf_path.exists() and emb_path.exists():
            logger.info(f"[{idx+1}/{len(shard_pockets)}] {family}: SKIP (exists)")
            results_summary.append({
                "family": family, "status": "skipped",
            })
            continue

        logger.info(
            f"[{idx+1}/{len(shard_pockets)}] {family} ({source}): "
            f"generating {args.num_samples} molecules..."
        )

        try:
            pocket_out.mkdir(parents=True, exist_ok=True)

            if source == "test_set":
                # Use PDB file directly
                pdb_path = pocket["pdb_path"]
                mols, embeddings = sampler.sample_and_embed(
                    pocket_pdb=pdb_path,
                    num_samples=args.num_samples,
                    save_sdf=sdf_path,
                )
            elif source == "lmdb":
                # Load pre-processed .pt file
                data_path = pocket["data_path"]
                data = sampler.load_pocket_data(data_path)
                mols, embeddings = sampler.sample_and_embed_data(
                    data,
                    num_samples=args.num_samples,
                    save_sdf=sdf_path,
                )
            else:
                logger.warning(f"  Unknown source: {source}, skipping")
                continue

            # Save embeddings
            np.save(emb_path, embeddings)

            n_valid = sum(1 for m in mols if m is not None)
            elapsed = time.time() - t0

            logger.info(
                f"  Done: {n_valid}/{len(mols)} valid molecules, "
                f"embeddings shape {embeddings.shape}, "
                f"{elapsed:.1f}s"
            )

            results_summary.append({
                "family": family,
                "status": "success",
                "n_valid": n_valid,
                "n_total": len(mols),
                "elapsed": elapsed,
                "pKd": pocket["pKd"],
            })

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  FAILED: {e} ({elapsed:.1f}s)")
            results_summary.append({
                "family": family,
                "status": "failed",
                "error": str(e),
                "elapsed": elapsed,
            })
            # Clean up CUDA memory
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # Save shard summary
    summary_path = output_dir / f"shard_{shard_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "shard_index": shard_id,
            "num_shards": args.num_shards,
            "n_total": len(shard_pockets),
            "n_success": sum(1 for r in results_summary if r["status"] == "success"),
            "n_failed": sum(1 for r in results_summary if r["status"] == "failed"),
            "n_skipped": sum(1 for r in results_summary if r["status"] == "skipped"),
            "results": results_summary,
        }, f, indent=2)

    logger.info(
        f"\n=== Shard {shard_id} complete ===\n"
        f"  Success: {sum(1 for r in results_summary if r['status'] == 'success')}\n"
        f"  Failed:  {sum(1 for r in results_summary if r['status'] == 'failed')}\n"
        f"  Skipped: {sum(1 for r in results_summary if r['status'] == 'skipped')}"
    )


if __name__ == "__main__":
    main()
