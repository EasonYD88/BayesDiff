"""
scripts/02_sample_molecules.py
──────────────────────────────
Sample molecules for each pocket using TargetDiff.

Mac debug mode:  python scripts/02_sample_molecules.py \
    --pocket_list data/splits/debug_pockets.txt \
    --pdbbind_dir data/pdbbind \
    --num_samples 4 --device cpu \
    --output_dir results/generated_molecules

HPC batch mode:  See slurm/sample_job.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bayesdiff.data import find_pocket_file
from bayesdiff.sampler import TargetDiffSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sample molecules with TargetDiff")
    parser.add_argument(
        "--pocket_list",
        type=str,
        required=True,
        help="Text file with one PDB code per line",
    )
    parser.add_argument(
        "--pdbbind_dir",
        type=str,
        required=True,
        help="Path to PDBbind root directory",
    )
    parser.add_argument(
        "--targetdiff_dir",
        type=str,
        default="external/targetdiff",
        help="Path to cloned TargetDiff repo",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (auto-detects in targetdiff_dir if not set)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Molecules per pocket"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (default: num_samples)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/generated_molecules",
        help="Output directory",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="DDPM diffusion steps"
    )

    args = parser.parse_args()

    pdbbind_dir = Path(args.pdbbind_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read pocket list
    pocket_list = Path(args.pocket_list)
    pdb_codes = [
        line.strip()
        for line in pocket_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info(f"Will sample {args.num_samples} molecules for {len(pdb_codes)} pockets")

    # Auto-detect checkpoint
    targetdiff_dir = Path(args.targetdiff_dir)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        candidates = [
            targetdiff_dir / "pretrained_model.pt",
            targetdiff_dir / "pretrained_models" / "pretrained_diffusion.pt",
            targetdiff_dir / "checkpoints" / "pretrained_model.pt",
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            logger.error(
                "Cannot find TargetDiff checkpoint. Provide --checkpoint or download to:\n"
                f"  {candidates[0]}"
            )
            sys.exit(1)

    # Initialize sampler
    sampler = TargetDiffSampler(
        targetdiff_dir=targetdiff_dir,
        checkpoint_path=ckpt_path,
        device=args.device,
        num_steps=args.num_steps,
    )

    # Sample for each pocket
    all_embeddings = {}
    timing_log = []

    for i, pdb_code in enumerate(pdb_codes):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(pdb_codes)}] Sampling for pocket: {pdb_code}")

        # Find pocket file
        pocket_file = find_pocket_file(pdbbind_dir, pdb_code)
        if pocket_file is None:
            logger.warning(f"Pocket file not found for {pdb_code}, skipping")
            continue

        # Create output subdir
        pocket_out = output_dir / pdb_code
        pocket_out.mkdir(parents=True, exist_ok=True)
        sdf_path = pocket_out / f"{pdb_code}_generated.sdf"

        # Sample
        t0 = time.time()
        try:
            mols, embeddings = sampler.sample_and_embed(
                pocket_pdb=pocket_file,
                num_samples=args.num_samples,
                save_sdf=sdf_path,
            )
            elapsed = time.time() - t0

            # Save embeddings
            emb_path = pocket_out / f"{pdb_code}_embeddings.npy"
            np.save(emb_path, embeddings)

            all_embeddings[pdb_code] = embeddings

            n_valid = sum(1 for m in mols if m is not None)
            logger.info(
                f"  Generated {n_valid}/{args.num_samples} valid molecules "
                f"in {elapsed:.1f}s"
            )
            logger.info(f"  Embeddings shape: {embeddings.shape}")
            logger.info(f"  SDF: {sdf_path}")

            timing_log.append(
                {
                    "pdb_code": pdb_code,
                    "n_valid": n_valid,
                    "n_total": args.num_samples,
                    "elapsed_s": round(elapsed, 1),
                    "emb_shape": list(embeddings.shape),
                }
            )

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  Failed after {elapsed:.1f}s: {e}")
            timing_log.append(
                {
                    "pdb_code": pdb_code,
                    "n_valid": 0,
                    "n_total": args.num_samples,
                    "elapsed_s": round(elapsed, 1),
                    "error": str(e),
                }
            )

    # Save summary
    summary_path = output_dir / "sampling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "n_pockets": len(pdb_codes),
                "n_sampled": len(all_embeddings),
                "timing": timing_log,
            },
            f,
            indent=2,
        )

    # Save combined embeddings
    if all_embeddings:
        combined_path = output_dir / "all_embeddings.npz"
        np.savez(combined_path, **{k: v for k, v in all_embeddings.items()})
        logger.info(f"\nSaved combined embeddings to {combined_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Sampling complete: {len(all_embeddings)}/{len(pdb_codes)} pockets")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
