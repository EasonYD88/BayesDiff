"""
scripts/03_extract_embeddings.py
────────────────────────────────
Extract SE(3)-equivariant embeddings from *existing* generated molecules.

This script is useful when:
  - You already have SDF files from TargetDiff (e.g., from HPC sampling)
  - You want to re-extract embeddings with a different encoder
  - You want to extract embeddings for reference ligands (crystal structures)

Usage:
    # Extract from generated molecules
    python scripts/03_extract_embeddings.py \
        --input_dir results/generated_molecules \
        --pdbbind_dir data/pdbbind \
        --targetdiff_dir external/targetdiff \
        --output results/embeddings.npz

    # Extract from reference (crystal) ligands
    python scripts/03_extract_embeddings.py \
        --mode reference \
        --pocket_list data/splits/train_pockets.txt \
        --pdbbind_dir data/pdbbind \
        --targetdiff_dir external/targetdiff \
        --output results/reference_embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bayesdiff.data import find_ligand_file, find_pocket_file
from bayesdiff.sampler import TargetDiffSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_from_generated(args):
    """Extract embeddings from SDF files in input_dir/<pdb_code>/."""
    input_dir = Path(args.input_dir)
    pdbbind_dir = Path(args.pdbbind_dir)

    sampler = TargetDiffSampler(
        targetdiff_dir=args.targetdiff_dir,
        checkpoint_path=args.checkpoint or "auto",
        device=args.device,
    )

    all_embeddings = {}
    pocket_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()]
    )

    for i, pocket_dir in enumerate(pocket_dirs):
        pdb_code = pocket_dir.name
        sdf_files = list(pocket_dir.glob("*.sdf"))
        if not sdf_files:
            logger.warning(f"No SDF files found in {pocket_dir}")
            continue

        pocket_file = find_pocket_file(pdbbind_dir, pdb_code)
        if pocket_file is None:
            logger.warning(f"Pocket PDB not found for {pdb_code}")
            continue

        logger.info(f"[{i+1}/{len(pocket_dirs)}] Extracting embeddings for {pdb_code}")

        try:
            embeddings = sampler.extract_embeddings(pocket_file, sdf_files[0])
            all_embeddings[pdb_code] = embeddings
            logger.info(f"  Shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return all_embeddings


def extract_from_reference(args):
    """Extract embeddings for reference (crystal) ligands."""
    pdbbind_dir = Path(args.pdbbind_dir)
    pocket_list_path = Path(args.pocket_list)

    pdb_codes = [
        line.strip()
        for line in pocket_list_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    sampler = TargetDiffSampler(
        targetdiff_dir=args.targetdiff_dir,
        checkpoint_path=args.checkpoint or "auto",
        device=args.device,
    )

    all_embeddings = {}

    for i, pdb_code in enumerate(pdb_codes):
        pocket_file = find_pocket_file(pdbbind_dir, pdb_code)
        ligand_file = find_ligand_file(pdbbind_dir, pdb_code)

        if pocket_file is None or ligand_file is None:
            logger.warning(
                f"[{i+1}/{len(pdb_codes)}] Missing files for {pdb_code}, skipping"
            )
            continue

        logger.info(
            f"[{i+1}/{len(pdb_codes)}] Extracting reference embedding for {pdb_code}"
        )

        try:
            embeddings = sampler.extract_embeddings(pocket_file, ligand_file)
            # Reference ligand: single embedding
            all_embeddings[pdb_code] = embeddings[0]
            logger.info(f"  Shape: {embeddings[0].shape}")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract molecular embeddings")
    parser.add_argument(
        "--mode",
        type=str,
        default="generated",
        choices=["generated", "reference"],
        help="Extract from generated SDFs or reference crystal ligands",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/generated_molecules",
        help="Directory with generated molecules (for mode=generated)",
    )
    parser.add_argument(
        "--pocket_list",
        type=str,
        default=None,
        help="Pocket list file (for mode=reference)",
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
        help="Checkpoint path (auto-detect if not set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/embeddings.npz",
        help="Output .npz file",
    )

    args = parser.parse_args()

    if args.mode == "generated":
        all_embeddings = extract_from_generated(args)
    elif args.mode == "reference":
        if args.pocket_list is None:
            logger.error("--pocket_list is required for mode=reference")
            sys.exit(1)
        all_embeddings = extract_from_reference(args)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **all_embeddings)

    logger.info(f"\nSaved {len(all_embeddings)} embeddings to {output_path}")
    for k, v in list(all_embeddings.items())[:3]:
        logger.info(f"  {k}: {v.shape}")
    if len(all_embeddings) > 3:
        logger.info(f"  ... and {len(all_embeddings) - 3} more")


if __name__ == "__main__":
    main()
