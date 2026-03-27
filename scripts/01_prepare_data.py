"""
scripts/01_prepare_data.py
──────────────────────────
Parse PDBbind v2020 INDEX, create protein-family splits, extract pockets,
and produce label CSV + split JSON files.

Usage:
    python scripts/01_prepare_data.py \
        --pdbbind_dir data/pdbbind \
        --output_dir data/splits \
        [--casf_dir data/pdbbind/CASF-2016] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bayesdiff.data import (
    extract_pocket_from_protein,
    find_ligand_file,
    find_pocket_file,
    load_casf2016_codes,
    parse_pdbbind_index,
    protein_family_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare BayesDiff data from PDBbind")
    parser.add_argument(
        "--pdbbind_dir",
        type=str,
        required=True,
        help="Path to PDBbind v2020 root (containing refined-set/ and INDEX files)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/splits", help="Output directory"
    )
    parser.add_argument(
        "--casf_dir",
        type=str,
        default=None,
        help="Path to CASF-2016 directory (optional)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extract_pockets",
        action="store_true",
        help="Extract 10Å pockets from protein PDB files",
    )
    args = parser.parse_args()

    pdbbind_dir = Path(args.pdbbind_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse INDEX file ──────────────────────────────────────────────
    index_candidates = [
        pdbbind_dir / "INDEX_refined_data.2020",
        pdbbind_dir / "index" / "INDEX_refined_data.2020",
        pdbbind_dir / "refined-set" / "index" / "INDEX_refined_data.2020",
    ]
    index_path = None
    for p in index_candidates:
        if p.exists():
            index_path = p
            break

    if index_path is None:
        logger.error(
            f"Cannot find INDEX file. Searched:\n"
            + "\n".join(f"  {p}" for p in index_candidates)
        )
        sys.exit(1)

    logger.info(f"Parsing INDEX file: {index_path}")
    df = parse_pdbbind_index(index_path)

    # Verify that PDB files actually exist
    existing = []
    for _, row in df.iterrows():
        pocket = find_pocket_file(pdbbind_dir, row["pdb_code"])
        if pocket is not None:
            existing.append(row["pdb_code"])

    logger.info(
        f"Found structural files for {len(existing)}/{len(df)} complexes"
    )
    df = df[df["pdb_code"].isin(existing)].reset_index(drop=True)

    # ── 2. Save labels CSV ───────────────────────────────────────────────
    labels_path = output_dir / "labels.csv"
    df.to_csv(labels_path, index=False)
    logger.info(f"Saved labels to {labels_path}")

    # Print statistics
    logger.info(f"  pKd range: [{df['pkd'].min():.2f}, {df['pkd'].max():.2f}]")
    logger.info(f"  Mean pKd: {df['pkd'].mean():.2f} ± {df['pkd'].std():.2f}")
    logger.info(f"  Affinity types: {df['affinity_type'].value_counts().to_dict()}")
    logger.info(
        f"  Active (pKd >= 7): {(df['pkd'] >= 7).sum()} / {len(df)} "
        f"({(df['pkd'] >= 7).mean():.1%})"
    )

    # ── 3. Create splits ─────────────────────────────────────────────────
    logger.info("Creating protein-family splits...")
    splits = protein_family_split(df, pdbbind_dir, seed=args.seed)

    # Remove CASF-2016 codes from splits if they exist
    casf_codes = set()
    if args.casf_dir:
        casf_dir = Path(args.casf_dir)
        if casf_dir.exists():
            casf_codes = set(load_casf2016_codes(casf_dir))
            logger.info(f"CASF-2016 core set: {len(casf_codes)} complexes")
            for split_name in splits:
                splits[split_name] = [
                    c for c in splits[split_name] if c not in casf_codes
                ]

    # Save splits
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits to {splits_path}")

    for name, codes in splits.items():
        logger.info(f"  {name}: {len(codes)} complexes")
        # Also save individual text files
        txt_path = output_dir / f"{name}_pockets.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(codes) + "\n")

    # Save CASF codes
    if casf_codes:
        casf_path = output_dir / "casf2016_pockets.txt"
        with open(casf_path, "w") as f:
            f.write("\n".join(sorted(casf_codes)) + "\n")
        logger.info(f"Saved CASF-2016 list to {casf_path}")

    # ── 4. Debug pocket list (3 pockets for local testing) ───────────────
    # Pick 3 test pockets with diverse pKd values
    test_codes = splits.get("test", list(df["pdb_code"][:10]))
    if len(test_codes) >= 3:
        test_df = df[df["pdb_code"].isin(test_codes)].sort_values("pkd")
        # Pick low/mid/high affinity
        indices = [0, len(test_df) // 2, len(test_df) - 1]
        debug_codes = [test_df.iloc[i]["pdb_code"] for i in indices]
    else:
        debug_codes = test_codes[:3]

    debug_path = output_dir / "debug_pockets.txt"
    with open(debug_path, "w") as f:
        f.write("\n".join(debug_codes) + "\n")
    logger.info(f"Debug pockets ({len(debug_codes)}): {debug_codes}")

    # ── 5. Extract pockets (optional) ────────────────────────────────────
    if args.extract_pockets:
        logger.info("Extracting 10Å pockets...")
        n_extracted = 0
        for pdb_code in existing:
            pocket_file = find_pocket_file(pdbbind_dir, pdb_code)
            if pocket_file and "pocket" in pocket_file.name:
                continue  # Already a pocket file

            ligand_file = find_ligand_file(pdbbind_dir, pdb_code)
            if ligand_file is None:
                continue

            protein_pdb = (
                pdbbind_dir / "refined-set" / pdb_code / f"{pdb_code}_protein.pdb"
            )
            if not protein_pdb.exists():
                continue

            out_pocket = (
                pdbbind_dir / "refined-set" / pdb_code / f"{pdb_code}_pocket10.pdb"
            )
            if out_pocket.exists():
                continue

            try:
                extract_pocket_from_protein(protein_pdb, ligand_file, out_pocket)
                n_extracted += 1
            except Exception as e:
                logger.warning(f"Failed to extract pocket for {pdb_code}: {e}")

        logger.info(f"Extracted {n_extracted} new pockets")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 0 data preparation complete!")
    logger.info(f"  Labels:       {labels_path}")
    logger.info(f"  Splits:       {splits_path}")
    logger.info(f"  Debug list:   {debug_path}")
    logger.info(f"  Total:        {len(df)} complexes")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Clone TargetDiff:  git clone https://github.com/guanjq/targetdiff.git external/targetdiff")
    logger.info("  2. Download weights:  cd external/targetdiff && mkdir -p pretrained_models")
    logger.info("     (get pretrained_diffusion.pt from the TargetDiff release page)")
    logger.info(f"  3. Debug sampling:    python scripts/02_sample_molecules.py \\")
    logger.info(f"       --pocket_list {debug_path} --num_samples 20 --device cpu")


if __name__ == "__main__":
    main()
