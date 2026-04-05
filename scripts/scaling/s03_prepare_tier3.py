#!/usr/bin/env python
"""
scripts/scaling/s03_prepare_tier3.py
───────────────────────────
Prepare pocket data for Tier 3 dataset construction.

Steps:
  1. Scan LMDB for all pocket families
  2. Match with affinity_info.pkl for pKd labels
  3. Inventory test set pockets (existing PDB files)
  4. Save pre-processed pocket Data objects as .pt files
  5. Generate comprehensive pocket list JSON

Run on login node (no GPU needed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TD_DIR = PROJECT_ROOT / "external" / "targetdiff"
LMDB_PATH = TD_DIR / "data" / "data" / "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
AFFINITY_PATH = TD_DIR / "data" / "affinity_info.pkl"
TEST_SET_DIR = TD_DIR / "data" / "test_set"
SPLIT_PATH = TD_DIR / "data" / "split_by_name.pt"

# Output directories
POCKET_DATA_DIR = PROJECT_ROOT / "data" / "tier3_pockets"
RESULTS_DIR = PROJECT_ROOT / "results" / "tier3_sampling"


def load_affinity_info() -> dict:
    """Load affinity_info.pkl → {family_name: pKd_value}."""
    with open(AFFINITY_PATH, "rb") as f:
        raw = pickle.load(f)

    # Build family → best pKd mapping
    # Values are dicts with keys: rmsd, pk, vina
    family_pk = {}
    for key, info in raw.items():
        if isinstance(info, dict):
            pk = info.get("pk", 0)
        else:
            pk = info
        if pk is None or pk == 0:
            continue
        # key format: "FAMILY_NAME/pdb_rec_lig..."
        if "/" in key:
            family = key.split("/")[0]
        else:
            family = key
        # Keep the highest pKd per family (most potent)
        if family not in family_pk or pk > family_pk[family]:
            family_pk[family] = float(pk)

    logger.info(f"Affinity info: {len(raw)} total entries, {len(family_pk)} families with pKd")
    return family_pk


def scan_lmdb() -> dict:
    """Scan LMDB to build family → [entry_keys] mapping.

    Returns dict: {family_name: [(key, protein_filename), ...]}
    """
    logger.info(f"Scanning LMDB: {LMDB_PATH}")
    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, subdir=False,
                     map_size=10 * 1024**3)

    family_entries = defaultdict(list)
    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0
        for key_bytes, val_bytes in cursor:
            data = pickle.loads(val_bytes)
            pf = data.get("protein_filename", "")
            family = pf.split("/")[0] if "/" in pf else pf
            family_entries[family].append((key_bytes.decode(), pf))
            count += 1
            if count % 20000 == 0:
                logger.info(f"  Scanned {count} entries, {len(family_entries)} families...")

    env.close()
    logger.info(f"LMDB scan complete: {count} entries, {len(family_entries)} families")
    return dict(family_entries)


def extract_pocket_data(family_entries: dict, labeled_families: set) -> dict:
    """Extract one representative pocket Data for each labeled family.

    Returns dict: {family: {key, protein_filename, n_atoms}}
    """
    logger.info(f"Extracting pocket data for {len(labeled_families)} labeled families...")
    POCKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, subdir=False,
                     map_size=10 * 1024**3)

    extracted = {}
    with env.begin() as txn:
        for i, family in enumerate(sorted(labeled_families)):
            if family not in family_entries:
                continue

            # Pick the first entry as representative
            entry_key, protein_filename = family_entries[family][0]
            raw = txn.get(entry_key.encode())
            if raw is None:
                logger.warning(f"  Key {entry_key} not found for family {family}")
                continue

            data_dict = pickle.loads(raw)

            # Build PyG-compatible pocket data dict
            # Keep only protein fields (sampling generates ligand from scratch)
            pocket_data = {
                "protein_element": data_dict["protein_element"],
                "protein_pos": data_dict["protein_pos"],
                "protein_is_backbone": data_dict["protein_is_backbone"],
                "protein_atom_name": data_dict["protein_atom_name"],
                "protein_atom_to_aa_type": data_dict["protein_atom_to_aa_type"],
                "protein_molecule_name": data_dict.get("protein_molecule_name", "pocket"),
                "protein_filename": protein_filename,
            }

            # Save as .pt file
            out_path = POCKET_DATA_DIR / f"{family}.pt"
            torch.save(pocket_data, out_path)

            n_atoms = data_dict["protein_element"].shape[0]
            extracted[family] = {
                "key": entry_key,
                "protein_filename": protein_filename,
                "n_atoms": n_atoms,
                "data_path": str(out_path),
            }

            if (i + 1) % 100 == 0:
                logger.info(f"  Extracted {i + 1}/{len(labeled_families)} families...")

    env.close()
    logger.info(f"Extracted {len(extracted)} pocket data files to {POCKET_DATA_DIR}")
    return extracted


def inventory_test_set() -> dict:
    """Inventory test set pockets with PDB files.

    Returns dict: {pdb_code: pdb_path}
    """
    test_pockets = {}
    if not TEST_SET_DIR.exists():
        logger.warning(f"Test set directory not found: {TEST_SET_DIR}")
        return test_pockets

    for d in sorted(TEST_SET_DIR.iterdir()):
        if not d.is_dir():
            continue
        pdb_code = d.name
        # Find pocket PDB file
        pdb_files = list(d.glob("*_pocket10.pdb")) + list(d.glob("*_rec.pdb"))
        if pdb_files:
            test_pockets[pdb_code] = str(pdb_files[0])

    logger.info(f"Test set: {len(test_pockets)} pockets with PDB files")
    return test_pockets


def check_existing_sdfs() -> set:
    """Check which pockets already have SDF sampling results."""
    existing = set()
    sdf_dir = PROJECT_ROOT / "results" / "sampling"
    if sdf_dir.exists():
        for sdf_file in sdf_dir.glob("*/molecules.sdf"):
            pocket = sdf_file.parent.name
            # Check if SDF is non-empty
            if sdf_file.stat().st_size > 100:
                existing.add(pocket)

    # Also check tier3 results
    if RESULTS_DIR.exists():
        for sdf_file in RESULTS_DIR.glob("*/molecules.sdf"):
            pocket = sdf_file.parent.name
            if sdf_file.stat().st_size > 100:
                existing.add(pocket)

    logger.info(f"Existing SDF results: {len(existing)} pockets")
    return existing


def build_pocket_list(
    family_pk: dict,
    lmdb_families: dict,
    extracted: dict,
    test_pockets: dict,
    existing_sdfs: set,
) -> list:
    """Build comprehensive pocket list with metadata.

    Returns list of dicts, each with:
      - family: pocket family name
      - pKd: affinity value
      - source: 'test_set' or 'lmdb'
      - pdb_path: path to PDB file (for test_set)
      - data_path: path to .pt file (for LMDB)
      - has_existing_sdf: bool
      - n_atoms: protein atom count (for LMDB pockets)
    """
    pocket_list = []

    # Test set pockets with pKd
    for pdb_code, pdb_path in test_pockets.items():
        if pdb_code in family_pk:
            pocket_list.append({
                "family": pdb_code,
                "pKd": family_pk[pdb_code],
                "source": "test_set",
                "pdb_path": pdb_path,
                "data_path": None,
                "has_existing_sdf": pdb_code in existing_sdfs,
                "n_atoms": None,
            })

    # LMDB families with pKd
    for family in sorted(extracted.keys()):
        if family in family_pk:
            info = extracted[family]
            pocket_list.append({
                "family": family,
                "pKd": family_pk[family],
                "source": "lmdb",
                "pdb_path": None,
                "data_path": info["data_path"],
                "has_existing_sdf": family in existing_sdfs,
                "n_atoms": info["n_atoms"],
            })

    # Sort by source (test first) then family name
    pocket_list.sort(key=lambda x: (0 if x["source"] == "test_set" else 1, x["family"]))

    logger.info(
        f"Pocket list: {len(pocket_list)} total "
        f"({sum(1 for p in pocket_list if p['source'] == 'test_set')} test, "
        f"{sum(1 for p in pocket_list if p['source'] == 'lmdb')} LMDB)"
    )

    # Stats
    needs_generation = [p for p in pocket_list if not p["has_existing_sdf"]]
    logger.info(f"  Need generation: {len(needs_generation)} pockets")
    logger.info(f"  Already have SDF: {len(pocket_list) - len(needs_generation)} pockets")

    return pocket_list


def main():
    parser = argparse.ArgumentParser(description="Prepare Tier 3 pocket data")
    parser.add_argument("--skip-lmdb-scan", action="store_true",
                        help="Skip LMDB scanning (use cached results)")
    parser.add_argument("--output-json", type=str,
                        default=str(PROJECT_ROOT / "data" / "tier3_pocket_list.json"))
    args = parser.parse_args()

    # Step 1: Load affinity info
    family_pk = load_affinity_info()

    # Step 2: Scan LMDB
    lmdb_families = scan_lmdb()

    # Step 3: Find labeled families in LMDB
    lmdb_labeled = set(lmdb_families.keys()) & set(family_pk.keys())
    logger.info(f"LMDB families with pKd: {len(lmdb_labeled)}")

    # Step 4: Extract pocket data for labeled families
    extracted = extract_pocket_data(lmdb_families, lmdb_labeled)

    # Step 5: Inventory test set
    test_pockets = inventory_test_set()

    # Step 6: Check existing results
    existing_sdfs = check_existing_sdfs()

    # Step 7: Build comprehensive pocket list
    pocket_list = build_pocket_list(
        family_pk, lmdb_families, extracted, test_pockets, existing_sdfs
    )

    # Step 8: Save pocket list JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_total": len(pocket_list),
            "n_test": sum(1 for p in pocket_list if p["source"] == "test_set"),
            "n_lmdb": sum(1 for p in pocket_list if p["source"] == "lmdb"),
            "n_need_generation": sum(1 for p in pocket_list if not p["has_existing_sdf"]),
            "pockets": pocket_list,
        }, f, indent=2)
    logger.info(f"Saved pocket list to {output_path}")

    # Step 9: Summary
    pk_values = [p["pKd"] for p in pocket_list]
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total pockets with pKd: {len(pocket_list)}")
    logger.info(f"  Test set: {sum(1 for p in pocket_list if p['source'] == 'test_set')}")
    logger.info(f"  LMDB (train): {sum(1 for p in pocket_list if p['source'] == 'lmdb')}")
    logger.info(f"  Need generation: {sum(1 for p in pocket_list if not p['has_existing_sdf'])}")
    logger.info(f"  pKd range: [{min(pk_values):.2f}, {max(pk_values):.2f}]")
    logger.info(f"  pKd mean ± std: {np.mean(pk_values):.2f} ± {np.std(pk_values):.2f}")


if __name__ == "__main__":
    main()
