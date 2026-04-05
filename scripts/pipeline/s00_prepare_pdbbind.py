"""
scripts/pipeline/s00_prepare_pdbbind.py
───────────────────────────────────────
Prepare PDBbind v2020 refined set for supervised pretraining.

Stages:
  1. Parse INDEX file → labels DataFrame
  2. Extract 10Å pockets from protein PDB files
  3. Featurize each complex (pocket + ligand) using TargetDiff transforms
  4. Save each complex as a .pt file
  5. Create protein-family splits (mmseqs2 clustering)

Supports SLURM array jobs via --shard_index / --num_shards for HPC parallelism.
Within each shard, uses multiprocessing for additional speedup.

Usage (single node, all complexes):
    python scripts/pipeline/s00_prepare_pdbbind.py \\
        --pdbbind_dir data/pdbbind \\
        --output_dir data/pdbbind_v2020

Usage (HPC shard):
    python scripts/pipeline/s00_prepare_pdbbind.py \\
        --pdbbind_dir data/pdbbind \\
        --output_dir data/pdbbind_v2020 \\
        --shard_index 3 --num_shards 50 \\
        --stage featurize

Usage (merge after all shards):
    python scripts/pipeline/s00_prepare_pdbbind.py \\
        --pdbbind_dir data/pdbbind \\
        --output_dir data/pdbbind_v2020 \\
        --stage merge

Usage (split):
    python scripts/pipeline/s00_prepare_pdbbind.py \\
        --pdbbind_dir data/pdbbind \\
        --output_dir data/pdbbind_v2020 \\
        --stage split
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bayesdiff.data import (
    extract_pocket_from_protein,
    find_ligand_file,
    parse_pdbbind_index,
    protein_family_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("s00_prepare_pdbbind")


# ── Featurization helpers ────────────────────────────────────────────────────


def _parse_pdb_atoms(pdb_path: Path) -> dict:
    """Parse pocket PDB file into tensors expected by TargetDiff featurizers.

    Returns dict with keys: element, pos, is_backbone, atom_name, atom_to_aa_type.
    """
    from rdkit import Chem

    BACKBONE_ATOMS = {"N", "CA", "C", "O"}

    # Amino acid name → integer mapping (same as TargetDiff)
    AA_NAME_TO_IDX = {
        "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
        "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
        "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
        "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
        "UNK": 20,
    }

    # Element symbol → atomic number
    ELEMENT_TO_NUM = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
        "Cl": 17, "Se": 34, "Br": 35, "I": 53, "Fe": 26, "Zn": 30,
        "Mg": 12, "Ca": 20, "Mn": 25, "Co": 27, "Cu": 29, "Na": 11,
    }

    elements = []
    positions = []
    is_backbone = []
    atom_names = []
    aa_types = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # Element from columns 76-78 or infer from atom name
            elem_str = line[76:78].strip() if len(line) > 76 else ""
            if not elem_str:
                elem_str = atom_name[0]

            atomic_num = ELEMENT_TO_NUM.get(elem_str, 0)
            if atomic_num == 0:
                continue  # skip unknown elements

            elements.append(atomic_num)
            positions.append([x, y, z])
            is_backbone.append(atom_name in BACKBONE_ATOMS)
            atom_names.append(atom_name)
            aa_types.append(AA_NAME_TO_IDX.get(resname, AA_NAME_TO_IDX["UNK"]))

    if not elements:
        raise ValueError(f"No atoms parsed from {pdb_path}")

    return {
        "element": torch.tensor(elements, dtype=torch.long),
        "pos": torch.tensor(positions, dtype=torch.float32),
        "is_backbone": torch.tensor(is_backbone, dtype=torch.bool),
        "atom_name": atom_names,
        "atom_to_aa_type": torch.tensor(aa_types, dtype=torch.long),
    }


def _parse_ligand_sdf(sdf_path: Path) -> dict:
    """Parse ligand SDF file into tensors expected by TargetDiff featurizers.

    Returns dict with keys: element, pos, atom_feature, bond_index, bond_type.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=True)
    if mol is None:
        # Try mol2
        mol2_path = sdf_path.with_suffix(".mol2")
        if mol2_path.exists():
            mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True, sanitize=True)
    if mol is None:
        raise ValueError(f"Cannot read ligand: {sdf_path}")

    conf = mol.GetConformer()

    elements = []
    positions = []
    for atom in mol.GetAtoms():
        elements.append(atom.GetAtomicNum())
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])

    # Build atom features (8-dim: consistent with TargetDiff)
    # Features: [atomic_num_onehot(6), degree, formal_charge, ...]
    # Simplified: store atomic number and let featurizer handle it
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
        ]
        atom_features.append(feat)

    # Bond connectivity
    bond_indices = []
    bond_types = []
    BOND_TYPE_MAP = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        bond_indices.extend([[i, j], [j, i]])
        bond_types.extend([bt, bt])

    n_atoms = len(elements)
    result = {
        "element": torch.tensor(elements, dtype=torch.long),
        "pos": torch.tensor(positions, dtype=torch.float32),
        "atom_feature": torch.tensor(atom_features, dtype=torch.float32),
    }

    if bond_indices:
        result["bond_index"] = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
        result["bond_type"] = torch.tensor(bond_types, dtype=torch.long)
    else:
        result["bond_index"] = torch.empty([2, 0], dtype=torch.long)
        result["bond_type"] = torch.empty([0], dtype=torch.long)

    return result


def _process_one_complex(
    pdb_code: str,
    pdbbind_dir: Path,
    output_dir: Path,
    pkd_map: dict,
    radius: float = 10.0,
) -> dict:
    """Process a single PDBbind complex: extract pocket, featurize, save .pt.

    Returns a status dict with keys: pdb_code, status, error (if any).
    """
    out_pt = output_dir / f"{pdb_code}.pt"
    if out_pt.exists():
        return {"pdb_code": pdb_code, "status": "skipped", "error": None}

    try:
        base = pdbbind_dir / "refined-set" / pdb_code
        protein_pdb = base / f"{pdb_code}_protein.pdb"
        ligand_sdf = base / f"{pdb_code}_ligand.sdf"
        pocket_pdb = base / f"{pdb_code}_pocket.pdb"

        if not protein_pdb.exists():
            return {"pdb_code": pdb_code, "status": "failed", "error": "protein PDB not found"}
        if not ligand_sdf.exists():
            ligand_sdf = base / f"{pdb_code}_ligand.mol2"
            if not ligand_sdf.exists():
                return {"pdb_code": pdb_code, "status": "failed", "error": "ligand file not found"}

        # Extract pocket if not already present
        if not pocket_pdb.exists():
            pocket_pdb = base / f"{pdb_code}_pocket10.pdb"
            if not pocket_pdb.exists():
                extract_pocket_from_protein(protein_pdb, ligand_sdf, pocket_pdb, radius=radius)

        # Parse pocket atoms
        protein_data = _parse_pdb_atoms(pocket_pdb)

        # Parse ligand atoms
        ligand_data = _parse_ligand_sdf(ligand_sdf)

        # Get pKd label
        pkd = pkd_map.get(pdb_code, float("nan"))

        # Save as .pt
        data = {
            "pdb_code": pdb_code,
            "protein_element": protein_data["element"],
            "protein_pos": protein_data["pos"],
            "protein_is_backbone": protein_data["is_backbone"],
            "protein_atom_name": protein_data["atom_name"],
            "protein_atom_to_aa_type": protein_data["atom_to_aa_type"],
            "ligand_element": ligand_data["element"],
            "ligand_pos": ligand_data["pos"],
            "ligand_atom_feature": ligand_data["atom_feature"],
            "ligand_bond_index": ligand_data["bond_index"],
            "ligand_bond_type": ligand_data["bond_type"],
            "pkd": torch.tensor(pkd, dtype=torch.float32),
            "n_protein_atoms": protein_data["element"].shape[0],
            "n_ligand_atoms": ligand_data["element"].shape[0],
        }
        torch.save(data, out_pt)

        return {
            "pdb_code": pdb_code,
            "status": "ok",
            "error": None,
            "n_protein": protein_data["element"].shape[0],
            "n_ligand": ligand_data["element"].shape[0],
        }

    except Exception as e:
        return {"pdb_code": pdb_code, "status": "failed", "error": str(e)}


# ── Stages ───────────────────────────────────────────────────────────────────


def find_index_file(pdbbind_dir: Path) -> Path:
    """Locate the PDBbind INDEX file."""
    candidates = [
        pdbbind_dir / "INDEX_refined_data.2020",
        pdbbind_dir / "index" / "INDEX_refined_data.2020",
        pdbbind_dir / "refined-set" / "index" / "INDEX_refined_data.2020",
        pdbbind_dir / "INDEX_general_PL_data.2020",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find INDEX file. Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def stage_parse(pdbbind_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Stage 1: Parse INDEX file and save labels.csv."""
    index_path = find_index_file(pdbbind_dir)
    logger.info(f"Parsing INDEX file: {index_path}")

    df = parse_pdbbind_index(index_path)

    # Verify structural files exist
    existing = []
    missing = []
    for code in df["pdb_code"]:
        base = pdbbind_dir / "refined-set" / code
        if base.exists():
            existing.append(code)
        else:
            missing.append(code)

    logger.info(f"Found {len(existing)}/{len(df)} complexes with structural files")
    if missing:
        logger.warning(f"Missing {len(missing)} complexes (first 10): {missing[:10]}")

    df = df[df["pdb_code"].isin(existing)].reset_index(drop=True)

    # Save labels
    labels_path = output_dir / "labels.csv"
    df.to_csv(labels_path, index=False)
    logger.info(f"Saved {len(df)} labels → {labels_path}")
    logger.info(f"  pKd: [{df['pkd'].min():.2f}, {df['pkd'].max():.2f}], "
                f"mean={df['pkd'].mean():.2f}±{df['pkd'].std():.2f}")
    logger.info(f"  Affinity types: {df['affinity_type'].value_counts().to_dict()}")

    return df


def stage_featurize(
    pdbbind_dir: Path,
    output_dir: Path,
    df: pd.DataFrame,
    shard_index: int = 0,
    num_shards: int = 1,
    num_workers: int = 1,
    radius: float = 10.0,
) -> list[dict]:
    """Stage 2: Extract pockets + featurize complexes, optionally sharded."""
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdb_codes = sorted(df["pdb_code"].tolist())
    pkd_map = dict(zip(df["pdb_code"], df["pkd"]))

    # Shard the codes
    shard_codes = [c for i, c in enumerate(pdb_codes) if i % num_shards == shard_index]
    logger.info(
        f"Shard {shard_index}/{num_shards}: processing {len(shard_codes)} / {len(pdb_codes)} complexes "
        f"with {num_workers} workers"
    )

    process_fn = partial(
        _process_one_complex,
        pdbbind_dir=pdbbind_dir,
        output_dir=processed_dir,
        pkd_map=pkd_map,
        radius=radius,
    )

    results = []
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_fn, shard_codes)):
                results.append(res)
                if (i + 1) % 100 == 0 or (i + 1) == len(shard_codes):
                    n_ok = sum(1 for r in results if r["status"] == "ok")
                    n_skip = sum(1 for r in results if r["status"] == "skipped")
                    n_fail = sum(1 for r in results if r["status"] == "failed")
                    logger.info(f"  [{i+1}/{len(shard_codes)}] ok={n_ok} skipped={n_skip} failed={n_fail}")
    else:
        for i, code in enumerate(shard_codes):
            res = process_fn(code)
            results.append(res)
            if (i + 1) % 100 == 0 or (i + 1) == len(shard_codes):
                n_ok = sum(1 for r in results if r["status"] == "ok")
                n_skip = sum(1 for r in results if r["status"] == "skipped")
                n_fail = sum(1 for r in results if r["status"] == "failed")
                logger.info(f"  [{i+1}/{len(shard_codes)}] ok={n_ok} skipped={n_skip} failed={n_fail}")

    # Save shard status
    status_path = output_dir / f"shard_status_{shard_index}of{num_shards}.json"
    with open(status_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Shard status → {status_path}")

    # Summary
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    logger.info(f"Shard {shard_index} complete: ok={n_ok}, skipped={n_skip}, failed={n_fail}")

    if n_fail > 0:
        failed = [r for r in results if r["status"] == "failed"]
        for r in failed[:10]:
            logger.warning(f"  FAILED {r['pdb_code']}: {r['error']}")

    return results


def stage_merge(output_dir: Path) -> pd.DataFrame:
    """Stage 3: Merge shard status files and create summary."""
    all_results = []
    for status_file in sorted(output_dir.glob("shard_status_*.json")):
        with open(status_file) as f:
            all_results.extend(json.load(f))

    if not all_results:
        # No shard files — check processed directory directly
        processed_dir = output_dir / "processed"
        if processed_dir.exists():
            pt_files = list(processed_dir.glob("*.pt"))
            logger.info(f"No shard status files; found {len(pt_files)} .pt files in processed/")
            all_results = [{"pdb_code": p.stem, "status": "ok", "error": None} for p in pt_files]

    n_ok = sum(1 for r in all_results if r["status"] == "ok")
    n_skip = sum(1 for r in all_results if r["status"] == "skipped")
    n_fail = sum(1 for r in all_results if r["status"] == "failed")
    logger.info(f"Merged results: total={len(all_results)}, ok={n_ok}, skipped={n_skip}, failed={n_fail}")

    # Save quality report
    failed_df = pd.DataFrame([r for r in all_results if r["status"] == "failed"])
    if not failed_df.empty:
        report_path = output_dir / "failed_complexes.csv"
        failed_df.to_csv(report_path, index=False)
        logger.warning(f"Failed complexes → {report_path}")

    # Create list of successfully processed codes
    ok_codes = sorted(r["pdb_code"] for r in all_results if r["status"] in ("ok", "skipped"))
    manifest_path = output_dir / "processed_codes.txt"
    with open(manifest_path, "w") as f:
        f.write("\n".join(ok_codes) + "\n")
    logger.info(f"Processed codes ({len(ok_codes)}) → {manifest_path}")

    return ok_codes


def stage_split(
    pdbbind_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Stage 4: Create protein-family splits."""
    labels_path = output_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}. Run --stage parse first.")
    df = pd.read_csv(labels_path)

    # Filter to only successfully processed complexes
    manifest_path = output_dir / "processed_codes.txt"
    if manifest_path.exists():
        ok_codes = set(manifest_path.read_text().strip().split("\n"))
        before = len(df)
        df = df[df["pdb_code"].isin(ok_codes)].reset_index(drop=True)
        logger.info(f"Filtered labels: {before} → {len(df)} (matched processed codes)")

    logger.info(f"Creating protein-family splits for {len(df)} complexes (seed={seed})...")
    splits = protein_family_split(df, pdbbind_dir, seed=seed)

    # Save splits
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Splits → {splits_path}")

    for name, codes in splits.items():
        logger.info(f"  {name}: {len(codes)} complexes")
        # Per-split pKd stats
        split_df = df[df["pdb_code"].isin(codes)]
        if not split_df.empty:
            logger.info(f"    pKd: [{split_df['pkd'].min():.2f}, {split_df['pkd'].max():.2f}], "
                        f"mean={split_df['pkd'].mean():.2f}±{split_df['pkd'].std():.2f}")

    # Also save filtered labels
    labels_filtered_path = output_dir / "labels.csv"
    df.to_csv(labels_filtered_path, index=False)

    return splits


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PDBbind v2020 refined set for supervised pretraining"
    )
    parser.add_argument(
        "--pdbbind_dir", type=str, required=True,
        help="Path to PDBbind v2020 root (containing refined-set/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/pdbbind_v2020",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "parse", "featurize", "merge", "split"],
        help="Which stage to run (default: all)",
    )
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of multiprocessing workers (0=auto, 1=sequential)",
    )
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pdbbind_dir = Path(args.pdbbind_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = min(mp.cpu_count(), 32)

    stages = (
        ["parse", "featurize", "merge", "split"]
        if args.stage == "all"
        else [args.stage]
    )

    # Parse stage (always needed for featurize)
    labels_path = output_dir / "labels.csv"
    df = None

    if "parse" in stages:
        df = stage_parse(pdbbind_dir, output_dir)
    elif labels_path.exists():
        df = pd.read_csv(labels_path)
    else:
        # Need to parse first
        df = stage_parse(pdbbind_dir, output_dir)

    if "featurize" in stages:
        stage_featurize(
            pdbbind_dir, output_dir, df,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            num_workers=num_workers,
            radius=args.radius,
        )

    if "merge" in stages:
        stage_merge(output_dir)

    if "split" in stages:
        stage_split(pdbbind_dir, output_dir, seed=args.seed)

    logger.info("Done!")


if __name__ == "__main__":
    main()
