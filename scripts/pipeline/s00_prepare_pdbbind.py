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
    cluster_stratified_split,
    extract_pocket_from_protein,
    load_casf2016_codes,
    parse_casf_coreset,
    parse_pdbbind_index,
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


def build_pdb_code_to_dir(pdbbind_dir: Path) -> dict[str, Path]:
    """Scan PDBbind directory to build {pdb_code: path_to_complex_dir} mapping.

    Supports:
      - P-L/YEAR_RANGE/CODE/  (R1 format)
      - refined-set/CODE/     (legacy format)
    """
    mapping: dict[str, Path] = {}
    # R1 layout: P-L/1981-2000/XXXX/
    pl_dir = pdbbind_dir / "P-L"
    if pl_dir.exists():
        for yr_dir in sorted(pl_dir.iterdir()):
            if yr_dir.is_dir():
                for code_dir in yr_dir.iterdir():
                    if code_dir.is_dir() and len(code_dir.name) == 4:
                        mapping[code_dir.name.lower()] = code_dir
    # Legacy layout: refined-set/XXXX/
    refined_dir = pdbbind_dir / "refined-set"
    if refined_dir.exists():
        for code_dir in refined_dir.iterdir():
            if code_dir.is_dir() and len(code_dir.name) == 4:
                code = code_dir.name.lower()
                if code not in mapping:
                    mapping[code] = code_dir
    logger.info(f"Found {len(mapping)} complex directories in {pdbbind_dir}")
    return mapping


def _process_one_complex(
    pdb_code: str,
    code_to_dir: dict[str, Path],
    output_dir: Path,
    pkd_map: dict,
    radius: float = 10.0,
) -> dict:
    """Process a single PDBbind complex: extract pocket, featurize, save .pt."""
    out_pt = output_dir / f"{pdb_code}.pt"
    if out_pt.exists():
        return {"pdb_code": pdb_code, "status": "skipped", "error": None}

    try:
        base = code_to_dir.get(pdb_code)
        if base is None:
            return {"pdb_code": pdb_code, "status": "failed", "error": "directory not found"}

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
                pocket_pdb = output_dir / f"_pocket_{pdb_code}.pdb"
                extract_pocket_from_protein(protein_pdb, ligand_sdf, pocket_pdb, radius=radius)

        protein_data = _parse_pdb_atoms(pocket_pdb)
        ligand_data = _parse_ligand_sdf(ligand_sdf)
        pkd = pkd_map.get(pdb_code, float("nan"))

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

        # Clean up temp pocket
        temp_pocket = output_dir / f"_pocket_{pdb_code}.pdb"
        if temp_pocket.exists():
            temp_pocket.unlink()

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
        pdbbind_dir / "index" / "INDEX_general_PL.2020R1.lst",
        pdbbind_dir / "INDEX_general_PL.2020R1.lst",
        pdbbind_dir / "INDEX_refined_data.2020",
        pdbbind_dir / "index" / "INDEX_refined_data.2020",
        pdbbind_dir / "refined-set" / "index" / "INDEX_refined_data.2020",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find INDEX file. Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def find_casf_dir(pdbbind_dir: Path) -> Path:
    """Locate CASF-2016 directory."""
    candidates = [
        pdbbind_dir / "CASF-2016",
        pdbbind_dir.parent / "CASF-2016",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find CASF-2016 directory")


def stage_parse(pdbbind_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Stage 1: Parse INDEX + CASF-2016, produce labels.csv with source column."""
    index_path = find_index_file(pdbbind_dir)
    logger.info(f"Parsing INDEX file: {index_path}")
    df = parse_pdbbind_index(index_path, keep_inexact=False, keep_ic50=True)
    logger.info(f"  Valid entries: {len(df)}")
    logger.info(f"  pKd: [{df['pkd'].min():.2f}, {df['pkd'].max():.2f}], "
                f"mean={df['pkd'].mean():.2f}±{df['pkd'].std():.2f}")
    logger.info(f"  Affinity types: {df['affinity_type'].value_counts().to_dict()}")

    # Build code→dir mapping, filter to complexes with structural files
    code_to_dir = build_pdb_code_to_dir(pdbbind_dir)
    before = len(df)
    df = df[df["pdb_code"].isin(code_to_dir)].reset_index(drop=True)
    logger.info(f"  With structural files: {len(df)}/{before}")

    df["source"] = "pdbbind"

    # Parse CASF-2016 (optional — graceful if missing)
    try:
        casf_dir = find_casf_dir(pdbbind_dir)
        casf_codes = set(load_casf2016_codes(casf_dir))
        logger.info(f"  CASF-2016 core set: {len(casf_codes)} PDB codes")

        casf_in_pdbbind = casf_codes & set(df["pdb_code"])
        logger.info(f"  CASF codes found in PDBbind: {len(casf_in_pdbbind)}")
        df.loc[df["pdb_code"].isin(casf_codes), "source"] = "casf_test"

        # Also parse CASF CoreSet.dat for any codes missing from INDEX
        coreset_dat = casf_dir / "power_screening" / "CoreSet.dat"
        if coreset_dat.exists():
            casf_df = parse_casf_coreset(coreset_dat)
            missing_casf = set(casf_df["pdb_code"]) - set(df["pdb_code"])
            if missing_casf:
                logger.warning(f"  {len(missing_casf)} CASF codes not in PDBbind INDEX")
                for code in missing_casf:
                    if code in code_to_dir:
                        row = casf_df[casf_df["pdb_code"] == code].iloc[0]
                        new_row = {
                            "pdb_code": code, "resolution": row["resolution"],
                            "year": row["year"], "affinity_type": row.get("affinity_type", "Kd"),
                            "affinity_value_M": np.nan, "pkd": row["pkd"],
                            "is_exact": True, "source": "casf_test",
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        logger.warning("  CASF-2016 directory not found; all codes marked as pdbbind")

    labels_path = output_dir / "labels.csv"
    df.to_csv(labels_path, index=False)
    logger.info(f"Saved {len(df)} labels → {labels_path}")
    logger.info(f"  Train pool: {(df['source'] == 'pdbbind').sum()}")
    logger.info(f"  Test (CASF): {(df['source'] == 'casf_test').sum()}")

    # Save code_to_dir mapping for featurize stage
    dir_map = {k: str(v) for k, v in code_to_dir.items()}
    with open(output_dir / "code_to_dir.json", "w") as f:
        json.dump(dir_map, f)

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

    # Load code_to_dir mapping
    dir_map_path = output_dir / "code_to_dir.json"
    if dir_map_path.exists():
        with open(dir_map_path) as f:
            code_to_dir = {k: Path(v) for k, v in json.load(f).items()}
    else:
        code_to_dir = build_pdb_code_to_dir(pdbbind_dir)

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
        code_to_dir=code_to_dir,
        output_dir=processed_dir,
        pkd_map=pkd_map,
        radius=radius,
    )

    results = []
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_fn, shard_codes)):
                results.append(res)
                if (i + 1) % 200 == 0 or (i + 1) == len(shard_codes):
                    n_ok = sum(1 for r in results if r["status"] == "ok")
                    n_skip = sum(1 for r in results if r["status"] == "skipped")
                    n_fail = sum(1 for r in results if r["status"] == "failed")
                    logger.info(f"  [{i+1}/{len(shard_codes)}] ok={n_ok} skipped={n_skip} failed={n_fail}")
    else:
        for i, code in enumerate(shard_codes):
            res = process_fn(code)
            results.append(res)
            if (i + 1) % 200 == 0 or (i + 1) == len(shard_codes):
                n_ok = sum(1 for r in results if r["status"] == "ok")
                n_skip = sum(1 for r in results if r["status"] == "skipped")
                n_fail = sum(1 for r in results if r["status"] == "failed")
                logger.info(f"  [{i+1}/{len(shard_codes)}] ok={n_ok} skipped={n_skip} failed={n_fail}")

    # Save shard status
    status_path = output_dir / f"shard_status_{shard_index:04d}of{num_shards:04d}.json"
    with open(status_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Shard status → {status_path}")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    logger.info(f"Shard {shard_index} complete: ok={n_ok}, failed={n_fail}")

    if n_fail > 0:
        for r in [r for r in results if r["status"] == "failed"][:10]:
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
    val_frac: float = 0.12,
    seed: int = 42,
    n_folds: int = 5,
) -> dict:
    """Stage 4: Cluster-stratified Train/Val + CASF-2016 Test split.

    Generates both:
      - splits.json: default single split (backward-compatible, = fold 0)
      - splits_5fold.json: N-fold grouped splits for robustness evaluation
    """
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

    # Separate CASF test set
    test_codes = sorted(df[df["source"] == "casf_test"]["pdb_code"].tolist())
    train_pool = df[df["source"] != "casf_test"].reset_index(drop=True)
    logger.info(f"Test (CASF-2016): {len(test_codes)} complexes")
    logger.info(f"Train pool: {len(train_pool)} complexes")

    # Load code_to_dir for finding protein PDBs
    dir_map_path = output_dir / "code_to_dir.json"
    pdb_to_file = None
    if dir_map_path.exists():
        with open(dir_map_path) as f:
            code_to_dir = {k: Path(v) for k, v in json.load(f).items()}
        pdb_to_file = {
            code: d / f"{code}_protein.pdb"
            for code, d in code_to_dir.items()
        }

    # Cluster-stratified train/val split
    tv_splits = cluster_stratified_split(
        train_pool, pdbbind_dir, val_frac=val_frac, seed=seed,
        pdb_to_file=pdb_to_file,
    )

    splits = {
        "train": tv_splits["train"],
        "val": tv_splits["val"],
        "test": test_codes,
    }

    # Log stats
    for name, codes in splits.items():
        split_df = df[df["pdb_code"].isin(codes)]
        if not split_df.empty:
            logger.info(
                f"  {name}: {len(codes)} complexes, "
                f"pKd=[{split_df['pkd'].min():.2f}, {split_df['pkd'].max():.2f}], "
                f"mean={split_df['pkd'].mean():.2f}±{split_df['pkd'].std():.2f}"
            )

    # Verify no overlap
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert not (train_set & val_set), "Train/Val overlap!"
    assert not (train_set & test_set), "Train/Test overlap!"
    assert not (val_set & test_set), "Val/Test overlap!"
    logger.info("No overlap between splits")

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Splits → {splits_path}")

    # ── Write cluster files ──────────────────────────────────────────────
    cluster_info = tv_splits.get("_cluster_info", [])
    clusters_map = tv_splits.get("_clusters", {})

    if cluster_info:
        # clusters.json: {cluster_id: [pdb_code, ...]}
        clusters_json = {str(c["cluster_id"]): c["pdbs"] for c in cluster_info}
        clusters_path = output_dir / "clusters.json"
        with open(clusters_path, "w") as f:
            json.dump(clusters_json, f, indent=2)
        logger.info(f"Clusters ({len(clusters_json)}) → {clusters_path}")

        # cluster_assignments.csv: pdb_code, cluster_id, cluster_median_pkd, pkd_bin
        rows = []
        # Build quick lookup for cluster metadata
        cid_to_info = {c["cluster_id"]: c for c in cluster_info}
        for pdb_code, cid in sorted(clusters_map.items()):
            info = cid_to_info.get(cid, {})
            rows.append({
                "pdb_code": pdb_code,
                "cluster_id": cid,
                "cluster_median_pkd": info.get("median_pkd", np.nan),
                "pkd_bin": info.get("bin", -1),
            })
        ca_df = pd.DataFrame(rows)
        ca_path = output_dir / "cluster_assignments.csv"
        ca_df.to_csv(ca_path, index=False)
        logger.info(f"Cluster assignments ({len(ca_df)}) → {ca_path}")

    df.to_csv(labels_path, index=False)

    # ── Generate 5-fold grouped splits ───────────────────────────────────
    from bayesdiff.data import cluster_stratified_split_nfold

    clusters_path = output_dir / "clusters.json"
    if clusters_path.exists() and n_folds > 1:
        logger.info(f"Generating {n_folds}-fold grouped Train/Val splits...")
        nfold_result = cluster_stratified_split_nfold(
            df=train_pool,
            n_folds=n_folds,
            val_frac=val_frac,
            n_bins=10,
            base_seed=seed,
            clusters_json=clusters_path,
        )

        # Add fixed test set
        nfold_output = {
            "test": test_codes,
            "folds": nfold_result["folds"],
            "cluster_info": nfold_result["cluster_info"],
        }

        # Verify each fold
        for fid, fold in nfold_output["folds"].items():
            f_train = set(fold["train"])
            f_val = set(fold["val"])
            assert not (f_train & f_val), f"Fold {fid}: Train/Val overlap!"
            assert not (f_train & test_set), f"Fold {fid}: Train/Test overlap!"
            assert not (f_val & test_set), f"Fold {fid}: Val/Test overlap!"

        logger.info(f"All {n_folds} folds verified: no overlaps")

        fivefold_path = output_dir / "splits_5fold.json"
        with open(fivefold_path, "w") as f:
            json.dump(nfold_output, f, indent=2)
        logger.info(f"5-fold splits → {fivefold_path}")
    elif n_folds > 1:
        logger.warning("clusters.json not found; skipping 5-fold split generation")

    return splits


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PDBbind v2020 R1 + CASF-2016 for supervised pretraining"
    )
    parser.add_argument(
        "--pdbbind_dir", type=str, required=True,
        help="Path to PDBbind data root (containing P-L/, index/, CASF-2016/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/pdbbind_v2020",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "parse", "featurize", "merge", "split"],
    )
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="CPU workers per shard (0=auto)",
    )
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--val_frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of grouped Train/Val folds to generate (default: 5)",
    )
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
        stage_split(pdbbind_dir, output_dir, val_frac=args.val_frac, seed=args.seed,
                    n_folds=args.n_folds)

    logger.info("Done!")


if __name__ == "__main__":
    main()
