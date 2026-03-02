"""
Data loading, PDBbind INDEX parsing, protein-family split, and label transforms.

Usage:
    python scripts/01_prepare_data.py --pdbbind_dir data/pdbbind --output_dir data/splits
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RT_KCAL = 0.592  # RT at 298.15 K in kcal/mol  (R = 1.987e-3 kcal/(mol·K))

AFFINITY_TYPE_MAP = {"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"}

# Mapping from string like "Kd=5.9nM" → float in Molar
_UNIT_SCALE = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}


# ── INDEX file parsing ───────────────────────────────────────────────────────


def parse_pdbbind_index(index_path: str | Path) -> pd.DataFrame:
    """Parse PDBbind INDEX_refined_data.2020 (or general) into a DataFrame.

    Expected line format (space-separated, // starts comment):
        3ug2  1.80  2011  8.22  Kd=6nM  // reference info ...

    Returns DataFrame with columns:
        pdb_code, resolution, year, pka, affinity_type, affinity_value_M, pkd
    """
    records = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # strip comment after //
            data_part = line.split("//")[0].strip()
            parts = data_part.split()
            if len(parts) < 5:
                continue

            pdb_code = parts[0]
            resolution = float(parts[1]) if parts[1] != "NMR" else np.nan
            year = int(parts[2])
            pka = float(parts[3])  # -log10(affinity)
            kv_str = parts[4]  # e.g. "Kd=5.9nM", "Ki=3.5uM"

            # Parse affinity type and value
            aff_type, aff_val_M = _parse_affinity_string(kv_str)

            records.append(
                {
                    "pdb_code": pdb_code,
                    "resolution": resolution,
                    "year": year,
                    "pka": pka,
                    "affinity_type": aff_type,
                    "affinity_value_M": aff_val_M,
                    "pkd": pka,  # already -log10, treat as pKd/pKi/pIC50
                }
            )

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} entries from {index_path}")
    return df


def _parse_affinity_string(kv_str: str) -> tuple[str, float]:
    """Parse 'Kd=5.9nM' → ('Kd', 5.9e-9)."""
    match = re.match(r"(Kd|Ki|IC50)[~<>=]*([0-9.eE+-]+)(\w+)", kv_str)
    if match:
        aff_type = match.group(1)
        value = float(match.group(2))
        unit = match.group(3)
        scale = _UNIT_SCALE.get(unit, 1.0)
        return aff_type, value * scale
    return "unknown", np.nan


# ── Protein-family split ─────────────────────────────────────────────────────


def protein_family_split(
    df: pd.DataFrame,
    pdbbind_dir: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    cal_frac: float = 0.10,
    test_frac: float = 0.10,
    seq_identity: float = 0.30,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split PDB codes by protein family clustering.

    Uses mmseqs2 if available, otherwise falls back to time-based split.
    Returns dict with keys: train, val, cal, test → lists of PDB codes.
    """
    pdb_codes = df["pdb_code"].unique().tolist()

    # Try mmseqs2-based clustering
    clusters = _try_mmseqs_cluster(pdb_codes, pdbbind_dir, seq_identity)

    if clusters is not None:
        return _split_by_clusters(
            clusters, train_frac, val_frac, cal_frac, test_frac, seed
        )
    else:
        logger.warning(
            "mmseqs2 not found. Falling back to time-based split (year < 2016 train, ≥ 2019 test)."
        )
        return _time_based_split(df, seed)


def _try_mmseqs_cluster(
    pdb_codes: list[str],
    pdbbind_dir: str | Path,
    seq_identity: float,
) -> Optional[dict[str, int]]:
    """Try to cluster proteins by sequence identity using mmseqs2.

    Returns {pdb_code: cluster_id} or None if mmseqs2 not available.
    """
    # Check if mmseqs is available
    try:
        subprocess.run(["mmseqs", "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    pdbbind_dir = Path(pdbbind_dir)

    # Extract sequences from PDB files (simple CA extraction)
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "proteins.fasta"
        _extract_sequences_to_fasta(pdb_codes, pdbbind_dir, fasta_path)

        if not fasta_path.exists() or fasta_path.stat().st_size == 0:
            logger.warning("No sequences extracted; skipping mmseqs clustering.")
            return None

        # Run mmseqs2
        db_path = Path(tmpdir) / "seqDB"
        clu_path = Path(tmpdir) / "cluDB"
        tsv_path = Path(tmpdir) / "clusters.tsv"

        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_path)],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                "mmseqs",
                "easy-cluster",
                str(fasta_path),
                str(clu_path),
                str(tmpdir),
                "--min-seq-id",
                str(seq_identity),
            ],
            capture_output=True,
            check=True,
        )

        # Parse cluster TSV
        tsv_file = clu_path.with_name(clu_path.name + "_cluster.tsv")
        if not tsv_file.exists():
            # Try alternate naming
            for f in Path(tmpdir).glob("*cluster.tsv"):
                tsv_file = f
                break

        if not tsv_file.exists():
            return None

        clusters = {}
        cluster_id = 0
        current_rep = None
        with open(tsv_file) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if rep != current_rep:
                    current_rep = rep
                    cluster_id += 1
                clusters[member] = cluster_id

        logger.info(
            f"mmseqs2 clustering: {len(set(clusters.values()))} clusters from {len(clusters)} sequences"
        )
        return clusters


def _extract_sequences_to_fasta(
    pdb_codes: list[str], pdbbind_dir: Path, output_path: Path
) -> None:
    """Extract protein sequences from PDB files into a FASTA file."""
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    with open(output_path, "w") as out:
        for pdb_code in pdb_codes:
            protein_pdb = pdbbind_dir / "refined-set" / pdb_code / f"{pdb_code}_protein.pdb"
            if not protein_pdb.exists():
                continue
            # Extract CA atoms → sequence
            residues = []
            seen = set()
            with open(protein_pdb) as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        chain = line[21]
                        resi = line[22:27].strip()
                        resn = line[17:20].strip()
                        key = (chain, resi)
                        if key not in seen:
                            seen.add(key)
                            residues.append(three_to_one.get(resn, "X"))
            if residues:
                seq = "".join(residues)
                out.write(f">{pdb_code}\n{seq}\n")


def _split_by_clusters(
    clusters: dict[str, int],
    train_frac: float,
    val_frac: float,
    cal_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split PDB codes ensuring no cluster spans multiple splits."""
    rng = np.random.RandomState(seed)

    # Group by cluster
    cluster_to_pdbs = {}
    for pdb, cid in clusters.items():
        cluster_to_pdbs.setdefault(cid, []).append(pdb)

    # Shuffle cluster order
    cluster_ids = list(cluster_to_pdbs.keys())
    rng.shuffle(cluster_ids)

    total = len(clusters)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    n_cal = int(total * cal_frac)

    splits = {"train": [], "val": [], "cal": [], "test": []}
    count = 0
    for cid in cluster_ids:
        pdbs = cluster_to_pdbs[cid]
        if count < n_train:
            splits["train"].extend(pdbs)
        elif count < n_train + n_val:
            splits["val"].extend(pdbs)
        elif count < n_train + n_val + n_cal:
            splits["cal"].extend(pdbs)
        else:
            splits["test"].extend(pdbs)
        count += len(pdbs)

    return splits


def _time_based_split(df: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    """Fallback: split by deposition year."""
    rng = np.random.RandomState(seed)

    train = df[df["year"] < 2016]["pdb_code"].tolist()
    rest = df[df["year"] >= 2016]["pdb_code"].tolist()
    rng.shuffle(rest)

    n = len(rest)
    n_val = n // 3
    n_cal = n // 3

    return {
        "train": train,
        "val": rest[:n_val],
        "cal": rest[n_val : n_val + n_cal],
        "test": rest[n_val + n_cal :],
    }


# ── CASF-2016 ────────────────────────────────────────────────────────────────


def load_casf2016_codes(casf_dir: str | Path) -> list[str]:
    """Load CASF-2016 core set PDB codes from its directory listing."""
    casf_dir = Path(casf_dir)

    # Try CoreSet.dat
    coreset_dat = casf_dir / "CoreSet.dat"
    if coreset_dat.exists():
        codes = []
        with open(coreset_dat) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                codes.append(line.split()[0])
        return codes

    # Fallback: list subdirectories
    codes = [d.name for d in casf_dir.iterdir() if d.is_dir() and len(d.name) == 4]
    return sorted(codes)


# ── Label transforms ─────────────────────────────────────────────────────────


def pka_to_deltaG(pka: float) -> float:
    """Convert pKa (-log10 affinity) to ΔG in kcal/mol."""
    # ΔG = RT * ln(Kd) = RT * ln(10) * log10(Kd) = -RT * ln(10) * pKa
    return -RT_KCAL * np.log(10) * pka


def deltaG_to_pka(dG: float) -> float:
    """Convert ΔG (kcal/mol) to pKa."""
    return -dG / (RT_KCAL * np.log(10))


# ── Pocket file utilities ────────────────────────────────────────────────────


def find_pocket_file(pdbbind_dir: str | Path, pdb_code: str) -> Optional[Path]:
    """Find the pocket PDB file for a given PDB code."""
    pdbbind_dir = Path(pdbbind_dir)
    base = pdbbind_dir / "refined-set" / pdb_code

    # Try standard naming conventions
    for suffix in [f"{pdb_code}_pocket.pdb", f"{pdb_code}_pocket10.pdb"]:
        p = base / suffix
        if p.exists():
            return p

    # Fallback: use protein file (will need pocket extraction)
    protein = base / f"{pdb_code}_protein.pdb"
    if protein.exists():
        return protein

    return None


def find_ligand_file(pdbbind_dir: str | Path, pdb_code: str) -> Optional[Path]:
    """Find the reference ligand file for a given PDB code."""
    pdbbind_dir = Path(pdbbind_dir)
    base = pdbbind_dir / "refined-set" / pdb_code
    for suffix in [f"{pdb_code}_ligand.sdf", f"{pdb_code}_ligand.mol2"]:
        p = base / suffix
        if p.exists():
            return p
    return None


def extract_pocket_from_protein(
    protein_pdb: str | Path,
    ligand_sdf: str | Path,
    output_path: str | Path,
    radius: float = 10.0,
) -> Path:
    """Extract pocket residues within `radius` Å of ligand atoms.

    Simple PDB-based extraction (no MDTraj dependency).
    """
    from rdkit import Chem

    protein_pdb = Path(protein_pdb)
    output_path = Path(output_path)

    # Get ligand atom coordinates
    mol = Chem.MolFromMolFile(str(ligand_sdf))
    if mol is None:
        mol = Chem.MolFromMol2File(str(ligand_sdf))
    if mol is None:
        raise ValueError(f"Cannot read ligand: {ligand_sdf}")

    conf = mol.GetConformer()
    lig_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    # Parse protein atoms
    pocket_residues = set()
    protein_lines = []
    with open(protein_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord = np.array([x, y, z])

                # Check if within radius of any ligand atom
                dists = np.linalg.norm(lig_coords - coord, axis=1)
                if dists.min() <= radius:
                    chain = line[21]
                    resi = line[22:27].strip()
                    pocket_residues.add((chain, resi))
                protein_lines.append(line)

    # Write pocket residues
    with open(output_path, "w") as out:
        for line in protein_lines:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                resi = line[22:27].strip()
                if (chain, resi) in pocket_residues:
                    out.write(line)
        out.write("END\n")

    return output_path
