"""
bayesdiff/data.py — Data Utilities
───────────────────────────────────
Data loading, PDBbind INDEX parsing, protein-family split, and label transforms.

Usage:
    python scripts/pipeline/s01_prepare_data.py --pdbbind_dir data/pdbbind --output_dir data/splits
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
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


def parse_pdbbind_index(
    index_path: str | Path,
    keep_inexact: bool = False,
    keep_ic50: bool = True,
) -> pd.DataFrame:
    """Parse PDBbind INDEX file into a DataFrame.

    Supports two formats:
      - Refined (5 data fields): PDB  resl  year  pKa   Kd=...  // ...
      - R1 general (4 data fields): PDB  resl  year  Kd=...  // ...

    Args:
        index_path: Path to INDEX file.
        keep_inexact: If False (default), drop entries with <, >, ~ in binding data.
        keep_ic50: If True (default), keep IC50 entries; if False, drop them.

    Returns DataFrame with columns:
        pdb_code, resolution, year, affinity_type, affinity_value_M, pkd, is_exact
    """
    records = []
    n_skipped_inexact = 0
    n_skipped_ic50 = 0
    n_skipped_parse = 0

    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data_part = line.split("//")[0].strip()
            parts = data_part.split()
            if len(parts) < 4:
                continue

            pdb_code = parts[0]
            resolution = float(parts[1]) if parts[1] != "NMR" else np.nan
            year = int(parts[2])

            # Detect format: if parts[3] looks like a number, it's refined format
            # (5 fields: code resl year pKa Kd=...), else R1 format (4 fields)
            try:
                pka_explicit = float(parts[3])
                # Refined format: pKa is given explicitly
                kv_str = parts[4] if len(parts) >= 5 else None
            except ValueError:
                # R1 format: parts[3] is the binding data string
                pka_explicit = None
                kv_str = parts[3]

            if kv_str is None:
                n_skipped_parse += 1
                continue

            # Check if inexact
            is_exact = not bool(re.search(r"[<>~]", kv_str))
            if not is_exact and not keep_inexact:
                n_skipped_inexact += 1
                continue

            # Parse affinity type and value
            aff_type, aff_val_M = _parse_affinity_string(kv_str)

            if aff_type == "unknown":
                n_skipped_parse += 1
                continue

            if aff_type == "IC50" and not keep_ic50:
                n_skipped_ic50 += 1
                continue

            # Compute pKd = -log10(affinity_M)
            if pka_explicit is not None:
                pkd = pka_explicit
            elif aff_val_M > 0:
                pkd = -np.log10(aff_val_M)
            else:
                n_skipped_parse += 1
                continue

            records.append(
                {
                    "pdb_code": pdb_code,
                    "resolution": resolution,
                    "year": year,
                    "affinity_type": aff_type,
                    "affinity_value_M": aff_val_M,
                    "pkd": pkd,
                    "is_exact": is_exact,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        f"Parsed {len(df)} entries from {index_path} "
        f"(skipped: {n_skipped_inexact} inexact, {n_skipped_ic50} IC50, "
        f"{n_skipped_parse} parse errors)"
    )
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


def parse_casf_coreset(coreset_dat: str | Path) -> pd.DataFrame:
    """Parse CASF-2016 CoreSet.dat into a DataFrame.

    Format: #code  resl  year  logKa  Ka  target
    Returns DataFrame with columns: pdb_code, resolution, year, pkd, affinity_type, target_id
    """
    records = []
    with open(coreset_dat) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            pdb_code = parts[0]
            resolution = float(parts[1])
            year = int(parts[2])
            pkd = float(parts[3])  # logKa = -log10(Kd)
            kv_str = parts[4]
            target_id = int(parts[5])
            aff_type, _ = _parse_affinity_string(kv_str)
            records.append({
                "pdb_code": pdb_code,
                "resolution": resolution,
                "year": year,
                "pkd": pkd,
                "affinity_type": aff_type,
                "target_id": target_id,
            })
    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} CASF-2016 entries ({df['target_id'].nunique()} targets)")
    return df


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
    """Legacy split function. Use cluster_stratified_split() for new code."""
    return cluster_stratified_split(
        df, pdbbind_dir, val_frac=val_frac, seq_identity=seq_identity, seed=seed,
    )


def cluster_stratified_split(
    df: pd.DataFrame,
    pdbbind_dir: str | Path,
    val_frac: float = 0.12,
    seq_identity: float = 0.30,
    n_bins: int = 10,
    seed: int = 42,
    pdb_to_file: Optional[dict[str, Path]] = None,
) -> dict[str, list[str]]:
    """Split PDB codes by protein-cluster + pKd-stratified sampling.

    Steps:
      1. Extract protein sequences → FASTA
      2. Cluster with mmseqs2 at `seq_identity` threshold
      3. Bin clusters by median pKd (quantile-based)
      4. Within each bin, sample ~val_frac of clusters → val

    Args:
        df: DataFrame with 'pdb_code' and 'pkd' columns.
        pdbbind_dir: Root for finding protein PDB files.
        val_frac: Target fraction of *samples* for validation (0.10–0.15).
        seq_identity: mmseqs2 --min-seq-id threshold.
        n_bins: Number of pKd bins for stratified sampling.
        seed: Random seed.
        pdb_to_file: Optional {pdb_code: Path} mapping to protein PDB files.

    Returns dict with keys: train, val → lists of PDB codes.
    """
    rng = np.random.RandomState(seed)
    pdb_codes = df["pdb_code"].unique().tolist()
    pkd_map = dict(zip(df["pdb_code"], df["pkd"]))

    # Step 1-2: Cluster
    clusters = _try_mmseqs_cluster(pdb_codes, pdbbind_dir, seq_identity, pdb_to_file)

    if clusters is None:
        logger.warning("mmseqs2 not available. Falling back to time-based split.")
        return _time_based_split(df, seed)

    # Build cluster → pdb_codes mapping
    cluster_to_pdbs: dict[int, list[str]] = {}
    for pdb, cid in clusters.items():
        cluster_to_pdbs.setdefault(cid, []).append(pdb)

    # Assign unclustered codes to singleton clusters
    max_cid = max(clusters.values()) if clusters else 0
    for pdb in pdb_codes:
        if pdb not in clusters:
            max_cid += 1
            cluster_to_pdbs[max_cid] = [pdb]
            clusters[pdb] = max_cid

    # Step 3: Compute per-cluster median pKd
    cluster_info = []
    for cid, pdbs in cluster_to_pdbs.items():
        pkds = [pkd_map[p] for p in pdbs if p in pkd_map]
        if not pkds:
            continue
        cluster_info.append({
            "cluster_id": cid,
            "pdbs": pdbs,
            "size": len(pdbs),
            "median_pkd": np.median(pkds),
        })

    # Step 4: Bin clusters by median pKd quantiles
    median_pkds = np.array([c["median_pkd"] for c in cluster_info])
    bin_edges = np.quantile(median_pkds, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6  # ensure max value is included

    for c in cluster_info:
        c["bin"] = int(np.digitize(c["median_pkd"], bin_edges[1:])) 

    # Step 5: Within each bin, sample clusters for validation
    #   Sample from EVERY bin proportionally so val pKd distribution matches train.
    val_clusters = set()
    val_count = 0

    bins = {}
    for c in cluster_info:
        bins.setdefault(c["bin"], []).append(c)

    for bin_id in sorted(bins.keys()):
        bin_clusters = bins[bin_id]
        rng.shuffle(bin_clusters)
        bin_total = sum(c["size"] for c in bin_clusters)
        bin_target = int(bin_total * val_frac)
        bin_val = 0
        for c in bin_clusters:
            if bin_val + c["size"] <= bin_target + c["size"] // 2:
                val_clusters.add(c["cluster_id"])
                bin_val += c["size"]
                val_count += c["size"]

    train_codes = []
    val_codes = []
    for c in cluster_info:
        if c["cluster_id"] in val_clusters:
            val_codes.extend(c["pdbs"])
        else:
            train_codes.extend(c["pdbs"])

    logger.info(
        f"Cluster-stratified split: {len(train_codes)} train, {len(val_codes)} val "
        f"({len(val_codes)/(len(train_codes)+len(val_codes))*100:.1f}%) "
        f"from {len(cluster_info)} clusters ({len(val_clusters)} val clusters)"
    )

    return {
        "train": sorted(train_codes),
        "val": sorted(val_codes),
        "_cluster_info": cluster_info,
        "_clusters": clusters,
    }


def cluster_stratified_split_nfold(
    df: pd.DataFrame,
    n_folds: int = 5,
    val_frac: float = 0.12,
    n_bins: int = 10,
    base_seed: int = 42,
    clusters_json: Optional[str | Path] = None,
    pdbbind_dir: Optional[str | Path] = None,
    seq_identity: float = 0.30,
    pdb_to_file: Optional[dict[str, Path]] = None,
) -> dict:
    """Generate N grouped Train/Val splits from pre-computed protein clusters.

    Re-uses the same clustering for all folds; only the random sampling seed
    differs (seed = base_seed + fold_id).

    Args:
        df: DataFrame with 'pdb_code' and 'pkd' columns (non-CASF pool only).
        n_folds: Number of folds to generate.
        val_frac: Target validation fraction of samples.
        n_bins: Number of pKd quantile bins for stratified sampling.
        base_seed: Base random seed; fold i uses base_seed + i.
        clusters_json: Path to existing clusters.json. If provided, skip
            mmseqs2 clustering and load clusters from file.
        pdbbind_dir: PDBbind root (needed only if clusters_json is None).
        seq_identity: mmseqs2 --min-seq-id (needed only if clusters_json is None).
        pdb_to_file: Optional {pdb_code: Path} mapping.

    Returns dict:
        {
            "folds": {
                "0": {"train": [...], "val": [...], "seed": 42},
                ...
            },
            "cluster_info": {
                "n_clusters": N,
                "clustering_params": {"min_seq_id": 0.30, "coverage": 0.8}
            }
        }
    """
    pdb_codes = df["pdb_code"].unique().tolist()
    pkd_map = dict(zip(df["pdb_code"], df["pkd"]))

    # ── Load or compute clusters ──
    if clusters_json is not None:
        clusters_json = Path(clusters_json)
        with open(clusters_json) as f:
            raw = json.load(f)
        # clusters.json format: {cluster_id_str: [pdb_code, ...]}
        clusters: dict[str, int] = {}
        cluster_to_pdbs: dict[int, list[str]] = {}
        pool_set = set(pdb_codes)
        for cid_str, pdbs in raw.items():
            cid = int(cid_str)
            members = [p for p in pdbs if p in pool_set]
            if members:
                cluster_to_pdbs[cid] = members
                for p in members:
                    clusters[p] = cid
        logger.info(
            f"Loaded {len(cluster_to_pdbs)} clusters from {clusters_json} "
            f"({len(clusters)} PDB codes in pool)"
        )
    else:
        if pdbbind_dir is None:
            raise ValueError("Either clusters_json or pdbbind_dir must be provided")
        clusters = _try_mmseqs_cluster(pdb_codes, pdbbind_dir, seq_identity, pdb_to_file)
        if clusters is None:
            raise RuntimeError("mmseqs2 not available and no clusters_json provided")
        cluster_to_pdbs = {}
        for pdb, cid in clusters.items():
            cluster_to_pdbs.setdefault(cid, []).append(pdb)

    # Assign unclustered codes to singleton clusters
    max_cid = max(clusters.values()) if clusters else 0
    for pdb in pdb_codes:
        if pdb not in clusters:
            max_cid += 1
            cluster_to_pdbs[max_cid] = [pdb]
            clusters[pdb] = max_cid

    # Compute per-cluster median pKd
    cluster_info_list = []
    for cid, pdbs in cluster_to_pdbs.items():
        pkds = [pkd_map[p] for p in pdbs if p in pkd_map]
        if not pkds:
            continue
        cluster_info_list.append({
            "cluster_id": cid,
            "pdbs": pdbs,
            "size": len(pdbs),
            "median_pkd": float(np.median(pkds)),
        })

    # Bin clusters by median pKd quantiles (shared across folds)
    median_pkds = np.array([c["median_pkd"] for c in cluster_info_list])
    bin_edges = np.quantile(median_pkds, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6

    for c in cluster_info_list:
        c["bin"] = int(np.digitize(c["median_pkd"], bin_edges[1:]))

    bins: dict[int, list[dict]] = {}
    for c in cluster_info_list:
        bins.setdefault(c["bin"], []).append(c)

    # ── Generate each fold ──
    folds = {}
    for fold_id in range(n_folds):
        seed = base_seed + fold_id
        rng = np.random.RandomState(seed)

        val_clusters = set()
        for bin_id in sorted(bins.keys()):
            bin_clusters = list(bins[bin_id])  # copy to avoid in-place mutation
            rng.shuffle(bin_clusters)
            bin_total = sum(c["size"] for c in bin_clusters)
            bin_target = int(bin_total * val_frac)
            bin_val = 0
            for c in bin_clusters:
                if bin_val + c["size"] <= bin_target + c["size"] // 2:
                    val_clusters.add(c["cluster_id"])
                    bin_val += c["size"]

        train_codes = []
        val_codes = []
        for c in cluster_info_list:
            if c["cluster_id"] in val_clusters:
                val_codes.extend(c["pdbs"])
            else:
                train_codes.extend(c["pdbs"])

        folds[str(fold_id)] = {
            "train": sorted(train_codes),
            "val": sorted(val_codes),
            "seed": seed,
        }
        logger.info(
            f"Fold {fold_id} (seed={seed}): {len(train_codes)} train, "
            f"{len(val_codes)} val "
            f"({len(val_codes)/(len(train_codes)+len(val_codes))*100:.1f}%), "
            f"{len(val_clusters)} val clusters"
        )

    return {
        "folds": folds,
        "cluster_info": {
            "n_clusters": len(cluster_info_list),
            "clustering_params": {"min_seq_id": seq_identity, "coverage": 0.8},
        },
    }


def _try_mmseqs_cluster(
    pdb_codes: list[str],
    pdbbind_dir: str | Path,
    seq_identity: float,
    pdb_to_file: Optional[dict[str, Path]] = None,
) -> Optional[dict[str, int]]:
    """Cluster proteins by sequence identity using mmseqs2.

    Returns {pdb_code: cluster_id} or None if mmseqs2 not available.
    """
    try:
        subprocess.run(["mmseqs", "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    pdbbind_dir = Path(pdbbind_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "proteins.fasta"
        _extract_sequences_to_fasta(pdb_codes, pdbbind_dir, fasta_path, pdb_to_file)

        if not fasta_path.exists() or fasta_path.stat().st_size == 0:
            logger.warning("No sequences extracted; skipping mmseqs clustering.")
            return None

        clu_prefix = Path(tmpdir) / "cluDB"
        tmp_sub = Path(tmpdir) / "tmp"
        tmp_sub.mkdir()

        result = subprocess.run(
            [
                "mmseqs", "easy-cluster",
                str(fasta_path),
                str(clu_prefix),
                str(tmp_sub),
                "--min-seq-id", str(seq_identity),
                "-c", "0.8",
                "--threads", str(min(mp.cpu_count(), 32)),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"mmseqs2 failed: {result.stderr[:500]}")
            return None

        # Parse cluster TSV (rep \t member)
        tsv_file = Path(str(clu_prefix) + "_cluster.tsv")
        if not tsv_file.exists():
            for f in Path(tmpdir).glob("*cluster.tsv"):
                tsv_file = f
                break
        if not tsv_file.exists():
            logger.warning("mmseqs2 cluster TSV not found.")
            return None

        rep_to_id: dict[str, int] = {}
        clusters: dict[str, int] = {}
        next_id = 0
        with open(tsv_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                rep, member = parts[0], parts[1]
                if rep not in rep_to_id:
                    rep_to_id[rep] = next_id
                    next_id += 1
                clusters[member] = rep_to_id[rep]

        logger.info(
            f"mmseqs2 clustering: {len(rep_to_id)} clusters from {len(clusters)} sequences"
        )
        return clusters


def _extract_sequences_to_fasta(
    pdb_codes: list[str],
    pdbbind_dir: Path,
    output_path: Path,
    pdb_to_file: Optional[dict[str, Path]] = None,
) -> None:
    """Extract protein sequences from PDB files into a FASTA file.

    Supports both refined-set layout and P-L/YEAR_RANGE layout.
    """
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    n_ok = 0
    with open(output_path, "w") as out:
        for pdb_code in pdb_codes:
            protein_pdb = None
            if pdb_to_file and pdb_code in pdb_to_file:
                protein_pdb = pdb_to_file[pdb_code]
            if protein_pdb is None or not protein_pdb.exists():
                protein_pdb = _find_protein_pdb(pdbbind_dir, pdb_code)
            if protein_pdb is None:
                continue

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
                out.write(f">{pdb_code}\n{''.join(residues)}\n")
                n_ok += 1

    logger.info(f"Extracted {n_ok}/{len(pdb_codes)} protein sequences → {output_path}")


def _find_protein_pdb(pdbbind_dir: Path, pdb_code: str) -> Optional[Path]:
    """Find protein PDB file in various PDBbind directory layouts."""
    # Layout 1: refined-set/CODE/CODE_protein.pdb
    p = pdbbind_dir / "refined-set" / pdb_code / f"{pdb_code}_protein.pdb"
    if p.exists():
        return p
    # Layout 2: P-L/YEAR_RANGE/CODE/CODE_protein.pdb
    pl_dir = pdbbind_dir / "P-L"
    if pl_dir.exists():
        for yr_dir in pl_dir.iterdir():
            if yr_dir.is_dir():
                p = yr_dir / pdb_code / f"{pdb_code}_protein.pdb"
                if p.exists():
                    return p
    return None


def _split_by_clusters(
    clusters: dict[str, int],
    train_frac: float,
    val_frac: float,
    cal_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split PDB codes ensuring no cluster spans multiple splits. (Legacy)"""
    rng = np.random.RandomState(seed)
    cluster_to_pdbs = {}
    for pdb, cid in clusters.items():
        cluster_to_pdbs.setdefault(cid, []).append(pdb)
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
    n_val = max(1, int(n * 0.3))
    return {
        "train": train + rest[n_val:],
        "val": rest[:n_val],
    }


# ── CASF-2016 ────────────────────────────────────────────────────────────────


def load_casf2016_codes(casf_dir: str | Path) -> list[str]:
    """Load CASF-2016 core set PDB codes from CoreSet.dat or directory listing."""
    casf_dir = Path(casf_dir)

    # Try various CoreSet.dat locations
    for candidate in [
        casf_dir / "CoreSet.dat",
        casf_dir / "power_screening" / "CoreSet.dat",
        casf_dir / "power_scoring" / "CoreSet.dat",
    ]:
        if candidate.exists():
            codes = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    codes.append(line.split()[0])
            return sorted(codes)

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
    """Find the pocket PDB file for a given PDB code.

    Supports two layouts:
      1. PDBbind: pdbbind_dir/refined-set/{code}/{code}_pocket.pdb
      2. TargetDiff test_set: pdbbind_dir/{target_name}/*_rec.pdb
    """
    pdbbind_dir = Path(pdbbind_dir)

    # ── Layout 1: PDBbind refined-set ────────────────────────────
    base = pdbbind_dir / "refined-set" / pdb_code
    for suffix in [f"{pdb_code}_pocket.pdb", f"{pdb_code}_pocket10.pdb"]:
        p = base / suffix
        if p.exists():
            return p
    protein = base / f"{pdb_code}_protein.pdb"
    if protein.exists():
        return protein

    # ── Layout 2: TargetDiff test_set (target_name/*_rec.pdb) ──
    target_dir = pdbbind_dir / pdb_code
    if target_dir.is_dir():
        rec_files = list(target_dir.glob("*_rec.pdb"))
        if rec_files:
            return rec_files[0]
        # Any PDB file as fallback
        pdb_files = list(target_dir.glob("*.pdb"))
        if pdb_files:
            return pdb_files[0]

    # ── Layout 3: directory is the test_set root, search 1 level ──
    for sub in pdbbind_dir.iterdir():
        if sub.is_dir() and sub.name == pdb_code:
            rec = list(sub.glob("*_rec.pdb"))
            if rec:
                return rec[0]

    return None


def find_ligand_file(pdbbind_dir: str | Path, pdb_code: str) -> Optional[Path]:
    """Find the reference ligand file for a given PDB code.

    Supports PDBbind layout and TargetDiff test_set layout.
    """
    pdbbind_dir = Path(pdbbind_dir)

    # PDBbind layout
    base = pdbbind_dir / "refined-set" / pdb_code
    for suffix in [f"{pdb_code}_ligand.sdf", f"{pdb_code}_ligand.mol2"]:
        p = base / suffix
        if p.exists():
            return p

    # TargetDiff test_set layout
    target_dir = pdbbind_dir / pdb_code
    if target_dir.is_dir():
        sdf_files = list(target_dir.glob("*.sdf"))
        if sdf_files:
            return sdf_files[0]

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
