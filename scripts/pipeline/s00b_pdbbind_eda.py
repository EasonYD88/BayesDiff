"""
scripts/pipeline/s00b_pdbbind_eda.py
────────────────────────────────────
EDA (Exploratory Data Analysis) for PDBbind v2020 dataset.

Generates all visualizations and quality checks specified in
doc/Stage_2/00a_supervised_pretraining.md §2.

Output:
    results/pdbbind_eda/
    ├── pKd_distribution.png
    ├── pKd_by_split.png
    ├── affinity_type_pie.png
    ├── resolution_vs_pkd.png
    ├── pocket_size_hist.png
    ├── ligand_size_hist.png
    ├── ligand_properties.png
    ├── split_summary.png
    ├── data_quality_report.csv
    └── eda_summary.json

Usage:
    python scripts/pipeline/s00b_pdbbind_eda.py \\
        --data_dir data/pdbbind_v2020 \\
        --pdbbind_dir data/pdbbind \\
        --output_dir results/pdbbind_eda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("s00b_eda")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
})


def load_data(data_dir: Path) -> tuple[pd.DataFrame, dict, Path]:
    """Load labels.csv, splits.json, and locate processed dir."""
    labels = pd.read_csv(data_dir / "labels.csv")
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    processed_dir = data_dir / "processed"
    return labels, splits, processed_dir


def plot_pkd_distribution(df: pd.DataFrame, output_dir: Path):
    """pKd histogram for the full dataset."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["pkd"], bins=np.arange(0, 15, 0.5), edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("pKd")
    ax.set_ylabel("Count")
    ax.set_title(f"PDBbind v2020 pKd Distribution (N={len(df)})")
    ax.axvline(df["pkd"].mean(), color="red", linestyle="--", label=f'Mean={df["pkd"].mean():.2f}')
    ax.legend()
    fig.savefig(output_dir / "pKd_distribution.png")
    plt.close(fig)
    logger.info("Saved pKd_distribution.png")


def plot_pkd_by_split(df: pd.DataFrame, splits: dict, output_dir: Path):
    """pKd distributions compared across splits."""
    split_names = [s for s in ["train", "val", "test"] if s in splits]
    colors = {"train": "steelblue", "val": "orange", "test": "red"}
    n_plots = len(split_names)
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, squeeze=False)

    for i, name in enumerate(split_names):
        ax = axes[i // ncols][i % ncols]
        codes = splits.get(name, [])
        split_df = df[df["pdb_code"].isin(codes)]
        if split_df.empty:
            ax.text(0.5, 0.5, f"No data for {name}", transform=ax.transAxes, ha="center")
            continue
        ax.hist(split_df["pkd"], bins=np.arange(0, 15, 0.5), edgecolor="black", alpha=0.7,
                color=colors.get(name, "gray"))
        ax.set_title(f"{name} (N={len(split_df)}, \u03bc={split_df['pkd'].mean():.2f})")
        ax.set_ylabel("Count")

    # Hide unused axes
    for i in range(n_plots, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("pKd Distribution by Split", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "pKd_by_split.png")
    plt.close(fig)
    logger.info("Saved pKd_by_split.png")


def plot_affinity_type_pie(df: pd.DataFrame, output_dir: Path):
    """Affinity type (Kd/Ki/IC50) pie chart."""
    counts = df["affinity_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=sns.color_palette("pastel", len(counts)))
    ax.set_title("Affinity Type Distribution")
    fig.savefig(output_dir / "affinity_type_pie.png")
    plt.close(fig)
    logger.info("Saved affinity_type_pie.png")


def plot_resolution_vs_pkd(df: pd.DataFrame, output_dir: Path):
    """Scatter: crystal resolution vs pKd."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = df.dropna(subset=["resolution"])
    ax.scatter(valid["resolution"], valid["pkd"], alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel("pKd")
    ax.set_title("Crystal Resolution vs pKd")
    fig.savefig(output_dir / "resolution_vs_pkd.png")
    plt.close(fig)
    logger.info("Saved resolution_vs_pkd.png")


def plot_pocket_ligand_sizes(processed_dir: Path, output_dir: Path) -> dict:
    """Pocket and ligand atom count distributions from .pt files."""
    n_protein_list = []
    n_ligand_list = []

    pt_files = sorted(processed_dir.glob("*.pt"))
    logger.info(f"Scanning {len(pt_files)} .pt files for size stats...")

    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            n_protein_list.append(int(data["n_protein_atoms"]))
            n_ligand_list.append(int(data["n_ligand_atoms"]))
        except Exception:
            continue

    if not n_protein_list:
        logger.warning("No .pt files found; skipping size plots")
        return {}

    # Pocket size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_protein_list, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Number of pocket atoms")
    ax.set_ylabel("Count")
    ax.set_title(f"Pocket Size Distribution (N={len(n_protein_list)})")
    ax.axvline(np.mean(n_protein_list), color="red", linestyle="--",
               label=f"Mean={np.mean(n_protein_list):.0f}")
    ax.legend()
    fig.savefig(output_dir / "pocket_size_hist.png")
    plt.close(fig)

    # Ligand size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_ligand_list, bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Number of ligand heavy atoms")
    ax.set_ylabel("Count")
    ax.set_title(f"Ligand Size Distribution (N={len(n_ligand_list)})")
    ax.axvline(np.mean(n_ligand_list), color="red", linestyle="--",
               label=f"Mean={np.mean(n_ligand_list):.0f}")
    ax.legend()
    fig.savefig(output_dir / "ligand_size_hist.png")
    plt.close(fig)

    logger.info("Saved pocket_size_hist.png, ligand_size_hist.png")

    return {
        "n_protein_mean": float(np.mean(n_protein_list)),
        "n_protein_std": float(np.std(n_protein_list)),
        "n_protein_min": int(np.min(n_protein_list)),
        "n_protein_max": int(np.max(n_protein_list)),
        "n_ligand_mean": float(np.mean(n_ligand_list)),
        "n_ligand_std": float(np.std(n_ligand_list)),
        "n_ligand_min": int(np.min(n_ligand_list)),
        "n_ligand_max": int(np.max(n_ligand_list)),
    }


def plot_ligand_properties(data_dir: Path, labels: pd.DataFrame, output_dir: Path):
    """Ligand physicochemical properties (MW, logP, TPSA, HBD/HBA)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError:
        logger.warning("RDKit not available; skipping ligand properties plot")
        return

    # Load code_to_dir mapping
    code_to_dir_path = data_dir / "code_to_dir.json"
    if code_to_dir_path.exists():
        with open(code_to_dir_path) as f:
            code_to_dir = json.load(f)
    else:
        logger.warning("code_to_dir.json not found; skipping ligand properties plot")
        return

    mw_list, logp_list, tpsa_list, hbd_list, hba_list, rot_list = [], [], [], [], [], []

    for code in labels["pdb_code"]:
        base = code_to_dir.get(code)
        if base is None:
            continue
        sdf_path = Path(base) / f"{code}_ligand.sdf"
        if not sdf_path.exists():
            continue
        mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=True)
        if mol is None:
            continue
        mw_list.append(Descriptors.MolWt(mol))
        logp_list.append(Descriptors.MolLogP(mol))
        tpsa_list.append(Descriptors.TPSA(mol))
        hbd_list.append(Lipinski.NumHDonors(mol))
        hba_list.append(Lipinski.NumHAcceptors(mol))
        rot_list.append(Lipinski.NumRotatableBonds(mol))

    if not mw_list:
        logger.warning("No ligands processed; skipping properties plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    props = [
        (mw_list, "Molecular Weight", "MW (Da)"),
        (logp_list, "LogP", "LogP"),
        (tpsa_list, "TPSA", "TPSA (Å²)"),
        (hbd_list, "H-Bond Donors", "Count"),
        (hba_list, "H-Bond Acceptors", "Count"),
        (rot_list, "Rotatable Bonds", "Count"),
    ]
    for ax, (vals, title, xlabel) in zip(axes.flat, props):
        ax.hist(vals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)

    fig.suptitle("Ligand Physicochemical Properties", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "ligand_properties.png")
    plt.close(fig)
    logger.info("Saved ligand_properties.png")


def plot_split_summary(splits: dict, output_dir: Path):
    """Bar chart of split sizes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(splits.keys())
    sizes = [len(v) for v in splits.values()]
    total = sum(sizes)
    bars = ax.bar(names, sizes, color=["steelblue", "orange", "green", "red"][:len(names)],
                  edgecolor="black")
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{size}\n({size/total:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of complexes")
    ax.set_title(f"Dataset Splits (Total: {total})")
    fig.savefig(output_dir / "split_summary.png")
    plt.close(fig)
    logger.info("Saved split_summary.png")


def plot_family_distribution(data_dir: Path, output_dir: Path):
    """Top-20 cluster (protein family) sizes as bar chart."""
    clusters_path = data_dir / "clusters.json"
    if not clusters_path.exists():
        logger.warning("clusters.json not found; skipping family distribution plot")
        return
    with open(clusters_path) as f:
        clusters = json.load(f)

    sizes = [(cid, len(pdbs)) for cid, pdbs in clusters.items()]
    sizes.sort(key=lambda x: -x[1])
    top20 = sizes[:20]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top20)), [s for _, s in top20], color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels([cid for cid, _ in top20], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of complexes")
    ax.set_title(f"Top-20 Protein Clusters by Size (of {len(clusters)} total)")
    fig.tight_layout()
    fig.savefig(output_dir / "family_distribution.png")
    plt.close(fig)
    logger.info("Saved family_distribution.png")


def plot_family_split_heatmap(data_dir: Path, splits: dict, output_dir: Path):
    """Heatmap: cluster × split to verify no cluster spans train AND val."""
    ca_path = data_dir / "cluster_assignments.csv"
    if not ca_path.exists():
        logger.warning("cluster_assignments.csv not found; skipping family-split heatmap")
        return

    ca = pd.read_csv(ca_path)
    split_map = {}
    for split_name, codes in splits.items():
        for c in codes:
            split_map[c] = split_name
    ca["split"] = ca["pdb_code"].map(split_map)
    ca = ca.dropna(subset=["split"])

    # Cross-tab: cluster_id × split
    ct = pd.crosstab(ca["cluster_id"], ca["split"])
    # Only show clusters that have >1 member and appear in at least 2 splits
    multi_split = ct[(ct > 0).sum(axis=1) > 1]
    n_leaking = len(multi_split[multi_split.get("train", 0) > 0][multi_split.get("val", 0) > 0]) if "train" in ct.columns and "val" in ct.columns else 0

    # For the heatmap, show top-30 largest clusters
    cluster_sizes = ca.groupby("cluster_id").size().sort_values(ascending=False)
    top30 = cluster_sizes.head(30).index
    ct_top = ct.loc[ct.index.isin(top30)].reindex(columns=["train", "val", "test"], fill_value=0)
    ct_top = ct_top.sort_values("train", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(ct_top, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title(f"Top-30 Clusters × Split\n(train/val leaking clusters: {n_leaking})")
    ax.set_ylabel("Cluster ID")
    fig.tight_layout()
    fig.savefig(output_dir / "family_split_heatmap.png")
    plt.close(fig)
    logger.info(f"Saved family_split_heatmap.png (train/val leak: {n_leaking})")


def plot_pkd_kde_by_split(df: pd.DataFrame, splits: dict, output_dir: Path):
    """KDE overlay of train vs val pKd distributions + KS test."""
    from scipy import stats as sp_stats

    train_pkd = df[df["pdb_code"].isin(splits.get("train", []))]["pkd"]
    val_pkd = df[df["pdb_code"].isin(splits.get("val", []))]["pkd"]
    test_pkd = df[df["pdb_code"].isin(splits.get("test", []))]["pkd"]

    ks_stat, ks_p = sp_stats.ks_2samp(train_pkd, val_pkd) if len(val_pkd) > 0 else (0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(train_pkd) > 0:
        train_pkd.plot.kde(ax=ax, label=f"Train (N={len(train_pkd)})", color="steelblue", linewidth=2)
    if len(val_pkd) > 0:
        val_pkd.plot.kde(ax=ax, label=f"Val (N={len(val_pkd)})", color="orange", linewidth=2)
    if len(test_pkd) > 0:
        test_pkd.plot.kde(ax=ax, label=f"Test (N={len(test_pkd)})", color="red", linewidth=2, linestyle="--")
    ax.set_xlabel("pKd")
    ax.set_ylabel("Density")
    ax.set_title(f"pKd Distribution by Split (KS p={ks_p:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pkd_distribution_by_split.png")
    plt.close(fig)
    logger.info(f"Saved pkd_distribution_by_split.png (KS p={ks_p:.4f})")
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p)}


def plot_chemical_space_tsne(data_dir: Path, labels: pd.DataFrame, splits: dict, output_dir: Path):
    """t-SNE of Morgan fingerprints coloured by split."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("RDKit or sklearn not available; skipping chemical space t-SNE")
        return

    code_to_dir_path = data_dir / "code_to_dir.json"
    if not code_to_dir_path.exists():
        logger.warning("code_to_dir.json not found; skipping chemical space t-SNE")
        return
    with open(code_to_dir_path) as f:
        code_to_dir = json.load(f)

    # Build split lookup
    split_map = {}
    for s, codes in splits.items():
        for c in codes:
            split_map[c] = s

    fps, split_labels, sampled_codes = [], [], []
    # Sample for speed: up to 3000 from train, all val, all test
    rng = np.random.RandomState(42)
    codes_train = [c for c in labels["pdb_code"] if split_map.get(c) == "train"]
    codes_val = [c for c in labels["pdb_code"] if split_map.get(c) == "val"]
    codes_test = [c for c in labels["pdb_code"] if split_map.get(c) == "test"]
    sample_train = list(rng.choice(codes_train, min(3000, len(codes_train)), replace=False))
    all_sample = sample_train + codes_val + codes_test

    for code in all_sample:
        base = code_to_dir.get(code)
        if base is None:
            continue
        sdf_path = Path(base) / f"{code}_ligand.sdf"
        if not sdf_path.exists():
            continue
        mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=True)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(np.array(fp))
        split_labels.append(split_map.get(code, "unknown"))
        sampled_codes.append(code)

    if len(fps) < 50:
        logger.warning(f"Only {len(fps)} fingerprints; skipping t-SNE")
        return

    logger.info(f"Running t-SNE on {len(fps)} ligand fingerprints...")
    X = np.array(fps)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fps) - 1))
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"train": "steelblue", "val": "orange", "test": "red"}
    for s in ["train", "val", "test"]:
        mask = np.array([l == s for l in split_labels])
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1], c=colors.get(s, "gray"),
                       label=f"{s} (N={mask.sum()})", alpha=0.4, s=8)
    ax.set_title("Chemical Space t-SNE (Morgan FP)")
    ax.legend(markerscale=3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(output_dir / "chemical_space_tsne.png")
    plt.close(fig)
    logger.info("Saved chemical_space_tsne.png")


def plot_cluster_size_hist(data_dir: Path, output_dir: Path):
    """Histogram of cluster sizes."""
    clusters_path = data_dir / "clusters.json"
    if not clusters_path.exists():
        logger.warning("clusters.json not found; skipping cluster size histogram")
        return
    with open(clusters_path) as f:
        clusters = json.load(f)

    sizes = [len(pdbs) for pdbs in clusters.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: full distribution
    axes[0].hist(sizes, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Cluster size")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Cluster Size Distribution (N={len(sizes)})")
    axes[0].axvline(np.median(sizes), color="red", linestyle="--",
                    label=f"Median={np.median(sizes):.0f}")
    axes[0].legend()

    # Right: log-scale for long tail
    axes[1].hist(sizes, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].set_xlabel("Cluster size")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_yscale("log")
    axes[1].set_title("Cluster Size Distribution (log scale)")

    fig.tight_layout()
    fig.savefig(output_dir / "cluster_size_hist.png")
    plt.close(fig)
    logger.info(f"Saved cluster_size_hist.png (median={np.median(sizes):.0f}, max={np.max(sizes)})")


def check_data_quality(labels: pd.DataFrame, processed_dir: Path, output_dir: Path) -> list[dict]:
    """Run data quality checks and save report."""
    issues = []

    # Outlier pKd values
    for _, row in labels.iterrows():
        if row["pkd"] < 1:
            issues.append({"pdb_code": row["pdb_code"], "issue": "low_pkd", "value": row["pkd"]})
        elif row["pkd"] > 14:
            issues.append({"pdb_code": row["pdb_code"], "issue": "high_pkd", "value": row["pkd"]})

    # Check for extreme pocket/ligand sizes
    for pt_file in processed_dir.glob("*.pt"):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            n_p = int(data["n_protein_atoms"])
            n_l = int(data["n_ligand_atoms"])
            if n_p < 50:
                issues.append({"pdb_code": pt_file.stem, "issue": "small_pocket", "value": n_p})
            if n_p > 2000:
                issues.append({"pdb_code": pt_file.stem, "issue": "large_pocket", "value": n_p})
            if n_l < 3:
                issues.append({"pdb_code": pt_file.stem, "issue": "small_ligand", "value": n_l})
            if n_l > 100:
                issues.append({"pdb_code": pt_file.stem, "issue": "large_ligand", "value": n_l})
        except Exception:
            issues.append({"pdb_code": pt_file.stem, "issue": "corrupt_pt", "value": None})

    # Save report
    if issues:
        report_df = pd.DataFrame(issues)
        report_path = output_dir / "data_quality_report.csv"
        report_df.to_csv(report_path, index=False)
        logger.info(f"Data quality issues: {len(issues)} → {report_path}")
        for issue_type, count in report_df["issue"].value_counts().items():
            logger.info(f"  {issue_type}: {count}")
    else:
        logger.info("No data quality issues found")

    return issues


# ── 5-Fold Split Quality Checks (§2.3.1) ────────────────────────────────────


def load_5fold_splits(data_dir: Path) -> dict | None:
    """Load splits_5fold.json if it exists."""
    fivefold_path = data_dir / "splits_5fold.json"
    if not fivefold_path.exists():
        logger.warning("splits_5fold.json not found; skipping 5-fold EDA")
        return None
    with open(fivefold_path) as f:
        return json.load(f)


def plot_5fold_val_sizes(fivefold: dict, output_dir: Path):
    """Bar chart comparing Val sample counts across 5 folds."""
    folds = fivefold["folds"]
    fold_ids = sorted(folds.keys(), key=int)
    val_sizes = [len(folds[fid]["val"]) for fid in fold_ids]
    train_sizes = [len(folds[fid]["train"]) for fid in fold_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fold_ids))
    width = 0.35
    bars_train = ax.bar(x - width / 2, train_sizes, width, label="Train", color="steelblue",
                        edgecolor="black")
    bars_val = ax.bar(x + width / 2, val_sizes, width, label="Val", color="orange",
                      edgecolor="black")

    # Annotate val percentages
    for i, (tr, va) in enumerate(zip(train_sizes, val_sizes)):
        pct = va / (tr + va) * 100
        ax.text(x[i] + width / 2, va + 20, f"{pct:.1f}%", ha="center", fontsize=8)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Number of complexes")
    ax.set_title(f"5-Fold Val Sizes (Test fixed: {len(fivefold['test'])})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {fid}" for fid in fold_ids])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "5fold_val_sizes.png")
    plt.close(fig)
    logger.info("Saved 5fold_val_sizes.png")


def plot_5fold_val_pkd_kde(fivefold: dict, labels: pd.DataFrame, output_dir: Path):
    """Overlay KDE of Val pKd distributions across 5 folds."""
    folds = fivefold["folds"]
    fold_ids = sorted(folds.keys(), key=int)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(fold_ids)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, fid in enumerate(fold_ids):
        val_codes = set(folds[fid]["val"])
        val_pkd = labels[labels["pdb_code"].isin(val_codes)]["pkd"]
        if len(val_pkd) > 10:
            val_pkd.plot.kde(ax=ax, label=f"Fold {fid} (N={len(val_pkd)})",
                            color=colors[i], linewidth=1.5)

    # Also plot train of fold 0 as reference
    train_codes_0 = set(folds["0"]["train"])
    train_pkd_0 = labels[labels["pdb_code"].isin(train_codes_0)]["pkd"]
    if len(train_pkd_0) > 10:
        train_pkd_0.plot.kde(ax=ax, label=f"Train fold 0 (N={len(train_pkd_0)})",
                            color="black", linewidth=2, linestyle="--")

    ax.set_xlabel("pKd")
    ax.set_ylabel("Density")
    ax.set_title("5-Fold Val pKd Distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "5fold_val_pkd_kde.png")
    plt.close(fig)
    logger.info("Saved 5fold_val_pkd_kde.png")


def compute_5fold_ks_tests(fivefold: dict, labels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """KS test: Train vs Val pKd for each fold. Save CSV summary."""
    from scipy import stats as sp_stats

    folds = fivefold["folds"]
    fold_ids = sorted(folds.keys(), key=int)
    rows = []

    for fid in fold_ids:
        train_pkd = labels[labels["pdb_code"].isin(folds[fid]["train"])]["pkd"].values
        val_pkd = labels[labels["pdb_code"].isin(folds[fid]["val"])]["pkd"].values
        if len(train_pkd) > 0 and len(val_pkd) > 0:
            ks_stat, ks_p = sp_stats.ks_2samp(train_pkd, val_pkd)
        else:
            ks_stat, ks_p = float("nan"), float("nan")
        rows.append({
            "fold": int(fid),
            "seed": folds[fid]["seed"],
            "n_train": len(folds[fid]["train"]),
            "n_val": len(folds[fid]["val"]),
            "val_pct": len(folds[fid]["val"]) / (len(folds[fid]["train"]) + len(folds[fid]["val"])) * 100,
            "train_pkd_mean": float(np.mean(train_pkd)) if len(train_pkd) > 0 else float("nan"),
            "val_pkd_mean": float(np.mean(val_pkd)) if len(val_pkd) > 0 else float("nan"),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "pass": ks_p > 0.05,
        })

    ks_df = pd.DataFrame(rows)
    ks_path = output_dir / "5fold_ks_test_summary.csv"
    ks_df.to_csv(ks_path, index=False)
    logger.info(f"Saved 5fold_ks_test_summary.csv")

    # Log results
    for _, row in ks_df.iterrows():
        status = "PASS" if row["pass"] else "FAIL"
        logger.info(
            f"  Fold {int(row['fold'])}: KS stat={row['ks_statistic']:.4f}, "
            f"p={row['ks_p_value']:.4f} [{status}], "
            f"val={row['val_pct']:.1f}%"
        )

    return ks_df


def plot_5fold_val_overlap_heatmap(fivefold: dict, output_dir: Path):
    """Heatmap of pairwise Jaccard similarity between fold Val sets."""
    folds = fivefold["folds"]
    fold_ids = sorted(folds.keys(), key=int)
    n = len(fold_ids)
    jaccard = np.zeros((n, n))

    val_sets = [set(folds[fid]["val"]) for fid in fold_ids]

    for i in range(n):
        for j in range(n):
            inter = len(val_sets[i] & val_sets[j])
            union = len(val_sets[i] | val_sets[j])
            jaccard[i, j] = inter / union if union > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = sns.heatmap(
        jaccard,
        annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=[f"Fold {fid}" for fid in fold_ids],
        yticklabels=[f"Fold {fid}" for fid in fold_ids],
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title("Val Set Jaccard Overlap Between Folds")
    fig.tight_layout()
    fig.savefig(output_dir / "5fold_val_overlap_heatmap.png")
    plt.close(fig)

    # Check non-degeneracy (off-diagonal < 0.8)
    off_diag = jaccard[np.triu_indices(n, k=1)]
    max_overlap = float(np.max(off_diag)) if len(off_diag) > 0 else 0
    logger.info(
        f"Saved 5fold_val_overlap_heatmap.png "
        f"(max off-diag Jaccard={max_overlap:.3f}, "
        f"{'PASS' if max_overlap < 0.8 else 'WARN: high overlap'})"
    )
    return max_overlap


def main():
    parser = argparse.ArgumentParser(description="PDBbind v2020 EDA")
    parser.add_argument("--data_dir", type=str, default="data/pdbbind_v2020")
    parser.add_argument("--pdbbind_dir", type=str, default="data/pdbbind")
    parser.add_argument("--output_dir", type=str, default="results/pdbbind_eda")
    parser.add_argument(
        "--only_5fold", action="store_true",
        help="Only run 5-fold split quality checks (skip single-split EDA)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pdbbind_dir = Path(args.pdbbind_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5-fold outputs always go to a separate subdirectory
    fivefold_output_dir = output_dir / "5fold"
    fivefold_output_dir.mkdir(parents=True, exist_ok=True)

    labels, splits, processed_dir = load_data(data_dir)

    ks_stats = {}
    size_stats = {}
    issues = []

    if not args.only_5fold:
        # ── 2.1 Label distribution ──
        plot_pkd_distribution(labels, output_dir)
        plot_pkd_by_split(labels, splits, output_dir)
        plot_affinity_type_pie(labels, output_dir)
        plot_resolution_vs_pkd(labels, output_dir)

        # ── 2.2 Protein & ligand statistics ──
        size_stats = plot_pocket_ligand_sizes(processed_dir, output_dir)
        plot_ligand_properties(data_dir, labels, output_dir)

        # ── 2.3 Split quality ──
        plot_split_summary(splits, output_dir)
        plot_family_distribution(data_dir, output_dir)
        plot_family_split_heatmap(data_dir, splits, output_dir)
        ks_stats = plot_pkd_kde_by_split(labels, splits, output_dir)
        plot_chemical_space_tsne(data_dir, labels, splits, output_dir)
        plot_cluster_size_hist(data_dir, output_dir)

        # ── 2.4 Data quality ──
        issues = check_data_quality(labels, processed_dir, output_dir)

    # ── 2.3.1 5-Fold Split quality checks → output to 5fold/ subdirectory ──
    fivefold = load_5fold_splits(data_dir)
    fivefold_stats = {}
    if fivefold is not None:
        plot_5fold_val_sizes(fivefold, fivefold_output_dir)
        plot_5fold_val_pkd_kde(fivefold, labels, fivefold_output_dir)
        ks_df = compute_5fold_ks_tests(fivefold, labels, fivefold_output_dir)
        max_overlap = plot_5fold_val_overlap_heatmap(fivefold, fivefold_output_dir)
        fivefold_stats = {
            "n_folds": len(fivefold["folds"]),
            "all_ks_pass": bool(ks_df["pass"].all()),
            "max_val_overlap_jaccard": float(max_overlap),
            "fold_val_sizes": {
                fid: len(fivefold["folds"][fid]["val"])
                for fid in sorted(fivefold["folds"].keys(), key=int)
            },
        }

    # ── Summary JSON ──
    summary = {
        "n_total": len(labels),
        "n_processed": len(list(processed_dir.glob("*.pt"))),
        "pkd_mean": float(labels["pkd"].mean()),
        "pkd_std": float(labels["pkd"].std()),
        "pkd_min": float(labels["pkd"].min()),
        "pkd_max": float(labels["pkd"].max()),
        "affinity_types": labels["affinity_type"].value_counts().to_dict(),
        "splits": {k: len(v) for k, v in splits.items()},
        "n_quality_issues": len(issues),
        **(ks_stats if ks_stats else {}),
        **size_stats,
        **({"fivefold": fivefold_stats} if fivefold_stats else {}),
    }

    # Write main summary to output_dir (always)
    summary_path = output_dir / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"EDA summary → {summary_path}")

    # Also write 5-fold summary to the 5fold subdirectory
    if fivefold_stats:
        fivefold_summary_path = fivefold_output_dir / "5fold_eda_summary.json"
        with open(fivefold_summary_path, "w") as f:
            json.dump(fivefold_stats, f, indent=2)
        logger.info(f"5-fold summary → {fivefold_summary_path}")

    logger.info("=" * 60)
    logger.info("EDA complete!")
    logger.info(f"  Total complexes: {summary['n_total']}")
    logger.info(f"  Processed .pt:   {summary['n_processed']}")
    logger.info(f"  Quality issues:  {summary['n_quality_issues']}")
    logger.info(f"  Output dir:      {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
