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
    "savefig.bbox_inches": "tight",
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    split_names = ["train", "val", "cal", "test"]
    colors = ["steelblue", "orange", "green", "red"]

    for ax, name, color in zip(axes.flat, split_names, colors):
        codes = splits.get(name, [])
        split_df = df[df["pdb_code"].isin(codes)]
        if split_df.empty:
            ax.text(0.5, 0.5, f"No data for {name}", transform=ax.transAxes, ha="center")
            continue
        ax.hist(split_df["pkd"], bins=np.arange(0, 15, 0.5), edgecolor="black", alpha=0.7, color=color)
        ax.set_title(f"{name} (N={len(split_df)}, μ={split_df['pkd'].mean():.2f})")
        ax.set_ylabel("Count")

    axes[1, 0].set_xlabel("pKd")
    axes[1, 1].set_xlabel("pKd")
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


def plot_ligand_properties(pdbbind_dir: Path, labels: pd.DataFrame, output_dir: Path):
    """Ligand physicochemical properties (MW, logP, TPSA, HBD/HBA)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError:
        logger.warning("RDKit not available; skipping ligand properties plot")
        return

    mw_list, logp_list, tpsa_list, hbd_list, hba_list, rot_list = [], [], [], [], [], []

    for code in labels["pdb_code"]:
        sdf_path = pdbbind_dir / "refined-set" / code / f"{code}_ligand.sdf"
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


def main():
    parser = argparse.ArgumentParser(description="PDBbind v2020 EDA")
    parser.add_argument("--data_dir", type=str, default="data/pdbbind_v2020")
    parser.add_argument("--pdbbind_dir", type=str, default="data/pdbbind")
    parser.add_argument("--output_dir", type=str, default="results/pdbbind_eda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pdbbind_dir = Path(args.pdbbind_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, splits, processed_dir = load_data(data_dir)

    # ── 2.1 Label distribution ──
    plot_pkd_distribution(labels, output_dir)
    plot_pkd_by_split(labels, splits, output_dir)
    plot_affinity_type_pie(labels, output_dir)
    plot_resolution_vs_pkd(labels, output_dir)

    # ── 2.2 Protein & ligand statistics ──
    size_stats = plot_pocket_ligand_sizes(processed_dir, output_dir)
    plot_ligand_properties(pdbbind_dir, labels, output_dir)

    # ── 2.3 Split quality ──
    plot_split_summary(splits, output_dir)

    # ── 2.4 Data quality ──
    issues = check_data_quality(labels, processed_dir, output_dir)

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
        **size_stats,
    }
    summary_path = output_dir / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"EDA summary → {summary_path}")

    logger.info("=" * 60)
    logger.info("EDA complete!")
    logger.info(f"  Total complexes: {summary['n_total']}")
    logger.info(f"  Processed .pt:   {summary['n_processed']}")
    logger.info(f"  Quality issues:  {summary['n_quality_issues']}")
    logger.info(f"  Output dir:      {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
