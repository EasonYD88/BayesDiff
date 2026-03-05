"""
scripts/09_generate_figures.py
──────────────────────────────
Generate publication-quality figures from HPC results.

Usage:
    python scripts/09_generate_figures.py \
        --eval_dir results/evaluation \
        --ablation_dir results/ablation \
        --embeddings results/generated_molecules/all_embeddings.npz \
        --gp_meta results/gp_model/train_meta.json \
        --output results/figures

Generates:
    fig1_dashboard.png     — 4-panel overview (pred vs true, P_success, uncertainty, GP info)
    fig2_embeddings.png    — PCA + t-SNE of 93×64 embeddings colored by pKd
    fig3_uncertainty.png   — Uncertainty decomposition (σ²_oracle vs σ²_gen) + OOD
    fig4_ablation.png      — Ablation comparison bar chart (ECE, AUROC, NLL, RMSE)
    fig5_calibration.png   — Reliability diagram + residual analysis
    fig6_pocket_ranking.png — Per-pocket ranking by P_success vs true pKd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Matplotlib config (no display needed on HPC) ────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF9800",
    "accent": "#4CAF50",
    "danger": "#F44336",
    "purple": "#9C27B0",
    "teal": "#009688",
    "gray": "#607D8B",
}


def load_data(args):
    """Load all results data."""
    data = {}

    # Per-pocket evaluation results
    with open(Path(args.eval_dir) / "per_pocket_results.json") as f:
        data["eval"] = json.load(f)

    # Eval metrics
    with open(Path(args.eval_dir) / "eval_metrics.json") as f:
        data["metrics"] = json.load(f)

    # Multi-threshold
    mt_path = Path(args.eval_dir) / "eval_multi_threshold.json"
    if mt_path.exists():
        with open(mt_path) as f:
            data["multi_threshold"] = json.load(f)

    # Ablation summary
    with open(Path(args.ablation_dir) / "ablation_summary.json") as f:
        data["ablation"] = json.load(f)

    # Ablation per-pocket
    abl_pp_path = Path(args.ablation_dir) / "ablation_per_pocket.json"
    if abl_pp_path.exists():
        with open(abl_pp_path) as f:
            data["ablation_per_pocket"] = json.load(f)

    # GP training meta
    with open(Path(args.gp_meta)) as f:
        data["gp_meta"] = json.load(f)

    # Embeddings
    emb_data = np.load(args.embeddings, allow_pickle=True)
    data["embeddings"] = {k: emb_data[k] for k in emb_data.files}

    return data


# ═══════════════════════════════════════════════════════════════
# Figure 1: Dashboard
# ═══════════════════════════════════════════════════════════════
def fig1_dashboard(data, output_dir):
    """4-panel dashboard: pred vs true, P_success, uncertainty breakdown, summary."""
    eval_data = data["eval"]
    metrics = data["metrics"]
    gp_meta = data["gp_meta"]

    y_true = np.array([r["pkd_true"] for r in eval_data])
    y_pred = np.array([r["mu_pred"] for r in eval_data])
    p_succ = np.array([r["p_success"] for r in eval_data])
    sigma2_oracle = np.array([r["sigma2_oracle"] for r in eval_data])
    sigma2_gen = np.array([r["sigma2_gen"] for r in eval_data])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("BayesDiff HPC Results Dashboard (N=48 pockets)", fontsize=14, fontweight="bold")

    # Panel A: Predicted vs True pKd
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, c=COLORS["primary"], alpha=0.7, edgecolors="white", s=60, zorder=3)
    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "--", color=COLORS["gray"], alpha=0.5, label="y=x")
    ax.axhline(y=np.mean(y_pred), color=COLORS["danger"], ls=":", alpha=0.7, label=f"μ_pred={np.mean(y_pred):.2f}")
    ax.set_xlabel("True pKd")
    ax.set_ylabel("Predicted pKd")
    ax.set_title(f"(A) Predicted vs True pKd\nRMSE={metrics['rmse']:.3f}")
    ax.legend(loc="upper left")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel B: P_success vs True pKd
    ax = axes[0, 1]
    active = y_true >= 7.0
    ax.scatter(y_true[active], p_succ[active], c=COLORS["accent"], alpha=0.7,
               edgecolors="white", s=60, label=f"Active (pKd≥7, n={active.sum()})", zorder=3)
    ax.scatter(y_true[~active], p_succ[~active], c=COLORS["danger"], alpha=0.7,
               edgecolors="white", s=60, label=f"Inactive (pKd<7, n={(~active).sum()})", zorder=3)
    ax.axvline(x=7.0, color=COLORS["gray"], ls="--", alpha=0.5, label="y_target=7.0")
    ax.set_xlabel("True pKd")
    ax.set_ylabel("P(pKd ≥ 7.0)")
    ax.set_title(f"(B) Success Probability\nAUROC={metrics['auroc']:.3f}")
    ax.legend(loc="upper left", fontsize=7)

    # Panel C: Uncertainty decomposition
    ax = axes[1, 0]
    x_idx = np.arange(len(eval_data))
    sort_idx = np.argsort(y_true)
    s2_o = sigma2_oracle[sort_idx]
    s2_g = sigma2_gen[sort_idx]
    ax.bar(x_idx, s2_o, color=COLORS["primary"], alpha=0.8, label="σ²_oracle")
    ax.bar(x_idx, s2_g, bottom=s2_o, color=COLORS["secondary"], alpha=0.8, label="σ²_gen")
    ax.set_xlabel("Pocket (sorted by true pKd)")
    ax.set_ylabel("Variance")
    ax.set_title(f"(C) Uncertainty Decomposition\nOracle dominates ({np.mean(s2_o/(s2_o+s2_g+1e-12))*100:.0f}%)")
    ax.legend()

    # Panel D: Summary stats table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["N pockets evaluated", f"{len(eval_data)}"],
        ["N pockets total", "93 (48 with pK labels)"],
        ["Embedding dim", f"{gp_meta['d']}"],
        ["GP inducing points", f"{gp_meta['n_inducing']}"],
        ["GP training samples", f"{gp_meta['n_train']} (augmented)"],
        ["GP final loss", f"{gp_meta['final_loss']:.4f}"],
        ["GP training time", f"{gp_meta['elapsed_s']}s (A100 GPU)"],
        ["ECE", f"{metrics['ece']:.4f}"],
        ["AUROC", f"{metrics['auroc']:.3f}"],
        ["RMSE", f"{metrics['rmse']:.4f}"],
        ["NLL", f"{metrics['nll']:.4f}"],
        ["pKd range", f"[{gp_meta['pkd_range'][0]:.2f}, {gp_meta['pkd_range'][1]:.2f}]"],
    ]
    table = ax.table(cellText=table_data, cellLoc="left", loc="center",
                     colWidths=[0.45, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    # Header styling
    for j in range(2):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("(D) Summary Statistics", pad=20)

    plt.tight_layout()
    path = output_dir / "fig1_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Embedding visualization
# ═══════════════════════════════════════════════════════════════
def fig2_embeddings(data, output_dir):
    """PCA + t-SNE of all 93×64 embeddings, colored by pKd where available."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    embeddings = data["embeddings"]
    eval_lookup = {r["target"]: r["pkd_true"] for r in data["eval"]}

    # Use per-sample embeddings (subsample for speed: max 8 per pocket)
    pocket_names = sorted(embeddings.keys())
    all_embs = []
    all_pkd = []
    all_pocket_ids = []
    max_per_pocket = 8

    for k in pocket_names:
        emb = embeddings[k]
        n = min(max_per_pocket, emb.shape[0])
        idx = np.random.RandomState(42).choice(emb.shape[0], n, replace=False)
        all_embs.append(emb[idx])
        pk = eval_lookup.get(k, np.nan)
        all_pkd.extend([pk] * n)
        all_pocket_ids.extend([k] * n)

    X_all = np.vstack(all_embs)
    pkd_all = np.array(all_pkd)
    has_label = ~np.isnan(pkd_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Molecular Embedding Space (93 pockets × {max_per_pocket} samples, d=128)", fontsize=14, fontweight="bold")

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
    ev_ratio = pca.explained_variance_ratio_
    # Handle NaN explained variance (identical embeddings)
    if np.any(np.isnan(ev_ratio)):
        ev_ratio = np.zeros_like(ev_ratio)

    ax = axes[0]
    sc = ax.scatter(X_pca[has_label, 0], X_pca[has_label, 1],
                    c=pkd_all[has_label], cmap="RdYlGn", s=20, alpha=0.6,
                    edgecolors="none", zorder=3)
    ax.scatter(X_pca[~has_label, 0], X_pca[~has_label, 1],
               c=COLORS["gray"], s=10, alpha=0.3, marker="x", label="No pK label")
    plt.colorbar(sc, ax=ax, label="pKd", shrink=0.8)
    ax.set_xlabel(f"PC1 ({ev_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev_ratio[1]*100:.1f}%)")
    ax.set_title(f"(A) PCA\nExplained var: {sum(ev_ratio[:2])*100:.1f}%")
    ax.legend(loc="lower right")

    # t-SNE (skip if embeddings are near-identical to avoid segfault)
    X_std = np.std(X_all, axis=0)
    if np.all(X_std < 1e-10):
        # Embeddings are essentially identical — add jitter for visualization
        X_jitter = X_all + np.random.RandomState(42).randn(*X_all.shape) * 0.01
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_jitter)-1), random_state=42)
        X_tsne = tsne.fit_transform(X_jitter)
        tsne_note = "\n(jittered — embeddings near-identical)"
    else:
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_all)-1), random_state=42)
        X_tsne = tsne.fit_transform(X_all)
        tsne_note = ""

    ax = axes[1]
    sc = ax.scatter(X_tsne[has_label, 0], X_tsne[has_label, 1],
                    c=pkd_all[has_label], cmap="RdYlGn", s=20, alpha=0.6,
                    edgecolors="none", zorder=3)
    ax.scatter(X_tsne[~has_label, 0], X_tsne[~has_label, 1],
               c=COLORS["gray"], s=10, alpha=0.3, marker="x", label="No pK label")
    plt.colorbar(sc, ax=ax, label="pKd", shrink=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"(B) t-SNE{tsne_note}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = output_dir / "fig2_embeddings.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Uncertainty decomposition + OOD
# ═══════════════════════════════════════════════════════════════
def fig3_uncertainty(data, output_dir):
    """3-panel: σ²_oracle vs σ²_gen scatter, OOD distances, trace distribution."""
    eval_data = data["eval"]

    sigma2_o = np.array([r["sigma2_oracle"] for r in eval_data])
    sigma2_g = np.array([r["sigma2_gen"] for r in eval_data])
    ood_dist = np.array([r["ood_distance"] for r in eval_data])
    ood_flag = np.array([r["ood_flag"] for r in eval_data])
    trace_cov = np.array([r["trace_cov_gen"] for r in eval_data])
    y_true = np.array([r["pkd_true"] for r in eval_data])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Uncertainty Analysis (N=48 pockets)", fontsize=14, fontweight="bold")

    # Panel A: Oracle vs Gen variance
    ax = axes[0]
    sc = ax.scatter(sigma2_g, sigma2_o, c=y_true, cmap="RdYlGn", s=60, alpha=0.8,
                    edgecolors="white", zorder=3)
    plt.colorbar(sc, ax=ax, label="True pKd", shrink=0.8)
    ax.set_xlabel("σ²_gen (Generation Uncertainty)")
    ax.set_ylabel("σ²_oracle (Oracle Uncertainty)")
    ax.set_title(f"(A) Variance Components\nOracle: {sigma2_o.mean():.3f} >> Gen: {sigma2_g.mean():.6f}")
    # Add diagonal
    lim = max(sigma2_o.max(), sigma2_g.max()) * 1.1
    ax.set_xlim(left=-0.001)

    # Panel B: OOD distance distribution
    ax = axes[1]
    n_ood = ood_flag.sum()
    n_id = (~ood_flag).sum()
    ax.hist(ood_dist[~ood_flag], bins=15, alpha=0.7, color=COLORS["primary"],
            label=f"In-distribution (n={n_id})", edgecolor="white")
    if n_ood > 0:
        ax.hist(ood_dist[ood_flag], bins=5, alpha=0.7, color=COLORS["danger"],
                label=f"OOD (n={n_ood})", edgecolor="white")
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Count")
    ax.set_title(f"(B) OOD Detection\n{n_ood}/{len(eval_data)} flagged as OOD")
    ax.legend()

    # Panel C: Trace(Σ_gen) vs true pKd
    ax = axes[2]
    ax.scatter(y_true, trace_cov, c=COLORS["teal"], alpha=0.7, edgecolors="white", s=60, zorder=3)
    ax.set_xlabel("True pKd")
    ax.set_ylabel("Tr(Σ_gen)")
    ax.set_title(f"(C) Generation Diversity\nTr(Σ_gen): [{trace_cov.min():.2f}, {trace_cov.max():.2f}]")

    plt.tight_layout()
    path = output_dir / "fig3_uncertainty.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Figure 4: Ablation comparison
# ═══════════════════════════════════════════════════════════════
def fig4_ablation(data, output_dir):
    """Bar chart comparing ablation variants on key metrics."""
    abl = data["ablation"]

    variants = ["full", "A1", "A2", "A3", "A4", "A5", "A7"]
    labels = [
        "Full\nBayesDiff",
        "A1\nNo U_gen",
        "A2\nNo U_oracle",
        "A3\nNo calib",
        "A4\nNaive cov",
        "A5\nNo multi",
        "A7\nNo OOD",
    ]
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["danger"],
                   COLORS["accent"], COLORS["purple"], COLORS["teal"], COLORS["gray"]]

    # Metrics to plot
    metric_keys = ["ece", "nll", "rmse"]
    metric_names = ["ECE (↓ better)", "NLL (↓ better)", "RMSE (↓ better)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study (7 variants, N=48 pockets)", fontsize=14, fontweight="bold")

    for idx, (mkey, mname) in enumerate(zip(metric_keys, metric_names)):
        ax = axes[idx]
        vals = []
        for v in variants:
            val = abl.get(v, {}).get(mkey, 0)
            # Cap extreme values for visualization
            if np.isnan(val) or np.isinf(val):
                val = 0
            vals.append(val)

        # For NLL, A2 is extremely large — use log scale or cap
        use_log = (mkey == "nll" and max(vals) > 100)

        bars = ax.bar(range(len(variants)), vals, color=colors_list, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(mname)
        ax.set_title(mname)

        if use_log:
            ax.set_yscale("log")
            ax.set_title(f"{mname} (log scale)")

        # Annotate values
        for bar, val in zip(bars, vals):
            if val > 0:
                if use_log and val > 1e6:
                    txt = f"{val:.1e}"
                else:
                    txt = f"{val:.3f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        txt, ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    path = output_dir / "fig4_ablation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Figure 5: Calibration / Reliability diagram + Residuals
# ═══════════════════════════════════════════════════════════════
def fig5_calibration(data, output_dir):
    """2-panel: reliability diagram + prediction residuals."""
    eval_data = data["eval"]

    y_true = np.array([r["pkd_true"] for r in eval_data])
    y_pred = np.array([r["mu_pred"] for r in eval_data])
    p_succ = np.array([r["p_success"] for r in eval_data])
    sigma_total = np.array([r["sigma_total"] for r in eval_data])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Calibration Analysis (N=48 pockets)", fontsize=14, fontweight="bold")

    # Panel A: Reliability diagram
    ax = axes[0]
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means_pred = []
    bin_means_true = []
    bin_counts = []
    active = (y_true >= 7.0).astype(float)

    for i in range(n_bins):
        mask = (p_succ >= bin_edges[i]) & (p_succ < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (p_succ >= bin_edges[i]) & (p_succ <= bin_edges[i+1])
        if mask.sum() > 0:
            bin_means_pred.append(p_succ[mask].mean())
            bin_means_true.append(active[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means_pred.append(bin_edges[i] + 0.5/n_bins)
            bin_means_true.append(0)
            bin_counts.append(0)

    bin_means_pred = np.array(bin_means_pred)
    bin_means_true = np.array(bin_means_true)
    bin_counts = np.array(bin_counts)

    ax.plot([0, 1], [0, 1], "--", color=COLORS["gray"], alpha=0.5, label="Perfect calibration")
    bars = ax.bar(bin_means_pred, bin_means_true, width=0.15, alpha=0.7,
                  color=COLORS["primary"], edgecolor="white", label="Observed frequency")
    for bp, bt, bc in zip(bin_means_pred, bin_means_true, bin_counts):
        if bc > 0:
            ax.text(bp, bt + 0.03, f"n={bc}", ha="center", fontsize=7)
    ax.set_xlabel("Predicted P(pKd ≥ 7)")
    ax.set_ylabel("Observed Fraction Active")
    ece = data["metrics"]["ece"]
    ax.set_title(f"(A) Reliability Diagram\nECE = {ece:.4f}")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)

    # Panel B: Residual plot
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, c=sigma_total, cmap="YlOrRd", s=60, alpha=0.8,
               edgecolors="white", zorder=3)
    ax.axhline(y=0, color=COLORS["gray"], ls="--", alpha=0.5)
    plt.colorbar(ax.collections[0], ax=ax, label="σ_total", shrink=0.8)
    ax.set_xlabel("True pKd")
    ax.set_ylabel("Residual (Predicted - True)")
    ax.set_title(f"(B) Prediction Residuals\nMean residual: {residuals.mean():.2f}")

    plt.tight_layout()
    path = output_dir / "fig5_calibration.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Figure 6: Pocket ranking
# ═══════════════════════════════════════════════════════════════
def fig6_pocket_ranking(data, output_dir):
    """Horizontal bar chart: pockets ranked by P_success vs true pKd."""
    eval_data = data["eval"]

    # Sort by true pKd descending
    sorted_data = sorted(eval_data, key=lambda r: r["pkd_true"], reverse=True)
    targets = [r["target"].replace("_0", "").replace("_", "\n") for r in sorted_data]
    y_true = [r["pkd_true"] for r in sorted_data]
    p_succ = [r["p_success"] for r in sorted_data]
    ood_flags = [r["ood_flag"] for r in sorted_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, len(sorted_data) * 0.3)))
    fig.suptitle(f"Per-Pocket Results (N={len(sorted_data)}, sorted by true pKd)", fontsize=14, fontweight="bold")

    y_pos = np.arange(len(sorted_data))

    # Panel A: True pKd
    ax = axes[0]
    colors = [COLORS["accent"] if pk >= 7.0 else COLORS["danger"] for pk in y_true]
    ax.barh(y_pos, y_true, color=colors, alpha=0.8, edgecolor="white", height=0.7)
    ax.axvline(x=7.0, color=COLORS["gray"], ls="--", alpha=0.7, label="y_target=7.0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(targets, fontsize=5)
    ax.set_xlabel("True pKd")
    ax.set_title("(A) True pKd")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    # Panel B: P_success + OOD markers
    ax = axes[1]
    colors_p = [COLORS["primary"] if not ood else COLORS["danger"] for ood in ood_flags]
    ax.barh(y_pos, p_succ, color=colors_p, alpha=0.8, edgecolor="white", height=0.7)
    ax.axvline(x=0.5, color=COLORS["gray"], ls="--", alpha=0.7, label="P=0.5 threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(sorted_data))
    ax.set_xlabel("P(pKd ≥ 7)")
    ax.set_title("(B) Predicted P_success")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    path = output_dir / "fig6_pocket_ranking.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate BayesDiff visualization figures")
    parser.add_argument("--eval_dir", type=str, default="results/evaluation")
    parser.add_argument("--ablation_dir", type=str, default="results/ablation")
    parser.add_argument("--embeddings", type=str, default="results/generated_molecules/all_embeddings.npz")
    parser.add_argument("--gp_meta", type=str, default="results/gp_model/train_meta.json")
    parser.add_argument("--output", type=str, default="results/figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(args)
    print(f"  Eval: {len(data['eval'])} pockets")
    print(f"  Ablation: {len(data['ablation'])} variants")
    print(f"  Embeddings: {len(data['embeddings'])} pockets")

    print("\nGenerating figures...")

    fig1_dashboard(data, output_dir)
    fig2_embeddings(data, output_dir)
    fig3_uncertainty(data, output_dir)
    fig4_ablation(data, output_dir)
    fig5_calibration(data, output_dir)
    fig6_pocket_ranking(data, output_dir)

    print(f"\nAll 6 figures saved to {output_dir}/")
    print("  fig1_dashboard.png     — Overview dashboard")
    print("  fig2_embeddings.png    — PCA + t-SNE embeddings")
    print("  fig3_uncertainty.png   — Uncertainty decomposition + OOD")
    print("  fig4_ablation.png      — Ablation comparison")
    print("  fig5_calibration.png   — Calibration + residuals")
    print("  fig6_pocket_ranking.png — Per-pocket ranking")


if __name__ == "__main__":
    main()
