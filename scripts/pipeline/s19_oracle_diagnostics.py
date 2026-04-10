"""
scripts/pipeline/s19_oracle_diagnostics.py
──────────────────────────────────────────
Sub-Plan 4 Phase F.3: Diagnostic plots and analysis for oracle heads.

Generates figures D.2–D.8 and diagnostic JSON files from Tier 1/2 results.

Usage:
    python scripts/pipeline/s19_oracle_diagnostics.py \
        --results_dir results/stage2/oracle_heads \
        --frozen_embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --output_dir results/stage2/oracle_heads/figures \
        --format pdf

    # Specific figures only:
    python scripts/pipeline/s19_oracle_diagnostics.py \
        --results_dir results/stage2/oracle_heads \
        --frozen_embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --output_dir results/stage2/oracle_heads/figures \
        --plots err_scatter,calibration,tier1_bar

Output:
    results/stage2/oracle_heads/figures/
        err_vs_sigma_scatter.{pdf,png}       # Fig. D.2
        calibration_curve.{pdf,png}          # Fig. D.3
        feature_tsne.{pdf,png}               # Fig. D.4
        tier1_comparison_bar.{pdf,png}       # Fig. D.5
        training_curves.{pdf,png}            # Fig. D.6 (if training logs available)
        uncertainty_decomp.{pdf,png}         # Fig. D.7
        binned_err_sigma.{pdf,png}           # Fig. D.8
        uncertainty_diagnostics.json         # Per-head detailed analysis
        ensemble_diagnostics.json            # Pairwise correlations, effective M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.manifold import TSNE

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bayesdiff.hybrid_oracle import (
    DKLOracle,
    DKLEnsembleOracle,
    NNResidualOracle,
    PCA_GPOracle,
)
from bayesdiff.oracle_interface import OracleResult
from bayesdiff.gp_oracle import GPOracle

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HEAD_DISPLAY = {
    "svgp": "Raw SVGP",
    "pca_svgp": "PCA+SVGP",
    "dkl": "DKL",
    "dkl_ensemble": "DKL Ensemble",
    "nn_residual": "NN+GP Residual",
}

HEAD_COLORS = {
    "svgp": "#8da0cb",
    "pca_svgp": "#66c2a5",
    "dkl": "#fc8d62",
    "dkl_ensemble": "#e78ac3",
    "nn_residual": "#a6d854",
}

ALL_PLOTS = [
    "err_scatter",
    "calibration",
    "feature_tsne",
    "tier1_bar",
    "uncertainty_decomp",
    "binned_err_sigma",
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_embeddings(path: str) -> dict:
    """Load frozen embeddings NPZ."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_tier1_json(results_dir: str) -> dict:
    """Load Tier 1 comparison JSON."""
    p = Path(results_dir) / "tier1_comparison.json"
    if not p.exists():
        print(f"[WARN] {p} not found, skipping Tier 1 data")
        return {}
    with open(p) as f:
        return json.load(f)


def load_oracle_head(head_name: str, results_dir: str, input_dim: int,
                     device: str = "cpu"):
    """Load a trained oracle head from saved checkpoint."""
    base = Path(results_dir) / head_name
    if head_name == "dkl":
        oracle = DKLOracle(input_dim=input_dim, device=device)
        oracle.load(str(base))
    elif head_name == "dkl_ensemble":
        oracle = DKLEnsembleOracle(input_dim=input_dim, device=device)
        oracle.load(str(base))
    elif head_name == "nn_residual":
        oracle = NNResidualOracle(input_dim=input_dim, device=device)
        oracle.load(str(base))
    elif head_name == "pca_svgp":
        oracle = PCA_GPOracle(input_dim=input_dim, device=device)
        oracle.load(str(base))
    elif head_name == "svgp":
        oracle = GPOracle(d=input_dim, device=device)
        oracle.load(str(base / "gp_model.pt"))
    else:
        raise ValueError(f"Unknown head: {head_name}")
    return oracle


def oracle_predict(oracle, X: np.ndarray) -> OracleResult:
    """Unified predict call: returns OracleResult regardless of oracle type."""
    if isinstance(oracle, GPOracle):
        mu, var = oracle.predict(X)
        return OracleResult(mu=mu, sigma2=var, aux={})
    return oracle.predict(X)


# --------------------------------------------------------------------------- #
# Fig. D.2: |err| vs σ scatter
# --------------------------------------------------------------------------- #

def plot_err_vs_sigma_scatter(predictions: dict, y_test: np.ndarray,
                              out_path: str, fmt: str):
    """Scatter plot of |error| vs predicted σ for each oracle head."""
    n_heads = len(predictions)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4), squeeze=False)

    for i, (name, result) in enumerate(predictions.items()):
        ax = axes[0, i]
        errors = np.abs(y_test - result.mu)
        sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
        rho, p = spearmanr(errors, sigma)

        ax.scatter(sigma, errors, alpha=0.5, s=15,
                   color=HEAD_COLORS.get(name, "#999999"), edgecolors="none")
        # Fit line
        z = np.polyfit(sigma, errors, 1)
        x_line = np.linspace(sigma.min(), sigma.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1, alpha=0.7)

        ax.set_xlabel(r"Predicted $\sigma$")
        ax.set_ylabel(r"$|y - \mu|$")
        ax.set_title(f"{HEAD_DISPLAY.get(name, name)}\n"
                     rf"$\rho_{{|err|,\sigma}}={rho:.3f}$")

    fig.tight_layout()
    save_path = Path(out_path) / f"err_vs_sigma_scatter.{fmt}"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# --------------------------------------------------------------------------- #
# Fig. D.3: Calibration curves
# --------------------------------------------------------------------------- #

def plot_calibration_curves(predictions: dict, y_test: np.ndarray,
                            out_path: str, fmt: str, n_bins: int = 10):
    """Expected vs observed calibration curves for Gaussian predictions."""
    fig, ax = plt.subplots(figsize=(6, 6))

    confidence_levels = np.linspace(0.1, 0.95, 20)

    for name, result in predictions.items():
        sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
        observed_fracs = []
        for cl in confidence_levels:
            z_crit = norm.ppf((1 + cl) / 2)
            lower = result.mu - z_crit * sigma
            upper = result.mu + z_crit * sigma
            frac_in = np.mean((y_test >= lower) & (y_test <= upper))
            observed_fracs.append(frac_in)

        ax.plot(confidence_levels, observed_fracs, "o-", ms=4, lw=1.5,
                color=HEAD_COLORS.get(name, "#999999"),
                label=HEAD_DISPLAY.get(name, name))

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_xlabel("Expected confidence level")
    ax.set_ylabel("Observed fraction in interval")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    save_path = Path(out_path) / f"calibration_curve.{fmt}"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# --------------------------------------------------------------------------- #
# Fig. D.4: Feature space t-SNE
# --------------------------------------------------------------------------- #

def plot_feature_tsne(X_test: np.ndarray, y_test: np.ndarray,
                      predictions: dict, oracle_heads: dict,
                      out_path: str, fmt: str):
    """t-SNE of raw z vs learned features u, colored by |error|."""
    import torch

    # Get DKL ensemble features if available
    panels = [("Raw embeddings $z$", X_test)]
    for name in ["dkl_ensemble", "dkl"]:
        if name in oracle_heads:
            oracle = oracle_heads[name]
            if hasattr(oracle, "members"):
                # Use first member's feature extractor
                member = oracle.members[0]
                with torch.no_grad():
                    X_t = torch.tensor(X_test, dtype=torch.float32,
                                       device=member.device)
                    u = member.feature_extractor(X_t).cpu().numpy()
                panels.append((f"DKL features $u$ (member 0)", u))
            elif hasattr(oracle, "feature_extractor"):
                with torch.no_grad():
                    X_t = torch.tensor(X_test, dtype=torch.float32,
                                       device=oracle.device)
                    u = oracle.feature_extractor(X_t).cpu().numpy()
                panels.append(("DKL features $u$", u))
            break

    errors = np.abs(y_test - predictions.get("dkl_ensemble",
                     list(predictions.values())[0]).mu)

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, features) in zip(axes, panels):
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        emb = tsne.fit_transform(features)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=errors, cmap="RdYlGn_r",
                        s=20, alpha=0.7, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        plt.colorbar(sc, ax=ax, label=r"$|y - \mu|$")

    fig.tight_layout()
    save_path = Path(out_path) / f"feature_tsne.{fmt}"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# --------------------------------------------------------------------------- #
# Fig. D.5: Tier 1 grouped bar chart
# --------------------------------------------------------------------------- #

def plot_tier1_bar(tier1_data: dict, out_path: str, fmt: str):
    """Grouped bar chart of Tier 1 metrics by oracle head."""
    if not tier1_data:
        print("  [SKIP] No Tier 1 data for bar chart")
        return

    metrics = ["spearman_rho", "R2", "err_sigma_rho"]
    metric_labels = [r"$\rho$", r"$R^2$", r"$\rho_{|err|,\sigma}$"]
    heads = [h for h in HEAD_DISPLAY if h in tier1_data]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [tier1_data[h]["test"].get(metric, 0) for h in heads]
        bars = ax.bar(range(len(heads)), vals,
                      color=[HEAD_COLORS.get(h, "#999") for h in heads])
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([HEAD_DISPLAY[h] for h in heads],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)

        # Add value annotations
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Tier 1: Oracle Head Comparison (CASF-2016 Test)", fontsize=11)
    fig.tight_layout()
    save_path = Path(out_path) / f"tier1_comparison_bar.{fmt}"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# --------------------------------------------------------------------------- #
# Fig. D.7: Uncertainty decomposition
# --------------------------------------------------------------------------- #

def plot_uncertainty_decomp(predictions: dict, y_test: np.ndarray,
                            out_path: str, fmt: str):
    """Stacked σ²_alea vs σ²_epi sorted by |error| (for DKL Ensemble)."""
    name = "dkl_ensemble"
    if name not in predictions:
        print("  [SKIP] No DKL Ensemble predictions for decomposition")
        return

    result = predictions[name]
    if "sigma2_aleatoric" not in result.aux or "sigma2_epistemic" not in result.aux:
        print("  [SKIP] DKL Ensemble missing aleatoric/epistemic decomposition")
        return

    errors = np.abs(y_test - result.mu)
    sort_idx = np.argsort(errors)

    sigma2_a = result.aux["sigma2_aleatoric"][sort_idx]
    sigma2_e = result.aux["sigma2_epistemic"][sort_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(sort_idx))
    ax.bar(x, sigma2_a, label=r"$\sigma^2_{\mathrm{aleatoric}}$",
           color="#66c2a5", width=1.0)
    ax.bar(x, sigma2_e, bottom=sigma2_a,
           label=r"$\sigma^2_{\mathrm{epistemic}}$",
           color="#fc8d62", width=1.0)
    ax.set_xlabel("Sample index (sorted by |error|)")
    ax.set_ylabel(r"$\sigma^2$")
    ax.set_title("DKL Ensemble: Uncertainty Decomposition")
    ax.legend()

    # Overlay sorted errors on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, errors[sort_idx], "k-", lw=0.8, alpha=0.5, label="|error|")
    ax2.set_ylabel(r"$|y - \mu|$")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    save_path = Path(out_path) / f"uncertainty_decomp.{fmt}"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# --------------------------------------------------------------------------- #
# Fig. D.8: Binned error-sigma analysis
# --------------------------------------------------------------------------- #

def plot_binned_err_sigma(predictions: dict, y_test: np.ndarray,
                          out_path: str, fmt: str, n_bins: int = 5):
    """Mean |error| per σ-quantile bin — monotonicity check."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, result in predictions.items():
        sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
        errors = np.abs(y_test - result.mu)

        # Bin by sigma quantiles
        bin_edges = np.percentile(sigma, np.linspace(0, 100, n_bins + 1))
        bin_means_err = []
        bin_means_sig = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (sigma >= lo) & (sigma <= hi)
            if mask.sum() > 0:
                bin_means_err.append(errors[mask].mean())
                bin_means_sig.append(sigma[mask].mean())

        ax.plot(bin_means_sig, bin_means_err, "o-", lw=1.5, ms=6,
                color=HEAD_COLORS.get(name, "#999"),
                label=HEAD_DISPLAY.get(name, name))

    ax.set_xlabel(r"Mean $\sigma$ per bin")
    ax.set_ylabel(r"Mean $|y - \mu|$ per bin")
    ax.set_title("Binned Error vs Uncertainty (Monotonicity Check)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_path = Path(out_path) / f"binned_err_sigma.{fmt}"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# --------------------------------------------------------------------------- #
# Diagnostics JSON generation
# --------------------------------------------------------------------------- #

def compute_uncertainty_diagnostics(predictions: dict, y_test: np.ndarray) -> dict:
    """Compute per-head uncertainty diagnostics."""
    diagnostics = {}
    for name, result in predictions.items():
        errors = np.abs(y_test - result.mu)
        sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
        rho_err_sigma, p_val = spearmanr(errors, sigma)

        d = {
            "rho_err_sigma": float(rho_err_sigma),
            "rho_err_sigma_pval": float(p_val),
            "mean_sigma": float(sigma.mean()),
            "std_sigma": float(sigma.std()),
            "mean_error": float(errors.mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
        }

        # Decomposition if available
        if "sigma2_aleatoric" in result.aux and "sigma2_epistemic" in result.aux:
            sigma_a = np.sqrt(np.clip(result.aux["sigma2_aleatoric"], 1e-10, None))
            sigma_e = np.sqrt(np.clip(result.aux["sigma2_epistemic"], 1e-10, None))
            rho_a, _ = spearmanr(errors, sigma_a)
            rho_e, _ = spearmanr(errors, sigma_e)
            d["rho_err_sigma_aleatoric"] = float(rho_a)
            d["rho_err_sigma_epistemic"] = float(rho_e)
            d["mean_aleatoric_frac"] = float(
                (result.aux["sigma2_aleatoric"] / result.sigma2).mean()
            )
            d["mean_epistemic_frac"] = float(
                (result.aux["sigma2_epistemic"] / result.sigma2).mean()
            )

        diagnostics[name] = d

    return diagnostics


def compute_ensemble_diagnostics(predictions: dict) -> dict:
    """Compute ensemble diversity metrics for DKL Ensemble."""
    name = "dkl_ensemble"
    if name not in predictions:
        return {}

    result = predictions[name]
    if "member_mus" not in result.aux:
        return {}

    member_mus = result.aux["member_mus"]  # (M, N)
    M = member_mus.shape[0]

    # Pairwise Spearman correlations
    pairwise_rho = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            pairwise_rho[i, j], _ = spearmanr(member_mus[i], member_mus[j])

    # Effective ensemble size (Lakshminarayanan et al.)
    off_diag = pairwise_rho[np.triu_indices(M, k=1)]
    mean_corr = float(off_diag.mean())
    # M_eff = M / (1 + (M-1) * mean_corr)
    m_eff = M / (1 + (M - 1) * mean_corr) if (1 + (M - 1) * mean_corr) > 0 else M

    return {
        "n_members": M,
        "pairwise_rho_matrix": pairwise_rho.tolist(),
        "mean_pairwise_rho": mean_corr,
        "min_pairwise_rho": float(off_diag.min()),
        "max_pairwise_rho": float(off_diag.max()),
        "effective_ensemble_size": float(m_eff),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Sub-Plan 04 F.3: Oracle head diagnostic plots and analysis"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with trained oracle heads")
    parser.add_argument("--frozen_embeddings", type=str, default=None,
                        help="Path to frozen_embeddings.npz (default: results_dir/frozen_embeddings.npz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for figures")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"],
                        help="Figure output format")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plots", type=str, default="all",
                        help="Comma-separated plot names or 'all'")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = args.frozen_embeddings or str(Path(args.results_dir) / "frozen_embeddings.npz")
    print(f"Loading embeddings from {emb_path}")
    data = load_embeddings(emb_path)
    X_test, y_test = data["X_test"], data["y_test"]
    input_dim = X_test.shape[1]

    # Load Tier 1 JSON
    tier1_data = load_tier1_json(args.results_dir)

    # Determine which heads have saved models
    available_heads = []
    for head_name in HEAD_DISPLAY:
        head_dir = Path(args.results_dir) / head_name
        if head_dir.exists():
            available_heads.append(head_name)
    print(f"Available heads: {available_heads}")

    # Load models and generate predictions
    print("Loading oracle heads and generating predictions...")
    predictions = {}
    oracle_heads = {}
    for head_name in available_heads:
        try:
            oracle = load_oracle_head(head_name, args.results_dir, input_dim,
                                      device=args.device)
            result = oracle_predict(oracle, X_test)
            predictions[head_name] = result
            oracle_heads[head_name] = oracle
            rho, _ = spearmanr(y_test, result.mu)
            print(f"  {head_name}: ρ={rho:.3f}, mean_σ={np.sqrt(result.sigma2).mean():.3f}")
        except Exception as e:
            print(f"  [WARN] Failed to load {head_name}: {e}")

    if not predictions:
        print("[ERROR] No predictions generated. Check model paths.")
        sys.exit(1)

    # Select plots
    if args.plots == "all":
        plot_list = ALL_PLOTS
    else:
        plot_list = [p.strip() for p in args.plots.split(",")]

    fmt = args.format

    # Generate plots
    print(f"\nGenerating figures (format={fmt})...")

    if "err_scatter" in plot_list:
        print("  Fig. D.2: |err| vs σ scatter")
        plot_err_vs_sigma_scatter(predictions, y_test, str(out_dir), fmt)

    if "calibration" in plot_list:
        print("  Fig. D.3: Calibration curves")
        plot_calibration_curves(predictions, y_test, str(out_dir), fmt)

    if "feature_tsne" in plot_list:
        print("  Fig. D.4: Feature space t-SNE")
        plot_feature_tsne(X_test, y_test, predictions, oracle_heads,
                          str(out_dir), fmt)

    if "tier1_bar" in plot_list:
        print("  Fig. D.5: Tier 1 comparison bar chart")
        plot_tier1_bar(tier1_data, str(out_dir), fmt)

    if "uncertainty_decomp" in plot_list:
        print("  Fig. D.7: Uncertainty decomposition")
        plot_uncertainty_decomp(predictions, y_test, str(out_dir), fmt)

    if "binned_err_sigma" in plot_list:
        print("  Fig. D.8: Binned error-sigma")
        plot_binned_err_sigma(predictions, y_test, str(out_dir), fmt)

    # Generate diagnostic JSONs
    print("\nGenerating diagnostic JSONs...")
    unc_diag = compute_uncertainty_diagnostics(predictions, y_test)
    unc_path = out_dir / "uncertainty_diagnostics.json"
    with open(unc_path, "w") as f:
        json.dump(unc_diag, f, indent=2)
    print(f"  Saved {unc_path}")

    ens_diag = compute_ensemble_diagnostics(predictions)
    if ens_diag:
        ens_path = out_dir / "ensemble_diagnostics.json"
        with open(ens_path, "w") as f:
            json.dump(ens_diag, f, indent=2)
        print(f"  Saved {ens_path}")

    print("\n=== Diagnostics Summary ===")
    for name, d in unc_diag.items():
        print(f"  {HEAD_DISPLAY.get(name, name):20s}: "
              f"ρ_err_σ={d['rho_err_sigma']:.4f}  "
              f"RMSE={d['rmse']:.3f}  "
              f"mean_σ={d['mean_sigma']:.3f}")
    if ens_diag:
        print(f"  Ensemble: M_eff={ens_diag['effective_ensemble_size']:.2f} "
              f"(mean ρ={ens_diag['mean_pairwise_rho']:.3f})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
