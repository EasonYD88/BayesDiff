"""
scripts/12_robust_evaluation.py
────────────────────────────────
Robust GP evaluation with three protocols:
  1. Exact GP + analytic LOOCV
  2. 50× repeated 70/30 random split
  3. Pocket-level bootstrap (1000×)

Reports mean ± std for all metrics. Works with any embedding .npz.

Usage:
    python scripts/12_robust_evaluation.py \
        --embeddings results/embedding_rdkit/all_embeddings.npz \
        --affinity_pkl external/targetdiff/data/affinity_info.pkl \
        --output results/embedding_rdkit/robust_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gpytorch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
})


# ── Data helpers ─────────────────────────────────────────────
def load_affinity_pkl(pkl_path: Path) -> dict[str, float]:
    with open(pkl_path, "rb") as f:
        affinity = pickle.load(f)
    pocket_pks: dict[str, list[float]] = {}
    for key, info in affinity.items():
        pk = info.get("pk")
        if pk is None or float(pk) == 0.0:
            continue
        pocket_fam = str(key).split("/")[0]
        pocket_pks.setdefault(pocket_fam, []).append(float(pk))
    return {fam: float(np.mean(vals)) for fam, vals in pocket_pks.items()}


def build_dataset(emb_path: str, pkl_path: str):
    """Return X (N, d), y (N,), names (N,)."""
    emb_data = np.load(emb_path, allow_pickle=True)
    label_map = load_affinity_pkl(Path(pkl_path))
    X_list, y_list, names = [], [], []
    for name in emb_data.files:
        pk = label_map.get(name)
        if pk is None:
            continue
        emb = emb_data[name]
        z_mean = emb.mean(axis=0) if emb.ndim == 2 else emb
        X_list.append(z_mean)
        y_list.append(pk)
        names.append(name)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, names


# ── Exact GP model ───────────────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, use_ard=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ard = train_x.shape[1] if use_ard else None
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_exact_gp(X, y, n_epochs=100, lr=0.1, use_ard=False, verbose=False):
    """Train an Exact GP and return model + likelihood."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_t, y_t, likelihood, use_ard=use_ard)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = -mll(output, y_t)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 25 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}: MLL={-loss.item():.4f}, "
                        f"noise={likelihood.noise.item():.4f}")

    model.eval()
    likelihood.eval()
    return model, likelihood


def predict_exact_gp(model, likelihood, X):
    """Predict with an Exact GP."""
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    return pred.mean.numpy(), pred.variance.numpy()


# ── Metrics ──────────────────────────────────────────────────
def compute_metrics(y_true, mu, var):
    """Compute regression + calibration metrics."""
    sigma = np.sqrt(np.maximum(var, 1e-10))
    rmse = np.sqrt(np.mean((mu - y_true) ** 2))
    mae = np.mean(np.abs(mu - y_true))
    nll = 0.5 * np.mean(np.log(2 * np.pi * np.maximum(var, 1e-10))
                        + (y_true - mu) ** 2 / np.maximum(var, 1e-10))
    n = len(y_true)
    r2 = 1 - np.sum((y_true - mu) ** 2) / np.sum((y_true - y_true.mean()) ** 2) if n > 1 else float('nan')

    if n > 2:
        sp_rho, sp_p = spearmanr(y_true, mu)
        pe_r, pe_p = pearsonr(y_true, mu)
    else:
        sp_rho = sp_p = pe_r = pe_p = float('nan')

    ci_low = mu - 1.96 * sigma
    ci_hi = mu + 1.96 * sigma
    cov95 = np.mean((y_true >= ci_low) & (y_true <= ci_hi))
    ci_width = np.mean(2 * 1.96 * sigma)

    return {
        "rmse": float(rmse), "mae": float(mae), "nll": float(nll),
        "r2": float(r2), "spearman_rho": float(sp_rho), "spearman_p": float(sp_p),
        "pearson_r": float(pe_r), "coverage_95": float(cov95),
        "ci_width": float(ci_width), "n": n,
    }


# ═══════════════════════════════════════════════════════════════
# Protocol 1: Analytic LOOCV with Exact GP
# ═══════════════════════════════════════════════════════════════
def run_loocv(X, y, n_epochs=100, lr=0.1, use_ard=False):
    """
    Train Exact GP on ALL data, then compute analytic LOOCV predictions.

    For an Exact GP with kernel matrix K and noise σ²_n:
        K_full = K(X,X) + σ²_n I
        α = K_full⁻¹ y
        μ_LOO_i = y_i - α_i / [K_full⁻¹]_ii
        σ²_LOO_i = 1 / [K_full⁻¹]_ii
    """
    logger.info("  Training Exact GP on all data for analytic LOOCV...")
    model, likelihood = train_exact_gp(X, y, n_epochs=n_epochs, lr=lr,
                                        use_ard=use_ard, verbose=True)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    with torch.no_grad():
        # Build full covariance matrix
        K = model.covar_module(X_t).evaluate()
        noise = likelihood.noise.item()
        K_full = K + noise * torch.eye(len(X_t))

        # Analytic LOOCV
        K_inv = torch.linalg.inv(K_full)
        alpha = K_inv @ y_t

        loocv_mu = y_t - alpha / K_inv.diag()
        loocv_var = 1.0 / K_inv.diag()

    mu_loo = loocv_mu.numpy()
    var_loo = loocv_var.numpy()

    # Clamp negative variances (numerical)
    var_loo = np.maximum(var_loo, 1e-10)

    metrics = compute_metrics(y, mu_loo, var_loo)
    logger.info(f"  LOOCV: RMSE={metrics['rmse']:.3f}, ρ={metrics['spearman_rho']:.3f}, "
                f"R²={metrics['r2']:.3f}, NLL={metrics['nll']:.3f}, CI95%={metrics['coverage_95']:.0%}")

    return {
        "metrics": metrics,
        "mu": mu_loo.tolist(),
        "var": var_loo.tolist(),
        "noise": noise,
        "hyperparams": {
            "outputscale": model.covar_module.outputscale.item(),
            "noise": noise,
            "lengthscale_mean": model.covar_module.base_kernel.lengthscale.mean().item(),
        },
    }


# ═══════════════════════════════════════════════════════════════
# Protocol 2: Repeated Random Split
# ═══════════════════════════════════════════════════════════════
def run_repeated_split(X, y, n_repeats=50, test_frac=0.3,
                       n_epochs=60, lr=0.1, use_ard=False, seed=42):
    """Train/test split n_repeats times, collect test metrics."""
    rng = np.random.default_rng(seed)
    N = len(y)
    n_test = max(2, int(N * test_frac))
    n_train = N - n_test

    all_metrics = []
    for rep in range(n_repeats):
        idx = rng.permutation(N)
        tr_idx, te_idx = idx[:n_train], idx[n_train:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        model, lik = train_exact_gp(X_tr, y_tr, n_epochs=n_epochs, lr=lr,
                                     use_ard=use_ard)
        mu_te, var_te = predict_exact_gp(model, lik, X_te)
        m = compute_metrics(y_te, mu_te, var_te)
        all_metrics.append(m)

        if (rep + 1) % 10 == 0:
            logger.info(f"    Split {rep+1}/{n_repeats}: RMSE={m['rmse']:.3f}, ρ={m['spearman_rho']:.3f}")

    # Aggregate
    keys = ["rmse", "mae", "nll", "r2", "spearman_rho", "pearson_r", "coverage_95", "ci_width"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "min": float(np.min(vals)), "max": float(np.max(vals)),
                          "median": float(np.median(vals)), "n_valid": len(vals)}
        else:
            summary[k] = {"mean": float('nan'), "std": float('nan'), "n_valid": 0}

    logger.info(f"  Repeated split ({n_repeats}×): "
                f"RMSE={summary['rmse']['mean']:.3f}±{summary['rmse']['std']:.3f}, "
                f"ρ={summary['spearman_rho']['mean']:.3f}±{summary['spearman_rho']['std']:.3f}, "
                f"R²={summary['r2']['mean']:.3f}±{summary['r2']['std']:.3f}")

    return {"summary": summary, "all_metrics": all_metrics,
            "n_repeats": n_repeats, "test_frac": test_frac}


# ═══════════════════════════════════════════════════════════════
# Protocol 3: Pocket-Level Bootstrap
# ═══════════════════════════════════════════════════════════════
def run_bootstrap(X, y, n_bootstrap=1000, n_epochs=100, lr=0.1,
                  use_ard=False, seed=42):
    """
    Bootstrap: resample N pockets WITH replacement, train on resampled,
    evaluate LOOCV metrics on the resampled set.
    Uses analytic LOOCV for each bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    all_metrics = []

    for b in range(n_bootstrap):
        # Resample with replacement
        idx = rng.integers(0, N, size=N)
        X_b, y_b = X[idx], y[idx]

        # Add tiny jitter to avoid duplicate rows (GP needs unique inputs)
        X_b = X_b + rng.standard_normal(X_b.shape).astype(np.float32) * 1e-4

        try:
            model, lik = train_exact_gp(X_b, y_b, n_epochs=n_epochs, lr=lr,
                                         use_ard=use_ard)
            X_t = torch.tensor(X_b, dtype=torch.float32)
            y_t = torch.tensor(y_b, dtype=torch.float32)

            with torch.no_grad():
                K = model.covar_module(X_t).evaluate()
                noise = lik.noise.item()
                K_full = K + noise * torch.eye(N)
                K_inv = torch.linalg.inv(K_full)
                alpha = K_inv @ y_t
                loo_mu = y_t - alpha / K_inv.diag()
                loo_var = torch.clamp(1.0 / K_inv.diag(), min=1e-10)

            m = compute_metrics(y_b, loo_mu.numpy(), loo_var.numpy())
            all_metrics.append(m)
        except Exception as e:
            if (b + 1) % 100 == 0:
                logger.warning(f"    Bootstrap {b+1} failed: {e}")
            continue

        if (b + 1) % 200 == 0:
            logger.info(f"    Bootstrap {b+1}/{n_bootstrap}: "
                        f"RMSE={m['rmse']:.3f}, ρ={m['spearman_rho']:.3f}")

    keys = ["rmse", "mae", "nll", "r2", "spearman_rho", "pearson_r", "coverage_95", "ci_width"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if vals:
            vals_arr = np.array(vals)
            summary[k] = {
                "mean": float(np.mean(vals_arr)),
                "std": float(np.std(vals_arr)),
                "ci_2.5": float(np.percentile(vals_arr, 2.5)),
                "ci_97.5": float(np.percentile(vals_arr, 97.5)),
                "median": float(np.median(vals_arr)),
                "n_valid": len(vals),
            }
        else:
            summary[k] = {"mean": float('nan'), "std": float('nan'), "n_valid": 0}

    logger.info(f"  Bootstrap ({len(all_metrics)}/{n_bootstrap} successful): "
                f"RMSE={summary['rmse']['mean']:.3f} [{summary['rmse']['ci_2.5']:.3f}, {summary['rmse']['ci_97.5']:.3f}], "
                f"ρ={summary['spearman_rho']['mean']:.3f} [{summary['spearman_rho']['ci_2.5']:.3f}, {summary['spearman_rho']['ci_97.5']:.3f}]")

    return {"summary": summary, "n_bootstrap": n_bootstrap,
            "n_successful": len(all_metrics)}


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════
def generate_figures(loocv_res, split_res, bootstrap_res, y, names, out_dir):
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Figure 1: LOOCV pred vs true ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Robust GP Evaluation (Exact GP, ECFP4-128)", fontsize=14, fontweight="bold")

    # (0) LOOCV scatter
    ax = axes[0]
    mu_loo = np.array(loocv_res["mu"])
    var_loo = np.array(loocv_res["var"])
    sigma_loo = np.sqrt(var_loo)

    ax.errorbar(y, mu_loo, yerr=1.96 * sigma_loo, fmt="o", markersize=7,
                capsize=3, color="#3498DB", ecolor="#BDC3C7", alpha=0.8, zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm[:15], (y[i], mu_loo[i]), fontsize=4.5, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")

    lims = [min(y.min(), mu_loo.min()) - 0.5, max(y.max(), mu_loo.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True pKd"); ax.set_ylabel("LOOCV Predicted μ")
    m = loocv_res["metrics"]
    ax.set_title(f"Analytic LOOCV (N={m['n']})\n"
                 f"RMSE={m['rmse']:.2f}, ρ={m['spearman_rho']:.2f}, R²={m['r2']:.2f}")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (1) Repeated split distributions
    ax = axes[1]
    rs = split_res["summary"]
    metric_keys = ["rmse", "r2", "spearman_rho", "coverage_95"]
    metric_labels = ["RMSE", "R²", "Spearman ρ", "CI-95% Cov"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#E67E22"]

    for i, (k, label, c) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [m[k] for m in split_res["all_metrics"] if not np.isnan(m[k])]
        parts = ax.violinplot([vals], positions=[i], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('gray')
        ax.annotate(f"{np.mean(vals):.2f}±{np.std(vals):.2f}",
                    (i, np.mean(vals)), fontsize=7, ha="center",
                    xytext=(0, 12), textcoords="offset points")

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title(f"50× Repeated 70/30 Split\n(violin = distribution across splits)")
    ax.grid(alpha=0.3, axis="y")

    # (2) Bootstrap CI
    ax = axes[2]
    bs = bootstrap_res["summary"]
    x_pos = np.arange(len(metric_keys))
    means = [bs[k]["mean"] for k in metric_keys]
    ci_lo = [bs[k].get("ci_2.5", bs[k]["mean"] - 2*bs[k]["std"]) for k in metric_keys]
    ci_hi = [bs[k].get("ci_97.5", bs[k]["mean"] + 2*bs[k]["std"]) for k in metric_keys]
    errs = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]

    bars = ax.bar(x_pos, means, yerr=errs, capsize=6, color=colors, alpha=0.7,
                  edgecolor="gray")
    for i, (m_val, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.annotate(f"{m_val:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    (i, m_val), fontsize=6.5, ha="center",
                    xytext=(0, 12), textcoords="offset points")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels, fontsize=9)
    n_ok = bootstrap_res["n_successful"]
    ax.set_title(f"Bootstrap 95% CI ({n_ok} samples)\n(error bars = 2.5–97.5 percentile)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig_dir / "robust_evaluation.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'robust_evaluation.png'}")
    plt.close()

    # ── Figure 2: LOOCV residual + calibration ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("LOOCV Diagnostics", fontsize=14, fontweight="bold")

    residuals = mu_loo - y

    # (0) Residual vs true
    ax = axes[0]
    ax.scatter(y, residuals, s=50, c="#3498DB", alpha=0.7, edgecolors="gray")
    ax.axhline(0, color="black", ls="--", alpha=0.4)
    ax.set_xlabel("True pKd"); ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residuals vs True pKd"); ax.grid(alpha=0.3)

    # (1) Calibration: predicted σ vs |error|
    ax = axes[1]
    abs_err = np.abs(residuals)
    ax.scatter(sigma_loo, abs_err, s=50, c="#2ECC71", alpha=0.7, edgecolors="gray")
    lim = max(sigma_loo.max(), abs_err.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="Perfect cal.")
    ax.plot([0, lim], [0, 1.96 * lim], "r:", alpha=0.3, label="1.96σ bound")
    ax.set_xlabel("Predicted σ"); ax.set_ylabel("|Actual Error|")
    ax.set_title("Uncertainty Calibration"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (2) QQ-like: standardized residuals
    ax = axes[2]
    z_scores = residuals / sigma_loo
    z_sorted = np.sort(z_scores)
    n = len(z_sorted)
    expected = np.array([np.sqrt(2) * torch.erfinv(torch.tensor(2 * (i + 0.5) / n - 1)).item()
                         for i in range(n)])
    ax.scatter(expected, z_sorted, s=50, c="#E67E22", alpha=0.7, edgecolors="gray")
    lim = max(abs(expected).max(), abs(z_sorted).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.set_xlabel("Expected Normal Quantiles"); ax.set_ylabel("Observed Standardized Residuals")
    ax.set_title("Normal QQ Plot"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig_dir / "loocv_diagnostics.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'loocv_diagnostics.png'}")
    plt.close()

    # ── Figure 3: Comparison summary table as figure ─────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_title("Robust Evaluation Summary — Exact GP on ECFP4-128", fontsize=14, fontweight="bold", pad=20)

    headers = ["Protocol", "RMSE", "R²", "Spearman ρ", "NLL", "CI-95% Cov", "N"]
    lm = loocv_res["metrics"]
    table_data = [
        ["LOOCV (analytic)",
         f"{lm['rmse']:.3f}",
         f"{lm['r2']:.3f}",
         f"{lm['spearman_rho']:.3f}",
         f"{lm['nll']:.3f}",
         f"{lm['coverage_95']:.0%}",
         str(lm['n'])],
        [f"Repeated Split (×{split_res['n_repeats']})",
         f"{rs['rmse']['mean']:.3f}±{rs['rmse']['std']:.3f}",
         f"{rs['r2']['mean']:.3f}±{rs['r2']['std']:.3f}",
         f"{rs['spearman_rho']['mean']:.3f}±{rs['spearman_rho']['std']:.3f}",
         f"{rs['nll']['mean']:.3f}±{rs['nll']['std']:.3f}",
         f"{rs['coverage_95']['mean']:.0%}±{rs['coverage_95']['std']:.0%}",
         f"{int(len(y) * split_res['test_frac'])}"],
        [f"Bootstrap (×{bootstrap_res['n_successful']})",
         f"{bs['rmse']['mean']:.3f} [{bs['rmse'].get('ci_2.5', 0):.2f},{bs['rmse'].get('ci_97.5', 0):.2f}]",
         f"{bs['r2']['mean']:.3f} [{bs['r2'].get('ci_2.5', 0):.2f},{bs['r2'].get('ci_97.5', 0):.2f}]",
         f"{bs['spearman_rho']['mean']:.3f} [{bs['spearman_rho'].get('ci_2.5', 0):.2f},{bs['spearman_rho'].get('ci_97.5', 0):.2f}]",
         f"{bs['nll']['mean']:.3f}",
         f"{bs['coverage_95']['mean']:.0%}",
         str(len(y))],
    ]

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Color headers
    for j in range(len(headers)):
        table[0, j].set_facecolor("#3498DB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.savefig(fig_dir / "summary_table.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'summary_table.png'}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Robust GP Evaluation")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--affinity_pkl", required=True)
    parser.add_argument("--output", default="results/embedding_rdkit/robust_eval")
    parser.add_argument("--n_epochs", type=int, default=80,
                        help="Training epochs for Exact GP")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--use_ard", action="store_true",
                        help="Use ARD (per-dim lengthscales). Default: isotropic.")
    parser.add_argument("--n_repeats", type=int, default=50)
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, names = build_dataset(args.embeddings, args.affinity_pkl)
    logger.info(f"Dataset: {len(y)} labeled pockets, d={X.shape[1]}")
    logger.info(f"pKd range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")

    results = {"embedding": args.embeddings, "n_pockets": len(y), "d": X.shape[1],
               "use_ard": args.use_ard, "n_epochs": args.n_epochs, "lr": args.lr}

    # ── Protocol 1: LOOCV ────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Protocol 1: Analytic LOOCV (Exact GP)")
    logger.info("="*60)
    t0 = time.time()
    loocv_res = run_loocv(X, y, n_epochs=args.n_epochs, lr=args.lr,
                          use_ard=args.use_ard)
    results["loocv"] = loocv_res
    results["loocv"]["time_s"] = round(time.time() - t0, 1)

    # ── Protocol 2: Repeated Split ───────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"Protocol 2: {args.n_repeats}× Repeated 70/30 Split")
    logger.info("="*60)
    t0 = time.time()
    split_res = run_repeated_split(X, y, n_repeats=args.n_repeats,
                                    n_epochs=min(args.n_epochs, 60), lr=args.lr,
                                    use_ard=args.use_ard, seed=args.seed)
    results["repeated_split"] = {k: v for k, v in split_res.items() if k != "all_metrics"}
    results["repeated_split"]["time_s"] = round(time.time() - t0, 1)
    # Save full metrics separately (too verbose for main JSON)
    split_res_full = split_res

    # ── Protocol 3: Bootstrap ────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"Protocol 3: {args.n_bootstrap}× Pocket-Level Bootstrap")
    logger.info("="*60)
    t0 = time.time()
    bootstrap_res = run_bootstrap(X, y, n_bootstrap=args.n_bootstrap,
                                  n_epochs=min(args.n_epochs, 50), lr=args.lr,
                                  use_ard=args.use_ard, seed=args.seed)
    results["bootstrap"] = bootstrap_res
    results["bootstrap"]["time_s"] = round(time.time() - t0, 1)

    # ── Save results ─────────────────────────────────────────
    with open(out / "robust_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out / 'robust_eval_results.json'}")

    # ── Generate figures ─────────────────────────────────────
    logger.info("\nGenerating figures...")
    generate_figures(loocv_res, split_res_full, bootstrap_res, y, names, out)

    # ── Print final summary ──────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("ROBUST EVALUATION SUMMARY")
    logger.info("="*60)
    lm = loocv_res["metrics"]
    rs = split_res["summary"]
    bs = bootstrap_res["summary"]
    logger.info(f"  {'Protocol':<25s} {'RMSE':>10s} {'R²':>10s} {'ρ':>10s} {'NLL':>10s} {'CI95%':>8s}")
    logger.info(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    logger.info(f"  {'LOOCV':<25s} {lm['rmse']:10.3f} {lm['r2']:10.3f} {lm['spearman_rho']:10.3f} {lm['nll']:10.3f} {lm['coverage_95']:7.0%}")
    logger.info(f"  {'Repeated Split (mean)':<25s} {rs['rmse']['mean']:10.3f} {rs['r2']['mean']:10.3f} {rs['spearman_rho']['mean']:10.3f} {rs['nll']['mean']:10.3f} {rs['coverage_95']['mean']:7.0%}")
    logger.info(f"  {'Repeated Split (std)':<25s} {rs['rmse']['std']:10.3f} {rs['r2']['std']:10.3f} {rs['spearman_rho']['std']:10.3f} {rs['nll']['std']:10.3f} {rs['coverage_95']['std']:7.0%}")
    logger.info(f"  {'Bootstrap (mean)':<25s} {bs['rmse']['mean']:10.3f} {bs['r2']['mean']:10.3f} {bs['spearman_rho']['mean']:10.3f} {bs['nll']['mean']:10.3f} {bs['coverage_95']['mean']:7.0%}")
    logger.info(f"  {'Bootstrap [2.5,97.5]':<25s} [{bs['rmse'].get('ci_2.5',0):.2f},{bs['rmse'].get('ci_97.5',0):.2f}] "
                f"[{bs['r2'].get('ci_2.5',0):.2f},{bs['r2'].get('ci_97.5',0):.2f}] "
                f"[{bs['spearman_rho'].get('ci_2.5',0):.2f},{bs['spearman_rho'].get('ci_97.5',0):.2f}]")


if __name__ == "__main__":
    main()
