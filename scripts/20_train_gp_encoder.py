#!/usr/bin/env python3
"""Train GP on TargetDiff encoder embeddings and compare with FCFP4 baseline.

Loads X_encoder_128.npy (128-dim encoder embeddings) and trains GP with:
- LOOCV
- 5-fold CV
- 50× repeated random splits (70/30)
- 30× train/val/test (60/20/20)

Generates comparison figures against FCFP4-2048 baseline.
"""

import sys
import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
import gpytorch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = REPO / "results" / "tier3_gp"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── GP Model ──
class FlexibleGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type="rq"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == "rq":
            base = gpytorch.kernels.RQKernel()
        elif kernel_type == "rbf":
            base = gpytorch.kernels.RBFKernel()
        elif kernel_type == "matern25":
            base = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            base = gpytorch.kernels.RQKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gp(X_train, y_train, kernel_type="rq", n_epochs=200, lr=0.1, noise_lb=0.001):
    """Train GP and return model, likelihood, losses."""
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
    )
    model = FlexibleGP(X_t, y_t, likelihood, kernel_type=kernel_type)
    model = model.to(DEVICE)
    likelihood = likelihood.to(DEVICE)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = -mll(output, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, likelihood, losses


def predict_gp(model, likelihood, X_test):
    """Predict with GP, return mean and std."""
    model.eval()
    likelihood.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    rho, p = stats.spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"rmse": rmse, "mae": mae, "rho": rho, "p_value": p, "r2": r2}


def loocv(X, y, kernel_type="rq", n_epochs=200, lr=0.1):
    """Leave-one-out cross-validation."""
    n = len(y)
    preds = np.zeros(n)
    stds = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model, lik, _ = train_gp(X[mask], y[mask], kernel_type, n_epochs, lr)
        pred_mean, pred_std = predict_gp(model, lik, X[~mask])
        preds[i] = pred_mean[0]
        stds[i] = pred_std[0]

        if (i + 1) % 100 == 0:
            logger.info(f"  LOOCV: {i+1}/{n}")

    return preds, stds


def main():
    # ── Load data ──
    X_enc = np.load(DATA_DIR / "X_encoder_128.npy")
    y_enc = np.load(DATA_DIR / "y_pkd_encoder.npy")
    with open(DATA_DIR / "families_encoder.json") as f:
        families_enc = json.load(f)

    # Also load FCFP4 for comparison
    X_fcfp = np.load(DATA_DIR / "X_FCFP4_2048.npy")
    y_fcfp = np.load(DATA_DIR / "y_pkd.npy")

    logger.info(f"Encoder: {X_enc.shape[0]} pockets, {X_enc.shape[1]}-dim")
    logger.info(f"FCFP4:   {X_fcfp.shape[0]} pockets, {X_fcfp.shape[1]}-dim")
    logger.info(f"Encoder pKd: {y_enc.mean():.2f} ± {y_enc.std():.2f} "
                f"[{y_enc.min():.2f}, {y_enc.max():.2f}]")

    # Standardize
    scaler_enc = StandardScaler()
    X_enc_s = scaler_enc.fit_transform(X_enc)

    scaler_fcfp = StandardScaler()
    X_fcfp_s = scaler_fcfp.fit_transform(X_fcfp)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # 1. Training curve comparison
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== Training curves ===")
    _, _, losses_enc = train_gp(X_enc_s, y_enc, "rq", 200, 0.1)
    _, _, losses_fcfp = train_gp(X_fcfp_s, y_fcfp, "rq", 200, 0.1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses_enc, "b-", alpha=0.8, label=f"Encoder-128 (final={losses_enc[-1]:.2f})")
    ax.plot(losses_fcfp, "r--", alpha=0.8, label=f"FCFP4-2048 (final={losses_fcfp[-1]:.2f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative MLL")
    ax.set_title("GP Training Curves: Encoder vs FCFP4")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoder_vs_fcfp_training.png", dpi=150)
    plt.close()
    logger.info(f"  Encoder final loss: {losses_enc[-1]:.3f}")
    logger.info(f"  FCFP4 final loss:   {losses_fcfp[-1]:.3f}")

    # ══════════════════════════════════════════════════════════════
    # 2. LOOCV
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== LOOCV (Encoder) ===")
    loocv_preds, loocv_stds = loocv(X_enc_s, y_enc, "rq", 200, 0.1)
    loocv_metrics = compute_metrics(y_enc, loocv_preds)
    results["loocv_encoder"] = loocv_metrics
    logger.info(f"  RMSE={loocv_metrics['rmse']:.3f}, ρ={loocv_metrics['rho']:.3f} "
                f"(p={loocv_metrics['p_value']:.4f}), R²={loocv_metrics['r2']:.3f}")

    # Load FCFP4 LOOCV from previous results for comparison
    try:
        with open(DATA_DIR / "tier3_gp_results.json") as f:
            prev_results = json.load(f)
        fcfp_loocv = prev_results.get("loocv", {})
    except FileNotFoundError:
        fcfp_loocv = {}

    # ══════════════════════════════════════════════════════════════
    # 3. 5-fold CV
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== 5-Fold CV (Encoder) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_fold_preds = np.zeros(len(y_enc))
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_enc_s)):
        model, lik, _ = train_gp(X_enc_s[tr_idx], y_enc[tr_idx], "rq", 200, 0.1)
        pred, _ = predict_gp(model, lik, X_enc_s[te_idx])
        all_fold_preds[te_idx] = pred
        fm = compute_metrics(y_enc[te_idx], pred)
        fold_metrics.append(fm)
        logger.info(f"  Fold {fold+1}: RMSE={fm['rmse']:.3f}, ρ={fm['rho']:.3f}")

    overall_5fold = compute_metrics(y_enc, all_fold_preds)
    results["5fold_encoder"] = {
        "per_fold": fold_metrics,
        "overall": overall_5fold,
    }
    logger.info(f"  Overall: RMSE={overall_5fold['rmse']:.3f}, "
                f"ρ={overall_5fold['rho']:.3f}, R²={overall_5fold['r2']:.3f}")

    # ══════════════════════════════════════════════════════════════
    # 4. 50× repeated random splits (70/30)
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== 50× Repeated Splits (Encoder) ===")
    n_repeats = 50
    split_results = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(rep)
        idx = rng.permutation(len(y_enc))
        n_train = int(0.7 * len(y_enc))
        tr_idx, te_idx = idx[:n_train], idx[n_train:]
        model, lik, _ = train_gp(X_enc_s[tr_idx], y_enc[tr_idx], "rq", 200, 0.1)
        pred, _ = predict_gp(model, lik, X_enc_s[te_idx])
        sm = compute_metrics(y_enc[te_idx], pred)
        split_results.append(sm)

    rmses = [s["rmse"] for s in split_results]
    rhos = [s["rho"] for s in split_results]
    r2s = [s["r2"] for s in split_results]
    results["50x_splits_encoder"] = {
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "rho_mean": np.mean(rhos), "rho_std": np.std(rhos),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
    }
    logger.info(f"  RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    logger.info(f"  ρ:    {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    logger.info(f"  R²:   {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    # ══════════════════════════════════════════════════════════════
    # 5. 30× Train/Val/Test (60/20/20)
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== 30× Train/Val/Test (Encoder) ===")
    tvt_results = {"train": [], "val": [], "test": []}
    for rep in range(30):
        rng = np.random.RandomState(rep + 1000)
        idx = rng.permutation(len(y_enc))
        n_tr = int(0.6 * len(y_enc))
        n_val = int(0.2 * len(y_enc))
        tr_idx = idx[:n_tr]
        val_idx = idx[n_tr : n_tr + n_val]
        te_idx = idx[n_tr + n_val :]

        model, lik, _ = train_gp(X_enc_s[tr_idx], y_enc[tr_idx], "rq", 200, 0.1)

        for name, idxs in [("train", tr_idx), ("val", val_idx), ("test", te_idx)]:
            pred, _ = predict_gp(model, lik, X_enc_s[idxs])
            m = compute_metrics(y_enc[idxs], pred)
            tvt_results[name].append(m)

    for name in ["train", "val", "test"]:
        rmses = [m["rmse"] for m in tvt_results[name]]
        rhos = [m["rho"] for m in tvt_results[name]]
        r2s = [m["r2"] for m in tvt_results[name]]
        results[f"tvt_{name}_encoder"] = {
            "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            "rho_mean": np.mean(rhos), "rho_std": np.std(rhos),
            "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        }
        logger.info(f"  {name:5s}: RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f}, "
                     f"ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}, "
                     f"R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}")

    # ══════════════════════════════════════════════════════════════
    # 6. Comparison figures
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== Generating comparison figures ===")

    # 6a. LOOCV scatter comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, preds, y_true, metrics in [
        (axes[0], "Encoder-128", loocv_preds, y_enc, loocv_metrics),
    ]:
        ax.scatter(y_true, preds, alpha=0.3, s=15, c="steelblue")
        lims = [min(y_true.min(), preds.min()) - 0.5,
                max(y_true.max(), preds.max()) + 0.5]
        ax.plot(lims, lims, "r--", alpha=0.5, lw=1.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True pKd")
        ax.set_ylabel("Predicted pKd (LOOCV)")
        ax.set_title(f"{title}\nRMSE={metrics['rmse']:.2f}, "
                     f"ρ={metrics['rho']:.3f}, R²={metrics['r2']:.3f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Plot FCFP4 LOOCV if available
    if fcfp_loocv:
        axes[1].text(0.5, 0.5,
                     f"FCFP4-2048 LOOCV\n"
                     f"RMSE={fcfp_loocv.get('rmse', 'N/A')}\n"
                     f"ρ={fcfp_loocv.get('rho', 'N/A')}\n"
                     f"R²={fcfp_loocv.get('r2', 'N/A')}",
                     transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=14)
        axes[1].set_title("FCFP4-2048 (Previous)")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoder_loocv_scatter.png", dpi=150)
    plt.close()

    # 6b. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ["RMSE", "Spearman ρ", "R²"]
    enc_vals = [loocv_metrics["rmse"], loocv_metrics["rho"], loocv_metrics["r2"]]
    fcfp_vals = [
        fcfp_loocv.get("rmse", 2.068),
        fcfp_loocv.get("rho", 0.111),
        fcfp_loocv.get("r2", 0.013),
    ]

    for ax, name, ev, fv in zip(axes, metric_names, enc_vals, fcfp_vals):
        bars = ax.bar(["Encoder-128", "FCFP4-2048"], [ev, fv],
                      color=["steelblue", "salmon"], alpha=0.8, edgecolor="black")
        ax.set_title(name, fontsize=13)
        ax.set_ylabel(name)
        for bar, val in zip(bars, [ev, fv]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("LOOCV: Encoder vs FCFP4", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoder_vs_fcfp_metrics.png", dpi=150)
    plt.close()

    # 6c. Train/Val/Test boxplots comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric_key, ylabel in [
        (axes[0], "rmse", "RMSE"),
        (axes[1], "rho", "Spearman ρ"),
        (axes[2], "r2", "R²"),
    ]:
        enc_data = {
            "Train": [m[metric_key] for m in tvt_results["train"]],
            "Val": [m[metric_key] for m in tvt_results["val"]],
            "Test": [m[metric_key] for m in tvt_results["test"]],
        }
        bp = ax.boxplot(
            [enc_data["Train"], enc_data["Val"], enc_data["Test"]],
            labels=["Train", "Val", "Test"],
            patch_artist=True,
        )
        colors = ["#4CAF50", "#FF9800", "#F44336"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Encoder: {ylabel} (30× splits)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoder_tvt_boxplots.png", dpi=150)
    plt.close()

    # 6d. Embedding space analysis (PCA/t-SNE)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_enc_s)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_enc, cmap="viridis",
                    s=15, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="pKd")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Encoder Embedding PCA (colored by pKd)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoder_pca.png", dpi=150)
    plt.close()

    # 6e. Variance explained by PCA
    pca_full = PCA().fit(X_enc_s)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, "b-o", markersize=3)
    ax.axhline(0.95, color="r", linestyle="--", alpha=0.5, label="95% variance")
    ax.axhline(0.99, color="orange", linestyle="--", alpha=0.5, label="99% variance")
    n_95 = np.searchsorted(cum_var, 0.95) + 1
    n_99 = np.searchsorted(cum_var, 0.99) + 1
    ax.axvline(n_95, color="r", linestyle=":", alpha=0.3)
    ax.axvline(n_99, color="orange", linestyle=":", alpha=0.3)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title(f"Encoder PCA: 95% at {n_95} dims, 99% at {n_99} dims")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoder_pca_variance.png", dpi=150)
    plt.close()

    # ── Save results ──
    results_path = DATA_DIR / "encoder_gp_results.json"

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("Figures saved to:")
    for fig_name in sorted(FIG_DIR.glob("encoder_*.png")):
        logger.info(f"  {fig_name.name}")

    # ── Summary comparison ──
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Encoder-128 vs FCFP4-2048 (LOOCV)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<12} {'Encoder-128':>12} {'FCFP4-2048':>12} {'Δ':>10}")
    logger.info(f"{'-'*48}")
    for name, e, f_val in [
        ("RMSE", loocv_metrics["rmse"], fcfp_loocv.get("rmse", 2.068)),
        ("ρ", loocv_metrics["rho"], fcfp_loocv.get("rho", 0.111)),
        ("R²", loocv_metrics["r2"], fcfp_loocv.get("r2", 0.013)),
    ]:
        delta = e - f_val
        logger.info(f"{name:<12} {e:>12.3f} {f_val:>12.3f} {delta:>+10.3f}")
    logger.info(f"{'='*60}")

    go_no_go = loocv_metrics["rho"] > 0.3
    logger.info(f"\nGo/No-Go: LOOCV ρ = {loocv_metrics['rho']:.3f} "
                f"{'> 0.3 → GO ✓' if go_no_go else '≤ 0.3 → EXPLORE ALTERNATIVES'}")


if __name__ == "__main__":
    main()
