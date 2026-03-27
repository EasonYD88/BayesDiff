#!/usr/bin/env python
"""
scripts/17_train_gp_tier3.py
─────────────────────────────
Train GP on Tier 3 dataset (N≈932) with best BO config.
Evaluate with LOOCV + repeated random splits.
Generate training curves and performance visualizations.

Runs on GPU for speed but works on CPU too.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import gpytorch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "results" / "tier3_gp"
OUT_DIR = PROJECT / "results" / "tier3_gp"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── GP Model ─────────────────────────────────────────────────────────────
class FlexibleGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type="rq", ard_dims=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == "rq":
            base = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)
        elif kernel_type == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel_type == "matern25":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        elif kernel_type == "matern15":
            base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        else:
            base = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)

        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gp(X_train, y_train, kernel_type="rq", n_epochs=100, lr=0.1, noise_lb=0.001):
    """Train GP and return model, likelihood, training losses."""
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
    """Predict with GP. Returns mean, std."""
    model.eval()
    likelihood.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()


def evaluate(y_true, y_pred):
    """Compute RMSE, Spearman rho, R², MAE."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    rho, p_val = stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (0, 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"rmse": rmse, "mae": mae, "rho": rho, "p_val": p_val, "r2": r2}


def loocv(X, y, kernel_type="rq", n_epochs=100, lr=0.1, noise_lb=0.001):
    """Leave-one-out cross-validation (analytic for GP)."""
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

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

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = -mll(output, y_t)
        loss.backward()
        optimizer.step()

    # Analytic LOOCV
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        K = model.covar_module(X_t).evaluate()
        noise = likelihood.noise
        K_noise = K + noise * torch.eye(len(X_t), device=DEVICE)
        K_inv = torch.linalg.inv(K_noise)
        alpha = K_inv @ y_t
        loo_mean = y_t - alpha / K_inv.diagonal()
        loo_var = 1.0 / K_inv.diagonal()

    loo_pred = loo_mean.cpu().numpy()
    loo_std = torch.sqrt(loo_var).cpu().numpy()

    return loo_pred, loo_std


def repeated_splits(X, y, n_splits=50, test_frac=0.2, kernel_type="rq",
                    n_epochs=100, lr=0.1, noise_lb=0.001, seed=42):
    """Repeated random train/test splits."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_test = max(int(n * test_frac), 1)

    all_metrics = []
    for i in range(n_splits):
        idx = rng.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model, lik, _ = train_gp(X_tr, y_tr, kernel_type, n_epochs, lr, noise_lb)
        y_pred, _ = predict_gp(model, lik, X_te)

        metrics = evaluate(y_te, y_pred)
        all_metrics.append(metrics)

        if (i + 1) % 10 == 0:
            logger.info(f"  Split {i+1}/{n_splits}: RMSE={metrics['rmse']:.3f}, ρ={metrics['rho']:.3f}")

    return all_metrics


def kfold_cv(X, y, k=5, kernel_type="rq", n_epochs=100, lr=0.1, noise_lb=0.001, seed=42):
    """K-fold cross-validation."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    all_preds = np.zeros(n)
    all_stds = np.zeros(n)
    fold_metrics = []

    for fold_i in range(k):
        test_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_i])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model, lik, _ = train_gp(X_tr, y_tr, kernel_type, n_epochs, lr, noise_lb)
        y_pred, y_std = predict_gp(model, lik, X_te)

        all_preds[test_idx] = y_pred
        all_stds[test_idx] = y_std

        metrics = evaluate(y_te, y_pred)
        fold_metrics.append(metrics)
        logger.info(f"  Fold {fold_i+1}/{k}: RMSE={metrics['rmse']:.3f}, ρ={metrics['rho']:.3f}")

    return all_preds, all_stds, fold_metrics


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load data ──
    fp_name = "FCFP4_2048"  # Best from BO
    X_raw = np.load(DATA_DIR / f"X_{fp_name}.npy")
    y = np.load(DATA_DIR / "y_pkd.npy")
    with open(DATA_DIR / "families.json") as f:
        families = json.load(f)

    logger.info(f"Data: {X_raw.shape[0]} pockets, {X_raw.shape[1]}-dim {fp_name}")
    logger.info(f"pKd: {y.mean():.2f} ± {y.std():.2f} [{y.min():.2f}, {y.max():.2f}]")

    # ── PCA reduction ──
    pca_dims = [10, 20, 50, 100, None]
    best_pca = None
    best_rmse = float("inf")

    # Quick 5-fold CV to select PCA dims
    logger.info("\n=== PCA dimension selection (5-fold CV) ===")
    for pca_d in pca_dims:
        if pca_d is not None and pca_d >= X_raw.shape[1]:
            continue
        if pca_d is not None:
            pca = PCA(n_components=pca_d, random_state=42)
            X = pca.fit_transform(X_raw)
            var_explained = pca.explained_variance_ratio_.sum()
        else:
            X = X_raw.copy()
            var_explained = 1.0
            pca_d_label = X.shape[1]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        _, _, fold_m = kfold_cv(X, y, k=5, n_epochs=80, lr=0.15)
        mean_rmse = np.mean([m["rmse"] for m in fold_m])
        mean_rho = np.mean([m["rho"] for m in fold_m])

        label = f"PCA-{pca_d}" if pca_d is not None else f"Full-{pca_d_label}"
        logger.info(f"  {label}: RMSE={mean_rmse:.3f}, ρ={mean_rho:.3f}, var={var_explained:.2%}")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_pca = pca_d

    logger.info(f"\nBest PCA: {best_pca} (RMSE={best_rmse:.3f})")

    # ── Prepare final X ──
    if best_pca is not None:
        pca = PCA(n_components=best_pca, random_state=42)
        X = pca.fit_transform(X_raw)
    else:
        X = X_raw.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── 1. Training curve (full dataset) ──
    logger.info("\n=== Training curve ===")
    model, lik, losses = train_gp(X, y, kernel_type="rq", n_epochs=200, lr=0.1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, "b-", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative MLL")
    ax.set_title(f"GP Training Curve (N={len(y)}, {fp_name}, PCA→{best_pca or X.shape[1]})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=150)
    plt.close()
    logger.info(f"  Final loss: {losses[-1]:.3f} (from {losses[0]:.3f})")

    # ── 2. LOOCV ──
    logger.info("\n=== Leave-One-Out Cross-Validation ===")
    loo_pred, loo_std = loocv(X, y, kernel_type="rq", n_epochs=150, lr=0.1)
    loo_metrics = evaluate(y, loo_pred)
    logger.info(f"  LOOCV RMSE={loo_metrics['rmse']:.3f}, ρ={loo_metrics['rho']:.3f}, "
                f"R²={loo_metrics['r2']:.3f}, MAE={loo_metrics['mae']:.3f}")

    # LOOCV scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, loo_pred, c=loo_std, cmap="viridis", alpha=0.5, s=15, edgecolors="none")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Predictive σ")
    lims = [min(y.min(), loo_pred.min()) - 0.5, max(y.max(), loo_pred.max()) + 0.5]
    ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True pKd")
    ax.set_ylabel("LOOCV Predicted pKd")
    ax.set_title(f"LOOCV (N={len(y)}): RMSE={loo_metrics['rmse']:.2f}, "
                 f"ρ={loo_metrics['rho']:.3f}, R²={loo_metrics['r2']:.3f}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loocv_scatter.png", dpi=150)
    plt.close()

    # ── 3. 5-Fold CV ──
    logger.info("\n=== 5-Fold Cross-Validation ===")
    cv_pred, cv_std, cv_fold_metrics = kfold_cv(X, y, k=5, n_epochs=150, lr=0.1)
    cv_overall = evaluate(y, cv_pred)
    logger.info(f"  5-Fold overall: RMSE={cv_overall['rmse']:.3f}, ρ={cv_overall['rho']:.3f}, "
                f"R²={cv_overall['r2']:.3f}")

    # 5-Fold scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, cv_pred, c=cv_std, cmap="viridis", alpha=0.5, s=15, edgecolors="none")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Predictive σ")
    ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True pKd")
    ax.set_ylabel("5-Fold CV Predicted pKd")
    ax.set_title(f"5-Fold CV (N={len(y)}): RMSE={cv_overall['rmse']:.2f}, "
                 f"ρ={cv_overall['rho']:.3f}, R²={cv_overall['r2']:.3f}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv5_scatter.png", dpi=150)
    plt.close()

    # ── 4. 50× Repeated splits ──
    logger.info("\n=== 50× Repeated Random Splits (80/20) ===")
    split_metrics = repeated_splits(X, y, n_splits=50, test_frac=0.2,
                                    n_epochs=150, lr=0.1)

    rmse_vals = [m["rmse"] for m in split_metrics]
    rho_vals = [m["rho"] for m in split_metrics]
    r2_vals = [m["r2"] for m in split_metrics]
    logger.info(f"  RMSE: {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}")
    logger.info(f"  ρ:    {np.mean(rho_vals):.3f} ± {np.std(rho_vals):.3f}")
    logger.info(f"  R²:   {np.mean(r2_vals):.3f} ± {np.std(r2_vals):.3f}")

    # Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].hist(rmse_vals, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(np.mean(rmse_vals), color="red", linestyle="--", label=f"mean={np.mean(rmse_vals):.2f}")
    axes[0].set_xlabel("RMSE")
    axes[0].set_title("RMSE Distribution")
    axes[0].legend()

    axes[1].hist(rho_vals, bins=15, color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(np.mean(rho_vals), color="red", linestyle="--", label=f"mean={np.mean(rho_vals):.3f}")
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Spearman ρ")
    axes[1].set_title("Spearman ρ Distribution")
    axes[1].legend()

    axes[2].hist(r2_vals, bins=15, color="seagreen", edgecolor="white", alpha=0.8)
    axes[2].axvline(np.mean(r2_vals), color="red", linestyle="--", label=f"mean={np.mean(r2_vals):.3f}")
    axes[2].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("R²")
    axes[2].set_title("R² Distribution")
    axes[2].legend()

    fig.suptitle(f"50× Repeated Splits (N={len(y)}, 80/20)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "repeated_splits_dist.png", dpi=150)
    plt.close()

    # ── 5. Residual analysis ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    residuals = y - loo_pred
    axes[0].scatter(loo_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Predicted pKd")
    axes[0].set_ylabel("Residual (true - pred)")
    axes[0].set_title("Residual vs Predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})")

    # Calibration: fraction of true values within CI
    from scipy.stats import norm
    ci_levels = np.linspace(0.05, 0.95, 19)
    observed_coverage = []
    for ci in ci_levels:
        z = norm.ppf(0.5 + ci / 2)
        lower = loo_pred - z * loo_std
        upper = loo_pred + z * loo_std
        in_ci = ((y >= lower) & (y <= upper)).mean()
        observed_coverage.append(in_ci)

    axes[2].plot(ci_levels, observed_coverage, "bo-", markersize=4, label="Observed")
    axes[2].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Ideal")
    axes[2].set_xlabel("Expected Coverage")
    axes[2].set_ylabel("Observed Coverage")
    axes[2].set_title("Uncertainty Calibration (LOOCV)")
    axes[2].legend()
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Diagnostic Plots (N={len(y)})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnostics.png", dpi=150)
    plt.close()

    # ── 6. Comparison: N=24 vs N=932 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["N=24\n(old)", f"N={len(y)}\n(Tier 3)"]
    old_rmse = 2.07  # from BO results
    old_rho = -0.42
    new_rmse = loo_metrics["rmse"]
    new_rho = loo_metrics["rho"]

    x_pos = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, [old_rmse, new_rmse], width, label="RMSE", color="steelblue")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, [old_rho, new_rho], width, label="Spearman ρ", color="coral")

    ax.set_ylabel("RMSE")
    ax2.set_ylabel("Spearman ρ")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("GP Performance: Small vs Large Dataset")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    for bar, val in zip(bars1, [old_rmse, new_rmse]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontsize=10)
    for bar, val in zip(bars2, [old_rho, new_rho]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02 * (1 if val > 0 else -1),
                 f"{val:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_n24_vs_tier3.png", dpi=150)
    plt.close()

    # ── Save all results ──
    results = {
        "dataset": {
            "n_pockets": len(y),
            "fingerprint": fp_name,
            "pca_dims": best_pca,
            "final_dims": X.shape[1],
            "pkd_mean": float(y.mean()),
            "pkd_std": float(y.std()),
            "pkd_range": [float(y.min()), float(y.max())],
        },
        "loocv": {k: float(v) for k, v in loo_metrics.items()},
        "cv5_overall": {k: float(v) for k, v in cv_overall.items()},
        "cv5_per_fold": [{k: float(v) for k, v in m.items()} for m in cv_fold_metrics],
        "repeated_splits_50x": {
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "rho_mean": float(np.mean(rho_vals)),
            "rho_std": float(np.std(rho_vals)),
            "r2_mean": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)),
        },
        "training": {
            "final_loss": float(losses[-1]),
            "initial_loss": float(losses[0]),
            "n_epochs": len(losses),
        },
    }

    with open(OUT_DIR / "tier3_gp_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL RESULTS SAVED to {OUT_DIR}")
    logger.info(f"Figures: {FIG_DIR}")
    logger.info(f"{'='*60}")
    logger.info(f"\n=== FINAL SUMMARY ===")
    logger.info(f"Dataset: {len(y)} pockets, {fp_name}, PCA→{best_pca or X.shape[1]}")
    logger.info(f"LOOCV:    RMSE={loo_metrics['rmse']:.3f}, ρ={loo_metrics['rho']:.3f}, R²={loo_metrics['r2']:.3f}")
    logger.info(f"5-Fold:   RMSE={cv_overall['rmse']:.3f}, ρ={cv_overall['rho']:.3f}, R²={cv_overall['r2']:.3f}")
    logger.info(f"50×Split: RMSE={np.mean(rmse_vals):.3f}±{np.std(rmse_vals):.3f}, "
                f"ρ={np.mean(rho_vals):.3f}±{np.std(rho_vals):.3f}")


if __name__ == "__main__":
    main()
