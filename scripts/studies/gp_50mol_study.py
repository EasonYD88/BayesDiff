#!/usr/bin/env python3
"""28_50mol_gp_study.py — Comprehensive GP study on 50mol dataset.

Phases:
  1. Data Analysis & Visualization
  2. Grid Search (kernel × PCA × ARD) + Bayesian Optimization (Optuna)
  3. Training Curves (top configs, train/val/test)
  4. Ablation Study
  5. All Figures + Summary

Usage:
    python scripts/studies/gp_50mol_study.py [--n_bo_trials 200]
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "results" / "50mol_gp"
TIER3_DIR = REPO / "results" / "tier3_gp"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                      "axes.titlesize": 12, "axes.labelsize": 10})

# ── Kernel names ──
KERNELS = ["rbf", "matern15", "matern25", "rq"]
KERNEL_LABELS = {"rbf": "RBF", "matern15": "Matérn-3/2", "matern25": "Matérn-5/2", "rq": "RQ"}
PCA_DIMS_LIST = [0, 10, 20, 32, 64]  # 0 = no PCA
ARD_OPTIONS = [False, True]


# ═══════════════════════════════════════════════════════════════
# Section 1: GP Model
# ═══════════════════════════════════════════════════════════════

class FlexibleGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type="rq", ard_dims=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel_type == "matern15":
            base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        elif kernel_type == "matern25":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        elif kernel_type == "rq":
            base = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)
        else:
            base = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


# ═══════════════════════════════════════════════════════════════
# Section 2: Training & Evaluation
# ═══════════════════════════════════════════════════════════════

def prepare_data(X, y, pca_dims=None):
    """Standardize + optional PCA."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca_model = None
    if pca_dims and pca_dims > 0:
        actual = min(pca_dims, X_s.shape[0] - 1, X_s.shape[1])
        if actual < 2:
            return X_s, scaler, None, 1.0
        pca_model = PCA(n_components=actual)
        X_s = pca_model.fit_transform(X_s)
    var_exp = float(pca_model.explained_variance_ratio_.sum()) if pca_model else 1.0
    return X_s, scaler, pca_model, var_exp


def train_gp(X, y, kernel_type="rq", ard=False, n_epochs=200, lr=0.1, noise_lb=0.001):
    """Train ExactGP, return model + likelihood."""
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)).to(DEVICE)
    ard_dims = X.shape[1] if ard else None
    model = FlexibleGP(X_t, y_t, lik, kernel_type, ard_dims).to(DEVICE)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        opt.step()
    return model, lik


def predict_gp(model, lik, X_test):
    model.eval(); lik.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(X_t))
    return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()


def compute_metrics(y_true, y_pred, y_std=None):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, p = stats.spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    m = {"rmse": rmse, "rho": float(rho), "p_value": float(p), "r2": r2}
    if y_std is not None:
        residuals = y_true - y_pred
        nll = float(np.mean(0.5 * np.log(2 * np.pi * y_std ** 2 + 1e-10)
                            + 0.5 * residuals ** 2 / (y_std ** 2 + 1e-10)))
        coverage = float(np.mean(np.abs(residuals) < 1.96 * y_std))
        m["nll"] = nll
        m["ci95_coverage"] = coverage
        m["mean_ci_width"] = float(np.mean(2 * 1.96 * y_std))
    return m


def analytic_loocv(X, y, kernel_type="rq", ard=False, n_epochs=200, lr=0.1, noise_lb=0.001):
    """Train GP then compute analytic LOOCV. Returns metrics + predictions."""
    model, lik = train_gp(X, y, kernel_type, ard, n_epochs, lr, noise_lb)
    model.eval(); lik.eval()
    N = len(y)
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        K = model.covar_module(X_t).evaluate()
        K_noisy = K + lik.noise * torch.eye(N, device=DEVICE)
        try:
            K_inv = torch.linalg.inv(K_noisy)
        except Exception:
            K_inv = torch.linalg.inv(K_noisy + 1e-4 * torch.eye(N, device=DEVICE))
        alpha = K_inv @ y_t
        diag_inv = K_inv.diag()
        loo_pred = (y_t - alpha / diag_inv).cpu().numpy()
        loo_var = (1.0 / diag_inv).cpu().numpy()
    loo_std = np.sqrt(np.maximum(loo_var, 1e-10))
    metrics = compute_metrics(y, loo_pred, loo_std)
    return metrics, loo_pred, loo_std


def train_gp_with_curves(X_tr, y_tr, X_val, y_val, X_te, y_te,
                          kernel_type="rq", ard=False, n_epochs=300,
                          lr=0.1, noise_lb=0.001, eval_every=5):
    """Train GP recording per-epoch metrics on all splits."""
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    ard_dims = X_tr.shape[1] if ard else None
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)).to(DEVICE)
    model = FlexibleGP(X_tr_t, y_tr_t, lik, kernel_type, ard_dims).to(DEVICE)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    curves = {"epoch": [], "train_loss": [],
              "train_rmse": [], "train_rho": [], "train_r2": [],
              "val_rmse": [], "val_rho": [], "val_r2": [],
              "test_rmse": [], "test_rho": [], "test_r2": [],
              "noise": [], "outputscale": []}

    for epoch in range(n_epochs):
        model.train(); lik.train()
        opt.zero_grad()
        output = model(X_tr_t)
        loss = -mll(output, y_tr_t)
        loss.backward()
        opt.step()

        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            model.eval(); lik.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                tr_mu = lik(model(X_tr_t)).mean.cpu().numpy()
                val_mu = lik(model(X_val_t)).mean.cpu().numpy()
                te_mu = lik(model(X_te_t)).mean.cpu().numpy()
            tr_m = compute_metrics(y_tr, tr_mu)
            val_m = compute_metrics(y_val, val_mu)
            te_m = compute_metrics(y_te, te_mu)

            curves["epoch"].append(epoch)
            curves["train_loss"].append(loss.item())
            curves["train_rmse"].append(tr_m["rmse"])
            curves["train_rho"].append(tr_m["rho"])
            curves["train_r2"].append(tr_m["r2"])
            curves["val_rmse"].append(val_m["rmse"])
            curves["val_rho"].append(val_m["rho"])
            curves["val_r2"].append(val_m["r2"])
            curves["test_rmse"].append(te_m["rmse"])
            curves["test_rho"].append(te_m["rho"])
            curves["test_r2"].append(te_m["r2"])
            curves["noise"].append(float(lik.noise.item()))
            curves["outputscale"].append(float(model.covar_module.outputscale.item()))

    final_model = model
    final_lik = lik
    return final_model, final_lik, curves


def repeated_splits(X, y, kernel_type="rq", ard=False, n_repeats=10,
                    val_frac=0.2, test_frac=0.2, n_epochs=200, lr=0.1, noise_lb=0.001):
    """Repeated random train/val/test splits."""
    N = len(y)
    n_test = max(1, int(N * test_frac))
    n_val = max(1, int(N * val_frac))
    results = []
    for seed in range(n_repeats):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(N)
        te_idx = idx[:n_test]
        va_idx = idx[n_test:n_test + n_val]
        tr_idx = idx[n_test + n_val:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        model, lik = train_gp(X_tr, y_tr, kernel_type, ard, n_epochs, lr, noise_lb)
        mu_tr, std_tr = predict_gp(model, lik, X_tr)
        mu_va, std_va = predict_gp(model, lik, X_va)
        mu_te, std_te = predict_gp(model, lik, X_te)

        results.append({
            "train": compute_metrics(y_tr, mu_tr, std_tr),
            "val": compute_metrics(y_va, mu_va, std_va),
            "test": compute_metrics(y_te, mu_te, std_te),
        })
    return results


def summarize_splits(split_results):
    """Aggregate repeated split results."""
    out = {}
    for split_name in ["train", "val", "test"]:
        for metric in ["rmse", "rho", "r2"]:
            vals = [r[split_name][metric] for r in split_results]
            out[f"{split_name}_{metric}_mean"] = float(np.mean(vals))
            out[f"{split_name}_{metric}_std"] = float(np.std(vals))
    out["overfit_gap"] = out["train_rho_mean"] - out["test_rho_mean"]
    return out


# ═══════════════════════════════════════════════════════════════
# Section 3: Phase 1 — Data Analysis
# ═══════════════════════════════════════════════════════════════

def phase1_data_analysis(X, y, families, per_pocket_path):
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Data Analysis & Visualization")
    logger.info("=" * 60)

    N, D = X.shape
    logger.info(f"Dataset: N={N}, D={D}")
    logger.info(f"pKd: range=[{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")

    # PCA analysis
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    n_pc = min(50, N - 1, D)
    pca = PCA(n_components=n_pc)
    X_pca = pca.fit_transform(X_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    idx_90 = int(np.searchsorted(cumvar, 0.9))
    logger.info(f"PCA: {idx_90 + 1} PCs explain 90% variance, PC1={pca.explained_variance_ratio_[0]:.1%}")

    # U_gen per pocket
    u_gen = None
    if per_pocket_path.exists():
        pp = np.load(per_pocket_path)
        u_gen_vals = []
        for fam in families:
            if fam in pp:
                emb = pp[fam]  # (M, 128)
                cov = np.cov(emb.T)
                u_gen_vals.append(np.trace(cov))
            else:
                u_gen_vals.append(np.nan)
        u_gen = np.array(u_gen_vals)
        valid = ~np.isnan(u_gen)
        if valid.any():
            logger.info(f"U_gen: mean={np.nanmean(u_gen):.2f}, std={np.nanstd(u_gen):.2f}")

    # ── Figure 01: Data Overview ──
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # pKd distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(y, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(y.mean(), color="red", linestyle="--", label=f"mean={y.mean():.2f}")
    ax.set_xlabel("pKd"); ax.set_ylabel("Count")
    ax.set_title(f"pKd Distribution (N={N})"); ax.legend()

    # PCA scatter
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="pKd")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Embedding (colored by pKd)")

    # Cumulative variance
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", markersize=3, color="#2ecc71")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5)
    if idx_90 < len(cumvar):
        ax.annotate(f"{idx_90 + 1} PCs (90%)", xy=(idx_90 + 1, 0.9),
                    fontsize=9, color="red", ha="left", va="bottom")
    ax.set_xlabel("# PCs"); ax.set_ylabel("Cumulative Variance")
    ax.set_title("PCA Variance Explained"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # U_gen distribution
    ax = fig.add_subplot(gs[1, 0])
    if u_gen is not None and np.any(~np.isnan(u_gen)):
        ax.hist(u_gen[~np.isnan(u_gen)], bins=20, color="#e67e22", edgecolor="white", alpha=0.8)
        ax.set_xlabel("$U_{gen} = \\mathrm{tr}(\\hat{\\Sigma}_{gen})$")
        ax.set_ylabel("Count")
        ax.set_title("Generation Uncertainty")
    else:
        ax.text(0.5, 0.5, "No per-pocket data", ha="center", va="center", transform=ax.transAxes)

    # Embedding norm distribution
    ax = fig.add_subplot(gs[1, 1])
    norms = np.linalg.norm(X_s, axis=1)
    ax.hist(norms, bins=20, color="#9b59b6", edgecolor="white", alpha=0.8)
    ax.set_xlabel("L2 Norm (standardized)"); ax.set_ylabel("Count")
    ax.set_title("Embedding Norm Distribution")

    # pKd vs U_gen
    ax = fig.add_subplot(gs[1, 2])
    if u_gen is not None and np.any(~np.isnan(u_gen)):
        valid = ~np.isnan(u_gen)
        ax.scatter(u_gen[valid], y[valid], c="#e74c3c", s=30, alpha=0.7)
        rho_ug, p_ug = stats.spearmanr(u_gen[valid], y[valid])
        ax.set_xlabel("$U_{gen}$"); ax.set_ylabel("pKd")
        ax.set_title(f"pKd vs $U_{{gen}}$ (ρ={rho_ug:.3f}, p={p_ug:.3f})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No per-pocket data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("50mol GP Study — Data Overview", fontsize=16, fontweight="bold")
    fig.savefig(FIG_DIR / "01_data_overview.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 01_data_overview.png")
    return {"N": N, "D": D, "pca_90pct": int(idx_90 + 1)}


# ═══════════════════════════════════════════════════════════════
# Section 4: Phase 2 — Grid Search + Bayesian Optimization
# ═══════════════════════════════════════════════════════════════

def phase2_grid_search(X_raw, y):
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2a: Grid Search (kernel × PCA × ARD)")
    logger.info("=" * 60)

    results = []
    total = len(KERNELS) * len(PCA_DIMS_LIST) * len(ARD_OPTIONS)
    done = 0

    for kernel in KERNELS:
        for pca_d in PCA_DIMS_LIST:
            for ard in ARD_OPTIONS:
                done += 1
                pca_dims = pca_d if pca_d > 0 else None
                label = f"{KERNEL_LABELS[kernel]}, PCA={pca_d or 'None'}, ARD={ard}"
                try:
                    X_s, scaler, pca_model, var_exp = prepare_data(X_raw, y, pca_dims)
                    actual_dim = X_s.shape[1]

                    # LOOCV
                    loo_m, _, _ = analytic_loocv(X_s, y, kernel, ard)

                    # 10× splits
                    sp = repeated_splits(X_s, y, kernel, ard, n_repeats=10)
                    sp_sum = summarize_splits(sp)

                    result = {
                        "kernel": kernel, "pca_dims": pca_d, "ard": ard,
                        "actual_dim": actual_dim, "var_explained": var_exp,
                        "loocv": loo_m, "splits": sp_sum, "label": label,
                    }
                    results.append(result)
                    logger.info(f"  [{done}/{total}] {label}: "
                                f"LOOCV ρ={loo_m['rho']:.3f}, "
                                f"10× Test ρ={sp_sum['test_rho_mean']:.3f}±{sp_sum['test_rho_std']:.3f}")
                except Exception as e:
                    logger.error(f"  [{done}/{total}] FAILED {label}: {e}")

    # Sort by LOOCV rho
    results.sort(key=lambda r: r["loocv"]["rho"], reverse=True)

    with open(DATA_DIR / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Grid search done: {len(results)} configs evaluated")

    # Print top-5
    logger.info("\nTop-5 by LOOCV ρ:")
    for i, r in enumerate(results[:5]):
        logger.info(f"  #{i + 1} {r['label']}: ρ={r['loocv']['rho']:.3f}, "
                    f"RMSE={r['loocv']['rmse']:.3f}, R²={r['loocv']['r2']:.3f}")
    return results


def phase2_bayesian_opt(X_raw, y, n_trials=200):
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed, skipping Bayesian Optimization")
        return None

    logger.info("\n" + "=" * 60)
    logger.info(f"Phase 2b: Bayesian Optimization ({n_trials} trials)")
    logger.info("=" * 60)

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", KERNELS)
        pca_d = trial.suggest_categorical("pca_dims", PCA_DIMS_LIST)
        ard = trial.suggest_categorical("ard", [True, False])
        lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
        n_epochs = trial.suggest_int("n_epochs", 50, 300)
        noise_lb = trial.suggest_float("noise_lb", 1e-5, 0.1, log=True)

        pca_dims = pca_d if pca_d > 0 else None
        X_s, _, _, _ = prepare_data(X_raw, y, pca_dims)
        try:
            m, _, _ = analytic_loocv(X_s, y, kernel, ard, n_epochs, lr, noise_lb)
            return m["rmse"]
        except Exception:
            return 999.0

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["best_rmse"] = study.best_value
    logger.info(f"Best BO config: {best}")

    # Save
    bo_results = {
        "best_params": best,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "trials": [{"number": t.number, "value": t.value, "params": t.params}
                    for t in study.trials if t.value is not None],
    }
    with open(DATA_DIR / "bo_results.json", "w") as f:
        json.dump(bo_results, f, indent=2)

    return bo_results


# ═══════════════════════════════════════════════════════════════
# Section 5: Phase 3 — Training Curves
# ═══════════════════════════════════════════════════════════════

def phase3_training_curves(X_raw, y, top_configs):
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Training Curves (top-3 configs)")
    logger.info("=" * 60)

    # Fixed split for training curves
    N = len(y)
    n_te = max(1, int(N * 0.2))
    n_va = max(1, int(N * 0.2))
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    te_idx, va_idx, tr_idx = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

    all_curves = {}
    for i, cfg in enumerate(top_configs[:3]):
        kernel = cfg["kernel"]
        pca_d = cfg["pca_dims"]
        ard = cfg["ard"]
        label = cfg["label"]
        logger.info(f"  Config {i + 1}: {label}")

        pca_dims = pca_d if pca_d > 0 else None
        X_s, _, _, _ = prepare_data(X_raw, y, pca_dims)

        X_tr, y_tr = X_s[tr_idx], y[tr_idx]
        X_va, y_va = X_s[va_idx], y[va_idx]
        X_te, y_te = X_s[te_idx], y[te_idx]

        _, _, curves = train_gp_with_curves(
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            kernel_type=kernel, ard=ard, n_epochs=300, eval_every=5)

        all_curves[label] = curves
        logger.info(f"    Final: train ρ={curves['train_rho'][-1]:.3f}, "
                    f"val ρ={curves['val_rho'][-1]:.3f}, test ρ={curves['test_rho'][-1]:.3f}")

    with open(DATA_DIR / "training_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)

    # ── Figure 03: Training Curves ──
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    metrics_list = [("train_loss", "NLL Loss"), ("train_rmse", "RMSE"),
                    ("train_rho", "Spearman ρ"), ("train_r2", "R²")]

    for row, (label, curves) in enumerate(all_curves.items()):
        epochs = curves["epoch"]
        for col, (base_key, metric_name) in enumerate(metrics_list):
            ax = axes[row, col]
            if col == 0:
                ax.plot(epochs, curves["train_loss"], "b-", label="Train NLL", linewidth=1.5)
                ax.set_ylabel("NLL Loss")
            else:
                key = base_key.replace("train_", "")
                ax.plot(epochs, curves[f"train_{key}"], "b-", label="Train", linewidth=1.5)
                ax.plot(epochs, curves[f"val_{key}"], "orange", label="Val", linewidth=1.5)
                ax.plot(epochs, curves[f"test_{key}"], "g-", label="Test", linewidth=1.5)
                ax.set_ylabel(metric_name)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title(metric_name, fontsize=12, fontweight="bold")
            if col == 0:
                ax.annotate(label, xy=(0, 0.5), xytext=(-0.4, 0.5),
                            xycoords="axes fraction", textcoords="axes fraction",
                            fontsize=9, ha="center", va="center", rotation=90,
                            fontweight="bold")

    fig.suptitle("Training Curves — Top 3 Configurations", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "03_training_curves.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 03_training_curves.png")

    return all_curves


# ═══════════════════════════════════════════════════════════════
# Section 6: Phase 4 — Ablation Study
# ═══════════════════════════════════════════════════════════════

def phase4_ablation(X_raw, y, best_config):
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Ablation Study")
    logger.info("=" * 60)

    best_kernel = best_config["kernel"]
    best_pca = best_config["pca_dims"]
    best_ard = best_config["ard"]

    ablations = []

    # Baseline (best)
    ablations.append({"name": "Best (full)", "kernel": best_kernel,
                       "pca_dims": best_pca, "ard": best_ard})

    # A1: Vary kernel
    for k in KERNELS:
        if k != best_kernel:
            ablations.append({"name": f"Kernel→{KERNEL_LABELS[k]}",
                              "kernel": k, "pca_dims": best_pca, "ard": best_ard})

    # A2: Toggle ARD
    ablations.append({"name": f"ARD→{not best_ard}",
                       "kernel": best_kernel, "pca_dims": best_pca, "ard": not best_ard})

    # A3: Vary PCA
    for p in PCA_DIMS_LIST:
        if p != best_pca:
            ablations.append({"name": f"PCA→{p or 'None'}",
                              "kernel": best_kernel, "pca_dims": p, "ard": best_ard})

    results = []
    for abl in ablations:
        pca_dims = abl["pca_dims"] if abl["pca_dims"] > 0 else None
        try:
            X_s, _, _, var_exp = prepare_data(X_raw, y, pca_dims)
            loo_m, _, _ = analytic_loocv(X_s, y, abl["kernel"], abl["ard"])
            sp = repeated_splits(X_s, y, abl["kernel"], abl["ard"], n_repeats=10)
            sp_sum = summarize_splits(sp)
            abl_result = {
                "name": abl["name"], "kernel": abl["kernel"],
                "pca_dims": abl["pca_dims"], "ard": abl["ard"],
                "loocv": loo_m, "splits": sp_sum,
            }
            results.append(abl_result)
            logger.info(f"  {abl['name']:25s} | LOOCV ρ={loo_m['rho']:.3f} | "
                        f"10× Test ρ={sp_sum['test_rho_mean']:.3f}±{sp_sum['test_rho_std']:.3f}")
        except Exception as e:
            logger.error(f"  FAILED {abl['name']}: {e}")

    with open(DATA_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════
# Section 7: Phase 5 — Visualization
# ═══════════════════════════════════════════════════════════════

def plot_grid_search(grid_results):
    """Figure 02: Grid search heatmap (kernel × PCA for ARD=False and ARD=True)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ard_val, title in zip(axes, [False, True], ["ARD=False (Isotropic)", "ARD=True"]):
        matrix = np.full((len(KERNELS), len(PCA_DIMS_LIST)), np.nan)
        for r in grid_results:
            if r["ard"] == ard_val:
                ki = KERNELS.index(r["kernel"])
                pi = PCA_DIMS_LIST.index(r["pca_dims"])
                matrix[ki, pi] = r["loocv"]["rho"]

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.6)
        ax.set_xticks(range(len(PCA_DIMS_LIST)))
        ax.set_xticklabels([str(p) if p > 0 else "None" for p in PCA_DIMS_LIST])
        ax.set_yticks(range(len(KERNELS)))
        ax.set_yticklabels([KERNEL_LABELS[k] for k in KERNELS])
        ax.set_xlabel("PCA Dimensions")
        ax.set_ylabel("Kernel")
        ax.set_title(title, fontweight="bold")

        # Annotate
        for ki in range(len(KERNELS)):
            for pi in range(len(PCA_DIMS_LIST)):
                val = matrix[ki, pi]
                if not np.isnan(val):
                    ax.text(pi, ki, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="black" if 0.1 < val < 0.5 else "white")
        plt.colorbar(im, ax=ax, label="LOOCV ρ", shrink=0.8)

    fig.suptitle("Grid Search: LOOCV Spearman ρ", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_grid_search.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 02_grid_search.png")


def plot_test_results(X_raw, y, best_config):
    """Figure 04: Test results — scatter + calibration + residuals."""
    pca_dims = best_config["pca_dims"] if best_config["pca_dims"] > 0 else None
    X_s, _, _, _ = prepare_data(X_raw, y, pca_dims)
    loo_m, loo_pred, loo_std = analytic_loocv(
        X_s, y, best_config["kernel"], best_config["ard"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter
    ax = axes[0]
    sc = ax.scatter(y, loo_pred, c=loo_std, cmap="YlOrRd", s=40, alpha=0.8, edgecolors="gray", linewidth=0.5)
    lims = [min(y.min(), loo_pred.min()) - 0.5, max(y.max(), loo_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.fill_between(np.sort(y), np.sort(y) - 1.96 * np.mean(loo_std),
                     np.sort(y) + 1.96 * np.mean(loo_std), alpha=0.1, color="blue")
    plt.colorbar(sc, ax=ax, label="Predicted σ")
    ax.set_xlabel("True pKd"); ax.set_ylabel("LOOCV Predicted pKd")
    ax.set_title(f"Predicted vs True (ρ={loo_m['rho']:.3f}, R²={loo_m['r2']:.3f})")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    # Calibration
    ax = axes[1]
    alphas = np.linspace(0.05, 0.95, 19)
    z_scores = np.abs(y - loo_pred) / (loo_std + 1e-10)
    observed = []
    for a in alphas:
        z_crit = stats.norm.ppf(0.5 + a / 2)
        observed.append(float(np.mean(z_scores < z_crit)))
    ax.plot(alphas, observed, "bo-", markersize=4, label="Observed")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    ax.set_xlabel("Predicted Coverage"); ax.set_ylabel("Observed Coverage")
    ax.set_title("Uncertainty Calibration")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Residuals
    ax = axes[2]
    residuals = y - loo_pred
    ax.hist(residuals, bins=25, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Residual (True - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (RMSE={loo_m['rmse']:.3f})")

    cfg_str = f"{KERNEL_LABELS[best_config['kernel']]}, PCA={best_config['pca_dims'] or 'None'}, ARD={best_config['ard']}"
    fig.suptitle(f"Test Results — Best Config: {cfg_str}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_test_results.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 04_test_results.png")


def plot_ablation(ablation_results):
    """Figure 05: Ablation study bar chart + table."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    names = [r["name"] for r in ablation_results]
    loo_rhos = [r["loocv"]["rho"] for r in ablation_results]
    split_rhos = [r["splits"]["test_rho_mean"] for r in ablation_results]
    split_stds = [r["splits"]["test_rho_std"] for r in ablation_results]

    x = np.arange(len(names))
    w = 0.35

    # Bar chart
    ax = axes[0]
    ax.barh(x - w / 2, loo_rhos, w, label="LOOCV ρ", color="#3498db", alpha=0.8)
    ax.barh(x + w / 2, split_rhos, w, xerr=split_stds, label="10× Test ρ",
            color="#2ecc71", alpha=0.8, capsize=3)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Ablation: Spearman ρ Comparison", fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Table
    ax = axes[1]
    ax.axis("off")
    cols = ["Config", "LOOCV ρ", "LOOCV R²", "10× Test ρ", "Overfit Gap"]
    rows = []
    for r in ablation_results:
        rows.append([
            r["name"],
            f"{r['loocv']['rho']:.3f}",
            f"{r['loocv']['r2']:.3f}",
            f"{r['splits']['test_rho_mean']:.3f}±{r['splits']['test_rho_std']:.3f}",
            f"{r['splits']['overfit_gap']:.3f}",
        ])
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best
    table[1, 0].set_facecolor("#a8e6cf")

    fig.suptitle("Ablation Study", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_ablation.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 05_ablation.png")


def plot_bo_trace(bo_results):
    """Figure 06: Bayesian optimization trace."""
    if bo_results is None:
        return

    trials = bo_results["trials"]
    values = [t["value"] for t in trials]
    best_so_far = [min(values[:i + 1]) for i in range(len(values))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(range(len(values)), values, s=10, alpha=0.5, c="#3498db")
    ax.plot(range(len(best_so_far)), best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("LOOCV RMSE")
    ax.set_title("Optimization Trace"); ax.legend(); ax.grid(True, alpha=0.3)

    # Parameter importance: count how often each kernel/pca wins
    ax = axes[1]
    kernel_vals = {}
    for t in trials:
        k = t["params"]["kernel"]
        kernel_vals.setdefault(k, []).append(t["value"])
    kernel_names = list(kernel_vals.keys())
    kernel_means = [np.mean(v) for v in kernel_vals.values()]
    kernel_stds = [np.std(v) for v in kernel_vals.values()]
    x = range(len(kernel_names))
    ax.bar(x, kernel_means, yerr=kernel_stds, color="#2ecc71", alpha=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_LABELS.get(k, k) for k in kernel_names])
    ax.set_ylabel("LOOCV RMSE (mean ± std)")
    ax.set_title("RMSE by Kernel Type"); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Bayesian Optimization Results", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_bo_trace.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 06_bo_trace.png")


def plot_kernel_pca_analysis(grid_results):
    """Figures 07+08: Kernel comparison + PCA sweep."""

    # ── Figure 07: Kernel comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Best config per kernel
    best_per_kernel = {}
    for r in grid_results:
        k = r["kernel"]
        if k not in best_per_kernel or r["loocv"]["rho"] > best_per_kernel[k]["loocv"]["rho"]:
            best_per_kernel[k] = r

    ax = axes[0]
    knames = [KERNEL_LABELS[k] for k in KERNELS if k in best_per_kernel]
    kvals_loo = [best_per_kernel[k]["loocv"]["rho"] for k in KERNELS if k in best_per_kernel]
    kvals_sp = [best_per_kernel[k]["splits"]["test_rho_mean"] for k in KERNELS if k in best_per_kernel]
    kvals_sp_std = [best_per_kernel[k]["splits"]["test_rho_std"] for k in KERNELS if k in best_per_kernel]
    x = np.arange(len(knames))
    ax.bar(x - 0.15, kvals_loo, 0.3, label="LOOCV ρ", color="#3498db", alpha=0.8)
    ax.bar(x + 0.15, kvals_sp, 0.3, yerr=kvals_sp_std, label="10× Test ρ",
           color="#2ecc71", alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(knames)
    ax.set_ylabel("Spearman ρ"); ax.set_title("Best Config per Kernel", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # ARD effect
    ax = axes[1]
    ard_f = [r for r in grid_results if not r["ard"]]
    ard_t = [r for r in grid_results if r["ard"]]
    best_noARD = max(ard_f, key=lambda r: r["loocv"]["rho"]) if ard_f else None
    best_ARD = max(ard_t, key=lambda r: r["loocv"]["rho"]) if ard_t else None
    labels_ard = []
    vals_ard = []
    if best_noARD:
        labels_ard.append(f"Best Isotropic\n({best_noARD['label']})")
        vals_ard.append(best_noARD["loocv"]["rho"])
    if best_ARD:
        labels_ard.append(f"Best ARD\n({best_ARD['label']})")
        vals_ard.append(best_ARD["loocv"]["rho"])
    colors = ["#e74c3c", "#2ecc71"]
    ax.bar(range(len(labels_ard)), vals_ard, color=colors[:len(labels_ard)], alpha=0.8)
    ax.set_xticks(range(len(labels_ard))); ax.set_xticklabels(labels_ard, fontsize=8)
    ax.set_ylabel("LOOCV ρ"); ax.set_title("ARD Effect", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Kernel & ARD Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_kernel_ard_analysis.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 07_kernel_ard_analysis.png")

    # ── Figure 08: PCA sweep ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, ard_val, title in zip(axes, [False, True], ["Isotropic", "ARD"]):
        for kernel in KERNELS:
            pca_vals = []
            rho_vals = []
            for r in grid_results:
                if r["kernel"] == kernel and r["ard"] == ard_val:
                    pca_vals.append(r["pca_dims"] if r["pca_dims"] > 0 else 128)
                    rho_vals.append(r["loocv"]["rho"])
            if pca_vals:
                # sort by pca
                order = np.argsort(pca_vals)
                ax.plot([pca_vals[i] for i in order], [rho_vals[i] for i in order],
                        "o-", label=KERNEL_LABELS[kernel], markersize=5)
        ax.set_xlabel("PCA Dimensions (128 = None)")
        ax.set_ylabel("LOOCV ρ")
        ax.set_title(f"PCA Sweep ({title})", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("PCA Dimensionality Sweep", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_pca_sweep.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 08_pca_sweep.png")


def plot_overfit_analysis(grid_results):
    """Figure 09: Overfit analysis — train vs test ρ."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors_map = {"rbf": "#e74c3c", "matern15": "#3498db", "matern25": "#2ecc71", "rq": "#9b59b6"}
    markers_map = {False: "o", True: "^"}

    for r in grid_results:
        tr = r["splits"]["train_rho_mean"]
        te = r["splits"]["test_rho_mean"]
        c = colors_map.get(r["kernel"], "gray")
        m = markers_map.get(r["ard"], "o")
        ax.scatter(tr, te, c=c, marker=m, s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Legend
    for k in KERNELS:
        ax.scatter([], [], c=colors_map[k], label=KERNEL_LABELS[k], s=80)
    ax.scatter([], [], marker="o", c="gray", label="Isotropic", s=80)
    ax.scatter([], [], marker="^", c="gray", label="ARD", s=80)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="No overfit")

    ax.set_xlabel("Train ρ (10× mean)"); ax.set_ylabel("Test ρ (10× mean)")
    ax.set_title("Overfitting Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_overfit_analysis.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 09_overfit_analysis.png")


def plot_embedding_comparison(X_raw, y, best_config):
    """Figure 10: Compare embedding types using best GP config."""
    logger.info("Comparing embedding types with best GP config...")

    embeddings = {"Combined-128 (50mol)": (X_raw, y)}

    # Load tier3 originals for comparison
    for name, xf, yf in [
        ("Encoder-128 (tier3)", "X_encoder_128.npy", "y_pkd_encoder.npy"),
        ("FCFP4-2048", "X_FCFP4_2048.npy", "y_pkd.npy"),
    ]:
        xp = TIER3_DIR / xf
        yp = TIER3_DIR / yf
        if xp.exists() and yp.exists():
            embeddings[name] = (np.load(xp), np.load(yp))

    # 50mol standalone
    x50 = DATA_DIR / "X_50mol_128.npy"
    y50 = DATA_DIR / "y_pkd_50mol.npy"
    if x50.exists() and y50.exists():
        embeddings["50mol-only (N=37)"] = (np.load(x50), np.load(y50))

    results = []
    for ename, (X_e, y_e) in embeddings.items():
        try:
            pca_dims = best_config["pca_dims"] if best_config["pca_dims"] > 0 else None
            # Cap PCA dims for small datasets
            if pca_dims and pca_dims >= X_e.shape[0]:
                pca_dims = max(2, X_e.shape[0] - 2)
            X_s, _, _, var_exp = prepare_data(X_e, y_e, pca_dims)
            loo_m, _, _ = analytic_loocv(X_s, y_e, best_config["kernel"], best_config["ard"])
            results.append({"name": ename, "N": len(y_e), "D": X_e.shape[1],
                            "loocv": loo_m, "var_explained": var_exp})
            logger.info(f"  {ename:30s} N={len(y_e):4d}, LOOCV ρ={loo_m['rho']:.3f}")
        except Exception as e:
            logger.error(f"  FAILED {ename}: {e}")

    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = [r["name"] for r in results]
    x = np.arange(len(names))

    ax = axes[0]
    rhos = [r["loocv"]["rho"] for r in results]
    colors = ["#2ecc71" if v == max(rhos) else "#3498db" for v in rhos]
    ax.barh(x, rhos, color=colors, alpha=0.8)
    ax.set_yticks(x); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOOCV ρ"); ax.set_title("Spearman ρ by Embedding", fontweight="bold")
    for i, v in enumerate(rhos):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    r2s = [r["loocv"]["r2"] for r in results]
    colors2 = ["#2ecc71" if v == max(r2s) else "#e74c3c" if v < 0 else "#3498db" for v in r2s]
    ax.barh(x, r2s, color=colors2, alpha=0.8)
    ax.set_yticks(x); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOOCV R²"); ax.set_title("R² by Embedding", fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(r2s):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")

    cfg_str = f"{KERNEL_LABELS[best_config['kernel']]}, PCA={best_config['pca_dims'] or 'None'}, ARD={best_config['ard']}"
    fig.suptitle(f"Embedding Comparison (GP config: {cfg_str})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_embedding_comparison.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 10_embedding_comparison.png")

    with open(DATA_DIR / "embedding_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# Section 8: Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bo_trials", type=int, default=200)
    args = parser.parse_args()

    t_start = time.time()

    # ── Load data ──
    X_combined = np.load(DATA_DIR / "X_combined_128.npy")
    y_combined = np.load(DATA_DIR / "y_pkd_combined.npy")
    with open(DATA_DIR / "families_combined.json") as f:
        families_combined = json.load(f)

    logger.info(f"Combined dataset: X={X_combined.shape}, y={y_combined.shape}")

    # ── Phase 1: Data Analysis ──
    phase1_info = phase1_data_analysis(
        X_combined, y_combined, families_combined,
        DATA_DIR / "per_pocket_embeddings.npz")

    # ── Phase 2a: Grid Search ──
    grid_results = phase2_grid_search(X_combined, y_combined)

    # ── Phase 2b: Bayesian Optimization ──
    bo_results = phase2_bayesian_opt(X_combined, y_combined, n_trials=args.n_bo_trials)

    # ── Determine best config ──
    best_grid = grid_results[0]  # sorted by LOOCV rho
    best_config = {"kernel": best_grid["kernel"], "pca_dims": best_grid["pca_dims"],
                   "ard": best_grid["ard"]}

    # If BO found better, use BO config
    if bo_results:
        bp = bo_results["best_params"]
        pca_bo = bp["pca_dims"]
        X_s_bo, _, _, _ = prepare_data(X_combined, y_combined,
                                        pca_bo if pca_bo > 0 else None)
        try:
            bo_loo, _, _ = analytic_loocv(X_s_bo, y_combined, bp["kernel"], bp["ard"],
                                           bp.get("n_epochs", 200), bp.get("lr", 0.1),
                                           bp.get("noise_lb", 0.001))
            if bo_loo["rho"] > best_grid["loocv"]["rho"]:
                best_config = {"kernel": bp["kernel"], "pca_dims": bp["pca_dims"],
                               "ard": bp["ard"]}
                logger.info(f"BO config is better: ρ={bo_loo['rho']:.3f} vs grid {best_grid['loocv']['rho']:.3f}")
        except Exception:
            pass

    logger.info(f"\nBest config: {best_config}")
    with open(DATA_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # ── Phase 3: Training Curves ──
    phase3_training_curves(X_combined, y_combined, grid_results[:3])

    # ── Phase 4: Ablation ──
    ablation_results = phase4_ablation(X_combined, y_combined, best_config)

    # ── Phase 5: Visualization ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 5: Generating all figures")
    logger.info("=" * 60)

    plot_grid_search(grid_results)
    plot_test_results(X_combined, y_combined, best_config)
    plot_ablation(ablation_results)
    plot_bo_trace(bo_results)
    plot_kernel_pca_analysis(grid_results)
    plot_overfit_analysis(grid_results)
    plot_embedding_comparison(X_combined, y_combined, best_config)

    # ── Summary ──
    elapsed = time.time() - t_start
    summary = {
        "dataset": "combined_128",
        "N": int(X_combined.shape[0]),
        "D": int(X_combined.shape[1]),
        "best_config": best_config,
        "best_loocv_rho": best_grid["loocv"]["rho"],
        "best_loocv_rmse": best_grid["loocv"]["rmse"],
        "best_loocv_r2": best_grid["loocv"]["r2"],
        "best_10x_test_rho": best_grid["splits"]["test_rho_mean"],
        "best_overfit_gap": best_grid["splits"]["overfit_gap"],
        "n_grid_configs": len(grid_results),
        "n_bo_trials": args.n_bo_trials if HAS_OPTUNA else 0,
        "n_ablations": len(ablation_results),
        "elapsed_seconds": elapsed,
        "device": DEVICE,
    }
    with open(DATA_DIR / "50mol_gp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("STUDY COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Best config: {KERNEL_LABELS[best_config['kernel']]}, "
                f"PCA={best_config['pca_dims'] or 'None'}, ARD={best_config['ard']}")
    logger.info(f"LOOCV: ρ={best_grid['loocv']['rho']:.3f}, RMSE={best_grid['loocv']['rmse']:.3f}, "
                f"R²={best_grid['loocv']['r2']:.3f}")
    logger.info(f"10× Test: ρ={best_grid['splits']['test_rho_mean']:.3f}±"
                f"{best_grid['splits']['test_rho_std']:.3f}")
    logger.info(f"Overfit gap: {best_grid['splits']['overfit_gap']:.3f}")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
