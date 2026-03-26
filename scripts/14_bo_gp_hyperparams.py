"""
scripts/14_bo_gp_hyperparams.py
────────────────────────────────
Bayesian Optimization of GP hyperparameters via Optuna.

Searches over:
  - Embedding type (ECFP4-128, RDKit-2D, FCFP4-2048, ...)
  - Kernel type (Matérn-1.5, Matérn-2.5, RBF, Tanimoto, RQ)
  - PCA dimensions (5, 10, 15, 20, None)
  - ARD vs isotropic lengthscale
  - Noise constraint (lower bound)
  - Learning rate & epochs for MLL optimizer
  - Mean function (constant vs linear)
  - Lengthscale / outputscale priors

Objective: minimize LOOCV RMSE (analytic, no retraining).
Validation: best config tested with 50× repeated split + bootstrap.

Usage:
    python scripts/14_bo_gp_hyperparams.py \
        --sdf_dir results/embedding_1000step \
        --affinity_pkl external/targetdiff/data/affinity_info.pkl \
        --output results/bo_gp \
        --n_trials 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import gpytorch
import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# ── Data loading (reused from script 13) ─────────────────────
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


def find_sdf_files(sdf_dir: str) -> dict[str, Path]:
    sdf_dir = Path(sdf_dir)
    result = {}
    for sdf_path in sorted(sdf_dir.rglob("*.sdf")):
        if sdf_path.stat().st_size > 0:
            pocket_name = sdf_path.stem.replace("_generated", "")
            result[pocket_name] = sdf_path
    return result


def load_molecules(sdf_path: Path):
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    return [m for m in suppl if m is not None]


# ── Embedding extractors ────────────────────────────────────
def extract_ecfp(mols, radius=2, n_bits=128):
    from rdkit.Chem import AllChem
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps) if fps else None


def extract_fcfp(mols, radius=2, n_bits=2048):
    from rdkit.Chem import AllChem
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits,
                                                    useFeatures=True)
        fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps) if fps else None


def extract_rdkit2d(mols):
    from rdkit.Chem import Descriptors
    desc_names = [d[0] for d in Descriptors._descList]
    feats = []
    for mol in mols:
        try:
            vals = Descriptors.CalcMolDescriptors(mol)
            vec = [float(vals.get(n, 0.0)) for n in desc_names]
            vec = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in vec]
            feats.append(vec)
        except Exception:
            continue
    return np.array(feats, dtype=np.float32) if feats else None


EMBEDDING_CONFIGS = OrderedDict({
    "ECFP4-128": {"fn": lambda mols: extract_ecfp(mols, 2, 128), "type": "binary"},
    "ECFP4-2048": {"fn": lambda mols: extract_ecfp(mols, 2, 2048), "type": "binary"},
    "ECFP6-2048": {"fn": lambda mols: extract_ecfp(mols, 3, 2048), "type": "binary"},
    "FCFP4-2048": {"fn": lambda mols: extract_fcfp(mols, 2, 2048), "type": "binary"},
    "RDKit-2D": {"fn": lambda mols: extract_rdkit2d(mols), "type": "continuous"},
})


def extract_all_embeddings(sdf_files):
    results = {name: {} for name in EMBEDDING_CONFIGS}
    for i, (pocket, sdf_path) in enumerate(sdf_files.items()):
        mols = load_molecules(sdf_path)
        if not mols:
            continue
        for emb_name, cfg in EMBEDDING_CONFIGS.items():
            emb_matrix = cfg["fn"](mols)
            if emb_matrix is not None and len(emb_matrix) > 0:
                results[emb_name][pocket] = emb_matrix.mean(axis=0)
    # Combined
    combined = {}
    for pocket in results["ECFP4-2048"]:
        if pocket in results["RDKit-2D"]:
            combined[pocket] = np.concatenate([
                results["ECFP4-2048"][pocket], results["RDKit-2D"][pocket],
            ])
    results["Combined"] = combined
    return results


# ── Kernels ──────────────────────────────────────────────────
class TanimotoKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False
    def forward(self, x1, x2, diag=False, **params):
        x1x2 = x1 @ x2.T
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)
        denom = x1_sq + x2_sq.T - x1x2 + 1e-8
        K = x1x2 / denom
        return K.diag() if diag else K


class FlexibleExactGP(gpytorch.models.ExactGP):
    """Exact GP with configurable kernel, mean, and priors."""

    def __init__(self, train_x, train_y, likelihood,
                 kernel_type="matern25", ard=False,
                 mean_type="constant",
                 ls_prior=None, os_prior=None):
        super().__init__(train_x, train_y, likelihood)

        # Mean
        if mean_type == "linear":
            self.mean_module = gpytorch.means.LinearMean(train_x.shape[1])
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        # Base kernel
        d = train_x.shape[1]
        ard_dims = d if ard else None

        if kernel_type == "matern15":
            base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        elif kernel_type == "matern25":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        elif kernel_type == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel_type == "rq":
            base = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)
        elif kernel_type == "tanimoto":
            base = TanimotoKernel()
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

        # Priors on lengthscale
        if ls_prior is not None and hasattr(base, 'lengthscale'):
            mu_ls, sigma_ls = ls_prior
            base.register_prior(
                "lengthscale_prior",
                gpytorch.priors.LogNormalPrior(mu_ls, sigma_ls),
                "lengthscale",
            )

        self.covar_module = gpytorch.kernels.ScaleKernel(base)

        if os_prior is not None:
            mu_os, sigma_os = os_prior
            self.covar_module.register_prior(
                "outputscale_prior",
                gpytorch.priors.LogNormalPrior(mu_os, sigma_os),
                "outputscale",
            )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ── LOOCV engine ─────────────────────────────────────────────
def analytic_loocv(X, y, kernel_type="matern25", ard=False,
                   mean_type="constant", ls_prior=None, os_prior=None,
                   noise_lb=1e-4, n_epochs=100, lr=0.05):
    """Train ExactGP + compute analytic LOOCV. Returns metrics dict."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
    )

    model = FlexibleExactGP(
        X_t, y_t, likelihood,
        kernel_type=kernel_type, ard=ard, mean_type=mean_type,
        ls_prior=ls_prior, os_prior=os_prior,
    )

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = -mll(model(X_t), y_t)
        if torch.isnan(loss) or torch.isinf(loss):
            return None  # Diverged
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > 30:
            break

    model.eval(); likelihood.eval()

    with torch.no_grad():
        K = model.covar_module(X_t).evaluate()
        noise = likelihood.noise.item()
        K_full = K + noise * torch.eye(len(X_t))

        try:
            K_inv = torch.linalg.inv(K_full)
        except Exception:
            return None

        alpha = K_inv @ y_t
        loo_mu = y_t - alpha / K_inv.diag()
        loo_var = torch.clamp(1.0 / K_inv.diag(), min=1e-10)

    mu = loo_mu.numpy()
    var = loo_var.numpy()
    y_np = y

    rmse = float(np.sqrt(np.mean((mu - y_np) ** 2)))
    r2 = float(1 - np.sum((y_np - mu)**2) / np.sum((y_np - y_np.mean())**2))

    try:
        sp_rho = float(spearmanr(y_np, mu)[0])
    except Exception:
        sp_rho = float('nan')
    try:
        pe_r = float(pearsonr(y_np, mu)[0])
    except Exception:
        pe_r = float('nan')

    sigma = np.sqrt(var)
    cov95 = float(np.mean((y_np >= mu - 1.96*sigma) & (y_np <= mu + 1.96*sigma)))
    nll = float(0.5 * np.mean(np.log(2 * np.pi * var) + (y_np - mu)**2 / var))

    hparams = {"noise": noise, "mll": -best_loss}
    try:
        hparams["outputscale"] = model.covar_module.outputscale.item()
    except Exception:
        pass
    try:
        ls = model.covar_module.base_kernel.lengthscale
        hparams["lengthscale_mean"] = float(ls.mean())
        hparams["lengthscale_min"] = float(ls.min())
        hparams["lengthscale_max"] = float(ls.max())
    except Exception:
        pass

    return {
        "rmse": rmse, "r2": r2, "spearman_rho": sp_rho, "pearson_r": pe_r,
        "coverage_95": cov95, "nll": nll, "n": len(y), "mu": mu.tolist(),
        "var": var.tolist(), "hyperparams": hparams, "epochs_used": epoch + 1,
    }


# ── PCA helper ───────────────────────────────────────────────
def apply_pca(X, n_components):
    if n_components is None or n_components >= X.shape[1]:
        return X
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X_sc).astype(np.float32)


# ── Repeated split for validation ────────────────────────────
def repeated_split_eval(X, y, kernel_type, ard, mean_type, ls_prior, os_prior,
                        noise_lb, n_epochs, lr, n_repeats=50, seed=42):
    """Run n_repeats 70/30 splits and report mean±std metrics."""
    rng = np.random.default_rng(seed)
    N = len(y)
    n_test = max(2, int(N * 0.3))

    rmses, rhos, r2s, covs = [], [], [], []
    for rep in range(n_repeats):
        idx = rng.permutation(N)
        tr, te = idx[:N-n_test], idx[N-n_test:]

        X_t = torch.tensor(X[tr], dtype=torch.float32)
        y_t = torch.tensor(y[tr], dtype=torch.float32)
        X_te = torch.tensor(X[te], dtype=torch.float32)

        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
        )
        mdl = FlexibleExactGP(X_t, y_t, lik, kernel_type=kernel_type, ard=ard,
                               mean_type=mean_type, ls_prior=ls_prior, os_prior=os_prior)
        mdl.train(); lik.train()
        opt = torch.optim.Adam(mdl.parameters(), lr=lr)
        mll_fn = gpytorch.mlls.ExactMarginalLogLikelihood(lik, mdl)
        for _ in range(n_epochs):
            opt.zero_grad()
            loss = -mll_fn(mdl(X_t), y_t)
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()

        mdl.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = lik(mdl(X_te))
            mu_te = pred.mean.numpy()
            var_te = np.maximum(pred.variance.numpy(), 1e-10)

        y_te = y[te]
        rmses.append(float(np.sqrt(np.mean((mu_te - y_te)**2))))
        try:
            rhos.append(float(spearmanr(y_te, mu_te)[0]))
        except Exception:
            pass
        ss_res = np.sum((y_te - mu_te)**2)
        ss_tot = np.sum((y_te - y_te.mean())**2)
        r2s.append(float(1 - ss_res/ss_tot) if ss_tot > 0 else float('nan'))
        sig_te = np.sqrt(var_te)
        covs.append(float(np.mean((y_te >= mu_te - 1.96*sig_te) & (y_te <= mu_te + 1.96*sig_te))))

    def stats(arr):
        arr = [v for v in arr if not math.isnan(v)]
        if not arr:
            return {"mean": float('nan'), "std": float('nan')}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    return {
        "rmse": stats(rmses), "spearman_rho": stats(rhos),
        "r2": stats(r2s), "coverage_95": stats(covs),
    }


# ── Optuna objective ─────────────────────────────────────────
def make_objective(datasets: dict):
    """Create Optuna objective that searches over embedding + GP config."""

    def objective(trial: optuna.Trial) -> float:
        # Embedding choice
        emb_name = trial.suggest_categorical("embedding", list(datasets.keys()))
        ds = datasets[emb_name]
        X_raw, y = ds["X"], ds["y"]
        emb_type = ds.get("type", "continuous")

        # PCA dimensions — always suggest to keep Optuna search space static
        pca_dim = trial.suggest_categorical("pca_dim", [5, 8, 10, 15, 20])
        d_orig = X_raw.shape[1]
        if d_orig > 23:  # N-1 = 23
            X = apply_pca(X_raw, pca_dim)
        else:
            X = X_raw

        # Kernel — fixed choices to avoid Optuna dynamic space error
        kernel_type = trial.suggest_categorical(
            "kernel", ["matern15", "matern25", "rbf", "rq", "tanimoto"]
        )

        # Tanimoto only valid for binary embeddings; prune if mismatch
        if kernel_type == "tanimoto" and emb_type != "binary":
            raise optuna.TrialPruned()

        # ARD vs isotropic — always suggest (ignored for tanimoto)
        ard = trial.suggest_categorical("ard", [True, False])
        if kernel_type == "tanimoto":
            X = X_raw  # raw binary FP, no PCA
            ard = False  # override: tanimoto has no lengthscale

        # Mean function
        mean_type = trial.suggest_categorical("mean_type", ["constant", "linear"])

        # Noise lower bound (regularization)
        noise_lb = trial.suggest_float("noise_lb", 1e-4, 5.0, log=True)

        # Learning rate
        lr = trial.suggest_float("lr", 0.005, 0.2, log=True)

        # Epochs
        n_epochs = trial.suggest_int("n_epochs", 50, 300, step=50)

        # Lengthscale prior — always suggest params to keep space static
        use_ls_prior = trial.suggest_categorical("ls_prior", [True, False])
        ls_mu = trial.suggest_float("ls_mu", -1.0, 3.0)
        ls_sigma = trial.suggest_float("ls_sigma", 0.3, 2.0)
        ls_prior = None
        if use_ls_prior and kernel_type != "tanimoto":
            ls_prior = (ls_mu, ls_sigma)

        # Outputscale prior — always suggest params to keep space static
        use_os_prior = trial.suggest_categorical("os_prior", [True, False])
        os_mu = trial.suggest_float("os_mu", -1.0, 2.0)
        os_sigma = trial.suggest_float("os_sigma", 0.3, 2.0)
        os_prior = None
        if use_os_prior:
            os_prior = (os_mu, os_sigma)

        # Run LOOCV
        result = analytic_loocv(
            X, y, kernel_type=kernel_type, ard=ard, mean_type=mean_type,
            ls_prior=ls_prior, os_prior=os_prior, noise_lb=noise_lb,
            n_epochs=n_epochs, lr=lr,
        )

        if result is None or math.isnan(result["rmse"]):
            return float('inf')

        # Primary objective: LOOCV RMSE (lower is better)
        rmse = result["rmse"]

        # Log additional info
        trial.set_user_attr("r2", result["r2"])
        trial.set_user_attr("spearman_rho", result["spearman_rho"])
        trial.set_user_attr("pearson_r", result["pearson_r"])
        trial.set_user_attr("coverage_95", result["coverage_95"])
        trial.set_user_attr("nll", result["nll"])
        trial.set_user_attr("noise", result["hyperparams"].get("noise", float('nan')))
        trial.set_user_attr("mll", result["hyperparams"].get("mll", float('nan')))

        return rmse

    return objective


# ── Visualization ────────────────────────────────────────────
def plot_optimization_history(study, fig_dir):
    """Plot optimization history, parameter importance, and best config."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Optimization history
    ax = axes[0, 0]
    trials = [t for t in study.trials if t.value is not None and t.value < 100]
    vals = [t.value for t in trials]
    best_so_far = np.minimum.accumulate(vals) if vals else []
    ax.scatter(range(len(vals)), vals, alpha=0.4, s=15, label="Trial RMSE")
    if len(best_so_far):
        ax.plot(best_so_far, 'r-', lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("LOOCV RMSE")
    ax.set_title("Optimization History"); ax.legend()

    # 2. RMSE by embedding
    ax = axes[0, 1]
    emb_rmses = {}
    for t in trials:
        emb = t.params.get("embedding", "?")
        emb_rmses.setdefault(emb, []).append(t.value)
    if emb_rmses:
        labels = sorted(emb_rmses.keys())
        data = [emb_rmses[l] for l in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#7fb3d8')
        ax.set_ylabel("LOOCV RMSE"); ax.set_title("RMSE by Embedding")
        ax.tick_params(axis='x', rotation=30)

    # 3. RMSE by kernel
    ax = axes[1, 0]
    kern_rmses = {}
    for t in trials:
        k = t.params.get("kernel", "?")
        kern_rmses.setdefault(k, []).append(t.value)
    if kern_rmses:
        labels = sorted(kern_rmses.keys())
        data = [kern_rmses[l] for l in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#a8d8a8')
        ax.set_ylabel("LOOCV RMSE"); ax.set_title("RMSE by Kernel")
        ax.tick_params(axis='x', rotation=30)

    # 4. Spearman ρ by embedding
    ax = axes[1, 1]
    emb_rhos = {}
    for t in trials:
        emb = t.params.get("embedding", "?")
        rho = t.user_attrs.get("spearman_rho", float('nan'))
        if not math.isnan(rho):
            emb_rhos.setdefault(emb, []).append(rho)
    if emb_rhos:
        labels = sorted(emb_rhos.keys())
        data = [emb_rhos[l] for l in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#d8a87f')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_ylabel("Spearman ρ"); ax.set_title("ρ by Embedding")
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(fig_dir / "bo_optimization_history.png")
    plt.close()
    logger.info(f"  Saved {fig_dir / 'bo_optimization_history.png'}")


def plot_best_config_loocv(best_result, fig_dir):
    """Plot LOOCV scatter and residuals for the best configuration."""
    mu = np.array(best_result["mu"])
    y = np.array(best_result["y"])
    var = np.array(best_result["var"])
    sigma = np.sqrt(var)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Predicted vs true
    ax = axes[0]
    ax.errorbar(y, mu, yerr=1.96*sigma, fmt='o', capsize=3, alpha=0.7, ms=6)
    lims = [min(y.min(), mu.min()) - 0.5, max(y.max(), mu.max()) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel("True pKd"); ax.set_ylabel("LOOCV Predicted pKd")
    ax.set_title(f"Best Config: LOOCV\nRMSE={best_result['rmse']:.2f}, "
                 f"ρ={best_result['spearman_rho']:.3f}, R²={best_result['r2']:.3f}")

    # 2. Residuals
    ax = axes[1]
    residuals = mu - y
    ax.bar(range(len(residuals)), residuals, color=['red' if r > 0 else 'blue' for r in residuals], alpha=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residuals")

    # 3. Calibration
    ax = axes[2]
    z_scores = np.abs((mu - y) / sigma)
    expected_p = np.arange(0.1, 1.01, 0.05)
    from scipy.stats import norm
    expected_z = norm.ppf(0.5 + expected_p / 2)
    observed_p = [np.mean(z_scores <= z) for z in expected_z]
    ax.plot(expected_p, observed_p, 'bo-', label="Observed")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect")
    ax.set_xlabel("Expected coverage"); ax.set_ylabel("Observed coverage")
    ax.set_title("Calibration Plot"); ax.legend()
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(fig_dir / "bo_best_config_loocv.png")
    plt.close()
    logger.info(f"  Saved {fig_dir / 'bo_best_config_loocv.png'}")


def plot_validation_results(split_results, fig_dir):
    """Plot repeated split validation results."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    metrics_names = ["rmse", "spearman_rho", "r2", "coverage_95"]
    labels = ["RMSE", "Spearman ρ", "R²", "CI-95%"]
    means = [split_results[m]["mean"] for m in metrics_names]
    stds = [split_results[m]["std"] for m in metrics_names]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    for bar, m, s in zip(bars, means, stds):
        y_pos = bar.get_height() + s + 0.05
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{m:.3f}±{s:.3f}",
                ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_title("Best Config: 50× Repeated Split Validation")
    ax.set_ylabel("Metric Value")

    plt.tight_layout()
    plt.savefig(fig_dir / "bo_validation_splits.png")
    plt.close()
    logger.info(f"  Saved {fig_dir / 'bo_validation_splits.png'}")


def plot_summary_table(best_trial, loocv_result, split_result, fig_dir, n_total_trials=200):
    """Render summary table as image."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Config table
    params = best_trial.params
    config_rows = [
        ["Embedding", params.get("embedding", "?")],
        ["Kernel", params.get("kernel", "?")],
        ["PCA dim", str(params.get("pca_dim", "N/A"))],
        ["ARD", str(params.get("ard", "N/A"))],
        ["Mean type", params.get("mean_type", "?")],
        ["Noise LB", f"{params.get('noise_lb', 0):.4f}"],
        ["LR", f"{params.get('lr', 0):.4f}"],
        ["Epochs", str(params.get("n_epochs", "?"))],
        ["LS prior", str(params.get("ls_prior", False))],
        ["OS prior", str(params.get("os_prior", False))],
    ]

    metrics_rows = [
        ["", "LOOCV", "50× Split (mean±std)"],
        ["RMSE", f"{loocv_result['rmse']:.3f}",
         f"{split_result['rmse']['mean']:.3f}±{split_result['rmse']['std']:.3f}"],
        ["Spearman ρ", f"{loocv_result['spearman_rho']:.3f}",
         f"{split_result['spearman_rho']['mean']:.3f}±{split_result['spearman_rho']['std']:.3f}"],
        ["R²", f"{loocv_result['r2']:.3f}",
         f"{split_result['r2']['mean']:.3f}±{split_result['r2']['std']:.3f}"],
        ["CI-95%", f"{loocv_result['coverage_95']:.1%}",
         f"{split_result['coverage_95']['mean']:.1%}±{split_result['coverage_95']['std']:.1%}"],
    ]

    # Draw config table
    ax.text(0.25, 0.98, "Best Configuration", ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    t1 = ax.table(cellText=config_rows, colLabels=["Parameter", "Value"],
                  loc='upper left', bbox=[0.0, 0.3, 0.45, 0.65])
    t1.auto_set_font_size(False); t1.set_fontsize(9)
    for (r, c), cell in t1.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4a90d9'); cell.set_text_props(color='white', fontweight='bold')

    # Draw metrics table
    ax.text(0.75, 0.98, "Performance Metrics", ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    t2 = ax.table(cellText=metrics_rows[1:], colLabels=metrics_rows[0],
                  loc='upper right', bbox=[0.5, 0.45, 0.48, 0.50])
    t2.auto_set_font_size(False); t2.set_fontsize(9)
    for (r, c), cell in t2.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4a90d9'); cell.set_text_props(color='white', fontweight='bold')

    # Trial info
    ax.text(0.5, 0.22, f"Best trial #{best_trial.number} / {n_total_trials} total  |  "
            f"LOOCV RMSE = {best_trial.value:.3f}", ha='center', fontsize=11,
            transform=ax.transAxes, style='italic')

    plt.savefig(fig_dir / "bo_summary_table.png", bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {fig_dir / 'bo_summary_table.png'}")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization of GP hyperparameters")
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--affinity_pkl", required=True)
    parser.add_argument("--output", default="results/bo_gp")
    parser.add_argument("--n_trials", type=int, default=200,
                        help="Number of Optuna trials")
    parser.add_argument("--n_val_repeats", type=int, default=50,
                        help="Number of repeated splits for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading data...")
    label_map = load_affinity_pkl(Path(args.affinity_pkl))
    sdf_files = find_sdf_files(args.sdf_dir)
    logger.info(f"Found {len(sdf_files)} SDF files, {len(label_map)} pKd labels")

    logger.info("Extracting embeddings...")
    t0 = time.time()
    all_embs = extract_all_embeddings(sdf_files)
    logger.info(f"Extraction done in {time.time()-t0:.1f}s")

    # Build labeled datasets
    datasets = {}
    for emb_name, pocket_embs in all_embs.items():
        X_list, y_list = [], []
        for pocket, emb in pocket_embs.items():
            pk = label_map.get(pocket)
            if pk is not None:
                X_list.append(emb)
                y_list.append(pk)
        if X_list:
            cfg = EMBEDDING_CONFIGS.get(emb_name, {})
            datasets[emb_name] = {
                "X": np.stack(X_list).astype(np.float32),
                "y": np.array(y_list, dtype=np.float32),
                "type": cfg.get("type", "mixed"),
            }

    logger.info(f"\nDatasets:")
    for name, ds in datasets.items():
        logger.info(f"  {name}: N={len(ds['y'])}, d={ds['X'].shape[1]}")

    # ── Phase 1: Bayesian Optimization ───────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Bayesian Optimization ({args.n_trials} trials)")
    logger.info(f"{'='*60}")

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="gp_hyperparams",
    )

    objective = make_objective(datasets)
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    bo_time = time.time() - t0

    best = study.best_trial
    logger.info(f"\nBO completed in {bo_time:.1f}s")
    logger.info(f"Best trial #{best.number}: LOOCV RMSE = {best.value:.4f}")
    logger.info(f"Best params: {json.dumps(best.params, indent=2)}")
    logger.info(f"Best attrs: ρ={best.user_attrs.get('spearman_rho', 'N/A'):.3f}, "
                f"R²={best.user_attrs.get('r2', 'N/A'):.3f}")

    # ── Phase 2: Re-run best config for full LOOCV details ───
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: Detailed evaluation of best config")
    logger.info(f"{'='*60}")

    bp = best.params
    emb_name = bp["embedding"]
    ds = datasets[emb_name]
    X_raw, y = ds["X"], ds["y"]

    # Apply PCA if needed
    pca_dim = bp.get("pca_dim")
    if bp.get("kernel") == "tanimoto":
        X_best = X_raw
    elif pca_dim and pca_dim < X_raw.shape[1]:
        X_best = apply_pca(X_raw, pca_dim)
    else:
        X_best = X_raw

    # Re-run LOOCV
    loocv_result = analytic_loocv(
        X_best, y,
        kernel_type=bp.get("kernel", "matern25"),
        ard=bp.get("ard", False),
        mean_type=bp.get("mean_type", "constant"),
        ls_prior=(bp.get("ls_mu"), bp.get("ls_sigma")) if bp.get("ls_prior") else None,
        os_prior=(bp.get("os_mu"), bp.get("os_sigma")) if bp.get("os_prior") else None,
        noise_lb=bp.get("noise_lb", 1e-4),
        n_epochs=bp.get("n_epochs", 100),
        lr=bp.get("lr", 0.05),
    )
    loocv_result["y"] = y.tolist()

    logger.info(f"  LOOCV RMSE={loocv_result['rmse']:.3f}, ρ={loocv_result['spearman_rho']:.3f}, "
                f"R²={loocv_result['r2']:.3f}, CI-95%={loocv_result['coverage_95']:.1%}")

    # ── Phase 3: Validation with repeated splits ─────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 3: Validation ({args.n_val_repeats}× repeated split)")
    logger.info(f"{'='*60}")

    split_result = repeated_split_eval(
        X_best, y,
        kernel_type=bp.get("kernel", "matern25"),
        ard=bp.get("ard", False),
        mean_type=bp.get("mean_type", "constant"),
        ls_prior=(bp.get("ls_mu"), bp.get("ls_sigma")) if bp.get("ls_prior") else None,
        os_prior=(bp.get("os_mu"), bp.get("os_sigma")) if bp.get("os_prior") else None,
        noise_lb=bp.get("noise_lb", 1e-4),
        n_epochs=bp.get("n_epochs", 100),
        lr=bp.get("lr", 0.05),
        n_repeats=args.n_val_repeats,
    )

    logger.info(f"  Split RMSE={split_result['rmse']['mean']:.3f}±{split_result['rmse']['std']:.3f}")
    logger.info(f"  Split ρ={split_result['spearman_rho']['mean']:.3f}±{split_result['spearman_rho']['std']:.3f}")

    # ── Phase 4: Top-5 configs comparison ────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 4: Top-5 configurations")
    logger.info(f"{'='*60}")

    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value < 100],
        key=lambda t: t.value
    )
    top5 = []
    for i, t in enumerate(sorted_trials[:5]):
        info = {
            "rank": i + 1,
            "trial": t.number,
            "rmse": t.value,
            "spearman_rho": t.user_attrs.get("spearman_rho", float('nan')),
            "r2": t.user_attrs.get("r2", float('nan')),
            "params": t.params,
        }
        top5.append(info)
        logger.info(f"  #{i+1}: RMSE={t.value:.3f}, ρ={info['spearman_rho']:.3f}, "
                     f"emb={t.params.get('embedding')}, kern={t.params.get('kernel')}")

    # ── Phase 5: Save results ────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Phase 5: Saving results and figures")
    logger.info(f"{'='*60}")

    # All trial data
    all_trials = []
    for t in study.trials:
        if t.value is not None:
            all_trials.append({
                "number": t.number, "value": t.value,
                "params": t.params, "user_attrs": t.user_attrs,
            })

    full_results = {
        "best_trial": {
            "number": best.number, "rmse": best.value,
            "params": best.params, "user_attrs": {k: v for k, v in best.user_attrs.items()},
        },
        "loocv_detailed": {k: v for k, v in loocv_result.items() if k not in ("mu", "var")},
        "repeated_split": split_result,
        "top5": top5,
        "n_trials": len(study.trials),
        "bo_time_sec": bo_time,
        "all_trials": all_trials,
    }

    with open(out / "bo_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info(f"  Saved {out / 'bo_results.json'}")

    # Figures
    plot_optimization_history(study, fig_dir)
    plot_best_config_loocv(loocv_result, fig_dir)
    plot_validation_results(split_result, fig_dir)
    plot_summary_table(best, loocv_result, split_result, fig_dir, n_total_trials=len(study.trials))

    # ── Final summary ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("BAYESIAN OPTIMIZATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Trials:         {len(study.trials)}")
    logger.info(f"  Time:           {bo_time:.1f}s")
    logger.info(f"  Best embedding: {bp['embedding']}")
    logger.info(f"  Best kernel:    {bp.get('kernel', '?')}")
    logger.info(f"  Best LOOCV RMSE: {loocv_result['rmse']:.3f}")
    logger.info(f"  Best LOOCV ρ:    {loocv_result['spearman_rho']:.3f}")
    logger.info(f"  Best LOOCV R²:   {loocv_result['r2']:.3f}")
    logger.info(f"  Split RMSE:      {split_result['rmse']['mean']:.3f}±{split_result['rmse']['std']:.3f}")
    logger.info(f"  Split ρ:         {split_result['spearman_rho']['mean']:.3f}±{split_result['spearman_rho']['std']:.3f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
