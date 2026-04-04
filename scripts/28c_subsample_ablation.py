#!/usr/bin/env python3
"""28c_subsample_ablation.py — A5 ablation: Fewer molecules per pocket.

Subsamples 20/10/5 molecules per pocket from 50mol data, recomputes mean
embeddings, re-merges with tier3, and evaluates GP with the best config.
Appends results to existing ablation_results.json and regenerates the
ablation figure (05_ablation.png).

No GPU needed — uses pre-saved per-molecule embeddings.
"""

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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "results" / "50mol_gp"
TIER3_DIR = REPO / "results" / "tier3_gp"
FIG_DIR = DATA_DIR / "figures"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KERNEL_LABELS = {"rbf": "RBF", "matern15": "Matérn-3/2", "matern25": "Matérn-5/2", "rq": "RQ"}

# ── GP Model (same as script 28) ──

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


def prepare_data(X, y, pca_dims=None):
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
    return compute_metrics(y, loo_pred, loo_std), loo_pred, loo_std


def repeated_splits(X, y, kernel_type="rq", ard=False, n_repeats=10,
                    val_frac=0.2, test_frac=0.2, n_epochs=200, lr=0.1, noise_lb=0.001):
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
    out = {}
    for split_name in ["train", "val", "test"]:
        for metric in ["rmse", "rho", "r2"]:
            vals = [r[split_name][metric] for r in split_results]
            out[f"{split_name}_{metric}_mean"] = float(np.mean(vals))
            out[f"{split_name}_{metric}_std"] = float(np.std(vals))
    out["overfit_gap"] = out["train_rho_mean"] - out["test_rho_mean"]
    return out


# ── Subsample & merge ──

def subsample_and_merge(per_mol_embeddings, families_50mol_pkd, y_50mol,
                        X_tier3, y_tier3, fam_tier3, n_mols, seed=42):
    """Subsample n_mols per pocket, recompute mean, merge with tier3."""
    rng = np.random.RandomState(seed)

    # Recompute 50mol per-pocket means with subsampled molecules
    X_50mol_sub = []
    for fam in families_50mol_pkd:
        emb = per_mol_embeddings[fam]  # (M, 128)
        M = emb.shape[0]
        if M > n_mols:
            idx = rng.choice(M, size=n_mols, replace=False)
            X_50mol_sub.append(emb[idx].mean(axis=0))
        else:
            X_50mol_sub.append(emb.mean(axis=0))
    X_50mol_sub = np.stack(X_50mol_sub)

    # Merge with tier3 (same logic as script 29)
    X_combined = X_tier3.copy()
    y_combined = y_tier3.copy()
    fam_combined = list(fam_tier3)
    fam_t3_set = set(fam_tier3)
    updated, added = 0, 0

    for i, fam in enumerate(families_50mol_pkd):
        if fam in fam_t3_set:
            idx = fam_tier3.index(fam)
            X_combined[idx] = X_50mol_sub[i]
            updated += 1
        else:
            X_combined = np.vstack([X_combined, X_50mol_sub[i:i+1]])
            y_combined = np.append(y_combined, y_50mol[i])
            fam_combined.append(fam)
            added += 1

    return X_combined, y_combined, fam_combined, updated, added


def plot_ablation(ablation_results):
    """Regenerate figure 05: Ablation study bar chart + table."""
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(ablation_results) * 0.7)))

    names = [r["name"] for r in ablation_results]
    loo_rhos = [r["loocv"]["rho"] for r in ablation_results]
    split_rhos = [r["splits"]["test_rho_mean"] for r in ablation_results]
    split_stds = [r["splits"]["test_rho_std"] for r in ablation_results]

    x = np.arange(len(names))
    w = 0.35

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
    table[1, 0].set_facecolor("#a8e6cf")

    fig.suptitle("Ablation Study", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_ablation.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved 05_ablation.png")


def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("A5 Ablation: Fewer molecules per pocket")
    logger.info("=" * 60)

    # Load best config
    with open(DATA_DIR / "best_config.json") as f:
        best_config = json.load(f)
    kernel = best_config["kernel"]
    pca_dims_raw = best_config["pca_dims"]
    ard = best_config["ard"]
    logger.info(f"Best config: {best_config}")

    # Load per-molecule embeddings
    per_mol = np.load(DATA_DIR / "per_pocket_embeddings.npz")

    # Load 50mol families with pKd
    with open(DATA_DIR / "families_50mol.json") as f:
        fam_50mol = json.load(f)
    y_50mol = np.load(DATA_DIR / "y_pkd_50mol.npy")
    logger.info(f"50mol pockets with pKd: {len(fam_50mol)}")

    # Mol counts for 50mol pockets
    for fam in fam_50mol:
        logger.info(f"  {fam}: {per_mol[fam].shape[0]} mols")

    # Load tier3 data
    X_t3 = np.load(TIER3_DIR / "X_encoder_128.npy")
    y_t3 = np.load(TIER3_DIR / "y_pkd_encoder.npy")
    with open(TIER3_DIR / "families_encoder.json") as f:
        fam_t3 = json.load(f)
    logger.info(f"Tier3: N={len(y_t3)}")

    # Load existing ablation results
    with open(DATA_DIR / "ablation_results.json") as f:
        ablation_results = json.load(f)
    logger.info(f"Existing ablation configs: {len(ablation_results)}")

    # A5: Subsample ablation
    subsample_counts = [20, 10, 5]
    n_seeds = 3  # average over 3 random subsamples for stability

    for n_mols in subsample_counts:
        name = f"Subsample→{n_mols}mol"
        eligible = sum(1 for fam in fam_50mol if per_mol[fam].shape[0] >= n_mols)
        logger.info(f"\n{'─'*40}")
        logger.info(f"{name}: {eligible}/{len(fam_50mol)} pockets have >= {n_mols} molecules")

        # Average LOOCV over multiple random subsamples for stability
        all_loo_metrics = []
        all_split_results = []
        for seed in range(n_seeds):
            X_merged, y_merged, fam_merged, upd, add = subsample_and_merge(
                per_mol, fam_50mol, y_50mol, X_t3, y_t3, fam_t3, n_mols, seed=seed)

            pca_dims = pca_dims_raw if pca_dims_raw > 0 else None
            X_s, _, _, _ = prepare_data(X_merged, y_merged, pca_dims)
            loo_m, _, _ = analytic_loocv(X_s, y_merged, kernel, ard, n_epochs=100, lr=0.1)
            sp = repeated_splits(X_s, y_merged, kernel, ard, n_repeats=5, n_epochs=100)
            all_loo_metrics.append(loo_m)
            all_split_results.extend(sp)
            logger.info(f"  Seed {seed}: N={len(y_merged)}, LOOCV ρ={loo_m['rho']:.3f}")

        # Average LOOCV metrics
        avg_loo = {}
        for key in all_loo_metrics[0]:
            vals = [m[key] for m in all_loo_metrics]
            avg_loo[key] = float(np.mean(vals))

        # Summarize all splits
        sp_sum = summarize_splits(all_split_results)

        result = {
            "name": name,
            "kernel": kernel,
            "pca_dims": pca_dims_raw,
            "ard": ard,
            "n_mols_subsample": n_mols,
            "eligible_pockets": eligible,
            "n_seeds": n_seeds,
            "loocv": avg_loo,
            "splits": sp_sum,
        }
        ablation_results.append(result)
        logger.info(f"  → Avg LOOCV ρ={avg_loo['rho']:.3f} | "
                    f"10× Test ρ={sp_sum['test_rho_mean']:.3f}±{sp_sum['test_rho_std']:.3f}")

    # Save updated ablation results
    with open(DATA_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info(f"\nUpdated ablation_results.json: {len(ablation_results)} configs total")

    # Regenerate ablation figure
    plot_ablation(ablation_results)

    # Update summary
    summary_path = DATA_DIR / "50mol_gp_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["n_ablations"] = len(ablation_results)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Updated 50mol_gp_summary.json")

    elapsed = time.time() - t_start
    logger.info(f"\nA5 ablation done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
