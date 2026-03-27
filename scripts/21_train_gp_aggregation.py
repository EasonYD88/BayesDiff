#!/usr/bin/env python3
"""P0+: Compare aggregation strategies for encoder embeddings.

The current pipeline: per-atom embeddings → scatter_mean → per-molecule (128-dim)
                       → mean across molecules → per-pocket (128-dim)

This script explores better molecule→pocket aggregation:
1. Mean (current baseline)
2. Max-pooling
3. Mean + Max concatenation (256-dim)
4. Attention-weighted mean (learned attention weights per molecule)
5. Weighted by molecule validity score

Also re-extracts per-atom embeddings for atom→molecule aggregation experiments:
6. Atom-level max-pool → molecule → pocket mean
7. Atom-level attention → molecule → pocket mean

Uses existing encoder_embeddings.npy (per-molecule, shape [n_mols, 128]).
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
SAMPLING_DIR = REPO / "results" / "tier3_sampling"
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
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
    )
    model = FlexibleGP(X_t, y_t, likelihood, kernel_type=kernel_type)
    model, likelihood = model.to(DEVICE), likelihood.to(DEVICE)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, likelihood, losses


def predict_gp(model, likelihood, X_test):
    model.eval(); likelihood.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
    return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    rho, p = stats.spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"rmse": float(rmse), "mae": float(mae), "rho": float(rho),
            "p_value": float(p), "r2": float(r2)}


def loocv(X, y, kernel_type="rq", n_epochs=200, lr=0.1):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        model, lik, _ = train_gp(X[mask], y[mask], kernel_type, n_epochs, lr)
        pred, _ = predict_gp(model, lik, X[~mask])
        preds[i] = pred[0]
        if (i + 1) % 100 == 0:
            logger.info(f"    LOOCV: {i+1}/{n}")
    return preds


def repeated_splits(X, y, n_repeats=50, test_frac=0.3, kernel_type="rq"):
    results = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(rep)
        idx = rng.permutation(len(y))
        n_tr = int((1 - test_frac) * len(y))
        model, lik, _ = train_gp(X[idx[:n_tr]], y[idx[:n_tr]], kernel_type, 200, 0.1)
        pred, _ = predict_gp(model, lik, X[idx[n_tr:]])
        results.append(compute_metrics(y[idx[n_tr:]], pred))
    return results


def tvt_splits(X, y, n_repeats=30, kernel_type="rq"):
    results = {"train": [], "val": [], "test": []}
    for rep in range(n_repeats):
        rng = np.random.RandomState(rep + 1000)
        idx = rng.permutation(len(y))
        n_tr = int(0.6 * len(y))
        n_val = int(0.2 * len(y))
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]
        model, lik, _ = train_gp(X[tr], y[tr], kernel_type, 200, 0.1)
        for name, idxs in [("train", tr), ("val", va), ("test", te)]:
            pred, _ = predict_gp(model, lik, X[idxs])
            results[name].append(compute_metrics(y[idxs], pred))
    return results


# ── Aggregation strategies ──

def agg_mean(mol_embs):
    """Simple mean across molecules."""
    return mol_embs.mean(axis=0)


def agg_max(mol_embs):
    """Max-pooling across molecules."""
    return mol_embs.max(axis=0)


def agg_mean_max(mol_embs):
    """Concatenation of mean and max (256-dim)."""
    return np.concatenate([mol_embs.mean(axis=0), mol_embs.max(axis=0)])


def agg_std_augmented(mol_embs):
    """Mean + std concatenation (256-dim) — captures diversity."""
    return np.concatenate([mol_embs.mean(axis=0), mol_embs.std(axis=0)])


def agg_attention(mol_embs):
    """Self-attention weighted mean.
    
    Attention weight for each molecule = softmax(||emb_i|| / temperature).
    Molecules with larger norms (stronger signal) get higher weight.
    """
    norms = np.linalg.norm(mol_embs, axis=1)
    temperature = max(norms.std(), 1e-8)
    logits = norms / temperature
    logits = logits - logits.max()  # numerical stability
    weights = np.exp(logits)
    weights = weights / weights.sum()
    return (mol_embs * weights[:, None]).sum(axis=0)


def agg_pca_attention(mol_embs):
    """PCA-based attention: weight by variance contribution.
    
    Project each molecule onto first PC, use projection magnitude as weight.
    """
    if mol_embs.shape[0] < 2:
        return mol_embs.mean(axis=0)
    centered = mol_embs - mol_embs.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1_proj = np.abs(centered @ Vt[0])  # projection onto PC1
    temperature = max(pc1_proj.std(), 1e-8)
    logits = pc1_proj / temperature
    logits = logits - logits.max()  # numerical stability
    weights = np.exp(logits)
    weights = weights / weights.sum()
    return (mol_embs * weights[:, None]).sum(axis=0)


def agg_trimmed_mean(mol_embs, trim_frac=0.2):
    """Trimmed mean: remove outlier molecules by L2 norm."""
    if mol_embs.shape[0] <= 2:
        return mol_embs.mean(axis=0)
    norms = np.linalg.norm(mol_embs, axis=1)
    n_trim = max(1, int(len(norms) * trim_frac))
    keep = np.argsort(norms)[:-n_trim]  # remove highest-norm (outlier) molecules
    return mol_embs[keep].mean(axis=0)


AGGREGATION_STRATEGIES = {
    "mean": (agg_mean, 128),
    "max": (agg_max, 128),
    "mean+max": (agg_mean_max, 256),
    "mean+std": (agg_std_augmented, 256),
    "attn_norm": (agg_attention, 128),
    "attn_pca": (agg_pca_attention, 128),
    "trimmed_mean": (agg_trimmed_mean, 128),
}


def build_pocket_features(families, sampling_dir, agg_func):
    """Load per-molecule encoder embeddings, aggregate to per-pocket."""
    features = []
    valid_families = []
    for family in families:
        emb_path = sampling_dir / family / "encoder_embeddings.npy"
        if not emb_path.exists():
            continue
        mol_embs = np.load(emb_path)  # (n_mols, 128)
        if mol_embs.shape[0] == 0:
            continue
        pocket_emb = agg_func(mol_embs)
        features.append(pocket_emb)
        valid_families.append(family)
    return np.stack(features), valid_families


def main():
    # ── Load reference data ──
    with open(DATA_DIR / "families_encoder.json") as f:
        families = json.load(f)
    y_ref = np.load(DATA_DIR / "y_pkd_encoder.npy")

    logger.info(f"Reference: {len(families)} pockets, pKd range [{y_ref.min():.2f}, {y_ref.max():.2f}]")

    # Build family→pKd lookup
    family_to_pkd = dict(zip(families, y_ref))

    all_results = {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Build features for all aggregation strategies
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== Building features for all aggregation strategies ===")

    strategy_features = {}
    for name, (agg_func, expected_dim) in AGGREGATION_STRATEGIES.items():
        X, valid_fams = build_pocket_features(families, SAMPLING_DIR, agg_func)
        y = np.array([family_to_pkd[f] for f in valid_fams])

        # Standardize
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        n_nonzero = int(np.sum(np.abs(X).sum(axis=0) > 1e-10))
        valid = n_nonzero > 0 and not np.any(np.isnan(X_s))
        strategy_features[name] = {"X": X_s, "y": y, "dim": X.shape[1], "n": len(y), "valid": valid}
        logger.info(f"  {name:15s}: shape={X.shape}, non-zero features={n_nonzero}, valid={valid}")

    # ══════════════════════════════════════════════════════════════
    # Phase 2: LOOCV for all strategies
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== LOOCV for all aggregation strategies ===")

    for name, data in strategy_features.items():
        if not data["valid"]:
            logger.warning(f"  --- {name}: SKIPPED (degenerate features) ---")
            all_results[f"loocv_{name}"] = {"rmse": float("nan"), "rho": float("nan"),
                                             "r2": float("nan"), "p_value": float("nan"),
                                             "mae": float("nan"), "skipped": True}
            continue
        logger.info(f"\n  --- {name} (dim={data['dim']}) ---")
        preds = loocv(data["X"], data["y"], "rq", 200, 0.1)
        metrics = compute_metrics(data["y"], preds)
        all_results[f"loocv_{name}"] = metrics
        logger.info(f"  RMSE={metrics['rmse']:.3f}, ρ={metrics['rho']:.3f} "
                     f"(p={metrics['p_value']:.2e}), R²={metrics['r2']:.3f}")

    # ══════════════════════════════════════════════════════════════
    # Phase 3: 50× repeated splits for top strategies
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== 50× Repeated Splits ===")

    for name, data in strategy_features.items():
        if not data["valid"]:
            all_results[f"50x_{name}"] = {
                "rmse": "N/A", "rho": "N/A", "r2": "N/A",
                "rmse_mean": float("nan"), "rmse_std": float("nan"),
                "rho_mean": float("nan"), "rho_std": float("nan"),
                "r2_mean": float("nan"), "r2_std": float("nan"),
                "skipped": True,
            }
            logger.warning(f"  {name:15s}: SKIPPED (degenerate)")
            continue
        splits = repeated_splits(data["X"], data["y"], 50, 0.3, "rq")
        rmses = [s["rmse"] for s in splits]
        rhos = [s["rho"] for s in splits]
        r2s = [s["r2"] for s in splits]
        all_results[f"50x_{name}"] = {
            "rmse": f"{np.mean(rmses):.3f}±{np.std(rmses):.3f}",
            "rho": f"{np.mean(rhos):.3f}±{np.std(rhos):.3f}",
            "r2": f"{np.mean(r2s):.3f}±{np.std(r2s):.3f}",
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        }
        logger.info(f"  {name:15s}: RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f}, "
                     f"ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}, "
                     f"R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}")

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Train/Val/Test for top 3 strategies
    # ══════════════════════════════════════════════════════════════
    # Rank by LOOCV rho
    ranked = sorted(
        [(name, all_results[f"loocv_{name}"]["rho"]) for name in strategy_features
         if strategy_features[name].get("valid", True) and not all_results.get(f"loocv_{name}", {}).get("skipped", False)],
        key=lambda x: -x[1]
    )
    top3 = [name for name, _ in ranked[:3]]

    logger.info(f"\n=== 30× Train/Val/Test for top 3: {top3} ===")

    for name in top3:
        data = strategy_features[name]
        tvt = tvt_splits(data["X"], data["y"], 30, "rq")
        for split_name in ["train", "val", "test"]:
            rmses = [m["rmse"] for m in tvt[split_name]]
            rhos = [m["rho"] for m in tvt[split_name]]
            r2s = [m["r2"] for m in tvt[split_name]]
            all_results[f"tvt_{split_name}_{name}"] = {
                "rmse": f"{np.mean(rmses):.3f}±{np.std(rmses):.3f}",
                "rho": f"{np.mean(rhos):.3f}±{np.std(rhos):.3f}",
                "r2": f"{np.mean(r2s):.3f}±{np.std(r2s):.3f}",
                "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
                "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            }
        logger.info(f"  {name:15s}: Train ρ={all_results[f'tvt_train_{name}']['rho']}, "
                     f"Val ρ={all_results[f'tvt_val_{name}']['rho']}, "
                     f"Test ρ={all_results[f'tvt_test_{name}']['rho']}")

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Visualization
    # ══════════════════════════════════════════════════════════════
    logger.info("\n=== Generating comparison figures ===")

    # Only include valid strategies in plots
    valid_names = [n for n in AGGREGATION_STRATEGIES if not all_results.get(f"loocv_{n}", {}).get("skipped", False)]
    loocv_rmses = [all_results[f"loocv_{n}"]["rmse"] for n in valid_names]
    loocv_rhos = [all_results[f"loocv_{n}"]["rho"] for n in valid_names]
    loocv_r2s = [all_results[f"loocv_{n}"]["r2"] for n in valid_names]

    # 5a. LOOCV metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid_names)))

    for ax, metric_vals, ylabel, title in [
        (axes[0], loocv_rmses, "RMSE", "LOOCV RMSE (lower = better)"),
        (axes[1], loocv_rhos, "Spearman ρ", "LOOCV Spearman ρ (higher = better)"),
        (axes[2], loocv_r2s, "R²", "LOOCV R² (higher = better)"),
    ]:
        bars = ax.bar(range(len(valid_names)), metric_vals, color=colors,
                      edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(valid_names)))
        ax.set_xticklabels(valid_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, val in zip(bars, metric_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # Add FCFP4 baseline line
    axes[0].axhline(2.068, color="red", linestyle="--", alpha=0.5, label="FCFP4")
    axes[1].axhline(0.111, color="red", linestyle="--", alpha=0.5, label="FCFP4")
    axes[2].axhline(0.013, color="red", linestyle="--", alpha=0.5, label="FCFP4")
    for ax in axes:
        ax.legend()

    fig.suptitle("P0+: Aggregation Strategy Comparison (LOOCV)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0plus_aggregation_loocv.png", dpi=150)
    plt.close()

    # 5b. 50× splits boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric_key, ylabel in [
        (axes[0], "rho_mean", "Spearman ρ"),
        (axes[1], "r2_mean", "R²"),
        (axes[2], "rmse_mean", "RMSE"),
    ]:
        means = [all_results[f"50x_{n}"][metric_key] for n in valid_names]
        stds = [all_results[f"50x_{n}"][f"{metric_key.replace('_mean','_std')}"] for n in valid_names]
        x = range(len(valid_names))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
               alpha=0.8, capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(valid_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"50× Splits: {ylabel}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("P0+: Aggregation Robustness (50× Random Splits)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0plus_aggregation_splits.png", dpi=150)
    plt.close()

    # 5c. Top 3 train/val/test comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    split_colors = {"train": "#4CAF50", "val": "#FF9800", "test": "#F44336"}
    bar_width = 0.25

    for ax, metric_key, ylabel in [
        (axes[0], "rho", "Spearman ρ"),
        (axes[1], "r2", "R²"),
        (axes[2], "rmse", "RMSE"),
    ]:
        for j, split_name in enumerate(["train", "val", "test"]):
            vals = []
            errs = []
            for name in top3:
                key = f"tvt_{split_name}_{name}"
                vals.append(all_results[key][f"{metric_key}_mean"])
                errs.append(all_results[key][f"{metric_key}_std"])
            x = np.arange(len(top3)) + j * bar_width
            ax.bar(x, vals, bar_width, yerr=errs, label=split_name,
                   color=split_colors[split_name], alpha=0.7, capsize=3)
        ax.set_xticks(np.arange(len(top3)) + bar_width)
        ax.set_xticklabels(top3, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Train/Val/Test: {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("P0+: Top 3 Aggregations — Generalization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0plus_top3_tvt.png", dpi=150)
    plt.close()

    # ── Save results ──
    # Handle NaN values for JSON
    def json_safe(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        return obj

    with open(DATA_DIR / "p0plus_aggregation_results.json", "w") as f:
        json.dump(json_safe(all_results), f, indent=2)
    logger.info(f"\nResults saved to {DATA_DIR / 'p0plus_aggregation_results.json'}")

    # ── Summary table ──
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY: Aggregation Strategy Comparison (LOOCV)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Strategy':<15} {'Dim':>4} {'RMSE':>7} {'ρ':>7} {'R²':>7}")
    logger.info(f"{'-'*45}")
    for name in valid_names:
        m = all_results[f"loocv_{name}"]
        dim = AGGREGATION_STRATEGIES[name][1]
        logger.info(f"{name:<15} {dim:>4} {m['rmse']:>7.3f} {m['rho']:>7.3f} {m['r2']:>7.3f}")
    logger.info(f"{'-'*45}")
    logger.info(f"{'FCFP4-2048':<15} {'2048':>4} {'2.068':>7} {'0.111':>7} {'0.013':>7}")
    logger.info(f"{'='*70}")

    best_name = ranked[0][0]
    best_rho = ranked[0][1]
    logger.info(f"\nBest strategy: {best_name} (LOOCV ρ={best_rho:.3f})")

    # Compare best to mean baseline
    mean_rho = all_results["loocv_mean"]["rho"]
    logger.info(f"vs mean baseline: Δρ = {best_rho - mean_rho:+.3f}")


if __name__ == "__main__":
    main()
