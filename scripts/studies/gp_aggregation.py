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
    """Analytic LOOCV for Exact GP.

    Train once on all data, then compute leave-one-out predictions
    analytically from K_inv:
        mu_loo_i = y_i - alpha_i / K_inv_ii
        var_loo_i = 1 / K_inv_ii
    This is ~Nx faster than brute-force LOOCV.
    """
    # Train GP on ALL data
    model, lik, _ = train_gp(X, y, kernel_type, n_epochs, lr)
    model.eval()
    lik.eval()

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        # Compute full kernel matrix + noise
        K = model.covar_module(X_t).evaluate()
        noise = lik.noise
        K_noisy = K + noise * torch.eye(len(y), device=DEVICE)

        # Solve for alpha and K_inv diagonal
        K_inv = torch.linalg.inv(K_noisy)
        alpha = K_inv @ y_t

        # Analytic LOO predictions
        loo_mu = y_t - alpha / K_inv.diag()

    return loo_mu.cpu().numpy()


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


def json_safe(obj):
    """Handle NaN/Inf values for JSON serialization."""
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


# ── Single-strategy worker (for array jobs) ──

def run_single_strategy(strategy_name):
    """Run LOOCV + 50x splits + 30x TVT for a single aggregation strategy.

    Saves partial results to p0plus_partial_{strategy_name}.json
    """
    with open(DATA_DIR / "families_encoder.json") as f:
        families = json.load(f)
    y_ref = np.load(DATA_DIR / "y_pkd_encoder.npy")
    family_to_pkd = dict(zip(families, y_ref))

    agg_func, expected_dim = AGGREGATION_STRATEGIES[strategy_name]
    logger.info(f"=== Strategy: {strategy_name} (dim={expected_dim}) ===")

    X, valid_fams = build_pocket_features(families, SAMPLING_DIR, agg_func)
    y = np.array([family_to_pkd[f] for f in valid_fams])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    n_nonzero = int(np.sum(np.abs(X).sum(axis=0) > 1e-10))
    valid = n_nonzero > 0 and not np.any(np.isnan(X_s))
    logger.info(f"  shape={X.shape}, non-zero features={n_nonzero}, valid={valid}")

    results = {"strategy": strategy_name, "dim": expected_dim, "n": len(y), "valid": valid}

    if not valid:
        results["loocv"] = {"rmse": None, "rho": None, "r2": None, "p_value": None, "mae": None, "skipped": True}
        results["50x"] = {"skipped": True}
        results["tvt"] = {"skipped": True}
    else:
        # LOOCV (analytic — fast)
        logger.info("  Running analytic LOOCV ...")
        preds = loocv(X_s, y, "rq", 200, 0.1)
        results["loocv"] = compute_metrics(y, preds)
        logger.info(f"  LOOCV: RMSE={results['loocv']['rmse']:.3f}, "
                     f"ρ={results['loocv']['rho']:.3f}, R²={results['loocv']['r2']:.3f}")

        # 50× repeated splits
        logger.info("  Running 50× repeated splits ...")
        splits = repeated_splits(X_s, y, 50, 0.3, "rq")
        rmses = [s["rmse"] for s in splits]
        rhos = [s["rho"] for s in splits]
        r2s = [s["r2"] for s in splits]
        results["50x"] = {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        }
        logger.info(f"  50x: RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f}, "
                     f"ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}")

        # 30× TVT splits
        logger.info("  Running 30× TVT splits ...")
        tvt = tvt_splits(X_s, y, 30, "rq")
        results["tvt"] = {}
        for split_name in ["train", "val", "test"]:
            rmses = [m["rmse"] for m in tvt[split_name]]
            rhos = [m["rho"] for m in tvt[split_name]]
            r2s = [m["r2"] for m in tvt[split_name]]
            results["tvt"][split_name] = {
                "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
                "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            }
        logger.info(f"  TVT: Test ρ={results['tvt']['test']['rho_mean']:.3f}"
                     f"±{results['tvt']['test']['rho_std']:.3f}")

    out_path = DATA_DIR / f"p0plus_partial_{strategy_name}.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(results), f, indent=2)
    logger.info(f"  Saved → {out_path}")


# ── Merge partial results + generate figures ──

def merge_results():
    """Merge partial JSON files from array job into final results + figures."""
    strategy_names = list(AGGREGATION_STRATEGIES.keys())
    all_results = {}

    # Load partial results
    for name in strategy_names:
        path = DATA_DIR / f"p0plus_partial_{name}.json"
        if not path.exists():
            logger.warning(f"  Missing partial result: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        skipped = data.get("loocv", {}).get("skipped", False) or not data.get("valid", True)
        all_results[f"loocv_{name}"] = data["loocv"]
        if not skipped:
            all_results[f"50x_{name}"] = data["50x"]
            if "tvt" in data and not data["tvt"].get("skipped", False):
                for split_name in ["train", "val", "test"]:
                    all_results[f"tvt_{split_name}_{name}"] = data["tvt"][split_name]
        else:
            all_results[f"50x_{name}"] = {"skipped": True}
        logger.info(f"  Loaded {name}: LOOCV ρ={data['loocv'].get('rho', 'N/A')}")

    # Identify valid strategies
    valid_names = [n for n in strategy_names
                   if f"loocv_{n}" in all_results
                   and not all_results.get(f"loocv_{n}", {}).get("skipped", False)]

    if not valid_names:
        logger.error("No valid strategy results found!")
        return

    # Rank by LOOCV rho
    ranked = sorted(
        [(n, all_results[f"loocv_{n}"]["rho"]) for n in valid_names],
        key=lambda x: -x[1]
    )
    top3 = [name for name, _ in ranked[:3]]

    # ── Visualization ──
    logger.info("\n=== Generating comparison figures ===")
    loocv_rmses = [all_results[f"loocv_{n}"]["rmse"] for n in valid_names]
    loocv_rhos = [all_results[f"loocv_{n}"]["rho"] for n in valid_names]
    loocv_r2s = [all_results[f"loocv_{n}"]["r2"] for n in valid_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid_names)))

    # 1. LOOCV metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
    axes[0].axhline(2.068, color="red", linestyle="--", alpha=0.5, label="FCFP4")
    axes[1].axhline(0.111, color="red", linestyle="--", alpha=0.5, label="FCFP4")
    axes[2].axhline(0.013, color="red", linestyle="--", alpha=0.5, label="FCFP4")
    for ax in axes:
        ax.legend()
    fig.suptitle("P0+: Aggregation Strategy Comparison (LOOCV)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0plus_aggregation_loocv.png", dpi=150)
    plt.close()

    # 2. 50× splits bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric_key, ylabel in [
        (axes[0], "rho_mean", "Spearman ρ"),
        (axes[1], "r2_mean", "R²"),
        (axes[2], "rmse_mean", "RMSE"),
    ]:
        means = [all_results[f"50x_{n}"].get(metric_key, 0) for n in valid_names]
        stds = [all_results[f"50x_{n}"].get(f"{metric_key.replace('_mean','_std')}", 0) for n in valid_names]
        x = range(len(valid_names))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", alpha=0.8, capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(valid_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"50× Splits: {ylabel}")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("P0+: Aggregation Robustness (50× Random Splits)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0plus_aggregation_splits.png", dpi=150)
    plt.close()

    # 3. Top 3 TVT comparison
    top3_with_tvt = [n for n in top3 if f"tvt_test_{n}" in all_results]
    if top3_with_tvt:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        split_colors = {"train": "#4CAF50", "val": "#FF9800", "test": "#F44336"}
        bar_width = 0.25
        for ax, metric_key, ylabel in [
            (axes[0], "rho", "Spearman ρ"),
            (axes[1], "r2", "R²"),
            (axes[2], "rmse", "RMSE"),
        ]:
            for j, split_name in enumerate(["train", "val", "test"]):
                vals = [all_results[f"tvt_{split_name}_{n}"][f"{metric_key}_mean"] for n in top3_with_tvt]
                errs = [all_results[f"tvt_{split_name}_{n}"][f"{metric_key}_std"] for n in top3_with_tvt]
                x = np.arange(len(top3_with_tvt)) + j * bar_width
                ax.bar(x, vals, bar_width, yerr=errs, label=split_name,
                       color=split_colors[split_name], alpha=0.7, capsize=3)
            ax.set_xticks(np.arange(len(top3_with_tvt)) + bar_width)
            ax.set_xticklabels(top3_with_tvt, rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Train/Val/Test: {ylabel}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle("P0+: Top 3 Aggregations — Generalization", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "p0plus_top3_tvt.png", dpi=150)
        plt.close()

    # ── Save merged results ──
    with open(DATA_DIR / "p0plus_aggregation_results.json", "w") as f:
        json.dump(json_safe(all_results), f, indent=2)
    logger.info(f"\nMerged results saved to {DATA_DIR / 'p0plus_aggregation_results.json'}")

    # ── Summary table ──
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY: Aggregation Strategy Comparison")
    logger.info(f"{'='*70}")
    logger.info(f"{'Strategy':<15} {'Dim':>4} {'RMSE':>7} {'ρ':>7} {'R²':>7}  "
                f"{'50x ρ':>12} {'TVT test ρ':>14}")
    logger.info(f"{'-'*80}")
    for name in valid_names:
        m = all_results[f"loocv_{name}"]
        dim = AGGREGATION_STRATEGIES[name][1]
        s50 = all_results.get(f"50x_{name}", {})
        tvt_test = all_results.get(f"tvt_test_{name}", {})
        s50_str = f"{s50.get('rho_mean',0):.3f}±{s50.get('rho_std',0):.3f}" if not s50.get("skipped") else "N/A"
        tvt_str = f"{tvt_test.get('rho_mean',0):.3f}±{tvt_test.get('rho_std',0):.3f}" if tvt_test else "N/A"
        logger.info(f"{name:<15} {dim:>4} {m['rmse']:>7.3f} {m['rho']:>7.3f} {m['r2']:>7.3f}  "
                     f"{s50_str:>12} {tvt_str:>14}")
    logger.info(f"{'-'*80}")
    logger.info(f"{'FCFP4-2048':<15} {'2048':>4} {'2.068':>7} {'0.111':>7} {'0.013':>7}")
    logger.info(f"{'='*70}")

    best_name, best_rho = ranked[0]
    logger.info(f"\nBest strategy: {best_name} (LOOCV ρ={best_rho:.3f})")
    mean_rho = all_results["loocv_mean"]["rho"]
    logger.info(f"vs mean baseline: Δρ = {best_rho - mean_rho:+.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P0+: Aggregation strategy comparison")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run a single strategy (for array job parallelism). "
                             "One of: " + ", ".join(AGGREGATION_STRATEGIES.keys()))
    parser.add_argument("--strategy-index", type=int, default=None,
                        help="Strategy index (0-based, for SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial results and generate figures")
    args = parser.parse_args()

    if args.merge:
        merge_results()
        return

    # Determine which strategy to run
    strategy_list = list(AGGREGATION_STRATEGIES.keys())
    if args.strategy_index is not None:
        if args.strategy_index >= len(strategy_list):
            logger.error(f"Strategy index {args.strategy_index} out of range (0-{len(strategy_list)-1})")
            sys.exit(1)
        strategy_name = strategy_list[args.strategy_index]
    elif args.strategy is not None:
        strategy_name = args.strategy
    else:
        # Run ALL strategies sequentially (original behavior)
        logger.info("Running ALL strategies sequentially ...")
        for name in strategy_list:
            run_single_strategy(name)
        merge_results()
        return

    run_single_strategy(strategy_name)


if __name__ == "__main__":
    main()
