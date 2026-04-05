#!/usr/bin/env python3
"""P0++: Compare multi-layer embedding strategies for GP prediction.

Loads multilayer_embeddings.npz (10 layers × n_mols × 128) and evaluates
different layer combination strategies via analytic LOOCV.

Strategies:
  - Individual layers (0-9): which layer is most predictive?
  - Last layer (9) = P0 baseline
  - Concat last-K layers (K=2,3,4,5): multi-scale features
  - Concat all layers (1280-dim → PCA)
  - Uniform average of all layers
  - Weighted average (weight ∝ layer depth)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = REPO / "results" / "tier3_gp"
SAMPLING_DIR = REPO / "results" / "tier3_sampling"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


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
            self.mean_module(x), self.covar_module(x))


def train_gp(X, y, kernel_type="rq", n_epochs=200, lr=0.1, noise_lb=0.001):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb))
    model = FlexibleGP(X_t, y_t, lik, kernel_type).to(DEVICE)
    lik = lik.to(DEVICE)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        opt.step()
    return model, lik


def analytic_loocv(X, y, kernel_type="rq", n_epochs=200, lr=0.1):
    """Train once, compute LOO predictions analytically."""
    model, lik = train_gp(X, y, kernel_type, n_epochs, lr)
    model.eval(); lik.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        K = model.covar_module(X_t).evaluate()
        K_noisy = K + lik.noise * torch.eye(len(y), device=DEVICE)
        K_inv = torch.linalg.inv(K_noisy)
        alpha = K_inv @ y_t
        loo_mu = y_t - alpha / K_inv.diag()
    return loo_mu.cpu().numpy()


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rho, p = stats.spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"rmse": float(rmse), "rho": float(rho), "p_value": float(p), "r2": float(r2)}


def repeated_splits(X, y, n_repeats=50, test_frac=0.3, kernel_type="rq"):
    results = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(rep)
        idx = rng.permutation(len(y))
        n_tr = int((1 - test_frac) * len(y))
        model, lik = train_gp(X[idx[:n_tr]], y[idx[:n_tr]], kernel_type, 200, 0.1)
        model.eval(); lik.eval()
        X_te = torch.tensor(X[idx[n_tr:]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = lik(model(X_te)).mean.cpu().numpy()
        results.append(compute_metrics(y[idx[n_tr:]], pred))
    return results


def build_multilayer_features(families, sampling_dir):
    """Load multilayer embeddings for all pockets.

    Returns: dict layer_idx -> list of (n_mols, 128) arrays, one per pocket
             valid_families: list of family names
    """
    layer_data = {}
    valid_families = []

    for family in families:
        npz_path = sampling_dir / family / "multilayer_embeddings.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
        if len(keys) == 0 or data[keys[0]].shape[0] == 0:
            continue

        for k in keys:
            layer_idx = int(k.split("_")[1])
            if layer_idx not in layer_data:
                layer_data[layer_idx] = []
            layer_data[layer_idx].append(data[k])  # (n_mols, 128)
        valid_families.append(family)

    return layer_data, valid_families


def aggregate_to_pocket(mol_embs_list):
    """Mean-pool molecules → pocket embedding."""
    return np.array([emb.mean(axis=0) for emb in mol_embs_list])


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run single strategy (for array job)")
    parser.add_argument("--strategy-index", type=int, default=None,
                        help="Strategy index (for SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial results and generate figures")
    args = parser.parse_args()

    # ── Define strategies ──
    # Each strategy is: name -> function(layer_data, valid_families) -> (X, dim_description)
    # layer_data: dict layer_idx -> list of (n_mols_i, 128) per pocket
    n_layers = 10  # 0=init, 1-9=attention layers

    strategy_list = []
    # Individual layers
    for i in range(n_layers):
        strategy_list.append(f"layer_{i}")
    # Concat last-K
    for k in [2, 3, 4, 5]:
        strategy_list.append(f"last_{k}")
    # All layers concat
    strategy_list.append("concat_all")
    # Uniform average
    strategy_list.append("avg_all")
    # Depth-weighted average
    strategy_list.append("weighted_avg")
    # Skip-connection style: layer_0 + layer_9
    strategy_list.append("skip_0_9")

    if args.merge:
        merge_results(strategy_list)
        return

    # Determine which strategy to run
    if args.strategy_index is not None:
        if args.strategy_index >= len(strategy_list):
            logger.error(f"Index {args.strategy_index} out of range (0-{len(strategy_list)-1})")
            sys.exit(1)
        strat_name = strategy_list[args.strategy_index]
    elif args.strategy is not None:
        strat_name = args.strategy
    else:
        # Run all sequentially
        for name in strategy_list:
            run_single_strategy(name)
        merge_results(strategy_list)
        return

    run_single_strategy(strat_name)


def build_strategy_features(strat_name, layer_data, valid_families):
    """Build pocket-level feature matrix for a given strategy."""
    n_layers = max(layer_data.keys()) + 1
    n_pockets = len(valid_families)

    if strat_name.startswith("layer_"):
        layer_idx = int(strat_name.split("_")[1])
        X = aggregate_to_pocket(layer_data[layer_idx])
        dim_desc = f"128 (layer {layer_idx})"

    elif strat_name.startswith("last_"):
        k = int(strat_name.split("_")[1])
        layers_to_use = list(range(n_layers - k, n_layers))
        parts = [aggregate_to_pocket(layer_data[i]) for i in layers_to_use]
        X = np.concatenate(parts, axis=1)
        dim_desc = f"{128*k} (last {k} layers)"

    elif strat_name == "concat_all":
        parts = [aggregate_to_pocket(layer_data[i]) for i in range(n_layers)]
        X = np.concatenate(parts, axis=1)
        dim_desc = f"{128*n_layers} (all layers)"

    elif strat_name == "avg_all":
        layers = [aggregate_to_pocket(layer_data[i]) for i in range(n_layers)]
        X = np.mean(layers, axis=0)
        dim_desc = "128 (uniform avg)"

    elif strat_name == "weighted_avg":
        # Linearly increasing weights: layer 0 gets weight 1, layer 9 gets weight 10
        weights = np.arange(1, n_layers + 1, dtype=np.float64)
        weights /= weights.sum()
        layers = [aggregate_to_pocket(layer_data[i]) for i in range(n_layers)]
        X = sum(w * l for w, l in zip(weights, layers))
        dim_desc = "128 (depth-weighted avg)"

    elif strat_name == "skip_0_9":
        X = np.concatenate([
            aggregate_to_pocket(layer_data[0]),
            aggregate_to_pocket(layer_data[n_layers - 1])
        ], axis=1)
        dim_desc = "256 (skip: layer 0 + last)"

    else:
        raise ValueError(f"Unknown strategy: {strat_name}")

    return X, dim_desc


def run_single_strategy(strat_name):
    """Run LOOCV + 50× splits for one strategy."""
    with open(DATA_DIR / "families_encoder.json") as f:
        families = json.load(f)
    y_ref = np.load(DATA_DIR / "y_pkd_encoder.npy")
    family_to_pkd = dict(zip(families, y_ref))

    logger.info(f"=== Strategy: {strat_name} ===")

    layer_data, valid_families = build_multilayer_features(families, SAMPLING_DIR)
    if not valid_families:
        logger.error("No valid pockets with multilayer embeddings!")
        sys.exit(1)

    y = np.array([family_to_pkd[f] for f in valid_families])
    X, dim_desc = build_strategy_features(strat_name, layer_data, valid_families)

    # PCA for high-dim strategies
    from sklearn.decomposition import PCA
    pca_applied = False
    orig_dim = X.shape[1]
    if X.shape[1] > 256:
        n_components = min(50, X.shape[1], len(y) - 1)
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_.sum()
        dim_desc += f" → PCA({n_components}, {var_explained:.1%} var)"
        pca_applied = True

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    logger.info(f"  Features: {dim_desc}, shape={X_s.shape}")

    # LOOCV
    logger.info("  Running analytic LOOCV ...")
    preds = analytic_loocv(X_s, y, "rq", 200, 0.1)
    loocv = compute_metrics(y, preds)
    logger.info(f"  LOOCV: RMSE={loocv['rmse']:.3f}, ρ={loocv['rho']:.3f}, R²={loocv['r2']:.3f}")

    # 50× splits
    logger.info("  Running 50× splits ...")
    splits = repeated_splits(X_s, y, 50, 0.3, "rq")
    rhos = [s["rho"] for s in splits]
    rmses = [s["rmse"] for s in splits]
    r2s = [s["r2"] for s in splits]

    results = {
        "strategy": strat_name,
        "dim": int(X_s.shape[1]),
        "orig_dim": int(orig_dim),
        "dim_desc": dim_desc,
        "pca_applied": pca_applied,
        "n_pockets": len(y),
        "loocv": loocv,
        "splits_50x": {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        }
    }

    out_path = DATA_DIR / f"p0pp_partial_{strat_name}.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(results), f, indent=2)
    logger.info(f"  Saved → {out_path}")


def merge_results(strategy_list):
    """Merge partial results and generate comparison figures."""
    all_results = {}
    for name in strategy_list:
        path = DATA_DIR / f"p0pp_partial_{name}.json"
        if not path.exists():
            logger.warning(f"  Missing: {name}")
            continue
        with open(path) as f:
            all_results[name] = json.load(f)
        logger.info(f"  Loaded {name}: LOOCV ρ={all_results[name]['loocv']['rho']:.3f}")

    if not all_results:
        logger.error("No results to merge!")
        return

    # Save merged
    with open(DATA_DIR / "p0pp_multilayer_results.json", "w") as f:
        json.dump(json_safe(all_results), f, indent=2)

    # ── Figures ──
    names = list(all_results.keys())
    rhos = [all_results[n]["loocv"]["rho"] for n in names]
    rmses = [all_results[n]["loocv"]["rmse"] for n in names]
    r2s = [all_results[n]["loocv"]["r2"] for n in names]

    # Separate individual layers from combination strategies
    layer_names = [n for n in names if n.startswith("layer_")]
    combo_names = [n for n in names if not n.startswith("layer_")]
    layer_rhos = [all_results[n]["loocv"]["rho"] for n in layer_names]
    layer_indices = [int(n.split("_")[1]) for n in layer_names]

    # Fig 1: Per-layer ρ curve (the main insight: which layers carry information?)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sort by layer index
    sort_idx = np.argsort(layer_indices)
    sorted_layers = [layer_indices[i] for i in sort_idx]
    sorted_rhos = [layer_rhos[i] for i in sort_idx]
    sorted_rmses = [all_results[layer_names[i]]["loocv"]["rmse"] for i in sort_idx]
    sorted_r2s = [all_results[layer_names[i]]["loocv"]["r2"] for i in sort_idx]

    axes[0].plot(sorted_layers, sorted_rhos, "o-", color="#2196F3", linewidth=2, markersize=8)
    axes[0].axhline(0.111, color="red", linestyle="--", alpha=0.5, label="FCFP4 baseline")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("LOOCV ρ by Layer (higher = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sorted_layers, sorted_rmses, "s-", color="#FF9800", linewidth=2, markersize=8)
    axes[1].axhline(2.068, color="red", linestyle="--", alpha=0.5, label="FCFP4 baseline")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("LOOCV RMSE by Layer (lower = better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sorted_layers, sorted_r2s, "^-", color="#4CAF50", linewidth=2, markersize=8)
    axes[2].axhline(0.013, color="red", linestyle="--", alpha=0.5, label="FCFP4 baseline")
    axes[2].set_xlabel("Layer Index")
    axes[2].set_ylabel("R²")
    axes[2].set_title("LOOCV R² by Layer (higher = better)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("P0++: Per-Layer Embedding Quality", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p0pp_per_layer.png", dpi=150)
    plt.close()

    # Fig 2: Combination strategies comparison
    if combo_names:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        all_strat_names = layer_names + combo_names
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_strat_names)))

        for ax, metric, ylabel, title in [
            (axes[0], "rmse", "RMSE", "LOOCV RMSE"),
            (axes[1], "rho", "Spearman ρ", "LOOCV Spearman ρ"),
            (axes[2], "r2", "R²", "LOOCV R²"),
        ]:
            vals = [all_results[n]["loocv"][metric] for n in all_strat_names]
            bars = ax.bar(range(len(all_strat_names)), vals, color=colors, edgecolor="black", alpha=0.8)
            ax.set_xticks(range(len(all_strat_names)))
            ax.set_xticklabels(all_strat_names, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")
            # Highlight best
            best_idx = np.argmax(vals) if metric != "rmse" else np.argmin(vals)
            bars[best_idx].set_edgecolor("red")
            bars[best_idx].set_linewidth(2)

        fig.suptitle("P0++: All Strategies Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "p0pp_all_strategies.png", dpi=150)
        plt.close()

    # ── Summary ──
    ranked = sorted(all_results.items(), key=lambda kv: -kv[1]["loocv"]["rho"])
    logger.info(f"\n{'='*80}")
    logger.info("P0++ SUMMARY: Multi-Layer Embedding Comparison")
    logger.info(f"{'='*80}")
    logger.info(f"{'Strategy':<15} {'Dim':>6} {'RMSE':>7} {'ρ':>7} {'R²':>7}  {'50x ρ':>12}")
    logger.info(f"{'-'*65}")
    for name, data in ranked:
        s50 = data["splits_50x"]
        logger.info(f"{name:<15} {data['dim']:>6} {data['loocv']['rmse']:>7.3f} "
                     f"{data['loocv']['rho']:>7.3f} {data['loocv']['r2']:>7.3f}  "
                     f"{s50['rho_mean']:.3f}±{s50['rho_std']:.3f}")
    logger.info(f"{'-'*65}")
    logger.info(f"{'FCFP4-2048':<15} {'2048':>6} {'2.068':>7} {'0.111':>7} {'0.013':>7}")
    logger.info(f"{'P0 (layer 9)':<15} {'128':>6} {'1.949':>7} {'0.369':>7} {'0.119':>7}")
    logger.info(f"{'='*80}")
    logger.info(f"\nBest: {ranked[0][0]} (ρ={ranked[0][1]['loocv']['rho']:.3f})")


if __name__ == "__main__":
    main()
