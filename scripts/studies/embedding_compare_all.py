#!/usr/bin/env python3
"""Phase D+E: Compare all embedding strategies via GP evaluation + visualization.

Evaluates embeddings from:
1. Encoder-128 (TargetDiff final layer, baseline)
2. Multi-layer TargetDiff (concat, weighted avg, best single layer)
3. Uni-Mol 512 (pre-trained on 209M conformations)
4. SchNet 128 (QM9 pre-trained)
5. Fusion variants (concat best embeddings + PCA)

Evaluation: LOOCV + 50× repeated random 80/20 splits
Output: 6 figures + JSON results
"""

import json
import logging
import time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = REPO / "results" / "tier3_gp"
SAMPLING_DIR = REPO / "results" / "tier3_sampling"
FIG_DIR = DATA_DIR / "figures" / "pretrained_comparison"
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
            self.mean_module(x), self.covar_module(x))


def train_gp(X, y, kernel_type="rq", n_epochs=200, lr=0.1):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(0.001))
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
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, p = stats.spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "rho": float(rho), "p_value": float(p), "r2": float(r2)}


def repeated_splits(X, y, n_repeats=50, test_frac=0.2, kernel_type="rq"):
    """Repeated random 80/20 splits."""
    n = len(y)
    n_test = max(1, int(n * test_frac))
    results = []
    for seed in range(n_repeats):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model, lik = train_gp(X_tr_s, y_tr, kernel_type)
        model.eval(); lik.eval()
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=DEVICE)
        X_te_t = torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_train = model(X_tr_t).mean.cpu().numpy()
            pred_test = model(X_te_t).mean.cpu().numpy()

        train_m = compute_metrics(y_tr, pred_train)
        test_m = compute_metrics(y_te, pred_test)
        results.append({"train": train_m, "test": test_m})
    return results


def evaluate_embedding(name, X, y, n_repeats=50, pca_dims=None):
    """Full evaluation: LOOCV + repeated splits, optionally with PCA."""
    logger.info(f"Evaluating {name}: X={X.shape}, y={y.shape}")

    # Standardize
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Optional PCA
    if pca_dims is not None and pca_dims < X_s.shape[1]:
        actual_dims = min(pca_dims, X_s.shape[0] - 1)
        if actual_dims < 2:
            logger.warning(f"  Skipping PCA: too few samples ({X_s.shape[0]})")
            return None
        pca = PCA(n_components=actual_dims)
        X_s = pca.fit_transform(X_s)
        var_explained = float(pca.explained_variance_ratio_.sum())
        logger.info(f"  PCA {X.shape[1]}→{pca_dims}: {var_explained:.1%} variance explained")
    else:
        var_explained = 1.0

    # LOOCV
    loo_pred = analytic_loocv(X_s, y)
    loo_metrics = compute_metrics(y, loo_pred)
    logger.info(f"  LOOCV: RMSE={loo_metrics['rmse']:.3f}, ρ={loo_metrics['rho']:.3f}, R²={loo_metrics['r2']:.3f}")

    # Repeated splits
    split_results = repeated_splits(X_s, y, n_repeats=n_repeats)
    test_rhos = [r["test"]["rho"] for r in split_results]
    test_rmses = [r["test"]["rmse"] for r in split_results]
    test_r2s = [r["test"]["r2"] for r in split_results]
    train_rhos = [r["train"]["rho"] for r in split_results]

    result = {
        "name": name,
        "dim": int(X_s.shape[1]),
        "n_samples": int(X_s.shape[0]),
        "pca_dims": pca_dims,
        "var_explained": var_explained,
        "loocv": loo_metrics,
        "splits_50x": {
            "test_rho_mean": float(np.mean(test_rhos)),
            "test_rho_std": float(np.std(test_rhos)),
            "test_rmse_mean": float(np.mean(test_rmses)),
            "test_rmse_std": float(np.std(test_rmses)),
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "train_rho_mean": float(np.mean(train_rhos)),
            "train_rho_std": float(np.std(train_rhos)),
            "overfit_gap": float(np.mean(train_rhos) - np.mean(test_rhos)),
        },
        "all_test_rhos": [float(r) for r in test_rhos],
        "all_test_rmses": [float(r) for r in test_rmses],
    }
    logger.info(f"  50× splits: Test ρ={np.mean(test_rhos):.3f}±{np.std(test_rhos):.3f}, "
                f"Gap={result['splits_50x']['overfit_gap']:.3f}")
    return result


# ── Data Loading ──

def load_multilayer_embeddings(families_encoder):
    """Build multi-layer embedding matrices from per-pocket npz files."""
    n_layers = 10
    layer_embs = {i: [] for i in range(n_layers)}
    valid_families = []

    for family in families_encoder:
        npz_path = SAMPLING_DIR / family / "multilayer_embeddings.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        valid = True
        for i in range(n_layers):
            key = f"layer_{i}"
            if key not in data:
                valid = False
                break
            # Mean pool over molecules → (128,)
            layer_embs[i].append(data[key].mean(axis=0))
        if valid:
            valid_families.append(family)

    if len(valid_families) == 0:
        return None, None, None

    layer_arrays = {i: np.stack(embs) for i, embs in layer_embs.items()}
    return layer_arrays, valid_families, n_layers


def build_embedding_configs(families_encoder, y_encoder):
    """Build all embedding configurations to evaluate."""
    configs = []

    # 1. Baseline Encoder-128
    X_enc = np.load(DATA_DIR / "X_encoder_128.npy")
    configs.append(("Encoder-128 (baseline)", X_enc, y_encoder, None))

    # 2. Multi-layer TargetDiff strategies
    layer_arrays, ml_families, n_layers = load_multilayer_embeddings(families_encoder)
    if layer_arrays is not None:
        # Map to matching y values
        fam_to_idx = {f: i for i, f in enumerate(families_encoder)}
        ml_indices = [fam_to_idx[f] for f in ml_families if f in fam_to_idx]
        y_ml = y_encoder[ml_indices]

        # Best individual layers (try 0, 5, 8, 9)
        for li in [0, 5, 8, 9]:
            if li in layer_arrays:
                configs.append((f"Layer-{li} (128d)", layer_arrays[li], y_ml, None))

        # Concat all 10 layers → 1280-dim + PCA
        X_concat = np.concatenate([layer_arrays[i] for i in range(n_layers)], axis=1)
        for pca_d in [32, 64, 128]:
            configs.append((f"AllLayers→PCA-{pca_d}", X_concat, y_ml, pca_d))

        # Last 3 layers concat → 384-dim
        X_last3 = np.concatenate([layer_arrays[i] for i in [7, 8, 9]], axis=1)
        configs.append(("Last3→384d", X_last3, y_ml, None))

        # Weighted average (deeper layers weighted more)
        weights = np.array([i + 1 for i in range(n_layers)], dtype=np.float32)
        weights /= weights.sum()
        X_wavg = sum(weights[i] * layer_arrays[i] for i in range(n_layers))
        configs.append(("WeightedAvg-128d", X_wavg, y_ml, None))

        # Uniform average
        X_uavg = sum(layer_arrays[i] for i in range(n_layers)) / n_layers
        configs.append(("UniformAvg-128d", X_uavg, y_ml, None))

    # 3. Uni-Mol 512
    unimol_path = DATA_DIR / "X_unimol_512.npy"
    if unimol_path.exists():
        X_um = np.load(unimol_path)
        y_um = np.load(DATA_DIR / "y_pkd_unimol.npy")
        configs.append(("Uni-Mol-512", X_um, y_um, None))
        configs.append(("Uni-Mol→PCA-64", X_um, y_um, 64))
        configs.append(("Uni-Mol→PCA-128", X_um, y_um, 128))

        # Fusion: Encoder-128 + Uni-Mol
        with open(DATA_DIR / "families_unimol.json") as f:
            fam_um = json.load(f)
        fam_enc_set = set(families_encoder)
        fam_um_set = set(fam_um)
        common = sorted(fam_enc_set & fam_um_set)
        if len(common) > 100:
            enc_idx = [families_encoder.index(f) for f in common]
            um_idx = [fam_um.index(f) for f in common]
            X_fused = np.concatenate([X_enc[enc_idx], X_um[um_idx]], axis=1)
            y_fused = y_encoder[enc_idx]
            configs.append(("Encoder+UniMol→PCA-64", X_fused, y_fused, 64))
            configs.append(("Encoder+UniMol→PCA-128", X_fused, y_fused, 128))

    # 4. SchNet 128
    schnet_path = DATA_DIR / "X_schnet_128.npy"
    if schnet_path.exists():
        X_sn = np.load(schnet_path)
        y_sn = np.load(DATA_DIR / "y_pkd_schnet.npy")
        configs.append(("SchNet-128", X_sn, y_sn, None))

        # Fusion: Encoder-128 + SchNet
        with open(DATA_DIR / "families_schnet.json") as f:
            fam_sn = json.load(f)
        fam_enc_set = set(families_encoder)
        fam_sn_set = set(fam_sn)
        common_sn = sorted(fam_enc_set & fam_sn_set)
        if len(common_sn) > 100:
            enc_idx = [families_encoder.index(f) for f in common_sn]
            sn_idx = [fam_sn.index(f) for f in common_sn]
            X_fused_sn = np.concatenate([X_enc[enc_idx], X_sn[sn_idx]], axis=1)
            y_fused_sn = y_encoder[enc_idx]
            configs.append(("Encoder+SchNet→256d", X_fused_sn, y_fused_sn, None))

    # 5. FCFP4-2048 (reference)
    X_fcfp = np.load(DATA_DIR / "X_FCFP4_2048.npy")
    y_fcfp = np.load(DATA_DIR / "y_pkd.npy")
    configs.append(("FCFP4-2048 (ref)", X_fcfp, y_fcfp, None))

    return configs


# ── Visualization ──

def plot_comparison_bars(results, fig_path):
    """Bar chart comparing all embeddings on LOOCV and 50× splits."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Pre-trained Embedding Comparison", fontsize=16, fontweight="bold")

    names = [r["name"] for r in results]
    x = np.arange(len(names))

    # LOOCV metrics
    for ax, metric, label in zip(axes[0], ["rho", "rmse", "r2"],
                                  ["Spearman ρ", "RMSE", "R²"]):
        vals = [r["loocv"][metric] for r in results]
        colors = ['#2ecc71' if v == max(vals) and metric in ["rho", "r2"]
                  else '#e74c3c' if v == min(vals) and metric in ["rho", "r2"]
                  else '#3498db' for v in vals]
        if metric == "rmse":
            colors = ['#2ecc71' if v == min(vals)
                      else '#e74c3c' if v == max(vals)
                      else '#3498db' for v in vals]
        ax.barh(x, vals, color=colors, alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(f"LOOCV {label}")
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va='center', fontsize=7)

    # 50× split metrics
    for ax, metric_key, label in zip(axes[1],
                                      ["test_rho_mean", "test_rmse_mean", "overfit_gap"],
                                      ["Test ρ (50×)", "Test RMSE (50×)", "Overfit Gap"]):
        vals = [r["splits_50x"][metric_key] for r in results]
        if metric_key == "test_rho_mean":
            stds = [r["splits_50x"]["test_rho_std"] for r in results]
            colors = ['#2ecc71' if v == max(vals) else '#3498db' for v in vals]
        elif metric_key == "test_rmse_mean":
            stds = [r["splits_50x"]["test_rmse_std"] for r in results]
            colors = ['#2ecc71' if v == min(vals) else '#3498db' for v in vals]
        else:
            stds = [0] * len(vals)
            colors = ['#2ecc71' if v == min(vals) else '#e74c3c' if v == max(vals)
                      else '#3498db' for v in vals]

        ax.barh(x, vals, xerr=stds if metric_key != "overfit_gap" else None,
                color=colors, alpha=0.8, capsize=3)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path.name}")


def plot_boxplots(results, fig_path):
    """Boxplots of 50× test ρ distributions."""
    names = [r["name"] for r in results]
    data = [r["all_test_rhos"] for r in results]

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.5)))
    bp = ax.boxplot(data, positions=range(len(names)), vert=False, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Test Spearman ρ (50× random 80/20 splits)")
    ax.set_title("Embedding Comparison: Test ρ Distribution", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Highlight best
    means = [np.mean(d) for d in data]
    best_idx = np.argmax(means)
    bp['boxes'][best_idx].set_edgecolor('red')
    bp['boxes'][best_idx].set_linewidth(2)

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path.name}")


def plot_pca_comparison(results_with_X, fig_path):
    """PCA variance explained comparison for top embeddings."""
    # Filter out datasets too small for PCA
    valid = [(name, X) for name, X in results_with_X if X.shape[0] >= 50]
    if len(valid) == 0:
        logger.warning("No datasets large enough for PCA comparison")
        return

    fig, axes = plt.subplots(1, min(4, len(valid)), figsize=(5 * min(4, len(valid)), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (name, X) in zip(axes, valid[:4]):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        n_comp = min(50, X_s.shape[1], X_s.shape[0])
        pca = PCA(n_components=n_comp)
        pca.fit(X_s)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', markersize=3)
        ax.set_xlabel("# PCs")
        ax.set_ylabel("Cumulative Variance Explained")
        ax.set_title(f"{name}\n(dim={X.shape[1]})", fontsize=10)
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Annotate 90% point
        idx_90 = np.searchsorted(cumvar, 0.9)
        if idx_90 < len(cumvar):
            ax.annotate(f"{idx_90+1} PCs\n(90%)", xy=(idx_90+1, 0.9),
                       fontsize=8, color='red', ha='center', va='bottom')

    fig.suptitle("PCA Variance Explained per Embedding Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path.name}")


def plot_overfit_analysis(results, fig_path):
    """Overfitting analysis: train vs test ρ."""
    fig, ax = plt.subplots(figsize=(10, 8))

    names = [r["name"] for r in results]
    train_rhos = [r["splits_50x"]["train_rho_mean"] for r in results]
    test_rhos = [r["splits_50x"]["test_rho_mean"] for r in results]

    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))

    for i, (name, tr, te) in enumerate(zip(names, train_rhos, test_rhos)):
        ax.scatter(tr, te, c=[colors[i]], s=100, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (tr, te), fontsize=7, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points')

    # Diagonal (no overfitting)
    lims = [0, 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='No overfitting')
    ax.set_xlim(lims)
    ax.set_ylim([min(0, min(test_rhos) - 0.05), max(test_rhos) + 0.1])
    ax.set_xlabel("Train ρ (50× mean)")
    ax.set_ylabel("Test ρ (50× mean)")
    ax.set_title("Overfitting Analysis: Train vs Test ρ", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path.name}")


def plot_summary_table(results, fig_path):
    """Summary table figure."""
    columns = ["Embedding", "Dim", "N", "LOOCV ρ", "LOOCV R²",
               "50× Test ρ", "50× Test R²", "Overfit Gap"]

    rows = []
    for r in results:
        rows.append([
            r["name"],
            r["dim"],
            r["n_samples"],
            f"{r['loocv']['rho']:.3f}",
            f"{r['loocv']['r2']:.3f}",
            f"{r['splits_50x']['test_rho_mean']:.3f}±{r['splits_50x']['test_rho_std']:.3f}",
            f"{r['splits_50x']['test_r2_mean']:.3f}±{r['splits_50x']['test_r2_std']:.3f}",
            f"{r['splits_50x']['overfit_gap']:.3f}",
        ])

    # Sort by 50× test rho descending
    sort_idx = sorted(range(len(results)),
                      key=lambda i: results[i]["splits_50x"]["test_rho_mean"],
                      reverse=True)
    rows_sorted = [rows[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(20, max(4, len(rows) * 0.4 + 2)))
    ax.axis('off')

    table = ax.table(cellText=rows_sorted, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color best cells
    for j, col in enumerate(columns):
        if col in ["LOOCV ρ", "50× Test ρ"]:
            vals = [float(rows_sorted[i][j].split("±")[0]) for i in range(len(rows_sorted))]
            best_i = np.argmax(vals)
            table[best_i + 1, j].set_facecolor('#a8e6cf')
        elif col == "Overfit Gap":
            vals = [float(rows_sorted[i][j]) for i in range(len(rows_sorted))]
            best_i = np.argmin(vals)
            table[best_i + 1, j].set_facecolor('#a8e6cf')

    # Header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title("Pre-trained Embedding Comparison Summary\n(sorted by 50× Test ρ)",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path.name}")


def plot_dim_vs_performance(results, fig_path):
    """Scatter: embedding dimension vs test ρ."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["name"] for r in results]
    dims = [r["dim"] for r in results]
    test_rhos = [r["splits_50x"]["test_rho_mean"] for r in results]
    gaps = [r["splits_50x"]["overfit_gap"] for r in results]

    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))

    for ax, yvals, ylabel, title in zip(
        axes, [test_rhos, gaps],
        ["Test ρ (50× mean)", "Overfit Gap"],
        ["Dimension vs Test Performance", "Dimension vs Overfitting"]
    ):
        for i, (name, d, yv) in enumerate(zip(names, dims, yvals)):
            ax.scatter(d, yv, c=[colors[i]], s=100, zorder=5, edgecolors='black', linewidth=0.5)
            ax.annotate(name, (d, yv), fontsize=7, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {fig_path.name}")


# ── Main ──

def main():
    t_start = time.time()

    # Load baseline data
    y_encoder = np.load(DATA_DIR / "y_pkd_encoder.npy")
    with open(DATA_DIR / "families_encoder.json") as f:
        families_encoder = json.load(f)

    # Build all embedding configs
    configs = build_embedding_configs(families_encoder, y_encoder)
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {len(configs)} embedding configurations")
    logger.info(f"{'='*60}\n")

    # Evaluate all
    results = []
    results_with_X = []  # for PCA plot
    for name, X, y, pca_dims in configs:
        try:
            result = evaluate_embedding(name, X, y, n_repeats=50, pca_dims=pca_dims)
            if result is None:
                continue
            results.append(result)
            if pca_dims is None:  # Keep raw X for PCA analysis
                results_with_X.append((name, X))
        except Exception as e:
            logger.error(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

    if len(results) == 0:
        logger.error("No results! Check data availability.")
        return

    # Save results
    results_path = DATA_DIR / "pretrained_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    # Generate figures
    logger.info("\nGenerating figures...")
    plot_comparison_bars(results, FIG_DIR / "01_comparison_bars.png")
    plot_boxplots(results, FIG_DIR / "02_boxplots.png")
    plot_overfit_analysis(results, FIG_DIR / "03_overfit_analysis.png")
    plot_summary_table(results, FIG_DIR / "04_summary_table.png")
    plot_dim_vs_performance(results, FIG_DIR / "05_dim_vs_performance.png")

    if len(results_with_X) >= 2:
        plot_pca_comparison(results_with_X[:4], FIG_DIR / "06_pca_comparison.png")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RANKING (by 50× Test ρ):")
    logger.info(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r["splits_50x"]["test_rho_mean"], reverse=True)
    for i, r in enumerate(sorted_results):
        logger.info(f"  #{i+1} {r['name']:30s} | Test ρ={r['splits_50x']['test_rho_mean']:.3f}±"
                    f"{r['splits_50x']['test_rho_std']:.3f} | Gap={r['splits_50x']['overfit_gap']:.3f}")


if __name__ == "__main__":
    main()
