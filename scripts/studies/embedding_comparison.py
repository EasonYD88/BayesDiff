"""
scripts/studies/embedding_comparison.py
───────────────────────────────────
Extract multiple molecular embeddings from SDF files and compare
their predictive power for pKd using Exact GP + analytic LOOCV.

Embedding candidates:
  E1: ECFP4-128  (current baseline)
  E2: ECFP4-2048
  E3: ECFP6-2048 (radius=3)
  E4: FCFP4-2048 (pharmacophore)
  E5: RDKit-2D   (217 physicochemical descriptors)
  E6: Combined   (E2 + E5)

Usage:
    python scripts/studies/embedding_comparison.py \
        --sdf_dir results/embedding_1000step \
        --affinity_pkl external/targetdiff/data/affinity_info.pkl \
        --output results/embedding_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import gpytorch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


# ── Data loading ─────────────────────────────────────────────
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
    """Find all non-empty SDF files, return {pocket_name: path}."""
    sdf_dir = Path(sdf_dir)
    result = {}
    for sdf_path in sorted(sdf_dir.rglob("*.sdf")):
        if sdf_path.stat().st_size > 0:
            pocket_name = sdf_path.stem.replace("_generated", "")
            result[pocket_name] = sdf_path
    return result


def load_molecules(sdf_path: Path):
    """Load all valid RDKit Mol objects from an SDF file."""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols = [m for m in suppl if m is not None]
    return mols


# ── Embedding extractors ────────────────────────────────────
def extract_ecfp(mols, radius=2, n_bits=128):
    """Morgan fingerprint (ECFP)."""
    from rdkit.Chem import AllChem
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps) if fps else None


def extract_fcfp(mols, radius=2, n_bits=2048):
    """Feature-class fingerprint (pharmacophore-based Morgan)."""
    from rdkit.Chem import AllChem
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits,
                                                    useFeatures=True)
        fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps) if fps else None


def extract_rdkit2d(mols):
    """RDKit 2D physicochemical descriptors (217 descriptors)."""
    from rdkit.Chem import Descriptors
    desc_names = [d[0] for d in Descriptors._descList]
    feats = []
    for mol in mols:
        try:
            vals = Descriptors.CalcMolDescriptors(mol)
            vec = [float(vals.get(n, 0.0)) for n in desc_names]
            # Replace NaN/Inf
            vec = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in vec]
            feats.append(vec)
        except Exception:
            continue
    if not feats:
        return None
    return np.array(feats, dtype=np.float32)


EMBEDDING_CONFIGS = OrderedDict({
    "ECFP4-128": {"fn": lambda mols: extract_ecfp(mols, radius=2, n_bits=128),
                  "type": "binary", "dim": 128},
    "ECFP4-2048": {"fn": lambda mols: extract_ecfp(mols, radius=2, n_bits=2048),
                   "type": "binary", "dim": 2048},
    "ECFP6-2048": {"fn": lambda mols: extract_ecfp(mols, radius=3, n_bits=2048),
                   "type": "binary", "dim": 2048},
    "FCFP4-2048": {"fn": lambda mols: extract_fcfp(mols, radius=2, n_bits=2048),
                   "type": "binary", "dim": 2048},
    "RDKit-2D": {"fn": lambda mols: extract_rdkit2d(mols),
                 "type": "continuous", "dim": 217},
})


def extract_all_embeddings(sdf_files: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    """Extract all embedding types for all pockets.
    Returns {emb_name: {pocket_name: mean_embedding}}.
    """
    results = {name: {} for name in EMBEDDING_CONFIGS}

    for i, (pocket, sdf_path) in enumerate(sdf_files.items()):
        mols = load_molecules(sdf_path)
        if not mols:
            continue

        for emb_name, cfg in EMBEDDING_CONFIGS.items():
            emb_matrix = cfg["fn"](mols)
            if emb_matrix is not None and len(emb_matrix) > 0:
                results[emb_name][pocket] = emb_matrix.mean(axis=0)

        if (i + 1) % 10 == 0:
            logger.info(f"  Extracted {i+1}/{len(sdf_files)} pockets")

    return results


# ── Exact GP + LOOCV ─────────────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_dims=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto (Jaccard) kernel for binary fingerprints."""
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        x1x2 = x1 @ x2.T
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)
        denom = x1_sq + x2_sq.T - x1x2 + 1e-8
        K = x1x2 / denom
        if diag:
            return K.diag()
        return K


class TanimotoGPModel(gpytorch.models.ExactGP):
    """Exact GP with Tanimoto kernel (for binary fingerprints)."""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def run_loocv(X, y, kernel_type="matern_iso", n_epochs=80, lr=0.1):
    """Train ExactGP, compute analytic LOOCV, return metrics."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if kernel_type == "tanimoto":
        model = TanimotoGPModel(X_t, y_t, likelihood)
    else:
        model = ExactGPModel(X_t, y_t, likelihood)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval(); likelihood.eval()

    with torch.no_grad():
        K = model.covar_module(X_t).evaluate()
        noise = likelihood.noise.item()
        K_full = K + noise * torch.eye(len(X_t))
        K_inv = torch.linalg.inv(K_full)
        alpha = K_inv @ y_t
        loo_mu = y_t - alpha / K_inv.diag()
        loo_var = torch.clamp(1.0 / K_inv.diag(), min=1e-10)

    mu = loo_mu.numpy()
    var = loo_var.numpy()
    sigma = np.sqrt(var)

    rmse = np.sqrt(np.mean((mu - y) ** 2))
    mae = np.mean(np.abs(mu - y))
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y - mu)**2 / var)
    r2 = 1 - np.sum((y - mu)**2) / np.sum((y - y.mean())**2)
    sp_rho, sp_p = spearmanr(y, mu) if len(y) > 2 else (float('nan'), float('nan'))
    pe_r, pe_p = pearsonr(y, mu) if len(y) > 2 else (float('nan'), float('nan'))
    ci_lo = mu - 1.96 * sigma
    ci_hi = mu + 1.96 * sigma
    cov95 = np.mean((y >= ci_lo) & (y <= ci_hi))

    return {
        "rmse": float(rmse), "mae": float(mae), "nll": float(nll),
        "r2": float(r2), "spearman_rho": float(sp_rho), "spearman_p": float(sp_p),
        "pearson_r": float(pe_r), "coverage_95": float(cov95),
        "noise": noise, "n": len(y),
        "mu": mu.tolist(), "var": var.tolist(),
    }


def run_repeated_split(X, y, kernel_type="matern_iso", n_repeats=30,
                       test_frac=0.3, n_epochs=60, lr=0.1, seed=42):
    """Repeated random split evaluation."""
    rng = np.random.default_rng(seed)
    N = len(y)
    n_test = max(2, int(N * test_frac))

    metrics_list = []
    for rep in range(n_repeats):
        idx = rng.permutation(N)
        tr, te = idx[:N-n_test], idx[N-n_test:]
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        X_t = torch.tensor(X_tr, dtype=torch.float32)
        y_t = torch.tensor(y_tr, dtype=torch.float32)
        lik = gpytorch.likelihoods.GaussianLikelihood()

        if kernel_type == "tanimoto":
            mdl = TanimotoGPModel(X_t, y_t, lik)
        else:
            mdl = ExactGPModel(X_t, y_t, lik)

        mdl.train(); lik.train()
        opt = torch.optim.Adam(mdl.parameters(), lr=lr)
        mll_fn = gpytorch.mlls.ExactMarginalLogLikelihood(lik, mdl)
        for _ in range(n_epochs):
            opt.zero_grad()
            loss = -mll_fn(mdl(X_t), y_t)
            loss.backward()
            opt.step()

        mdl.eval(); lik.eval()
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = lik(mdl(X_te_t))
            mu_te = pred.mean.numpy()
            var_te = pred.variance.numpy()

        rmse = np.sqrt(np.mean((mu_te - y_te)**2))
        sp_rho = spearmanr(y_te, mu_te)[0] if len(y_te) > 2 else float('nan')
        r2 = 1 - np.sum((y_te - mu_te)**2) / np.sum((y_te - y_te.mean())**2) if len(y_te) > 1 else float('nan')
        sigma_te = np.sqrt(np.maximum(var_te, 1e-10))
        cov95 = np.mean((y_te >= mu_te - 1.96*sigma_te) & (y_te <= mu_te + 1.96*sigma_te))

        metrics_list.append({"rmse": float(rmse), "spearman_rho": float(sp_rho),
                             "r2": float(r2), "coverage_95": float(cov95)})

    keys = ["rmse", "r2", "spearman_rho", "coverage_95"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": float('nan'), "std": float('nan')}
    return summary


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--affinity_pkl", required=True)
    parser.add_argument("--output", default="results/embedding_comparison")
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--pca_dim", type=int, default=20,
                        help="PCA target dim for high-dim embeddings")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Load labels & SDF ────────────────────────────────────
    label_map = load_affinity_pkl(Path(args.affinity_pkl))
    sdf_files = find_sdf_files(args.sdf_dir)
    logger.info(f"Found {len(sdf_files)} SDF files, {len(label_map)} pKd labels")

    # ── Extract all embeddings ───────────────────────────────
    logger.info("\n=== Phase 1: Extracting embeddings ===")
    t0 = time.time()
    all_embs = extract_all_embeddings(sdf_files)

    # Add Combined (ECFP4-2048 + RDKit-2D)
    combined = {}
    for pocket in all_embs["ECFP4-2048"]:
        if pocket in all_embs["RDKit-2D"]:
            combined[pocket] = np.concatenate([
                all_embs["ECFP4-2048"][pocket],
                all_embs["RDKit-2D"][pocket],
            ])
    all_embs["Combined"] = combined
    EMBEDDING_CONFIGS["Combined"] = {"type": "mixed", "dim": "2048+217"}

    logger.info(f"Extraction done in {time.time()-t0:.1f}s")
    for name, pockets in all_embs.items():
        logger.info(f"  {name}: {len(pockets)} pockets")

    # ── Build labeled datasets ───────────────────────────────
    datasets = {}
    for emb_name, pocket_embs in all_embs.items():
        X_list, y_list, names = [], [], []
        for pocket, emb in pocket_embs.items():
            pk = label_map.get(pocket)
            if pk is not None:
                X_list.append(emb)
                y_list.append(pk)
                names.append(pocket)
        if X_list:
            datasets[emb_name] = {
                "X": np.stack(X_list).astype(np.float32),
                "y": np.array(y_list, dtype=np.float32),
                "names": names,
            }
    logger.info(f"\nLabeled datasets:")
    for name, ds in datasets.items():
        logger.info(f"  {name}: N={len(ds['y'])}, d={ds['X'].shape[1]}")

    # ── Run evaluation for each embedding ────────────────────
    logger.info("\n=== Phase 2: LOOCV evaluation ===")
    results = OrderedDict()

    for emb_name, ds in datasets.items():
        X, y = ds["X"], ds["y"]
        d = X.shape[1]
        cfg = EMBEDDING_CONFIGS.get(emb_name, {})
        emb_type = cfg.get("type", "unknown")

        logger.info(f"\n--- {emb_name} (d={d}, type={emb_type}) ---")
        entry = {"n": len(y), "d_orig": d, "type": emb_type}

        # PCA for high-dim embeddings
        X_eval = X
        if d > args.pca_dim:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_comp = min(args.pca_dim, len(X) - 1)
            pca = PCA(n_components=n_comp)
            X_eval = pca.fit_transform(X_scaled).astype(np.float32)
            var_explained = pca.explained_variance_ratio_.sum()
            logger.info(f"  PCA: {d} → {n_comp} dims ({var_explained:.1%} variance)")
            entry["d_pca"] = n_comp
            entry["pca_var_explained"] = float(var_explained)
        else:
            entry["d_pca"] = d

        # Isotropic Matérn LOOCV
        logger.info(f"  Running Matérn LOOCV...")
        loocv_matern = run_loocv(X_eval, y, kernel_type="matern_iso",
                                  n_epochs=args.n_epochs, lr=0.1)
        entry["loocv_matern"] = {k: v for k, v in loocv_matern.items()
                                  if k not in ("mu", "var")}
        entry["loocv_matern"]["mu"] = loocv_matern["mu"]
        entry["loocv_matern"]["var"] = loocv_matern["var"]
        logger.info(f"    Matérn: RMSE={loocv_matern['rmse']:.3f}, "
                    f"ρ={loocv_matern['spearman_rho']:.3f}, R²={loocv_matern['r2']:.3f}")

        # Tanimoto LOOCV (for binary embeddings)
        if emb_type == "binary":
            logger.info(f"  Running Tanimoto LOOCV...")
            loocv_tani = run_loocv(X, y, kernel_type="tanimoto",
                                    n_epochs=args.n_epochs, lr=0.1)
            entry["loocv_tanimoto"] = {k: v for k, v in loocv_tani.items()
                                        if k not in ("mu", "var")}
            entry["loocv_tanimoto"]["mu"] = loocv_tani["mu"]
            entry["loocv_tanimoto"]["var"] = loocv_tani["var"]
            logger.info(f"    Tanimoto: RMSE={loocv_tani['rmse']:.3f}, "
                        f"ρ={loocv_tani['spearman_rho']:.3f}, R²={loocv_tani['r2']:.3f}")

        # Repeated split
        logger.info(f"  Running {args.n_repeats}× repeated split...")
        split_res = run_repeated_split(X_eval, y, kernel_type="matern_iso",
                                        n_repeats=args.n_repeats,
                                        n_epochs=min(args.n_epochs, 60),
                                        seed=args.seed)
        entry["repeated_split"] = split_res
        logger.info(f"    Split: RMSE={split_res['rmse']['mean']:.3f}±{split_res['rmse']['std']:.3f}, "
                    f"ρ={split_res['spearman_rho']['mean']:.3f}±{split_res['spearman_rho']['std']:.3f}")

        results[emb_name] = entry

    # ── Save results ─────────────────────────────────────────
    # Strip mu/var from JSON for compact output
    results_json = {}
    for name, entry in results.items():
        rj = {k: v for k, v in entry.items()}
        for key in ["loocv_matern", "loocv_tanimoto"]:
            if key in rj:
                rj[key] = {k: v for k, v in rj[key].items() if k not in ("mu", "var")}
        results_json[name] = rj

    with open(out / "comparison_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out / 'comparison_results.json'}")

    # ── Generate figures ─────────────────────────────────────
    logger.info("\n=== Phase 3: Generating figures ===")

    # Figure 1: Comparison bar chart (LOOCV metrics)
    emb_names = list(results.keys())
    n_emb = len(emb_names)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Embedding Comparison — Exact GP + LOOCV (N=24)", fontsize=14, fontweight="bold")

    metric_keys = [("rmse", "RMSE ↓"), ("r2", "R² ↑"), ("spearman_rho", "Spearman ρ ↑"), ("coverage_95", "CI-95% Coverage")]
    colors = plt.cm.Set2(np.linspace(0, 1, n_emb))

    for ax, (mk, mlabel) in zip(axes, metric_keys):
        vals_matern = []
        vals_tani = []
        for name in emb_names:
            vals_matern.append(results[name]["loocv_matern"][mk])
            if "loocv_tanimoto" in results[name]:
                vals_tani.append(results[name]["loocv_tanimoto"][mk])
            else:
                vals_tani.append(None)

        x = np.arange(n_emb)
        width = 0.35
        bars1 = ax.bar(x - width/2, vals_matern, width, label="Matérn",
                       color=colors, edgecolor="gray", alpha=0.8)
        # Tanimoto bars where available
        tani_vals = [v if v is not None else 0 for v in vals_tani]
        tani_mask = [v is not None for v in vals_tani]
        if any(tani_mask):
            bars2 = ax.bar(x[tani_mask] + width/2,
                           [tani_vals[i] for i in range(n_emb) if tani_mask[i]],
                           width, label="Tanimoto", color="white",
                           edgecolor=[colors[i] for i in range(n_emb) if tani_mask[i]],
                           hatch="///", alpha=0.8)

        for i, v in enumerate(vals_matern):
            ax.annotate(f"{v:.2f}", (x[i] - width/2, v),
                        ha="center", va="bottom", fontsize=6, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(emb_names, rotation=45, ha="right", fontsize=8)
        ax.set_title(mlabel)
        ax.grid(alpha=0.3, axis="y")
        if mk == "rmse":
            ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig_dir / "embedding_comparison_bars.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'embedding_comparison_bars.png'}")
    plt.close()

    # Figure 2: LOOCV scatter for top embeddings (2×3 grid)
    n_show = min(6, n_emb)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("LOOCV Predicted vs True pKd — All Embeddings", fontsize=14, fontweight="bold")

    for idx, emb_name in enumerate(emb_names[:n_show]):
        ax = axes[idx // 3, idx % 3]
        entry = results[emb_name]

        # Use best kernel result
        if ("loocv_tanimoto" in entry and
            entry["loocv_tanimoto"]["spearman_rho"] > entry["loocv_matern"]["spearman_rho"]):
            mu = np.array(entry["loocv_tanimoto"]["mu"])
            var = np.array(entry["loocv_tanimoto"]["var"])
            kernel_used = "Tanimoto"
            metrics = entry["loocv_tanimoto"]
        else:
            mu = np.array(entry["loocv_matern"]["mu"])
            var = np.array(entry["loocv_matern"]["var"])
            kernel_used = "Matérn"
            metrics = entry["loocv_matern"]

        sigma = np.sqrt(var)
        y = datasets[emb_name]["y"]

        ax.errorbar(y, mu, yerr=1.96*sigma, fmt="o", markersize=6,
                     capsize=3, color=colors[idx], ecolor="#BDC3C7", alpha=0.8)
        lims = [min(y.min(), mu.min()) - 0.5, max(y.max(), mu.max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.4)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True pKd"); ax.set_ylabel("LOOCV μ")
        d_info = f"d={entry['d_orig']}"
        if entry.get("d_pca") and entry["d_pca"] != entry["d_orig"]:
            d_info += f"→{entry['d_pca']}"
        ax.set_title(f"{emb_name} ({d_info}, {kernel_used})\n"
                     f"RMSE={metrics['rmse']:.2f}, ρ={metrics['spearman_rho']:.2f}, "
                     f"R²={metrics['r2']:.2f}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(n_show, 6):
        axes[idx // 3, idx % 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_dir / "loocv_scatter_all.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'loocv_scatter_all.png'}")
    plt.close()

    # Figure 3: Summary table
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    ax.set_title("Embedding Comparison Summary — Exact GP + Analytic LOOCV", fontsize=14, fontweight="bold", pad=20)

    headers = ["Embedding", "Type", "d (orig→PCA)", "LOOCV RMSE", "LOOCV R²",
               "LOOCV ρ", "Split ρ (mean±std)", "CI-95%"]
    rows = []
    for name in emb_names:
        e = results[name]
        lm = e["loocv_matern"]
        rs = e["repeated_split"]["spearman_rho"]
        d_str = str(e["d_orig"])
        if e.get("d_pca") and e["d_pca"] != e["d_orig"]:
            d_str += f"→{e['d_pca']}"

        # Pick best kernel for display
        best_rho = lm["spearman_rho"]
        best_rmse = lm["rmse"]
        best_r2 = lm["r2"]
        best_cov = lm["coverage_95"]
        kernel_note = ""
        if "loocv_tanimoto" in e:
            lt = e["loocv_tanimoto"]
            if lt["spearman_rho"] > best_rho:
                best_rho = lt["spearman_rho"]
                best_rmse = lt["rmse"]
                best_r2 = lt["r2"]
                best_cov = lt["coverage_95"]
                kernel_note = " (T)"

        rows.append([
            name,
            e["type"],
            d_str,
            f"{best_rmse:.3f}",
            f"{best_r2:.3f}",
            f"{best_rho:.3f}{kernel_note}",
            f"{rs['mean']:.3f}±{rs['std']:.3f}",
            f"{best_cov:.0%}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best ρ row
    rho_vals = [results[n]["loocv_matern"]["spearman_rho"] for n in emb_names]
    if any("loocv_tanimoto" in results[n] for n in emb_names):
        for i, n in enumerate(emb_names):
            if "loocv_tanimoto" in results[n]:
                rho_vals[i] = max(rho_vals[i], results[n]["loocv_tanimoto"]["spearman_rho"])
    best_idx = int(np.argmax(rho_vals))
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#D5F5E3")

    plt.savefig(fig_dir / "comparison_table.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'comparison_table.png'}")
    plt.close()

    # ── Print final summary ──────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("EMBEDDING COMPARISON SUMMARY")
    logger.info("="*70)
    logger.info(f"  {'Embedding':<15s} {'d':>8s} {'LOOCV RMSE':>12s} {'LOOCV ρ':>10s} {'LOOCV R²':>10s} {'Split ρ':>15s}")
    logger.info(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*15}")
    for name in emb_names:
        e = results[name]
        lm = e["loocv_matern"]
        rs = e["repeated_split"]["spearman_rho"]
        d_str = f"{e['d_orig']}"
        if e.get("d_pca") and e["d_pca"] != e["d_orig"]:
            d_str += f"→{e['d_pca']}"
        best_rho = lm["spearman_rho"]
        if "loocv_tanimoto" in e and e["loocv_tanimoto"]["spearman_rho"] > best_rho:
            best_rho = e["loocv_tanimoto"]["spearman_rho"]
        logger.info(f"  {name:<15s} {d_str:>8s} {lm['rmse']:12.3f} {best_rho:10.3f} {lm['r2']:10.3f} {rs['mean']:7.3f}±{rs['std']:.3f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
