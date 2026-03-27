"""
scripts/11_gp_training_analysis.py
───────────────────────────────────
Re-train GP with train/val/test split, record per-epoch losses,
evaluate on all splits, and generate comprehensive visualizations.

Usage:
    python scripts/11_gp_training_analysis.py \
        --embeddings results/embedding_rdkit/all_embeddings.npz \
        --affinity_pkl external/targetdiff/data/affinity_info.pkl \
        --output results/embedding_rdkit/gp_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gpytorch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
})


# ── Data loading (reuse from 04_train_gp.py) ────────────────
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


def build_dataset(emb_dict, label_map):
    """Build per-pocket dataset: list of (pocket_name, mean_embedding, pkd)."""
    dataset = []
    for name, emb in emb_dict.items():
        pk = label_map.get(name)
        if pk is None:
            continue
        z_mean = emb.mean(axis=0) if emb.ndim == 2 else emb
        dataset.append({"name": name, "z_mean": z_mean, "pkd": pk, "n_samples": emb.shape[0] if emb.ndim == 2 else 1})
    return dataset


def augment_data(X, y, target_n=200, seed=42):
    if len(X) >= target_n:
        return X, y
    rng = np.random.default_rng(seed)
    n_aug = target_n - len(X)
    X_aug, y_aug = [X], [y]
    for _ in range(n_aug):
        idx = rng.integers(len(X))
        x_new = X[idx] + rng.standard_normal(X.shape[1]).astype(np.float32) * 0.3
        y_new = y[idx] + rng.standard_normal() * 0.5
        X_aug.append(x_new.reshape(1, -1))
        y_aug.append(np.array([y_new]))
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ── GP training with validation tracking ─────────────────────
from bayesdiff.gp_oracle import SVGPModel


def train_gp_with_validation(
    X_train, y_train, X_val, y_val,
    n_inducing=48, n_epochs=200, batch_size=64, lr=0.01, device="cpu",
):
    """Train SVGP and record train/val losses per epoch."""
    dev = torch.device(device)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=dev)
    N = X_t.shape[0]

    n_ind = min(n_inducing, N)
    idx = torch.randperm(N)[:n_ind]
    inducing = X_t[idx].clone()

    model = SVGPModel(inducing).to(dev)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": [],
        "train_nll": [], "val_nll": [],
        "noise": [], "lengthscale_mean": [],
    }

    for epoch in range(n_epochs):
        # Training
        model.train()
        likelihood.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            output = model(X_b)
            loss = -mll(output, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= N
        history["train_loss"].append(epoch_loss)

        # Validation (eval mode)
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Val loss
            val_out = model(X_v)
            val_mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_v))
            val_loss = -val_mll(val_out, y_v).item()
            history["val_loss"].append(val_loss)

            # Train predictions
            train_pred = likelihood(model(X_t))
            train_mu = train_pred.mean.cpu().numpy()
            train_var = train_pred.variance.cpu().numpy()
            train_rmse = np.sqrt(np.mean((train_mu - y_train) ** 2))
            train_nll = 0.5 * np.mean(np.log(2 * np.pi * train_var) + (y_train - train_mu)**2 / train_var)
            history["train_rmse"].append(float(train_rmse))
            history["train_nll"].append(float(train_nll))

            # Val predictions
            val_pred = likelihood(model(X_v))
            val_mu = val_pred.mean.cpu().numpy()
            val_var = val_pred.variance.cpu().numpy()
            val_rmse = np.sqrt(np.mean((val_mu - y_val) ** 2))
            val_nll = 0.5 * np.mean(np.log(2 * np.pi * val_var) + (y_val - val_mu)**2 / val_var)
            history["val_rmse"].append(float(val_rmse))
            history["val_nll"].append(float(val_nll))

        # Hyperparameters
        noise = likelihood.noise.item()
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().mean()
        history["noise"].append(float(noise))
        history["lengthscale_mean"].append(float(ls))

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, "
                f"train_rmse={train_rmse:.3f}, val_rmse={val_rmse:.3f}, "
                f"noise={noise:.4f}"
            )

    model.eval()
    likelihood.eval()
    return model, likelihood, history


def evaluate_split(model, likelihood, X, y, device="cpu"):
    """Evaluate GP on a data split, return detailed metrics."""
    dev = torch.device(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_t))
        mu = pred.mean.cpu().numpy()
        var = pred.variance.cpu().numpy()
    sigma = np.sqrt(var)

    rmse = np.sqrt(np.mean((mu - y) ** 2))
    mae = np.mean(np.abs(mu - y))
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y - mu)**2 / var)
    r2 = 1 - np.sum((y - mu)**2) / np.sum((y - y.mean())**2) if len(y) > 1 else float('nan')

    if len(y) > 2:
        spearman_rho, spearman_p = spearmanr(y, mu)
        pearson_r, pearson_p = pearsonr(y, mu)
    else:
        spearman_rho = spearman_p = pearson_r = pearson_p = float('nan')

    # Coverage: fraction of true values within 95% CI
    ci_low = mu - 1.96 * sigma
    ci_high = mu + 1.96 * sigma
    coverage_95 = np.mean((y >= ci_low) & (y <= ci_high))

    # Mean interval width
    ci_width = np.mean(2 * 1.96 * sigma)

    return {
        "mu": mu, "var": var, "sigma": sigma,
        "rmse": float(rmse), "mae": float(mae), "nll": float(nll), "r2": float(r2),
        "spearman_rho": float(spearman_rho), "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "coverage_95": float(coverage_95), "ci_width": float(ci_width),
        "n": len(y),
    }


def main():
    parser = argparse.ArgumentParser(description="GP Training Analysis with Train/Val/Test Split")
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--affinity_pkl", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/embedding_rdkit/gp_analysis")
    parser.add_argument("--n_inducing", type=int, default=48)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--augment_to", type=int, default=200)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────
    emb_data = np.load(args.embeddings, allow_pickle=True)
    emb_dict = {k: emb_data[k] for k in emb_data.files}
    label_map = load_affinity_pkl(Path(args.affinity_pkl))
    dataset = build_dataset(emb_dict, label_map)
    logger.info(f"Dataset: {len(dataset)} labeled pockets out of {len(emb_dict)} total")

    # ── Train / Val / Test split ─────────────────────────────
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(dataset))
    n_test = max(1, int(len(dataset) * args.test_frac))
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    X_train_raw = np.stack([d["z_mean"] for d in train_data]).astype(np.float32)
    y_train_raw = np.array([d["pkd"] for d in train_data], dtype=np.float32)
    X_val = np.stack([d["z_mean"] for d in val_data]).astype(np.float32)
    y_val = np.array([d["pkd"] for d in val_data], dtype=np.float32)
    X_test = np.stack([d["z_mean"] for d in test_data]).astype(np.float32)
    y_test = np.array([d["pkd"] for d in test_data], dtype=np.float32)

    logger.info(f"  Train pKd: [{y_train_raw.min():.2f}, {y_train_raw.max():.2f}], mean={y_train_raw.mean():.2f}")
    logger.info(f"  Val pKd:   [{y_val.min():.2f}, {y_val.max():.2f}], mean={y_val.mean():.2f}")
    logger.info(f"  Test pKd:  [{y_test.min():.2f}, {y_test.max():.2f}], mean={y_test.mean():.2f}")

    # Augment training set
    if args.augment_to > 0:
        X_train, y_train = augment_data(X_train_raw, y_train_raw, target_n=args.augment_to, seed=args.seed)
        logger.info(f"  Augmented train: {len(X_train_raw)} → {len(X_train)} samples")
    else:
        X_train, y_train = X_train_raw, y_train_raw

    # ── Train GP ─────────────────────────────────────────────
    logger.info(f"\nTraining SVGP (d={X_train.shape[1]}, J={args.n_inducing}, epochs={args.n_epochs})...")
    t0 = time.time()
    model, likelihood, history = train_gp_with_validation(
        X_train, y_train, X_val, y_val,
        n_inducing=args.n_inducing, n_epochs=args.n_epochs,
        batch_size=args.batch_size, lr=args.lr, device=device,
    )
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # ── Evaluate on all splits ───────────────────────────────
    logger.info("\nEvaluating on all splits...")
    eval_train = evaluate_split(model, likelihood, X_train_raw, y_train_raw, device)
    eval_val = evaluate_split(model, likelihood, X_val, y_val, device)
    eval_test = evaluate_split(model, likelihood, X_test, y_test, device)
    # Also evaluate on ALL labeled data
    X_all = np.vstack([X_train_raw, X_val, X_test])
    y_all = np.concatenate([y_train_raw, y_val, y_test])
    eval_all = evaluate_split(model, likelihood, X_all, y_all, device)

    for name, ev in [("Train", eval_train), ("Val", eval_val), ("Test", eval_test), ("All", eval_all)]:
        logger.info(
            f"  {name:>5s} (N={ev['n']:2d}): RMSE={ev['rmse']:.3f}, "
            f"R²={ev['r2']:.3f}, ρ={ev['spearman_rho']:.3f}, "
            f"NLL={ev['nll']:.3f}, CI95%={ev['coverage_95']:.0%}"
        )

    # ── Save results ─────────────────────────────────────────
    # Training history
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Split info
    split_info = {
        "train_pockets": [d["name"] for d in train_data],
        "val_pockets": [d["name"] for d in val_data],
        "test_pockets": [d["name"] for d in test_data],
        "train_pkd": y_train_raw.tolist(),
        "val_pkd": y_val.tolist(),
        "test_pkd": y_test.tolist(),
    }
    with open(out / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Evaluation metrics
    eval_summary = {
        "train": {k: v for k, v in eval_train.items() if k not in ("mu", "var", "sigma")},
        "val": {k: v for k, v in eval_val.items() if k not in ("mu", "var", "sigma")},
        "test": {k: v for k, v in eval_test.items() if k not in ("mu", "var", "sigma")},
        "all": {k: v for k, v in eval_all.items() if k not in ("mu", "var", "sigma")},
        "training_time_s": round(elapsed, 1),
        "n_epochs": args.n_epochs,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss_epoch": int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss": float(min(history["val_loss"])),
    }
    with open(out / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "d": X_train.shape[1],
        "n_inducing": min(args.n_inducing, len(X_train)),
    }, out / "gp_model.pt")

    # ══════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════
    logger.info("\nGenerating visualizations...")
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    epochs = np.arange(1, args.n_epochs + 1)

    # ── Figure 1: Training Curves (2×2) ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GP (SVGP) Training Curves", fontsize=14, fontweight="bold")

    # (0,0) ELBO Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train ELBO Loss", color="#3498DB", lw=1.5)
    ax.plot(epochs, history["val_loss"], label="Val ELBO Loss", color="#E74C3C", lw=1.5)
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, ls="--", color="gray", alpha=0.5, label=f"Best val (epoch {best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Negative ELBO")
    ax.set_title("Variational ELBO Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # (0,1) RMSE
    ax = axes[0, 1]
    ax.plot(epochs, history["train_rmse"], label="Train RMSE", color="#3498DB", lw=1.5)
    ax.plot(epochs, history["val_rmse"], label="Val RMSE", color="#E74C3C", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (pKd)")
    ax.set_title("Root Mean Squared Error")
    ax.legend(); ax.grid(alpha=0.3)

    # (1,0) NLL
    ax = axes[1, 0]
    ax.plot(epochs, history["train_nll"], label="Train NLL", color="#3498DB", lw=1.5)
    ax.plot(epochs, history["val_nll"], label="Val NLL", color="#E74C3C", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("NLL")
    ax.set_title("Negative Log-Likelihood")
    ax.legend(); ax.grid(alpha=0.3)

    # (1,1) Hyperparameters
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(epochs, history["noise"], label="Noise σ²", color="#E67E22", lw=1.5)
    l2, = ax2.plot(epochs, history["lengthscale_mean"], label="Mean lengthscale", color="#2ECC71", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Noise variance", color="#E67E22")
    ax2.set_ylabel("Mean lengthscale", color="#2ECC71")
    ax.set_title("Kernel Hyperparameters")
    ax.legend(handles=[l1, l2], loc="center right"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_dir / "training_curves.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'training_curves.png'}")
    plt.close()

    # ── Figure 2: Pred vs True on Train/Val/Test (1×3) ───────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GP Predictions: Train / Val / Test", fontsize=14, fontweight="bold")

    splits = [
        ("Train", X_train_raw, y_train_raw, train_data, eval_train),
        ("Val", X_val, y_val, val_data, eval_val),
        ("Test", X_test, y_test, test_data, eval_test),
    ]

    for ax, (name, X, y, data, ev) in zip(axes, splits):
        mu = ev["mu"]
        sigma = ev["sigma"]
        ax.errorbar(y, mu, yerr=1.96 * sigma, fmt="o", markersize=8, capsize=4,
                     color="#3498DB", ecolor="#BDC3C7", alpha=0.8, zorder=3)
        # Label each point
        for i, d in enumerate(data):
            ax.annotate(d["name"][:12], (y[i], mu[i]), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

        lims = [min(y.min(), mu.min()) - 0.5, max(y.max(), mu.max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect")
        ax.fill_between(lims, [l - 1 for l in lims], [l + 1 for l in lims], alpha=0.05, color="gray")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True pKd"); ax.set_ylabel("Predicted μ")
        ax.set_title(f"{name} (N={ev['n']})\n"
                     f"RMSE={ev['rmse']:.2f}, R²={ev['r2']:.2f}, ρ={ev['spearman_rho']:.2f}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig_dir / "pred_vs_true_splits.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'pred_vs_true_splits.png'}")
    plt.close()

    # ── Figure 3: Residuals + CI coverage ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Prediction Quality Analysis", fontsize=14, fontweight="bold")

    # (0,0) Residual histogram per split
    ax = axes[0, 0]
    for name, _, y, _, ev in splits:
        residuals = ev["mu"] - y
        ax.hist(residuals, bins=10, alpha=0.5, label=f"{name} (MAE={ev['mae']:.2f})")
    ax.axvline(0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("Residual (μ - pKd)"); ax.set_ylabel("Count")
    ax.set_title("Residual Distribution"); ax.legend()

    # (0,1) Uncertainty calibration (predicted σ vs actual error)
    ax = axes[0, 1]
    for name, _, y, _, ev in splits:
        actual_err = np.abs(ev["mu"] - y)
        pred_sigma = ev["sigma"]
        ax.scatter(pred_sigma, actual_err, s=50, alpha=0.7, label=f"{name}")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="Perfect cal.")
    ax.plot([0, lim], [0, 1.96 * lim], "r:", alpha=0.3, label="1.96σ bound")
    ax.set_xlabel("Predicted σ"); ax.set_ylabel("|Actual Error|")
    ax.set_title("Uncertainty Calibration"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (1,0) Metrics comparison bar chart
    ax = axes[1, 0]
    metric_names = ["RMSE", "MAE", "NLL", "R²", "Spearman ρ", "CI 95%\nCoverage"]
    x = np.arange(len(metric_names))
    width = 0.25
    for i, (name, _, _, _, ev) in enumerate(splits):
        vals = [ev["rmse"], ev["mae"], ev["nll"], ev["r2"], ev["spearman_rho"], ev["coverage_95"]]
        bars = ax.bar(x + i * width, vals, width, label=name, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x + width); ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylabel("Value"); ax.set_title("Split Comparison"); ax.legend()

    # (1,1) Confidence interval visualization (all data)
    ax = axes[1, 1]
    all_data_sorted = sorted(
        zip(y_all, eval_all["mu"], eval_all["sigma"],
            ["Train"]*len(train_data) + ["Val"]*len(val_data) + ["Test"]*len(test_data)),
        key=lambda x: x[0]
    )
    y_s = [d[0] for d in all_data_sorted]
    mu_s = [d[1] for d in all_data_sorted]
    sig_s = [d[2] for d in all_data_sorted]
    split_s = [d[3] for d in all_data_sorted]
    colors = {"Train": "#3498DB", "Val": "#E67E22", "Test": "#2ECC71"}
    for i in range(len(y_s)):
        c = colors[split_s[i]]
        ax.errorbar(i, mu_s[i], yerr=1.96*sig_s[i], fmt="o", color=c, markersize=5,
                     capsize=3, ecolor=c, alpha=0.6)
    ax.scatter(range(len(y_s)), y_s, marker="x", color="black", s=30, zorder=5, label="True pKd")
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[s], label=s) for s in ["Train", "Val", "Test"]]
    handles.append(plt.Line2D([0],[0], marker="x", color="black", ls="", label="True pKd"))
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel("Pocket (sorted by pKd)"); ax.set_ylabel("pKd")
    ax.set_title(f"95% CI Coverage = {eval_all['coverage_95']:.0%}")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_dir / "prediction_analysis.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'prediction_analysis.png'}")
    plt.close()

    # ── Figure 4: GP landscape + data distribution ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("GP Model Analysis", fontsize=14, fontweight="bold")

    # (0) PCA 2D projection with GP predictions
    from sklearn.decomposition import PCA
    X_all_emb = np.vstack([X_train_raw, X_val, X_test])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all_emb)
    split_labels = ["Train"]*len(train_data) + ["Val"]*len(val_data) + ["Test"]*len(test_data)

    ax = axes[0]
    for sname, marker, alpha in [("Train", "o", 0.8), ("Val", "s", 0.9), ("Test", "D", 0.9)]:
        mask = [s == sname for s in split_labels]
        sc = ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=y_all[mask],
                        cmap="RdYlGn", marker=marker, s=80, alpha=alpha,
                        edgecolors="gray", linewidths=0.5, label=sname,
                        vmin=y_all.min(), vmax=y_all.max())
    plt.colorbar(sc, ax=ax, label="pKd")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of Embeddings (colored by pKd)")
    ax.legend()

    # (1) pKd distribution per split
    ax = axes[1]
    split_colors = {"Train": "#3498DB", "Val": "#E67E22", "Test": "#2ECC71"}
    for sname, y_split in [("Train", y_train_raw), ("Val", y_val), ("Test", y_test)]:
        ax.hist(y_split, bins=8, alpha=0.5, color=split_colors[sname], label=f"{sname} (N={len(y_split)})")
        ax.axvline(y_split.mean(), color=split_colors[sname], ls="--", lw=1.5, alpha=0.8)
    ax.set_xlabel("pKd"); ax.set_ylabel("Count")
    ax.set_title("pKd Distribution per Split")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig_dir / "gp_model_analysis.png", bbox_inches="tight")
    logger.info(f"  Saved {fig_dir / 'gp_model_analysis.png'}")
    plt.close()

    # ── Summary ──────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("GP Training Analysis Complete!")
    logger.info(f"  Output:  {out}")
    logger.info(f"  Figures: {fig_dir}")
    logger.info(f"{'='*60}")
    logger.info(f"\n  Train: RMSE={eval_train['rmse']:.3f}, R²={eval_train['r2']:.3f}, ρ={eval_train['spearman_rho']:.3f}")
    logger.info(f"  Val:   RMSE={eval_val['rmse']:.3f}, R²={eval_val['r2']:.3f}, ρ={eval_val['spearman_rho']:.3f}")
    logger.info(f"  Test:  RMSE={eval_test['rmse']:.3f}, R²={eval_test['r2']:.3f}, ρ={eval_test['spearman_rho']:.3f}")
    logger.info(f"  All:   RMSE={eval_all['rmse']:.3f}, R²={eval_all['r2']:.3f}, ρ={eval_all['spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
