"""
scripts/10_merge_and_train_eval.py
──────────────────────────────────
Merge sharded 1000-step embeddings, train GP with train/val split,
and generate comprehensive visualizations including training curves.

Steps (selected via --step):
  merge     — Merge embeddings from 4 shard directories
  train     — GP training with train/val split + training curve viz
  visualize — Generate all figures including GP training diagnostics

Usage:
    python scripts/10_merge_and_train_eval.py --step merge --output_dir results/embedding_1000step/merged
    python scripts/10_merge_and_train_eval.py --step train --output_dir results/embedding_1000step/merged
    python scripts/10_merge_and_train_eval.py --step visualize --output_dir results/embedding_1000step/merged
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Shard directories from the 1000-step HPC run
SHARD_DIRS = [
    "results/embedding_1000step/20260305_085825_j3387783/shards/shard_0of4",
    "results/embedding_1000step/20260305_085827_j3387783/shards/shard_1of4",
    "results/embedding_1000step/20260305_085834_j3387783/shards/shard_2of4",
    "results/embedding_1000step/20260305_085839_j3387783/shards/shard_3of4",
]


# ═══════════════════════════════════════════════════════════════
# Step: MERGE
# ═══════════════════════════════════════════════════════════════

def step_merge(output_dir: Path):
    """Merge embeddings from all 4 shard directories."""
    logger.info("=" * 60)
    logger.info("MERGE: Collecting embeddings from 4 shards")
    logger.info("=" * 60)

    all_embeddings = {}
    for sd in SHARD_DIRS:
        sd = Path(sd)
        if not sd.exists():
            logger.warning(f"  Shard dir not found: {sd}")
            continue
        for pocket_dir in sorted(sd.iterdir()):
            if not pocket_dir.is_dir():
                continue
            emb_files = list(pocket_dir.glob("*_embeddings.npy"))
            if emb_files:
                emb = np.load(emb_files[0])
                all_embeddings[pocket_dir.name] = emb
                logger.debug(f"  {pocket_dir.name}: {emb.shape}")

    # Save merged
    out_path = output_dir / "all_embeddings.npz"
    np.savez(out_path, **all_embeddings)

    total_mols = sum(v.shape[0] for v in all_embeddings.values())
    dim = list(all_embeddings.values())[0].shape[1] if all_embeddings else 0

    summary = {
        "n_pockets": len(all_embeddings),
        "total_molecules": total_mols,
        "embedding_dim": dim,
        "samples_per_pocket": list(all_embeddings.values())[0].shape[0] if all_embeddings else 0,
        "pockets": sorted(all_embeddings.keys()),
    }
    with open(output_dir / "merge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Merged {len(all_embeddings)} pockets, {total_mols} molecules, dim={dim}")
    logger.info(f"  Saved to {out_path}")
    return all_embeddings


# ═══════════════════════════════════════════════════════════════
# Step: TRAIN (with train/val split)
# ═══════════════════════════════════════════════════════════════

def load_affinity_labels(affinity_pkl: str) -> dict[str, float]:
    """Load pocket_family -> mean pKd from affinity_info.pkl."""
    with open(affinity_pkl, "rb") as f:
        affinity = pickle.load(f)
    pocket_pks: dict[str, list[float]] = {}
    for key, info in affinity.items():
        pk = info.get("pk")
        if pk is None or float(pk) == 0.0:
            continue
        pocket_fam = str(key).split("/")[0]
        pocket_pks.setdefault(pocket_fam, []).append(float(pk))
    return {fam: float(np.mean(vals)) for fam, vals in pocket_pks.items()}


def step_train(output_dir: Path, affinity_pkl: str, n_epochs: int = 300,
               n_inducing: int = 128, val_fraction: float = 0.2,
               batch_size: int = 64, lr: float = 0.01, device: str = "cuda",
               augment_to: int = 200, seed: int = 42):
    """Train GP oracle with train/val split, save training curves."""
    import torch
    from bayesdiff.gp_oracle import GPOracle

    logger.info("=" * 60)
    logger.info("TRAIN: GP Oracle with Train/Val Split")
    logger.info("=" * 60)

    gp_dir = output_dir / "gp_model"
    gp_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    emb_path = output_dir / "all_embeddings.npz"
    data = np.load(emb_path, allow_pickle=True)
    embeddings_dict = {k: data[k] for k in data.files}
    logger.info(f"  Loaded {len(embeddings_dict)} pockets")

    # Load labels
    label_map = load_affinity_labels(affinity_pkl)
    logger.info(f"  Labels: {len(label_map)} pocket families with pKd")

    # Build dataset: mean embedding per pocket + pKd
    X_list, y_list, codes = [], [], []
    for pdb_code, emb in embeddings_dict.items():
        pk = label_map.get(pdb_code)
        if pk is None:
            # Try prefix match
            pdb_base = pdb_code.split("_")[0] if "_" in pdb_code else pdb_code
            pk = label_map.get(pdb_base)
        if pk is None:
            continue
        X_list.append(emb.mean(axis=0))
        y_list.append(pk)
        codes.append(pdb_code)

    if not X_list:
        logger.error("No matching embeddings+labels found")
        sys.exit(1)

    X_all = np.stack(X_list).astype(np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    logger.info(f"  Matched {len(X_all)} pockets with labels")
    logger.info(f"  pKd range: [{y_all.min():.2f}, {y_all.max():.2f}], mean={y_all.mean():.2f}")

    # Train/val split (stratified by pKd median)
    rng = np.random.default_rng(seed)
    n = len(X_all)
    n_val = max(3, int(n * val_fraction))
    n_train = n - n_val

    # Stratified: sort by pKd, interleave train/val
    sort_idx = np.argsort(y_all)
    val_mask = np.zeros(n, dtype=bool)
    # Pick every ~(1/val_fraction)-th sample for validation
    step = max(1, int(1.0 / val_fraction))
    for i in range(0, n, step):
        if val_mask.sum() < n_val:
            val_mask[sort_idx[i]] = True

    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    rng.shuffle(train_idx)

    X_train_raw = X_all[train_idx]
    y_train_raw = y_all[train_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]
    codes_train = [codes[i] for i in train_idx]
    codes_val = [codes[i] for i in val_idx]

    logger.info(f"  Train: {len(X_train_raw)} pockets, Val: {len(X_val)} pockets")
    logger.info(f"  Train pKd: [{y_train_raw.min():.2f}, {y_train_raw.max():.2f}]")
    logger.info(f"  Val pKd:   [{y_val.min():.2f}, {y_val.max():.2f}]")

    # Augment training set if small
    X_train = X_train_raw.copy()
    y_train = y_train_raw.copy()
    if augment_to > 0 and len(X_train) < augment_to:
        n_aug = augment_to - len(X_train)
        d = X_train.shape[1]
        X_aug, y_aug = [X_train], [y_train]
        for _ in range(n_aug):
            idx = rng.integers(len(X_train_raw))
            x_new = X_train_raw[idx] + rng.standard_normal(d).astype(np.float32) * 0.3
            y_new = y_train_raw[idx] + rng.standard_normal() * 0.5
            X_aug.append(x_new.reshape(1, -1))
            y_aug.append(np.array([y_new], dtype=np.float32))
        X_train = np.concatenate(X_aug)
        y_train = np.concatenate(y_aug)
        logger.info(f"  Augmented train: {len(X_train)} samples")

    # Train GP with epoch-level train and val loss tracking
    d = X_train.shape[1]
    n_ind = min(n_inducing, len(X_train))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"\n  Training SVGP (d={d}, J={n_ind}, epochs={n_epochs}, device={device})")

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    N = X_t.shape[0]

    # Initialize GP
    import gpytorch
    from bayesdiff.gp_oracle import SVGPModel, GPOracle

    if N <= n_ind:
        inducing_points = X_t.clone()
    else:
        idx = torch.randperm(N)[:n_ind]
        inducing_points = X_t[idx].clone()

    model = SVGPModel(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

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
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_nll": [],
        "noise": [],
        "train_rmse": [],
    }

    t0 = time.time()
    for epoch in range(n_epochs):
        # Train
        model.train()
        likelihood.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= N
        history["train_loss"].append(epoch_loss)
        history["noise"].append(likelihood.noise.item())

        # Validate
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Train RMSE (on raw, non-augmented data)
            X_train_raw_t = torch.tensor(X_train_raw, dtype=torch.float32, device=device)
            y_train_raw_t = torch.tensor(y_train_raw, dtype=torch.float32, device=device)
            train_pred = likelihood(model(X_train_raw_t))
            train_rmse = (train_pred.mean - y_train_raw_t).pow(2).mean().sqrt().item()
            history["train_rmse"].append(train_rmse)

            # Val metrics
            val_pred = likelihood(model(X_val_t))
            val_mu = val_pred.mean
            val_var = val_pred.variance

            val_rmse = (val_mu - y_val_t).pow(2).mean().sqrt().item()
            history["val_rmse"].append(val_rmse)

            # NLL on val
            val_nll = -torch.distributions.Normal(val_mu, val_var.sqrt()).log_prob(y_val_t).mean().item()
            history["val_nll"].append(val_nll)

            # Val ELBO (approximate)
            val_output = model(X_val_t)
            val_loss = -mll(val_output, y_val_t).item()
            history["val_loss"].append(val_loss)

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch+1:>4d}/{n_epochs}: "
                f"train_loss={epoch_loss:.4f}, "
                f"train_rmse={train_rmse:.3f}, "
                f"val_rmse={val_rmse:.3f}, "
                f"val_nll={val_nll:.3f}, "
                f"noise={likelihood.noise.item():.4f}"
            )

    elapsed = time.time() - t0
    logger.info(f"\n  Training done in {elapsed:.1f}s")

    # Final predictions on train and val
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        train_final = likelihood(model(torch.tensor(X_train_raw, dtype=torch.float32, device=device)))
        val_final = likelihood(model(X_val_t))

    train_mu = train_final.mean.cpu().numpy()
    train_var = train_final.variance.cpu().numpy()
    val_mu_np = val_final.mean.cpu().numpy()
    val_var_np = val_final.variance.cpu().numpy()

    # Save GP model via GPOracle wrapper
    gp = GPOracle(d=d, n_inducing=n_ind, device=device)
    gp.model = model
    gp.likelihood = likelihood
    gp.save(gp_dir / "gp_model.pt")
    logger.info(f"  Model saved to {gp_dir / 'gp_model.pt'}")

    # Save training data (full augmented for OOD detector)
    np.savez(gp_dir / "train_data.npz", X=X_train, y=y_train)

    # Save training history
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(gp_dir / "training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)

    # Save train/val predictions for visualization
    np.savez(gp_dir / "predictions.npz",
             train_mu=train_mu, train_var=train_var, train_y=y_train_raw,
             train_codes=np.array(codes_train),
             val_mu=val_mu_np, val_var=val_var_np, val_y=y_val,
             val_codes=np.array(codes_val))

    # Save metadata
    meta = {
        "n_train_raw": int(len(X_train_raw)),
        "n_train_augmented": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_total_labeled": int(len(X_all)),
        "n_total_pockets": int(len(embeddings_dict)),
        "d": int(d),
        "n_inducing": int(n_ind),
        "n_epochs": int(n_epochs),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_train_rmse": float(history["train_rmse"][-1]),
        "final_val_rmse": float(history["val_rmse"][-1]),
        "final_val_nll": float(history["val_nll"][-1]),
        "final_noise": float(history["noise"][-1]),
        "elapsed_s": round(elapsed, 1),
        "pkd_range": [float(y_all.min()), float(y_all.max())],
        "pkd_mean": float(y_all.mean()),
        "val_fraction": val_fraction,
        "augment_to": augment_to,
        "device": device,
        "train_codes": codes_train,
        "val_codes": codes_val,
        # Compat with 09_generate_figures.py
        "n_train": int(len(X_train)),
        "final_loss": float(history["train_loss"][-1]),
    }
    with open(gp_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  Final train RMSE: {history['train_rmse'][-1]:.3f}")
    logger.info(f"  Final val RMSE:   {history['val_rmse'][-1]:.3f}")
    logger.info(f"  Final val NLL:    {history['val_nll'][-1]:.3f}")


# ═══════════════════════════════════════════════════════════════
# Step: VISUALIZE
# ═══════════════════════════════════════════════════════════════

def step_visualize(output_dir: Path):
    """Generate comprehensive figures including GP training diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gp_dir = output_dir / "gp_model"

    logger.info("=" * 60)
    logger.info("VISUALIZE: Generating comprehensive figures")
    logger.info("=" * 60)

    # Load data
    history = json.load(open(gp_dir / "training_history.json"))
    meta = json.load(open(gp_dir / "train_meta.json"))
    preds = np.load(gp_dir / "predictions.npz", allow_pickle=True)

    train_mu = preds["train_mu"]
    train_var = preds["train_var"]
    train_y = preds["train_y"]
    train_codes = preds["train_codes"]
    val_mu = preds["val_mu"]
    val_var = preds["val_var"]
    val_y = preds["val_y"]
    val_codes = preds["val_codes"]

    # ── Figure A: GP Training Curves (4-panel) ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GP Oracle Training Diagnostics\n"
                 f"(d={meta['d']}, J={meta['n_inducing']}, "
                 f"Train={meta['n_train_raw']} raw + {meta['n_train_augmented']-meta['n_train_raw']} aug, "
                 f"Val={meta['n_val']}, {meta['n_epochs']} epochs, {meta['elapsed_s']}s)",
                 fontsize=13, fontweight="bold")

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # A1: ELBO Loss (train vs val)
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2196F3", linewidth=1.5, label="Train ELBO Loss")
    ax.plot(epochs, history["val_loss"], color="#F44336", linewidth=1.5, label="Val ELBO Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative ELBO")
    ax.set_title("(A) ELBO Loss: Train vs Validation")
    ax.legend()

    # A2: RMSE (train vs val)
    ax = axes[0, 1]
    ax.plot(epochs, history["train_rmse"], color="#2196F3", linewidth=1.5, label=f"Train RMSE (final: {history['train_rmse'][-1]:.3f})")
    ax.plot(epochs, history["val_rmse"], color="#F44336", linewidth=1.5, label=f"Val RMSE (final: {history['val_rmse'][-1]:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (pKd)")
    ax.set_title("(B) RMSE: Train vs Validation")
    ax.legend()

    # A3: Val NLL over time
    ax = axes[1, 0]
    ax.plot(epochs, history["val_nll"], color="#9C27B0", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title(f"(C) Validation NLL (final: {history['val_nll'][-1]:.3f})")

    # A4: Noise parameter over time
    ax = axes[1, 1]
    ax.plot(epochs, history["noise"], color="#FF9800", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learned Noise σ²")
    ax.set_title(f"(D) Likelihood Noise (final: {history['noise'][-1]:.4f})")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = fig_dir / "fig_gp_training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # ── Figure B: Train/Val Predictions (2-panel) ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GP Oracle: Predicted vs True pKd", fontsize=13, fontweight="bold")

    for idx, (mu, var, y, codes_arr, title) in enumerate([
        (train_mu, train_var, train_y, train_codes, "Training Set"),
        (val_mu, val_var, val_y, val_codes, "Validation Set"),
    ]):
        ax = axes[idx]
        sigma = np.sqrt(var)
        rmse = np.sqrt(np.mean((mu - y) ** 2))
        corr = np.corrcoef(mu, y)[0, 1] if len(y) > 2 else 0

        # Error bars (2σ)
        ax.errorbar(y, mu, yerr=2 * sigma, fmt="o", capsize=3,
                    color="#2196F3" if idx == 0 else "#F44336",
                    ecolor="#90CAF9" if idx == 0 else "#FFCDD2",
                    markersize=6, linewidth=1, alpha=0.8,
                    label=f"μ ± 2σ (n={len(y)})")

        # y=x line
        lims = [min(y.min(), mu.min()) - 0.5, max(y.max(), mu.max()) + 0.5]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")

        # Annotate points
        for i, code in enumerate(codes_arr):
            short = str(code).split("_")[0]
            ax.annotate(short, (y[i], mu[i]), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel("True pKd")
        ax.set_ylabel("Predicted pKd")
        ax.set_title(f"({chr(65+idx)}) {title}\nRMSE={rmse:.3f}, r={corr:.3f}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    path = fig_dir / "fig_gp_predictions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # ── Figure C: Residual Analysis (train + val) ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GP Oracle: Residual Analysis", fontsize=13, fontweight="bold")

    all_mu = np.concatenate([train_mu, val_mu])
    all_y = np.concatenate([train_y, val_y])
    all_var = np.concatenate([train_var, val_var])
    all_residuals = all_mu - all_y
    all_sigma = np.sqrt(all_var)
    is_train = np.array([True] * len(train_mu) + [False] * len(val_mu))

    # C1: Residuals vs predicted
    ax = axes[0]
    ax.scatter(all_mu[is_train], all_residuals[is_train], c="#2196F3", alpha=0.6, s=40, label="Train", edgecolors="white")
    ax.scatter(all_mu[~is_train], all_residuals[~is_train], c="#F44336", alpha=0.8, s=60, label="Val", edgecolors="white", zorder=3)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Predicted pKd")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("(A) Residuals vs Predicted")
    ax.legend()

    # C2: Residuals vs uncertainty
    ax = axes[1]
    ax.scatter(all_sigma[is_train], np.abs(all_residuals[is_train]), c="#2196F3", alpha=0.6, s=40, label="Train")
    ax.scatter(all_sigma[~is_train], np.abs(all_residuals[~is_train]), c="#F44336", alpha=0.8, s=60, label="Val", zorder=3)
    # Add y=2σ reference line
    sigma_range = np.linspace(0, all_sigma.max(), 50)
    ax.plot(sigma_range, 2 * sigma_range, "--", color="gray", alpha=0.5, label="2σ boundary")
    ax.set_xlabel("σ (Predictive Uncertainty)")
    ax.set_ylabel("|Residual|")
    ax.set_title("(B) Calibration Check: |Error| vs σ")
    ax.legend()

    # C3: Residual histogram
    ax = axes[2]
    ax.hist(all_residuals[is_train], bins=15, alpha=0.6, color="#2196F3", label="Train", edgecolor="white")
    ax.hist(all_residuals[~is_train], bins=8, alpha=0.7, color="#F44336", label="Val", edgecolor="white")
    ax.axvline(x=0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Residual (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title(f"(C) Residual Distribution\nMean: {all_residuals.mean():.3f}, Std: {all_residuals.std():.3f}")
    ax.legend()

    plt.tight_layout()
    path = fig_dir / "fig_gp_residuals.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # ── Figure D: Uncertainty Calibration Scatter ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Uncertainty Calibration: Coverage Analysis", fontsize=13, fontweight="bold")

    for idx, (mu, var, y, title) in enumerate([
        (train_mu, train_var, train_y, "Training Set"),
        (val_mu, val_var, val_y, "Validation Set"),
    ]):
        ax = axes[idx]
        sigma = np.sqrt(var)
        z_scores = (mu - y) / np.clip(sigma, 1e-6, None)

        # Expected coverage: what fraction of points fall within k-sigma
        k_vals = np.linspace(0.5, 3.0, 20)
        from scipy import stats as scipy_stats
        expected_coverage = [2 * scipy_stats.norm.cdf(k) - 1 for k in k_vals]
        observed_coverage = [np.mean(np.abs(z_scores) <= k) for k in k_vals]

        ax.plot(expected_coverage, observed_coverage, "o-",
                color="#2196F3" if idx == 0 else "#F44336",
                markersize=5, linewidth=1.5, label=f"{title} (n={len(y)})")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(f"({chr(65+idx)}) {title}")
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = fig_dir / "fig_gp_calibration.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # ── Figure E: Embedding PCA colored by split + pKd ────────
    emb_path = output_dir / "all_embeddings.npz"
    if emb_path.exists():
        from sklearn.decomposition import PCA

        emb_data = np.load(emb_path, allow_pickle=True)
        label_map = {}
        aff_pkl = Path("external/targetdiff/data/affinity_info.pkl")
        if aff_pkl.exists():
            label_map = load_affinity_labels(str(aff_pkl))

        # Collect mean embeddings for all pockets
        pocket_embs, pocket_pkd, pocket_split = [], [], []
        train_set = set(str(c) for c in train_codes)
        val_set = set(str(c) for c in val_codes)

        for key in emb_data.files:
            emb = emb_data[key]
            pocket_embs.append(emb.mean(axis=0))
            pk = label_map.get(key)
            if pk is None:
                pdb_base = key.split("_")[0]
                pk = label_map.get(pdb_base)
            pocket_pkd.append(pk if pk is not None else np.nan)
            if key in train_set:
                pocket_split.append("train")
            elif key in val_set:
                pocket_split.append("val")
            else:
                pocket_split.append("unlabeled")

        Z = np.stack(pocket_embs)
        pkd = np.array(pocket_pkd)
        splits = np.array(pocket_split)

        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Z)
        ev = pca.explained_variance_ratio_

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Pocket Embedding Space (88 pockets, PCA of mean embeddings)",
                     fontsize=13, fontweight="bold")

        # Panel A: Colored by pKd
        ax = axes[0]
        has_pk = ~np.isnan(pkd)
        sc = ax.scatter(Z_pca[has_pk, 0], Z_pca[has_pk, 1], c=pkd[has_pk],
                        cmap="RdYlGn", s=60, alpha=0.8, edgecolors="white", zorder=3)
        ax.scatter(Z_pca[~has_pk, 0], Z_pca[~has_pk, 1], c="gray", s=30,
                   alpha=0.4, marker="x", label=f"No label (n={int((~has_pk).sum())})")
        plt.colorbar(sc, ax=ax, label="pKd", shrink=0.8)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title("(A) Colored by pKd")
        ax.legend()

        # Panel B: Colored by train/val/unlabeled split
        ax = axes[1]
        for split, color, marker, label in [
            ("train", "#2196F3", "o", f"Train (n={int((splits=='train').sum())})"),
            ("val", "#F44336", "s", f"Val (n={int((splits=='val').sum())})"),
            ("unlabeled", "#9E9E9E", "x", f"Unlabeled (n={int((splits=='unlabeled').sum())})"),
        ]:
            mask = splits == split
            ax.scatter(Z_pca[mask, 0], Z_pca[mask, 1], c=color, marker=marker,
                       s=60 if split != "unlabeled" else 30,
                       alpha=0.8 if split != "unlabeled" else 0.4,
                       edgecolors="white" if split != "unlabeled" else "none",
                       label=label, zorder=3 if split != "unlabeled" else 1)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title("(B) Colored by Train/Val Split")
        ax.legend()

        plt.tight_layout()
        path = fig_dir / "fig_embedding_space.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {path}")

    logger.info(f"\n  All custom figures saved to {fig_dir}/")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BayesDiff: Merge + Train + Visualize")
    parser.add_argument("--step", type=str, required=True,
                        choices=["merge", "train", "visualize", "all"],
                        help="Which step to run")
    parser.add_argument("--output_dir", type=str, default="results/embedding_1000step/merged")
    parser.add_argument("--affinity_pkl", type=str, default="external/targetdiff/data/affinity_info.pkl")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_inducing", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--augment_to", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step in ("merge", "all"):
        step_merge(output_dir)

    if args.step in ("train", "all"):
        step_train(
            output_dir, args.affinity_pkl,
            n_epochs=args.n_epochs, n_inducing=args.n_inducing,
            val_fraction=args.val_fraction, batch_size=args.batch_size,
            lr=args.lr, device=args.device, augment_to=args.augment_to,
            seed=args.seed,
        )

    if args.step in ("visualize", "all"):
        step_visualize(output_dir)


if __name__ == "__main__":
    main()
