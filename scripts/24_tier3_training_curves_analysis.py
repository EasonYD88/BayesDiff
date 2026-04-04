#!/usr/bin/env python
"""
scripts/24_tier3_training_curves_analysis.py
─────────────────────────────────────────────
Tier 3 数据的全面训练曲线 + 数据分析可视化。

生成图表：
  1. Training curves（FCFP4 vs Encoder）— epoch-by-epoch loss
  2. Train/Val/Test learning curves — 随训练epoch变化的三集指标
  3. 数据分布分析 — pKd 分布、embedding PCA、分子数统计
  4. Train/Val/Test 散点图 + 残差分析
  5. 模型对比 dashboard — FCFP4 vs Encoder 在 T/V/T 上的表现
  6. Calibration + Uncertainty 分析

使用 Tier 3 数据 (N≈932)，同时训练 FCFP4 和 Encoder 表征的 GP。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import gpytorch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "results" / "tier3_gp"
FIG_DIR = DATA_DIR / "figures" / "tier3_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# GP Model
# ═══════════════════════════════════════════════════════════════
class FlexGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gp_with_val(X_tr, y_tr, X_va, y_va, X_te, y_te,
                      n_epochs=200, lr=0.1, noise_lb=0.001, eval_every=5):
    """Train GP and record train/val/test metrics at each eval_every epochs."""
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_v = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    X_e = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
    )
    model = FlexGP(X_t, y_t, lik).to(DEVICE)
    lik = lik.to(DEVICE)

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    history = {
        "epoch": [], "train_loss": [],
        "train_rmse": [], "train_rho": [], "train_r2": [],
        "val_rmse": [], "val_rho": [], "val_r2": [],
        "test_rmse": [], "test_rho": [], "test_r2": [],
        "noise": [], "outputscale": [],
    }

    for ep in range(n_epochs):
        model.train(); lik.train()
        opt.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        opt.step()

        if ep % eval_every == 0 or ep == n_epochs - 1:
            model.eval(); lik.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                tr_p = lik(model(X_t)).mean.cpu().numpy()
                va_p = lik(model(X_v)).mean.cpu().numpy()
                te_p = lik(model(X_e)).mean.cpu().numpy()

            history["epoch"].append(ep)
            history["train_loss"].append(loss.item())

            for prefix, yt, yp in [("train", y_tr, tr_p), ("val", y_va, va_p), ("test", y_te, te_p)]:
                rmse = np.sqrt(np.mean((yt - yp)**2))
                rho, _ = stats.spearmanr(yt, yp) if len(yt) > 2 else (0, 1)
                ss_res = np.sum((yt - yp)**2)
                ss_tot = np.sum((yt - yt.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                history[f"{prefix}_rmse"].append(rmse)
                history[f"{prefix}_rho"].append(rho)
                history[f"{prefix}_r2"].append(r2)

            history["noise"].append(lik.noise.item())
            history["outputscale"].append(model.covar_module.outputscale.item())

    # Final predictions with std
    model.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        tr_out = lik(model(X_t))
        va_out = lik(model(X_v))
        te_out = lik(model(X_e))

    preds = {
        "train": (tr_out.mean.cpu().numpy(), tr_out.stddev.cpu().numpy()),
        "val": (va_out.mean.cpu().numpy(), va_out.stddev.cpu().numpy()),
        "test": (te_out.mean.cpu().numpy(), te_out.stddev.cpu().numpy()),
    }

    return model, lik, history, preds


def met(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    rho, p = stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (0, 1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"rmse": rmse, "mae": mae, "rho": rho, "p_val": p, "r2": r2}


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
print("Loading Tier 3 data...")
X_fcfp_raw = np.load(DATA_DIR / "X_FCFP4_2048.npy")
y_fcfp = np.load(DATA_DIR / "y_pkd.npy")
with open(DATA_DIR / "families.json") as f:
    families_fcfp = json.load(f)

X_enc_raw = np.load(DATA_DIR / "X_encoder_128.npy")
y_enc = np.load(DATA_DIR / "y_pkd_encoder.npy")
mol_counts = np.load(DATA_DIR / "mol_counts.npy")
mol_counts_enc = np.load(DATA_DIR / "mol_counts_encoder.npy")

N_fcfp = len(y_fcfp)
N_enc = len(y_enc)
print(f"FCFP4 dataset: {N_fcfp} pockets, {X_fcfp_raw.shape[1]}-dim")
print(f"Encoder dataset: {N_enc} pockets, {X_enc_raw.shape[1]}-dim")
print(f"pKd range: [{y_fcfp.min():.2f}, {y_fcfp.max():.2f}], mean={y_fcfp.mean():.2f}±{y_fcfp.std():.2f}")

# Standardize
scaler_fcfp = StandardScaler()
X_fcfp = scaler_fcfp.fit_transform(X_fcfp_raw)
scaler_enc = StandardScaler()
X_enc = scaler_enc.fit_transform(X_enc_raw)


# ═══════════════════════════════════════════════════════════════
# Split data (60/20/20)
# ═══════════════════════════════════════════════════════════════
rng = np.random.RandomState(42)

# FCFP split
idx_f = rng.permutation(N_fcfp)
n_tr_f = int(N_fcfp * 0.6)
n_va_f = int(N_fcfp * 0.2)
tr_f, va_f, te_f = idx_f[:n_tr_f], idx_f[n_tr_f:n_tr_f+n_va_f], idx_f[n_tr_f+n_va_f:]

# Encoder split
rng2 = np.random.RandomState(42)
idx_e = rng2.permutation(N_enc)
n_tr_e = int(N_enc * 0.6)
n_va_e = int(N_enc * 0.2)
tr_e, va_e, te_e = idx_e[:n_tr_e], idx_e[n_tr_e:n_tr_e+n_va_e], idx_e[n_tr_e+n_va_e:]

print(f"\nFCFP4 split: Train={len(tr_f)}, Val={len(va_f)}, Test={len(te_f)}")
print(f"Encoder split: Train={len(tr_e)}, Val={len(va_e)}, Test={len(te_e)}")


# ═══════════════════════════════════════════════════════════════
# Train both models with detailed tracking
# ═══════════════════════════════════════════════════════════════
N_EPOCHS = 200
EVAL_EVERY = 2  # Record metrics every 2 epochs

print(f"\n--- Training FCFP4 GP ({N_EPOCHS} epochs) ---")
mdl_f, lik_f, hist_f, preds_f = train_gp_with_val(
    X_fcfp[tr_f], y_fcfp[tr_f], X_fcfp[va_f], y_fcfp[va_f],
    X_fcfp[te_f], y_fcfp[te_f], n_epochs=N_EPOCHS, eval_every=EVAL_EVERY
)
print(f"  Final loss: {hist_f['train_loss'][-1]:.3f}")

print(f"\n--- Training Encoder GP ({N_EPOCHS} epochs) ---")
mdl_e, lik_e, hist_e, preds_e = train_gp_with_val(
    X_enc[tr_e], y_enc[tr_e], X_enc[va_e], y_enc[va_e],
    X_enc[te_e], y_enc[te_e], n_epochs=N_EPOCHS, eval_every=EVAL_EVERY
)
print(f"  Final loss: {hist_e['train_loss'][-1]:.3f}")


# ═══════════════════════════════════════════════════════════════
# Repeated splits for stability (10x for speed)
# ═══════════════════════════════════════════════════════════════
print("\n--- 10× Repeated splits (FCFP4) ---")
rep_f = {"train": [], "val": [], "test": []}
for run in range(10):
    rstate = np.random.RandomState(run * 7 + 1)
    idx = rstate.permutation(N_fcfp)
    ti, vi, ei = idx[:n_tr_f], idx[n_tr_f:n_tr_f+n_va_f], idx[n_tr_f+n_va_f:]
    _, _, _, rp = train_gp_with_val(
        X_fcfp[ti], y_fcfp[ti], X_fcfp[vi], y_fcfp[vi],
        X_fcfp[ei], y_fcfp[ei], n_epochs=150, eval_every=150
    )
    for sp, idxs in [("train", ti), ("val", vi), ("test", ei)]:
        rep_f[sp].append(met(y_fcfp[idxs], rp[sp][0]))
    if (run+1) % 5 == 0:
        print(f"  Run {run+1}/10")

print("--- 10× Repeated splits (Encoder) ---")
rep_e = {"train": [], "val": [], "test": []}
for run in range(10):
    rstate = np.random.RandomState(run * 7 + 1)
    idx = rstate.permutation(N_enc)
    ti, vi, ei = idx[:n_tr_e], idx[n_tr_e:n_tr_e+n_va_e], idx[n_tr_e+n_va_e:]
    _, _, _, rp = train_gp_with_val(
        X_enc[ti], y_enc[ti], X_enc[vi], y_enc[vi],
        X_enc[ei], y_enc[ei], n_epochs=150, eval_every=150
    )
    for sp, idxs in [("train", ti), ("val", vi), ("test", ei)]:
        rep_e[sp].append(met(y_enc[idxs], rp[sp][0]))
    if (run+1) % 5 == 0:
        print(f"  Run {run+1}/10")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Training Curves (Loss + Hyperparams)
# ═══════════════════════════════════════════════════════════════
print("\nGenerating Figure 1: Training Curves...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1a: NLL Loss
ax = axes[0, 0]
ax.plot(hist_f["epoch"], hist_f["train_loss"], "b-", alpha=0.8, linewidth=2, label="FCFP4-2048")
ax.plot(hist_e["epoch"], hist_e["train_loss"], "r-", alpha=0.8, linewidth=2, label="Encoder-128")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Negative Marginal Log-Likelihood", fontsize=12)
ax.set_title("(a) Training Loss", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 1b: Noise variance
ax = axes[0, 1]
ax.plot(hist_f["epoch"], hist_f["noise"], "b-", alpha=0.8, linewidth=2, label="FCFP4")
ax.plot(hist_e["epoch"], hist_e["noise"], "r-", alpha=0.8, linewidth=2, label="Encoder")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("σ²_noise", fontsize=12)
ax.set_title("(b) Learned Noise Variance", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 1c: Output scale
ax = axes[1, 0]
ax.plot(hist_f["epoch"], hist_f["outputscale"], "b-", alpha=0.8, linewidth=2, label="FCFP4")
ax.plot(hist_e["epoch"], hist_e["outputscale"], "r-", alpha=0.8, linewidth=2, label="Encoder")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Output Scale", fontsize=12)
ax.set_title("(c) Learned Output Scale", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 1d: Loss difference
ax = axes[1, 1]
min_len = min(len(hist_f["train_loss"]), len(hist_e["train_loss"]))
diff = [hist_f["train_loss"][i] - hist_e["train_loss"][i] for i in range(min_len)]
ep = hist_f["epoch"][:min_len]
ax.fill_between(ep, 0, diff, where=[d > 0 for d in diff], alpha=0.3, color="red", label="FCFP4 loss > Encoder")
ax.fill_between(ep, 0, diff, where=[d <= 0 for d in diff], alpha=0.3, color="blue", label="FCFP4 loss < Encoder")
ax.plot(ep, diff, "k-", linewidth=1, alpha=0.7)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss(FCFP4) − Loss(Encoder)", fontsize=12)
ax.set_title("(d) Loss Difference", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle("Training Curves: FCFP4-2048 vs Encoder-128 (Tier 3, N≈932)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 01_training_curves.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Learning Curves (T/V/T metrics vs epoch)
# ═══════════════════════════════════════════════════════════════
print("Generating Figure 2: Learning Curves (RMSE/ρ/R² vs Epoch)...")
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

for row, (hist, label, color_base) in enumerate([
    (hist_f, "FCFP4-2048", ("steelblue", "coral", "seagreen")),
    (hist_e, "Encoder-128", ("steelblue", "coral", "seagreen")),
]):
    colors = color_base
    epochs = hist["epoch"]

    # RMSE
    ax = axes[row, 0]
    ax.plot(epochs, hist["train_rmse"], "-", color=colors[0], linewidth=2, label="Train", alpha=0.8)
    ax.plot(epochs, hist["val_rmse"], "-", color=colors[1], linewidth=2, label="Val", alpha=0.8)
    ax.plot(epochs, hist["test_rmse"], "-", color=colors[2], linewidth=2, label="Test", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"{label}: RMSE", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Spearman ρ
    ax = axes[row, 1]
    ax.plot(epochs, hist["train_rho"], "-", color=colors[0], linewidth=2, label="Train", alpha=0.8)
    ax.plot(epochs, hist["val_rho"], "-", color=colors[1], linewidth=2, label="Val", alpha=0.8)
    ax.plot(epochs, hist["test_rho"], "-", color=colors[2], linewidth=2, label="Test", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title(f"{label}: Spearman ρ", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[row, 2]
    ax.plot(epochs, hist["train_r2"], "-", color=colors[0], linewidth=2, label="Train", alpha=0.8)
    ax.plot(epochs, hist["val_r2"], "-", color=colors[1], linewidth=2, label="Val", alpha=0.8)
    ax.plot(epochs, hist["test_r2"], "-", color=colors[2], linewidth=2, label="Test", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title(f"{label}: R²", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle("Learning Curves: Train / Val / Test Metrics vs Epoch",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 02_learning_curves.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Data Distribution Analysis
# ═══════════════════════════════════════════════════════════════
print("Generating Figure 3: Data Distribution Analysis...")
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 3a: pKd distribution (overall + per-split)
ax = fig.add_subplot(gs[0, 0])
ax.hist(y_fcfp, bins=40, alpha=0.7, color="steelblue", edgecolor="white", label="All")
ax.axvline(y_fcfp.mean(), color="red", linestyle="--", linewidth=2,
           label=f"mean={y_fcfp.mean():.2f}")
ax.set_xlabel("pKd", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"(a) pKd Distribution (N={N_fcfp})", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# 3b: pKd per split
ax = fig.add_subplot(gs[0, 1])
bp = ax.boxplot([y_fcfp[tr_f], y_fcfp[va_f], y_fcfp[te_f]],
                labels=["Train", "Val", "Test"], patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["steelblue", "coral", "seagreen"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_ylabel("pKd", fontsize=11)
ax.set_title("(b) pKd Distribution by Split", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2)
for i, (idxs, c) in enumerate([(tr_f, "steelblue"), (va_f, "coral"), (te_f, "seagreen")]):
    mean_v = y_fcfp[idxs].mean()
    ax.text(i + 1, y_fcfp[idxs].max() + 0.3, f"μ={mean_v:.2f}\nN={len(idxs)}",
            ha="center", fontsize=8, fontweight="bold")

# 3c: Molecules per pocket
ax = fig.add_subplot(gs[0, 2])
mc = mol_counts if len(mol_counts) == N_fcfp else mol_counts_enc
ax.hist(mc[:N_fcfp] if len(mc) >= N_fcfp else mc, bins=30, alpha=0.7,
        color="mediumpurple", edgecolor="white")
ax.axvline(mc.mean(), color="red", linestyle="--", linewidth=2,
           label=f"mean={mc.mean():.1f}")
ax.set_xlabel("Molecules per Pocket", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("(c) Generated Molecules per Pocket", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# 3d: FCFP4 PCA projection
ax = fig.add_subplot(gs[1, 0])
pca_f = PCA(n_components=2)
X_pca_f = pca_f.fit_transform(X_fcfp)
sc = ax.scatter(X_pca_f[tr_f, 0], X_pca_f[tr_f, 1], c=y_fcfp[tr_f], cmap="RdYlBu_r",
                s=10, alpha=0.5, label="Train")
ax.scatter(X_pca_f[va_f, 0], X_pca_f[va_f, 1], c=y_fcfp[va_f], cmap="RdYlBu_r",
           s=15, alpha=0.7, marker="s", label="Val")
ax.scatter(X_pca_f[te_f, 0], X_pca_f[te_f, 1], c=y_fcfp[te_f], cmap="RdYlBu_r",
           s=15, alpha=0.7, marker="^", label="Test")
plt.colorbar(sc, ax=ax, label="pKd", shrink=0.8)
ax.set_xlabel(f"PC1 ({pca_f.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca_f.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
ax.set_title("(d) FCFP4-2048 PCA (colored by pKd)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, markerscale=2)
ax.grid(True, alpha=0.2)

# 3e: Encoder PCA projection
ax = fig.add_subplot(gs[1, 1])
pca_e = PCA(n_components=2)
X_pca_e = pca_e.fit_transform(X_enc)
sc = ax.scatter(X_pca_e[tr_e, 0], X_pca_e[tr_e, 1], c=y_enc[tr_e], cmap="RdYlBu_r",
                s=10, alpha=0.5, label="Train")
ax.scatter(X_pca_e[va_e, 0], X_pca_e[va_e, 1], c=y_enc[va_e], cmap="RdYlBu_r",
           s=15, alpha=0.7, marker="s", label="Val")
ax.scatter(X_pca_e[te_e, 0], X_pca_e[te_e, 1], c=y_enc[te_e], cmap="RdYlBu_r",
           s=15, alpha=0.7, marker="^", label="Test")
plt.colorbar(sc, ax=ax, label="pKd", shrink=0.8)
ax.set_xlabel(f"PC1 ({pca_e.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca_e.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
ax.set_title("(e) Encoder-128 PCA (colored by pKd)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, markerscale=2)
ax.grid(True, alpha=0.2)

# 3f: PCA variance explained comparison
ax = fig.add_subplot(gs[1, 2])
pca_f_full = PCA(n_components=min(50, X_fcfp.shape[1]))
pca_f_full.fit(X_fcfp)
pca_e_full = PCA(n_components=min(50, X_enc.shape[1]))
pca_e_full.fit(X_enc)
cum_f = np.cumsum(pca_f_full.explained_variance_ratio_) * 100
cum_e = np.cumsum(pca_e_full.explained_variance_ratio_) * 100
ax.plot(range(1, len(cum_f)+1), cum_f, "b-", linewidth=2, label="FCFP4-2048")
ax.plot(range(1, len(cum_e)+1), cum_e, "r-", linewidth=2, label="Encoder-128")
ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
ax.set_xlabel("Number of PCs", fontsize=11)
ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
ax.set_title("(f) PCA Variance Explained", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3g: pKd vs mol count
ax = fig.add_subplot(gs[2, 0])
mc_use = mol_counts_enc[:N_enc]
ax.scatter(mc_use, y_enc, s=10, alpha=0.3, color="mediumpurple")
rho_mc, p_mc = stats.spearmanr(mc_use, y_enc) if len(mc_use) > 2 else (0, 1)
ax.set_xlabel("Molecules per Pocket", fontsize=11)
ax.set_ylabel("pKd", fontsize=11)
ax.set_title(f"(g) pKd vs Mol Count (ρ={rho_mc:.3f})", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2)

# 3h: Embedding norm distribution (encoder)
ax = fig.add_subplot(gs[2, 1])
norms = np.linalg.norm(X_enc_raw, axis=1)
ax.hist(norms, bins=30, alpha=0.7, color="indianred", edgecolor="white")
ax.axvline(norms.mean(), color="black", linestyle="--", linewidth=2,
           label=f"mean={norms.mean():.2f}")
ax.set_xlabel("L2 Norm of Encoder Embedding", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("(h) Encoder Embedding Norms", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# 3i: Embedding norm vs pKd
ax = fig.add_subplot(gs[2, 2])
sc = ax.scatter(norms, y_enc, c=y_enc, cmap="RdYlBu_r", s=10, alpha=0.4)
rho_n, p_n = stats.spearmanr(norms, y_enc)
ax.set_xlabel("Embedding L2 Norm", fontsize=11)
ax.set_ylabel("pKd", fontsize=11)
ax.set_title(f"(i) pKd vs Embedding Norm (ρ={rho_n:.3f})", fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=ax, label="pKd", shrink=0.8)
ax.grid(True, alpha=0.2)

fig.suptitle("Data Distribution Analysis (Tier 3 Dataset)",
             fontsize=16, fontweight="bold", y=1.01)
fig.savefig(FIG_DIR / "03_data_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 03_data_distribution.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Train/Val/Test Scatter Plots (both models)
# ═══════════════════════════════════════════════════════════════
print("Generating Figure 4: Train/Val/Test Scatter Plots...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
lims_f = [y_fcfp.min() - 0.5, y_fcfp.max() + 0.5]
lims_e = [y_enc.min() - 0.5, y_enc.max() + 0.5]

for row, (model_name, pred_dict, y_arr, idx_dict, lims) in enumerate([
    ("FCFP4-2048", preds_f,
     y_fcfp, {"Train": tr_f, "Val": va_f, "Test": te_f}, lims_f),
    ("Encoder-128", preds_e,
     y_enc, {"Train": tr_e, "Val": va_e, "Test": te_e}, lims_e),
]):
    for col, (sp_name, (sp_key, idx), c) in enumerate(zip(
        ["Train", "Validation", "Test"],
        [("train", idx_dict["Train"]), ("val", idx_dict["Val"]), ("test", idx_dict["Test"])],
        ["steelblue", "coral", "seagreen"],
    )):
        ax = axes[row, col]
        yp, ys = pred_dict[sp_key]
        yt = y_arr[idx]
        m = met(yt, yp)

        sc = ax.scatter(yt, yp, c=ys, cmap="viridis", alpha=0.5, s=12, edgecolors="none")
        ax.plot(lims, lims, "r--", alpha=0.4, linewidth=1.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True pKd", fontsize=11)
        ax.set_ylabel("Predicted pKd", fontsize=11)
        ax.set_title(f"{model_name} — {sp_name} (N={len(yt)})\n"
                     f"RMSE={m['rmse']:.2f}, ρ={m['rho']:.3f}, R²={m['r2']:.3f}",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label="Pred σ", shrink=0.8)

fig.suptitle("Predicted vs True pKd: FCFP4 (top) vs Encoder (bottom)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_scatter_tvt.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 04_scatter_tvt.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Model Comparison Dashboard
# ═══════════════════════════════════════════════════════════════
print("Generating Figure 5: Model Comparison Dashboard...")
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# Row 1: Single-split metrics comparison
# 5a: RMSE comparison
metrics_fcfp = {sp: met(y_fcfp[idx], preds_f[sp][0])
                for sp, idx in [("train", tr_f), ("val", va_f), ("test", te_f)]}
metrics_enc = {sp: met(y_enc[idx], preds_e[sp][0])
               for sp, idx in [("train", tr_e), ("val", va_e), ("test", te_e)]}

x_pos = np.arange(3)
w = 0.35
splits = ["train", "val", "test"]
split_labels = ["Train", "Val", "Test"]

for col, metric_name, metric_label in [
    (0, "rmse", "RMSE"), (1, "rho", "Spearman ρ"), (2, "r2", "R²")
]:
    ax = axes[0, col]
    vals_f = [metrics_fcfp[s][metric_name] for s in splits]
    vals_e = [metrics_enc[s][metric_name] for s in splits]

    b1 = ax.bar(x_pos - w/2, vals_f, w, label="FCFP4-2048", color="steelblue", alpha=0.8)
    b2 = ax.bar(x_pos + w/2, vals_e, w, label="Encoder-128", color="indianred", alpha=0.8)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            y_off = 0.02 * max(abs(v) for v in vals_f + vals_e) if max(abs(v) for v in vals_f + vals_e) > 0 else 0.01
            va = "bottom" if h >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, h + (y_off if h >= 0 else -y_off),
                    f"{h:.3f}", ha="center", va=va, fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"(a{col+1}) {metric_label} by Split", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    if metric_name in ["rho", "r2"]:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="y")

# Row 2: Repeated split distributions
for col, metric_name, metric_label in [
    (0, "rmse", "RMSE"), (1, "rho", "Spearman ρ"), (2, "r2", "R²")
]:
    ax = axes[1, col]
    data_f = [[m[metric_name] for m in rep_f[s]] for s in splits]
    data_e = [[m[metric_name] for m in rep_e[s]] for s in splits]

    positions_f = np.arange(3) * 2 - 0.3
    positions_e = np.arange(3) * 2 + 0.3

    bp1 = ax.boxplot(data_f, positions=positions_f, widths=0.4, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    bp2 = ax.boxplot(data_e, positions=positions_e, widths=0.4, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))

    for patch in bp1["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor("indianred")
        patch.set_alpha(0.6)

    ax.set_xticks(np.arange(3) * 2)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"(b{col+1}) {metric_label} (10× Repeated Splits)", fontsize=12, fontweight="bold")
    if metric_name in ["rho", "r2"]:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="y")

    # Add legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="steelblue", alpha=0.6, label="FCFP4"),
                       Patch(facecolor="indianred", alpha=0.6, label="Encoder")],
              fontsize=9)

fig.suptitle("Model Comparison Dashboard: FCFP4-2048 vs Encoder-128",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "05_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 05_model_comparison.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Residual & Calibration Analysis
# ═══════════════════════════════════════════════════════════════
print("Generating Figure 6: Residual & Calibration Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for row, (model_name, pred_dict, y_arr, idx_dict) in enumerate([
    ("FCFP4-2048", preds_f, y_fcfp, {"Train": tr_f, "Val": va_f, "Test": te_f}),
    ("Encoder-128", preds_e, y_enc, {"Train": tr_e, "Val": va_e, "Test": te_e}),
]):
    # 6a/d: Residual distribution
    ax = axes[row, 0]
    for sp, key, idx, c in [
        ("Train", "train", idx_dict["Train"], "steelblue"),
        ("Val", "val", idx_dict["Val"], "coral"),
        ("Test", "test", idx_dict["Test"], "seagreen"),
    ]:
        resid = y_arr[idx] - pred_dict[key][0]
        ax.hist(resid, bins=25, alpha=0.5, color=c,
                label=f"{sp} (σ={resid.std():.2f})", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Residual (True − Predicted)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{model_name}: Residual Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 6b/e: True vs Predicted distribution
    ax = axes[row, 1]
    ax.hist(y_arr[idx_dict["Test"]], bins=25, alpha=0.5, color="seagreen",
            label="True (Test)", density=True)
    ax.hist(pred_dict["test"][0], bins=25, alpha=0.7, color="seagreen",
            label="Pred (Test)", density=True, histtype="step", linewidth=2)
    ax.hist(y_arr[idx_dict["Train"]], bins=25, alpha=0.3, color="steelblue",
            label="True (Train)", density=True)
    ax.hist(pred_dict["train"][0], bins=25, alpha=0.7, color="steelblue",
            label="Pred (Train)", density=True, histtype="step", linewidth=2)
    ax.set_xlabel("pKd", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{model_name}: True vs Predicted Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 6c/f: Calibration — predicted σ vs actual |error|
    ax = axes[row, 2]
    for sp, key, idx, c, mk in [
        ("Train", "train", idx_dict["Train"], "steelblue", "o"),
        ("Val", "val", idx_dict["Val"], "coral", "s"),
        ("Test", "test", idx_dict["Test"], "seagreen", "^"),
    ]:
        pred_std = pred_dict[key][1]
        actual_err = np.abs(y_arr[idx] - pred_dict[key][0])
        ax.scatter(pred_std, actual_err, c=c, s=8, alpha=0.3, label=sp, marker=mk)

    # Add ideal line
    all_std = np.concatenate([pred_dict[k][1] for k in ["train", "val", "test"]])
    max_v = max(all_std.max(), 5)
    ax.plot([0, max_v], [0, max_v], "r--", alpha=0.5, linewidth=1.5, label="Ideal")
    ax.set_xlabel("Predicted σ", fontsize=11)
    ax.set_ylabel("|True − Predicted|", fontsize=11)
    ax.set_title(f"{model_name}: Uncertainty Calibration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

fig.suptitle("Residual & Uncertainty Analysis: FCFP4 (top) vs Encoder (bottom)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "06_residual_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 06_residual_calibration.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 7: Summary Table + Overfitting Gap Analysis
# ═══════════════════════════════════════════════════════════════
print("Generating Figure 7: Summary Table...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 7a: Overfitting gap (Train metric - Test metric over epochs)
ax = ax1
for hist, label, colors in [
    (hist_f, "FCFP4", ("steelblue", "steelblue")),
    (hist_e, "Encoder", ("indianred", "indianred")),
]:
    gap_rmse = [t - e for t, e in zip(hist["train_rmse"], hist["test_rmse"])]
    gap_rho = [t - e for t, e in zip(hist["train_rho"], hist["test_rho"])]
    ax.plot(hist["epoch"], gap_rho, "-", color=colors[0], linewidth=2,
            label=f"{label} ρ gap", alpha=0.8)

ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Train ρ − Test ρ (Overfitting Gap)", fontsize=12)
ax.set_title("(a) Overfitting Gap: Train ρ − Test ρ vs Epoch", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 7b: Summary table
ax = ax2
ax.axis("off")

cell_data = [
    ["", "FCFP4-2048", "", "", "Encoder-128", "", ""],
    ["Metric", "Train", "Val", "Test", "Train", "Val", "Test"],
]
for mn, ml in [("rmse", "RMSE"), ("rho", "ρ"), ("r2", "R²")]:
    row = [ml]
    for metrics_dict in [metrics_fcfp, metrics_enc]:
        for sp in splits:
            row.append(f"{metrics_dict[sp][mn]:.3f}")
    cell_data.append(row)

# Add repeated split summary
cell_data.append(["", "── 10× Repeated Splits ──", "", "", "── 10× Repeated Splits ──", "", ""])
for mn, ml in [("rmse", "RMSE"), ("rho", "ρ"), ("r2", "R²")]:
    row = [f"{ml} (mean±std)"]
    for rep in [rep_f, rep_e]:
        for sp in splits:
            vals = [m[mn] for m in rep[sp]]
            row.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
    cell_data.append(row)

table = ax.table(cellText=cell_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# Style header rows
for i in range(2):
    for j in range(7):
        table[i, j].set_facecolor("#e0e0e0")
        table[i, j].set_text_props(fontweight="bold")
# Style separator row
for j in range(7):
    table[5, j].set_facecolor("#f0f0f0")
    table[5, j].set_text_props(fontsize=8, fontstyle="italic")

ax.set_title("(b) Comprehensive Metrics Summary", fontsize=13, fontweight="bold", pad=20)

fig.suptitle("Overfitting Analysis & Summary",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "07_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 07_summary.png")


# ═══════════════════════════════════════════════════════════════
# Save results JSON
# ═══════════════════════════════════════════════════════════════
results = {
    "fcfp4_single": {sp: metrics_fcfp[sp] for sp in splits},
    "encoder_single": {sp: metrics_enc[sp] for sp in splits},
    "fcfp4_repeated_10x": {
        sp: {
            mn: {"mean": float(np.mean([m[mn] for m in rep_f[sp]])),
                 "std": float(np.std([m[mn] for m in rep_f[sp]]))}
            for mn in ["rmse", "rho", "r2"]
        } for sp in splits
    },
    "encoder_repeated_10x": {
        sp: {
            mn: {"mean": float(np.mean([m[mn] for m in rep_e[sp]])),
                 "std": float(np.std([m[mn] for m in rep_e[sp]]))}
            for mn in ["rmse", "rho", "r2"]
        } for sp in splits
    },
    "training_history": {
        "fcfp4": {k: [float(v) for v in hist_f[k]] for k in hist_f},
        "encoder": {k: [float(v) for v in hist_e[k]] for k in hist_e},
    },
}

out_json = FIG_DIR / "tier3_analysis_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Results saved to {out_json}")

print(f"\n{'='*60}")
print(f"All 7 figures saved to: {FIG_DIR}")
print(f"{'='*60}")

# Print quick summary
print("\n=== Quick Summary ===")
print(f"{'Model':<15} {'Split':<8} {'RMSE':>8} {'ρ':>8} {'R²':>8}")
print("-" * 50)
for model_name, metrics in [("FCFP4-2048", metrics_fcfp), ("Encoder-128", metrics_enc)]:
    for sp in splits:
        m = metrics[sp]
        print(f"{model_name:<15} {sp:<8} {m['rmse']:>8.3f} {m['rho']:>8.3f} {m['r2']:>8.3f}")
    print()
