#!/usr/bin/env python
"""
scripts/25_regularization_study.py
───────────────────────────────────
正则化实验：对比 Encoder-128 在不同正则化策略下的泛化能力。

当前模型 (ExactGP) 是纯核方法，没有 MLP 层，因此不能直接加 Dropout / Weight Decay。
本脚本引入 Deep Kernel Learning (DKL)，在嵌入和 GP 之间插入 MLP 特征提取器，
然后系统对比以下策略：

  1. Baseline: 纯 GP (ExactGP + RQ kernel, 无 MLP)
  2. DKL:      MLP(128→64→32) + GP, 无正则化
  3. DKL+D03:  MLP + Dropout(0.3) + GP
  4. DKL+D05:  MLP + Dropout(0.5) + GP
  5. DKL+WD:   MLP + AdamW(weight_decay=0.01) + GP
  6. DKL+D03+WD: MLP + Dropout(0.3) + AdamW(wd=0.01) + GP
  7. GP+NoisePrior: 纯 GP，增大噪声下界 (0.1) + 超参先验

生成图表：
  1. 学习曲线对比 (RMSE/ρ/R² vs epoch, train & test)
  2. 模型对比柱状图 + 10× 重复划分箱线图
  3. 散点图 (predicted vs true, test set, top 4 configs)
  4. 过拟合差距对比
  5. 汇总表
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from scipy import stats
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "results" / "tier3_gp"
FIG_DIR = DATA_DIR / "figures" / "regularization"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# MLP Feature Extractor
# ═══════════════════════════════════════════════════════════════
class MLPExtractor(nn.Module):
    """MLP feature extractor for Deep Kernel Learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# GP Models
# ═══════════════════════════════════════════════════════════════
class BaselineGP(gpytorch.models.ExactGP):
    """Pure GP with RQ kernel (no MLP)."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class DKLGP(gpytorch.models.ExactGP):
    """Deep Kernel Learning: MLP feature extractor + GP."""

    def __init__(self, train_x, train_y, likelihood, extractor: MLPExtractor):
        super().__init__(train_x, train_y, likelihood)
        self.extractor = extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        features = self.extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(features), self.covar_module(features)
        )


class PriorGP(gpytorch.models.ExactGP):
    """Pure GP with higher noise floor and hyperparameter priors."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        # Priors to constrain hyperparameters
        self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.5)
        likelihood.noise_prior = gpytorch.priors.GammaPrior(1.5, 1.0)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ═══════════════════════════════════════════════════════════════
# Training functions
# ═══════════════════════════════════════════════════════════════
def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, p = stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "rho": float(rho), "r2": r2, "p": float(p)}


def train_config(X_tr, y_tr, X_va, y_va, X_te, y_te, *,
                 config_name: str, n_epochs: int = 200,
                 lr: float = 0.1, eval_every: int = 2,
                 dropout: float = 0.0, weight_decay: float = 0.0,
                 use_dkl: bool = False, use_prior: bool = False,
                 noise_lb: float = 0.001):
    """Train one configuration, return history + final predictions."""
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_v = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    X_e = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
    ).to(DEVICE)

    if use_dkl:
        extractor = MLPExtractor(
            input_dim=X_tr.shape[1], hidden_dim=64,
            output_dim=32, dropout=dropout,
        ).to(DEVICE)
        model = DKLGP(X_t, y_t, lik, extractor).to(DEVICE)
    elif use_prior:
        model = PriorGP(X_t, y_t, lik).to(DEVICE)
    else:
        model = BaselineGP(X_t, y_t, lik).to(DEVICE)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    history = {
        "epoch": [], "train_loss": [],
        "train_rmse": [], "train_rho": [], "train_r2": [],
        "val_rmse": [], "val_rho": [], "val_r2": [],
        "test_rmse": [], "test_rho": [], "test_r2": [],
        "noise": [], "outputscale": [],
    }

    for ep in range(n_epochs):
        model.train()
        lik.train()
        opt.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        opt.step()

        if ep % eval_every == 0 or ep == n_epochs - 1:
            model.eval()
            lik.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                tr_p = lik(model(X_t)).mean.cpu().numpy()
                va_p = lik(model(X_v)).mean.cpu().numpy()
                te_p = lik(model(X_e)).mean.cpu().numpy()

            history["epoch"].append(ep)
            history["train_loss"].append(loss.item())
            for prefix, yt, yp in [
                ("train", y_tr, tr_p), ("val", y_va, va_p), ("test", y_te, te_p)
            ]:
                m = _metrics(yt, yp)
                history[f"{prefix}_rmse"].append(m["rmse"])
                history[f"{prefix}_rho"].append(m["rho"])
                history[f"{prefix}_r2"].append(m["r2"])

            history["noise"].append(lik.noise.item())
            history["outputscale"].append(model.covar_module.outputscale.item())

    # Final predictions
    model.eval()
    lik.eval()
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


# ═══════════════════════════════════════════════════════════════
# Configuration definitions
# ═══════════════════════════════════════════════════════════════
CONFIGS = [
    {
        "name": "Baseline GP",
        "short": "baseline",
        "use_dkl": False, "use_prior": False,
        "dropout": 0.0, "weight_decay": 0.0,
        "noise_lb": 0.001, "lr": 0.1,
        "color": "steelblue",
    },
    {
        "name": "DKL (no reg)",
        "short": "dkl",
        "use_dkl": True, "use_prior": False,
        "dropout": 0.0, "weight_decay": 0.0,
        "noise_lb": 0.001, "lr": 0.01,
        "color": "orange",
    },
    {
        "name": "DKL + Dropout 0.3",
        "short": "dkl_d03",
        "use_dkl": True, "use_prior": False,
        "dropout": 0.3, "weight_decay": 0.0,
        "noise_lb": 0.001, "lr": 0.01,
        "color": "mediumseagreen",
    },
    {
        "name": "DKL + Dropout 0.5",
        "short": "dkl_d05",
        "use_dkl": True, "use_prior": False,
        "dropout": 0.5, "weight_decay": 0.0,
        "noise_lb": 0.001, "lr": 0.01,
        "color": "mediumpurple",
    },
    {
        "name": "DKL + WD 0.01",
        "short": "dkl_wd",
        "use_dkl": True, "use_prior": False,
        "dropout": 0.0, "weight_decay": 0.01,
        "noise_lb": 0.001, "lr": 0.01,
        "color": "indianred",
    },
    {
        "name": "DKL + D0.3 + WD",
        "short": "dkl_d03_wd",
        "use_dkl": True, "use_prior": False,
        "dropout": 0.3, "weight_decay": 0.01,
        "noise_lb": 0.001, "lr": 0.01,
        "color": "darkcyan",
    },
    {
        "name": "GP + NoisePrior",
        "short": "gp_prior",
        "use_dkl": False, "use_prior": True,
        "dropout": 0.0, "weight_decay": 0.0,
        "noise_lb": 0.1, "lr": 0.1,
        "color": "goldenrod",
    },
]


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
print("Loading Encoder-128 data...")
X_raw = np.load(DATA_DIR / "X_encoder_128.npy")
y = np.load(DATA_DIR / "y_pkd_encoder.npy")
N = len(y)
print(f"  N={N}, dim={X_raw.shape[1]}, pKd=[{y.min():.2f}, {y.max():.2f}]")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)


# ═══════════════════════════════════════════════════════════════
# Single 60/20/20 split
# ═══════════════════════════════════════════════════════════════
rng = np.random.RandomState(42)
idx = rng.permutation(N)
n_tr = int(N * 0.6)
n_va = int(N * 0.2)
tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
print(f"  Split: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")

N_EPOCHS = 200
EVAL_EVERY = 2


# ═══════════════════════════════════════════════════════════════
# Train all configs (single split)
# ═══════════════════════════════════════════════════════════════
results = {}
for cfg in CONFIGS:
    print(f"\n--- Training: {cfg['name']} ---")
    _, _, hist, preds = train_config(
        X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X[te_idx], y[te_idx],
        config_name=cfg["name"],
        n_epochs=N_EPOCHS, lr=cfg["lr"], eval_every=EVAL_EVERY,
        dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
        use_dkl=cfg["use_dkl"], use_prior=cfg["use_prior"],
        noise_lb=cfg["noise_lb"],
    )

    final = {}
    for sp, sp_idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        final[sp] = _metrics(y[sp_idx], preds[sp][0])
    print(f"  Train: RMSE={final['train']['rmse']:.3f}, ρ={final['train']['rho']:.3f}")
    print(f"  Val:   RMSE={final['val']['rmse']:.3f}, ρ={final['val']['rho']:.3f}")
    print(f"  Test:  RMSE={final['test']['rmse']:.3f}, ρ={final['test']['rho']:.3f}")

    results[cfg["short"]] = {
        "name": cfg["name"], "color": cfg["color"],
        "history": hist, "preds": preds, "final": final,
    }


# ═══════════════════════════════════════════════════════════════
# 10× Repeated splits
# ═══════════════════════════════════════════════════════════════
print("\n\n=== 10× Repeated splits ===")
repeated = {cfg["short"]: {"train": [], "val": [], "test": []} for cfg in CONFIGS}

for run in range(10):
    rstate = np.random.RandomState(run * 7 + 1)
    idx_r = rstate.permutation(N)
    ti = idx_r[:n_tr]
    vi = idx_r[n_tr:n_tr + n_va]
    ei = idx_r[n_tr + n_va:]

    for cfg in CONFIGS:
        _, _, _, rp = train_config(
            X[ti], y[ti], X[vi], y[vi], X[ei], y[ei],
            config_name=cfg["name"],
            n_epochs=150, lr=cfg["lr"], eval_every=150,
            dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
            use_dkl=cfg["use_dkl"], use_prior=cfg["use_prior"],
            noise_lb=cfg["noise_lb"],
        )
        for sp, sp_idx in [("train", ti), ("val", vi), ("test", ei)]:
            repeated[cfg["short"]][sp].append(_metrics(y[sp_idx], rp[sp][0]))

    print(f"  Repeat {run + 1}/10 done")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Learning curves (train & test RMSE / ρ vs epoch)
# ═══════════════════════════════════════════════════════════════
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 3, figsize=(22, 11))

for col, (y_key, y_label) in enumerate([
    ("rmse", "RMSE"), ("rho", "Spearman ρ"), ("r2", "R²"),
]):
    # Train
    ax = axes[0, col]
    for cfg in CONFIGS:
        r = results[cfg["short"]]
        ax.plot(r["history"]["epoch"], r["history"][f"train_{y_key}"],
                color=cfg["color"], linewidth=1.8, alpha=0.85, label=cfg["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(f"Train {y_label}", fontsize=12, fontweight="bold")
    if col == 0:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    if y_key in ("rho", "r2"):
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

    # Test
    ax = axes[1, col]
    for cfg in CONFIGS:
        r = results[cfg["short"]]
        ax.plot(r["history"]["epoch"], r["history"][f"test_{y_key}"],
                color=cfg["color"], linewidth=1.8, alpha=0.85, label=cfg["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(f"Test {y_label}", fontsize=12, fontweight="bold")
    if col == 0:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    if y_key in ("rho", "r2"):
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

fig.suptitle("Regularization Study: Learning Curves (Encoder-128, Tier 3)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 01_learning_curves.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Model comparison bar chart + 10× boxplots
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 11))

x_pos = np.arange(len(CONFIGS))
cfg_labels = [c["name"] for c in CONFIGS]
cfg_colors = [c["color"] for c in CONFIGS]

# Row 1: Single-split bar charts
for col, (mkey, mlabel) in enumerate([
    ("rmse", "RMSE"), ("rho", "Spearman ρ"), ("r2", "R²"),
]):
    ax = axes[0, col]
    # Train
    vals_tr = [results[c["short"]]["final"]["train"][mkey] for c in CONFIGS]
    vals_te = [results[c["short"]]["final"]["test"][mkey] for c in CONFIGS]

    w = 0.35
    b1 = ax.bar(x_pos - w / 2, vals_tr, w, color=[c["color"] for c in CONFIGS],
                alpha=0.45, label="Train", edgecolor="gray")
    b2 = ax.bar(x_pos + w / 2, vals_te, w, color=[c["color"] for c in CONFIGS],
                alpha=0.9, label="Test", edgecolor="black", linewidth=0.5)

    for bar in b2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va=va, fontsize=6.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cfg_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel(mlabel)
    ax.set_title(f"{mlabel} (Train vs Test)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    if mkey in ("rho", "r2"):
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

# Row 2: 10× repeated split boxplots (test only)
for col, (mkey, mlabel) in enumerate([
    ("rmse", "RMSE"), ("rho", "Spearman ρ"), ("r2", "R²"),
]):
    ax = axes[1, col]
    data = [[m[mkey] for m in repeated[c["short"]]["test"]] for c in CONFIGS]

    bp = ax.boxplot(data, labels=cfg_labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], cfg_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Add means
    for i, d in enumerate(data):
        mean_v = np.mean(d)
        ax.scatter(i + 1, mean_v, color="black", marker="D", s=20, zorder=5)
        ax.text(i + 1, mean_v + 0.01, f"{mean_v:.3f}", ha="center", fontsize=6.5,
                fontweight="bold")

    ax.set_xticklabels(cfg_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel(mlabel)
    ax.set_title(f"Test {mlabel} (10× Repeated Splits)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    if mkey in ("rho", "r2"):
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

fig.suptitle("Regularization Study: Model Comparison (Encoder-128)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 02_model_comparison.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Scatter plots (test set, top 4 configs)
# ═══════════════════════════════════════════════════════════════
# Pick top 4 by test ρ
sorted_cfgs = sorted(CONFIGS, key=lambda c: results[c["short"]]["final"]["test"]["rho"],
                     reverse=True)
top4 = sorted_cfgs[:4]

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
lims = [y.min() - 0.5, y.max() + 0.5]

for i, cfg in enumerate(top4):
    ax = axes[i]
    r = results[cfg["short"]]
    yp, ys = r["preds"]["test"]
    yt = y[te_idx]
    m = r["final"]["test"]

    sc = ax.scatter(yt, yp, c=ys, cmap="viridis", alpha=0.5, s=12, edgecolors="none")
    ax.plot(lims, lims, "r--", alpha=0.4, linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True pKd")
    ax.set_ylabel("Predicted pKd")
    ax.set_title(f"{cfg['name']}\nRMSE={m['rmse']:.3f}, ρ={m['rho']:.3f}, R²={m['r2']:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="Pred σ", shrink=0.8)

fig.suptitle("Test Set Predictions: Top 4 Configurations",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_scatter_top4.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 03_scatter_top4.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Overfitting gap analysis
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

for col, (mkey, mlabel, gap_sign) in enumerate([
    ("rmse", "RMSE", 1),     # test - train (positive = overfit)
    ("rho", "Spearman ρ", 1),  # train - test (positive = overfit)
    ("r2", "R²", 1),           # train - test (positive = overfit)
]):
    ax = axes[col]
    for cfg in CONFIGS:
        r = results[cfg["short"]]
        epochs = r["history"]["epoch"]
        if mkey == "rmse":
            gap = [r["history"][f"test_{mkey}"][i] - r["history"][f"train_{mkey}"][i]
                   for i in range(len(epochs))]
            ylabel = "Test RMSE − Train RMSE"
        else:
            gap = [r["history"][f"train_{mkey}"][i] - r["history"][f"test_{mkey}"][i]
                   for i in range(len(epochs))]
            ylabel = f"Train {mlabel} − Test {mlabel}"

        ax.plot(epochs, gap, color=cfg["color"], linewidth=1.8, alpha=0.85,
                label=cfg["name"])

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Overfitting Gap: {mlabel}", fontsize=12, fontweight="bold")
    if col == 0:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

fig.suptitle("Regularization Effect on Overfitting Gap",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_overfitting_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 04_overfitting_gap.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Summary table + train/test comparison
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis("off")

# Build table data
headers = ["Config", "Train RMSE", "Test RMSE", "Train ρ", "Test ρ",
           "Train R²", "Test R²", "ρ Gap", "10× Test ρ (mean±std)"]
rows = []
for cfg in CONFIGS:
    r = results[cfg["short"]]
    f = r["final"]
    rep_rho = [m["rho"] for m in repeated[cfg["short"]]["test"]]
    gap = f["train"]["rho"] - f["test"]["rho"]
    rows.append([
        cfg["name"],
        f"{f['train']['rmse']:.3f}", f"{f['test']['rmse']:.3f}",
        f"{f['train']['rho']:.3f}", f"{f['test']['rho']:.3f}",
        f"{f['train']['r2']:.3f}", f"{f['test']['r2']:.3f}",
        f"{gap:.3f}",
        f"{np.mean(rep_rho):.3f}±{np.std(rep_rho):.3f}",
    ])

table = ax.table(cellText=rows, colLabels=headers, loc="center",
                 cellLoc="center", colWidths=[0.14] + [0.095] * 7 + [0.14])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# Color header
for j in range(len(headers)):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

# Highlight best test ρ row
best_rho_idx = max(range(len(CONFIGS)),
                   key=lambda i: results[CONFIGS[i]["short"]]["final"]["test"]["rho"])
for j in range(len(headers)):
    table[best_rho_idx + 1, j].set_facecolor("#E2EFDA")

# Alternate row colors
for i in range(len(CONFIGS)):
    if i != best_rho_idx:
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

ax.set_title("Regularization Study: Comprehensive Results Summary",
             fontsize=14, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "05_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 05_summary_table.png")


# ═══════════════════════════════════════════════════════════════
# Save JSON results
# ═══════════════════════════════════════════════════════════════
json_out = {}
for cfg in CONFIGS:
    r = results[cfg["short"]]
    rep_data = {}
    for sp in ("train", "val", "test"):
        rep_list = repeated[cfg["short"]][sp]
        rep_data[sp] = {
            mk: {
                "mean": float(np.mean([m[mk] for m in rep_list])),
                "std": float(np.std([m[mk] for m in rep_list])),
            }
            for mk in ("rmse", "rho", "r2")
        }
    json_out[cfg["short"]] = {
        "name": cfg["name"],
        "single_split": r["final"],
        "repeated_10x": rep_data,
    }

with open(FIG_DIR / "regularization_results.json", "w") as f:
    json.dump(json_out, f, indent=2)
print(f"\n✓ Results saved to {FIG_DIR / 'regularization_results.json'}")

# Print summary
print("\n" + "=" * 80)
print("REGULARIZATION STUDY SUMMARY (Encoder-128, Tier 3)")
print("=" * 80)
print(f"{'Config':<22} {'Train ρ':>8} {'Test ρ':>8} {'Gap':>6} {'10× Test ρ':>16}")
print("-" * 70)
for cfg in CONFIGS:
    r = results[cfg["short"]]
    f = r["final"]
    rep_rho = [m["rho"] for m in repeated[cfg["short"]]["test"]]
    gap = f["train"]["rho"] - f["test"]["rho"]
    print(f"{cfg['name']:<22} {f['train']['rho']:>8.3f} {f['test']['rho']:>8.3f} "
          f"{gap:>6.3f} {np.mean(rep_rho):>7.3f}±{np.std(rep_rho):.3f}")
print("=" * 80)
print(f"\n✓ All figures saved to {FIG_DIR}")
