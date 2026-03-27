#!/usr/bin/env python
"""
scripts/18_train_val_test_analysis.py
──────────────────────────────────────
Detailed train/val/test split analysis for Tier 3 GP.
Generates 5 diagnostic figures + JSON results.
"""
import json, warnings
import numpy as np
import torch
import gpytorch
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "results" / "tier3_gp"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load data ──
X_raw = np.load(DATA_DIR / "X_FCFP4_2048.npy")
y = np.load(DATA_DIR / "y_pkd.npy")
with open(DATA_DIR / "families.json") as f:
    families = json.load(f)

N = len(y)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"Dataset: {N} pockets, {X.shape[1]}-dim, device={DEVICE}")


class FlexGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_and_predict(X_tr, y_tr, X_te, n_epochs=150, lr=0.1):
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(0.001)
    )
    model = FlexGP(X_t, y_t, lik).to(DEVICE)
    lik = lik.to(DEVICE)

    model.train()
    lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    losses = []
    for ep in range(n_epochs):
        opt.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    model.eval()
    lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        tr_pred = lik(model(X_t))
        te_pred = lik(model(X_te_t))

    return (tr_pred.mean.cpu().numpy(), tr_pred.stddev.cpu().numpy(),
            te_pred.mean.cpu().numpy(), te_pred.stddev.cpu().numpy(),
            losses, model, lik)


def met(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    rho, p = stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (0, 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"rmse": rmse, "mae": mae, "rho": rho, "p_val": p, "r2": r2}


# ═══════════════════════════════════════════════════════════════
# 1. Single split (60/20/20)
# ═══════════════════════════════════════════════════════════════
print("\n=== SINGLE SPLIT (60/20/20) ===")
rng = np.random.RandomState(42)
idx = rng.permutation(N)
n_train = int(N * 0.6)
n_val = int(N * 0.2)
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_tr, y_tr = X[train_idx], y[train_idx]
X_va, y_va = X[val_idx], y[val_idx]
X_te, y_te = X[test_idx], y[test_idx]

print(f"  Train: {len(y_tr)}, Val: {len(y_va)}, Test: {len(y_te)}")

tr_pred, tr_std, va_pred, va_std, losses_s, mdl, lk = \
    train_and_predict(X_tr, y_tr, X_va, n_epochs=200)

# Predict test set
mdl.eval()
lk.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    te_out = lk(mdl(torch.tensor(X_te, dtype=torch.float32, device=DEVICE)))
te_pred = te_out.mean.cpu().numpy()
te_std = te_out.stddev.cpu().numpy()

m_tr = met(y_tr, tr_pred)
m_va = met(y_va, va_pred)
m_te = met(y_te, te_pred)

for name, m in [("Train", m_tr), ("Val", m_va), ("Test", m_te)]:
    rho_s = m["rho"]
    rmse_s = m["rmse"]
    mae_s = m["mae"]
    r2_s = m["r2"]
    p_s = m["p_val"]
    print(f"  {name:5s}: RMSE={rmse_s:.3f}, MAE={mae_s:.3f}, "
          f"rho={rho_s:.3f} (p={p_s:.4f}), R2={r2_s:.3f}")

# Prediction range analysis
print(f"\n  Train pred range: [{tr_pred.min():.2f}, {tr_pred.max():.2f}] "
      f"(true: [{y_tr.min():.2f}, {y_tr.max():.2f}])")
print(f"  Val   pred range: [{va_pred.min():.2f}, {va_pred.max():.2f}] "
      f"(true: [{y_va.min():.2f}, {y_va.max():.2f}])")
print(f"  Test  pred range: [{te_pred.min():.2f}, {te_pred.max():.2f}] "
      f"(true: [{y_te.min():.2f}, {y_te.max():.2f}])")

# ═══════════════════════════════════════════════════════════════
# 2. 30× Repeated splits
# ═══════════════════════════════════════════════════════════════
print("\n=== 30× REPEATED SPLITS ===")
all_tr_m, all_va_m, all_te_m = [], [], []

for run in range(30):
    rng_r = np.random.RandomState(run * 7 + 1)
    idx_r = rng_r.permutation(N)
    tr_i = idx_r[:n_train]
    va_i = idx_r[n_train:n_train + n_val]
    te_i = idx_r[n_train + n_val:]

    trp, _, vap, _, _, m2, l2 = train_and_predict(X[tr_i], y[tr_i], X[va_i])
    m2.eval()
    l2.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        teo = l2(m2(torch.tensor(X[te_i], dtype=torch.float32, device=DEVICE)))
    tep = teo.mean.cpu().numpy()

    all_tr_m.append(met(y[tr_i], trp))
    all_va_m.append(met(y[va_i], vap))
    all_te_m.append(met(y[te_i], tep))

    if (run + 1) % 10 == 0:
        print(f"  Run {run + 1}/30")

for name, ms in [("Train", all_tr_m), ("Val", all_va_m), ("Test", all_te_m)]:
    rmse_mean = np.mean([m["rmse"] for m in ms])
    rmse_std = np.std([m["rmse"] for m in ms])
    rho_mean = np.mean([m["rho"] for m in ms])
    rho_std = np.std([m["rho"] for m in ms])
    r2_mean = np.mean([m["r2"] for m in ms])
    r2_std = np.std([m["r2"] for m in ms])
    print(f"  {name:5s}: RMSE={rmse_mean:.3f}±{rmse_std:.3f}, "
          f"rho={rho_mean:.3f}±{rho_std:.3f}, R2={r2_mean:.3f}±{r2_std:.3f}")

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
lims = [y.min() - 1, y.max() + 1]

# Fig1: Train/Val/Test scatter
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (name, yt, yp, ys, c) in zip(axes, [
    ("Train", y_tr, tr_pred, tr_std, "steelblue"),
    ("Validation", y_va, va_pred, va_std, "coral"),
    ("Test", y_te, te_pred, te_std, "seagreen"),
]):
    m = met(yt, yp)
    sc = ax.scatter(yt, yp, c=ys, cmap="viridis", alpha=0.5, s=20, edgecolors="none")
    ax.plot(lims, lims, "r--", alpha=0.4, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True pKd", fontsize=12)
    ax.set_ylabel("Predicted pKd", fontsize=12)
    ax.set_title(f"{name} (N={len(yt)})\n"
                 f"RMSE={m['rmse']:.2f}, ρ={m['rho']:.3f}, R²={m['r2']:.3f}",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label="Pred σ", shrink=0.8)
fig.suptitle("GP Performance: Train / Validation / Test (60/20/20)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "train_val_test_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: train_val_test_scatter.png")

# Fig2: Training curve + metrics bar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(losses_s, "b-", alpha=0.7, linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Negative MLL")
ax1.set_title("Training Curve (NLL Loss)")
ax1.grid(True, alpha=0.3)

sets_names = ["Train", "Val", "Test"]
rmses_bar = [m_tr["rmse"], m_va["rmse"], m_te["rmse"]]
rhos_bar = [m_tr["rho"], m_va["rho"], m_te["rho"]]
r2s_bar = [m_tr["r2"], m_va["r2"], m_te["r2"]]
x_pos = np.arange(3)
w = 0.25

b1 = ax2.bar(x_pos - w, rmses_bar, w, label="RMSE", color="steelblue", alpha=0.8)
b2 = ax2.bar(x_pos, rhos_bar, w, label="ρ", color="coral", alpha=0.8)
b3 = ax2.bar(x_pos + w, r2s_bar, w, label="R²", color="seagreen", alpha=0.8)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        offset = 0.02 if h >= 0 else -0.02
        va = "bottom" if h >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + offset,
                 f"{h:.2f}", ha="center", va=va, fontsize=8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(sets_names)
ax2.set_ylabel("Score")
ax2.set_title("Metrics by Split")
ax2.legend(loc="upper left")
ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax2.grid(True, alpha=0.2, axis="y")
fig.suptitle("Training & Evaluation Overview", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "training_and_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: training_and_metrics.png")

# Fig3: Boxplots from 30 runs
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, mn, yl in zip(axes, ["rmse", "rho", "r2"], ["RMSE", "Spearman ρ", "R²"]):
    tr_v = [m[mn] for m in all_tr_m]
    va_v = [m[mn] for m in all_va_m]
    te_v = [m[mn] for m in all_te_m]
    bp = ax.boxplot([tr_v, va_v, te_v], labels=["Train", "Val", "Test"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], ["steelblue", "coral", "seagreen"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(yl, fontsize=12)
    ax.set_title(f"{yl} Distribution (30 runs)", fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    if mn in ["rho", "r2"]:
        ax.axhline(0, color="red", linestyle="--", alpha=0.4)
    for i, vals in enumerate([tr_v, va_v, te_v]):
        all_vals = tr_v + va_v + te_v
        span = max(all_vals) - min(all_vals)
        ax.text(i + 1, max(vals) + 0.05 * span,
                f"{np.mean(vals):.3f}±{np.std(vals):.3f}",
                ha="center", fontsize=8, fontweight="bold")
fig.suptitle("30× Repeated Train/Val/Test Split Distributions",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "repeated_split_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: repeated_split_boxplots.png")

# Fig4: Distribution analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(y_tr, bins=25, alpha=0.4, color="steelblue", label="True train", density=True)
ax.hist(tr_pred, bins=25, alpha=0.6, color="steelblue", label="Pred train",
        density=True, histtype="step", linewidth=2)
ax.hist(y_te, bins=20, alpha=0.4, color="seagreen", label="True test", density=True)
ax.hist(te_pred, bins=20, alpha=0.6, color="seagreen", label="Pred test",
        density=True, histtype="step", linewidth=2)
ax.set_xlabel("pKd")
ax.set_ylabel("Density")
ax.set_title("True vs Predicted pKd Distribution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

ax = axes[1]
tr_resid = y_tr - tr_pred
va_resid = y_va - va_pred
te_resid = y_te - te_pred
ax.hist(tr_resid, bins=20, alpha=0.5, color="steelblue",
        label=f"Train (σ={tr_resid.std():.2f})", density=True)
ax.hist(va_resid, bins=20, alpha=0.5, color="coral",
        label=f"Val (σ={va_resid.std():.2f})", density=True)
ax.hist(te_resid, bins=20, alpha=0.5, color="seagreen",
        label=f"Test (σ={te_resid.std():.2f})", density=True)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("Residual (True − Predicted)")
ax.set_ylabel("Density")
ax.set_title("Residual Distribution by Split")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
fig.suptitle("Distribution Analysis", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "distribution_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: distribution_analysis.png")

# Fig5: Cross-split overlay + confidence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for name, yt, yp, c, mk in [
    ("Train", y_tr, tr_pred, "steelblue", "o"),
    ("Val", y_va, va_pred, "coral", "s"),
    ("Test", y_te, te_pred, "seagreen", "^"),
]:
    ax.scatter(yt, yp, c=c, alpha=0.4, s=15, label=name, marker=mk, edgecolors="none")
ax.plot(lims, lims, "r--", alpha=0.4)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("True pKd")
ax.set_ylabel("Predicted pKd")
ax.set_title("All Splits Overlay")
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

ax = axes[1]
for name, yp, ys, c in [
    ("Train", tr_pred, tr_std, "steelblue"),
    ("Val", va_pred, va_std, "coral"),
    ("Test", te_pred, te_std, "seagreen"),
]:
    ax.scatter(yp, ys, c=c, alpha=0.4, s=15, label=name, edgecolors="none")
ax.set_xlabel("Predicted pKd")
ax.set_ylabel("Predictive σ")
ax.set_title("Prediction Confidence by Split")
ax.legend()
ax.grid(True, alpha=0.2)
fig.suptitle("Cross-Split Comparison", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "cross_split_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cross_split_comparison.png")

# ── Save results JSON ──
results = {
    "single_split": {
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "train": {k: float(v) for k, v in m_tr.items()},
        "val": {k: float(v) for k, v in m_va.items()},
        "test": {k: float(v) for k, v in m_te.items()},
        "train_pred_range": [float(tr_pred.min()), float(tr_pred.max())],
        "val_pred_range": [float(va_pred.min()), float(va_pred.max())],
        "test_pred_range": [float(te_pred.min()), float(te_pred.max())],
    },
    "repeated_30x": {},
}
for sname, ms in [("train", all_tr_m), ("val", all_va_m), ("test", all_te_m)]:
    results["repeated_30x"][sname] = {
        "rmse_mean": float(np.mean([m["rmse"] for m in ms])),
        "rmse_std": float(np.std([m["rmse"] for m in ms])),
        "rho_mean": float(np.mean([m["rho"] for m in ms])),
        "rho_std": float(np.std([m["rho"] for m in ms])),
        "r2_mean": float(np.mean([m["r2"] for m in ms])),
        "r2_std": float(np.std([m["r2"] for m in ms])),
    }

with open(DATA_DIR / "train_val_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print("ALL DONE — Results saved to results/tier3_gp/")
print(f"{'=' * 60}")
