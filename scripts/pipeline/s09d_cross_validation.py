"""
scripts/pipeline/s09d_cross_validation.py
─────────────────────────────────────────
Stage 2.5: 5-Fold Cross-Validation Robustness Check

Re-evaluate 4 models across 5 grouped train/val splits to eliminate
single-split bias observed in Stage 2–3. Test set (CASF-2016) is
fixed across all folds.

Models:
  L8       — single-layer GP on layer 8
  WS-all   — WeightedSumFusion over all 10 layers
  Attn-top2 — LayerAttentionFusion on layers [8, 6]
  Attn-all  — LayerAttentionFusion on all 10 layers

Usage:
    python scripts/pipeline/s09d_cross_validation.py \\
        --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits_5fold data/pdbbind_v2020/splits_5fold.json \\
        --output results/stage2/cross_validation \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_LAYERS = 10


# ─────────────────────────────────────────────
# Data loading (loads once, splits per fold)
# ─────────────────────────────────────────────

def load_all_embeddings(emb_path, labels_path, n_layers=N_LAYERS):
    """Load ALL embeddings + labels into a dict keyed by pdb_code.

    Returns
    -------
    code_layers : dict  {code: {layer_idx: ndarray (d,)}}
    label_map   : dict  {code: float}
    """
    import pandas as pd

    logger.info(f"Loading {emb_path} into memory...")
    emb = dict(np.load(emb_path, allow_pickle=True))
    logger.info(f"  Loaded {len(emb)} keys")

    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))

    codes = set()
    for key in emb:
        if key.endswith("_layer_0"):
            codes.add(key[:-8])

    code_layers = {}
    for code in sorted(codes):
        if code not in label_map:
            continue
        layers = {}
        for li in range(n_layers):
            k = f"{code}_layer_{li}"
            if k in emb:
                layers[li] = emb[k]
        if len(layers) == n_layers:
            code_layers[code] = layers

    logger.info(f"  {len(code_layers)} complexes with all {n_layers} layers + labels")
    return code_layers, label_map


def build_split_arrays(code_layers, label_map, codes_list, layer_indices):
    """Build X_layers dict + y array for a list of pdb codes.

    Returns
    -------
    X_layers : dict {layer_idx: ndarray (N, d)}
    y        : ndarray (N,)
    """
    X_per_layer = {li: [] for li in layer_indices}
    y = []
    for code in codes_list:
        if code not in code_layers:
            continue
        for li in layer_indices:
            X_per_layer[li].append(code_layers[code][li])
        y.append(label_map[code])

    X_layers = {li: np.stack(X_per_layer[li]) for li in layer_indices}
    y = np.array(y, dtype=np.float32)
    return X_layers, y


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true, mu, var):
    from scipy.stats import spearmanr

    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rho, _ = spearmanr(y_true, mu)
    rmse = np.sqrt(np.mean((y_true - mu) ** 2))
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var)
    return {"R2": float(r2), "spearman_rho": float(rho),
            "RMSE": float(rmse), "NLL": float(nll)}


# ─────────────────────────────────────────────
# Model trainers
# ─────────────────────────────────────────────

def train_single_layer_gp(X_train_layers, y_train, layer_idx,
                          n_inducing=512, n_epochs=200, batch_size=256,
                          lr=0.01, device="cuda"):
    """Train a GP on a single layer's embeddings."""
    from bayesdiff.gp_oracle import GPOracle

    X = X_train_layers[layer_idx]
    d = X.shape[1]
    oracle = GPOracle(d=d, n_inducing=n_inducing, device=device)
    oracle.train(X, y_train, n_epochs=n_epochs, batch_size=batch_size,
                 lr=lr, verbose=False)
    return oracle


def predict_single_layer(oracle, X_layers, layer_idx):
    X = X_layers[layer_idx]
    mu, var = oracle.predict(X)
    return mu, var


def train_ws_gp(X_train_layers, y_train, layer_indices,
                n_inducing=512, n_epochs=200, batch_size=256,
                lr=0.01, fusion_lr=0.05, device="cuda"):
    """Train WeightedSumFusion + GP jointly."""
    import gpytorch
    from bayesdiff.gp_oracle import SVGPModel
    from bayesdiff.layer_fusion import WeightedSumFusion

    dev = torch.device(device)
    n_layers = len(layer_indices)
    d = X_train_layers[layer_indices[0]].shape[1]
    N = len(y_train)

    layer_tensors = [torch.tensor(X_train_layers[li], dtype=torch.float32)
                     for li in layer_indices]
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    fusion = WeightedSumFusion(n_layers=n_layers).to(dev)

    with torch.no_grad():
        X_init, _ = fusion([lt.to(dev) for lt in layer_tensors])
    idx = torch.randperm(N)[:min(n_inducing, N)]
    inducing_points = X_init[idx].clone()

    model = SVGPModel(inducing_points).to(dev)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model.train(); lik.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": lik.parameters(), "lr": lr},
        {"params": fusion.parameters(), "lr": fusion_lr},
    ])
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=N)

    dataset = torch.utils.data.TensorDataset(torch.arange(N), y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lt_dev = [lt.to(dev) for lt in layer_tensors]

    for epoch in range(n_epochs):
        for idx_b, y_b in loader:
            y_b = y_b.to(dev)
            optimizer.zero_grad()
            z, _ = fusion([lt[idx_b] for lt in lt_dev])
            loss = -mll(model(z), y_b)
            loss.backward()
            optimizer.step()

    model.eval(); lik.eval(); fusion.eval()
    return fusion, model, lik


def predict_fusion_gp(fusion, model, lik, X_layers, layer_indices, device="cuda"):
    dev = torch.device(device)
    lts = [torch.tensor(X_layers[li], dtype=torch.float32, device=dev)
           for li in layer_indices]
    with torch.no_grad():
        z, _ = fusion(lts)
        dist = lik(model(z))
        return dist.mean.cpu().numpy(), dist.variance.cpu().numpy()


def train_attn_gp(X_train_layers, y_train, layer_indices,
                  n_inducing=512, n_epochs=200, batch_size=256,
                  lr=0.01, fusion_lr=0.01, hidden_dim=64, device="cuda"):
    """Train LayerAttentionFusion + GP jointly."""
    import gpytorch
    from bayesdiff.gp_oracle import SVGPModel
    from bayesdiff.layer_fusion import LayerAttentionFusion

    dev = torch.device(device)
    n_layers = len(layer_indices)
    d = X_train_layers[layer_indices[0]].shape[1]
    N = len(y_train)

    layer_tensors = [torch.tensor(X_train_layers[li], dtype=torch.float32)
                     for li in layer_indices]
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    fusion = LayerAttentionFusion(embed_dim=d, hidden_dim=hidden_dim).to(dev)

    with torch.no_grad():
        X_init, _ = fusion([lt.to(dev) for lt in layer_tensors])
    idx = torch.randperm(N)[:min(n_inducing, N)]
    inducing_points = X_init[idx].clone()

    model = SVGPModel(inducing_points).to(dev)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model.train(); lik.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": lik.parameters(), "lr": lr},
        {"params": fusion.parameters(), "lr": fusion_lr},
    ])
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=N)

    dataset = torch.utils.data.TensorDataset(torch.arange(N), y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lt_dev = [lt.to(dev) for lt in layer_tensors]

    for epoch in range(n_epochs):
        for idx_b, y_b in loader:
            y_b = y_b.to(dev)
            optimizer.zero_grad()
            z, _ = fusion([lt[idx_b] for lt in lt_dev])
            loss = -mll(model(z), y_b)
            loss.backward()
            optimizer.step()

    model.eval(); lik.eval(); fusion.eval()
    return fusion, model, lik


# ─────────────────────────────────────────────
# Main CV loop
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "L8": {
        "type": "single_layer",
        "layer_idx": 8,
        "layers_needed": [8],
    },
    "WS-all": {
        "type": "weighted_sum",
        "layer_indices": list(range(10)),
        "layers_needed": list(range(10)),
        "fusion_lr": 0.05,
    },
    "Attn-top2": {
        "type": "attention",
        "layer_indices": [8, 6],
        "layers_needed": [8, 6],
        "fusion_lr": 0.01,
        "hidden_dim": 64,
    },
    "Attn-all": {
        "type": "attention",
        "layer_indices": list(range(10)),
        "layers_needed": list(range(10)),
        "fusion_lr": 0.01,
        "hidden_dim": 64,
    },
}


def run_cv(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(args.splits_5fold) as f:
        splits_5f = json.load(f)
    test_codes = splits_5f["test"]
    folds = splits_5f["folds"]
    n_folds = len(folds)
    logger.info(f"Loaded 5-fold splits: {n_folds} folds, test={len(test_codes)}")

    # Load all embeddings once
    code_layers, label_map = load_all_embeddings(
        args.embeddings, args.labels, n_layers=N_LAYERS,
    )

    # All layers needed across all models
    all_layer_indices = list(range(N_LAYERS))

    # Pre-build test arrays (fixed across folds)
    X_test, y_test = build_split_arrays(
        code_layers, label_map, test_codes, all_layer_indices,
    )
    logger.info(f"Test set: {len(y_test)} samples")

    # Results storage
    all_results = []

    for fold_id in sorted(folds.keys(), key=int):
        fold = folds[fold_id]
        train_codes = fold["train"]
        val_codes = fold["val"]
        seed = fold.get("seed", 42 + int(fold_id))

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_id} (seed={seed}, train={len(train_codes)}, val={len(val_codes)})")
        logger.info(f"{'='*60}")

        # Build train/val arrays
        X_train, y_train = build_split_arrays(
            code_layers, label_map, train_codes, all_layer_indices,
        )
        X_val, y_val = build_split_arrays(
            code_layers, label_map, val_codes, all_layer_indices,
        )
        logger.info(f"  Arrays: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        for model_name, cfg in MODEL_CONFIGS.items():
            logger.info(f"  --- {model_name} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            if cfg["type"] == "single_layer":
                oracle = train_single_layer_gp(
                    X_train, y_train, cfg["layer_idx"],
                    n_inducing=args.n_inducing, n_epochs=args.n_epochs,
                    batch_size=args.batch_size, lr=args.lr, device=args.device,
                )
                mu_val, var_val = predict_single_layer(oracle, X_val, cfg["layer_idx"])
                mu_test, var_test = predict_single_layer(oracle, X_test, cfg["layer_idx"])

            elif cfg["type"] == "weighted_sum":
                fusion, model, lik = train_ws_gp(
                    X_train, y_train, cfg["layer_indices"],
                    n_inducing=args.n_inducing, n_epochs=args.n_epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    fusion_lr=cfg["fusion_lr"], device=args.device,
                )
                mu_val, var_val = predict_fusion_gp(
                    fusion, model, lik, X_val, cfg["layer_indices"], args.device)
                mu_test, var_test = predict_fusion_gp(
                    fusion, model, lik, X_test, cfg["layer_indices"], args.device)

            elif cfg["type"] == "attention":
                fusion, model, lik = train_attn_gp(
                    X_train, y_train, cfg["layer_indices"],
                    n_inducing=args.n_inducing, n_epochs=args.n_epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    fusion_lr=cfg["fusion_lr"], hidden_dim=cfg["hidden_dim"],
                    device=args.device,
                )
                mu_val, var_val = predict_fusion_gp(
                    fusion, model, lik, X_val, cfg["layer_indices"], args.device)
                mu_test, var_test = predict_fusion_gp(
                    fusion, model, lik, X_test, cfg["layer_indices"], args.device)

            val_m = compute_metrics(y_val, mu_val, var_val)
            test_m = compute_metrics(y_test, mu_test, var_test)

            row = {
                "model": model_name,
                "fold": int(fold_id),
                "seed": seed,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "val_R2": val_m["R2"],
                "val_spearman_rho": val_m["spearman_rho"],
                "val_RMSE": val_m["RMSE"],
                "val_NLL": val_m["NLL"],
                "test_R2": test_m["R2"],
                "test_spearman_rho": test_m["spearman_rho"],
                "test_RMSE": test_m["RMSE"],
                "test_NLL": test_m["NLL"],
            }
            all_results.append(row)

            logger.info(
                f"    Val:  R²={val_m['R2']:.4f}  ρ={val_m['spearman_rho']:.4f}  "
                f"RMSE={val_m['RMSE']:.3f}"
            )
            logger.info(
                f"    Test: R²={test_m['R2']:.4f}  ρ={test_m['spearman_rho']:.4f}  "
                f"RMSE={test_m['RMSE']:.3f}"
            )

    # ─── Save per-fold results ───
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "cv_results.csv", index=False)
    logger.info(f"\nSaved per-fold results to {output_dir / 'cv_results.csv'}")

    # ─── Compute summary statistics ───
    summary_rows = []
    for model_name in MODEL_CONFIGS:
        mdf = df[df["model"] == model_name]
        row = {"model": model_name}
        for metric in ["val_R2", "val_spearman_rho", "val_RMSE", "val_NLL",
                       "test_R2", "test_spearman_rho", "test_RMSE", "test_NLL"]:
            row[f"mean_{metric}"] = mdf[metric].mean()
            row[f"std_{metric}"] = mdf[metric].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("5-FOLD CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Model':<12} {'Val R² (mean±std)':<22} {'Val ρ (mean±std)':<22} "
                f"{'Test R² (mean±std)':<22} {'Test ρ (mean±std)':<22}")
    logger.info("-" * 100)
    for _, r in summary_df.iterrows():
        logger.info(
            f"{r['model']:<12} "
            f"{r['mean_val_R2']:.4f}±{r['std_val_R2']:.4f}     "
            f"{r['mean_val_spearman_rho']:.4f}±{r['std_val_spearman_rho']:.4f}     "
            f"{r['mean_test_R2']:.4f}±{r['std_test_R2']:.4f}     "
            f"{r['mean_test_spearman_rho']:.4f}±{r['std_test_spearman_rho']:.4f}"
        )
    logger.info("=" * 80)

    # ─── Conclusion ───
    best_model = summary_df.loc[summary_df["mean_val_R2"].idxmax()]
    conclusion = {
        "best_model_by_val_R2": best_model["model"],
        "best_mean_val_R2": float(best_model["mean_val_R2"]),
        "best_std_val_R2": float(best_model["std_val_R2"]),
        "per_model": {
            r["model"]: {
                "mean_val_R2": float(r["mean_val_R2"]),
                "std_val_R2": float(r["std_val_R2"]),
                "mean_val_rho": float(r["mean_val_spearman_rho"]),
                "std_val_rho": float(r["std_val_spearman_rho"]),
                "mean_test_R2": float(r["mean_test_R2"]),
                "std_test_R2": float(r["std_test_R2"]),
                "mean_test_rho": float(r["mean_test_spearman_rho"]),
                "std_test_rho": float(r["std_test_spearman_rho"]),
            }
            for _, r in summary_df.iterrows()
        },
        "per_fold": all_results,
    }

    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(conclusion, f, indent=2)

    logger.info(f"\nBest model by mean val R²: {best_model['model']} "
                f"({best_model['mean_val_R2']:.4f}±{best_model['std_val_R2']:.4f})")

    # ─── Plots ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(MODEL_CONFIGS.keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Val R² boxplot
        val_r2_data = [df[df["model"] == m]["val_R2"].values for m in models]
        bp1 = axes[0].boxplot(val_r2_data, labels=models, patch_artist=True)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for patch, color in zip(bp1["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel("Val R²")
        axes[0].set_title("5-Fold Val R² Distribution")
        axes[0].grid(axis="y", alpha=0.3)

        # Test R² boxplot
        test_r2_data = [df[df["model"] == m]["test_R2"].values for m in models]
        bp2 = axes[1].boxplot(test_r2_data, labels=models, patch_artist=True)
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel("Test R² (CASF-2016)")
        axes[1].set_title("5-Fold Test R² Distribution")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "cv_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {output_dir / 'cv_comparison.png'}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(description="Stage 2.5: 5-Fold CV")
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits_5fold", type=str,
                        default="data/pdbbind_v2020/splits_5fold.json")
    parser.add_argument("--output", type=str,
                        default="results/stage2/cross_validation")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_cv(args)


if __name__ == "__main__":
    main()
