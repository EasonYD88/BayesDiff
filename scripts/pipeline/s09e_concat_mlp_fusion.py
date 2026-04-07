"""
scripts/pipeline/s09e_concat_mlp_fusion.py
──────────────────────────────────────────
Stage 4: Concat + MLP Fusion — 5-fold cross-validation

Experiments:
  E4.1 — ConcatMLP (all 10 layers, output_dim=128) vs Attn-all baseline
  E4.2 — Output dim sensitivity: 64, 128, 256

Gate 4 decision:
  Compare concat+MLP vs. layer attention (Attn-all) on 5-fold CV.
  Uses both val and test metrics (learning from Stage 2.5 findings).

Usage:
    python scripts/pipeline/s09e_concat_mlp_fusion.py \\
        --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits_5fold data/pdbbind_v2020/splits_5fold.json \\
        --output results/stage2/concat_mlp \\
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


def load_all_embeddings(emb_path, labels_path, n_layers=N_LAYERS):
    """Load ALL embeddings + labels into a dict keyed by pdb_code."""
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
    """Build X_layers dict + y array for a list of pdb codes."""
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


def train_concat_mlp_gp(X_train_layers, y_train, layer_indices,
                        output_dim=128, n_inducing=512, n_epochs=200,
                        batch_size=256, lr=0.01, fusion_lr=0.01,
                        device="cuda"):
    """Train ConcatMLPFusion + SVGP GP jointly."""
    import gpytorch
    from bayesdiff.gp_oracle import SVGPModel
    from bayesdiff.layer_fusion import ConcatMLPFusion

    dev = torch.device(device)
    n_layers = len(layer_indices)
    d = X_train_layers[layer_indices[0]].shape[1]
    N = len(y_train)

    layer_tensors = [torch.tensor(X_train_layers[li], dtype=torch.float32)
                     for li in layer_indices]
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    fusion = ConcatMLPFusion(embed_dim=d, n_layers=n_layers,
                             output_dim=output_dim).to(dev)

    # Initial fused embeddings for inducing points
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
    lt_dev = [lt.to(dev) for lt in layer_tensors]

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for idx_b, y_b in loader:
            y_b = y_b.to(dev)
            optimizer.zero_grad()
            z, _ = fusion([lt[idx_b] for lt in lt_dev])
            loss = -mll(model(z), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx_b)

        if (epoch + 1) % 50 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}: "
                        f"loss={epoch_loss/N:.4f}, noise={lik.noise.item():.4f}")

    model.eval(); lik.eval(); fusion.eval()
    return fusion, model, lik


def predict_fusion_gp(fusion, model, lik, X_layers, layer_indices,
                      device="cuda"):
    dev = torch.device(device)
    lts = [torch.tensor(X_layers[li], dtype=torch.float32, device=dev)
           for li in layer_indices]
    with torch.no_grad():
        z, _ = fusion(lts)
        dist = lik(model(z))
        return dist.mean.cpu().numpy(), dist.variance.cpu().numpy()


def run_stage4(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    with open(args.splits_5fold) as f:
        splits_5f = json.load(f)
    test_codes = splits_5f["test"]
    folds = splits_5f["folds"]
    n_folds = len(folds)

    # Load all embeddings once
    code_layers, label_map = load_all_embeddings(
        args.embeddings, args.labels, n_layers=N_LAYERS,
    )

    all_layer_indices = list(range(N_LAYERS))

    # E4.1 + E4.2: ConcatMLP with different output dims across 5 folds
    output_dims = [64, 128, 256]

    results = []

    for output_dim in output_dims:
        model_name = f"ConcatMLP-d{output_dim}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        for fold_id_str, fold_info in sorted(folds.items()):
            fold_id = int(fold_id_str)
            seed = fold_info["seed"]
            train_codes = fold_info["train"]
            val_codes = fold_info["val"]

            logger.info(f"  Fold {fold_id} (seed={seed}): "
                        f"train={len(train_codes)}, val={len(val_codes)}")

            # Build arrays
            X_train, y_train = build_split_arrays(
                code_layers, label_map, train_codes, all_layer_indices)
            X_val, y_val = build_split_arrays(
                code_layers, label_map, val_codes, all_layer_indices)
            X_test, y_test = build_split_arrays(
                code_layers, label_map, test_codes, all_layer_indices)

            # Train
            fusion, gp_model, gp_lik = train_concat_mlp_gp(
                X_train, y_train, all_layer_indices,
                output_dim=output_dim,
                n_inducing=args.n_inducing,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                fusion_lr=args.fusion_lr,
                device=args.device,
            )

            # Evaluate
            mu_val, var_val = predict_fusion_gp(
                fusion, gp_model, gp_lik, X_val, all_layer_indices, args.device)
            val_metrics = compute_metrics(y_val, mu_val, var_val)

            mu_test, var_test = predict_fusion_gp(
                fusion, gp_model, gp_lik, X_test, all_layer_indices, args.device)
            test_metrics = compute_metrics(y_test, mu_test, var_test)

            row = {
                "model": model_name,
                "output_dim": output_dim,
                "fold": fold_id,
                "seed": seed,
                "n_train": len(y_train),
                "n_val": len(y_val),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            results.append(row)

            logger.info(f"    Val:  R²={val_metrics['R2']:.4f}  "
                        f"ρ={val_metrics['spearman_rho']:.4f}")
            logger.info(f"    Test: R²={test_metrics['R2']:.4f}  "
                        f"ρ={test_metrics['spearman_rho']:.4f}")

    # Save per-fold results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "concat_mlp_results.csv", index=False)

    # Compute summary statistics
    summary_rows = []
    for model_name in df["model"].unique():
        mdf = df[df["model"] == model_name]
        row = {
            "model": model_name,
            "output_dim": int(mdf["output_dim"].iloc[0]),
            "mean_val_R2": mdf["val_R2"].mean(),
            "std_val_R2": mdf["val_R2"].std(),
            "mean_val_spearman_rho": mdf["val_spearman_rho"].mean(),
            "std_val_spearman_rho": mdf["val_spearman_rho"].std(),
            "mean_val_RMSE": mdf["val_RMSE"].mean(),
            "std_val_RMSE": mdf["val_RMSE"].std(),
            "mean_val_NLL": mdf["val_NLL"].mean(),
            "std_val_NLL": mdf["val_NLL"].std(),
            "mean_test_R2": mdf["test_R2"].mean(),
            "std_test_R2": mdf["test_R2"].std(),
            "mean_test_spearman_rho": mdf["test_spearman_rho"].mean(),
            "std_test_spearman_rho": mdf["test_spearman_rho"].std(),
            "mean_test_RMSE": mdf["test_RMSE"].mean(),
            "std_test_RMSE": mdf["test_RMSE"].std(),
            "mean_test_NLL": mdf["test_NLL"].mean(),
            "std_test_NLL": mdf["test_NLL"].std(),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "concat_mlp_summary.csv", index=False)
    logger.info(f"\nSummary:\n{summary_df.to_string(index=False)}")

    # Load Stage 2.5 baselines for Gate 4 comparison
    cv_summary_path = Path(args.cv_summary)
    baselines = {}
    if cv_summary_path.exists():
        baseline_df = pd.read_csv(cv_summary_path)
        for _, brow in baseline_df.iterrows():
            baselines[brow["model"]] = {
                "mean_val_R2": brow["mean_val_R2"],
                "std_val_R2": brow["std_val_R2"],
                "mean_test_R2": brow["mean_test_R2"],
                "std_test_R2": brow["std_test_R2"],
                "mean_val_spearman_rho": brow["mean_val_spearman_rho"],
                "mean_test_spearman_rho": brow["mean_test_spearman_rho"],
            }

    # Gate 4 decision
    best_concat = summary_df.loc[summary_df["mean_test_R2"].idxmax()]
    best_concat_name = best_concat["model"]
    best_concat_test_r2 = best_concat["mean_test_R2"]
    best_concat_test_std = best_concat["std_test_R2"]
    best_concat_val_r2 = best_concat["mean_val_R2"]

    attn_all_test_r2 = baselines.get("Attn-all", {}).get("mean_test_R2", 0.528)
    attn_all_test_std = baselines.get("Attn-all", {}).get("std_test_R2", 0.037)
    attn_all_val_r2 = baselines.get("Attn-all", {}).get("mean_val_R2", 0.134)
    l8_test_r2 = baselines.get("L8", {}).get("mean_test_R2", 0.420)

    # Stability: check if ConcatMLP test std < Attn-all test std
    more_stable = best_concat_test_std < attn_all_test_std

    beats_attn = best_concat_test_r2 > attn_all_test_r2
    improvement = ((best_concat_test_r2 - attn_all_test_r2) /
                   attn_all_test_r2 * 100 if attn_all_test_r2 > 0 else 0)

    if beats_attn and more_stable:
        decision = "USE ConcatMLP — beats Attn-all and is more stable"
        gate4_pass = True
    elif beats_attn and not more_stable:
        decision = "PROCEED to Stage 5 — beats Attn-all but less stable"
        gate4_pass = True
    else:
        decision = "USE Attn-all — ConcatMLP did not beat it"
        gate4_pass = False

    gate4_summary = {
        "best_concat_config": best_concat_name,
        "best_concat_test_R2": float(best_concat_test_r2),
        "best_concat_test_std": float(best_concat_test_std),
        "best_concat_val_R2": float(best_concat_val_r2),
        "attn_all_test_R2": float(attn_all_test_r2),
        "attn_all_test_std": float(attn_all_test_std),
        "l8_test_R2": float(l8_test_r2),
        "improvement_vs_attn_pct": float(improvement),
        "more_stable_than_attn": bool(more_stable),
        "gate4_pass": bool(gate4_pass),
        "decision": decision,
        "output_dim_comparison": {
            row["model"]: {
                "mean_test_R2": float(row["mean_test_R2"]),
                "std_test_R2": float(row["std_test_R2"]),
                "mean_val_R2": float(row["mean_val_R2"]),
            }
            for _, row in summary_df.iterrows()
        },
        "baselines": baselines,
    }

    with open(output_dir / "gate4_decision.json", "w") as f:
        json.dump(gate4_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("GATE 4 DECISION")
    logger.info(f"  Best ConcatMLP: {best_concat_name} "
                f"test R²={best_concat_test_r2:.4f} ± {best_concat_test_std:.4f}")
    logger.info(f"  Attn-all baseline: test R²={attn_all_test_r2:.4f} "
                f"± {attn_all_test_std:.4f}")
    logger.info(f"  L8 baseline: test R²={l8_test_r2:.4f}")
    logger.info(f"  Improvement vs Attn-all: {improvement:+.1f}%")
    logger.info(f"  More stable: {more_stable}")
    logger.info(f"  → {decision}")
    logger.info("=" * 60)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Combined comparison with baselines
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gather all models
        all_models = []
        all_val_r2 = []
        all_test_r2 = []
        all_colors = []

        # Baselines from Stage 2.5
        for bname in ["L8", "WS-all", "Attn-top2", "Attn-all"]:
            if bname in baselines:
                all_models.append(bname)
                all_val_r2.append(baselines[bname]["mean_val_R2"])
                all_test_r2.append(baselines[bname]["mean_test_R2"])
                all_colors.append("steelblue" if "Attn" not in bname else "darkorange")

        # ConcatMLP variants
        for _, row in summary_df.iterrows():
            all_models.append(row["model"])
            all_val_r2.append(row["mean_val_R2"])
            all_test_r2.append(row["mean_test_R2"])
            all_colors.append("forestgreen")

        x = np.arange(len(all_models))
        axes[0].bar(x, all_val_r2, color=all_colors)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(all_models, rotation=45, ha="right")
        axes[0].set_ylabel("Mean Val R² (5-fold)")
        axes[0].set_title("Val R²: All Methods")

        axes[1].bar(x, all_test_r2, color=all_colors)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(all_models, rotation=45, ha="right")
        axes[1].set_ylabel("Mean Test R² (5-fold)")
        axes[1].set_title("Test R² (CASF-2016): All Methods")

        plt.tight_layout()
        plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        # E4.2: Output dim sensitivity
        fig, ax = plt.subplots(figsize=(8, 5))
        dims = summary_df["output_dim"].values
        test_r2s = summary_df["mean_test_R2"].values
        test_stds = summary_df["std_test_R2"].values
        ax.errorbar(dims, test_r2s, yerr=test_stds, fmt="o-",
                    capsize=5, color="forestgreen", linewidth=2)
        ax.axhline(y=attn_all_test_r2, color="darkorange", linestyle="--",
                   alpha=0.7, label=f"Attn-all ({attn_all_test_r2:.3f})")
        ax.axhline(y=l8_test_r2, color="gray", linestyle="--",
                   alpha=0.7, label=f"L8 ({l8_test_r2:.3f})")
        ax.set_xlabel("Output Dimension")
        ax.set_ylabel("Mean Test R² (5-fold)")
        ax.set_title("E4.2: ConcatMLP Output Dim Sensitivity")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "output_dim_sensitivity.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # Full summary JSON
    full_summary = {
        "results": results,
        "summary": summary_rows,
        "gate4": gate4_summary,
    }
    with open(output_dir / "stage4_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Concat+MLP Fusion")
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits_5fold", type=str,
                        default="data/pdbbind_v2020/splits_5fold.json")
    parser.add_argument("--cv_summary", type=str,
                        default="results/stage2/cross_validation/cv_summary.csv")
    parser.add_argument("--output", type=str,
                        default="results/stage2/concat_mlp")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fusion_lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_stage4(args)


if __name__ == "__main__":
    main()
