"""
scripts/pipeline/s09b_weighted_sum_fusion.py
─────────────────────────────────────────────
Stage 2: Weighted Sum Fusion — jointly train WeightedSumFusion + GP
on multi-layer embeddings from Stage 0.

Experiments:
  E2.1 — Weighted sum over top-k layers (k=2,4,all) vs best single layer
  E2.2 — Inspect learned layer weights

Gate 2 decision:
  PROCEED if weighted sum R² > best single layer R² on val set

Usage:
    python scripts/pipeline/s09b_weighted_sum_fusion.py \\
        --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/weighted_sum \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_layer_data(emb_path, labels_path, splits_path, n_layers=10):
    """Load multi-layer embeddings grouped by split.

    Returns dict with X_{split}[layer_idx] = (N, d), y_{split} = (N,).
    """
    import pandas as pd

    logger.info(f"  Loading {emb_path} into memory...")
    emb = dict(np.load(emb_path, allow_pickle=True))
    logger.info(f"  Loaded {len(emb)} keys")

    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))

    with open(splits_path) as f:
        splits = json.load(f)

    code_to_split = {}
    for split_name in ["train", "val", "test"]:
        for code in splits[split_name]:
            code_to_split[code] = split_name

    split_data = {
        s: {"codes": [], "layers": {i: [] for i in range(n_layers)}, "y": []}
        for s in ["train", "val", "test"]
    }

    codes_in_emb = set()
    for key in emb:
        if key.endswith("_layer_0"):
            codes_in_emb.add(key[:-8])

    for code in sorted(codes_in_emb):
        if code not in code_to_split or code not in label_map:
            continue
        split_name = code_to_split[code]
        split_data[split_name]["codes"].append(code)
        split_data[split_name]["y"].append(label_map[code])
        for layer_idx in range(n_layers):
            split_data[split_name]["layers"][layer_idx].append(
                emb[f"{code}_layer_{layer_idx}"]
            )

    result = {}
    for split_name in ["train", "val", "test"]:
        sd = split_data[split_name]
        X_per_layer = {}
        for layer_idx in range(n_layers):
            X_per_layer[layer_idx] = np.stack(sd["layers"][layer_idx], axis=0)
        result[f"X_{split_name}"] = X_per_layer
        result[f"y_{split_name}"] = np.array(sd["y"], dtype=np.float32)
        result[f"codes_{split_name}"] = sd["codes"]

    return result


def compute_metrics(y_true, mu, var):
    """Compute R², Spearman ρ, RMSE, and NLL."""
    from scipy.stats import spearmanr

    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rho, p_val = spearmanr(y_true, mu)
    rmse = np.sqrt(np.mean((y_true - mu) ** 2))
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var)
    return {
        "R2": float(r2),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "RMSE": float(rmse),
        "NLL": float(nll),
    }


def train_weighted_sum_gp(
    X_train_layers: dict,
    y_train: np.ndarray,
    layer_indices: list[int],
    n_inducing: int = 512,
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.01,
    fusion_lr: float = 0.05,
    device: str = "cuda",
) -> tuple:
    """Jointly train WeightedSumFusion + SVGP GP.

    Parameters
    ----------
    X_train_layers : dict
        {layer_idx: ndarray (N, d)} for all layers.
    y_train : ndarray (N,)
    layer_indices : list[int]
        Which layer indices to fuse.
    n_inducing, n_epochs, batch_size, lr : GP hyperparams
    fusion_lr : float
        Learning rate for fusion weights (higher than GP to let them adapt first).
    device : str

    Returns
    -------
    fusion_module, gp_model, gp_likelihood, history
    """
    import gpytorch
    from bayesdiff.gp_oracle import SVGPModel
    from bayesdiff.layer_fusion import WeightedSumFusion

    dev = torch.device(device)
    n_layers = len(layer_indices)
    d = X_train_layers[layer_indices[0]].shape[1]
    N = len(y_train)

    # Stack layer tensors: (N, n_layers, d)
    layer_tensors = []
    for li in layer_indices:
        layer_tensors.append(
            torch.tensor(X_train_layers[li], dtype=torch.float32)
        )

    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Initialize fusion module
    fusion = WeightedSumFusion(n_layers=n_layers).to(dev)

    # Compute initial fused embeddings for inducing point init
    with torch.no_grad():
        init_embeds = [lt.to(dev) for lt in layer_tensors]
        X_fused_init, _ = fusion(init_embeds)

    # Initialize inducing points
    if N <= n_inducing:
        inducing_points = X_fused_init.clone()
    else:
        idx = torch.randperm(N)[:n_inducing]
        inducing_points = X_fused_init[idx].clone()

    gp_model = SVGPModel(inducing_points).to(dev)
    gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)

    gp_model.train()
    gp_likelihood.train()

    # Separate optimizers: fusion weights converge faster
    optimizer = torch.optim.Adam([
        {"params": gp_model.parameters(), "lr": lr},
        {"params": gp_likelihood.parameters(), "lr": lr},
        {"params": fusion.parameters(), "lr": fusion_lr},
    ])

    mll = gpytorch.mlls.VariationalELBO(gp_likelihood, gp_model, num_data=N)

    # Create dataset with indices (we need full layer stack per sample)
    dataset = torch.utils.data.TensorDataset(
        torch.arange(N), y_tensor,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    # Pre-move layer tensors to device
    layer_tensors_dev = [lt.to(dev) for lt in layer_tensors]

    history = {"loss": [], "weights": []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for idx_batch, y_batch in loader:
            y_batch = y_batch.to(dev)
            optimizer.zero_grad()

            # Gather per-layer embeddings for this batch
            batch_layers = [lt[idx_batch] for lt in layer_tensors_dev]
            z_fuse, _ = fusion(batch_layers)

            output = gp_model(z_fuse)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx_batch)

        epoch_loss /= N
        history["loss"].append(epoch_loss)

        # Record weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                _, w = fusion([lt[:1] for lt in layer_tensors_dev])
                history["weights"].append(w.cpu().tolist())

        if (epoch + 1) % 50 == 0:
            noise = gp_likelihood.noise.item()
            with torch.no_grad():
                _, w = fusion([lt[:1] for lt in layer_tensors_dev])
            logger.info(
                f"    Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                f"noise={noise:.4f}, weights={[f'{x:.3f}' for x in w.cpu().tolist()]}"
            )

    gp_model.eval()
    gp_likelihood.eval()
    fusion.eval()

    return fusion, gp_model, gp_likelihood, history


def predict_fused(
    fusion, gp_model, gp_likelihood, X_layers, layer_indices, device="cuda",
):
    """Predict using fusion + GP pipeline."""
    dev = torch.device(device)
    layer_tensors = [
        torch.tensor(X_layers[li], dtype=torch.float32, device=dev)
        for li in layer_indices
    ]

    with torch.no_grad():
        z_fuse, weights = fusion(layer_tensors)
        pred_dist = gp_likelihood(gp_model(z_fuse))
        mu = pred_dist.mean.cpu().numpy()
        var = pred_dist.variance.cpu().numpy()

    return mu, var, weights.cpu().numpy()


def run_weighted_sum(args):
    """E2.1 + E2.2: Weighted sum fusion experiments."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    data = load_layer_data(
        Path(args.embeddings), Path(args.labels), Path(args.splits),
    )

    n_total_layers = len(data["X_train"])
    d = data["X_train"][0].shape[1]
    logger.info(
        f"Total layers: {n_total_layers}, dim: {d}, "
        f"train: {len(data['y_train'])}, val: {len(data['y_val'])}, "
        f"test: {len(data['y_test'])}"
    )

    # Load Stage 1 baseline for comparison
    stage1_csv = Path(args.stage1_results)
    if stage1_csv.exists():
        import pandas as pd
        stage1_df = pd.read_csv(stage1_csv)
        best_single_layer = stage1_df.loc[stage1_df["val_R2"].idxmax()]
        best_layer_idx = int(best_single_layer["layer_idx"])
        best_layer_val_r2 = float(best_single_layer["val_R2"])
        best_layer_val_rho = float(best_single_layer["val_spearman_rho"])
        logger.info(
            f"Stage 1 best single layer: L{best_layer_idx} "
            f"(val R²={best_layer_val_r2:.4f}, ρ={best_layer_val_rho:.4f})"
        )
        # Rank layers by val R² to pick top-k
        layer_ranking = stage1_df.sort_values("val_R2", ascending=False)["layer_idx"].tolist()
        layer_ranking = [int(x) for x in layer_ranking]
    else:
        logger.warning("Stage 1 results not found, using default layer ranking")
        best_layer_idx = n_total_layers - 1
        best_layer_val_r2 = 0.0
        best_layer_val_rho = 0.0
        layer_ranking = list(range(n_total_layers))

    # E2.1: Weighted sum over top-k layers
    k_configs = {
        "top2": layer_ranking[:2],
        "top4": layer_ranking[:4],
        "all": list(range(n_total_layers)),
    }

    results = []
    all_weights = {}

    for config_name, layer_indices in k_configs.items():
        logger.info(f"=== E2.1: config={config_name}, layers={layer_indices} ===")

        fusion, gp, gp_lik, history = train_weighted_sum_gp(
            X_train_layers=data["X_train"],
            y_train=data["y_train"],
            layer_indices=layer_indices,
            n_inducing=args.n_inducing,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fusion_lr=args.fusion_lr,
            device=args.device,
        )

        # Val metrics
        mu_val, var_val, w_val = predict_fused(
            fusion, gp, gp_lik, data["X_val"], layer_indices, args.device,
        )
        val_metrics = compute_metrics(data["y_val"], mu_val, var_val)

        # Test metrics
        mu_test, var_test, w_test = predict_fused(
            fusion, gp, gp_lik, data["X_test"], layer_indices, args.device,
        )
        test_metrics = compute_metrics(data["y_test"], mu_test, var_test)

        # E2.2: learned weights
        learned_weights = {
            f"L{li}": float(w_val[i])
            for i, li in enumerate(layer_indices)
        }
        all_weights[config_name] = learned_weights

        row = {
            "config": config_name,
            "layers": str(layer_indices),
            "n_layers_fused": len(layer_indices),
            "final_loss": float(history["loss"][-1]),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "learned_weights": json.dumps(learned_weights),
        }
        results.append(row)

        logger.info(
            f"  Val:  R²={val_metrics['R2']:.4f}  ρ={val_metrics['spearman_rho']:.4f}  "
            f"RMSE={val_metrics['RMSE']:.3f}  NLL={val_metrics['NLL']:.3f}"
        )
        logger.info(
            f"  Test: R²={test_metrics['R2']:.4f}  ρ={test_metrics['spearman_rho']:.4f}  "
            f"RMSE={test_metrics['RMSE']:.3f}  NLL={test_metrics['NLL']:.3f}"
        )
        logger.info(f"  Weights: {learned_weights}")

        # Save model
        torch.save({
            "fusion_state": fusion.state_dict(),
            "layer_indices": layer_indices,
        }, output_dir / f"fusion_{config_name}.pt")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = output_dir / "weighted_sum_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # --- Gate 2 Decision ---
    best_ws = max(results, key=lambda r: r["val_R2"])
    best_ws_r2 = best_ws["val_R2"]
    best_ws_config = best_ws["config"]

    gate2_pass = best_ws_r2 > best_layer_val_r2
    improvement = (best_ws_r2 - best_layer_val_r2) / best_layer_val_r2 * 100 if best_layer_val_r2 > 0 else 0

    gate2_summary = {
        "best_single_layer": best_layer_idx,
        "best_single_layer_val_R2": best_layer_val_r2,
        "best_single_layer_val_rho": best_layer_val_rho,
        "best_weighted_sum_config": best_ws_config,
        "best_weighted_sum_val_R2": best_ws_r2,
        "best_weighted_sum_val_rho": best_ws["val_spearman_rho"],
        "improvement_pct": improvement,
        "gate2_pass": gate2_pass,
        "decision": (
            f"PROCEED to Stage 3 — weighted sum ({best_ws_config}) val R²={best_ws_r2:.4f} "
            f"vs best single layer L{best_layer_idx} val R²={best_layer_val_r2:.4f} "
            f"({improvement:+.1f}%)"
            if gate2_pass
            else f"STOP — weighted sum did not beat best single layer L{best_layer_idx}"
        ),
        "all_weights": all_weights,
    }

    with open(output_dir / "gate2_decision.json", "w") as f:
        json.dump(gate2_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("GATE 2 DECISION")
    logger.info(f"  Best single layer: L{best_layer_idx} val R²={best_layer_val_r2:.4f}")
    logger.info(f"  Best weighted sum: {best_ws_config} val R²={best_ws_r2:.4f}")
    logger.info(f"  Improvement: {improvement:+.1f}%")
    logger.info(f"  → {'PROCEED' if gate2_pass else 'STOP'}")
    logger.info("=" * 60)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig L.3: Learned layer weights for 'all' config
        if "all" in all_weights:
            fig, ax = plt.subplots(figsize=(10, 5))
            ws = all_weights["all"]
            layers = list(ws.keys())
            vals = list(ws.values())
            colors = ["#ff7f0e" if v == max(vals) else "#1f77b4" for v in vals]
            ax.bar(layers, vals, color=colors)
            ax.set_ylabel("Learned Weight (β)")
            ax.set_xlabel("Encoder Layer")
            ax.set_title("WeightedSumFusion: Learned Layer Weights")
            ax.axhline(y=1.0 / len(vals), color="gray", linestyle="--",
                       alpha=0.5, label="Uniform")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "learned_weights.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved weight plot to {output_dir / 'learned_weights.png'}")

        # Fig L.4: Comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = [f"Best Single\n(L{best_layer_idx})"] + [
            r["config"] for r in results
        ]
        val_r2s = [best_layer_val_r2] + [r["val_R2"] for r in results]
        val_rhos = [best_layer_val_rho] + [r["val_spearman_rho"] for r in results]

        x = np.arange(len(labels))
        axes[0].bar(x, val_r2s, color=["gray"] + ["steelblue"] * len(results))
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Val R²")
        axes[0].set_title("E2.1: Weighted Sum vs. Best Single Layer")

        axes[1].bar(x, val_rhos, color=["gray"] + ["steelblue"] * len(results))
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Val Spearman ρ")
        axes[1].set_title("E2.1: Weighted Sum vs. Best Single Layer")

        plt.tight_layout()
        plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved comparison plot to {output_dir / 'comparison.png'}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # Full summary
    summary = {
        "n_total_layers": n_total_layers,
        "d": d,
        "n_train": len(data["y_train"]),
        "n_val": len(data["y_val"]),
        "n_test": len(data["y_test"]),
        "results": results,
        "gate2": gate2_summary,
    }
    with open(output_dir / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Weighted Sum Fusion")
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--stage1_results", type=str,
                        default="results/stage2/layer_probing/layer_probing.csv")
    parser.add_argument("--output", type=str,
                        default="results/stage2/weighted_sum")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fusion_lr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_weighted_sum(args)


if __name__ == "__main__":
    main()
