"""
scripts/pipeline/s09a_single_layer_probe.py
────────────────────────────────────────────
Stage 1: Single-Layer Probing — train an independent GP on each encoder
layer's mean-pooled embedding and evaluate per-layer predictive quality.

Experiments:
  E1.1 — Per-layer GP metrics (R², Spearman ρ, NLL)
  E1.2 — CKA similarity matrix across layers

Gate 1 decision:
  PROCEED if best non-final layer R² ≥ 0.9 × last layer R²

Usage:
    python scripts/pipeline/s09a_single_layer_probe.py \\
        --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/layer_probing \\
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_layer_data(
    emb_path: Path,
    labels_path: Path,
    splits_path: Path,
    n_layers: int = 10,
) -> dict:
    """Load multi-layer embeddings, labels, and splits.

    Returns dict with keys:
        'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        Each X is a dict: {layer_idx: ndarray (N, d)}
        Each y is ndarray (N,)
        'codes_train', 'codes_val', 'codes_test': list[str]
    """
    import pandas as pd

    # Load ALL embeddings into memory at once (faster than random key access)
    logger.info(f"  Loading {emb_path} into memory...")
    emb = dict(np.load(emb_path, allow_pickle=True))
    logger.info(f"  Loaded {len(emb)} keys")

    # Load labels
    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    # Build split → code mapping
    code_to_split = {}
    for split_name in ["train", "val", "test"]:
        for code in splits[split_name]:
            code_to_split[code] = split_name

    # Single pass: classify embeddings by split
    split_data = {s: {"codes": [], "layers": {i: [] for i in range(n_layers)}, "y": []}
                  for s in ["train", "val", "test"]}

    # Get all unique codes from embeddings
    codes_in_emb = set()
    for key in emb:
        if key.endswith("_layer_0"):
            codes_in_emb.add(key[:-8])  # strip "_layer_0"

    for code in sorted(codes_in_emb):
        if code not in code_to_split or code not in label_map:
            continue
        split_name = code_to_split[code]
        split_data[split_name]["codes"].append(code)
        split_data[split_name]["y"].append(label_map[code])
        for layer_idx in range(n_layers):
            split_data[split_name]["layers"][layer_idx].append(emb[f"{code}_layer_{layer_idx}"])

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


def compute_metrics(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray) -> dict:
    """Compute R², Spearman ρ, RMSE, and NLL."""
    from scipy.stats import spearmanr

    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rho, p_val = spearmanr(y_true, mu)

    rmse = np.sqrt(np.mean((y_true - mu) ** 2))

    # Gaussian NLL
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var)

    return {
        "R2": float(r2),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "RMSE": float(rmse),
        "NLL": float(nll),
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    X, Y: (N, d1) and (N, d2) — centered and then compared.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC(K, L) = ||Y^T X||_F^2 / (N-1)^2
    # but for linear CKA we compute:
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    return float(num / denom) if denom > 0 else 0.0


def run_layer_probing(args):
    """E1.1: Train GP per layer, collect metrics."""
    from bayesdiff.gp_oracle import GPOracle

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    data = load_layer_data(
        Path(args.embeddings),
        Path(args.labels),
        Path(args.splits),
    )

    n_layers = len(data["X_train"])
    d = data["X_train"][0].shape[1]
    logger.info(
        f"Layers: {n_layers}, dim: {d}, "
        f"train: {len(data['y_train'])}, val: {len(data['y_val'])}, "
        f"test: {len(data['y_test'])}"
    )

    # E1.1: Per-layer GP
    results = []
    for layer_idx in range(n_layers):
        logger.info(f"=== Layer {layer_idx}/{n_layers-1} ===")
        X_train = data["X_train"][layer_idx]
        y_train = data["y_train"]
        X_val = data["X_val"][layer_idx]
        y_val = data["y_val"]
        X_test = data["X_test"][layer_idx]
        y_test = data["y_test"]

        gp = GPOracle(d=d, n_inducing=args.n_inducing, device=args.device)
        train_info = gp.train(
            X_train, y_train,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=False,
        )

        # Val metrics
        mu_val, var_val = gp.predict(X_val)
        val_metrics = compute_metrics(y_val, mu_val, var_val)

        # Test metrics
        mu_test, var_test = gp.predict(X_test)
        test_metrics = compute_metrics(y_test, mu_test, var_test)

        row = {
            "layer_idx": layer_idx,
            "final_loss": float(train_info["loss"][-1]),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
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

        # Save per-layer GP model
        gp.save(output_dir / f"gp_layer_{layer_idx}.pt")

    # Save results CSV
    import pandas as pd

    df = pd.DataFrame(results)
    csv_path = output_dir / "layer_probing.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved per-layer metrics to {csv_path}")

    # E1.2: CKA similarity matrix (on val set for reasonable size)
    logger.info("Computing CKA similarity matrix...")
    cka_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(i, n_layers):
            cka_val = linear_cka(data["X_val"][i], data["X_val"][j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val

    np.save(output_dir / "cka_matrix.npy", cka_matrix)
    logger.info(f"Saved CKA matrix to {output_dir / 'cka_matrix.npy'}")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig L.1: Per-layer GP performance bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        layer_labels = [f"L{i}" for i in range(n_layers)]

        # R² bars
        ax = axes[0]
        val_r2 = [r["val_R2"] for r in results]
        test_r2 = [r["test_R2"] for r in results]
        x = np.arange(n_layers)
        ax.bar(x - 0.2, val_r2, 0.35, label="Val", alpha=0.8)
        ax.bar(x + 0.2, test_r2, 0.35, label="Test", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel("R²")
        ax.set_title("Per-Layer R²")
        ax.legend()

        # Spearman ρ bars
        ax = axes[1]
        val_rho = [r["val_spearman_rho"] for r in results]
        test_rho = [r["test_spearman_rho"] for r in results]
        ax.bar(x - 0.2, val_rho, 0.35, label="Val", alpha=0.8)
        ax.bar(x + 0.2, test_rho, 0.35, label="Test", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Per-Layer Spearman ρ")
        ax.legend()

        # NLL bars
        ax = axes[2]
        val_nll = [r["val_NLL"] for r in results]
        test_nll = [r["test_NLL"] for r in results]
        ax.bar(x - 0.2, val_nll, 0.35, label="Val", alpha=0.8)
        ax.bar(x + 0.2, test_nll, 0.35, label="Test", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel("NLL")
        ax.set_title("Per-Layer NLL (lower is better)")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "layer_probing.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved per-layer bar chart to {output_dir / 'layer_probing.png'}")

        # Fig L.2: CKA similarity heatmap
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layer_labels)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_labels)
        ax.set_title("CKA Similarity Matrix")
        # Add text annotations
        for i in range(n_layers):
            for j in range(n_layers):
                ax.text(j, i, f"{cka_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if cka_matrix[i, j] < 0.5 else "black")
        plt.colorbar(im, ax=ax, label="CKA")
        plt.tight_layout()
        plt.savefig(output_dir / "cka_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved CKA heatmap to {output_dir / 'cka_heatmap.png'}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # --- Gate 1 Decision ---
    last_layer_r2 = results[-1]["val_R2"]
    best_nonfinal_r2 = max(r["val_R2"] for r in results[:-1])
    best_nonfinal_layer = max(range(n_layers - 1), key=lambda i: results[i]["val_R2"])

    gate1_ratio = best_nonfinal_r2 / last_layer_r2 if last_layer_r2 > 0 else 0
    gate1_pass = gate1_ratio >= 0.9

    gate1_summary = {
        "last_layer_R2": last_layer_r2,
        "best_nonfinal_R2": best_nonfinal_r2,
        "best_nonfinal_layer": best_nonfinal_layer,
        "ratio": gate1_ratio,
        "threshold": 0.9,
        "gate1_pass": gate1_pass,
        "decision": "PROCEED to Stage 2" if gate1_pass else "STOP — multi-layer fusion unlikely to help",
    }

    with open(output_dir / "gate1_decision.json", "w") as f:
        json.dump(gate1_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("GATE 1 DECISION")
    logger.info(f"  Last layer (L{n_layers-1}) val R²: {last_layer_r2:.4f}")
    logger.info(f"  Best non-final (L{best_nonfinal_layer}) val R²: {best_nonfinal_r2:.4f}")
    logger.info(f"  Ratio: {gate1_ratio:.4f} (threshold: 0.9)")
    logger.info(f"  → {'PROCEED' if gate1_pass else 'STOP'}")
    logger.info("=" * 60)

    # Save full summary
    summary = {
        "n_layers": n_layers,
        "d": d,
        "n_train": len(data["y_train"]),
        "n_val": len(data["y_val"]),
        "n_test": len(data["y_test"]),
        "per_layer_results": results,
        "cka_matrix": cka_matrix.tolist(),
        "gate1": gate1_summary,
    }
    with open(output_dir / "stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Single-Layer Probing")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to all_multilayer_embeddings.npz")
    parser.add_argument("--labels", type=str,
                        default="data/pdbbind_v2020/labels.csv",
                        help="CSV with pdb_code and pkd columns")
    parser.add_argument("--splits", type=str,
                        default="data/pdbbind_v2020/splits.json",
                        help="JSON with train/val/test splits")
    parser.add_argument("--output", type=str,
                        default="results/stage2/layer_probing",
                        help="Output directory for results")
    parser.add_argument("--n_inducing", type=int, default=512,
                        help="Number of inducing points for SVGP")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Training epochs per GP")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for GP training")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for GP training")

    args = parser.parse_args()
    run_layer_probing(args)


if __name__ == "__main__":
    main()
