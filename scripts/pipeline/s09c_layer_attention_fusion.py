"""
scripts/pipeline/s09c_layer_attention_fusion.py
────────────────────────────────────────────────
Stage 3: Layer Attention Fusion — input-dependent layer weighting
jointly trained with GP.

Experiments:
  E3.1 — Layer attention over same layer sets as E2.1
  E3.2 — Per-sample weight variance analysis (do weights vary?)

Gate 3 decision:
  Compare layer attention vs weighted sum (and best single layer).
  If attention > weighted sum → PROCEED (input-dependent weighting matters)
  If attention ≈ weighted sum → USE weighted sum (simpler)

Usage:
    python scripts/pipeline/s09c_layer_attention_fusion.py \\
        --embeddings results/multilayer_embeddings/all_multilayer_embeddings.npz \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/layer_attention \\
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
    """Load multi-layer embeddings grouped by split."""
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


def train_layer_attention_gp(
    X_train_layers: dict,
    y_train: np.ndarray,
    layer_indices: list[int],
    n_inducing: int = 512,
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.01,
    fusion_lr: float = 0.01,
    hidden_dim: int = 64,
    device: str = "cuda",
) -> tuple:
    """Jointly train LayerAttentionFusion + SVGP GP.

    Returns
    -------
    fusion_module, gp_model, gp_likelihood, history
    """
    import gpytorch
    from bayesdiff.gp_oracle import SVGPModel
    from bayesdiff.layer_fusion import LayerAttentionFusion

    dev = torch.device(device)
    n_layers = len(layer_indices)
    d = X_train_layers[layer_indices[0]].shape[1]
    N = len(y_train)

    layer_tensors = []
    for li in layer_indices:
        layer_tensors.append(
            torch.tensor(X_train_layers[li], dtype=torch.float32)
        )

    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    fusion = LayerAttentionFusion(embed_dim=d, hidden_dim=hidden_dim).to(dev)

    # Initial fused embeddings for inducing points
    with torch.no_grad():
        init_embeds = [lt.to(dev) for lt in layer_tensors]
        X_fused_init, _ = fusion(init_embeds)

    if N <= n_inducing:
        inducing_points = X_fused_init.clone()
    else:
        idx = torch.randperm(N)[:n_inducing]
        inducing_points = X_fused_init[idx].clone()

    gp_model = SVGPModel(inducing_points).to(dev)
    gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)

    gp_model.train()
    gp_likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": gp_model.parameters(), "lr": lr},
        {"params": gp_likelihood.parameters(), "lr": lr},
        {"params": fusion.parameters(), "lr": fusion_lr},
    ])

    mll = gpytorch.mlls.VariationalELBO(gp_likelihood, gp_model, num_data=N)

    dataset = torch.utils.data.TensorDataset(torch.arange(N), y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    layer_tensors_dev = [lt.to(dev) for lt in layer_tensors]

    history = {"loss": []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for idx_batch, y_batch in loader:
            y_batch = y_batch.to(dev)
            optimizer.zero_grad()

            batch_layers = [lt[idx_batch] for lt in layer_tensors_dev]
            z_fuse, _ = fusion(batch_layers)

            output = gp_model(z_fuse)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx_batch)

        epoch_loss /= N
        history["loss"].append(epoch_loss)

        if (epoch + 1) % 50 == 0:
            noise = gp_likelihood.noise.item()
            logger.info(
                f"    Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                f"noise={noise:.4f}"
            )

    gp_model.eval()
    gp_likelihood.eval()
    fusion.eval()

    return fusion, gp_model, gp_likelihood, history


def predict_fused(
    fusion, gp_model, gp_likelihood, X_layers, layer_indices, device="cuda",
):
    """Predict using fusion + GP pipeline. Returns mu, var, weights."""
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


def analyze_weight_variance(weights_val, weights_test, layer_indices, output_dir):
    """E3.2: Analyze per-sample layer weight variance.

    Do attention weights actually vary across molecules, or do they
    collapse to a fixed weighting (making attention equivalent to
    weighted sum)?
    """
    analysis = {}

    for name, W in [("val", weights_val), ("test", weights_test)]:
        # W: (N, n_layers)
        # Per-layer statistics
        mean_weights = W.mean(axis=0)
        std_weights = W.std(axis=0)

        # Entropy of weights per sample (higher = more uniform)
        eps = 1e-10
        entropy = -np.sum(W * np.log(W + eps), axis=1)
        max_entropy = np.log(W.shape[1])

        # Coefficient of variation per layer (std/mean)
        cv = std_weights / (mean_weights + eps)

        analysis[name] = {
            "mean_weights": {
                f"L{li}": float(mean_weights[i])
                for i, li in enumerate(layer_indices)
            },
            "std_weights": {
                f"L{li}": float(std_weights[i])
                for i, li in enumerate(layer_indices)
            },
            "cv_weights": {
                f"L{li}": float(cv[i])
                for i, li in enumerate(layer_indices)
            },
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std()),
            "max_entropy": float(max_entropy),
            "normalized_entropy": float(entropy.mean() / max_entropy),
        }

    # Interpretation
    #   normalized_entropy close to 1.0 → uniform weights → attention ≈ weighted sum
    #   normalized_entropy << 1.0 → peaked weights → input-dependent
    #   high CV → weights vary a lot across samples → attention is useful
    max_cv = max(
        max(v for v in analysis["val"]["cv_weights"].values()),
        max(v for v in analysis["test"]["cv_weights"].values()),
    )

    analysis["interpretation"] = {
        "max_cv": float(max_cv),
        "weights_vary": max_cv > 0.1,
        "summary": (
            "Attention weights vary meaningfully across samples (CV > 0.1)"
            if max_cv > 0.1
            else "Attention weights are nearly constant — equivalent to weighted sum"
        ),
    }

    return analysis


def run_layer_attention(args):
    """E3.1 + E3.2: Layer attention fusion experiments."""
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

    # Load Stage 1 baseline
    stage1_csv = Path(args.stage1_results)
    if stage1_csv.exists():
        import pandas as pd
        stage1_df = pd.read_csv(stage1_csv)
        best_single = stage1_df.loc[stage1_df["val_R2"].idxmax()]
        best_layer_idx = int(best_single["layer_idx"])
        best_layer_val_r2 = float(best_single["val_R2"])
        best_layer_val_rho = float(best_single["val_spearman_rho"])
        layer_ranking = stage1_df.sort_values("val_R2", ascending=False)["layer_idx"].tolist()
        layer_ranking = [int(x) for x in layer_ranking]
    else:
        best_layer_idx = n_total_layers - 1
        best_layer_val_r2 = 0.0
        best_layer_val_rho = 0.0
        layer_ranking = list(range(n_total_layers))

    # Load Stage 2 baseline
    stage2_json = Path(args.stage2_results)
    if stage2_json.exists():
        with open(stage2_json) as f:
            stage2 = json.load(f)
        best_ws_r2 = stage2["best_weighted_sum_val_R2"]
        best_ws_config = stage2["best_weighted_sum_config"]
        logger.info(
            f"Stage 1 best: L{best_layer_idx} val R²={best_layer_val_r2:.4f}"
        )
        logger.info(
            f"Stage 2 best: {best_ws_config} val R²={best_ws_r2:.4f}"
        )
    else:
        best_ws_r2 = 0.0
        best_ws_config = "N/A"

    baseline_r2 = max(best_layer_val_r2, best_ws_r2)

    # E3.1: Layer attention over same configs as E2.1
    k_configs = {
        "top2": layer_ranking[:2],
        "top4": layer_ranking[:4],
        "all": list(range(n_total_layers)),
    }

    results = []
    all_weight_analysis = {}

    for config_name, layer_indices in k_configs.items():
        logger.info(f"=== E3.1: config={config_name}, layers={layer_indices} ===")

        fusion, gp, gp_lik, history = train_layer_attention_gp(
            X_train_layers=data["X_train"],
            y_train=data["y_train"],
            layer_indices=layer_indices,
            n_inducing=args.n_inducing,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fusion_lr=args.fusion_lr,
            hidden_dim=args.hidden_dim,
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

        # E3.2: Weight variance analysis
        weight_analysis = analyze_weight_variance(
            w_val, w_test, layer_indices, output_dir,
        )
        all_weight_analysis[config_name] = weight_analysis

        # Mean weights for summary
        mean_w_val = {
            f"L{li}": float(w_val[:, i].mean())
            for i, li in enumerate(layer_indices)
        }

        row = {
            "config": config_name,
            "layers": str(layer_indices),
            "n_layers_fused": len(layer_indices),
            "final_loss": float(history["loss"][-1]),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "mean_weights": json.dumps(mean_w_val),
            "weight_entropy_norm": weight_analysis["val"]["normalized_entropy"],
            "max_cv": weight_analysis["interpretation"]["max_cv"],
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
        logger.info(f"  Mean weights (val): {mean_w_val}")
        logger.info(
            f"  Weight entropy (normalized): {weight_analysis['val']['normalized_entropy']:.4f}"
        )
        logger.info(f"  Max CV: {weight_analysis['interpretation']['max_cv']:.4f}")
        logger.info(f"  {weight_analysis['interpretation']['summary']}")

        # Save model
        torch.save({
            "fusion_state": fusion.state_dict(),
            "layer_indices": layer_indices,
            "embed_dim": d,
            "hidden_dim": args.hidden_dim,
        }, output_dir / f"attn_{config_name}.pt")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = output_dir / "layer_attention_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Save weight analysis
    with open(output_dir / "weight_variance_analysis.json", "w") as f:
        json.dump(all_weight_analysis, f, indent=2)

    # --- Gate 3 Decision ---
    best_attn = max(results, key=lambda r: r["val_R2"])
    best_attn_r2 = best_attn["val_R2"]
    best_attn_config = best_attn["config"]

    # Compare against best of (single layer, weighted sum)
    beats_baseline = best_attn_r2 > baseline_r2
    improvement_vs_single = (
        (best_attn_r2 - best_layer_val_r2) / best_layer_val_r2 * 100
        if best_layer_val_r2 > 0 else 0
    )
    improvement_vs_ws = (
        (best_attn_r2 - best_ws_r2) / best_ws_r2 * 100
        if best_ws_r2 > 0 else 0
    )

    gate3_summary = {
        "best_single_layer": best_layer_idx,
        "best_single_layer_val_R2": best_layer_val_r2,
        "best_weighted_sum_config": best_ws_config,
        "best_weighted_sum_val_R2": best_ws_r2,
        "best_attention_config": best_attn_config,
        "best_attention_val_R2": best_attn_r2,
        "best_attention_val_rho": best_attn["val_spearman_rho"],
        "improvement_vs_single_pct": improvement_vs_single,
        "improvement_vs_ws_pct": improvement_vs_ws,
        "gate3_pass": beats_baseline,
        "decision": (
            f"PROCEED — layer attention ({best_attn_config}) val R²={best_attn_r2:.4f} "
            f"beats baseline R²={baseline_r2:.4f} "
            f"(vs single: {improvement_vs_single:+.1f}%, vs WS: {improvement_vs_ws:+.1f}%)"
            if beats_baseline
            else f"STOP — layer attention ({best_attn_config}) val R²={best_attn_r2:.4f} "
            f"did not beat baseline R²={baseline_r2:.4f}"
        ),
        "weight_analysis_summary": {
            config_name: {
                "weights_vary": all_weight_analysis[config_name]["interpretation"]["weights_vary"],
                "max_cv": all_weight_analysis[config_name]["interpretation"]["max_cv"],
                "summary": all_weight_analysis[config_name]["interpretation"]["summary"],
            }
            for config_name in all_weight_analysis
        },
    }

    with open(output_dir / "gate3_decision.json", "w") as f:
        json.dump(gate3_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("GATE 3 DECISION")
    logger.info(f"  Best single layer: L{best_layer_idx} val R²={best_layer_val_r2:.4f}")
    logger.info(f"  Best weighted sum: {best_ws_config} val R²={best_ws_r2:.4f}")
    logger.info(f"  Best attention: {best_attn_config} val R²={best_attn_r2:.4f}")
    logger.info(f"  vs single layer: {improvement_vs_single:+.1f}%")
    logger.info(f"  vs weighted sum: {improvement_vs_ws:+.1f}%")
    logger.info(f"  → {'PROCEED' if beats_baseline else 'STOP'}")
    logger.info("=" * 60)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig: Comparison bar chart (single, WS, attention)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels_plot = [f"Best Single\n(L{best_layer_idx})"]
        val_r2s = [best_layer_val_r2]
        val_rhos = [best_layer_val_rho]
        colors = ["gray"]

        # Load WS results if available
        ws_csv = Path(args.stage2_csv)
        if ws_csv.exists():
            ws_df = pd.read_csv(ws_csv)
            for _, row in ws_df.iterrows():
                labels_plot.append(f"WS-{row['config']}")
                val_r2s.append(row["val_R2"])
                val_rhos.append(row["val_spearman_rho"])
                colors.append("steelblue")

        for r in results:
            labels_plot.append(f"Attn-{r['config']}")
            val_r2s.append(r["val_R2"])
            val_rhos.append(r["val_spearman_rho"])
            colors.append("darkorange")

        x = np.arange(len(labels_plot))
        axes[0].bar(x, val_r2s, color=colors)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels_plot, rotation=45, ha="right")
        axes[0].set_ylabel("Val R²")
        axes[0].set_title("E3.1: Layer Attention vs. Weighted Sum vs. Single")

        axes[1].bar(x, val_rhos, color=colors)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels_plot, rotation=45, ha="right")
        axes[1].set_ylabel("Val Spearman ρ")
        axes[1].set_title("E3.1: Layer Attention vs. Weighted Sum vs. Single")

        plt.tight_layout()
        plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Fig: Weight heatmap for 'all' config (E3.2)
        if "all" in all_weight_analysis:
            wa = all_weight_analysis["all"]
            fig, ax = plt.subplots(figsize=(10, 4))
            layers_list = list(wa["val"]["mean_weights"].keys())
            means = [wa["val"]["mean_weights"][l] for l in layers_list]
            stds = [wa["val"]["std_weights"][l] for l in layers_list]
            ax.bar(layers_list, means, yerr=stds, capsize=4,
                   color="darkorange", alpha=0.8)
            ax.set_ylabel("Mean Attention Weight ± std")
            ax.set_xlabel("Encoder Layer")
            ax.set_title("E3.2: Per-Sample Layer Attention Weights (val set)")
            uniform = 1.0 / len(layers_list)
            ax.axhline(y=uniform, color="gray", linestyle="--",
                       alpha=0.5, label="Uniform")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "attention_weights.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved attention weight plot")

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
        "gate3": gate3_summary,
    }
    with open(output_dir / "stage3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Layer Attention Fusion")
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--stage1_results", type=str,
                        default="results/stage2/layer_probing/layer_probing.csv")
    parser.add_argument("--stage2_results", type=str,
                        default="results/stage2/weighted_sum/gate2_decision.json")
    parser.add_argument("--stage2_csv", type=str,
                        default="results/stage2/weighted_sum/weighted_sum_results.csv")
    parser.add_argument("--output", type=str,
                        default="results/stage2/layer_attention")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fusion_lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_layer_attention(args)


if __name__ == "__main__":
    main()
