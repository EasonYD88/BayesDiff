"""Evaluate multi-task trunk + oracle head against SP4 baseline.

Generates:
  - Comparison table (JSON + stdout)
  - Per-task loss curves (PDF)
  - t-SNE: unshaped vs. shaped trunk (PDF)
  - Within-group NDCG curves (PDF, if v2)

Usage:
    python scripts/pipeline/s21_evaluate_multitask.py \
        --results_dir results/stage2/multitask_trunk \
        --embeddings results/stage2/oracle_heads/frozen_embeddings_augmented.npz \
        --output results/stage2/multitask_trunk/figures
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

log = logging.getLogger(__name__)


def load_all_results(results_dir: Path) -> dict:
    """Walk results_dir and load eval_results.json for each experiment/seed."""
    all_results = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            eval_path = seed_dir / "eval_results.json"
            if eval_path.exists():
                with open(eval_path) as f:
                    metrics = json.load(f)
                key = f"{exp_dir.name}/{seed_dir.name}"
                all_results[key] = metrics
    return all_results


def print_comparison_table(all_results: dict):
    """Print a comparison table to stdout."""
    # Group by experiment (aggregate seeds)
    from collections import defaultdict

    experiments = defaultdict(list)
    for key, metrics in all_results.items():
        exp_name = key.split("/")[0]
        experiments[exp_name].append(metrics)

    header = f"{'Experiment':<30} {'rho':>8} {'R2':>8} {'RMSE':>8} {'NLL':>8} {'err_sig_rho':>12}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    summary = {}
    for exp_name, seed_metrics in sorted(experiments.items()):
        n_seeds = len(seed_metrics)
        row = {}
        for metric_key in ["spearman_rho", "R2", "rmse", "nll", "err_sigma_rho"]:
            values = [m[metric_key] for m in seed_metrics if metric_key in m]
            if values:
                mean = np.mean(values)
                std = np.std(values) if len(values) > 1 else 0.0
                row[metric_key] = (mean, std)

        rho_str = f"{row.get('spearman_rho', (0,0))[0]:.3f}±{row.get('spearman_rho', (0,0))[1]:.3f}"
        r2_str = f"{row.get('R2', (0,0))[0]:.3f}±{row.get('R2', (0,0))[1]:.3f}"
        rmse_str = f"{row.get('rmse', (0,0))[0]:.3f}±{row.get('rmse', (0,0))[1]:.3f}"
        nll_str = f"{row.get('nll', (0,0))[0]:.3f}±{row.get('nll', (0,0))[1]:.3f}"
        esr_str = f"{row.get('err_sigma_rho', (0,0))[0]:.3f}±{row.get('err_sigma_rho', (0,0))[1]:.3f}"
        print(f"{exp_name:<30} {rho_str:>8} {r2_str:>8} {rmse_str:>8} {nll_str:>8} {esr_str:>12}")
        summary[exp_name] = row

    print("=" * len(header))
    return summary


def plot_training_curves(results_dir: Path, output_dir: Path):
    """Plot per-task training curves for each experiment."""
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            history_path = seed_dir / "history.json"
            if not history_path.exists():
                continue
            with open(history_path) as f:
                history = json.load(f)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"{exp_dir.name} / {seed_dir.name}", fontsize=12)

            for ax, (key, label) in zip(
                axes,
                [("L_reg", "Regression"), ("L_cls", "Classification"), ("L_rank", "Ranking")],
            ):
                train_key = f"train_{key}"
                val_key = f"val_{key}"
                if train_key in history and history[train_key]:
                    ax.plot(history[train_key], label="train", alpha=0.8)
                if val_key in history and history[val_key]:
                    ax.plot(history[val_key], label="val", alpha=0.8)
                ax.set_title(label)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()

            plt.tight_layout()
            out_path = output_dir / f"training_curves_{exp_dir.name}_{seed_dir.name}.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved {out_path}")


def plot_tsne(results_dir: Path, embeddings_path: Path, output_dir: Path):
    """Plot t-SNE of unshaped vs. shaped trunk features for the first seed."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        log.warning("sklearn not available, skipping t-SNE")
        return

    data = np.load(embeddings_path, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    # Unshaped embedding t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_test) - 1))
    Z_unshaped = tsne.fit_transform(X_test)

    # Look for shaped features from one experiment
    import torch
    from bayesdiff.multi_task import MultiTaskTrunk

    shaped_features = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in ("figures", "A5.0_no_trunk"):
            continue
        seed_dir = exp_dir / "seed42"
        trunk_path = seed_dir / "trunk.pt"
        config_path = seed_dir / "config.json"
        if not trunk_path.exists() or not config_path.exists():
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        trunk = MultiTaskTrunk(
            input_dim=X_test.shape[1],
            hidden_dim=cfg.get("hidden_dim", 256),
            trunk_dim=cfg.get("trunk_dim", 128),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            residual=True,
            activity_threshold=cfg.get("threshold", 7.0),
            enable_ranking=cfg.get("phase", "v1") == "v2",
        )
        trunk.load_state_dict(torch.load(trunk_path, weights_only=False))
        trunk.eval()
        h = trunk.extract_trunk_features(X_test)
        shaped_features[exp_dir.name] = h

    # Plot
    n_plots = 1 + len(shaped_features)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    sc = axes[0].scatter(Z_unshaped[:, 0], Z_unshaped[:, 1], c=y_test, cmap="viridis", s=8, alpha=0.7)
    axes[0].set_title("Unshaped (frozen)")
    plt.colorbar(sc, ax=axes[0], label="pKd")

    for ax, (name, h) in zip(axes[1:], shaped_features.items()):
        Z = tsne.fit_transform(h)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y_test, cmap="viridis", s=8, alpha=0.7)
        ax.set_title(f"Shaped: {name}")
        plt.colorbar(sc, ax=ax, label="pKd")

    fig.suptitle("t-SNE: Frozen vs. Shaped Trunk Features", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / "tsne_trunk.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def tier1_go_nogo(summary: dict) -> bool:
    """Check the 3-condition Tier 1 decision gate.

    Tier 1 passes if ALL of:
        (a) rho(A5.2) >= rho(A5.0) - 0.01   (no regression from SP4 baseline)
        (b) nll(A5.2) <= nll(A5.0) + 0.05    (uncertainty not blown)
        (c) At least one of:
            - rho(A5.2) >= rho(A5.0) + 0.005
            - err_sigma_rho(A5.2) >= err_sigma_rho(A5.0) + 0.01
    """
    baseline_name = None
    primary_name = None
    for name in summary:
        if "A5.0" in name or "no_trunk" in name:
            baseline_name = name
        if "A5.2" in name or "reg_cls" in name:
            primary_name = name

    if not baseline_name or not primary_name:
        log.warning("Cannot find A5.0 and/or A5.2 — skipping go/no-go gate")
        return False

    b = summary[baseline_name]
    p = summary[primary_name]

    rho_b = b["spearman_rho"][0]
    rho_p = p["spearman_rho"][0]
    nll_b = b["nll"][0]
    nll_p = p["nll"][0]
    esr_b = b.get("err_sigma_rho", (0, 0))[0]
    esr_p = p.get("err_sigma_rho", (0, 0))[0]

    cond_a = rho_p >= rho_b - 0.01
    cond_b = nll_p <= nll_b + 0.05
    cond_c = (rho_p >= rho_b + 0.005) or (esr_p >= esr_b + 0.01)

    print("\n=== Tier 1 Go/No-Go Gate ===")
    print(f"  (a) rho(A5.2)={rho_p:.4f} >= rho(A5.0)-0.01={rho_b - 0.01:.4f}  → {'PASS' if cond_a else 'FAIL'}")
    print(f"  (b) nll(A5.2)={nll_p:.4f} <= nll(A5.0)+0.05={nll_b + 0.05:.4f}  → {'PASS' if cond_b else 'FAIL'}")
    print(f"  (c) rho gain={rho_p - rho_b:.4f} OR esr gain={esr_p - esr_b:.4f}  → {'PASS' if cond_c else 'FAIL'}")

    passed = cond_a and cond_b and cond_c
    print(f"\n  Tier 1 DECISION: {'GO — proceed to Tier 2' if passed else 'NO-GO — stop or revise'}")
    return passed


def main(args):
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all results
    all_results = load_all_results(results_dir)
    if not all_results:
        log.error(f"No eval_results.json found under {results_dir}")
        return

    # 2. Print comparison table
    summary = print_comparison_table(all_results)

    # 3. Save summary JSON
    with open(output_dir / "comparison_table.json", "w") as f:
        json.dump(
            {k: {mk: {"mean": v[0], "std": v[1]} for mk, v in mv.items()} for k, mv in summary.items()},
            f,
            indent=2,
        )

    # 4. Tier 1 go/no-go gate
    tier1_go_nogo(summary)

    # 5. Training curves
    plot_training_curves(results_dir, output_dir)

    # 6. t-SNE
    if args.embeddings:
        plot_tsne(results_dir, Path(args.embeddings), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SP05: Evaluate multi-task experiments")
    parser.add_argument(
        "--results_dir",
        default="results/stage2/multitask_trunk",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--embeddings",
        default="results/stage2/oracle_heads/frozen_embeddings_augmented.npz",
        help="Path to augmented NPZ for t-SNE",
    )
    parser.add_argument(
        "--output",
        default="results/stage2/multitask_trunk/figures",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main(args)
