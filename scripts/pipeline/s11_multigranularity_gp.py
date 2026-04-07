"""
scripts/pipeline/s11_multigranularity_gp.py
───────────────────────────────────────────
Stage 2 Sub-Plan 1: Multi-Granularity GP Evaluation

Combines z_interaction (from trained InteractionGNN) with z_global
(from encoder layer embeddings) and trains a GP on the concatenated
representation.

Experiments run:
  1. z_global only (baseline — layer 8 or WS-all)
  2. z_interaction only (interaction graph branch alone)
  3. z_global + z_interaction (multi-granularity concat)
  4. z_global + z_interaction_shuffled (ablation A1.10 — shuffled edges)

Gate criteria (from plan §8):
  - ΔR² ≥ +0.03 AND Δρ ≥ +0.04 over best baseline
  - z_interaction must beat shuffled-edge control

Usage:
    python scripts/pipeline/s11_multigranularity_gp.py \\
        --multilayer_emb results/multilayer_embeddings/all_multilayer_embeddings.npz \\
        --z_interaction_dir results/stage2/interaction_gnn \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/multigranularity_gp \\
        --device cuda

    # With shuffled-edge ablation:
    python scripts/pipeline/s11_multigranularity_gp.py \\
        ... \\
        --z_shuffled_dir results/stage2/interaction_gnn_shuffled
"""

from __future__ import annotations

import argparse
import csv
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


def compute_metrics(y_true, mu, var):
    """R², Spearman ρ, Pearson r, RMSE, MAE, NLL."""
    from scipy.stats import spearmanr, pearsonr

    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rho, _ = spearmanr(y_true, mu)
    pr, _ = pearsonr(y_true, mu)
    rmse = np.sqrt(np.mean((y_true - mu) ** 2))
    mae = np.mean(np.abs(y_true - mu))
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var)
    return {
        "R2": float(r2),
        "spearman_rho": float(rho),
        "pearson_r": float(pr),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "NLL": float(nll),
    }


def load_multilayer_embeddings(emb_path, splits_path, labels_path, layer_idx=8):
    """Load z_global from a specific encoder layer, split by train/val/test.

    Returns dict with X_{split} (N, d), y_{split} (N,), codes_{split} (list).
    """
    import pandas as pd

    logger.info(f"Loading multilayer embeddings (layer {layer_idx})...")
    emb = dict(np.load(emb_path, allow_pickle=True))

    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))

    with open(splits_path) as f:
        splits = json.load(f)

    result = {}
    for split_name in ["train", "val", "test"]:
        codes, X, y = [], [], []
        for code in splits[split_name]:
            key = f"{code}_layer_{layer_idx}"
            if key not in emb or code not in label_map:
                continue
            codes.append(code)
            X.append(emb[key])
            y.append(label_map[code])
        result[f"X_{split_name}"] = np.stack(X, axis=0)
        result[f"y_{split_name}"] = np.array(y, dtype=np.float32)
        result[f"codes_{split_name}"] = codes
        logger.info(f"  {split_name}: {len(codes)} samples, dim={result[f'X_{split_name}'].shape[1]}")

    return result


def load_z_interaction(z_dir, splits_codes):
    """Load z_interaction embeddings, aligned to splits_codes ordering.

    Args:
        z_dir: directory containing z_interaction_{split}.npz
        splits_codes: dict of {split_name: [codes]} for alignment

    Returns dict with X_{split} arrays aligned to the same codes as z_global.
    """
    result = {}
    for split_name in ["train", "val", "test"]:
        npz = np.load(Path(z_dir) / f"z_interaction_{split_name}.npz", allow_pickle=True)
        z_emb = npz["embeddings"]  # (N, d)
        z_codes = list(npz["pdb_codes"])
        z_pkd = npz["pkd"]

        # Build code → index mapping
        code_to_idx = {c: i for i, c in enumerate(z_codes)}

        # Align to splits_codes ordering
        target_codes = splits_codes[split_name]
        indices = []
        missing = 0
        for code in target_codes:
            if code in code_to_idx:
                indices.append(code_to_idx[code])
            else:
                missing += 1

        if missing > 0:
            logger.warning(f"  {split_name}: {missing}/{len(target_codes)} codes missing in z_interaction")

        # Only keep codes that exist in both
        aligned_codes = [c for c in target_codes if c in code_to_idx]
        aligned_indices = [code_to_idx[c] for c in aligned_codes]

        result[f"X_{split_name}"] = z_emb[aligned_indices]
        result[f"codes_{split_name}"] = aligned_codes
        logger.info(f"  {split_name}: {len(aligned_codes)} aligned samples, dim={result[f'X_{split_name}'].shape[1]}")

    return result


def align_and_concat(global_data, interaction_data):
    """Align two embedding sources by pdb_code and concatenate.

    Returns dict with X_{split} (N_common, d_g + d_i), y_{split}, codes_{split}.
    """
    result = {}
    for split_name in ["train", "val", "test"]:
        g_codes = global_data[f"codes_{split_name}"]
        i_codes = set(interaction_data[f"codes_{split_name}"])

        # Build index maps
        g_code_to_idx = {c: i for i, c in enumerate(g_codes)}
        i_code_to_idx = {c: i for i, c in enumerate(interaction_data[f"codes_{split_name}"])}

        common_codes = [c for c in g_codes if c in i_codes]

        g_indices = [g_code_to_idx[c] for c in common_codes]
        i_indices = [i_code_to_idx[c] for c in common_codes]

        X_g = global_data[f"X_{split_name}"][g_indices]
        X_i = interaction_data[f"X_{split_name}"][i_indices]
        y = global_data[f"y_{split_name}"][g_indices]

        result[f"X_{split_name}"] = np.concatenate([X_g, X_i], axis=1)
        result[f"y_{split_name}"] = y
        result[f"codes_{split_name}"] = common_codes
        logger.info(f"  {split_name}: {len(common_codes)} common, dim={result[f'X_{split_name}'].shape[1]}")

    return result


def train_and_eval_gp(X_train, y_train, X_val, y_val, X_test, y_test,
                      n_inducing=512, n_epochs=200, batch_size=256, lr=0.01,
                      device="cuda"):
    """Train GP and evaluate on val + test."""
    from bayesdiff.gp_oracle import GPOracle

    d = X_train.shape[1]
    gp = GPOracle(d=d, n_inducing=n_inducing, device=device)

    logger.info(f"  Training GP: d={d}, n_train={X_train.shape[0]}, n_inducing={n_inducing}")
    t0 = time.time()
    history = gp.train(X_train, y_train, n_epochs=n_epochs, batch_size=batch_size, lr=lr)
    train_time = time.time() - t0
    logger.info(f"  GP training took {train_time:.1f}s")

    # Evaluate
    mu_val, var_val = gp.predict(X_val)
    mu_test, var_test = gp.predict(X_test)

    val_metrics = compute_metrics(y_val, mu_val, var_val)
    test_metrics = compute_metrics(y_test, mu_test, var_test)

    return val_metrics, test_metrics, train_time


def main():
    parser = argparse.ArgumentParser(description="Multi-Granularity GP Evaluation")
    parser.add_argument("--multilayer_emb", type=str,
                        default="results/multilayer_embeddings/all_multilayer_embeddings.npz")
    parser.add_argument("--z_interaction_dir", type=str,
                        default="results/stage2/interaction_gnn")
    parser.add_argument("--z_shuffled_dir", type=str, default=None,
                        help="Dir with shuffled-edge z_interaction (for ablation A1.10)")
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--output", type=str, default="results/stage2/multigranularity_gp")
    parser.add_argument("--layer_idx", type=int, default=8,
                        help="Encoder layer index for z_global (default: 8 = best single layer)")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("=== Loading embeddings ===")
    global_data = load_multilayer_embeddings(
        args.multilayer_emb, args.splits, args.labels, layer_idx=args.layer_idx
    )

    logger.info("Loading z_interaction (real topology)...")
    z_real = load_z_interaction(
        args.z_interaction_dir,
        {s: global_data[f"codes_{s}"] for s in ["train", "val", "test"]}
    )

    z_shuffled = None
    if args.z_shuffled_dir and Path(args.z_shuffled_dir).exists():
        logger.info("Loading z_interaction (shuffled edges)...")
        z_shuffled = load_z_interaction(
            args.z_shuffled_dir,
            {s: global_data[f"codes_{s}"] for s in ["train", "val", "test"]}
        )

    # ── Align and prepare configurations ──────────────────────────────
    logger.info("Aligning embeddings...")
    combined_real = align_and_concat(global_data, z_real)

    configs = {
        "z_global_only": {
            "X_train": global_data["X_train"],
            "y_train": global_data["y_train"],
            "X_val": global_data["X_val"],
            "y_val": global_data["y_val"],
            "X_test": global_data["X_test"],
            "y_test": global_data["y_test"],
        },
        "z_interaction_only": {
            "X_train": z_real["X_train"],
            "y_train": global_data["y_train"][:z_real["X_train"].shape[0]],
            "X_val": z_real["X_val"],
            "y_val": global_data["y_val"][:z_real["X_val"].shape[0]],
            "X_test": z_real["X_test"],
            "y_test": global_data["y_test"][:z_real["X_test"].shape[0]],
        },
        "z_global+z_interaction": {
            "X_train": combined_real["X_train"],
            "y_train": combined_real["y_train"],
            "X_val": combined_real["X_val"],
            "y_val": combined_real["y_val"],
            "X_test": combined_real["X_test"],
            "y_test": combined_real["y_test"],
        },
    }

    # z_interaction_only needs proper y alignment
    i_train_codes = z_real["codes_train"]
    i_val_codes = z_real["codes_val"]
    i_test_codes = z_real["codes_test"]

    import pandas as pd
    labels_df = pd.read_csv(args.labels)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))

    configs["z_interaction_only"]["y_train"] = np.array(
        [label_map[c] for c in i_train_codes], dtype=np.float32
    )
    configs["z_interaction_only"]["y_val"] = np.array(
        [label_map[c] for c in i_val_codes], dtype=np.float32
    )
    configs["z_interaction_only"]["y_test"] = np.array(
        [label_map[c] for c in i_test_codes], dtype=np.float32
    )

    if z_shuffled is not None:
        combined_shuffled = align_and_concat(global_data, z_shuffled)
        configs["z_global+z_shuffled"] = {
            "X_train": combined_shuffled["X_train"],
            "y_train": combined_shuffled["y_train"],
            "X_val": combined_shuffled["X_val"],
            "y_val": combined_shuffled["y_val"],
            "X_test": combined_shuffled["X_test"],
            "y_test": combined_shuffled["y_test"],
        }

    # ── Run experiments ────────────────────────────────────────────────
    all_results = {}

    for config_name, data in configs.items():
        logger.info(f"\n=== Config: {config_name} ===")
        logger.info(f"  X_train: {data['X_train'].shape}, X_val: {data['X_val'].shape}, X_test: {data['X_test'].shape}")

        val_metrics, test_metrics, train_time = train_and_eval_gp(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
            n_inducing=args.n_inducing,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        logger.info(f"  Val:  R²={val_metrics['R2']:.4f}  ρ={val_metrics['spearman_rho']:.4f}  RMSE={val_metrics['RMSE']:.3f}  NLL={val_metrics['NLL']:.3f}")
        logger.info(f"  Test: R²={test_metrics['R2']:.4f}  ρ={test_metrics['spearman_rho']:.4f}  RMSE={test_metrics['RMSE']:.3f}  NLL={test_metrics['NLL']:.3f}")

        all_results[config_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "train_time_s": train_time,
            "dim": int(data["X_train"].shape[1]),
            "n_train": int(data["X_train"].shape[0]),
        }

    # ── Gate evaluation ────────────────────────────────────────────────
    logger.info("\n=== Gate Evaluation ===")
    baseline = all_results["z_global_only"]
    combined = all_results["z_global+z_interaction"]

    delta_r2_val = combined["val"]["R2"] - baseline["val"]["R2"]
    delta_rho_val = combined["val"]["spearman_rho"] - baseline["val"]["spearman_rho"]
    delta_r2_test = combined["test"]["R2"] - baseline["test"]["R2"]
    delta_rho_test = combined["test"]["spearman_rho"] - baseline["test"]["spearman_rho"]

    logger.info(f"  Val  ΔR² = {delta_r2_val:+.4f}, Δρ = {delta_rho_val:+.4f}")
    logger.info(f"  Test ΔR² = {delta_r2_test:+.4f}, Δρ = {delta_rho_test:+.4f}")

    gate_r2 = delta_r2_val >= 0.03
    gate_rho = delta_rho_val >= 0.04
    gate_pass = gate_r2 and gate_rho

    # Shuffled-edge check
    shuffled_check = None
    if "z_global+z_shuffled" in all_results:
        shuffled = all_results["z_global+z_shuffled"]
        shuffle_delta_r2 = combined["val"]["R2"] - shuffled["val"]["R2"]
        shuffle_delta_rho = combined["val"]["spearman_rho"] - shuffled["val"]["spearman_rho"]
        shuffled_check = {
            "real_val_R2": combined["val"]["R2"],
            "shuffled_val_R2": shuffled["val"]["R2"],
            "delta_R2": float(shuffle_delta_r2),
            "real_val_rho": combined["val"]["spearman_rho"],
            "shuffled_val_rho": shuffled["val"]["spearman_rho"],
            "delta_rho": float(shuffle_delta_rho),
            "real_beats_shuffled": shuffle_delta_r2 > 0 and shuffle_delta_rho > 0,
        }
        logger.info(f"  Shuffled control: real R²={combined['val']['R2']:.4f} vs shuffled R²={shuffled['val']['R2']:.4f} (Δ={shuffle_delta_r2:+.4f})")
        gate_pass = gate_pass and shuffled_check["real_beats_shuffled"]

    gate_decision = {
        "baseline_config": "z_global_only",
        "baseline_layer": args.layer_idx,
        "combined_config": "z_global+z_interaction",
        "delta_val_R2": float(delta_r2_val),
        "delta_val_rho": float(delta_rho_val),
        "delta_test_R2": float(delta_r2_test),
        "delta_test_rho": float(delta_rho_test),
        "gate_R2_pass": gate_r2,
        "gate_rho_pass": gate_rho,
        "shuffled_check": shuffled_check,
        "gate_pass": gate_pass,
        "decision": (
            "PASS — Multi-granularity representation meets gate criteria"
            if gate_pass
            else "FAIL — Multi-granularity representation does NOT meet gate criteria"
        ),
    }

    status = "PASS ✓" if gate_pass else "FAIL ✗"
    logger.info(f"  Gate decision: {status}")
    logger.info(f"    ΔR² ≥ 0.03? {'YES' if gate_r2 else 'NO'} ({delta_r2_val:+.4f})")
    logger.info(f"    Δρ  ≥ 0.04? {'YES' if gate_rho else 'NO'} ({delta_rho_val:+.4f})")
    if shuffled_check:
        logger.info(f"    Real > Shuffled? {'YES' if shuffled_check['real_beats_shuffled'] else 'NO'}")

    # ── Save results ───────────────────────────────────────────────────
    with open(output_dir / "results.json", "w") as f:
        json.dump({"configs": all_results, "gate": gate_decision, "args": vars(args)}, f, indent=2)

    # Summary CSV
    rows = []
    for config_name, data in all_results.items():
        row = {"config": config_name, "dim": data["dim"], "n_train": data["n_train"]}
        for split in ["val", "test"]:
            for metric, value in data[split].items():
                row[f"{split}_{metric}"] = value
        rows.append(row)

    with open(output_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "gate_decision.json", "w") as f:
        json.dump(gate_decision, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
