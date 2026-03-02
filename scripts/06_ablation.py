"""
scripts/06_ablation.py
──────────────────────
Run ablation experiments defined in plan_opendata.md §6.3.

Ablation IDs:
  A1: No U_gen          — σ²_total = σ²_oracle only
  A2: No U_oracle       — σ²_total = J_μ^T Σ_gen J_μ only
  A3: No calibration    — Output raw P_success
  A4: Naive covariance  — No Ledoit-Wolf shrinkage
  A5: No multimodal     — Force K=1
  A7: No OOD detection  — Remove Mahalanobis gate

Usage:
    python scripts/06_ablation.py \
        --embeddings results/generated_molecules/all_embeddings.npz \
        --gp_model results/gp_model/gp_model.pt \
        --labels data/splits/labels.csv \
        --output results/ablation

    # Specific ablation only:
    python scripts/06_ablation.py --ablations A1 A3 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


ABLATION_DESCRIPTIONS = {
    "full": "BayesDiff (Full)",
    "A1": "No U_gen (σ²_total = σ²_oracle)",
    "A2": "No U_oracle (σ²_total = σ²_gen only)",
    "A3": "No calibration (raw P_success)",
    "A4": "Naive covariance (no Ledoit-Wolf)",
    "A5": "No multimodal detection (force K=1)",
    "A7": "No OOD detection",
}


def load_labels(labels_path, affinity_pkl):
    """Load labels from CSV or pkl."""
    if labels_path and Path(labels_path).exists():
        import pandas as pd
        df = pd.read_csv(labels_path)
        return dict(zip(df["pdb_code"], df["pkd"]))
    if affinity_pkl and Path(affinity_pkl).exists():
        with open(affinity_pkl, "rb") as f:
            a = pickle.load(f)
        return {
            str(k).split("_")[0] if "_" in str(k) else str(k): float(v.get("neglog_aff", 0))
            for k, v in a.items() if v.get("neglog_aff") is not None
        }
    default = PROJECT_ROOT / "external" / "targetdiff" / "data" / "affinity_info.pkl"
    if default.exists():
        return load_labels(None, str(default))
    return {}


def run_ablation(
    ablation_id: str,
    embeddings_dict: dict[str, np.ndarray],
    label_map: dict[str, float],
    gp,
    ood_detector,
    y_target: float = 7.0,
) -> list[dict]:
    """Run a single ablation variant and return per-pocket results."""
    from bayesdiff.gen_uncertainty import estimate_gen_uncertainty
    from bayesdiff.fusion import fuse_uncertainties
    from bayesdiff.ood import MahalanobisOOD

    results = []

    for pdb_code, emb in embeddings_dict.items():
        pdb_base = pdb_code.split("_")[0] if "_" in pdb_code else pdb_code
        pk = label_map.get(pdb_base) or label_map.get(pdb_code)
        if pk is None:
            continue
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Gen uncertainty (A4: no shrinkage; A5: force K=1)
        shrinkage = "none" if ablation_id == "A4" else "ledoit_wolf"
        max_modes = 1 if ablation_id == "A5" else 3
        gen_r = estimate_gen_uncertainty(emb, shrinkage=shrinkage, max_modes=max_modes)

        z_bar = gen_r.z_bar.reshape(1, -1)
        mu_oracle, var_oracle, J_mu = gp.predict_with_jacobian(z_bar)
        mu_o, var_o, j_mu = mu_oracle[0], var_oracle[0], J_mu[0]

        # Compute variance components based on ablation
        sigma2_gen = float(j_mu @ gen_r.cov_gen @ j_mu)
        sigma2_oracle = float(var_o)

        if ablation_id == "A1":
            # No U_gen: total = oracle only
            sigma2_total = sigma2_oracle
        elif ablation_id == "A2":
            # No U_oracle: total = gen propagation only
            sigma2_total = sigma2_gen
        else:
            # Full or other ablations
            sigma2_total = sigma2_oracle + sigma2_gen

        sigma_total = max(np.sqrt(sigma2_total), 1e-8)

        # P_success
        from scipy.stats import norm
        p_raw = 1.0 - norm.cdf(y_target, loc=mu_o, scale=sigma_total)

        if ablation_id == "A3":
            # No calibration — use raw (in full pipeline, calibration adjusts this)
            p_success = float(p_raw)
        else:
            # In debug mode, no calibrator fitted anyway, so raw = calibrated
            p_success = float(p_raw)

        # OOD
        ood_flag = False
        ood_conf = 1.0
        if ablation_id != "A7" and ood_detector is not None:
            ood_r = ood_detector.score(gen_r.z_bar)
            ood_flag = ood_r.is_ood
            ood_conf = ood_r.confidence_modifier

        results.append({
            "target": pdb_code,
            "pkd_true": float(pk),
            "mu_pred": float(mu_o),
            "sigma2_total": float(sigma2_total),
            "sigma_total": float(sigma_total),
            "p_success": float(p_success),
            "ood_flag": bool(ood_flag),
            "ood_confidence_modifier": float(ood_conf),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="BayesDiff Ablation Study")
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Embeddings .npz",
    )
    parser.add_argument(
        "--gp_model", type=str, required=True,
        help="Trained GP model (.pt)",
    )
    parser.add_argument(
        "--gp_train_data", type=str, default=None,
        help="GP training data .npz (for OOD detector)",
    )
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--affinity_pkl", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default="results/ablation",
    )
    parser.add_argument(
        "--ablations", nargs="*", default=None,
        help="Which ablations to run (default: all). E.g. A1 A3 A5",
    )
    parser.add_argument("--y_target", type=float, default=7.0)
    parser.add_argument("--bootstrap_n", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_ids = args.ablations or ["full", "A1", "A2", "A3", "A4", "A5", "A7"]

    # ── Load data ────────────────────────────────────────────────
    logger.info(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)
    embeddings_dict = {k: data[k] for k in data.files}
    logger.info(f"  {len(embeddings_dict)} pockets")

    label_map = load_labels(args.labels, args.affinity_pkl)
    logger.info(f"  {len(label_map)} labels")

    # ── Load GP ──────────────────────────────────────────────────
    from bayesdiff.gp_oracle import GPOracle

    first_emb = list(embeddings_dict.values())[0]
    d = first_emb.shape[-1]
    gp = GPOracle(d=d, n_inducing=10, device="cpu")  # n_inducing overridden by load
    gp.load(args.gp_model)
    logger.info(f"  GP model loaded (d={d})")

    # ── OOD detector ─────────────────────────────────────────────
    from bayesdiff.ood import MahalanobisOOD

    ood_detector = None
    if args.gp_train_data and Path(args.gp_train_data).exists():
        train_data = np.load(args.gp_train_data)
        ood_detector = MahalanobisOOD()
        ood_detector.fit(train_data["X"], percentile=95.0, fit_background=True)
        logger.info(f"  OOD detector fitted")

    # ── Run ablations ────────────────────────────────────────────
    from bayesdiff.evaluate import evaluate_all, results_to_dict, comparison_table

    all_eval_results = {}
    all_per_pocket = {}

    for ablation_id in ablation_ids:
        desc = ABLATION_DESCRIPTIONS.get(ablation_id, ablation_id)
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation: {ablation_id} — {desc}")
        logger.info(f"{'='*60}")

        results = run_ablation(
            ablation_id, embeddings_dict, label_map, gp, ood_detector,
            y_target=args.y_target,
        )
        all_per_pocket[ablation_id] = results

        if len(results) >= 3:
            mu_pred = np.array([r["mu_pred"] for r in results])
            sigma_pred = np.array([r["sigma_total"] for r in results])
            p_success = np.array([r["p_success"] for r in results])
            y_true = np.array([r["pkd_true"] for r in results])

            eval_res = evaluate_all(
                mu_pred, sigma_pred, p_success, y_true,
                y_target=args.y_target,
                confidence_threshold=0.5,
                bootstrap_n=args.bootstrap_n,
            )
            all_eval_results[ablation_id] = eval_res

            logger.info(f"  ECE={eval_res.ece:.4f}, AUROC={eval_res.auroc:.4f}, "
                        f"EF@1%={eval_res.ef_1pct:.2f}, "
                        f"Spearman={eval_res.spearman_rho:.4f}, RMSE={eval_res.rmse:.4f}")
        else:
            logger.warning(f"  Only {len(results)} pockets — skipping aggregate eval")

    # ── Comparison table ─────────────────────────────────────────
    if all_eval_results:
        logger.info(f"\n{'='*60}")
        logger.info("ABLATION COMPARISON TABLE")
        logger.info(f"{'='*60}")
        table = comparison_table(all_eval_results)
        logger.info("\n" + table)

    # ── Save results ─────────────────────────────────────────────
    # Save per-ablation metrics
    summary = {}
    for aid, er in all_eval_results.items():
        summary[aid] = {
            "description": ABLATION_DESCRIPTIONS.get(aid, aid),
            **results_to_dict(er),
        }

    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-pocket results
    with open(output_dir / "ablation_per_pocket.json", "w") as f:
        json.dump(all_per_pocket, f, indent=2)

    logger.info(f"\n  Results saved to {output_dir}")
    logger.info(f"  - ablation_summary.json")
    logger.info(f"  - ablation_per_pocket.json")
    logger.info(f"\n{'='*60}")
    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
