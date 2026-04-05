"""
scripts/pipeline/s05_evaluate.py
──────────────────────
Evaluate the full BayesDiff pipeline: GP prediction → fusion → calibration
→ OOD detection → compute all metrics.

Usage (Mac debug):
    python scripts/pipeline/s05_evaluate.py \
        --embeddings results/generated_molecules/all_embeddings.npz \
        --gp_model results/gp_model/gp_model.pt \
        --labels data/splits/labels.csv \
        --output results/evaluation

Usage (full with training data for OOD):
    python scripts/pipeline/s05_evaluate.py \
        --embeddings data/embeddings/casf_test.npz \
        --gp_model results/gp_model/gp_model.pt \
        --gp_train_data results/gp_model/train_data.npz \
        --labels data/splits/labels.csv \
        --output results/evaluation
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


def load_labels(labels_path: str | None, affinity_pkl: str | None) -> dict[str, float]:
    """Load pdb_code -> pKd mapping."""
    if labels_path and Path(labels_path).exists():
        import pandas as pd
        df = pd.read_csv(labels_path)
        return dict(zip(df["pdb_code"], df["pkd"]))
    if affinity_pkl and Path(affinity_pkl).exists():
        import numpy as _np
        with open(affinity_pkl, "rb") as f:
            a = pickle.load(f)
        pocket_pks: dict[str, list[float]] = {}
        for k, v in a.items():
            pk = v.get("pk")
            if pk is None or float(pk) == 0.0:
                continue
            pocket_fam = str(k).split("/")[0]
            pocket_pks.setdefault(pocket_fam, []).append(float(pk))
        return {fam: float(_np.mean(vals)) for fam, vals in pocket_pks.items()}
    # Auto-detect
    default = PROJECT_ROOT / "external" / "targetdiff" / "data" / "affinity_info.pkl"
    if default.exists():
        return load_labels(None, str(default))
    return {}


def main():
    parser = argparse.ArgumentParser(description="BayesDiff Evaluation Pipeline")
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Embeddings .npz (per-pocket, each key has shape (M, d))",
    )
    parser.add_argument(
        "--gp_model", type=str, required=True,
        help="Path to trained GP model (.pt)",
    )
    parser.add_argument(
        "--gp_train_data", type=str, default=None,
        help="Path to GP training data .npz (for OOD detector)",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Labels CSV (pdb_code, pkd)",
    )
    parser.add_argument(
        "--affinity_pkl", type=str, default=None,
        help="affinity_info.pkl fallback",
    )
    parser.add_argument(
        "--output", type=str, default="results/evaluation",
        help="Output directory",
    )
    parser.add_argument("--y_target", type=float, default=7.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap_n", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ──────────────────────────────────────────
    logger.info(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)
    embeddings_dict = {k: data[k] for k in data.files}
    logger.info(f"  {len(embeddings_dict)} pockets loaded")

    # ── Load labels ──────────────────────────────────────────────
    label_map = load_labels(args.labels, args.affinity_pkl)
    logger.info(f"  {len(label_map)} labels available")

    # ── Load GP model ────────────────────────────────────────────
    from bayesdiff.gp_oracle import GPOracle

    # Determine d from first embedding
    first_key = list(embeddings_dict.keys())[0]
    first_emb = embeddings_dict[first_key]
    d = first_emb.shape[-1]

    gp = GPOracle(d=d, n_inducing=10, device="cpu")  # n_inducing overridden by load
    gp.load(args.gp_model)
    logger.info(f"  GP model loaded from {args.gp_model}")

    # ── Load training data for OOD ───────────────────────────────
    from bayesdiff.ood import MahalanobisOOD

    ood_detector = None
    if args.gp_train_data and Path(args.gp_train_data).exists():
        train_data = np.load(args.gp_train_data)
        X_train = train_data["X"]
        ood_detector = MahalanobisOOD()
        ood_detector.fit(X_train, percentile=95.0, fit_background=True)
        logger.info(f"  OOD detector fitted on {X_train.shape[0]} training points")

    # ── Compute gen uncertainty + fusion for each pocket ─────────
    from bayesdiff.gen_uncertainty import estimate_gen_uncertainty
    from bayesdiff.fusion import fuse_uncertainties
    from bayesdiff.evaluate import (
        evaluate_all, evaluate_multi_threshold, save_results_json,
        print_results, results_to_dict,
    )

    # ── Pre-flight diagnostics ──────────────────────────────────
    sample_counts = {k: (v.shape if v.ndim > 1 else (1, v.shape[0]))
                     for k, v in embeddings_dict.items()}
    n_single = sum(1 for s in sample_counts.values() if s[0] <= 1)
    logger.info(f"  Embedding shapes: {len(sample_counts)} pockets, "
                f"{n_single} with M<=1 (will have zero gen uncertainty)")
    if n_single > 0:
        logger.warning(f"  ⚠ {n_single} pockets have <=1 sample — "
                       "sigma2_gen will be 0 for these")

    results_list = []
    targets_evaluated = []

    for pdb_code, emb in embeddings_dict.items():
        pdb_base = pdb_code.split("_")[0] if "_" in pdb_code else pdb_code
        pk = label_map.get(pdb_base) or label_map.get(pdb_code)
        if pk is None:
            logger.debug(f"  Skipping {pdb_code}: no label")
            continue

        # Guarantee 2D
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Gen uncertainty
        gen_r = estimate_gen_uncertainty(emb)

        # GP prediction + Jacobian
        z_bar = gen_r.z_bar.reshape(1, -1)
        mu_oracle, var_oracle, J_mu = gp.predict_with_jacobian(z_bar)

        # OOD detection (must come before fusion for P_final)
        ood_flag, ood_conf, ood_dist = False, 1.0, 0.0
        if ood_detector is not None:
            ood_r = ood_detector.score(gen_r.z_bar)
            ood_flag = ood_r.is_ood
            ood_conf = ood_r.confidence_modifier
            ood_dist = ood_r.mahalanobis_distance

        # Fusion (with OOD confidence for P_final = w(z) · P_success)
        fusion_r = fuse_uncertainties(
            mu_oracle=mu_oracle[0],
            sigma2_oracle=var_oracle[0],
            J_mu=J_mu[0],
            cov_gen=gen_r.cov_gen,
            y_target=args.y_target,
            ood_confidence=ood_conf,
        )

        result = {
            "target": pdb_code,
            "pkd_true": float(pk),
            "mu_pred": float(fusion_r.mu),
            "sigma2_oracle": float(fusion_r.sigma2_oracle),
            "sigma2_gen": float(fusion_r.sigma2_gen),
            "sigma2_total": float(fusion_r.sigma2_total),
            "sigma_total": float(fusion_r.sigma_total),
            "p_success": float(fusion_r.p_success),
            "p_final": float(fusion_r.p_final),
            "trace_cov_gen": float(gen_r.trace_cov),
            "n_modes": int(gen_r.n_modes),
            "n_samples": int(emb.shape[0]),
            "ood_flag": bool(ood_flag),
            "ood_confidence_modifier": float(ood_conf),
            "ood_distance": float(ood_dist),
        }
        results_list.append(result)
        targets_evaluated.append(pdb_code)

    logger.info(f"\n  Evaluated {len(results_list)} pockets")

    # ── Post-flight diagnostics (check_01 §9) ────────────────────
    if results_list:
        mu_arr = np.array([r["mu_pred"] for r in results_list])
        s2g_arr = np.array([r["sigma2_gen"] for r in results_list])
        tr_arr = np.array([r["trace_cov_gen"] for r in results_list])
        n_arr = np.array([r["n_samples"] for r in results_list])
        logger.info("  ── Diagnostics ──")
        logger.info(f"  mu_pred:  unique={len(np.unique(np.round(mu_arr,4)))}, "
                     f"std={mu_arr.std():.6f}, range=[{mu_arr.min():.4f}, {mu_arr.max():.4f}]")
        logger.info(f"  sigma2_gen: nonzero={np.count_nonzero(s2g_arr)}/{len(s2g_arr)}, "
                     f"mean={s2g_arr.mean():.6f}")
        logger.info(f"  trace_cov: nonzero={np.count_nonzero(tr_arr)}/{len(tr_arr)}, "
                     f"mean={tr_arr.mean():.4f}")
        logger.info(f"  n_samples: min={n_arr.min()}, max={n_arr.max()}, "
                     f"median={np.median(n_arr):.0f}")
        if mu_arr.std() < 1e-6:
            logger.warning("  ⚠ ALL mu_pred identical — check pipeline upstream!")
        if s2g_arr.max() == 0:
            logger.warning("  ⚠ ALL sigma2_gen=0 — check embedding M per pocket!")

    # ── Aggregate evaluation ─────────────────────────────────────
    if len(results_list) >= 3:
        mu_pred = np.array([r["mu_pred"] for r in results_list])
        sigma_pred = np.array([r["sigma_total"] for r in results_list])
        p_success = np.array([r["p_success"] for r in results_list])
        y_true = np.array([r["pkd_true"] for r in results_list])

        # Main evaluation
        eval_res = evaluate_all(
            mu_pred, sigma_pred, p_success, y_true,
            y_target=args.y_target,
            confidence_threshold=args.confidence_threshold,
            bootstrap_n=args.bootstrap_n,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Metrics (y≥{args.y_target}, N={len(results_list)})")
        logger.info(f"{'='*60}")
        print_results(eval_res)

        # Multi-threshold
        mt = evaluate_multi_threshold(
            mu_pred, sigma_pred, p_success, y_true,
            thresholds=(7.0, 8.0),
            confidence_threshold=args.confidence_threshold,
        )
        logger.info(f"\nMulti-threshold results:")
        for r in mt.results:
            logger.info(f"  y≥{r.y_target}: ECE={r.ece:.4f}, AUROC={r.auroc:.4f}, "
                        f"Hit={r.hit_rate:.4f}, Spearman={r.spearman_rho:.4f}")

        # Save metrics
        save_results_json(eval_res, output_dir / "eval_metrics.json")
        logger.info(f"\n  Metrics saved to {output_dir / 'eval_metrics.json'}")

        # Save multi-threshold
        mt_dict = {}
        for r in mt.results:
            mt_dict[f"y>={r.y_target}"] = results_to_dict(r)
        with open(output_dir / "eval_multi_threshold.json", "w") as f:
            json.dump(mt_dict, f, indent=2)

    # ── Save per-pocket results ──────────────────────────────────
    with open(output_dir / "per_pocket_results.json", "w") as f:
        json.dump(results_list, f, indent=2)
    logger.info(f"  Per-pocket results saved to {output_dir / 'per_pocket_results.json'}")

    # ── Summary ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation complete!")
    logger.info(f"  Pockets evaluated: {len(results_list)}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
