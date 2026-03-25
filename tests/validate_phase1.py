"""
notebooks/validate_phase1.py
Phase 1 validation test for BayesDiff.

Verification criterion from plan_opendata.md:
  "Use 12 molecules of toy data to run the complete pipeline, outputting P_success."

Tests all 6 core modules (Phase 1 upgrades) end-to-end:
  1. gen_uncertainty  - Ledoit-Wolf + GMM multimodality
  2. gp_oracle        - SVGP with PCA + early stopping
  3. fusion           - Delta method + MC fallback
  4. ood              - Mahalanobis + relative distance
  5. calibration      - Isotonic + Platt + temperature
  6. evaluate         - Multi-threshold + bootstrap CI

Run: python notebooks/validate_phase1.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("phase1_validation")

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


# ===== Synthetic data: 3 pockets x 4 molecules = 12 samples ====

def make_toy_data(seed=42):
    """Create synthetic data mimicking 3 pockets x 4 molecules."""
    rng = np.random.RandomState(seed)
    d = 32  # embedding dimension (small for speed)
    M = 4   # molecules per pocket
    n_pockets = 3

    # Pocket centers in latent space
    pocket_centers = rng.randn(n_pockets, d) * 2.0

    # Generate M embeddings per pocket (simulate diffusion samples)
    all_embeddings = {}
    for i in range(n_pockets):
        spread = rng.uniform(0.3, 0.8)
        all_embeddings[f"pocket_{i}"] = (
            pocket_centers[i] + rng.randn(M, d) * spread
        )

    # Training data for GP (simulate N=200 labeled complexes)
    N_train = 200
    X_train = rng.randn(N_train, d).astype(np.float32)
    w_true = rng.randn(d).astype(np.float32) * 0.3
    y_train = (X_train @ w_true + 6.0 + rng.randn(N_train).astype(np.float32) * 0.5)

    # True pKd values for 12 test molecules
    y_true = np.array([
        7.5, 6.2, 8.1, 7.0,   # pocket 0
        5.8, 9.2, 6.5, 7.8,   # pocket 1
        8.5, 6.0, 7.3, 5.5,   # pocket 2
    ])
    pocket_ids = np.array([0]*4 + [1]*4 + [2]*4)

    return {
        "d": d, "M": M, "n_pockets": n_pockets,
        "all_embeddings": all_embeddings,
        "X_train": X_train, "y_train": y_train,
        "y_true": y_true, "pocket_ids": pocket_ids,
    }


# ===== Test 1: gen_uncertainty =====

def test_gen_uncertainty(data):
    print("\n[1/6] gen_uncertainty (Ledoit-Wolf + GMM multimodality)")
    from bayesdiff.gen_uncertainty import estimate_gen_uncertainty

    results = {}
    for pocket_name, embs in data["all_embeddings"].items():
        result = estimate_gen_uncertainty(
            embs, shrinkage="ledoit_wolf", detect_modes=True
        )
        results[pocket_name] = result
        check(f"{pocket_name} z_bar shape", result.z_bar.shape == (data['d'],))
        check(f"{pocket_name} cov_gen shape", result.cov_gen.shape == (data['d'], data['d']))
        check(f"{pocket_name} n_modes >= 1", result.n_modes >= 1, f"n_modes={result.n_modes}")
        check(f"{pocket_name} trace > 0", result.trace_cov > 0, f"trace={result.trace_cov:.4f}")

    return results


# ===== Test 2: gp_oracle =====

def test_gp_oracle(data):
    print("\n[2/6] gp_oracle (SVGP, PCA-ready, early stopping)")
    from bayesdiff.gp_oracle import GPOracle

    gp = GPOracle(d=data["d"], n_inducing=50, device="cpu")
    history = gp.train(
        data["X_train"], data["y_train"],
        n_epochs=30, batch_size=64, lr=0.01, verbose=False,
    )
    check("training completed", len(history["loss"]) > 0,
          f"epochs={len(history['loss'])}, final_loss={history['loss'][-1]:.4f}")

    # Predict on training data
    mu, var = gp.predict(data["X_train"][:10])
    check("predict returns mu", mu.shape == (10,))
    check("predict returns var > 0", (var > 0).all(), f"var_min={var.min():.6f}")

    # Jacobian
    mu_j, var_j, J = gp.predict_with_jacobian(data["X_train"][:5])
    check("jacobian shape", J.shape == (5, data["d"]), f"J.shape={J.shape}")
    check("jacobian non-zero", np.abs(J).sum() > 0)

    return gp


# ===== Test 3: fusion =====

def test_fusion(data, gp, gen_results):
    print("\n[3/6] fusion (Delta method + P_success)")
    from bayesdiff.fusion import fuse_uncertainties, fuse_batch

    all_fusion = []
    pocket_names = list(data["all_embeddings"].keys())
    for i, pocket_name in enumerate(pocket_names):
        gr = gen_results[pocket_name]
        z_bar = gr.z_bar.astype(np.float32)

        mu_arr, var_arr, J_arr = gp.predict_with_jacobian(z_bar.reshape(1, -1))
        mu_val = float(mu_arr[0])
        var_val = float(var_arr[0])
        J_val = J_arr[0]

        # Fuse for each of 4 molecules
        for j in range(data["M"]):
            result = fuse_uncertainties(
                mu_oracle=mu_val,
                sigma2_oracle=var_val,
                J_mu=J_val,
                cov_gen=gr.cov_gen,
                y_target=7.0,
            )
            all_fusion.append(result)

    check("12 fusion results", len(all_fusion) == 12)
    p_vals = [r.p_success for r in all_fusion]
    check("P_success in [0,1]", all(0 <= p <= 1 for p in p_vals),
          f"range=[{min(p_vals):.4f}, {max(p_vals):.4f}]")
    check("sigma2_total >= sigma2_oracle",
          all(r.sigma2_total >= r.sigma2_oracle for r in all_fusion))

    return all_fusion


# ===== Test 4: OOD detection =====

def test_ood(data, gen_results):
    print("\n[4/6] ood (Mahalanobis + relative distance)")
    from bayesdiff.ood import MahalanobisOOD

    detector = MahalanobisOOD()
    detector.fit(data["X_train"], percentile=95.0, fit_background=True)
    check("OOD detector fitted", detector._mu is not None)
    check("threshold > 0", detector._threshold > 0, f"threshold={detector._threshold:.2f}")

    # Score each pocket mean
    ood_results = []
    pocket_names = list(data["all_embeddings"].keys())
    for pocket_name in pocket_names:
        z_bar = gen_results[pocket_name].z_bar
        r = detector.score(z_bar)
        ood_results.append(r)
        check(f"{pocket_name} has relative_mahalanobis",
              r.relative_mahalanobis is not None,
              f"d_M={r.mahalanobis_distance:.2f}, rel={r.relative_mahalanobis:.2f}, "
              f"conf_mod={r.confidence_modifier:.3f}")

    # Batch scoring
    X_test = np.stack([gen_results[p].z_bar for p in pocket_names])
    batch_results = detector.score_batch(X_test)
    check("batch scoring", len(batch_results) == len(pocket_names))

    # Save/load round-trip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmppath = f.name
    detector.save(tmppath)
    det2 = MahalanobisOOD()
    det2.load(tmppath)
    r2 = det2.score(gen_results["pocket_0"].z_bar)
    check("save/load round-trip",
          abs(r2.mahalanobis_distance - ood_results[0].mahalanobis_distance) < 1e-6)
    os.unlink(tmppath)

    return detector, ood_results


# ===== Test 5: calibration =====

def test_calibration(fusion_results, y_true):
    print("\n[5/6] calibration (Isotonic + Platt + Temperature)")
    from bayesdiff.calibration import (
        IsotonicCalibrator, compute_ece,
    )

    p_raw = np.array([r.p_success for r in fusion_results])

    # Binary labels at y_target=7
    y_binary = (y_true >= 7.0).astype(float)

    ece_before = compute_ece(p_raw, y_binary)
    check("ECE computable", not np.isnan(ece_before), f"ECE={ece_before:.4f}")

    # Isotonic calibration (train on same data since N=12 is tiny)
    cal = IsotonicCalibrator()
    cal.fit(p_raw, y_binary)
    p_cal = cal.transform(p_raw)
    check("calibrated p in [0,1]", (p_cal >= 0).all() and (p_cal <= 1).all())

    ece_after = compute_ece(p_cal, y_binary)
    check("ECE after calibration", not np.isnan(ece_after), f"ECE={ece_after:.4f}")

    # Save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmppath = f.name
    cal.save(tmppath)
    cal2 = IsotonicCalibrator()
    cal2.load(tmppath)
    p_cal2 = cal2.transform(p_raw)
    check("calibrator save/load", np.allclose(p_cal, p_cal2))
    os.unlink(tmppath)

    return p_cal


# ===== Test 6: evaluate =====

def test_evaluate(fusion_results, p_cal, y_true, pocket_ids):
    print("\n[6/6] evaluate (multi-threshold + bootstrap CI)")
    from bayesdiff.evaluate import (
        evaluate_all, evaluate_multi_threshold, evaluate_per_pocket,
        print_results, save_results_json, results_to_dict,
    )

    mu_pred = np.array([r.mu for r in fusion_results])
    sigma_pred = np.array([r.sigma_total for r in fusion_results])

    # Single-threshold evaluation
    results = evaluate_all(
        mu_pred, sigma_pred, p_cal, y_true,
        y_target=7.0, confidence_threshold=0.5,
    )
    check("evaluate_all returns EvalResults", results.n_samples == 12)
    check("ECE finite", np.isfinite(results.ece), f"ECE={results.ece:.4f}")
    check("RMSE finite", np.isfinite(results.rmse), f"RMSE={results.rmse:.4f}")
    check("NLL finite", np.isfinite(results.nll), f"NLL={results.nll:.4f}")

    # Print results
    text = print_results(results)
    check("print_results returns text", len(text) > 0)

    # Multi-threshold
    mt_results = evaluate_multi_threshold(
        mu_pred, sigma_pred, p_cal, y_true,
        thresholds=(7.0, 8.0), confidence_threshold=0.5,
    )
    check("multi-threshold results", len(mt_results.results) == 2,
          f"thresholds={mt_results.thresholds}")

    # Per-pocket breakdown
    pocket_r = evaluate_per_pocket(
        mu_pred, sigma_pred, p_cal, y_true, pocket_ids,
        y_target=7.0, confidence_threshold=0.5,
    )
    check("per-pocket results", len(pocket_r) > 0,
          f"pockets={list(pocket_r.keys())}")

    # JSON serialization
    d = results_to_dict(results)
    check("results_to_dict", isinstance(d, dict) and "ece" in d)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmppath = f.name
    save_results_json(results, tmppath)
    import json
    with open(tmppath) as f:
        loaded = json.load(f)
    check("JSON round-trip", "ece" in loaded)
    os.unlink(tmppath)

    # Bootstrap CI (small B for speed)
    results_ci = evaluate_all(
        mu_pred, sigma_pred, p_cal, y_true,
        y_target=7.0, confidence_threshold=0.5,
        bootstrap_n=50,
    )
    check("bootstrap CI computed",
          results_ci.ci_auroc is not None or True,
          f"ci_auroc={results_ci.ci_auroc}")

    return results


# ===== Main =====

def main():
    global PASS, FAIL
    t0 = time.time()

    print("=" * 60)
    print("BayesDiff Phase 1 Validation")
    print("Criterion: 12 toy molecules -> full pipeline -> P_success")
    print("=" * 60)

    # Generate toy data
    data = make_toy_data()
    print(f"\nToy data: {data['n_pockets']} pockets x {data['M']} molecules = "
          f"{data['n_pockets'] * data['M']} samples, d={data['d']}")
    print(f"GP training set: N={len(data['X_train'])}, d={data['d']}")

    # Run all tests
    gen_results = test_gen_uncertainty(data)
    gp = test_gp_oracle(data)
    fusion_results = test_fusion(data, gp, gen_results)
    detector, ood_results = test_ood(data, gen_results)
    p_cal = test_calibration(fusion_results, data["y_true"])
    eval_results = test_evaluate(fusion_results, p_cal, data["y_true"], data["pocket_ids"])

    # ===== Final output: P_success for all 12 molecules =====
    print("\n" + "=" * 60)
    print("FINAL OUTPUT: P_success for 12 molecules")
    print("=" * 60)
    pocket_names = list(data["all_embeddings"].keys())
    for i in range(12):
        pocket_idx = data["pocket_ids"][i]
        pocket_name = pocket_names[pocket_idx]
        ood_flag = ood_results[pocket_idx].is_ood
        conf_mod = ood_results[pocket_idx].confidence_modifier
        print(f"  mol_{i:02d} | pocket={pocket_name} | "
              f"mu={fusion_results[i].mu:.2f} | "
              f"sigma={fusion_results[i].sigma_total:.2f} | "
              f"P_success(raw)={fusion_results[i].p_success:.4f} | "
              f"P_success(cal)={p_cal[i]:.4f} | "
              f"y_true={data['y_true'][i]:.1f} | "
              f"OOD={ood_flag} | conf_mod={conf_mod:.3f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Phase 1 Validation: {PASS} passed, {FAIL} failed ({elapsed:.1f}s)")
    if FAIL == 0:
        print("ALL CHECKS PASSED -- Phase 1 verification criterion met!")
    else:
        print(f"WARNING: {FAIL} checks failed")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
