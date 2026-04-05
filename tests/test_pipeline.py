"""
tests/test_pipeline.py
───────────────────────────
Quick sanity check for Phase 0 — verifies that all modules import correctly
and the data pipeline works end-to-end with synthetic data.

Run: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all bayesdiff modules import."""
    print("Testing imports...")
    from bayesdiff import __version__

    print(f"  bayesdiff v{__version__}")

    from bayesdiff import data

    print(f"  data: {data.__name__}")

    from bayesdiff import sampler

    print(f"  sampler: {sampler.__name__}")

    from bayesdiff import gen_uncertainty

    print(f"  gen_uncertainty: {gen_uncertainty.__name__}")

    from bayesdiff import gp_oracle

    print(f"  gp_oracle: {gp_oracle.__name__}")

    from bayesdiff import fusion

    print(f"  fusion: {fusion.__name__}")

    from bayesdiff import calibration

    print(f"  calibration: {calibration.__name__}")

    from bayesdiff import ood

    print(f"  ood: {ood.__name__}")

    from bayesdiff import evaluate

    print(f"  evaluate: {evaluate.__name__}")

    print("  All imports OK!\n")


def test_gen_uncertainty():
    """Test gen_uncertainty with synthetic embeddings."""
    import numpy as np
    from bayesdiff.gen_uncertainty import estimate_gen_uncertainty

    print("Testing gen_uncertainty...")
    np.random.seed(42)

    # Simulate M=16 embeddings of dimension d=64
    M, d = 16, 64
    embeddings = np.random.randn(M, d) * 0.5 + np.random.randn(d)

    result = estimate_gen_uncertainty(embeddings, detect_modes=True)
    print(f"  z_bar shape: {result.z_bar.shape}")
    print(f"  cov_gen shape: {result.cov_gen.shape}")
    print(f"  n_modes: {result.n_modes}")
    print(f"  trace(Σ̂_gen): {result.trace_cov:.4f}")
    assert result.z_bar.shape == (d,)
    assert result.cov_gen.shape == (d, d)
    print("  gen_uncertainty OK!\n")


def test_gp_oracle():
    """Test GP training + prediction with synthetic data."""
    import numpy as np
    from bayesdiff.gp_oracle import GPOracle

    print("Testing gp_oracle...")
    np.random.seed(42)

    N, d = 200, 32
    X = np.random.randn(N, d).astype(np.float32)
    # Simple linear relationship + noise
    w = np.random.randn(d).astype(np.float32)
    y = (X @ w + np.random.randn(N) * 0.5).astype(np.float32)

    gp = GPOracle(d=d, n_inducing=50, device="cpu")
    history = gp.train(X, y, n_epochs=20, batch_size=64, verbose=False)
    print(f"  Training done. Final loss: {history['loss'][-1]:.4f}")

    mu, var = gp.predict(X[:10])
    print(f"  Predictions: μ range [{mu.min():.2f}, {mu.max():.2f}]")
    print(f"  Predictions: σ² range [{var.min():.4f}, {var.max():.4f}]")

    # Test Jacobian
    mu_j, var_j, J = gp.predict_with_jacobian(X[:5])
    print(f"  Jacobian shape: {J.shape}")
    assert J.shape == (5, d)
    print("  gp_oracle OK!\n")


def test_fusion():
    """Test uncertainty fusion."""
    import numpy as np
    from bayesdiff.fusion import fuse_uncertainties

    print("Testing fusion...")
    d = 32
    result = fuse_uncertainties(
        mu_oracle=7.5,
        sigma2_oracle=0.5,
        J_mu=np.random.randn(d) * 0.1,
        cov_gen=np.eye(d) * 0.2,
        y_target=7.0,
    )
    print(f"  μ = {result.mu:.3f}")
    print(f"  σ²_oracle = {result.sigma2_oracle:.3f}")
    print(f"  σ²_gen = {result.sigma2_gen:.3f}")
    print(f"  σ²_total = {result.sigma2_total:.3f}")
    print(f"  P_success = {result.p_success:.3f}")
    assert result.sigma2_total >= result.sigma2_oracle
    print("  fusion OK!\n")


def test_calibration():
    """Test calibration + ECE."""
    import numpy as np
    from bayesdiff.calibration import IsotonicCalibrator, compute_ece

    print("Testing calibration...")
    np.random.seed(42)

    N = 500
    p_raw = np.random.beta(2, 5, N)  # Uncalibrated predictions
    y_true = (np.random.rand(N) < p_raw * 1.5).astype(float)  # Noisy labels

    ece_before = compute_ece(p_raw, y_true)
    print(f"  ECE before calibration: {ece_before:.4f}")

    cal = IsotonicCalibrator()
    cal.fit(p_raw[:300], y_true[:300])
    p_cal = cal.transform(p_raw[300:])
    ece_after = compute_ece(p_cal, y_true[300:])
    print(f"  ECE after calibration: {ece_after:.4f}")
    print("  calibration OK!\n")


def test_ood():
    """Test OOD detection."""
    import numpy as np
    from bayesdiff.ood import MahalanobisOOD

    print("Testing OOD detection...")
    np.random.seed(42)

    d = 32
    X_train = np.random.randn(100, d)

    detector = MahalanobisOOD()
    detector.fit(X_train, percentile=95.0)

    # In-distribution
    x_id = np.random.randn(d) * 0.5
    r_id = detector.score(x_id)
    print(f"  In-dist: d_M={r_id.mahalanobis_distance:.2f}, OOD={r_id.is_ood}")

    # Out-of-distribution
    x_ood = np.ones(d) * 10
    r_ood = detector.score(x_ood)
    print(f"  OOD:     d_M={r_ood.mahalanobis_distance:.2f}, OOD={r_ood.is_ood}")
    assert r_ood.is_ood
    print("  OOD detection OK!\n")


def test_evaluate():
    """Test evaluation metrics."""
    import numpy as np
    from bayesdiff.evaluate import evaluate_all, print_results

    print("Testing evaluate...")
    np.random.seed(42)

    N = 200
    y_true = np.random.uniform(4, 12, N)
    mu_pred = y_true + np.random.randn(N) * 0.8
    sigma_pred = np.abs(np.random.randn(N) * 0.3) + 0.5
    p_success = 1.0 / (1.0 + np.exp(-(mu_pred - 7.0)))

    results = evaluate_all(
        mu_pred=mu_pred,
        sigma_pred=sigma_pred,
        p_success=p_success,
        y_true=y_true,
        y_target=7.0,
    )
    print_results(results)
    print("  evaluate OK!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("BayesDiff Phase 0 Debug Pipeline")
    print("=" * 60 + "\n")

    test_imports()
    test_gen_uncertainty()
    test_gp_oracle()
    test_fusion()
    test_calibration()
    test_ood()
    test_evaluate()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
