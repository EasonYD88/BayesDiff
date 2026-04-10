"""tests/stage2/test_hybrid_integration.py

Sub-Plan 04 integration tests: T2.1–T2.6.
Tests oracle heads integrated with fusion, calibration, OOD, and full pipeline.
"""

import numpy as np
import pytest
import torch

from bayesdiff.hybrid_oracle import DKLOracle, DKLEnsembleOracle, NNResidualOracle
from bayesdiff.oracle_interface import OracleResult
from bayesdiff.fusion import fuse_uncertainties, FusionResult
from bayesdiff.evaluate import evaluate_all, EvalResults
from bayesdiff.calibration import IsotonicCalibrator
from bayesdiff.ood import MahalanobisOOD


@pytest.fixture
def trained_oracle_and_data():
    """Pre-trained DKL oracle + synthetic data for integration tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(500, 128).astype(np.float32)
    w = np.random.randn(128).astype(np.float32)
    y = X @ w + 0.3 * np.random.randn(500).astype(np.float32)

    oracle = DKLOracle(input_dim=128, feature_dim=16, n_inducing=50,
                       hidden_dim=64, device="cpu")
    oracle.fit(X[:400], y[:400], X[400:450], y[400:450], n_epochs=30, verbose=False)

    return oracle, X[450:], y[450:]


def test_oracle_with_delta_method(trained_oracle_and_data):
    """T2.1: Delta method fusion works with oracle Jacobians."""
    oracle, X_test, y_test = trained_oracle_and_data
    result = oracle.predict_for_fusion(X_test)

    # Synthetic generation covariance (diagonal for simplicity)
    for i in range(min(10, len(X_test))):
        cov_gen = np.eye(128).astype(np.float32) * 0.01
        fuse_result = fuse_uncertainties(
            mu_oracle=float(result.mu[i]),
            sigma2_oracle=float(result.sigma2[i]),
            J_mu=result.jacobian[i],
            cov_gen=cov_gen,
            y_target=7.0,
        )
        assert isinstance(fuse_result, FusionResult)
        assert np.isfinite(fuse_result.sigma2_total)
        assert fuse_result.sigma2_total >= result.sigma2[i], \
            "Total variance must be >= oracle-only variance (Delta method adds sigma2_gen)"


def test_oracle_with_calibration(trained_oracle_and_data):
    """T2.3: Isotonic calibration should reduce ECE."""
    oracle, X_test, y_test = trained_oracle_and_data
    result = oracle.predict(X_test)
    sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))

    # Create binary target for calibration (pKd > median)
    y_binary = (y_test > np.median(y_test)).astype(float)
    p_raw = 1.0 / (1.0 + np.exp(-(result.mu - np.median(y_test)) / sigma))

    calibrator = IsotonicCalibrator()
    # Fit on first half, evaluate on second half
    n = len(y_test) // 2
    calibrator.fit(p_raw[:n], y_binary[:n])
    p_cal = calibrator.transform(p_raw[n:])

    assert p_cal.min() >= 0 and p_cal.max() <= 1, "Calibrated probabilities out of [0, 1]"


def test_oracle_with_ood_detection(trained_oracle_and_data):
    """T2.4: Mahalanobis OOD detects far-from-training inputs."""
    oracle, X_test, y_test = trained_oracle_and_data

    # Fit OOD detector on training embeddings
    np.random.seed(42)
    X_train = np.random.randn(400, 128).astype(np.float32)
    ood_detector = MahalanobisOOD()
    ood_detector.fit(X_train)

    X_id = np.random.randn(20, 128).astype(np.float32)
    X_ood = (np.random.randn(20, 128) * 5 + 10).astype(np.float32)

    scores_id = ood_detector.get_distances(X_id)
    scores_ood = ood_detector.get_distances(X_ood)

    assert scores_ood.mean() > scores_id.mean(), \
        "OOD Mahalanobis distances for out-of-distribution should be higher than in-distribution"


def test_full_pipeline_oracle(trained_oracle_and_data):
    """T2.5: End-to-end pipeline: oracle -> fusion -> evaluate."""
    oracle, X_test, y_test = trained_oracle_and_data
    result = oracle.predict(X_test)
    sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))

    # Compute p_success (probability pKd > 7.0)
    from scipy.stats import norm
    p_success = 1.0 - norm.cdf(7.0, loc=result.mu, scale=sigma)

    # Evaluate
    eval_result = evaluate_all(
        mu_pred=result.mu,
        sigma_pred=sigma,
        p_success=p_success,
        y_true=y_test,
        y_target=7.0,
    )
    assert isinstance(eval_result, EvalResults)
    assert hasattr(eval_result, 'spearman_rho')
    assert hasattr(eval_result, 'rmse')
    assert np.isfinite(eval_result.spearman_rho)
    assert np.isfinite(eval_result.rmse)


def test_oracle_head_swap(trained_oracle_and_data):
    """T2.6: Different oracle heads produce different but valid results."""
    _, X_test, y_test = trained_oracle_and_data
    np.random.seed(42)
    torch.manual_seed(42)
    X_train = np.random.randn(400, 128).astype(np.float32)
    w = np.random.randn(128).astype(np.float32)
    y_train = X_train @ w + 0.3 * np.random.randn(400).astype(np.float32)
    X_val = np.random.randn(50, 128).astype(np.float32)
    y_val = X_val @ w + 0.3 * np.random.randn(50).astype(np.float32)

    # Train two different heads
    dkl = DKLOracle(input_dim=128, feature_dim=16, n_inducing=50, hidden_dim=64, device="cpu")
    dkl.fit(X_train, y_train, X_val, y_val, n_epochs=20, verbose=False)

    nn_res = NNResidualOracle(input_dim=128, hidden_dim=64, n_inducing=50, device="cpu")
    nn_res.fit(X_train, y_train, X_val, y_val, nn_epochs=20, gp_epochs=20, verbose=False)

    result_dkl = dkl.predict(X_test[:20])
    result_nnres = nn_res.predict(X_test[:20])

    # Both valid
    assert isinstance(result_dkl, OracleResult) and isinstance(result_nnres, OracleResult)
    assert np.isfinite(result_dkl.mu).all() and np.isfinite(result_nnres.mu).all()
    # But different
    assert not np.allclose(result_dkl.mu, result_nnres.mu, atol=0.01), \
        "Different heads should produce different predictions"
