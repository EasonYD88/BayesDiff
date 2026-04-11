"""Integration tests for multi-task trunk + DKL Ensemble (Sub-Plan 05, Phase B)."""

import pytest
import torch
import numpy as np
import tempfile

from bayesdiff.multi_task import MultiTaskTrunk, MultiTaskHybridOracle, within_group_ndcg
from bayesdiff.hybrid_oracle import DKLEnsembleOracle
from bayesdiff.oracle_interface import OracleResult


# ── T2.1: Trunk → DKL Ensemble ───────────────────────────────────


def test_trunk_to_dkl_ensemble():
    """Full two-stage training on tiny synthetic data."""
    torch.manual_seed(42)
    np.random.seed(42)
    N_train, N_val = 200, 50
    d = 128

    X_train = np.random.randn(N_train, d).astype(np.float32)
    y_train = np.random.randn(N_train).astype(np.float32) * 2 + 7
    X_val = np.random.randn(N_val, d).astype(np.float32)
    y_val = np.random.randn(N_val).astype(np.float32) * 2 + 7

    trunk = MultiTaskTrunk(input_dim=d, trunk_dim=64, hidden_dim=128)
    oracle = DKLEnsembleOracle(
        input_dim=64, n_members=2, feature_dim=16, n_inducing=32, device="cpu"
    )
    hybrid = MultiTaskHybridOracle(trunk, oracle)

    # Stage 1
    history = hybrid.train_trunk(
        X_train, y_train, X_val, y_val, n_epochs=10, device="cpu"
    )
    assert len(history["train_loss"]) > 0

    # Stage 2
    oracle_hist = hybrid.train_oracle(
        X_train, y_train, X_val, y_val, n_epochs=10, verbose=False
    )
    assert "member_histories" in oracle_hist

    # Predict
    result = hybrid.predict(X_val)
    assert isinstance(result, OracleResult)
    assert result.mu.shape == (N_val,)
    assert result.sigma2.shape == (N_val,)
    assert np.all(result.sigma2 > 0)
    assert "cls_prob" in result.aux


# ── T2.2: h_trunk satisfies OracleHead contract ──────────────────


def test_trunk_to_oracle_interface():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64)
    X = np.random.randn(100, 128).astype(np.float32)
    h = trunk.extract_trunk_features(X)

    assert h.dtype == np.float32
    assert h.shape == (100, 64)
    assert np.isfinite(h).all()


# ── T2.3: Jacobian through trunk + oracle ────────────────────────


def test_jacobian_through_trunk():
    """Test that predict_for_fusion returns a Jacobian in z-space."""
    torch.manual_seed(42)
    np.random.seed(42)
    N = 10
    d = 32  # small for speed

    X_train = np.random.randn(50, d).astype(np.float32)
    y_train = np.random.randn(50).astype(np.float32) * 2 + 7
    X_val = np.random.randn(20, d).astype(np.float32)
    y_val = np.random.randn(20).astype(np.float32) * 2 + 7

    trunk = MultiTaskTrunk(input_dim=d, trunk_dim=16, hidden_dim=32)
    oracle = DKLEnsembleOracle(
        input_dim=16, n_members=2, feature_dim=8, n_inducing=16, device="cpu"
    )
    hybrid = MultiTaskHybridOracle(trunk, oracle)
    hybrid.train_trunk(X_train, y_train, X_val, y_val, n_epochs=5, device="cpu")
    hybrid.train_oracle(X_train, y_train, X_val, y_val, n_epochs=5, verbose=False)

    X_test = np.random.randn(N, d).astype(np.float32)
    result = hybrid.predict_for_fusion(X_test)

    assert result.jacobian is not None
    assert result.jacobian.shape == (N, d)  # ∂μ/∂z in original z-space
    assert np.isfinite(result.jacobian).all()


# ── T2.5: Full pipeline v1 ───────────────────────────────────────


def test_full_pipeline_v1():
    """End-to-end: trunk(reg+cls) → DKL Ensemble → evaluate."""
    torch.manual_seed(42)
    np.random.seed(42)

    N_train, N_val, N_test = 100, 30, 20
    d = 64

    X_train = np.random.randn(N_train, d).astype(np.float32)
    y_train = np.random.randn(N_train).astype(np.float32) * 2 + 7
    X_val = np.random.randn(N_val, d).astype(np.float32)
    y_val = np.random.randn(N_val).astype(np.float32) * 2 + 7
    X_test = np.random.randn(N_test, d).astype(np.float32)
    y_test = np.random.randn(N_test).astype(np.float32) * 2 + 7

    trunk = MultiTaskTrunk(input_dim=d, trunk_dim=32, hidden_dim=64)
    oracle = DKLEnsembleOracle(
        input_dim=32, n_members=2, feature_dim=8, n_inducing=16, device="cpu"
    )
    hybrid = MultiTaskHybridOracle(trunk, oracle)

    hybrid.train_trunk(X_train, y_train, X_val, y_val, n_epochs=5, device="cpu")
    hybrid.train_oracle(
        X_train, y_train, X_val, y_val, n_epochs=5, verbose=False
    )

    # Use OracleHead.evaluate() — the Tier 1 evaluation path
    metrics = oracle.evaluate(
        hybrid.multi_task.extract_trunk_features(X_test), y_test
    )
    assert "spearman_rho" in metrics
    assert "nll" in metrics
    assert "err_sigma_rho" in metrics
    assert np.isfinite(metrics["spearman_rho"])


# ── T2.7: Save/load roundtrip ────────────────────────────────────


def test_save_load_roundtrip():
    torch.manual_seed(42)
    np.random.seed(42)

    trunk = MultiTaskTrunk(input_dim=32, trunk_dim=16, hidden_dim=32)
    oracle = DKLEnsembleOracle(
        input_dim=16, n_members=2, feature_dim=8, n_inducing=16, device="cpu"
    )
    hybrid = MultiTaskHybridOracle(trunk, oracle)

    X = np.random.randn(50, 32).astype(np.float32)
    y = np.random.randn(50).astype(np.float32) * 2 + 7

    hybrid.train_trunk(X, y, X, y, n_epochs=5, device="cpu")
    hybrid.train_oracle(X, y, X, y, n_epochs=5, verbose=False)

    result_before = hybrid.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        hybrid.save(tmpdir)

        trunk2 = MultiTaskTrunk(input_dim=32, trunk_dim=16, hidden_dim=32)
        oracle2 = DKLEnsembleOracle(
            input_dim=16, n_members=2, feature_dim=8, n_inducing=16, device="cpu"
        )
        hybrid2 = MultiTaskHybridOracle(trunk2, oracle2)
        hybrid2.load(tmpdir)

        result_after = hybrid2.predict(X)

    np.testing.assert_allclose(result_before.mu, result_after.mu, atol=1e-5)
    np.testing.assert_allclose(result_before.sigma2, result_after.sigma2, atol=1e-5)


# ── T2.8: Within-group NDCG ──────────────────────────────────────


def test_within_group_ndcg_integration():
    y_true = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    y_pred = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    result = within_group_ndcg(y_true, y_pred, groups, k=5)
    assert result["ndcg_mean"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_groups_evaluated"] == 2
