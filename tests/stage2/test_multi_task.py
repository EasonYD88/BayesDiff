"""Unit tests for bayesdiff/multi_task.py (Sub-Plan 05, Phase B)."""

import pytest
import torch
import numpy as np

from bayesdiff.multi_task import (
    SharedTrunk,
    RegressionHead,
    ClassificationHead,
    RankingHead,
    MultiTaskTrunk,
    GroupedBatchSampler,
    within_group_ndcg,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def trunk_v1():
    return MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=False)


@pytest.fixture
def trunk_v2():
    return MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)


@pytest.fixture
def synthetic_data():
    torch.manual_seed(42)
    B = 64
    z = torch.randn(B, 128)
    w = torch.randn(128)
    y = z @ w / 128 * 3 + 7
    groups = torch.tensor([i % 8 for i in range(B)])
    return z, y, groups


# ── T1.1: Trunk output shape ─────────────────────────────────────


def test_trunk_output_shape():
    trunk = SharedTrunk(input_dim=128, hidden_dim=256, output_dim=64)
    z = torch.randn(32, 128)
    h = trunk(z)
    assert h.shape == (32, 64)


# ── T1.2: Residual connection ────────────────────────────────────


def test_trunk_residual_connection():
    z = torch.randn(16, 128)
    trunk_res = SharedTrunk(128, 256, 64, residual=True)
    trunk_nores = SharedTrunk(128, 256, 64, residual=False)
    trunk_nores.mlp.load_state_dict(trunk_res.mlp.state_dict())
    h_res = trunk_res(z)
    h_nores = trunk_nores(z)
    assert not torch.allclose(h_res, h_nores, atol=1e-6)


# ── T1.3: Regression head shape ──────────────────────────────────


def test_reg_head_shape():
    head = RegressionHead(64)
    h = torch.randn(32, 64)
    out = head(h)
    assert out.shape == (32,)


# ── T1.4: Classification head range ──────────────────────────────


def test_cls_head_range():
    head = ClassificationHead(64)
    h = torch.randn(100, 64) * 10
    probs = head.predict_prob(h)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert probs.shape == (100,)


# ── T1.5: Classification head logits unbounded ───────────────────


def test_cls_head_logits_unbounded():
    head = ClassificationHead(64)
    h = torch.randn(100, 64) * 10
    logits = head(h)
    assert logits.min() < 0 or logits.max() > 1  # not bounded to [0,1]


# ── T1.6: Ranking head shape ─────────────────────────────────────


def test_rank_head_shape():
    head = RankingHead(64)
    h = torch.randn(32, 64)
    out = head(h)
    assert out.shape == (32,)


# ── T1.7: v1 forward (no ranking) ────────────────────────────────


def test_multitask_forward_v1(trunk_v1):
    z = torch.randn(32, 128)
    h, reg, cls, rank = trunk_v1(z)
    assert h.shape == (32, 64)
    assert reg.shape == (32,)
    assert cls.shape == (32,)
    assert rank is None


# ── T1.8: v2 forward (with ranking) ──────────────────────────────


def test_multitask_forward_v2(trunk_v2):
    z = torch.randn(32, 128)
    h, reg, cls, rank = trunk_v2(z)
    assert h.shape == (32, 64)
    assert reg.shape == (32,)
    assert cls.shape == (32,)
    assert rank is not None
    assert rank.shape == (32,)


# ── T1.9: Grouped pair generation ────────────────────────────────


def test_grouped_pair_generation():
    y = torch.tensor([8.0, 6.0, 7.5, 5.0, 9.0, 4.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups)

    for i, j in pairs:
        assert y[i] > y[j], f"Pair ({i},{j}): y[{i}]={y[i]} should > y[{j}]={y[j]}"
        assert groups[i] == groups[j], (
            f"Cross-group pair: group[{i}]={groups[i]} != group[{j}]={groups[j]}"
        )
    assert len(pairs) == 6


# ── T1.10: No cross-group pairs ──────────────────────────────────


def test_no_cross_group_pairs():
    torch.manual_seed(0)
    y = torch.randn(100) * 3 + 7
    groups = torch.randint(0, 10, (100,))
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups)

    for i, j in pairs:
        assert groups[i] == groups[j]


# ── T1.11: Empty groups produce no pairs ──────────────────────────


def test_pair_generation_empty_group():
    y = torch.tensor([7.0, 6.0, 8.0])
    groups = torch.tensor([0, 1, 2])  # each group has 1 member
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups)
    assert len(pairs) == 0


# ── T1.12: Pair subsample cap ────────────────────────────────────


def test_pair_subsample_cap():
    y = torch.arange(20, dtype=torch.float)
    groups = torch.zeros(20, dtype=torch.long)
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups, max_pairs_per_group=30)
    assert len(pairs) <= 30


# ── T1.13: BPR loss gradient direction ───────────────────────────


def test_bpr_loss_correct_gradient():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    z = torch.randn(10, 128)
    y = torch.arange(10, dtype=torch.float)
    groups = torch.zeros(10, dtype=torch.long)

    loss, ld = trunk.compute_loss(z, y, groups=groups)
    assert loss.requires_grad
    assert ld["n_pairs"] > 0
    assert ld["L_rank"] > 0


# ── T1.14: Classification loss correct labels ────────────────────


def test_cls_loss_correct_labels():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, activity_threshold=7.0)
    z = torch.randn(100, 128)
    # All high pKd → all active
    y_high = torch.full((100,), 9.0)
    _, ld_high = trunk.compute_loss(z, y_high)
    # All low pKd → all inactive
    y_low = torch.full((100,), 4.0)
    _, ld_low = trunk.compute_loss(z, y_low)
    # Both should produce finite loss
    assert np.isfinite(ld_high["L_cls"])
    assert np.isfinite(ld_low["L_cls"])


# ── T1.15: v1 joint loss composition ─────────────────────────────


def test_joint_loss_v1():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=False)
    z = torch.randn(32, 128)
    y = torch.randn(32) * 2 + 7

    loss, ld = trunk.compute_loss(z, y, lambda_reg=1.0, lambda_cls=0.5)
    assert ld["L_rank"] == 0.0
    assert ld["n_pairs"] == 0
    expected = 1.0 * ld["L_reg"] + 0.5 * ld["L_cls"]
    assert abs(ld["L_total"] - expected) < 1e-4


# ── T1.16: v2 joint loss composition ─────────────────────────────


def test_joint_loss_v2():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    z = torch.randn(32, 128)
    y = torch.arange(32, dtype=torch.float)
    groups = torch.tensor([i % 4 for i in range(32)])

    loss, ld = trunk.compute_loss(
        z, y, groups=groups, lambda_reg=1.0, lambda_cls=0.5, lambda_rank=0.3
    )
    assert ld["n_pairs"] > 0
    expected = 1.0 * ld["L_reg"] + 0.5 * ld["L_cls"] + 0.3 * ld["L_rank"]
    assert abs(ld["L_total"] - expected) < 1e-4


# ── T1.17: Ranking requires groups ───────────────────────────────


def test_joint_loss_ranking_requires_groups():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    z = torch.randn(32, 128)
    y = torch.randn(32) * 2 + 7

    with pytest.raises(AssertionError):
        trunk.compute_loss(z, y, groups=None)


# ── T1.18: Learned task weights ──────────────────────────────────


def test_learned_task_weights():
    trunk = MultiTaskTrunk(
        input_dim=128, trunk_dim=64, enable_ranking=False, learned_weights=True
    )
    assert hasattr(trunk, "log_sigma")
    assert trunk.log_sigma.shape == (2,)
    # Weights should produce positive sigmas
    sigmas = torch.exp(trunk.log_sigma)
    assert (sigmas > 0).all()


# ── T1.19: Learned weights gradient ──────────────────────────────


def test_learned_weights_gradient():
    trunk = MultiTaskTrunk(
        input_dim=128, trunk_dim=64, learned_weights=True
    )
    z = torch.randn(16, 128)
    y = torch.randn(16) * 2 + 7
    loss, _ = trunk.compute_loss(z, y)
    loss.backward()
    assert trunk.log_sigma.grad is not None
    assert trunk.log_sigma.grad.abs().sum() > 0


# ── T1.20: Gradient flow to all heads ─────────────────────────────


def test_gradient_flow_all_heads(trunk_v2, synthetic_data):
    z, y, groups = synthetic_data
    loss, _ = trunk_v2.compute_loss(z, y, groups=groups)
    loss.backward()

    for name, param in trunk_v2.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ── T1.21: Training convergence ──────────────────────────────────


def test_training_convergence():
    torch.manual_seed(42)
    X = torch.randn(200, 128)
    w = torch.randn(128)
    y = X @ w / 128 * 3 + 7

    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, activity_threshold=7.0)
    optimizer = torch.optim.Adam(trunk.parameters(), lr=1e-3)

    losses = []
    for epoch in range(50):
        total_loss, _ = trunk.compute_loss(X, y)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"Loss did not decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


# ── T1.22: Feature extraction ────────────────────────────────────


def test_trunk_features_extractable(trunk_v1):
    z_np = np.random.randn(50, 128).astype(np.float32)
    h = trunk_v1.extract_trunk_features(z_np)
    assert isinstance(h, np.ndarray)
    assert h.shape == (50, 64)
    assert h.dtype == np.float32


# ── T1.23: cls_pos_weight ────────────────────────────────────────


def test_cls_pos_weight():
    z = torch.randn(32, 128)
    y = torch.randn(32) * 2 + 7  # mixed active/inactive

    t_no_pw = MultiTaskTrunk(input_dim=128, trunk_dim=64)
    t_pw = MultiTaskTrunk(input_dim=128, trunk_dim=64, cls_pos_weight=3.0)
    # Copy weights
    t_pw.load_state_dict(t_no_pw.state_dict())

    _, ld_no = t_no_pw.compute_loss(z, y)
    _, ld_pw = t_pw.compute_loss(z, y)
    # pos_weight should change the cls loss
    assert ld_no["L_cls"] != pytest.approx(ld_pw["L_cls"], abs=1e-6)


# ── T1.24: Loss dict keys ────────────────────────────────────────


def test_loss_dict_keys(trunk_v2, synthetic_data):
    z, y, groups = synthetic_data
    _, ld = trunk_v2.compute_loss(z, y, groups=groups)
    expected_keys = {"L_reg", "L_cls", "L_rank", "L_total", "n_pairs"}
    assert expected_keys.issubset(set(ld.keys()))


# ── GroupedBatchSampler ───────────────────────────────────────────


def test_grouped_batch_sampler_basic():
    group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    sampler = GroupedBatchSampler(group_ids, batch_size=8, min_group_size=2)
    batches = list(sampler)
    # All indices should appear
    all_indices = sorted(sum(batches, []))
    assert len(all_indices) == 12


def test_grouped_batch_sampler_min_group_size():
    group_ids = np.array([0, 0, 0, 1, 2, 2, 2])
    sampler = GroupedBatchSampler(group_ids, batch_size=4, min_group_size=3)
    batches = list(sampler)
    all_indices = sorted(sum(batches, []))
    # Group 1 has only 1 member, should be excluded
    assert 3 not in all_indices


# ── within_group_ndcg ─────────────────────────────────────────────


def test_within_group_ndcg_perfect():
    y_true = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    y_pred = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    result = within_group_ndcg(y_true, y_pred, groups, k=5)
    assert result["ndcg_mean"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_groups_evaluated"] == 2


def test_within_group_ndcg_scrambled():
    y_true = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    y_pred_good = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    y_pred_bad = np.array([0, 1, 2, 3, 4, 9, 8, 7, 6, 5], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    result_good = within_group_ndcg(y_true, y_pred_good, groups, k=5)
    result_bad = within_group_ndcg(y_true, y_pred_bad, groups, k=5)
    assert result_bad["ndcg_mean"] < result_good["ndcg_mean"]
