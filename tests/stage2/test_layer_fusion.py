"""Tests for layer_fusion.py — T1.1, T1.2."""

import pytest
import torch

from bayesdiff.layer_fusion import WeightedSumFusion


class TestWeightedSumFusion:
    """T1.1–T1.2: WeightedSumFusion correctness."""

    def test_T1_1_weights_sum_to_one_and_positive(self):
        """T1.1a: Softmax weights sum to 1 and are all positive."""
        fuse = WeightedSumFusion(n_layers=10)
        embeds = [torch.randn(4, 128) for _ in range(10)]
        z_fuse, weights = fuse(embeds)

        assert weights.shape == (10,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert (weights > 0).all()

    def test_T1_1_output_shape(self):
        """T1.1b: Output shape matches input embedding shape."""
        fuse = WeightedSumFusion(n_layers=5)
        embeds = [torch.randn(8, 64) for _ in range(5)]
        z_fuse, weights = fuse(embeds)

        assert z_fuse.shape == (8, 64)
        assert weights.shape == (5,)

    def test_T1_1_gradients_flow(self):
        """T1.1c: Gradients flow through to logits."""
        fuse = WeightedSumFusion(n_layers=3)
        embeds = [torch.randn(4, 32, requires_grad=True) for _ in range(3)]
        z_fuse, weights = fuse(embeds)
        loss = z_fuse.sum()
        loss.backward()

        assert fuse.logits.grad is not None
        assert fuse.logits.grad.shape == (3,)
        assert not torch.all(fuse.logits.grad == 0)

    def test_T1_2_uniform_initialization(self):
        """T1.2: Initial weights are uniform (all equal)."""
        fuse = WeightedSumFusion(n_layers=10)
        embeds = [torch.randn(2, 128) for _ in range(10)]
        _, weights = fuse(embeds)

        expected = torch.full((10,), 0.1)
        assert torch.allclose(weights, expected, atol=1e-6)

    def test_T1_2_logits_initialized_zero(self):
        """T1.2b: Logits are initialized to zero."""
        fuse = WeightedSumFusion(n_layers=7)
        assert torch.allclose(fuse.logits, torch.zeros(7))

    def test_single_layer_passthrough(self):
        """Edge case: single layer returns that layer unchanged."""
        fuse = WeightedSumFusion(n_layers=1)
        embed = torch.randn(4, 128)
        z_fuse, weights = fuse([embed])

        assert torch.allclose(z_fuse, embed)
        assert torch.allclose(weights, torch.tensor([1.0]))
