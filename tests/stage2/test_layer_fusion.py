"""Tests for layer_fusion.py — T1.1, T1.2, T1.3, T1.4."""

import pytest
import torch

from bayesdiff.layer_fusion import WeightedSumFusion, LayerAttentionFusion, ConcatMLPFusion


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


class TestLayerAttentionFusion:
    """T1.3: LayerAttentionFusion correctness."""

    def test_T1_3_weights_sum_to_one_per_sample(self):
        """T1.3a: Per-sample attention weights sum to 1."""
        fuse = LayerAttentionFusion(embed_dim=128, hidden_dim=64)
        embeds = [torch.randn(8, 128) for _ in range(10)]
        z_fuse, weights = fuse(embeds)

        assert weights.shape == (8, 10)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-6)
        assert (weights > 0).all()

    def test_T1_3_output_shape(self):
        """T1.3b: Output shape matches input embedding shape."""
        fuse = LayerAttentionFusion(embed_dim=64, hidden_dim=32)
        embeds = [torch.randn(4, 64) for _ in range(5)]
        z_fuse, weights = fuse(embeds)

        assert z_fuse.shape == (4, 64)
        assert weights.shape == (4, 5)

    def test_T1_3_gradients_flow(self):
        """T1.3c: Gradients flow through to attention parameters."""
        fuse = LayerAttentionFusion(embed_dim=32, hidden_dim=16)
        embeds = [torch.randn(4, 32, requires_grad=True) for _ in range(3)]
        z_fuse, weights = fuse(embeds)
        loss = z_fuse.sum()
        loss.backward()

        assert fuse.W.weight.grad is not None
        assert fuse.u.weight.grad is not None
        assert not torch.all(fuse.W.weight.grad == 0)

    def test_T1_3_weights_vary_across_samples(self):
        """T1.3d: Different inputs produce different layer weights."""
        fuse = LayerAttentionFusion(embed_dim=128, hidden_dim=64)
        # Use very different embeddings to ensure weight variation
        embeds = [torch.randn(16, 128) * (i + 1) for i in range(5)]
        _, weights = fuse(embeds)

        # Weights should differ across samples (not all identical rows)
        weight_std = weights.std(dim=0)  # std across samples per layer
        assert weight_std.max() > 1e-6, "Weights are identical across samples"

    def test_T1_3_single_layer(self):
        """Edge case: single layer returns weight=1 for all samples."""
        fuse = LayerAttentionFusion(embed_dim=64, hidden_dim=32)
        embed = torch.randn(4, 64)
        z_fuse, weights = fuse([embed])

        assert weights.shape == (4, 1)
        assert torch.allclose(weights, torch.ones(4, 1))


class TestConcatMLPFusion:
    """T1.4: ConcatMLPFusion correctness."""

    def test_T1_4_output_shape(self):
        """T1.4a: Output shape = (B, output_dim) for various configs."""
        for n_layers, embed_dim, output_dim, batch in [
            (10, 128, 128, 8),
            (10, 128, 64, 4),
            (10, 128, 256, 16),
            (3, 64, 32, 2),
        ]:
            fuse = ConcatMLPFusion(embed_dim, n_layers, output_dim)
            embeds = [torch.randn(batch, embed_dim) for _ in range(n_layers)]
            z_fuse, weights = fuse(embeds)

            assert z_fuse.shape == (batch, output_dim), (
                f"Expected ({batch}, {output_dim}), got {z_fuse.shape}"
            )
            assert weights is None

    def test_T1_4_gradients_flow(self):
        """T1.4b: Gradients flow through MLP to all parameters."""
        fuse = ConcatMLPFusion(embed_dim=128, n_layers=10, output_dim=128)
        embeds = [torch.randn(4, 128, requires_grad=True) for _ in range(10)]
        z_fuse, _ = fuse(embeds)
        loss = z_fuse.sum()
        loss.backward()

        for name, param in fuse.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert not torch.all(param.grad == 0), f"Zero grad for {name}"

    def test_T1_4_different_output_dims(self):
        """T1.4c: Output dim can differ from embed_dim."""
        for out_d in [64, 128, 256]:
            fuse = ConcatMLPFusion(embed_dim=128, n_layers=10, output_dim=out_d)
            embeds = [torch.randn(4, 128) for _ in range(10)]
            z_fuse, _ = fuse(embeds)
            assert z_fuse.shape == (4, out_d)

    def test_T1_4_no_nan_inf(self):
        """T1.4d: No NaN/Inf with normal inputs."""
        fuse = ConcatMLPFusion(embed_dim=128, n_layers=10, output_dim=128)
        embeds = [torch.randn(8, 128) for _ in range(10)]
        z_fuse, _ = fuse(embeds)
        assert torch.isfinite(z_fuse).all()
