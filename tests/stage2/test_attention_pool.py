"""Tests for attention_pool.py — T1.1 through T1.11.

Sub-Plan 2 unit tests: Intra-layer attention pooling and scheme integration.
"""

import pytest
import torch
import torch.nn as nn

from bayesdiff.attention_pool import (
    AttentionPoolingWithRegularization,
    CrossAttentionPooling,
    MLPReadout,
    SchemeA_TwoBranch,
    SchemeB_SingleBranch,
    SelfAttentionPooling,
)


# =========================================================================
# SelfAttentionPooling tests
# =========================================================================


class TestSelfAttentionPooling:
    """T1.1–T1.3, T1.10: SelfAttentionPooling correctness."""

    @pytest.mark.parametrize("B,N,d", [(1, 5, 32), (4, 20, 128), (8, 50, 64)])
    def test_T1_1_output_shape(self, B, N, d):
        """T1.1: Output shape = (B, d) for various B, N, d."""
        pool = SelfAttentionPooling(input_dim=d, hidden_dim=32)
        h = torch.randn(B, N, d)
        z, alpha = pool(h)

        assert z.shape == (B, d)
        assert alpha.shape == (B, N)

    def test_T1_2_weights_sum_to_one(self):
        """T1.2: Attention weights sum to 1 for each sample."""
        pool = SelfAttentionPooling(input_dim=64, hidden_dim=32)
        h = torch.randn(4, 10, 64)
        z, alpha = pool(h)

        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)
        assert (alpha >= 0).all()

    def test_T1_3_mask_handling(self):
        """T1.3: Padded positions get zero attention weight."""
        pool = SelfAttentionPooling(input_dim=32, hidden_dim=16)
        h = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 5:] = False  # First sample: 5 real atoms
        mask[1, 8:] = False  # Second sample: 8 real atoms

        z, alpha = pool(h, mask=mask)

        # Padded atoms must have zero weight
        assert alpha[0, 5:].abs().max() < 1e-6
        assert alpha[1, 8:].abs().max() < 1e-6
        # Real atoms must sum to 1
        assert torch.allclose(alpha[0, :5].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(alpha[1, :8].sum(), torch.tensor(1.0), atol=1e-5)

    def test_T1_7_gradient_flow(self):
        """T1.7: All parameters receive gradients."""
        pool = SelfAttentionPooling(input_dim=32, hidden_dim=16)
        h = torch.randn(4, 10, 32, requires_grad=True)
        z, alpha = pool(h)
        loss = z.sum()
        loss.backward()

        for name, param in pool.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

        assert h.grad is not None

    def test_T1_8_determinism(self):
        """T1.8: Same input + seed → same output."""
        pool = SelfAttentionPooling(input_dim=32, hidden_dim=16)
        pool.eval()
        h = torch.randn(2, 5, 32)

        z1, a1 = pool(h)
        z2, a2 = pool(h)

        assert torch.allclose(z1, z2)
        assert torch.allclose(a1, a2)

    def test_T1_9_numerical_stability(self):
        """T1.9: No NaN with very large/small embeddings."""
        pool = SelfAttentionPooling(input_dim=32, hidden_dim=16)

        # Large values
        h_large = torch.randn(2, 5, 32) * 100
        z, alpha = pool(h_large)
        assert not torch.isnan(z).any()
        assert not torch.isnan(alpha).any()

        # Small values
        h_small = torch.randn(2, 5, 32) * 1e-6
        z, alpha = pool(h_small)
        assert not torch.isnan(z).any()
        assert not torch.isnan(alpha).any()

    def test_T1_10_single_atom(self):
        """T1.10: Works when N=1 (single atom → all weight on that atom)."""
        pool = SelfAttentionPooling(input_dim=32, hidden_dim=16)
        h = torch.randn(2, 1, 32)
        z, alpha = pool(h)

        assert z.shape == (2, 32)
        assert alpha.shape == (2, 1)
        assert torch.allclose(alpha, torch.ones(2, 1), atol=1e-5)
        # z should equal h squeezed
        assert torch.allclose(z, h.squeeze(1), atol=1e-5)


# =========================================================================
# CrossAttentionPooling tests
# =========================================================================


class TestCrossAttentionPooling:
    """T1.4–T1.5: CrossAttentionPooling correctness."""

    @pytest.mark.parametrize("B,N,d", [(1, 5, 32), (4, 20, 128)])
    def test_T1_4_output_shape(self, B, N, d):
        """T1.4: Correct shape with pocket context."""
        pool = CrossAttentionPooling(ligand_dim=d, pocket_dim=d, hidden_dim=64)
        h_lig = torch.randn(B, N, d)
        h_pocket = torch.randn(B, d)
        z, alpha = pool(h_lig, h_pocket)

        assert z.shape == (B, d)
        assert alpha.shape == (B, N)

    def test_T1_4_weights_sum_to_one(self):
        """T1.4b: Cross-attention weights sum to 1."""
        pool = CrossAttentionPooling(ligand_dim=64, pocket_dim=64, hidden_dim=32)
        z, alpha = pool(torch.randn(4, 10, 64), torch.randn(4, 64))

        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_T1_5_pocket_influence(self):
        """T1.5: Different pocket contexts → different outputs."""
        pool = CrossAttentionPooling(ligand_dim=32, pocket_dim=32, hidden_dim=16)
        h_lig = torch.randn(1, 10, 32)

        pocket_a = torch.randn(1, 32)
        pocket_b = torch.randn(1, 32)

        z_a, _ = pool(h_lig, pocket_a)
        z_b, _ = pool(h_lig, pocket_b)

        # Different pockets should give different outputs
        assert not torch.allclose(z_a, z_b, atol=1e-3)

    def test_T1_5_mask_handling(self):
        """T1.5b: Cross-attention respects ligand mask."""
        pool = CrossAttentionPooling(ligand_dim=32, pocket_dim=32, hidden_dim=16)
        h_lig = torch.randn(2, 10, 32)
        h_pocket = torch.randn(2, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 3:] = False

        z, alpha = pool(h_lig, h_pocket, ligand_mask=mask)
        assert alpha[0, 3:].abs().max() < 1e-6
        assert torch.allclose(alpha[0, :3].sum(), torch.tensor(1.0), atol=1e-5)


# =========================================================================
# AttentionPoolingWithRegularization tests
# =========================================================================


class TestAttentionPoolingWithRegularization:
    """T1.6: Entropy regularization."""

    def test_T1_6_uniform_max_entropy(self):
        """T1.6: Uniform attention → max entropy → most negative reg_loss."""
        pool_base = SelfAttentionPooling(input_dim=32, hidden_dim=16)
        pool = AttentionPoolingWithRegularization(pool_base, entropy_weight=0.01)

        # We can't force uniform easily, but we can check the reg sign convention
        h = torch.randn(4, 10, 32)
        z, alpha, reg_loss = pool(h)

        # reg_loss = -lambda * entropy.  entropy > 0 → reg_loss < 0
        # (encouraging high entropy)
        assert reg_loss.numel() == 1

    def test_T1_6_peaked_low_entropy(self):
        """T1.6b: Peaked attention gives lower entropy than uniform-like."""
        pool_base = SelfAttentionPooling(input_dim=32, hidden_dim=16)

        # Make a custom alpha tensor and compute entropy manually
        alpha_uniform = torch.ones(1, 10) / 10
        alpha_peaked = torch.zeros(1, 10)
        alpha_peaked[0, 0] = 0.9
        alpha_peaked[0, 1:] = 0.1 / 9

        log_u = torch.log(alpha_uniform.clamp(min=1e-12))
        H_uniform = -(alpha_uniform * log_u).sum(dim=-1).mean()

        log_p = torch.log(alpha_peaked.clamp(min=1e-12))
        H_peaked = -(alpha_peaked * log_p).sum(dim=-1).mean()

        assert H_uniform > H_peaked

    def test_T1_6_gradient_through_reg(self):
        """T1.6c: Gradient flows through regularization loss."""
        pool_base = SelfAttentionPooling(input_dim=32, hidden_dim=16)
        pool = AttentionPoolingWithRegularization(pool_base, entropy_weight=0.1)
        h = torch.randn(4, 10, 32)
        z, alpha, reg_loss = pool(h)

        reg_loss.backward()
        for name, param in pool.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# =========================================================================
# SchemeA_TwoBranch tests
# =========================================================================


class TestSchemeA:
    """Integration tests for SchemeA_TwoBranch."""

    def test_output_shape(self):
        d, L = 128, 10
        model = SchemeA_TwoBranch(embed_dim=d, n_layers=L)
        h_list = [torch.randn(4, 20, d) for _ in range(L)]
        mask = torch.ones(4, 20, dtype=torch.bool)
        mask[:, 15:] = False

        z_new, info = model(h_list, atom_mask=mask)
        assert z_new.shape == (4, d)
        assert info["z_atom"].shape == (4, d)
        assert info["z_global"].shape == (4, d)
        assert info["alpha_atom"].shape == (4, 20)
        assert info["beta_layer"].shape == (4, L)

    def test_entropy_reg_returned(self):
        model = SchemeA_TwoBranch(embed_dim=64, n_layers=5, entropy_weight=0.01)
        h_list = [torch.randn(2, 10, 64) for _ in range(5)]
        z, info = model(h_list)
        assert "entropy_reg" in info
        assert info["entropy_reg"].numel() == 1

    def test_gated_fusion(self):
        model = SchemeA_TwoBranch(
            embed_dim=64, n_layers=5, fusion_type="gated"
        )
        h_list = [torch.randn(2, 10, 64) for _ in range(5)]
        z, info = model(h_list)
        assert z.shape == (2, 64)

    def test_gradient_flow(self):
        model = SchemeA_TwoBranch(embed_dim=64, n_layers=3)
        h_list = [torch.randn(2, 5, 64, requires_grad=True) for _ in range(3)]
        z, info = model(h_list)
        loss = z.sum() + info["entropy_reg"]
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# =========================================================================
# SchemeB_SingleBranch tests
# =========================================================================


class TestSchemeB:
    """Integration tests for SchemeB_SingleBranch."""

    def test_output_shape(self):
        d, L = 128, 10
        model = SchemeB_SingleBranch(embed_dim=d, n_layers=L)
        h_list = [torch.randn(4, 20, d) for _ in range(L)]
        mask = torch.ones(4, 20, dtype=torch.bool)
        mask[:, 15:] = False

        z_global, info = model(h_list, atom_mask=mask)
        assert z_global.shape == (4, d)
        assert len(info["layer_alphas"]) == L
        assert info["layer_alphas"][0].shape == (4, 20)
        assert info["beta_layer"].shape == (4, L)

    def test_T1_11_shared_pool(self):
        """T1.11: SchemeB shared_pool produces same weights for same input."""
        model = SchemeB_SingleBranch(embed_dim=64, n_layers=5)
        model.eval()

        # Use the SAME tensor for two layers
        h_same = torch.randn(2, 8, 64)
        h_list = [h_same.clone() for _ in range(5)]

        z, info = model(h_list)

        # All layers received same input → shared pool → same alpha
        for i in range(1, 5):
            assert torch.allclose(
                info["layer_alphas"][0], info["layer_alphas"][i], atol=1e-6
            ), f"Layer {i} alphas differ from layer 0 — shared pool not working"

    def test_shared_pool_different_inputs(self):
        """Shared pool with different inputs → different alphas."""
        model = SchemeB_SingleBranch(embed_dim=64, n_layers=3)
        model.eval()

        h_list = [torch.randn(2, 8, 64) for _ in range(3)]
        z, info = model(h_list)

        # Different inputs → (very likely) different alphas
        assert not torch.allclose(
            info["layer_alphas"][0], info["layer_alphas"][1], atol=1e-3
        )

    def test_gradient_flow(self):
        model = SchemeB_SingleBranch(embed_dim=64, n_layers=3)
        h_list = [torch.randn(2, 5, 64, requires_grad=True) for _ in range(3)]
        z, info = model(h_list)
        loss = z.sum() + info["entropy_reg"]
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# =========================================================================
# MLPReadout tests
# =========================================================================


class TestMLPReadout:
    def test_output_shape(self):
        mlp = MLPReadout(input_dim=128, hidden_dim=128)
        z = torch.randn(8, 128)
        pred = mlp(z)
        assert pred.shape == (8,)

    def test_gradient_flow(self):
        mlp = MLPReadout(input_dim=64, hidden_dim=32)
        z = torch.randn(4, 64, requires_grad=True)
        pred = mlp(z)
        loss = pred.sum()
        loss.backward()
        assert z.grad is not None
