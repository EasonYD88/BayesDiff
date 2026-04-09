"""
Tests for frozen_embedder.py — SP2→Predictor Head integration smoke tests.

These tests validate the interface contract between the frozen SP2
representation (Sub-Plan 2) and downstream predictor heads (Sub-Plan 4).

Tests use synthetic data (no checkpoints needed) unless marked with
@pytest.mark.checkpoint which requires real saved weights.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesdiff.attention_pool import (
    MLPReadout,
    SchemeB_Independent,
    SchemeB_SingleBranch,
)
from bayesdiff.frozen_embedder import FrozenSP2Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_batch(B=4, N=20, d=128, L=10):
    """Create a synthetic batch of layer embeddings + mask."""
    layer_embs = [torch.randn(B, N, d) for _ in range(L)]
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -3:] = False  # last 3 atoms are padding
    pkd = torch.randn(B)
    return layer_embs, mask, pkd


def _make_frozen_embedder(variant="independent", d=128, L=10):
    """Create a FrozenSP2Embedder from a randomly-initialised SchemeB."""
    if variant == "independent":
        model = SchemeB_Independent(
            embed_dim=d, n_layers=L, attn_hidden_dim=64, entropy_weight=0.01
        )
    else:
        model = SchemeB_SingleBranch(
            embed_dim=d, n_layers=L, attn_hidden_dim=64, entropy_weight=0.01
        )
    return FrozenSP2Embedder(model)


# =========================================================================
# T_F1: FrozenSP2Embedder basic contract
# =========================================================================


class TestFrozenContract:
    """Verify that FrozenSP2Embedder actually freezes parameters and
    produces correct output shapes."""

    def test_all_params_frozen(self):
        """Every parameter in the embedder must have requires_grad=False."""
        emb = _make_frozen_embedder()
        for name, p in emb.named_parameters():
            assert not p.requires_grad, f"{name} still requires grad"

    def test_output_shape(self):
        """forward() returns z of shape (B, d)."""
        emb = _make_frozen_embedder()
        layers, mask, _ = _make_synthetic_batch()
        z = emb(layers, atom_mask=mask)
        assert z.shape == (4, 128)

    def test_no_grad_output(self):
        """Output tensor should not require grad."""
        emb = _make_frozen_embedder()
        layers, mask, _ = _make_synthetic_batch()
        z = emb(layers, atom_mask=mask)
        assert not z.requires_grad

    def test_eval_mode_sticky(self):
        """Calling .train(True) should NOT switch embedder to train mode."""
        emb = _make_frozen_embedder()
        emb.train(True)
        assert not emb.training

    def test_embed_with_info_returns_dict(self):
        """embed_with_info() returns (z, info) with expected keys."""
        emb = _make_frozen_embedder()
        layers, mask, _ = _make_synthetic_batch()
        z, info = emb.embed_with_info(layers, atom_mask=mask)
        assert z.shape == (4, 128)
        assert "entropy_reg" in info

    def test_embed_dim_property(self):
        emb = _make_frozen_embedder(d=64, L=5)
        assert emb.embed_dim == 64

    @pytest.mark.parametrize("variant", ["independent", "shared"])
    def test_both_variants_frozen(self, variant):
        """Both Independent and SingleBranch variants freeze correctly."""
        emb = _make_frozen_embedder(variant=variant)
        n_frozen = sum(1 for p in emb.parameters() if not p.requires_grad)
        n_total = sum(1 for p in emb.parameters())
        assert n_frozen == n_total > 0


# =========================================================================
# T_F2: SchemeB → MLP → loss.backward() — gradient flows through head only
# =========================================================================


class TestSchemeBToMLP:
    """Verify that with a frozen embedder + trainable MLP head,
    gradients flow through the head but NOT the frozen encoder."""

    def test_mlp_backward_flow(self):
        """loss.backward() updates MLP grads but NOT embedder params."""
        emb = _make_frozen_embedder()
        mlp = MLPReadout(input_dim=128, hidden_dim=128)
        layers, mask, pkd = _make_synthetic_batch()

        # Forward
        z = emb(layers, atom_mask=mask)
        # z is detached (no_grad), so we need to re-enable grad for the head
        z_for_head = z.detach().requires_grad_(True)
        pred = mlp(z_for_head)
        loss = F.mse_loss(pred, pkd)

        # Backward
        loss.backward()

        # MLP should have gradients
        mlp_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mlp.parameters()
        )
        assert mlp_has_grad, "MLP should have non-zero gradients"

        # Embedder should NOT have gradients
        for name, p in emb.named_parameters():
            assert p.grad is None, f"Frozen param {name} should have no grad"

    def test_mlp_step_changes_weights(self):
        """An optimizer step actually changes MLP weights."""
        emb = _make_frozen_embedder()
        mlp = MLPReadout(input_dim=128, hidden_dim=128)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)

        layers, mask, pkd = _make_synthetic_batch()
        w_before = mlp.mlp[0].weight.data.clone()

        z = emb(layers, atom_mask=mask).detach().requires_grad_(True)
        pred = mlp(z)
        loss = F.mse_loss(pred, pkd)
        opt.zero_grad(); loss.backward(); opt.step()

        w_after = mlp.mlp[0].weight.data
        assert not torch.allclose(w_before, w_after), "MLP weights should change after step"

    def test_embedder_unchanged_after_step(self):
        """Optimizer step must NOT change embedder weights even if wrongly
        included in the optimizer."""
        emb = _make_frozen_embedder()
        mlp = MLPReadout(input_dim=128, hidden_dim=128)
        # Intentionally include embedder params (shouldn't matter — grad is None)
        opt = torch.optim.Adam(
            list(emb.parameters()) + list(mlp.parameters()), lr=1e-2
        )

        layers, mask, pkd = _make_synthetic_batch()
        enc_snap = {n: p.data.clone() for n, p in emb.named_parameters()}

        z = emb(layers, atom_mask=mask).detach().requires_grad_(True)
        pred = mlp(z)
        loss = F.mse_loss(pred, pkd)
        opt.zero_grad(); loss.backward(); opt.step()

        for name, p in emb.named_parameters():
            assert torch.equal(enc_snap[name], p.data), (
                f"Frozen param {name} changed after optimizer step"
            )


# =========================================================================
# T_F3: FrozenSP2Embedder → GPOracle — GP pipeline roundtrip
# =========================================================================


class TestEmbedderToGP:
    """Verify that frozen embeddings can feed into GPOracle for
    train → predict roundtrip."""

    def test_gp_roundtrip(self):
        """Embed synthetic data → GPOracle.train → GPOracle.predict
        should return (mu, var) with correct shapes."""
        # Import GPOracle here — it requires gpytorch
        from bayesdiff.gp_oracle import GPOracle

        emb = _make_frozen_embedder()
        layers, mask, pkd = _make_synthetic_batch(B=32)  # need enough for inducing

        z = emb(layers, atom_mask=mask)
        assert z.shape == (32, 128)

        gp = GPOracle(d=128, n_inducing=16)   # small for speed
        history = gp.train(z.numpy(), pkd.numpy(), n_epochs=5, lr=0.05, verbose=False)
        assert len(history["loss"]) == 5

        # Predict
        z_test = emb(layers, atom_mask=mask)
        mu, var = gp.predict(z_test.numpy())
        assert mu.shape == (32,)
        assert var.shape == (32,)
        assert (var >= 0).all(), "GP variance must be non-negative"


# =========================================================================
# T_F4: Jacobian / autograd through unfrozen head
# =========================================================================


class TestJacobianAutograd:
    """Ensure autograd graph works correctly:
    z (frozen, detached) → head (unfrozen) → ∂output/∂z should be computable.
    This is critical for future gradient-based methods on the head."""

    def test_jacobian_through_mlp(self):
        """torch.autograd.grad(output, z) should produce a gradient
        of shape (B, d) when the head is a simple MLP."""
        emb = _make_frozen_embedder()
        mlp = MLPReadout(input_dim=128, hidden_dim=128)
        layers, mask, _ = _make_synthetic_batch(B=2)

        z = emb(layers, atom_mask=mask).detach().requires_grad_(True)
        pred = mlp(z)  # (B, 1)

        # Compute ∂pred/∂z for each sample
        grad_z = torch.autograd.grad(
            pred.sum(), z, create_graph=False
        )[0]
        assert grad_z.shape == (2, 128)
        assert grad_z.abs().sum() > 0, "Jacobian should be non-zero"

    def test_second_order_through_head(self):
        """Second-order gradient (Hessian-vector product) should be
        computable through the head — needed for Laplace approximation."""
        mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        z = torch.randn(2, 128, requires_grad=True)

        pred = mlp(z).sum()
        grad = torch.autograd.grad(pred, z, create_graph=True)[0]
        # Hessian-vector product: differentiate grad w.r.t. z
        hvp = torch.autograd.grad(grad.sum(), z)[0]
        assert hvp.shape == (2, 128)


# =========================================================================
# T_F5: Determinism / reproducibility
# =========================================================================


class TestDeterminism:
    """FrozenSP2Embedder should produce identical outputs across calls."""

    def test_same_input_same_output(self):
        torch.manual_seed(0)
        emb = _make_frozen_embedder()
        layers, mask, _ = _make_synthetic_batch(B=3)

        z1 = emb(layers, atom_mask=mask)
        z2 = emb(layers, atom_mask=mask)
        assert torch.equal(z1, z2), "Same input must give identical output"

    def test_deterministic_across_eval_calls(self):
        """Even after .train(True), output must be the same (eval-locked)."""
        emb = _make_frozen_embedder()
        layers, mask, _ = _make_synthetic_batch(B=3)

        z1 = emb(layers, atom_mask=mask)
        emb.train(True)  # Should be a no-op
        z2 = emb(layers, atom_mask=mask)
        assert torch.equal(z1, z2)
