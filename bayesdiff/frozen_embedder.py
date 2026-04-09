"""
bayesdiff/frozen_embedder.py — Frozen SP2 Entry Points for Predictor Heads
──────────────────────────────────────────────────────────────────────────
Provides two frozen representation entry points for Sub-Plan 4
(Hybrid Predictor Head) and beyond.

Entry points
────────────
1. **Main entry** (A3.6 SchemeB Independent):
   - Best point-prediction representation: Test R²=0.574, ρ=0.778
   - All AttnPool + LayerFusion params frozen
   - Output: z ∈ ℝ^{B × 128}

2. **GP-reference entry** (A3.4c DKL-compatible):
   - SchemeB_SingleBranch (shared AttnPool) → DKLFeatureExtractor(128→32)
   - For GP/DKL heads as a recoverable-upper-bound reference
   - Output: z ∈ ℝ^{B × 32}   (DKL-reduced)
   - Or z ∈ ℝ^{B × 128} (pre-DKL) with optional .embed_for_dkl()

After freezing, downstream code compares *heads only* —
representation is held constant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from bayesdiff.attention_pool import SchemeB_Independent, SchemeB_SingleBranch

logger = logging.getLogger(__name__)

# Default checkpoint paths (relative to project root)
_DEFAULT_A36_CKPT = "results/stage2/ablation_viz/A36_independent_model.pt"
_DEFAULT_A34_CKPT = "results/stage2/phase3_refinement/A34_step1_model.pt"


class FrozenSP2Embedder(nn.Module):
    """Frozen SchemeB embedder — loads a trained SchemeB model and freezes all params.

    This is the **interface contract** between Sub-Plan 2 (representation)
    and Sub-Plan 4+ (predictor heads).  Downstream code calls `.embed()`
    and never touches the representation weights.

    Parameters
    ----------
    scheme_model : SchemeB_Independent | SchemeB_SingleBranch
        A trained SchemeB model (already loaded state_dict).
    """

    def __init__(self, scheme_model: nn.Module):
        super().__init__()
        self.encoder = scheme_model
        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    def forward(
        self,
        all_layer_atom_embs: list[Tensor],
        atom_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Produce frozen embedding z ∈ ℝ^{B × d}.

        Parameters
        ----------
        all_layer_atom_embs : list of L Tensors, each (B, N, d)
        atom_mask : Tensor (B, N), optional

        Returns
        -------
        z : Tensor (B, d) — detached, no grad
        """
        with torch.no_grad():
            z, _ = self.encoder(all_layer_atom_embs, atom_mask=atom_mask)
        return z

    def embed_with_info(
        self,
        all_layer_atom_embs: list[Tensor],
        atom_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Like forward(), but also returns attention info (for analysis)."""
        with torch.no_grad():
            z, info = self.encoder(all_layer_atom_embs, atom_mask=atom_mask)
        return z, info

    def train(self, mode: bool = True):
        # Always stay in eval mode — frozen
        return super().train(False)


# ---------------------------------------------------------------------------
# Factory: Main entry — A3.6 SchemeB Independent
# ---------------------------------------------------------------------------

def load_main_embedder(
    checkpoint_path: str | Path | None = None,
    embed_dim: int = 128,
    n_layers: int = 10,
    attn_hidden_dim: int = 64,
    device: str = "cpu",
) -> FrozenSP2Embedder:
    """Load the A3.6 SchemeB Independent model as a frozen embedder.

    This is the **primary representation** for Sub-Plan 4 predictor heads.
    Test R²=0.574, ρ=0.778.

    Parameters
    ----------
    checkpoint_path : path to saved state_dict (.pt)
        If None, uses default path.
    embed_dim : int
    n_layers : int
    attn_hidden_dim : int
    device : str

    Returns
    -------
    FrozenSP2Embedder with SchemeB_Independent inside.
    """
    if checkpoint_path is None:
        checkpoint_path = _DEFAULT_A36_CKPT

    model = SchemeB_Independent(
        embed_dim=embed_dim,
        n_layers=n_layers,
        attn_hidden_dim=attn_hidden_dim,
        entropy_weight=0.0,  # Irrelevant when frozen
    )

    ckpt = torch.load(str(checkpoint_path), weights_only=False, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    embedder = FrozenSP2Embedder(model).to(device)
    logger.info(
        f"Loaded main embedder (A3.6 Independent) from {checkpoint_path} "
        f"[{sum(p.numel() for p in model.parameters()):,} params, all frozen]"
    )
    return embedder


# ---------------------------------------------------------------------------
# Factory: GP-reference entry — A3.4c DKL-compatible
# ---------------------------------------------------------------------------

def load_gp_reference_embedder(
    checkpoint_path: str | Path | None = None,
    embed_dim: int = 128,
    n_layers: int = 10,
    attn_hidden_dim: int = 64,
    device: str = "cpu",
) -> FrozenSP2Embedder:
    """Load the A3.4 SchemeB_SingleBranch (shared) model as a frozen embedder.

    This is the **GP-reference representation** — the same checkpoint used
    by A3.4c DKL (ρ=0.760).  Downstream GP/DKL heads can be attached to
    its 128-d output without re-mixing representation changes.

    Parameters
    ----------
    checkpoint_path : path to saved state_dict (.pt)
    Returns FrozenSP2Embedder with SchemeB_SingleBranch inside.
    """
    if checkpoint_path is None:
        checkpoint_path = _DEFAULT_A34_CKPT

    model = SchemeB_SingleBranch(
        embed_dim=embed_dim,
        n_layers=n_layers,
        attn_hidden_dim=attn_hidden_dim,
        entropy_weight=0.0,
    )

    ckpt = torch.load(str(checkpoint_path), weights_only=False, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    embedder = FrozenSP2Embedder(model).to(device)
    logger.info(
        f"Loaded GP-reference embedder (A3.4 Shared) from {checkpoint_path} "
        f"[{sum(p.numel() for p in model.parameters()):,} params, all frozen]"
    )
    return embedder
