"""
bayesdiff/attention_pool.py — Attention-Based Aggregation Modules
─────────────────────────────────────────────────────────────────
Sub-Plan 2: Intra-layer attention pooling and inter-layer attention fusion
for improved molecular representation.

Modules:
  - SelfAttentionPooling: atom-level self-attention pooling within one layer
  - CrossAttentionPooling: pocket-conditioned cross-attention pooling
  - AttentionPoolingWithRegularization: entropy reg wrapper
  - SchemeA_TwoBranch: z_atom (last-layer AttnPool) + z_global (MeanPool→AttnFusion)
  - SchemeB_SingleBranch: per-layer shared AttnPool → AttnFusion

See doc/Stage_2/02_attention_aggregation.md
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bayesdiff.layer_fusion import LayerAttentionFusion


# ---------------------------------------------------------------------------
# Intra-layer pooling
# ---------------------------------------------------------------------------

class SelfAttentionPooling(nn.Module):
    """Intra-layer self-attention pooling over atoms.

    Computes:  s_i = w^T tanh(W h_i + b),  alpha = softmax(s),  z = sum(alpha_i h_i)

    Parameters
    ----------
    input_dim : int
        Atom embedding dimension (e.g. 128).
    hidden_dim : int
        Hidden dimension of the attention MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, h: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        h : Tensor, shape (B, N, d)
            Atom embeddings.
        mask : Tensor, shape (B, N), optional
            True = real atom, False = padding.

        Returns
        -------
        z : Tensor, shape (B, d)
            Attention-pooled embedding.
        alpha : Tensor, shape (B, N)
            Attention weights.
        """
        scores = self.attn(h).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)  # (B, N)
        # Zero out any NaN from all-masked rows (shouldn't happen in practice)
        alpha = alpha.masked_fill(alpha.isnan(), 0.0)
        z = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, d)
        return z, alpha


class CrossAttentionPooling(nn.Module):
    """Pocket-conditioned cross-attention pooling.

    Uses mean pocket embedding as query, ligand atom embeddings as keys/values.

    Parameters
    ----------
    ligand_dim : int
        Ligand atom embedding dimension.
    pocket_dim : int
        Pocket embedding dimension (may equal ligand_dim).
    hidden_dim : int
        Projected key/query dimension.
    """

    def __init__(
        self, ligand_dim: int, pocket_dim: int, hidden_dim: int = 128
    ):
        super().__init__()
        self.W_q = nn.Linear(pocket_dim, hidden_dim)
        self.W_k = nn.Linear(ligand_dim, hidden_dim)
        self.W_v = nn.Linear(ligand_dim, ligand_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        h_ligand: Tensor,
        h_pocket: Tensor,
        ligand_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        h_ligand : Tensor, shape (B, N_L, d_L)
        h_pocket : Tensor, shape (B, d_P)
            Pre-pooled pocket embedding (mean or summary).
        ligand_mask : Tensor, shape (B, N_L), optional

        Returns
        -------
        z : Tensor, shape (B, d_L)
        alpha : Tensor, shape (B, N_L)
        """
        q = self.W_q(h_pocket)  # (B, d_k)
        k = self.W_k(h_ligand)  # (B, N_L, d_k)
        v = self.W_v(h_ligand)  # (B, N_L, d_L)

        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / self.scale  # (B, N_L)
        if ligand_mask is not None:
            scores = scores.masked_fill(~ligand_mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)  # (B, N_L)
        alpha = alpha.masked_fill(alpha.isnan(), 0.0)
        z = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)  # (B, d_L)
        return z, alpha


# ---------------------------------------------------------------------------
# Regularization wrapper
# ---------------------------------------------------------------------------

class AttentionPoolingWithRegularization(nn.Module):
    """Wraps a pooling module and computes entropy regularization loss.

    Entropy reg:  L_ent = -lambda * sum(alpha_i log alpha_i)
    Higher lambda → encourages more uniform attention (prevents collapse).

    Parameters
    ----------
    pooling_module : nn.Module
        Must return (z, alpha, ...) where alpha is (B, N) attention weights.
    entropy_weight : float
        Coefficient for entropy regularization.
    """

    def __init__(self, pooling_module: nn.Module, entropy_weight: float = 0.01):
        super().__init__()
        self.pool = pooling_module
        self.entropy_weight = entropy_weight

    def forward(self, *args, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (z, alpha, reg_loss)."""
        z, alpha = self.pool(*args, **kwargs)
        # Entropy: H = -sum(alpha * log(alpha)), reg = -lambda * H (minimize → maximize entropy)
        # Clamp for numerical stability
        log_alpha = torch.log(alpha.clamp(min=1e-12))
        entropy = -(alpha * log_alpha).sum(dim=-1).mean()  # scalar, mean over batch
        reg_loss = -self.entropy_weight * entropy
        return z, alpha, reg_loss


# ---------------------------------------------------------------------------
# Scheme A: Two-Branch
# ---------------------------------------------------------------------------

class SchemeA_TwoBranch(nn.Module):
    """Two-branch architecture: z_atom (last-layer AttnPool) + z_global (MeanPool→AttnFusion).

    Branch 1: Self-attention pooling on last encoder layer → z_atom
    Branch 2: Mean-pool each layer → LayerAttentionFusion → z_global
    Fusion: concat + MLP or gated fusion

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (128 for TargetDiff).
    n_layers : int
        Number of encoder layers (10 for TargetDiff).
    fusion_type : str
        'concat_mlp' or 'gated'.
    attn_hidden_dim : int
        Hidden dim for attention MLP.
    entropy_weight : float
        Entropy regularization on atom attention.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 10,
        fusion_type: str = "concat_mlp",
        attn_hidden_dim: int = 64,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.fusion_type = fusion_type
        self.entropy_weight = entropy_weight

        # Branch 1: atom-level attention pooling on last layer
        self.atom_pool = SelfAttentionPooling(embed_dim, attn_hidden_dim)

        # Branch 2: layer-level attention fusion (reuse from layer_fusion.py)
        self.layer_fusion = LayerAttentionFusion(embed_dim, hidden_dim=attn_hidden_dim)

        # Fusion head
        if fusion_type == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif fusion_type == "gated":
            self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        all_layer_atom_embs: list[Tensor],
        atom_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """
        Parameters
        ----------
        all_layer_atom_embs : list of L Tensors, each (B, N, d)
            Per-layer atom-level embeddings.
        atom_mask : Tensor, shape (B, N), optional
            True = real atom, False = padding.

        Returns
        -------
        z_new : Tensor, shape (B, d)
        info : dict with z_atom, z_global, alpha_atom, beta_layer, entropy_reg
        """
        # Branch 1: last-layer AttnPool → z_atom
        z_atom, alpha_atom = self.atom_pool(all_layer_atom_embs[-1], mask=atom_mask)

        # Branch 2: per-layer MeanPool → AttnFusion → z_global
        layer_means = []
        for l_emb in all_layer_atom_embs:
            if atom_mask is not None:
                mask_f = atom_mask.unsqueeze(-1).float()  # (B, N, 1)
                z_l = (l_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                z_l = l_emb.mean(dim=1)
            layer_means.append(z_l)
        z_global, beta_layer = self.layer_fusion(layer_means)

        # Fusion
        if self.fusion_type == "concat_mlp":
            z_new = self.fusion(torch.cat([z_atom, z_global], dim=-1))
        else:  # gated
            g = torch.sigmoid(self.gate_proj(torch.cat([z_atom, z_global], dim=-1)))
            z_new = g * z_atom + (1 - g) * z_global

        # Entropy regularization on atom attention
        log_alpha = torch.log(alpha_atom.clamp(min=1e-12))
        entropy = -(alpha_atom * log_alpha).sum(dim=-1).mean()
        entropy_reg = -self.entropy_weight * entropy

        return z_new, {
            "z_atom": z_atom,
            "z_global": z_global,
            "alpha_atom": alpha_atom,
            "beta_layer": beta_layer,
            "entropy_reg": entropy_reg,
        }


# ---------------------------------------------------------------------------
# Scheme B: Single-Branch (shared AttnPool)
# ---------------------------------------------------------------------------

class SchemeB_SingleBranch(nn.Module):
    """Single-branch: per-layer shared AttnPool → LayerAttentionFusion.

    All layers share the same SelfAttentionPooling parameters.
    Stored as `self.shared_pool` (not ModuleList with repeated refs).

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    n_layers : int
        Number of encoder layers.
    attn_hidden_dim : int
        Hidden dim for attention MLP.
    entropy_weight : float
        Entropy regularization on atom attention.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 10,
        attn_hidden_dim: int = 64,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.entropy_weight = entropy_weight

        self.shared_pool = SelfAttentionPooling(embed_dim, attn_hidden_dim)
        self.layer_fusion = LayerAttentionFusion(embed_dim, hidden_dim=attn_hidden_dim)

    def forward(
        self,
        all_layer_atom_embs: list[Tensor],
        atom_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """
        Parameters
        ----------
        all_layer_atom_embs : list of L Tensors, each (B, N, d)
        atom_mask : Tensor, shape (B, N), optional

        Returns
        -------
        z_global : Tensor, shape (B, d)
        info : dict with layer_alphas, beta_layer, entropy_reg
        """
        layer_vecs = []
        layer_alphas = []
        total_entropy = 0.0

        for l_emb in all_layer_atom_embs:
            z_l, alpha_l = self.shared_pool(l_emb, mask=atom_mask)
            layer_vecs.append(z_l)
            layer_alphas.append(alpha_l)
            # Accumulate entropy
            log_alpha = torch.log(alpha_l.clamp(min=1e-12))
            total_entropy += -(alpha_l * log_alpha).sum(dim=-1).mean()

        z_global, beta_layer = self.layer_fusion(layer_vecs)

        # Average entropy across layers
        avg_entropy = total_entropy / len(all_layer_atom_embs)
        entropy_reg = -self.entropy_weight * avg_entropy

        return z_global, {
            "layer_alphas": layer_alphas,
            "beta_layer": beta_layer,
            "entropy_reg": entropy_reg,
        }


# ---------------------------------------------------------------------------
# MLP Readout Head (for Step 1 representation validation)
# ---------------------------------------------------------------------------

class MLPReadout(nn.Module):
    """Lightweight MLP readout for pKd prediction.

    Used in Step 1 of training strategy to validate representation quality
    before connecting to GP/DKL.

    Parameters
    ----------
    input_dim : int
        Input embedding dimension.
    hidden_dim : int
        Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor, shape (B, d)

        Returns
        -------
        pred : Tensor, shape (B,)
        """
        return self.mlp(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Multi-Head Attention Pooling (A3.5)
# ---------------------------------------------------------------------------

class MultiHeadAttentionPooling(nn.Module):
    """Multi-head atom-level attention pooling.

    Each head learns an independent attention pattern over atoms,
    then outputs are concatenated and projected back to embed_dim.

    Parameters
    ----------
    input_dim : int
        Atom embedding dimension.
    n_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dim for each head's attention MLP.
    """

    def __init__(self, input_dim: int, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        # Each head: Linear(input_dim, hidden_dim) → Tanh → Linear(hidden_dim, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(input_dim * n_heads, input_dim)

    def forward(
        self, h: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Parameters
        ----------
        h : Tensor, shape (B, N, d)
        mask : Tensor, shape (B, N), optional

        Returns
        -------
        z : Tensor, shape (B, d)
            Multi-head attention-pooled embedding.
        alphas : list of H Tensors, each (B, N)
            Per-head attention weights.
        """
        head_outputs = []
        alphas = []
        for head in self.heads:
            scores = head(h).squeeze(-1)  # (B, N)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            alpha = F.softmax(scores, dim=-1)
            alpha = alpha.masked_fill(alpha.isnan(), 0.0)
            z_h = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, d)
            head_outputs.append(z_h)
            alphas.append(alpha)
        z = self.out_proj(torch.cat(head_outputs, dim=-1))  # (B, d)
        return z, alphas


# ---------------------------------------------------------------------------
# Scheme B with Multi-Head (A3.5)
# ---------------------------------------------------------------------------

class SchemeB_MultiHead(nn.Module):
    """Scheme B with multi-head attention pooling (shared across layers).

    Parameters
    ----------
    embed_dim : int
    n_layers : int
    n_heads : int
    attn_hidden_dim : int
    entropy_weight : float
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 10,
        n_heads: int = 4,
        attn_hidden_dim: int = 64,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.entropy_weight = entropy_weight

        self.shared_pool = MultiHeadAttentionPooling(
            embed_dim, n_heads=n_heads, hidden_dim=attn_hidden_dim
        )
        self.layer_fusion = LayerAttentionFusion(embed_dim, hidden_dim=attn_hidden_dim)

    def forward(
        self,
        all_layer_atom_embs: list[Tensor],
        atom_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        layer_vecs = []
        all_head_alphas = []  # list of (n_layers, n_heads) attention weights
        total_entropy = 0.0

        for l_emb in all_layer_atom_embs:
            z_l, alphas = self.shared_pool(l_emb, mask=atom_mask)
            layer_vecs.append(z_l)
            all_head_alphas.append(alphas)
            # Entropy: average over heads
            for alpha_h in alphas:
                log_a = torch.log(alpha_h.clamp(min=1e-12))
                total_entropy += -(alpha_h * log_a).sum(dim=-1).mean()

        z_global, beta_layer = self.layer_fusion(layer_vecs)

        n_total = len(all_layer_atom_embs) * self.n_heads
        avg_entropy = total_entropy / n_total
        entropy_reg = -self.entropy_weight * avg_entropy

        return z_global, {
            "head_alphas": all_head_alphas,  # [L][H] each (B, N)
            "beta_layer": beta_layer,
            "entropy_reg": entropy_reg,
        }


# ---------------------------------------------------------------------------
# Scheme B with Independent per-layer AttnPool (A3.6)
# ---------------------------------------------------------------------------

class SchemeB_Independent(nn.Module):
    """Scheme B with independent AttnPool parameters per layer.

    Unlike SchemeB_SingleBranch which shares one AttnPool across all layers,
    this variant gives each layer its own set of attention parameters.

    Parameters
    ----------
    embed_dim : int
    n_layers : int
    attn_hidden_dim : int
    entropy_weight : float
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 10,
        attn_hidden_dim: int = 64,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.entropy_weight = entropy_weight

        self.layer_pools = nn.ModuleList([
            SelfAttentionPooling(embed_dim, attn_hidden_dim)
            for _ in range(n_layers)
        ])
        self.layer_fusion = LayerAttentionFusion(embed_dim, hidden_dim=attn_hidden_dim)

    def forward(
        self,
        all_layer_atom_embs: list[Tensor],
        atom_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        layer_vecs = []
        layer_alphas = []
        total_entropy = 0.0

        for l_idx, l_emb in enumerate(all_layer_atom_embs):
            z_l, alpha_l = self.layer_pools[l_idx](l_emb, mask=atom_mask)
            layer_vecs.append(z_l)
            layer_alphas.append(alpha_l)
            log_alpha = torch.log(alpha_l.clamp(min=1e-12))
            total_entropy += -(alpha_l * log_alpha).sum(dim=-1).mean()

        z_global, beta_layer = self.layer_fusion(layer_vecs)

        avg_entropy = total_entropy / len(all_layer_atom_embs)
        entropy_reg = -self.entropy_weight * avg_entropy

        return z_global, {
            "layer_alphas": layer_alphas,
            "beta_layer": beta_layer,
            "entropy_reg": entropy_reg,
        }
