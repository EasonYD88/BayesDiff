"""
bayesdiff/layer_fusion.py — Multi-Layer Fusion Modules
──────────────────────────────────────────────────────
Fuse per-layer encoder embeddings into a single representation
for downstream GP oracle prediction.

Stage 2: WeightedSumFusion — learnable softmax weights per layer
Stage 3: LayerAttentionFusion — input-dependent attention weights
Stage 4: ConcatMLPFusion — nonlinear projection of concatenated layers
Stage 5: ConcatDropoutFusion — concat+MLP with MC dropout
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSumFusion(nn.Module):
    """Learned weighted sum of per-layer embeddings.

    β_l = exp(w_l) / Σ exp(w_k),  z_fuse = Σ β_l z^(l)

    Parameters
    ----------
    n_layers : int
        Number of encoder layers to fuse.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(
        self, layer_embeddings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-layer embeddings via learned softmax weights.

        Parameters
        ----------
        layer_embeddings : list of Tensor
            Each tensor has shape (B, d).

        Returns
        -------
        z_fuse : Tensor, shape (B, d)
            Weighted-sum fused embedding.
        weights : Tensor, shape (n_layers,)
            Softmax layer weights (for interpretability).
        """
        weights = F.softmax(self.logits, dim=0)
        z_fuse = torch.zeros_like(layer_embeddings[0])
        for w, z in zip(weights, layer_embeddings):
            z_fuse = z_fuse + w * z
        return z_fuse, weights


class LayerAttentionFusion(nn.Module):
    """Input-dependent layer attention fusion.

    For each sample, computes attention scores over layers using a
    small MLP: score_l = u^T tanh(W z^(l)), then softmax over layers.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of per-layer embeddings.
    hidden_dim : int
        Hidden dimension of the attention MLP.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(embed_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, layer_embeddings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse per-layer embeddings via input-dependent attention.

        Parameters
        ----------
        layer_embeddings : list of Tensor
            Each tensor has shape (B, d).

        Returns
        -------
        z_fuse : Tensor, shape (B, d)
            Attention-weighted fused embedding.
        weights : Tensor, shape (B, n_layers)
            Per-sample layer attention weights.
        """
        # scores: list of (B, 1)
        scores = [self.u(torch.tanh(self.W(z))) for z in layer_embeddings]
        scores = torch.cat(scores, dim=-1)  # (B, n_layers)
        weights = F.softmax(scores, dim=-1)  # (B, n_layers)
        z_fuse = torch.zeros_like(layer_embeddings[0])
        for i, z in enumerate(layer_embeddings):
            z_fuse = z_fuse + weights[:, i].unsqueeze(-1) * z
        return z_fuse, weights


class ConcatMLPFusion(nn.Module):
    """Concatenation + bottleneck MLP fusion.

    Concatenates all layer embeddings and projects through a two-layer
    MLP with LayerNorm and ReLU, producing a fixed-size output.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each per-layer embedding.
    n_layers : int
        Number of layers to fuse.
    output_dim : int
        Output embedding dimension.
    """

    def __init__(self, embed_dim: int, n_layers: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * n_layers, 2 * output_dim),
            nn.LayerNorm(2 * output_dim),
            nn.ReLU(),
            nn.Linear(2 * output_dim, output_dim),
        )

    def forward(
        self, layer_embeddings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, None]:
        """Fuse per-layer embeddings via concatenation + MLP.

        Parameters
        ----------
        layer_embeddings : list of Tensor
            Each tensor has shape (B, d).

        Returns
        -------
        z_fuse : Tensor, shape (B, output_dim)
            MLP-projected fused embedding.
        weights : None
            No interpretable weights for this method.
        """
        z_concat = torch.cat(layer_embeddings, dim=-1)
        return self.mlp(z_concat), None
