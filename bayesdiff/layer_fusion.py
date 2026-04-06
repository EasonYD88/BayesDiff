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
