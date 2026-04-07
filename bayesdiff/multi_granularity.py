"""
bayesdiff/multi_granularity.py — Multi-Granularity Representation Encoder
─────────────────────────────────────────────────────────────────────────
Combines interaction-level (z_interaction from GNN) with existing
global-level embeddings (z_global from encoder).

MVP design: simple concatenation of z_global + z_interaction.
No learnable MLP fusion to avoid representation collapse during GP training
(lesson from ConcatMLP failure, see results/stage2/concat_mlp/).

See doc/Stage_2/01_multi_granularity_repr.md §2.4 and §8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from bayesdiff.interaction_gnn import InteractionGNN


class MultiGranularityEncoder(nn.Module):
    """Combines interaction-level and global-level representations.

    MVP: z_new = concat(z_global, z_interaction)
    Optional: z_new = MLP(concat(z_global, z_interaction))

    The interaction graph is built externally (in the DataLoader), so this
    module simply runs the GNN and concatenates with the provided z_global.
    """

    def __init__(
        self,
        interaction_gnn: InteractionGNN,
        z_global_dim: int = 128,
        fusion: str = "concat",
        output_dim: int = 128,
    ):
        """
        Args:
            interaction_gnn: Pre-built InteractionGNN module.
            z_global_dim:    Dimension of the global embedding (e.g., 128 from encoder).
            fusion:          'concat' (default, safe) or 'concat_mlp' (learnable, risky).
            output_dim:      Output dim when using 'concat_mlp' fusion.
        """
        super().__init__()
        self.gnn = interaction_gnn
        self.fusion_mode = fusion
        self.z_global_dim = z_global_dim

        combined_dim = z_global_dim + interaction_gnn.output_dim

        if fusion == "concat":
            self.output_dim = combined_dim
        elif fusion == "concat_mlp":
            self.output_dim = output_dim
            self.fusion_mlp = nn.Sequential(
                nn.Linear(combined_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")

    def forward(
        self,
        graph_data: Data,
        z_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            graph_data: PyG Data/Batch from InteractionGraphBuilder.
            z_global:   (B, z_global_dim) pre-computed global embeddings.
                        If None, returns z_interaction only.

        Returns:
            z_new: (B, output_dim) multi-granularity representation
        """
        z_interaction = self.gnn(graph_data)  # (B, gnn.output_dim)

        if z_global is None:
            return z_interaction

        z_combined = torch.cat([z_global, z_interaction], dim=-1)

        if self.fusion_mode == "concat":
            return z_combined
        elif self.fusion_mode == "concat_mlp":
            return self.fusion_mlp(z_combined)
        return z_combined
