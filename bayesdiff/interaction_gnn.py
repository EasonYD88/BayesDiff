"""
bayesdiff/interaction_gnn.py — Lightweight GNN for Pocket-Ligand Interaction Encoding
────────────────────────────────────────────────────────────────────────────────────────
Two-layer message-passing GNN on the bipartite pocket-ligand contact graph.
Produces z_interaction ∈ ℝ^{output_dim} per complex.

See doc/Stage_2/01_multi_granularity_repr.md §2.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from bayesdiff.interaction_graph import NUM_ELEMENTS, NUM_AA_TYPES


class BipartiteMessagePassing(nn.Module):
    """Single message-passing layer for bipartite (or general) graphs.

    Message:  m_{ij} = MLP_msg([h_i; h_j; e_{ij}])
    Update:   h_i' = h_i + MLP_upd(Σ_j m_{ij})
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, node_dim) node features
            edge_index: (2, E) edges
            edge_attr:  (E, edge_dim) edge features

        Returns:
            x':         (N, node_dim) updated node features
        """
        src, dst = edge_index  # src → dst

        # Compute messages
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate messages at destination nodes
        agg = torch.zeros(x.shape[0], messages.shape[1], device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update with residual
        x = self.norm(x + self.upd_mlp(agg))
        return x


class InteractionGNN(nn.Module):
    """Lightweight GNN for encoding pocket-ligand interaction graphs.

    Embeds atoms via learned element/residue embeddings, runs K message-passing
    layers, then pools over edges to produce z_interaction.

    Architecture:
        1. Node embedding: element_embed + (pocket only) aa_embed → project to node_dim
        2. K BipartiteMessagePassing layers
        3. Readout: 'edge' (mean over edge MLP), 'node' (mean over ligand nodes),
           or 'both' (concat of edge + node readouts)
    """

    def __init__(
        self,
        edge_dim: int = 51,
        hidden_dim: int = 128,
        n_layers: int = 2,
        output_dim: int = 128,
        n_elements: int = NUM_ELEMENTS,
        n_aa_types: int = NUM_AA_TYPES,
        element_embed_dim: int = 32,
        readout_mode: str = "node",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_mode = readout_mode

        # ── Node embeddings ──────────────────────────────────────────────
        self.element_embed = nn.Embedding(n_elements, element_embed_dim)
        self.aa_embed = nn.Embedding(n_aa_types, element_embed_dim)

        # Projection to common node dim
        # Ligand: element_embed(32) → node_dim
        self.lig_proj = nn.Sequential(
            nn.Linear(element_embed_dim, hidden_dim),
            nn.ReLU(),
        )
        # Pocket: element_embed(32) + aa_embed(32) = 64 → node_dim
        self.pkt_proj = nn.Sequential(
            nn.Linear(2 * element_embed_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── Message-passing layers ───────────────────────────────────────
        self.layers = nn.ModuleList([
            BipartiteMessagePassing(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # ── Readout heads ────────────────────────────────────────────────
        self.edge_readout = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.node_readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

        # Set output_dim based on readout mode
        if readout_mode == "both":
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data/Batch from InteractionGraphBuilder with fields:
                  edge_index, edge_attr, element_idx, aa_type_idx, node_type, batch

        Returns:
            z_interaction: (batch_size, output_dim) interaction-level embedding
        """
        # ── Embed nodes ──────────────────────────────────────────────────
        elem_emb = self.element_embed(data.element_idx)  # (N, 32)
        aa_emb = self.aa_embed(data.aa_type_idx)          # (N, 32)

        lig_mask = data.node_type == 0
        pkt_mask = data.node_type == 1

        x = torch.zeros(data.num_nodes, self.hidden_dim, device=elem_emb.device)
        if lig_mask.any():
            x[lig_mask] = self.lig_proj(elem_emb[lig_mask])
        if pkt_mask.any():
            pkt_input = torch.cat([elem_emb[pkt_mask], aa_emb[pkt_mask]], dim=-1)
            x[pkt_mask] = self.pkt_proj(pkt_input)

        # ── Message passing ──────────────────────────────────────────────
        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)

        # ── Readout ──────────────────────────────────────────────────────
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
        batch_size = batch.max().item() + 1
        n_edges = data.edge_index.shape[1]

        z_node = None
        z_edge = None

        # Node-level readout: mean over ligand nodes (which received pocket messages)
        if self.readout_mode in ("node", "both") or n_edges == 0:
            z_node = scatter(
                self.node_readout(x[lig_mask]),
                batch[lig_mask],
                dim=0,
                dim_size=batch_size,
                reduce='mean',
            )

        # Edge-level readout: mean over MLP([h_src; h_dst; e_ij])
        if self.readout_mode in ("edge", "both") and n_edges > 0:
            src, dst = data.edge_index
            edge_repr = torch.cat([x[src], x[dst], data.edge_attr], dim=-1)
            edge_out = self.edge_readout(edge_repr)  # (E, output_dim)
            edge_batch = batch[src]
            z_edge = scatter(edge_out, edge_batch, dim=0, dim_size=batch_size, reduce='mean')

        # Combine based on mode
        if self.readout_mode == "both" and z_edge is not None:
            z = torch.cat([z_node, z_edge], dim=-1)
        elif self.readout_mode == "edge" and z_edge is not None:
            z = z_edge
        else:
            z = z_node

        return z  # (batch_size, output_dim)


class InteractionGNNPredictor(nn.Module):
    """InteractionGNN + linear head for supervised pre-training on pKd.

    Used in Phase 1 to pre-train the GNN features before GP training.
    """

    def __init__(self, gnn: InteractionGNN):
        super().__init__()
        self.gnn = gnn
        self.head = nn.Sequential(
            nn.Linear(gnn.output_dim, gnn.output_dim // 2),
            nn.ReLU(),
            nn.Linear(gnn.output_dim // 2, 1),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (prediction, z_interaction)."""
        z = self.gnn(data)
        pred = self.head(z).squeeze(-1)
        return pred, z
