"""
bayesdiff/interaction_graph.py — Bipartite Pocket-Ligand Interaction Graph Builder
───────────────────────────────────────────────────────────────────────────────────
Constructs bipartite contact graphs between ligand heavy atoms and pocket heavy atoms
at atomic resolution (NOT Cα). See doc/Stage_2/01_multi_granularity_repr.md §2.2.

Edge features (MVP): distance RBF + element one-hots + AA type one-hot.
No chemistry-aware interaction type labels in this version.
"""

from __future__ import annotations

import torch
import numpy as np
from torch_geometric.data import Data


# ── Element vocabularies ────────────────────────────────────────────────
# Unified element mapping (atomic number → index)
ELEMENT_TO_IDX = {
    6: 0,    # C
    7: 1,    # N
    8: 2,    # O
    16: 3,   # S
    9: 4,    # F
    17: 5,   # Cl
    35: 6,   # Br
    15: 7,   # P
    53: 8,   # I
}
NUM_ELEMENTS = 10  # 9 known + 1 "other"

# Ligand element subset for edge features
LIGAND_ELEMENT_TO_IDX = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 5, 35: 6, 15: 7}
NUM_LIGAND_ELEMENTS = 9  # 8 known + 1 "other"

# Pocket element subset for edge features
POCKET_ELEMENT_TO_IDX = {6: 0, 7: 1, 8: 2, 16: 3}
NUM_POCKET_ELEMENTS = 5  # 4 known + 1 "other"

NUM_AA_TYPES = 21  # 20 standard amino acids + 1 unknown


class InteractionGraphBuilder:
    """Constructs pocket-ligand bipartite interaction graphs at heavy-atom resolution.

    Given ligand and pocket atom positions + types, builds a bipartite graph
    where edges connect ligand-pocket atom pairs within a distance cutoff.

    Edge features (per edge, dim = 16 + 9 + 5 + 21 = 51):
        - Gaussian RBF expansion of distance (16 centers, 0–8 Å)
        - Ligand atom element one-hot (9 types)
        - Pocket atom element one-hot (5 types)
        - Pocket residue type one-hot (21 types)
    """

    EDGE_DIM = NUM_LIGAND_ELEMENTS + NUM_POCKET_ELEMENTS + NUM_AA_TYPES  # one-hot parts

    def __init__(
        self,
        cutoff: float = 4.5,
        rbf_centers: int = 16,
        max_rbf_dist: float = 8.0,
    ):
        self.cutoff = cutoff
        self.rbf_centers = rbf_centers
        self.max_rbf_dist = max_rbf_dist

        # Pre-compute RBF center positions
        self.rbf_mu = torch.linspace(0.0, max_rbf_dist, rbf_centers)
        self.rbf_sigma = (max_rbf_dist / rbf_centers) * 0.5  # half the spacing

        self.edge_dim = rbf_centers + self.EDGE_DIM  # 16 + 35 = 51

    def build_graph(
        self,
        ligand_pos: torch.Tensor,
        ligand_element: torch.Tensor,
        pocket_pos: torch.Tensor,
        pocket_element: torch.Tensor,
        pocket_aa_type: torch.Tensor,
        pkd: float | torch.Tensor | None = None,
    ) -> Data:
        """Build a bipartite interaction graph.

        Args:
            ligand_pos:      (N_L, 3) ligand atom coordinates
            ligand_element:  (N_L,) atomic numbers for ligand atoms
            pocket_pos:      (N_P, 3) pocket heavy-atom coordinates
            pocket_element:  (N_P,) atomic numbers for pocket atoms
            pocket_aa_type:  (N_P,) amino acid type index (0–20) for each pocket atom
            pkd:             optional scalar pKd label

        Returns:
            PyG Data with:
                edge_index:     (2, E) bipartite edge indices (src=ligand, dst=pocket+offset)
                edge_attr:      (E, edge_dim) edge features
                node_type:      (N_L+N_P,) 0=ligand, 1=pocket
                element_idx:    (N_L+N_P,) unified element index
                aa_type_idx:    (N_L+N_P,) AA type index (0 for ligand atoms)
                num_lig_atoms:  int
                num_pkt_atoms:  int
                y:              optional pKd
        """
        n_lig = ligand_pos.shape[0]
        n_pkt = pocket_pos.shape[0]

        # ── Find contact edges within cutoff ─────────────────────────────
        # Compute pairwise distances: (N_L, N_P)
        dist = torch.cdist(ligand_pos.float(), pocket_pos.float())

        # Find edges: ligand atom i, pocket atom j with dist <= cutoff
        lig_idx, pkt_idx = torch.where(dist <= self.cutoff)

        # Edge indices into the concatenated node array [lig_0..lig_{N_L-1}, pkt_0..pkt_{N_P-1}]
        src = lig_idx                     # ligand node indices
        dst = pkt_idx + n_lig             # pocket node indices (offset)

        # Bidirectional edges for message passing
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )

        # ── Edge features ────────────────────────────────────────────────
        if len(lig_idx) > 0:
            edge_dist = dist[lig_idx, pkt_idx]

            # RBF expansion
            rbf = self._gaussian_rbf(edge_dist)  # (E_half, 16)

            # Element one-hots for edge features
            lig_elem_idx = self._map_elements(ligand_element[lig_idx], LIGAND_ELEMENT_TO_IDX)
            lig_elem_oh = torch.nn.functional.one_hot(
                lig_elem_idx, NUM_LIGAND_ELEMENTS
            ).float()  # (E_half, 9)

            pkt_elem_idx = self._map_elements(pocket_element[pkt_idx], POCKET_ELEMENT_TO_IDX)
            pkt_elem_oh = torch.nn.functional.one_hot(
                pkt_elem_idx, NUM_POCKET_ELEMENTS
            ).float()  # (E_half, 5)

            # AA type one-hot
            aa_idx = pocket_aa_type[pkt_idx].long().clamp(0, NUM_AA_TYPES - 1)
            aa_oh = torch.nn.functional.one_hot(
                aa_idx, NUM_AA_TYPES
            ).float()  # (E_half, 21)

            # Concatenate: RBF + lig_elem + pkt_elem + AA
            edge_feat_half = torch.cat([rbf, lig_elem_oh, pkt_elem_oh, aa_oh], dim=-1)

            # For reverse edges: swap ligand/pocket roles
            # Reverse: src=pocket, dst=ligand → lig features come from dst, pkt from src
            reverse_feat = torch.cat([rbf, lig_elem_oh, pkt_elem_oh, aa_oh], dim=-1)

            edge_attr = torch.cat([edge_feat_half, reverse_feat], dim=0)
        else:
            edge_attr = torch.zeros(0, self.edge_dim)

        # ── Node-level metadata ──────────────────────────────────────────
        node_type = torch.cat([
            torch.zeros(n_lig, dtype=torch.long),
            torch.ones(n_pkt, dtype=torch.long),
        ])

        element_idx = torch.cat([
            self._map_elements(ligand_element, ELEMENT_TO_IDX),
            self._map_elements(pocket_element, ELEMENT_TO_IDX),
        ])

        aa_type_idx = torch.cat([
            torch.zeros(n_lig, dtype=torch.long),  # ligand: no AA type
            pocket_aa_type.long().clamp(0, NUM_AA_TYPES - 1),
        ])

        data = Data(
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            node_type=node_type,
            element_idx=element_idx,
            aa_type_idx=aa_type_idx,
            num_lig_atoms=n_lig,
            num_pkt_atoms=n_pkt,
            num_nodes=n_lig + n_pkt,
        )

        if pkd is not None:
            data.y = torch.tensor([pkd], dtype=torch.float32) if not isinstance(pkd, torch.Tensor) else pkd.float().view(1)

        return data

    def build_graph_shuffled(
        self,
        ligand_pos: torch.Tensor,
        ligand_element: torch.Tensor,
        pocket_pos: torch.Tensor,
        pocket_element: torch.Tensor,
        pocket_aa_type: torch.Tensor,
        pkd: float | torch.Tensor | None = None,
    ) -> Data:
        """Build an interaction graph with SHUFFLED edges (ablation A1.10).

        Same node features, same edge count, but edges connect random
        ligand-pocket pairs instead of distance-based contacts.
        """
        n_lig = ligand_pos.shape[0]
        n_pkt = pocket_pos.shape[0]

        # First build the real graph to get the edge count
        real_graph = self.build_graph(
            ligand_pos, ligand_element, pocket_pos, pocket_element, pocket_aa_type, pkd
        )
        n_edges_half = real_graph.edge_index.shape[1] // 2  # bidirectional

        if n_edges_half == 0:
            return real_graph  # no edges to shuffle

        # Random ligand-pocket pairs (with replacement if needed)
        rand_lig = torch.randint(0, n_lig, (n_edges_half,))
        rand_pkt = torch.randint(0, n_pkt, (n_edges_half,))

        # Compute distances for the random pairs (for RBF features)
        rand_dist = (ligand_pos[rand_lig] - pocket_pos[rand_pkt]).norm(dim=-1)

        # Build edge features with the random distances
        rbf = self._gaussian_rbf(rand_dist)
        lig_elem_idx = self._map_elements(ligand_element[rand_lig], LIGAND_ELEMENT_TO_IDX)
        lig_elem_oh = torch.nn.functional.one_hot(lig_elem_idx, NUM_LIGAND_ELEMENTS).float()
        pkt_elem_idx = self._map_elements(pocket_element[rand_pkt], POCKET_ELEMENT_TO_IDX)
        pkt_elem_oh = torch.nn.functional.one_hot(pkt_elem_idx, NUM_POCKET_ELEMENTS).float()
        aa_idx = pocket_aa_type[rand_pkt].long().clamp(0, NUM_AA_TYPES - 1)
        aa_oh = torch.nn.functional.one_hot(aa_idx, NUM_AA_TYPES).float()
        edge_feat_half = torch.cat([rbf, lig_elem_oh, pkt_elem_oh, aa_oh], dim=-1)

        # Bidirectional
        src = rand_lig
        dst = rand_pkt + n_lig
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        ).long()
        edge_attr = torch.cat([edge_feat_half, edge_feat_half], dim=0).float()

        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=real_graph.node_type,
            element_idx=real_graph.element_idx,
            aa_type_idx=real_graph.aa_type_idx,
            num_lig_atoms=n_lig,
            num_pkt_atoms=n_pkt,
            num_nodes=n_lig + n_pkt,
        )
        if pkd is not None:
            data.y = torch.tensor([pkd], dtype=torch.float32) if not isinstance(pkd, torch.Tensor) else pkd.float().view(1)
        return data

    def _gaussian_rbf(self, dist: torch.Tensor) -> torch.Tensor:
        """Gaussian radial basis function expansion.

        Args:
            dist: (E,) pairwise distances

        Returns:
            (E, rbf_centers) RBF-expanded distances
        """
        dist = dist.float().unsqueeze(-1)  # (E, 1)
        mu = self.rbf_mu.to(dist.device)   # (C,)
        return torch.exp(-((dist - mu) ** 2) / (2 * self.rbf_sigma ** 2))

    @staticmethod
    def _map_elements(atomic_numbers: torch.Tensor, vocab: dict) -> torch.Tensor:
        """Map atomic numbers to vocabulary indices. Unknown → last index."""
        max_idx = max(vocab.values()) + 1  # "other" index
        result = torch.full_like(atomic_numbers, max_idx, dtype=torch.long)
        for z, idx in vocab.items():
            result[atomic_numbers == z] = idx
        return result
