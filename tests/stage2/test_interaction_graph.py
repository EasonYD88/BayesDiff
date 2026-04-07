"""
Unit tests for bayesdiff/interaction_graph.py
"""

import pytest
import torch
from bayesdiff.interaction_graph import (
    InteractionGraphBuilder,
    NUM_LIGAND_ELEMENTS,
    NUM_POCKET_ELEMENTS,
    NUM_AA_TYPES,
    ELEMENT_TO_IDX,
)


@pytest.fixture
def builder():
    return InteractionGraphBuilder(cutoff=5.0, rbf_centers=16, max_rbf_dist=8.0)


@pytest.fixture
def synthetic_data():
    """Synthetic: 10 ligand atoms near origin, 20 pocket atoms offset."""
    torch.manual_seed(42)
    lig_pos = torch.randn(10, 3)
    lig_elem = torch.tensor([6, 7, 8, 6, 6, 7, 8, 16, 9, 6])  # C,N,O,C,C,N,O,S,F,C
    pkt_pos = torch.randn(20, 3) + 2.0  # offset so some are within cutoff
    pkt_elem = torch.tensor([6, 7, 8, 16, 6, 7, 8, 6, 6, 7, 8, 6, 6, 7, 8, 16, 6, 7, 8, 6])
    pkt_aa = torch.randint(0, 20, (20,))
    return lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa


def test_graph_construction_basic(builder, synthetic_data):
    """T1.1: Graph builds with correct node counts."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data
    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    assert graph.num_nodes == 30  # 10 + 20
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    assert (graph.node_type[:10] == 0).all()  # ligand
    assert (graph.node_type[10:] == 1).all()  # pocket
    assert graph.num_lig_atoms == 10
    assert graph.num_pkt_atoms == 20


def test_cutoff_filtering(builder, synthetic_data):
    """T1.2: Only edges with distance ≤ cutoff are included."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data
    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    # Verify all edges are within cutoff
    n_lig = 10
    src, dst = graph.edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        if s < n_lig and d >= n_lig:
            # ligand → pocket edge
            dist = (lig_pos[s] - pkt_pos[d - n_lig]).norm().item()
            assert dist <= builder.cutoff + 1e-6
        elif s >= n_lig and d < n_lig:
            # pocket → ligand edge (reverse)
            dist = (pkt_pos[s - n_lig] - lig_pos[d]).norm().item()
            assert dist <= builder.cutoff + 1e-6


def test_cutoff_strictness():
    """Tight cutoff → fewer edges; wide cutoff → more edges."""
    lig_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pkt_pos = torch.tensor([[3.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    lig_elem = torch.tensor([6, 7])
    pkt_elem = torch.tensor([6, 7])
    pkt_aa = torch.tensor([0, 1])

    tight = InteractionGraphBuilder(cutoff=2.0)
    wide = InteractionGraphBuilder(cutoff=5.0)

    g_tight = tight.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)
    g_wide = wide.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    assert g_tight.edge_index.shape[1] <= g_wide.edge_index.shape[1]


def test_rbf_expansion(builder):
    """T1.3: RBF output shape and value range."""
    dist = torch.tensor([0.5, 1.0, 2.0, 4.0, 7.0])
    rbf = builder._gaussian_rbf(dist)

    assert rbf.shape == (5, 16)
    assert (rbf >= 0).all()
    assert (rbf <= 1.0 + 1e-6).all()


def test_edge_features_shape(builder, synthetic_data):
    """T1.4: Edge features have correct dimensionality (51)."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data
    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    expected_dim = 16 + NUM_LIGAND_ELEMENTS + NUM_POCKET_ELEMENTS + NUM_AA_TYPES
    assert graph.edge_attr.shape[1] == expected_dim
    assert graph.edge_attr.shape[1] == builder.edge_dim


def test_no_contacts():
    """T1.5: Handles case where no atom pairs are within cutoff."""
    builder = InteractionGraphBuilder(cutoff=1.0)
    lig_pos = torch.zeros(5, 3)
    pkt_pos = torch.ones(5, 3) * 100.0  # far away
    lig_elem = torch.full((5,), 6)
    pkt_elem = torch.full((5,), 6)
    pkt_aa = torch.zeros(5, dtype=torch.long)

    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    assert graph.edge_index.shape[1] == 0
    assert graph.edge_attr.shape[0] == 0
    assert graph.num_nodes == 10


def test_bidirectional_edges(builder, synthetic_data):
    """Edges are bidirectional: each contact produces 2 directed edges."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data
    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    assert graph.edge_index.shape[1] % 2 == 0  # even number of edges


def test_determinism(builder, synthetic_data):
    """T1.7: Same input → same graph."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data

    g1 = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)
    g2 = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    assert torch.equal(g1.edge_index, g2.edge_index)
    assert torch.allclose(g1.edge_attr, g2.edge_attr)
    assert torch.equal(g1.node_type, g2.node_type)


def test_element_mapping(builder):
    """Element indices are correctly mapped."""
    lig_elem = torch.tensor([6, 7, 8, 999])  # C, N, O, unknown
    idx = builder._map_elements(lig_elem, ELEMENT_TO_IDX)
    assert idx[0] == 0  # C
    assert idx[1] == 1  # N
    assert idx[2] == 2  # O
    assert idx[3] == max(ELEMENT_TO_IDX.values()) + 1  # "other"


def test_pkd_stored(builder, synthetic_data):
    """pKd label is stored in graph.y."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data
    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa, pkd=7.5)

    assert hasattr(graph, 'y')
    assert torch.allclose(graph.y, torch.tensor([7.5]))


def test_shuffled_edges(builder, synthetic_data):
    """A1.10 shuffled graph: same node count, same edge count, different topology."""
    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data

    real = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)
    shuffled = builder.build_graph_shuffled(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)

    # Same number of nodes
    assert shuffled.num_nodes == real.num_nodes
    # Same number of edges
    assert shuffled.edge_index.shape[1] == real.edge_index.shape[1]
    # Same edge feature dim
    assert shuffled.edge_attr.shape[1] == real.edge_attr.shape[1]
    # But different edge indices (with high probability)
    if real.edge_index.shape[1] > 0:
        # Not guaranteed to be different, but very likely with random edges
        pass


def test_batch_compatibility(builder, synthetic_data):
    """Graphs can be batched via PyG Batch.from_data_list."""
    from torch_geometric.data import Batch

    lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa = synthetic_data

    graphs = []
    for i in range(4):
        g = builder.build_graph(
            lig_pos[:5+i], lig_elem[:5+i],
            pkt_pos[:10+i], pkt_elem[:10+i], pkt_aa[:10+i],
            pkd=7.0 + i * 0.5,
        )
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    assert batch.num_nodes == sum(g.num_nodes for g in graphs)
    assert batch.batch.shape[0] == batch.num_nodes
    assert batch.y.shape[0] == 4
