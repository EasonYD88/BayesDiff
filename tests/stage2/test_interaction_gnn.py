"""
Unit tests for bayesdiff/interaction_gnn.py
"""

import pytest
import torch
from torch_geometric.data import Batch

from bayesdiff.interaction_graph import InteractionGraphBuilder
from bayesdiff.interaction_gnn import InteractionGNN, InteractionGNNPredictor


@pytest.fixture
def gnn():
    return InteractionGNN(edge_dim=51, hidden_dim=64, n_layers=2, output_dim=64)


@pytest.fixture
def sample_batch():
    """Create a batched graph from synthetic data."""
    builder = InteractionGraphBuilder(cutoff=5.0)
    torch.manual_seed(42)

    graphs = []
    for i in range(4):
        n_lig = 8 + i
        n_pkt = 15 + i * 2
        lig_pos = torch.randn(n_lig, 3)
        pkt_pos = torch.randn(n_pkt, 3) + 1.5
        lig_elem = torch.randint(6, 9, (n_lig,))  # C, N, O
        pkt_elem = torch.randint(6, 9, (n_pkt,))
        pkt_aa = torch.randint(0, 20, (n_pkt,))
        g = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa, pkd=5.0 + i)
        graphs.append(g)

    return Batch.from_data_list(graphs)


def test_forward_shape(gnn, sample_batch):
    """T2.1: Output shape = (batch_size, output_dim)."""
    z = gnn(sample_batch)
    assert z.shape == (4, 64)


def test_gradient_flow(gnn, sample_batch):
    """T2.2: Gradients flow through all parameters."""
    z = gnn(sample_batch)
    loss = z.sum()
    loss.backward()

    for name, param in gnn.named_parameters():
        if param.requires_grad:
            # node_readout is only used for 0-edge graphs when readout_mode='edge'
            if 'node_readout' in name and gnn.readout_mode == 'edge':
                continue
            # edge_readout is unused when readout_mode='node'
            if 'edge_readout' in name and gnn.readout_mode == 'node':
                continue
            assert param.grad is not None, f"No gradient for {name}"


def test_output_finite(gnn, sample_batch):
    """Output values are finite (no NaN/Inf)."""
    z = gnn(sample_batch)
    assert torch.isfinite(z).all()


def test_empty_graph(gnn):
    """T2.4: Graceful handling of graph with no edges."""
    builder = InteractionGraphBuilder(cutoff=0.1)
    lig_pos = torch.zeros(5, 3)
    pkt_pos = torch.ones(5, 3) * 100.0
    lig_elem = torch.full((5,), 6)
    pkt_elem = torch.full((5,), 6)
    pkt_aa = torch.zeros(5, dtype=torch.long)

    graph = builder.build_graph(lig_pos, lig_elem, pkt_pos, pkt_elem, pkt_aa)
    batch = Batch.from_data_list([graph])

    z = gnn(batch)
    assert z.shape == (1, 64)
    assert torch.isfinite(z).all()


def test_variable_size(gnn):
    """T2.5: Different-sized molecules in same batch."""
    builder = InteractionGraphBuilder(cutoff=5.0)
    torch.manual_seed(0)

    graphs = []
    for n_lig, n_pkt in [(3, 5), (10, 30), (20, 50)]:
        g = builder.build_graph(
            torch.randn(n_lig, 3),
            torch.randint(6, 9, (n_lig,)),
            torch.randn(n_pkt, 3) + 1.0,
            torch.randint(6, 9, (n_pkt,)),
            torch.randint(0, 20, (n_pkt,)),
        )
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    z = gnn(batch)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_predictor(gnn, sample_batch):
    """InteractionGNNPredictor returns prediction + z_interaction."""
    predictor = InteractionGNNPredictor(gnn)
    pred, z = predictor(sample_batch)

    assert pred.shape == (4,)
    assert z.shape == (4, 64)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(z).all()

    # Test backward
    loss = ((pred - sample_batch.y.squeeze()) ** 2).mean()
    loss.backward()
    for name, param in predictor.named_parameters():
        if param.requires_grad:
            # Skip readout heads not used in current mode
            if 'node_readout' in name and gnn.readout_mode == 'edge':
                continue
            if 'edge_readout' in name and gnn.readout_mode == 'node':
                continue
            assert param.grad is not None, f"No gradient for {name}"


def test_single_graph(gnn):
    """Works with batch_size=1."""
    builder = InteractionGraphBuilder(cutoff=5.0)
    torch.manual_seed(42)
    g = builder.build_graph(
        torch.randn(10, 3),
        torch.randint(6, 9, (10,)),
        torch.randn(20, 3) + 1.0,
        torch.randint(6, 9, (20,)),
        torch.randint(0, 20, (20,)),
    )
    batch = Batch.from_data_list([g])
    z = gnn(batch)
    assert z.shape == (1, 64)
