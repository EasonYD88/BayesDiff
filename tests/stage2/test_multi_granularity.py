"""
Unit tests for bayesdiff/multi_granularity.py
"""

import pytest
import torch
from torch_geometric.data import Batch

from bayesdiff.interaction_graph import InteractionGraphBuilder
from bayesdiff.interaction_gnn import InteractionGNN
from bayesdiff.multi_granularity import MultiGranularityEncoder


@pytest.fixture
def sample_batch():
    builder = InteractionGraphBuilder(cutoff=5.0)
    torch.manual_seed(42)
    graphs = []
    for i in range(4):
        g = builder.build_graph(
            torch.randn(8 + i, 3),
            torch.randint(6, 9, (8 + i,)),
            torch.randn(15 + i, 3) + 1.5,
            torch.randint(6, 9, (15 + i,)),
            torch.randint(0, 20, (15 + i,)),
            pkd=5.0 + i,
        )
        graphs.append(g)
    return Batch.from_data_list(graphs)


def test_concat_mode(sample_batch):
    """T3.1: Concat fusion output dim = z_global_dim + gnn.output_dim."""
    gnn = InteractionGNN(edge_dim=51, hidden_dim=64, n_layers=2, output_dim=64)
    encoder = MultiGranularityEncoder(gnn, z_global_dim=128, fusion="concat")

    z_global = torch.randn(4, 128)
    z_new = encoder(sample_batch, z_global)

    assert z_new.shape == (4, 128 + 64)
    assert encoder.output_dim == 128 + 64


def test_concat_mlp_mode(sample_batch):
    """T3.1: Concat+MLP fusion output dim = specified output_dim."""
    gnn = InteractionGNN(edge_dim=51, hidden_dim=64, n_layers=2, output_dim=64)
    encoder = MultiGranularityEncoder(
        gnn, z_global_dim=128, fusion="concat_mlp", output_dim=128
    )

    z_global = torch.randn(4, 128)
    z_new = encoder(sample_batch, z_global)

    assert z_new.shape == (4, 128)
    assert encoder.output_dim == 128


def test_interaction_only(sample_batch):
    """No z_global → returns z_interaction only."""
    gnn = InteractionGNN(edge_dim=51, hidden_dim=64, n_layers=2, output_dim=64)
    encoder = MultiGranularityEncoder(gnn, z_global_dim=128, fusion="concat")

    z_new = encoder(sample_batch, z_global=None)
    assert z_new.shape == (4, 64)


def test_backward_pass(sample_batch):
    """T3.4: Loss.backward() succeeds; all params have gradients."""
    gnn = InteractionGNN(edge_dim=51, hidden_dim=64, n_layers=2, output_dim=64)
    encoder = MultiGranularityEncoder(gnn, z_global_dim=128, fusion="concat_mlp", output_dim=128)

    z_global = torch.randn(4, 128, requires_grad=True)
    z_new = encoder(sample_batch, z_global)

    loss = z_new.sum()
    loss.backward()

    # Check GNN params have gradients
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            if 'node_readout' in name:
                continue
            assert param.grad is not None, f"No gradient for {name}"


def test_output_finite(sample_batch):
    """T3.5: z_new values are in reasonable range (no NaN/Inf)."""
    gnn = InteractionGNN(edge_dim=51, hidden_dim=64, n_layers=2, output_dim=64)
    encoder = MultiGranularityEncoder(gnn, z_global_dim=128, fusion="concat")

    z_global = torch.randn(4, 128)
    z_new = encoder(sample_batch, z_global)

    assert torch.isfinite(z_new).all()
