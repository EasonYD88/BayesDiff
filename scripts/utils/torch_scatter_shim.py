"""
Compatibility shim for torch_scatter.
Maps scatter operations to torch_geometric's native implementations.
"""
import torch
from torch_geometric.utils import scatter


def scatter_sum(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    result = scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
    return result, None  # torch_scatter returns (values, indices)


def scatter_min(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    result = scatter(src, index, dim=dim, dim_size=dim_size, reduce='min')
    return result, None


def segment_coo(src, index, out=None, dim_size=None, reduce="sum"):
    """segment_coo equivalent using scatter."""
    return scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce)


def scatter_softmax(src, index, dim=-1, dim_size=None):
    """scatter_softmax: compute softmax within groups defined by index."""
    # Get max per group for numerical stability
    max_val = scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
    max_val_expanded = max_val.index_select(dim, index)
    exp_src = torch.exp(src - max_val_expanded)
    sum_exp = scatter(exp_src, index, dim=dim, dim_size=dim_size, reduce='sum')
    sum_exp_expanded = sum_exp.index_select(dim, index)
    return exp_src / (sum_exp_expanded + 1e-16)
