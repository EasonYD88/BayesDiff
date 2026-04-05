#!/usr/bin/env python3
"""Phase C: Extract SchNet QM9-pretrained embeddings for all pockets.

Uses PyTorch Geometric's SchNet with pre-trained QM9 weights as a feature
extractor. Extracts 128-dim intermediate representations.

Output: results/tier3_gp/X_schnet_128.npy, y_pkd_schnet.npy, families_schnet.json
"""

import sys
import json
import logging
import time
import unittest.mock
from pathlib import Path

# Workaround: schnetpack → pytorch_lightning → torchmetrics → torchvision
# causes circular import. Mock torchvision and all submodules before importing.
_tv_mock = unittest.mock.MagicMock()
for submod in ['torchvision', 'torchvision.transforms', 'torchvision.models',
               'torchvision.ops', 'torchvision.datasets', 'torchvision.io',
               'torchvision.utils', 'torchvision._meta_registrations',
               'torchvision.extension']:
    if submod not in sys.modules:
        sys.modules[submod] = _tv_mock
    sys.modules['torchvision.transforms'] = unittest.mock.MagicMock()

import types
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Stub modules for old schnetpack 0.3.x classes referenced in the QM9 pickle.
# schnetpack 2.1.1 renamed/removed these modules. We create minimal stubs so
# torch.load() can unpickle the saved model into nn.Module objects whose
# state_dict / attributes we then read manually.
# ---------------------------------------------------------------------------

class _StubModule(nn.Module):
    """Generic nn.Module stub that accepts any constructor args."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return {}

class _ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))

def _register_stub_module(dotted_path, attrs=None):
    """Register a fake module at *dotted_path* with optional class attributes."""
    mod = types.ModuleType(dotted_path)
    if attrs:
        for name, obj in attrs.items():
            setattr(mod, name, obj)
    sys.modules[dotted_path] = mod

_STUBS = {
    'schnetpack.atomistic.model': {
        'AtomisticModel': _StubModule,
        'ModelError': _StubModule,
    },
    'schnetpack.atomistic.output_modules': {
        'Atomwise': _StubModule,
        'Output': _StubModule,
        'ElementalAtomwise': _StubModule,
    },
    'schnetpack.nn.acsf': {
        'GaussianSmearing': _StubModule,
    },
    'schnetpack.nn.cfconv': {
        'CFConv': _StubModule,
    },
    'schnetpack.nn.neighbors': {
        'AtomDistances': _StubModule,
    },
}

for _mod_path, _cls_map in _STUBS.items():
    if _mod_path not in sys.modules:
        _register_stub_module(_mod_path, _cls_map)

# Also ensure shifted_softplus callable is available at old path
if 'schnetpack.nn.activations' not in sys.modules:
    _register_stub_module('schnetpack.nn.activations',
                          {'shifted_softplus': _ShiftedSoftplus,
                           'ShiftedSoftplus': _ShiftedSoftplus})
else:
    # Module exists in 2.1.1 — just ensure the old name is present
    _m = sys.modules['schnetpack.nn.activations']
    if not hasattr(_m, 'shifted_softplus'):
        _m.shifted_softplus = _ShiftedSoftplus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = REPO / "results" / "tier3_gp"
SAMPLING_DIR = REPO / "results" / "tier3_sampling"


def load_schnet_pretrained():
    """Load SchNet pre-trained on QM9 (target 7 = U0).

    We manually download and load weights to avoid requiring the full QM9
    dataset object that from_qm9_pretrained() needs.
    """
    import os
    import warnings
    from torch_geometric.nn.models import SchNet

    root = str(REPO / "data" / "schnet_pretrained")
    os.makedirs(root, exist_ok=True)

    # Download pre-trained weights if needed
    folder = os.path.join(root, 'trained_schnet_models')
    if not os.path.exists(folder):
        import zipfile
        import urllib.request
        url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'
        zip_path = os.path.join(root, 'trained_schnet_models.zip')
        logger.info(f"Downloading SchNet pretrained weights from {url}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        os.unlink(zip_path)

    # Load the saved schnetpack model
    # schnetpack classes must be importable for torch.load to unpickle
    import schnetpack  # noqa: F401

    # Monkey-patch missing old 0.3.x classes into existing 2.1.1 modules
    import schnetpack.nn.base as _nn_base
    for _cn in ('Aggregate', 'GetItem', 'ScaleShift'):
        if not hasattr(_nn_base, _cn):
            setattr(_nn_base, _cn, _StubModule)
    import schnetpack.nn.blocks as _nn_blocks
    if not hasattr(_nn_blocks, 'MLP'):
        _nn_blocks.MLP = _StubModule

    model_path = os.path.join(root, 'trained_schnet_models', 'qm9_energy_U0', 'best_model')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        state = torch.load(model_path, map_location='cpu', weights_only=False)

    # Build a PyG SchNet with matching architecture
    net = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    )

    # Map schnetpack weights → PyG SchNet
    net.embedding.weight = state.representation.embedding.weight

    for int1, int2 in zip(state.representation.interactions, net.interactions):
        int2.mlp[0].weight = int1.filter_network[0].weight
        int2.mlp[0].bias = int1.filter_network[0].bias
        int2.mlp[2].weight = int1.filter_network[1].weight
        int2.mlp[2].bias = int1.filter_network[1].bias
        int2.lin.weight = int1.dense.weight
        int2.lin.bias = int1.dense.bias
        int2.conv.lin1.weight = int1.cfconv.in2f.weight
        int2.conv.lin2.weight = int1.cfconv.f2out.weight
        int2.conv.lin2.bias = int1.cfconv.f2out.bias

    net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
    net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
    net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
    net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

    net = net.to(DEVICE)
    net.eval()
    logger.info(f"SchNet loaded: {sum(p.numel() for p in net.parameters())} parameters")
    return net


def extract_schnet_embeddings_from_sdf(model, sdf_path):
    """Extract SchNet intermediate embeddings from SDF molecules.

    SchNet architecture:
    - embedding(z) → x (128-dim)
    - 6 interaction blocks: x = x + interaction(x, edge_index, edge_weight, edge_attr)
    - lin1(x) → lin2 → readout (per-atom → sum → scalar)

    We hook into the output of the last interaction block (128-dim per atom),
    then mean-pool over atoms per molecule.
    """
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol_embeddings = []

    for mol in supplier:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
        except Exception:
            continue

        z_list, pos_list = [], []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num > 0:
                z_list.append(atomic_num)
                p = conf.GetAtomPosition(atom.GetIdx())
                pos_list.append([p.x, p.y, p.z])

        if len(z_list) == 0:
            continue

        z = torch.tensor(z_list, dtype=torch.long, device=DEVICE)
        pos = torch.tensor(pos_list, dtype=torch.float32, device=DEVICE)
        batch = torch.zeros(len(z), dtype=torch.long, device=DEVICE)

        # Extract intermediate representation via hook
        intermediate = {}

        def hook_fn(module, input, output):
            intermediate['h'] = output.detach()

        # Hook the last interaction block
        handle = model.interactions[-1].register_forward_hook(hook_fn)

        with torch.no_grad():
            try:
                model(z, pos, batch)
            except Exception:
                handle.remove()
                continue

        handle.remove()

        if 'h' in intermediate:
            # Mean pool over atoms → (128,)
            h = intermediate['h']
            mol_emb = h.mean(dim=0).cpu().numpy()
            mol_embeddings.append(mol_emb)

    return mol_embeddings


def main():
    logger.info("Loading SchNet pre-trained model...")
    model = load_schnet_pretrained()

    with open(REPO / "data" / "tier3_pocket_list.json") as f:
        pocket_data = json.load(f)
    family_to_pkd = {p["family"]: p["pKd"] for p in pocket_data["pockets"]}

    pocket_dirs = sorted([
        d for d in SAMPLING_DIR.iterdir()
        if d.is_dir() and (d / "molecules.sdf").exists() and d.name in family_to_pkd
    ])
    logger.info(f"Found {len(pocket_dirs)} pockets")

    families, embeddings, pkd_values = [], [], []
    n_ok, n_fail = 0, 0
    t_start = time.time()

    for idx, pocket_dir in enumerate(pocket_dirs):
        family = pocket_dir.name
        sdf_path = pocket_dir / "molecules.sdf"

        try:
            mol_embs = extract_schnet_embeddings_from_sdf(model, sdf_path)
            if len(mol_embs) == 0:
                n_fail += 1
                continue

            pocket_emb = np.mean(mol_embs, axis=0)
            embeddings.append(pocket_emb)
            families.append(family)
            pkd_values.append(family_to_pkd[family])
            n_ok += 1

            if (idx + 1) % 50 == 0 or idx == 0:
                logger.info(f"[{idx+1}/{len(pocket_dirs)}] {family}: {len(mol_embs)} mols OK")

        except Exception as e:
            logger.error(f"[{idx+1}] {family}: FAILED - {e}")
            n_fail += 1

    elapsed = time.time() - t_start
    logger.info(f"\nDONE: {n_ok} OK, {n_fail} failed, {elapsed:.0f}s")

    X = np.stack(embeddings)
    y = np.array(pkd_values)
    logger.info(f"Final: X={X.shape}, y={y.shape}")

    np.save(DATA_DIR / "X_schnet_128.npy", X)
    np.save(DATA_DIR / "y_pkd_schnet.npy", y)
    with open(DATA_DIR / "families_schnet.json", "w") as f:
        json.dump(families, f)

    logger.info(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
