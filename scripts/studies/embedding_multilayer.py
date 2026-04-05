#!/usr/bin/env python3
"""P0++: Extract multi-layer TargetDiff encoder embeddings.

Extends script 19 to extract hidden states from ALL 9 UniTransformer layers
(+ initial embedding = 10 layers total), not just the final layer.

For each pocket, saves: multilayer_embeddings.npz with keys:
  layer_0 ... layer_9: each (n_mols, 128) — per-molecule mean-pooled embeddings
  (layer_0 = initial embedding, layer_1..9 = after attention layers 1-9)

Then runs GP comparison across layer combinations:
  - Individual layers (0-9)
  - Last layer only (current P0 baseline)
  - Concatenated all layers (10×128 = 1280-dim)
  - Concatenated last-K layers
  - Weighted sum (learned or uniform)
"""

import sys
import os
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TD_DIR = REPO / "external" / "targetdiff"
sys.path.insert(0, str(TD_DIR))
os.chdir(TD_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0, (6, False): 1, (6, True): 2, (7, False): 3, (7, True): 4,
    (8, False): 5, (8, True): 6, (9, False): 7, (15, False): 8, (15, True): 9,
    (16, False): 10, (16, True): 11, (17, False): 12,
}


def load_model():
    from models.molopt_score_model import ScorePosNet3D
    import utils.transforms as trans

    ckpt_path = TD_DIR / "pretrained_models" / "pretrained_diffusion.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    logger.info(f"Model loaded: hidden_dim={config.model.hidden_dim}, "
                f"num_layers={config.model.num_layers}, device={DEVICE}")
    return model, protein_featurizer, config


def load_pocket_protein_data(pocket_info, protein_featurizer):
    """Load protein data (identical to script 19)."""
    from torch_geometric.transforms import Compose

    transform = Compose([protein_featurizer])
    source = pocket_info["source"]

    if source == "lmdb":
        pt_path = Path(pocket_info["data_path"])
        pocket_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        try:
            from datasets.pl_data import ProteinLigandData
        except ImportError:
            ProteinLigandData = None

        protein_dict = {
            "element": pocket_dict["protein_element"],
            "pos": pocket_dict["protein_pos"],
            "is_backbone": pocket_dict["protein_is_backbone"],
            "atom_name": pocket_dict["protein_atom_name"],
            "atom_to_aa_type": pocket_dict["protein_atom_to_aa_type"],
        }
        ligand_dict = {
            "element": torch.empty([0], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0], dtype=torch.long),
        }
        if ProteinLigandData and hasattr(ProteinLigandData, "from_protein_ligand_dicts"):
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=protein_dict, ligand_dict=ligand_dict)
        else:
            from torch_geometric.data import Data
            data = Data()
            for k, v in protein_dict.items():
                setattr(data, f"protein_{k}", v)
            for k, v in ligand_dict.items():
                setattr(data, f"ligand_{k}", v)
        data = transform(data)

    elif source == "test_set":
        pdb_path = pocket_info["pdb_path"]
        from utils.data import PDBProtein
        with open(pdb_path, "r") as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)
        protein_dict = protein.to_dict_atom()
        try:
            from datasets.pl_data import ProteinLigandData
        except ImportError:
            ProteinLigandData = None
        ligand_dict = {
            "element": torch.empty([0], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0], dtype=torch.long),
        }
        if ProteinLigandData and hasattr(ProteinLigandData, "from_protein_ligand_dicts"):
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=protein_dict, ligand_dict=ligand_dict)
        else:
            from torch_geometric.data import Data
            data = Data()
            for k, v in protein_dict.items():
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                setattr(data, f"protein_{k}", v)
            for k, v in ligand_dict.items():
                setattr(data, f"ligand_{k}", v)
        data = transform(data)
    else:
        raise ValueError(f"Unknown source: {source}")
    return data


def sdf_to_ligand_data(sdf_path):
    """Read molecules from SDF (identical to script 19)."""
    from rdkit import Chem
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    positions_list, types_list = [], []
    for mol in supplier:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
        except Exception:
            continue
        positions, types, valid = [], [], True
        for atom in mol.GetAtoms():
            key = (atom.GetAtomicNum(), atom.GetIsAromatic())
            if key in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
                type_idx = MAP_ATOM_TYPE_AROMATIC_TO_INDEX[key]
            elif (atom.GetAtomicNum(), False) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
                type_idx = MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(atom.GetAtomicNum(), False)]
            else:
                valid = False
                break
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            types.append(type_idx)
        if valid and len(positions) > 0:
            positions_list.append(torch.tensor(positions, dtype=torch.float32))
            types_list.append(torch.tensor(types, dtype=torch.long))
    return positions_list, types_list


def extract_multilayer_embeddings(model, protein_data, ligand_positions, ligand_types, batch_size=8):
    """Extract per-layer encoder embeddings for molecules.

    Returns: dict of {layer_idx: (n_mols, 128) numpy array}
             layer_idx 0 = initial embedding, 1-9 = after attention layers
    """
    from torch_scatter import scatter_mean

    n_mols = len(ligand_positions)
    protein_pos = protein_data.protein_pos.float().to(DEVICE)
    protein_v = protein_data.protein_atom_feature.float().to(DEVICE)
    n_protein = len(protein_pos)

    # Accumulate per-layer embeddings across batches
    layer_embeddings = None  # will be initialized on first batch

    for start in range(0, n_mols, batch_size):
        end = min(start + batch_size, n_mols)
        batch_mols = end - start

        bp_pos = protein_pos.repeat(batch_mols, 1)
        bp_v = protein_v.repeat(batch_mols, 1)
        bp_idx = torch.arange(batch_mols, device=DEVICE).repeat_interleave(n_protein)

        bl_pos = torch.cat([ligand_positions[i].to(DEVICE) for i in range(start, end)])
        bl_v = torch.cat([ligand_types[i].to(DEVICE) for i in range(start, end)])
        n_atoms_per_mol = torch.tensor(
            [len(ligand_positions[i]) for i in range(start, end)], device=DEVICE)
        bl_idx = torch.arange(batch_mols, device=DEVICE).repeat_interleave(n_atoms_per_mol)

        center = scatter_mean(bp_pos, bp_idx, dim=0)
        bp_pos_c = bp_pos - center[bp_idx]
        bl_pos_c = bl_pos - center[bl_idx]

        with torch.no_grad():
            preds = model(
                protein_pos=bp_pos_c, protein_v=bp_v, batch_protein=bp_idx,
                init_ligand_pos=bl_pos_c, init_ligand_v=bl_v, batch_ligand=bl_idx,
                fix_x=True, return_layer_h=True,
            )

        # preds['layer_ligand_h'] = list of (n_lig_atoms, 128) tensors, length 10
        layer_ligand_h = preds['layer_ligand_h']
        n_layers = len(layer_ligand_h)

        if layer_embeddings is None:
            layer_embeddings = {i: [] for i in range(n_layers)}

        for layer_idx in range(n_layers):
            mol_embs = scatter_mean(layer_ligand_h[layer_idx], bl_idx, dim=0)
            layer_embeddings[layer_idx].append(mol_embs.cpu().numpy())

    # Concatenate batches
    return {i: np.concatenate(embs, axis=0) for i, embs in layer_embeddings.items()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P0++: Extract multi-layer encoder embeddings")
    parser.add_argument("--pocket-list", default=str(REPO / "data" / "tier3_pocket_list.json"))
    parser.add_argument("--sampling-dir", default=str(REPO / "results" / "tier3_sampling"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=None, help="Shard index for array jobs")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    with open(args.pocket_list) as f:
        pocket_data = json.load(f)
    pockets = pocket_data["pockets"]
    family_to_info = {p["family"]: p for p in pockets}
    logger.info(f"Loaded {len(pockets)} pockets from {args.pocket_list}")

    model, protein_featurizer, config = load_model()
    sampling_dir = Path(args.sampling_dir)

    pocket_dirs = sorted([
        d for d in sampling_dir.iterdir()
        if d.is_dir() and (d / "molecules.sdf").exists() and d.name in family_to_info
    ])
    logger.info(f"Found {len(pocket_dirs)} pockets with SDF files")

    # Shard if running as array job
    if args.shard_index is not None:
        pocket_dirs = [d for i, d in enumerate(pocket_dirs) if i % args.num_shards == args.shard_index]
        logger.info(f"Shard {args.shard_index}/{args.num_shards}: processing {len(pocket_dirs)} pockets")

    n_ok, n_fail = 0, 0
    t_start = time.time()

    for idx, pocket_dir in enumerate(pocket_dirs):
        family = pocket_dir.name
        pocket_info = family_to_info[family]
        sdf_path = pocket_dir / "molecules.sdf"

        try:
            protein_data = load_pocket_protein_data(pocket_info, protein_featurizer)
            lig_positions, lig_types = sdf_to_ligand_data(sdf_path)
            if len(lig_positions) == 0:
                n_fail += 1
                continue

            layer_embs = extract_multilayer_embeddings(
                model, protein_data, lig_positions, lig_types, batch_size=args.batch_size)

            # Save as npz: layer_0, layer_1, ..., layer_9
            save_dict = {f"layer_{i}": emb for i, emb in layer_embs.items()}
            np.savez(pocket_dir / "multilayer_embeddings.npz", **save_dict)
            n_ok += 1

            if (idx + 1) % 100 == 0 or idx == 0:
                n_layers = len(layer_embs)
                n_mols = layer_embs[0].shape[0]
                logger.info(f"[{idx+1}/{len(pocket_dirs)}] {family}: "
                            f"{n_mols} mols × {n_layers} layers, OK")

        except Exception as e:
            logger.error(f"[{idx+1}] {family}: FAILED - {e}")
            n_fail += 1

        if DEVICE == "cuda" and (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    logger.info(f"\nDONE: {n_ok} OK, {n_fail} failed, {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
