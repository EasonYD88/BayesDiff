"""
scripts/11_reextract_embeddings.py
──────────────────────────────────
Re-extract SE(3)-invariant embeddings from existing generated SDF molecules
using a single TargetDiff encoder forward pass (NOT full diffusion sampling).

The HPC-generated embeddings are all zeros because the embedding capture
failed during sampling. This script:
  1. Loads the pretrained TargetDiff model
  2. For each pocket, loads the pocket PDB + generated SDF
  3. Runs a SINGLE forward pass at t=0 to extract final_ligand_h
  4. Applies scatter_mean for per-molecule embeddings (128-dim)
  5. Saves corrected all_embeddings.npz

Usage:
    python scripts/11_reextract_embeddings.py \
        --targetdiff_dir external/targetdiff \
        --test_set_dir external/targetdiff/data/test_set \
        --sdf_root results/embedding_1000step \
        --output results/embedding_1000step/merged/all_embeddings.npz \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
# Monkey-patch numpy for TargetDiff compatibility (deprecated aliases removed in NumPy 2.x)
for _alias, _target in [('long', np.int64), ('int', np.int64), ('bool', np.bool_),
                         ('float', np.float64), ('complex', np.complex128),
                         ('str', np.str_), ('object', np.object_)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ReExtract")


def parse_rdmol_to_ligand_dict(rdmol):
    """Convert an RDKit Mol to a ligand dict compatible with TargetDiff."""
    import os
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem import RDConfig as _RDConfig

    fdefName = os.path.join(_RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    rd_num_atoms = rdmol.GetNumAtoms()

    # Atom families (pharmacophore features)
    ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe',
                     'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                     'ZnBinder']
    ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.int64)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Hybridization
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybridization.append(str(atom.GetHybridization()))

    # Positions and elements
    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = np.array([a.GetAtomicNum() for a in rdmol.GetAtoms()], dtype=np.int64)

    # Bonds
    row, col, edge_type = [], [], []
    BOND_TYPES = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bt = BOND_TYPES.get(bond.GetBondType(), 1)
        row += [start, end]
        col += [end, start]
        edge_type += [bt, bt]

    if row:
        edge_index = np.array([row, col], dtype=np.int64)
        edge_type_arr = np.array(edge_type, dtype=np.int64)
        perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type_arr = edge_type_arr[perm]
    else:
        edge_index = np.zeros([2, 0], dtype=np.int64)
        edge_type_arr = np.zeros([0], dtype=np.int64)

    return {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type_arr,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
    }


def extract_embeddings_forward(
    model,
    protein_featurizer,
    ligand_featurizer,
    pocket_pdb: str,
    sdf_path: str,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract embeddings via single forward pass (no diffusion sampling).

    Returns shape (N_mols, hidden_dim).
    """
    from rdkit import Chem
    from datasets.pl_data import ProteinLigandData, torchify_dict
    from utils.data import PDBProtein
    from torch_geometric.transforms import Compose

    transform = Compose([protein_featurizer])

    # Parse pocket once
    pocket_dict = PDBProtein(pocket_pdb).to_dict_atom()
    pocket_dict_torch = torchify_dict(pocket_dict)

    # Load molecules from SDF
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
    rdmols = []
    for mol in suppl:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveHs(mol)
                if mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0:
                    rdmols.append(mol)
            except Exception:
                pass

    if not rdmols:
        logger.warning(f"No valid molecules in {sdf_path}")
        return np.zeros((0, 128), dtype=np.float32)

    all_embeddings = []

    # Process in batches
    for batch_start in range(0, len(rdmols), batch_size):
        batch_mols = rdmols[batch_start:batch_start + batch_size]

        # Build PyG data objects for this batch
        data_list = []
        for mol in batch_mols:
            try:
                lig_dict = parse_rdmol_to_ligand_dict(mol)
                lig_dict_torch = torchify_dict(lig_dict)

                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict={k: v.clone() if torch.is_tensor(v) else v
                                  for k, v in pocket_dict_torch.items()},
                    ligand_dict=lig_dict_torch,
                )
                data = transform(data)
                data = ligand_featurizer(data)
                data_list.append(data)
            except Exception as e:
                logger.debug(f"Skip mol: {e}")
                continue

        if not data_list:
            continue

        # Manual batching (avoid Batch.from_data_list complexity)
        protein_pos_list = []
        protein_feat_list = []
        ligand_pos_list = []
        ligand_feat_list = []
        batch_protein_list = []
        batch_ligand_list = []

        for i, d in enumerate(data_list):
            n_prot = d.protein_pos.size(0)
            n_lig = d.ligand_pos.size(0)

            protein_pos_list.append(d.protein_pos)
            protein_feat_list.append(d.protein_atom_feature)
            ligand_pos_list.append(d.ligand_pos)
            ligand_feat_list.append(d.ligand_atom_feature_full)
            batch_protein_list.append(torch.full((n_prot,), i, dtype=torch.long))
            batch_ligand_list.append(torch.full((n_lig,), i, dtype=torch.long))

        protein_pos = torch.cat(protein_pos_list, dim=0).float().to(device)
        protein_v = torch.cat(protein_feat_list, dim=0).float().to(device)
        batch_protein = torch.cat(batch_protein_list, dim=0).to(device)
        ligand_pos = torch.cat(ligand_pos_list, dim=0).float().to(device)
        ligand_v = torch.cat(ligand_feat_list, dim=0).to(device)
        batch_ligand = torch.cat(batch_ligand_list, dim=0).to(device)

        n_ligand_total = ligand_pos.size(0)
        time_step = torch.zeros(len(data_list), dtype=torch.long, device=device)

        with torch.no_grad():
            preds = model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=time_step,
            )

        final_ligand_h = preds['final_ligand_h']  # (N_ligand_total, hidden_dim)

        # scatter_mean to get per-molecule embeddings
        from torch_scatter import scatter_mean
        mol_embs = scatter_mean(
            final_ligand_h, batch_ligand, dim=0
        )  # (n_mols_in_batch, hidden_dim)

        all_embeddings.append(mol_embs.cpu().numpy())

    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    else:
        return np.zeros((0, 128), dtype=np.float32)


def find_sdf_for_pocket(sdf_root: Path, pdb_code: str) -> Path | None:
    """Find the generated SDF file for a pocket across shard directories."""
    # Search in shard directories
    for shard_dir in sorted(sdf_root.glob("*/shards/*/{}".format(pdb_code))):
        sdf_files = list(shard_dir.glob("*_generated.sdf"))
        if sdf_files:
            return sdf_files[0]
    # Also check direct structure
    for sdf_file in sdf_root.glob(f"**/{pdb_code}/*_generated.sdf"):
        return sdf_file
    return None


def find_pocket_pdb(test_set_dir: Path, pdb_code: str) -> Path | None:
    """Find the pocket PDB file for a given pdb_code."""
    pocket_dir = test_set_dir / pdb_code
    if not pocket_dir.exists():
        return None
    pdb_files = list(pocket_dir.glob("*_rec.pdb"))
    if pdb_files:
        return pdb_files[0]
    pdb_files = list(pocket_dir.glob("*.pdb"))
    return pdb_files[0] if pdb_files else None


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract SE(3) embeddings from generated SDF files"
    )
    parser.add_argument("--targetdiff_dir", type=str,
                        default="external/targetdiff")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detect if not set)")
    parser.add_argument("--test_set_dir", type=str,
                        default="external/targetdiff/data/test_set")
    parser.add_argument("--sdf_root", type=str,
                        default="results/embedding_1000step")
    parser.add_argument("--output", type=str,
                        default="results/embedding_1000step/merged/all_embeddings.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    targetdiff_dir = Path(args.targetdiff_dir).resolve()
    test_set_dir = Path(args.test_set_dir).resolve()
    sdf_root = Path(args.sdf_root).resolve()
    output_path = Path(args.output).resolve()
    device = torch.device(args.device)

    # Add TargetDiff to path
    td_str = str(targetdiff_dir)
    if td_str not in sys.path:
        sys.path.insert(0, td_str)

    # Load model
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = targetdiff_dir / "pretrained_models" / "pretrained_diffusion.pt"

    logger.info(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    import utils.transforms as trans
    from models.molopt_score_model import ScorePosNet3D

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    hidden_dim = config.model.hidden_dim
    logger.info(f"Model loaded. hidden_dim={hidden_dim}, mode={ligand_atom_mode}, device={device}")

    # Find all pockets
    pocket_dirs = sorted([d for d in test_set_dir.iterdir() if d.is_dir()])
    pdb_codes = [d.name for d in pocket_dirs]
    logger.info(f"Found {len(pdb_codes)} pockets in {test_set_dir}")

    # Extract embeddings
    all_embeddings = {}
    stats = {"success": 0, "no_sdf": 0, "no_pdb": 0, "error": 0}
    t_start = time.time()

    for i, pdb_code in enumerate(pdb_codes):
        pocket_pdb = find_pocket_pdb(test_set_dir, pdb_code)
        if pocket_pdb is None:
            logger.warning(f"[{i+1}/{len(pdb_codes)}] {pdb_code}: no PDB found")
            stats["no_pdb"] += 1
            continue

        sdf_file = find_sdf_for_pocket(sdf_root, pdb_code)
        if sdf_file is None:
            logger.warning(f"[{i+1}/{len(pdb_codes)}] {pdb_code}: no SDF found")
            stats["no_sdf"] += 1
            continue

        try:
            t0 = time.time()
            emb = extract_embeddings_forward(
                model, protein_featurizer, ligand_featurizer,
                str(pocket_pdb), str(sdf_file), device,
                batch_size=args.batch_size,
            )
            dt = time.time() - t0
            all_embeddings[pdb_code] = emb
            stats["success"] += 1
            logger.info(
                f"[{i+1}/{len(pdb_codes)}] {pdb_code}: "
                f"shape={emb.shape}, mean={emb.mean():.4f}, "
                f"std={emb.std():.4f}, time={dt:.1f}s"
            )
        except Exception as e:
            logger.error(f"[{i+1}/{len(pdb_codes)}] {pdb_code}: ERROR {e}")
            stats["error"] += 1

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.0f}s. Stats: {stats}")

    # Verify non-zero
    n_nonzero = sum(1 for v in all_embeddings.values() if not np.all(v == 0))
    logger.info(f"Non-zero embeddings: {n_nonzero}/{len(all_embeddings)}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup old file
    if output_path.exists():
        backup = output_path.with_suffix('.npz.bak')
        import shutil
        shutil.copy2(output_path, backup)
        logger.info(f"Backed up old embeddings to {backup}")

    np.savez(output_path, **all_embeddings)
    logger.info(f"Saved {len(all_embeddings)} pocket embeddings to {output_path}")

    # Also save per-pocket .npy files
    for pdb_code, emb in all_embeddings.items():
        # Find shard dir for this pocket and update .npy
        for npy_file in sdf_root.glob(f"**/shards/*/{pdb_code}/{pdb_code}_embeddings.npy"):
            np.save(npy_file, emb)
            logger.debug(f"Updated {npy_file}")

    # Save summary
    summary = {
        "n_pockets": len(all_embeddings),
        "hidden_dim": hidden_dim,
        "elapsed_s": elapsed,
        "stats": stats,
        "non_zero_pockets": n_nonzero,
        "shapes": {k: list(v.shape) for k, v in all_embeddings.items()},
    }
    summary_path = output_path.parent / "reextract_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
