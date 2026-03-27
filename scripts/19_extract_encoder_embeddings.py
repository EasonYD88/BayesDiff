#!/usr/bin/env python3
"""Extract TargetDiff encoder embeddings for all sampled pockets.

For each pocket with generated molecules in results/tier3_sampling/:
1. Load protein data (from .pt or PDB)
2. Read generated molecules from SDF → positions + atom types
3. Center positions (match training convention)
4. One forward pass through ScorePosNet3D with fix_x=True
5. Extract final_ligand_h (128-dim per atom)
6. Aggregate per molecule with scatter_mean → (n_mols, 128)
7. Save encoder_embeddings.npy per pocket + X_encoder_128.npy global

Since time_emb_dim=0 in TargetDiff training config, no time_step is needed.
This is a SINGLE forward pass per batch (cheap), not 1000-step diffusion.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

# ── Paths ──
REPO = Path(__file__).resolve().parent.parent
TD_DIR = REPO / "external" / "targetdiff"
sys.path.insert(0, str(TD_DIR))
os.chdir(TD_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TargetDiff atom type mapping (add_aromatic mode, 13 classes)
MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12,
}


def load_model():
    """Load the TargetDiff model."""
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

    logger.info(
        f"Model loaded: hidden_dim={config.model.hidden_dim}, "
        f"time_emb_dim={config.model.time_emb_dim}, device={DEVICE}"
    )
    return model, protein_featurizer, config


def load_pocket_protein_data(pocket_info, protein_featurizer):
    """Load protein data and apply featurizer transform."""
    from torch_geometric.transforms import Compose
    from torch_geometric.data import Data

    transform = Compose([protein_featurizer])
    source = pocket_info["source"]

    if source == "lmdb":
        pt_path = Path(pocket_info["data_path"])
        if not pt_path.exists():
            raise FileNotFoundError(f"Pocket .pt not found: {pt_path}")
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
                protein_dict=protein_dict, ligand_dict=ligand_dict
            )
        else:
            data = Data()
            for k, v in protein_dict.items():
                setattr(data, f"protein_{k}", v)
            for k, v in ligand_dict.items():
                setattr(data, f"ligand_{k}", v)
        data = transform(data)

    elif source == "test_set":
        pdb_path = pocket_info["pdb_path"]
        if not Path(pdb_path).exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
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
                protein_dict=protein_dict, ligand_dict=ligand_dict
            )
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
    """Read molecules from SDF and convert to TargetDiff ligand format.

    Returns:
        positions: list of (n_atoms, 3) float tensors
        types: list of (n_atoms,) long tensors (TargetDiff atom type indices)
    """
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    positions_list = []
    types_list = []

    for mol in supplier:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
        except Exception:
            continue

        positions = []
        types = []
        valid = True

        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            is_aromatic = atom.GetIsAromatic()

            key = (atomic_num, is_aromatic)
            if key in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
                type_idx = MAP_ATOM_TYPE_AROMATIC_TO_INDEX[key]
            elif (atomic_num, False) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
                # Fallback: treat as non-aromatic
                type_idx = MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(atomic_num, False)]
            else:
                # Unknown atom type (e.g., Br, I) — skip molecule
                valid = False
                break

            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            types.append(type_idx)

        if valid and len(positions) > 0:
            positions_list.append(torch.tensor(positions, dtype=torch.float32))
            types_list.append(torch.tensor(types, dtype=torch.long))

    return positions_list, types_list


def extract_embeddings_for_pocket(
    model, protein_data, ligand_positions, ligand_types, batch_size=8
):
    """Extract encoder embeddings for molecules in a pocket.

    Returns: (n_mols, hidden_dim) numpy array
    """
    from torch_scatter import scatter_mean

    n_mols = len(ligand_positions)
    all_embeddings = []

    protein_pos = protein_data.protein_pos.float().to(DEVICE)
    protein_v = protein_data.protein_atom_feature.float().to(DEVICE)
    n_protein = len(protein_pos)

    for start in range(0, n_mols, batch_size):
        end = min(start + batch_size, n_mols)
        batch_mols = end - start

        # Replicate protein for each molecule in batch
        bp_pos = protein_pos.repeat(batch_mols, 1)
        bp_v = protein_v.repeat(batch_mols, 1)
        bp_idx = torch.arange(batch_mols, device=DEVICE).repeat_interleave(n_protein)

        # Concatenate ligand atoms
        bl_pos = torch.cat(
            [ligand_positions[i].to(DEVICE) for i in range(start, end)]
        )
        bl_v = torch.cat(
            [ligand_types[i].to(DEVICE) for i in range(start, end)]
        )

        # Build batch_ligand index
        n_atoms_per_mol = torch.tensor(
            [len(ligand_positions[i]) for i in range(start, end)],
            device=DEVICE,
        )
        bl_idx = torch.arange(batch_mols, device=DEVICE).repeat_interleave(
            n_atoms_per_mol
        )

        # Center positions (match training convention: center on protein mean)
        center = scatter_mean(bp_pos, bp_idx, dim=0)  # (batch_mols, 3)
        bp_pos_c = bp_pos - center[bp_idx]
        bl_pos_c = bl_pos - center[bl_idx]

        # Forward pass with fix_x=True (positions not updated, pure encoding)
        with torch.no_grad():
            preds = model(
                protein_pos=bp_pos_c,
                protein_v=bp_v,
                batch_protein=bp_idx,
                init_ligand_pos=bl_pos_c,
                init_ligand_v=bl_v,
                batch_ligand=bl_idx,
                fix_x=True,
            )

        # Extract per-molecule embeddings
        final_ligand_h = preds["final_ligand_h"]  # (total_lig_atoms, 128)
        mol_embs = scatter_mean(final_ligand_h, bl_idx, dim=0)  # (batch_mols, 128)
        all_embeddings.append(mol_embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)  # (n_mols, 128)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract TargetDiff encoder embeddings for sampled molecules"
    )
    parser.add_argument(
        "--pocket-list",
        default=str(REPO / "data" / "tier3_pocket_list.json"),
    )
    parser.add_argument(
        "--sampling-dir",
        default=str(REPO / "results" / "tier3_sampling"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "tier3_gp"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # ── Load pocket list ──
    with open(args.pocket_list) as f:
        pocket_data = json.load(f)
    pockets = pocket_data["pockets"]
    logger.info(f"Loaded {len(pockets)} pockets from {args.pocket_list}")

    # ── Load model ──
    model, protein_featurizer, config = load_model()
    hidden_dim = config.model.hidden_dim

    sampling_dir = Path(args.sampling_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build family → pocket info map
    family_to_info = {p["family"]: p for p in pockets}

    # Find pockets with valid SDF files
    pocket_dirs = sorted(
        [
            d
            for d in sampling_dir.iterdir()
            if d.is_dir() and (d / "molecules.sdf").exists()
        ]
    )
    logger.info(f"Found {len(pocket_dirs)} pockets with SDF files")

    # ── Process each pocket ──
    all_embeddings = []
    all_families = []
    all_pkds = []
    all_mol_counts = []
    n_failed = 0
    t_start = time.time()

    for idx, pocket_dir in enumerate(pocket_dirs):
        family = pocket_dir.name
        if family not in family_to_info:
            continue

        pocket_info = family_to_info[family]
        sdf_path = pocket_dir / "molecules.sdf"

        t0 = time.time()
        try:
            # 1. Load protein data
            protein_data = load_pocket_protein_data(
                pocket_info, protein_featurizer
            )

            # 2. Read molecules from SDF
            lig_positions, lig_types = sdf_to_ligand_data(sdf_path)
            if len(lig_positions) == 0:
                logger.warning(f"[{idx+1}] {family}: no valid molecules, skipping")
                n_failed += 1
                continue

            # 3. Extract encoder embeddings
            mol_embeddings = extract_embeddings_for_pocket(
                model,
                protein_data,
                lig_positions,
                lig_types,
                batch_size=args.batch_size,
            )

            # 4. Save per-pocket encoder embeddings
            emb_path = pocket_dir / "encoder_embeddings.npy"
            np.save(emb_path, mol_embeddings)

            # 5. Per-pocket mean embedding
            pocket_emb = mol_embeddings.mean(axis=0)

            all_embeddings.append(pocket_emb)
            all_families.append(family)
            all_pkds.append(pocket_info["pKd"])
            all_mol_counts.append(len(lig_positions))

            elapsed = time.time() - t0
            if (idx + 1) % 100 == 0 or idx == 0:
                logger.info(
                    f"[{idx+1}/{len(pocket_dirs)}] {family}: "
                    f"{len(lig_positions)} mols → emb {mol_embeddings.shape}, "
                    f"{elapsed:.1f}s"
                )

        except Exception as e:
            logger.error(f"[{idx+1}] {family}: FAILED - {e}")
            n_failed += 1
            continue

        # Clean GPU memory periodically
        if DEVICE == "cuda" and (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

    # ── Save global arrays ──
    X = np.stack(all_embeddings)  # (N, 128)
    y = np.array(all_pkds)

    np.save(output_dir / "X_encoder_128.npy", X)
    np.save(output_dir / "y_pkd_encoder.npy", y)
    with open(output_dir / "families_encoder.json", "w") as f:
        json.dump(all_families, f)
    np.save(output_dir / "mol_counts_encoder.npy", np.array(all_mol_counts))

    total_time = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: {len(all_embeddings)} pockets processed, {n_failed} failed")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}, range [{y.min():.2f}, {y.max():.2f}]")
    logger.info(f"Mol counts: {np.array(all_mol_counts).mean():.1f} ± "
                f"{np.array(all_mol_counts).std():.1f}")
    logger.info(f"Embedding stats: mean={X.mean():.4f}, std={X.std():.4f}")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Saved to {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
