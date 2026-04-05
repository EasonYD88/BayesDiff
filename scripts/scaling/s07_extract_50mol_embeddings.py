#!/usr/bin/env python3
"""Extract encoder embeddings for 50mol shard data, then merge with tier3.

The 50mol sampling pipeline only ran diffusion (no encoder extraction),
producing all-zero embeddings. This script does a post-hoc encoder pass:
  1. Load TargetDiff model
  2. For each pocket in embedding_50mol shards, load protein + SDF
  3. Forward pass with fix_x=True → extract final_ligand_h
  4. Mean-pool per molecule → scatter_mean per pocket
  5. Match with pKd labels
  6. Merge with tier3 data (replace tier3 embeddings with 50mol where overlapping)
  7. Save everything to results/50mol_gp/
"""

import sys
import os
import glob
import json
import pickle
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

EMB_DIR = REPO / "results" / "embedding_50mol"
TIER3_DIR = REPO / "results" / "tier3_gp"
OUTPUT_DIR = REPO / "results" / "50mol_gp"
TEST_SET_DIR = TD_DIR / "data" / "test_set"
AFF_PKL = TD_DIR / "data" / "affinity_info.pkl"

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0, (6, False): 1, (6, True): 2,
    (7, False): 3, (7, True): 4, (8, False): 5, (8, True): 6,
    (9, False): 7, (15, False): 8, (15, True): 9,
    (16, False): 10, (16, True): 11, (17, False): 12,
}


def load_model():
    from models.molopt_score_model import ScorePosNet3D
    import utils.transforms as trans

    ckpt_path = TD_DIR / "pretrained_models" / "pretrained_diffusion.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    logger.info(f"Model loaded: hidden_dim={config.model.hidden_dim}, device={DEVICE}")
    return model, protein_featurizer, config


def load_test_set_protein(pocket_name, protein_featurizer):
    """Load protein data from test_set directory."""
    from torch_geometric.transforms import Compose
    from torch_geometric.data import Data
    from utils.data import PDBProtein

    pocket_dir = TEST_SET_DIR / pocket_name
    pdb_files = list(pocket_dir.glob("*_rec.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB for {pocket_name} in {pocket_dir}")

    with open(pdb_files[0], "r") as f:
        pdb_block = f.read()

    protein = PDBProtein(pdb_block)
    protein_dict = protein.to_dict_atom()

    ligand_dict = {
        "element": torch.empty([0], dtype=torch.long),
        "pos": torch.empty([0, 3], dtype=torch.float),
        "atom_feature": torch.empty([0, 8], dtype=torch.float),
        "bond_index": torch.empty([2, 0], dtype=torch.long),
        "bond_type": torch.empty([0], dtype=torch.long),
    }

    # Build Data manually — ProteinLigandData.from_protein_ligand_dicts
    # leaves numpy arrays unconverted, causing FeaturizeProteinAtom to fail
    data = Data()
    for k, v in protein_dict.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        setattr(data, f"protein_{k}", v)
    for k, v in ligand_dict.items():
        setattr(data, f"ligand_{k}", v)

    transform = Compose([protein_featurizer])
    data = transform(data)
    return data


def sdf_to_ligand_data(sdf_path):
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

        if valid and positions:
            positions_list.append(torch.tensor(positions, dtype=torch.float32))
            types_list.append(torch.tensor(types, dtype=torch.long))

    return positions_list, types_list


def extract_embeddings_for_pocket(model, protein_data, ligand_positions, ligand_types, batch_size=8):
    from torch_scatter import scatter_mean
    n_mols = len(ligand_positions)
    all_embeddings = []

    protein_pos = protein_data.protein_pos.float().to(DEVICE)
    protein_v = protein_data.protein_atom_feature.float().to(DEVICE)
    n_protein = len(protein_pos)

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
                init_ligand_pos=bl_pos_c, init_ligand_v=bl_v,
                batch_ligand=bl_idx, fix_x=True,
            )

        final_ligand_h = preds["final_ligand_h"]
        mol_embs = scatter_mean(final_ligand_h, bl_idx, dim=0)
        all_embeddings.append(mol_embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def load_affinity():
    with open(AFF_PKL, "rb") as f:
        aff = pickle.load(f)
    pk_map = {}
    for key, info in aff.items():
        pk = info.get("pk")
        if pk is None or float(pk) == 0.0:
            continue
        fam = str(key).split("/")[0]
        pk_map.setdefault(fam, []).append(float(pk))
    return {fam: float(np.mean(vals)) for fam, vals in pk_map.items()}


def collect_shard_sdf_paths():
    """Find all pocket SDF paths across 50mol shards."""
    shard_dirs = sorted(glob.glob(str(EMB_DIR / "*/shards/shard_*of31")))
    logger.info(f"Found {len(shard_dirs)} shard directories")
    pocket_sdfs = {}
    for sd in shard_dirs:
        sd = Path(sd)
        for item in sorted(sd.iterdir()):
            if item.is_dir():
                sdf = item / f"{item.name}_generated.sdf"
                if sdf.exists() and sdf.stat().st_size > 0:
                    pocket_sdfs[item.name] = sdf
                else:
                    logger.warning(f"  {item.name}: SDF missing or empty")
    logger.info(f"Found {len(pocket_sdfs)} pockets with valid SDF files")
    return pocket_sdfs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Load model
    logger.info("=" * 60)
    logger.info("Phase 0: Loading model")
    model, protein_featurizer, config = load_model()
    hidden_dim = config.model.hidden_dim

    # 2. Find all pocket SDF paths
    logger.info("=" * 60)
    logger.info("Phase 1: Collecting SDF paths from shards")
    pocket_sdfs = collect_shard_sdf_paths()

    # 3. Extract embeddings
    logger.info("=" * 60)
    logger.info("Phase 2: Extracting encoder embeddings")
    all_embeddings = {}  # pocket_name → (n_mols, 128)
    n_failed = 0

    for idx, (pocket_name, sdf_path) in enumerate(sorted(pocket_sdfs.items())):
        t0 = time.time()
        try:
            # Check if pocket exists in test_set
            if not (TEST_SET_DIR / pocket_name).exists():
                logger.warning(f"[{idx+1}] {pocket_name}: not in test_set, skipping")
                n_failed += 1
                continue

            protein_data = load_test_set_protein(pocket_name, protein_featurizer)
            lig_positions, lig_types = sdf_to_ligand_data(sdf_path)

            if not lig_positions:
                logger.warning(f"[{idx+1}] {pocket_name}: no valid molecules")
                n_failed += 1
                continue

            mol_embeddings = extract_embeddings_for_pocket(
                model, protein_data, lig_positions, lig_types, batch_size=8)

            all_embeddings[pocket_name] = mol_embeddings
            elapsed = time.time() - t0
            logger.info(f"[{idx+1}/{len(pocket_sdfs)}] {pocket_name}: "
                        f"{len(lig_positions)} mols → emb {mol_embeddings.shape}, "
                        f"norm={np.linalg.norm(mol_embeddings.mean(0)):.4f}, {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"[{idx+1}] {pocket_name}: FAILED - {e}")
            n_failed += 1

        if DEVICE == "cuda" and (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    logger.info(f"\nExtraction done: {len(all_embeddings)} pockets, {n_failed} failed")

    if not all_embeddings:
        logger.error("No embeddings extracted! Exiting.")
        sys.exit(1)

    # 4. Save per-pocket embeddings
    np.savez(OUTPUT_DIR / "per_pocket_embeddings.npz", **all_embeddings)

    # 5. Mean-pool per pocket
    families = sorted(all_embeddings.keys())
    X_perpocket = np.stack([all_embeddings[f].mean(axis=0) for f in families])
    mol_counts = np.array([all_embeddings[f].shape[0] for f in families])
    logger.info(f"Mean-pooled: {X_perpocket.shape}, "
                f"norm range=[{np.linalg.norm(X_perpocket, axis=1).min():.4f}, "
                f"{np.linalg.norm(X_perpocket, axis=1).max():.4f}]")

    # 6. Match with pKd
    pk_map = load_affinity()
    matched = [(i, f) for i, f in enumerate(families) if f in pk_map]
    idx_m = [m[0] for m in matched]
    fam_m = [m[1] for m in matched]

    X_50mol = X_perpocket[idx_m]
    y_50mol = np.array([pk_map[f] for f in fam_m])
    logger.info(f"Matched with pKd: {len(fam_m)}/{len(families)}")
    logger.info(f"X_50mol: {X_50mol.shape}, y range=[{y_50mol.min():.2f},{y_50mol.max():.2f}]")
    logger.info(f"Embedding norm: mean={np.linalg.norm(X_50mol, axis=1).mean():.4f}")

    np.save(OUTPUT_DIR / "X_50mol_128.npy", X_50mol)
    np.save(OUTPUT_DIR / "y_pkd_50mol.npy", y_50mol)
    with open(OUTPUT_DIR / "families_50mol.json", "w") as f:
        json.dump(fam_m, f)
    np.save(OUTPUT_DIR / "mol_counts_50mol.npy", mol_counts[idx_m])

    # 7. Merge with tier3 (replace matching pockets with better 50mol embeddings)
    logger.info("=" * 60)
    logger.info("Phase 3: Merging with tier3 data")

    if (TIER3_DIR / "X_encoder_128.npy").exists():
        X_t3 = np.load(TIER3_DIR / "X_encoder_128.npy")
        y_t3 = np.load(TIER3_DIR / "y_pkd_encoder.npy")
        with open(TIER3_DIR / "families_encoder.json") as f:
            fam_t3 = json.load(f)

        X_combined = X_t3.copy()
        y_combined = y_t3.copy()
        fam_combined = list(fam_t3)
        fam_t3_set = set(fam_t3)
        updated, added = 0, 0

        for i, fam in enumerate(fam_m):
            if fam in fam_t3_set:
                idx = fam_t3.index(fam)
                X_combined[idx] = X_50mol[i]
                updated += 1
            else:
                X_combined = np.vstack([X_combined, X_50mol[i:i+1]])
                y_combined = np.append(y_combined, y_50mol[i])
                fam_combined.append(fam)
                added += 1

        logger.info(f"Tier3 merge: {updated} updated, {added} added, total={len(fam_combined)}")

        np.save(OUTPUT_DIR / "X_combined_128.npy", X_combined)
        np.save(OUTPUT_DIR / "y_pkd_combined.npy", y_combined)
        with open(OUTPUT_DIR / "families_combined.json", "w") as f:
            json.dump(fam_combined, f)
    else:
        logger.warning("Tier3 data not found, using 50mol only as combined")
        np.save(OUTPUT_DIR / "X_combined_128.npy", X_50mol)
        np.save(OUTPUT_DIR / "y_pkd_combined.npy", y_50mol)
        with open(OUTPUT_DIR / "families_combined.json", "w") as f:
            json.dump(fam_m, f)

    total_time = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Files saved to {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
