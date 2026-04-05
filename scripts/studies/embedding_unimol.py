#!/usr/bin/env python3
"""Phase B: Extract Uni-Mol pre-trained embeddings for all pockets.

Uses unimol_tools (pre-trained on 209M molecular conformations) to extract
512-dim CLS embeddings for each generated molecule, then mean-pools per pocket.

Output: results/tier3_gp/X_unimol_512.npy, y_pkd_unimol.npy, families_unimol.json
"""

import sys
import json
import logging
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = REPO / "results" / "tier3_gp"
SAMPLING_DIR = REPO / "results" / "tier3_sampling"


def extract_unimol_embeddings():
    """Extract Uni-Mol 512-dim embeddings for all pockets."""
    from rdkit import Chem

    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        logger.error("unimol_tools not installed. Run: pip install unimol_tools")
        sys.exit(1)

    # Initialize Uni-Mol representation model
    logger.info("Loading Uni-Mol pre-trained model...")
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    logger.info("Uni-Mol model loaded successfully")

    # Load pocket list
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
            # Read SMILES from SDF
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
            smiles_list = []
            for mol in supplier:
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_list.append(smi)

            if len(smiles_list) == 0:
                n_fail += 1
                continue

            # Get Uni-Mol representations (batch)
            # get_repr returns list of numpy arrays, each (512,)
            reprs = clf.get_repr(smiles_list)
            cls_reprs = np.stack(reprs)  # (n_mols, 512)

            # Mean pool over molecules
            pocket_emb = cls_reprs.mean(axis=0)  # (512,)
            embeddings.append(pocket_emb)
            families.append(family)
            pkd_values.append(family_to_pkd[family])
            n_ok += 1

            if (idx + 1) % 50 == 0 or idx == 0:
                logger.info(f"[{idx+1}/{len(pocket_dirs)}] {family}: "
                            f"{len(smiles_list)} mols → {cls_reprs.shape}")

        except Exception as e:
            logger.error(f"[{idx+1}] {family}: FAILED - {e}")
            n_fail += 1

    elapsed = time.time() - t_start
    logger.info(f"\nDONE: {n_ok} OK, {n_fail} failed, {elapsed:.0f}s")

    X = np.stack(embeddings)  # (N, 512)
    y = np.array(pkd_values)
    logger.info(f"Final: X={X.shape}, y={y.shape}")

    np.save(DATA_DIR / "X_unimol_512.npy", X)
    np.save(DATA_DIR / "y_pkd_unimol.npy", y)
    with open(DATA_DIR / "families_unimol.json", "w") as f:
        json.dump(families, f)

    logger.info(f"Saved to {DATA_DIR}")
    return X, y, families


if __name__ == "__main__":
    extract_unimol_embeddings()
