#!/usr/bin/env python3
"""Phase 0: Merge 50mol embedding shards and prepare datasets.

Collects embeddings from 31 shards (handles both all_embeddings.npz and
per-pocket _embeddings.npy fallback). Builds:
  1. 50mol standalone dataset (X_50mol_128, y_pkd_50mol)
  2. Combined dataset (tier3 base + 50mol updates for overlapping pockets)
  3. Per-pocket embeddings for U_gen computation
"""

import json
import glob
import pickle
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
EMB_DIR = REPO / "results" / "embedding_50mol"
TIER3_DIR = REPO / "results" / "tier3_gp"
OUTPUT_DIR = REPO / "results" / "50mol_gp"
AFF_PKL = REPO / "external" / "targetdiff" / "data" / "affinity_info.pkl"


def load_affinity():
    """Load pKd labels from affinity_info.pkl, grouped by pocket family."""
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


def collect_shards():
    """Collect all per-pocket embeddings from 31 shards."""
    all_embeddings = {}
    shard_dirs = sorted(glob.glob(str(EMB_DIR / "*/shards/shard_*of31")))
    logger.info(f"Found {len(shard_dirs)} shard directories")

    for sd in shard_dirs:
        sd = Path(sd)
        npz = sd / "all_embeddings.npz"

        if npz.exists():
            data = np.load(npz)
            for k in data.files:
                all_embeddings[k] = data[k]
            logger.info(f"  {sd.name}: {len(data.files)} pockets from npz")
        else:
            # Fallback: load per-pocket npy files
            count = 0
            for pocket_dir in sorted(sd.iterdir()):
                if not pocket_dir.is_dir():
                    continue
                npy = pocket_dir / f"{pocket_dir.name}_embeddings.npy"
                if npy.exists():
                    all_embeddings[pocket_dir.name] = np.load(npy)
                    count += 1
            logger.info(f"  {sd.name}: {count} pockets from npy (no npz)")

    return all_embeddings


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect all shard embeddings
    logger.info("=" * 60)
    logger.info("Phase 0: Merging 50mol shards")
    logger.info("=" * 60)
    all_emb = collect_shards()
    logger.info(f"Total pockets collected: {len(all_emb)}")

    # Verify shapes
    shapes = {k: v.shape for k, v in all_emb.items()}
    dims = set(s[1] for s in shapes.values())
    logger.info(f"Embedding dimensions found: {dims}")
    mol_counts_all = {k: v.shape[0] for k, v in all_emb.items()}
    logger.info(f"Molecules per pocket: min={min(mol_counts_all.values())}, "
                f"max={max(mol_counts_all.values())}, "
                f"mean={np.mean(list(mol_counts_all.values())):.1f}")

    # 2. Mean pool per pocket → (N, 128)
    families_all = sorted(all_emb.keys())
    X_all = np.stack([all_emb[f].mean(axis=0) for f in families_all])
    mol_counts = np.array([all_emb[f].shape[0] for f in families_all])
    logger.info(f"Mean-pooled: {X_all.shape}")

    # 3. Match with pKd labels
    pk_map = load_affinity()
    matched = [(i, f) for i, f in enumerate(families_all) if f in pk_map]
    idx_matched = [m[0] for m in matched]
    families_matched = [m[1] for m in matched]

    X_50mol = X_all[idx_matched]
    y_50mol = np.array([pk_map[f] for f in families_matched])
    mc_50mol = mol_counts[idx_matched]

    logger.info(f"Matched with pKd: {len(families_matched)} / {len(families_all)}")
    logger.info(f"X_50mol: {X_50mol.shape}, y range: [{y_50mol.min():.2f}, {y_50mol.max():.2f}]")
    logger.info(f"y mean: {y_50mol.mean():.2f}, std: {y_50mol.std():.2f}")

    # Save 50mol standalone
    np.save(OUTPUT_DIR / "X_50mol_128.npy", X_50mol)
    np.save(OUTPUT_DIR / "y_pkd_50mol.npy", y_50mol)
    with open(OUTPUT_DIR / "families_50mol.json", "w") as f:
        json.dump(families_matched, f)
    np.save(OUTPUT_DIR / "mol_counts_50mol.npy", mc_50mol)

    # Save per-pocket embeddings (for U_gen)
    per_pocket = {f: all_emb[f] for f in families_matched}
    np.savez(OUTPUT_DIR / "per_pocket_embeddings.npz", **per_pocket)
    logger.info(f"Saved per-pocket embeddings for {len(per_pocket)} pockets")

    # 4. Merge with tier3
    t3_enc = TIER3_DIR / "X_encoder_128.npy"
    if t3_enc.exists():
        X_t3 = np.load(TIER3_DIR / "X_encoder_128.npy")
        y_t3 = np.load(TIER3_DIR / "y_pkd_encoder.npy")
        with open(TIER3_DIR / "families_encoder.json") as f:
            fam_t3 = json.load(f)

        logger.info(f"Tier3 encoder data: {X_t3.shape}, {len(fam_t3)} families")

        # Replace tier3 embeddings with 50mol where available
        X_combined = X_t3.copy()
        y_combined = y_t3.copy()
        fam_combined = list(fam_t3)
        fam_t3_set = set(fam_t3)

        updated, added = 0, 0
        for i, fam in enumerate(families_matched):
            if fam in fam_t3_set:
                idx = fam_t3.index(fam)
                X_combined[idx] = X_50mol[i]
                updated += 1
            else:
                X_combined = np.vstack([X_combined, X_50mol[i:i + 1]])
                y_combined = np.append(y_combined, y_50mol[i])
                fam_combined.append(fam)
                added += 1

        logger.info(f"Tier3 merge: {updated} updated, {added} new, total={len(fam_combined)}")

        np.save(OUTPUT_DIR / "X_combined_128.npy", X_combined)
        np.save(OUTPUT_DIR / "y_pkd_combined.npy", y_combined)
        with open(OUTPUT_DIR / "families_combined.json", "w") as f:
            json.dump(fam_combined, f)
    else:
        logger.warning("Tier3 encoder data not found, skipping combined dataset")
        # Use 50mol as combined
        np.save(OUTPUT_DIR / "X_combined_128.npy", X_50mol)
        np.save(OUTPUT_DIR / "y_pkd_combined.npy", y_50mol)
        with open(OUTPUT_DIR / "families_combined.json", "w") as f:
            json.dump(families_matched, f)

    # 5. Summary
    logger.info("=" * 60)
    logger.info("Merge complete. Output files:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        sz = p.stat().st_size
        logger.info(f"  {p.name:40s}  {sz / 1024:.1f} KB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
