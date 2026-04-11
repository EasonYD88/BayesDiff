"""Augment frozen_embeddings.npz with PDB codes and cluster group arrays.

The frozen NPZ only has X_train/val/test and y_train/val/test arrays.
For SP05 ranking (v2), we need per-sample group assignments aligned with
the NPZ row order.  This script adds:
  - codes_train, codes_val, codes_test  (string arrays of PDB codes)
  - groups_train, groups_val            (int arrays of cluster_id, -1 if missing)

Usage:
    python scripts/pipeline/s20_augment_npz.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --splits data/pdbbind_v2020/splits.json \
        --clusters data/pdbbind_v2020/cluster_assignments.csv \
        --output results/stage2/oracle_heads/frozen_embeddings_augmented.npz
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def main(args):
    log.info(f"Loading splits from {args.splits}")
    with open(args.splits) as f:
        splits = json.load(f)
    codes_train = np.array(splits["train"])
    codes_val = np.array(splits["val"])
    codes_test = np.array(splits["test"])

    log.info(f"Loading embeddings from {args.embeddings}")
    data = dict(np.load(args.embeddings))
    assert len(codes_train) == data["X_train"].shape[0], (
        f"Train size mismatch: splits has {len(codes_train)}, "
        f"NPZ has {data['X_train'].shape[0]}"
    )
    assert len(codes_val) == data["X_val"].shape[0], (
        f"Val size mismatch: splits has {len(codes_val)}, "
        f"NPZ has {data['X_val'].shape[0]}"
    )
    assert len(codes_test) == data["X_test"].shape[0], (
        f"Test size mismatch: splits has {len(codes_test)}, "
        f"NPZ has {data['X_test'].shape[0]}"
    )

    log.info(f"Loading clusters from {args.clusters}")
    clusters = pd.read_csv(args.clusters)
    code_to_cluster = dict(zip(clusters["pdb_code"], clusters["cluster_id"]))

    groups_train = np.array(
        [code_to_cluster.get(c, -1) for c in codes_train], dtype=np.int64
    )
    groups_val = np.array(
        [code_to_cluster.get(c, -1) for c in codes_val], dtype=np.int64
    )

    coverage_train = (groups_train >= 0).mean()
    coverage_val = (groups_val >= 0).mean()
    log.info(f"Cluster coverage — train: {coverage_train:.1%}, val: {coverage_val:.1%}")

    data["codes_train"] = codes_train
    data["codes_val"] = codes_val
    data["codes_test"] = codes_test
    data["groups_train"] = groups_train
    data["groups_val"] = groups_val

    log.info(f"Saving augmented NPZ to {args.output}")
    np.savez(args.output, **data)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        default="results/stage2/oracle_heads/frozen_embeddings.npz",
    )
    parser.add_argument("--splits", default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--clusters", default="data/pdbbind_v2020/cluster_assignments.csv")
    parser.add_argument(
        "--output",
        default="results/stage2/oracle_heads/frozen_embeddings_augmented.npz",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(args)
