"""
scripts/04_train_gp.py
──────────────────────
Train the SVGP oracle on pre-computed embeddings + pKd labels.

Usage (Mac debug):
    python scripts/04_train_gp.py \
        --embeddings results/generated_molecules/all_embeddings.npz \
        --labels data/splits/labels.csv \
        --output results/gp_model \
        --n_inducing 128 --n_epochs 100

Usage (full):
    python scripts/04_train_gp.py \
        --embeddings data/embeddings/casf_test.npz \
        --labels data/splits/labels.csv \
        --output results/gp_model \
        --n_inducing 512 --n_epochs 300

The script can also train on affinity_info.pkl from the TargetDiff data.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_labels_csv(labels_csv: Path) -> dict[str, float]:
    """Load pdb_code -> pKd mapping from labels.csv."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    return dict(zip(df["pdb_code"], df["pkd"]))


def load_affinity_pkl(pkl_path: Path) -> dict[str, float]:
    """Load pdb_code -> pKd from TargetDiff's affinity_info.pkl."""
    with open(pkl_path, "rb") as f:
        affinity = pickle.load(f)
    label_map = {}
    for key, info in affinity.items():
        pk = info.get("neglog_aff") or info.get("pk") or info.get("pkd")
        if pk is not None:
            # Extract PDB code from the key (e.g., "1a2b_0" -> "1a2b")
            pdb_code = str(key).split("_")[0] if "_" in str(key) else str(key)
            label_map[pdb_code] = float(pk)
    return label_map


def build_training_set(
    embeddings_dict: dict[str, np.ndarray],
    label_map: dict[str, float],
    use_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y) from embeddings + labels.

    Parameters
    ----------
    embeddings_dict : {pdb_code: (M, d) or (d,) array}
    label_map : {pdb_code: pKd}
    use_mean : if True, take mean over M samples; else use all samples
    """
    X_list, y_list, codes = [], [], []

    for pdb_code, emb in embeddings_dict.items():
        pdb_base = pdb_code.split("_")[0] if "_" in pdb_code else pdb_code
        pk = label_map.get(pdb_base) or label_map.get(pdb_code)
        if pk is None:
            continue

        if emb.ndim == 1:
            X_list.append(emb)
            y_list.append(pk)
            codes.append(pdb_code)
        elif use_mean:
            X_list.append(emb.mean(axis=0))
            y_list.append(pk)
            codes.append(pdb_code)
        else:
            for m in range(emb.shape[0]):
                X_list.append(emb[m])
                y_list.append(pk)
                codes.append(f"{pdb_code}_s{m}")

    if not X_list:
        return np.empty((0, 0)), np.empty(0), []

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, codes


def augment_data(
    X: np.ndarray, y: np.ndarray, target_n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Augment small datasets by adding Gaussian noise."""
    if len(X) >= target_n:
        return X, y

    rng = np.random.default_rng(seed)
    n_aug = target_n - len(X)
    d = X.shape[1]

    X_aug = [X]
    y_aug = [y]
    for _ in range(n_aug):
        idx = rng.integers(len(X))
        x_new = X[idx] + rng.standard_normal(d).astype(np.float32) * 0.3
        y_new = y[idx] + rng.standard_normal() * 0.5
        X_aug.append(x_new.reshape(1, -1))
        y_aug.append(np.array([y_new]))

    return np.concatenate(X_aug), np.concatenate(y_aug)


def main():
    parser = argparse.ArgumentParser(description="Train SVGP Oracle")
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to embeddings .npz file",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to labels.csv (pdb_code, pkd columns)",
    )
    parser.add_argument(
        "--affinity_pkl", type=str, default=None,
        help="Path to affinity_info.pkl (fallback label source)",
    )
    parser.add_argument(
        "--output", type=str, default="results/gp_model",
        help="Output directory for trained GP model",
    )
    parser.add_argument("--n_inducing", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--augment_to", type=int, default=0,
                        help="Augment to this many samples if dataset is small (0=off)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ──────────────────────────────────────────
    emb_path = Path(args.embeddings)
    logger.info(f"Loading embeddings from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    embeddings_dict = {k: data[k] for k in data.files}
    logger.info(f"  Loaded {len(embeddings_dict)} entries")

    # ── Load labels ──────────────────────────────────────────────
    label_map = {}
    if args.labels and Path(args.labels).exists():
        label_map = load_labels_csv(Path(args.labels))
        logger.info(f"  Labels from CSV: {len(label_map)} entries")

    if not label_map and args.affinity_pkl and Path(args.affinity_pkl).exists():
        label_map = load_affinity_pkl(Path(args.affinity_pkl))
        logger.info(f"  Labels from affinity_pkl: {len(label_map)} entries")

    if not label_map:
        # Try auto-detect affinity_info.pkl
        default_pkl = PROJECT_ROOT / "external" / "targetdiff" / "data" / "affinity_info.pkl"
        if default_pkl.exists():
            label_map = load_affinity_pkl(default_pkl)
            logger.info(f"  Labels from auto-detected affinity_info.pkl: {len(label_map)} entries")

    if not label_map:
        logger.error("No labels found. Provide --labels or --affinity_pkl")
        sys.exit(1)

    # ── Build training set ───────────────────────────────────────
    X, y, codes = build_training_set(embeddings_dict, label_map, use_mean=True)
    logger.info(f"  Training set: {X.shape[0]} samples, d={X.shape[1]}")

    if X.shape[0] == 0:
        logger.error("No matching embeddings+labels found")
        sys.exit(1)

    if X.shape[0] < 10:
        logger.warning(f"Very small training set ({X.shape[0]}). Consider augmentation.")

    # Augment if requested
    if args.augment_to > 0 and X.shape[0] < args.augment_to:
        logger.info(f"  Augmenting from {X.shape[0]} to {args.augment_to} samples")
        X, y = augment_data(X, y, target_n=args.augment_to, seed=args.seed)
        logger.info(f"  After augmentation: {X.shape[0]} samples")

    logger.info(f"  pKd range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

    # ── Train GP ─────────────────────────────────────────────────
    from bayesdiff.gp_oracle import GPOracle

    d = X.shape[1]
    n_inducing = min(args.n_inducing, X.shape[0])

    logger.info(f"\nTraining SVGP (d={d}, J={n_inducing}, epochs={args.n_epochs})...")
    gp = GPOracle(d=d, n_inducing=n_inducing, device="cpu")

    t0 = time.time()
    history = gp.train(
        X, y,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    elapsed = time.time() - t0
    logger.info(f"  Training done in {elapsed:.1f}s, final loss={history['loss'][-1]:.4f}")

    # ── Save model ───────────────────────────────────────────────
    model_path = output_dir / "gp_model.pt"
    gp.save(model_path)
    logger.info(f"  Model saved to {model_path}")

    # Save training metadata
    meta = {
        "n_train": int(X.shape[0]),
        "d": int(d),
        "n_inducing": int(n_inducing),
        "n_epochs": int(args.n_epochs),
        "final_loss": float(history["loss"][-1]),
        "elapsed_s": round(elapsed, 1),
        "pkd_range": [float(y.min()), float(y.max())],
        "pkd_mean": float(y.mean()),
        "train_codes": codes[:50],  # First 50 for reference
    }
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save training data for OOD detector later
    np.savez(output_dir / "train_data.npz", X=X, y=y)
    logger.info(f"  Training data saved to {output_dir / 'train_data.npz'}")

    logger.info(f"\n{'='*60}")
    logger.info("GP training complete!")
    logger.info(f"  Model:    {model_path}")
    logger.info(f"  Metadata: {output_dir / 'train_meta.json'}")
    logger.info(f"{'='*60}")
    logger.info("\nNext: python scripts/05_evaluate.py \\")
    logger.info(f"        --embeddings {args.embeddings} \\")
    logger.info(f"        --gp_model {model_path} \\")
    logger.info(f"        --labels {args.labels or args.affinity_pkl or '<labels>'}")


if __name__ == "__main__":
    main()
