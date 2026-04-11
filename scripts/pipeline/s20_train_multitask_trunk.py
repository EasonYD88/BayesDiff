"""Train multi-task trunk shaping module and connect to best SP4 oracle head.

Two-stage pipeline:
  Stage 1: Train MultiTaskTrunk (reg + cls [+ rank]) on frozen embeddings
  Stage 2: Freeze trunk, extract h_trunk, fit DKL Ensemble on top
  Stage 3: Evaluate on CASF-2016 test set via OracleHead.evaluate()

Usage:
    # A5.0: True SP4 baseline (no trunk)
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings_augmented.npz \
        --output results/stage2/multitask_trunk/A5.0_no_trunk \
        --phase v1 --no_trunk --seed 42 --device cuda

    # A5.1: Regression-only trunk
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings_augmented.npz \
        --output results/stage2/multitask_trunk/A5.1_reg_only \
        --phase v1 --lambda_cls 0.0 --seed 42 --device cuda

    # A5.2: Reg + Cls trunk
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings_augmented.npz \
        --output results/stage2/multitask_trunk/A5.2_reg_cls \
        --phase v1 --lambda_cls 0.5 --seed 42 --device cuda

    # A5.4: Reg + Cls + Rank (v2, needs groups)
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings_augmented.npz \
        --output results/stage2/multitask_trunk/A5.4_reg_cls_rank \
        --phase v2 --lambda_cls 0.5 --lambda_rank 0.3 --seed 42 --device cuda
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from bayesdiff.multi_task import MultiTaskTrunk, MultiTaskHybridOracle
from bayesdiff.hybrid_oracle import DKLEnsembleOracle

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    seed_everything(args.seed)
    log.info(f"Seed: {args.seed}")

    # ── 0. Load data ──────────────────────────────────────────────
    log.info(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    log.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Load cluster assignments for v2 ranking
    groups_train = None
    groups_val = None
    if args.phase == "v2":
        if "groups_train" in data and "groups_val" in data:
            groups_train = data["groups_train"].astype(np.int64)
            groups_val = data["groups_val"].astype(np.int64)
            log.info(
                f"  Loaded groups from NPZ — "
                f"train coverage: {(groups_train >= 0).mean():.1%}, "
                f"val coverage: {(groups_val >= 0).mean():.1%}"
            )
        else:
            raise RuntimeError(
                "Phase v2 requires 'groups_train' and 'groups_val' in the NPZ. "
                "Run s20_augment_npz.py first."
            )

    # Class balance check
    pos_rate = float((y_train >= args.threshold).mean())
    log.info(f"  Classification positive rate (tau={args.threshold}): {pos_rate:.3f}")
    cls_pos_weight = None
    if pos_rate < 0.25:
        cls_pos_weight = (1 - pos_rate) / pos_rate
        log.info(f"  Using pos_weight={cls_pos_weight:.2f} for imbalanced BCE")

    # ── Output dir ────────────────────────────────────────────────
    output_dir = Path(args.output) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── A5.0: Skip trunk, fit oracle directly ─────────────────────
    if args.no_trunk:
        log.info("--no_trunk: skipping Stage 1, fitting DKL Ensemble directly")
        oracle_head = DKLEnsembleOracle(
            input_dim=X_train.shape[1],
            n_members=args.n_members,
            bootstrap=True,
            feature_dim=args.feature_dim,
            n_inducing=args.n_inducing,
            residual=True,
            device=args.device,
        )
        seed_everything(args.seed)
        oracle_head.fit(X_train, y_train, X_val, y_val, seed_base=args.seed)

        # Evaluate via OracleHead.evaluate() — no p_success needed
        log.info("Evaluating on CASF-2016 test set (no trunk)...")
        metrics = oracle_head.evaluate(X_test, y_test, y_target=args.threshold)
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        oracle_head.save(str(output_dir / "oracle_head"))
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        log.info(f"Saved to {output_dir}")
        return

    # ── 1. Create trunk ───────────────────────────────────────────
    trunk = MultiTaskTrunk(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        trunk_dim=args.trunk_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        residual=True,
        activity_threshold=args.threshold,
        enable_ranking=(args.phase == "v2"),
        learned_weights=args.learned_weights,
        cls_pos_weight=cls_pos_weight,
    )
    log.info(f"MultiTaskTrunk: {sum(p.numel() for p in trunk.parameters())} params")

    # ── 2. Create oracle head ─────────────────────────────────────
    oracle_head = DKLEnsembleOracle(
        input_dim=args.trunk_dim,
        n_members=args.n_members,
        bootstrap=True,
        feature_dim=args.feature_dim,
        n_inducing=args.n_inducing,
        residual=True,
        device=args.device,
    )

    hybrid = MultiTaskHybridOracle(trunk, oracle_head)

    # ── 3. Stage 1: Train trunk ───────────────────────────────────
    log.info("Stage 1: Training multi-task trunk...")
    history = hybrid.train_trunk(
        X_train,
        y_train,
        X_val,
        y_val,
        groups_train=groups_train,
        groups_val=groups_val,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
        lambda_rank=args.lambda_rank,
        patience=args.patience,
        device=args.device,
    )
    log.info(f"  Best epoch: {history['best_epoch']}")
    log.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    log.info(f"  Best val loss: {min(history['val_loss']):.4f}")

    # ── 4. Stage 2: Fit oracle head on shaped features ────────────
    log.info("Stage 2: Fitting DKL Ensemble on shaped trunk features...")
    seed_everything(args.seed)
    oracle_history = hybrid.train_oracle(
        X_train, y_train, X_val, y_val, seed_base=args.seed
    )

    # ── 5. Evaluate on CASF-2016 (Tier 1 metrics only) ───────────
    log.info("Evaluating on CASF-2016 test set...")
    # Use OracleHead.evaluate() — returns R2, spearman_rho, rmse, nll, err_sigma_rho
    # Does NOT need p_success (deferred to post-Tier 1)
    h_test = hybrid.multi_task.extract_trunk_features(X_test)
    metrics = oracle_head.evaluate(h_test, y_test, y_target=args.threshold)
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── 6. Save ───────────────────────────────────────────────────
    hybrid.save(str(output_dir))

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {
                k: (
                    v
                    if not isinstance(v, list)
                    else [float(x) for x in v]
                )
                for k, v in history.items()
            },
            f,
            indent=2,
        )

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SP05: Train multi-task trunk")
    # Data paths
    parser.add_argument(
        "--embeddings",
        default="results/stage2/oracle_heads/frozen_embeddings_augmented.npz",
    )
    parser.add_argument("--output", required=True)

    # Phase
    parser.add_argument("--phase", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--no_trunk",
        action="store_true",
        help="Skip trunk, fit oracle directly on frozen embeddings (A5.0 baseline)",
    )

    # Trunk architecture
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--trunk_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Task weights
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lambda_rank", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--learned_weights", action="store_true")

    # Training
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    # Oracle head
    parser.add_argument("--n_members", type=int, default=5)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--n_inducing", type=int, default=512)

    # Device and seed
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (3-seed protocol: 42, 123, 777)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main(args)
