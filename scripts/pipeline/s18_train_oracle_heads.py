"""
scripts/pipeline/s18_train_oracle_heads.py
──────────────────────────────────────────
Sub-Plan 4: Train and evaluate oracle heads on frozen embeddings.

Phase 4.1: Extract frozen embeddings (if not cached)
Phase 4.2: Train all oracle heads
Phase 4.3: Unified evaluation + comparison table

Usage:
    # Train all Tier 1 oracle heads:
    python scripts/pipeline/s18_train_oracle_heads.py \\
        --frozen_embeddings results/stage2/frozen_embeddings.npz \\
        --output results/stage2/oracle_heads \\
        --heads dkl,dkl_ensemble,nn_residual,svgp,pca_svgp \\
        --device cuda \\
        --seed 42

    # Train a single head for debugging:
    python scripts/pipeline/s18_train_oracle_heads.py \\
        --frozen_embeddings results/stage2/frozen_embeddings.npz \\
        --output results/stage2/oracle_heads \\
        --heads dkl \\
        --device cpu

    # Extract frozen embeddings first (requires model checkpoint + data):
    python scripts/pipeline/s18_train_oracle_heads.py \\
        --extract_embeddings \\
        --schemeb_checkpoint results/stage2/ablation_viz/A36_independent_model.pt \\
        --model_type independent \\
        --atom_emb_dir results/atom_embeddings \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/oracle_heads \\
        --device cuda

Output:
    results/stage2/oracle_heads/
        frozen_embeddings.npz
        dkl/dkl_model.pt
        dkl_ensemble/member_0/..., ensemble_config.json
        nn_residual/nn_model.pt, gp_model.pt
        svgp/gp_model.pt
        pca_svgp/pca.pkl, gp_model.pt
        tier1_comparison.json
        tier1_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# GPOracle wrapper (to conform to OracleHead interface)
# ============================================================================


class GPOracleWrapper:
    """Thin wrapper around GPOracle to match OracleHead interface for comparison."""

    def __init__(self, gp):
        self.gp = gp

    def fit(self, X_train, y_train, X_val, y_val, n_epochs=200, batch_size=256, lr=0.01, verbose=True, **kwargs):
        return self.gp.train(X_train, y_train, n_epochs=n_epochs, batch_size=batch_size, lr=lr, verbose=verbose)

    def predict(self, X):
        from bayesdiff.oracle_interface import OracleResult
        mu, var = self.gp.predict(X)
        return OracleResult(mu=mu, sigma2=var, aux={})

    def predict_for_fusion(self, X):
        from bayesdiff.oracle_interface import OracleResult
        mu, var, J = self.gp.predict_with_jacobian(X)
        return OracleResult(mu=mu, sigma2=var, jacobian=J, aux={})

    def evaluate(self, X, y, y_target=7.0):
        from bayesdiff.evaluate import gaussian_nll
        result = self.predict(X)
        sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
        ss_res = np.sum((y - result.mu) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rho, _ = spearmanr(result.mu, y)
        rmse = np.sqrt(np.mean((y - result.mu) ** 2))
        nll = gaussian_nll(result.mu, sigma, y)
        errors = np.abs(y - result.mu)
        err_sigma_rho, err_sigma_p = spearmanr(errors, sigma)
        return {
            "R2": float(r2), "spearman_rho": float(rho), "rmse": float(rmse),
            "nll": float(nll), "err_sigma_rho": float(err_sigma_rho),
            "err_sigma_p": float(err_sigma_p), "mean_sigma": float(sigma.mean()),
        }

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.gp.save(Path(path) / "gp_model.pt")


# ============================================================================
# Embedding extraction (Phase A.3)
# ============================================================================


@torch.no_grad()
def extract_frozen_embeddings(
    checkpoint_path: str,
    atom_emb_dir: str,
    labels_path: str,
    splits_path: str,
    output_path: str,
    model_type: str = "independent",
    embed_dim: int = 128,
    attn_hidden_dim: int = 64,
    entropy_weight: float = 0.01,
    batch_size: int = 64,
    device: str = "cuda",
) -> str:
    """Extract and cache frozen embeddings from SchemeB model.

    Returns path to the saved .npz file.
    """
    import pandas as pd
    from bayesdiff.attention_pool import SchemeB_Independent, SchemeB_SingleBranch
    from scripts.pipeline.s12_train_attn_pool import AtomEmbeddingDataset, collate_atom_emb

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info(f"Loading SchemeB ({model_type}) checkpoint from {checkpoint_path}")
    if model_type == "independent":
        model = SchemeB_Independent(
            embed_dim=embed_dim, n_layers=10,
            attn_hidden_dim=attn_hidden_dim, entropy_weight=entropy_weight,
        ).to(device_t)
    else:
        model = SchemeB_SingleBranch(
            embed_dim=embed_dim, n_layers=10,
            attn_hidden_dim=attn_hidden_dim, entropy_weight=entropy_weight,
        ).to(device_t)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device_t, weights_only=False))
    model.eval()

    # Load labels and splits
    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))
    with open(splits_path) as f:
        splits = json.load(f)

    def extract_split(split_codes, desc):
        ds = AtomEmbeddingDataset(atom_emb_dir, split_codes, label_map)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_atom_emb, num_workers=0, pin_memory=True,
        )
        all_z, all_y = [], []
        for batch in loader:
            layers = [l.to(device_t) for l in batch["layer_embs"]]
            mask = batch["mask"].to(device_t)
            z_global, _ = model(layers, atom_mask=mask)
            all_z.append(z_global.cpu().numpy())
            all_y.append(batch["pkd"].numpy())
        X = np.concatenate(all_z, axis=0)
        y = np.concatenate(all_y, axis=0)
        logger.info(f"  {desc}: {X.shape}")
        return X.astype(np.float32), y.astype(np.float32)

    X_train, y_train = extract_split(splits["train"], "train")
    X_val, y_val = extract_split(splits["val"], "val")
    X_test, y_test = extract_split(splits["test"], "test")

    output_file = output_path
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    logger.info(f"Frozen embeddings saved to {output_file}")
    return output_file


# ============================================================================
# Head registry
# ============================================================================


def build_head_registry(args):
    """Build registry of oracle head constructors."""
    from bayesdiff.hybrid_oracle import (
        DKLOracle, DKLEnsembleOracle, NNResidualOracle, PCA_GPOracle,
        SNGPOracle, EvidentialOracle,
    )
    from bayesdiff.gp_oracle import GPOracle

    return {
        "dkl": lambda: DKLOracle(
            input_dim=128, feature_dim=args.feature_dim, n_inducing=args.n_inducing,
            hidden_dim=args.hidden_dim, n_layers=args.dkl_n_layers,
            residual=bool(args.residual), dropout=args.dropout, device=args.device,
        ),
        "dkl_ensemble": lambda: DKLEnsembleOracle(
            input_dim=128, n_members=args.ensemble_members,
            bootstrap=bool(args.bootstrap),
            feature_dim=args.feature_dim, n_inducing=args.n_inducing,
            hidden_dim=args.hidden_dim, n_layers=args.dkl_n_layers,
            residual=bool(args.residual), dropout=args.dropout, device=args.device,
        ),
        "nn_residual": lambda: NNResidualOracle(
            input_dim=128, hidden_dim=args.hidden_dim, n_inducing=args.n_inducing,
            mc_dropout=bool(args.mc_dropout), mc_samples=args.mc_samples,
            dropout=args.dropout, device=args.device,
        ),
        "svgp": lambda: GPOracleWrapper(
            GPOracle(d=128, n_inducing=args.n_inducing, device=args.device)
        ),
        "pca_svgp": lambda: PCA_GPOracle(
            input_dim=128, pca_dim=32, n_inducing=args.n_inducing,
            device=args.device,
        ),
        "sngp": lambda: SNGPOracle(
            input_dim=128, hidden_dim=args.hidden_dim, n_layers=args.dkl_n_layers,
            n_rff=args.n_rff, dropout=args.dropout, device=args.device,
        ),
        "evidential": lambda: EvidentialOracle(
            input_dim=128, hidden_dim=args.hidden_dim,
            dropout=args.dropout, device=args.device,
        ),
    }


def _print_summary_table(results: dict):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 90)
    print(f"{'Head':<18} {'R²':>6} {'ρ':>6} {'RMSE':>6} {'NLL':>6} {'ρ_err_σ':>8} {'Time':>8}")
    print("-" * 90)
    for name, res in results.items():
        t = res["test"]
        elapsed = res["elapsed_seconds"]
        print(
            f"{name:<18} {t['R2']:>6.4f} {t['spearman_rho']:>6.4f} "
            f"{t['rmse']:>6.3f} {t['nll']:>6.3f} {t['err_sigma_rho']:>8.4f} "
            f"{elapsed:>7.1f}s"
        )
    print("=" * 90)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Sub-Plan 4: Oracle Head Comparison")
    parser.add_argument("--frozen_embeddings", type=str, default="results/stage2/frozen_embeddings.npz")
    parser.add_argument("--output", type=str, default="results/stage2/oracle_heads")
    parser.add_argument("--heads", type=str, default="dkl,dkl_ensemble,nn_residual,svgp,pca_svgp")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--ensemble_members", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    # Tier 2: DKL hyperparameter overrides
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dkl_n_layers", type=int, default=2)
    parser.add_argument("--residual", type=int, default=1, help="1=True, 0=False")
    parser.add_argument("--bootstrap", type=int, default=1, help="1=True, 0=False (ensemble)")
    parser.add_argument("--mc_dropout", type=int, default=1, help="1=True, 0=False (nn_residual)")
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_rff", type=int, default=1024, help="Number of random Fourier features (SNGP)")
    # Embedding extraction
    parser.add_argument("--extract_embeddings", action="store_true")
    parser.add_argument("--schemeb_checkpoint", type=str, default="results/stage2/ablation_viz/A36_independent_model.pt")
    parser.add_argument("--model_type", type=str, default="independent", choices=["independent", "shared"])
    parser.add_argument("--atom_emb_dir", type=str, default="results/atom_embeddings")
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Phase A.3: Extract embeddings if requested
    if args.extract_embeddings:
        emb_path = str(Path(args.output) / "frozen_embeddings.npz")
        extract_frozen_embeddings(
            checkpoint_path=args.schemeb_checkpoint,
            atom_emb_dir=args.atom_emb_dir,
            labels_path=args.labels,
            splits_path=args.splits,
            output_path=emb_path,
            model_type=args.model_type,
            device=args.device,
        )
        args.frozen_embeddings = emb_path

    # Early exit for extraction-only mode
    if args.heads.strip().lower() == "none":
        logger.info("Extraction-only mode (--heads none). Exiting.")
        return

    # Load frozen embeddings
    emb_path = args.frozen_embeddings
    if not Path(emb_path).exists():
        logger.error(f"Frozen embeddings not found at {emb_path}. Use --extract_embeddings to create them.")
        sys.exit(1)

    data = np.load(emb_path)
    X_train, y_train = data["X_train"].astype(np.float32), data["y_train"].astype(np.float32)
    X_val, y_val = data["X_val"].astype(np.float32), data["y_val"].astype(np.float32)
    X_test, y_test = data["X_test"].astype(np.float32), data["y_test"].astype(np.float32)

    logger.info(f"Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Build head registry
    registry = build_head_registry(args)

    # Train and evaluate each head
    heads_to_train = [h.strip() for h in args.heads.split(",")]
    results = {}
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for head_name in heads_to_train:
        if head_name not in registry:
            logger.warning(f"Unknown head: {head_name}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training oracle head: {head_name}")
        logger.info(f"{'='*60}")

        t_start = time.time()
        oracle = registry[head_name]()

        # Fit with appropriate kwargs
        fit_kwargs = {"n_epochs": args.n_epochs, "verbose": True}
        if head_name == "dkl_ensemble":
            fit_kwargs["seed_base"] = args.seed
        history = oracle.fit(X_train, y_train, X_val, y_val, **fit_kwargs)

        # Evaluate on val and test
        val_metrics = oracle.evaluate(X_val, y_val)
        test_metrics = oracle.evaluate(X_test, y_test)

        elapsed = time.time() - t_start

        results[head_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "elapsed_seconds": elapsed,
        }

        # Save checkpoint
        oracle.save(output_dir / head_name)

        logger.info(
            f"  {head_name} Test: R²={test_metrics['R2']:.4f}, "
            f"ρ={test_metrics['spearman_rho']:.4f}, "
            f"|err|-σ ρ={test_metrics['err_sigma_rho']:.4f}, "
            f"NLL={test_metrics['nll']:.4f} "
            f"({elapsed:.1f}s)"
        )

    # Save comparison table
    with open(output_dir / "tier1_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save as CSV
    with open(output_dir / "tier1_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "split", "R2", "spearman_rho", "rmse", "nll", "err_sigma_rho", "mean_sigma", "elapsed_s"])
        for name, res in results.items():
            for split in ["val", "test"]:
                m = res[split]
                writer.writerow([
                    name, split, f"{m['R2']:.4f}", f"{m['spearman_rho']:.4f}",
                    f"{m['rmse']:.4f}", f"{m['nll']:.4f}", f"{m['err_sigma_rho']:.4f}",
                    f"{m['mean_sigma']:.4f}", f"{res['elapsed_seconds']:.1f}",
                ])

    _print_summary_table(results)
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
