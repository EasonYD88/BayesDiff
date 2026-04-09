"""
scripts/pipeline/s14_phase3_refinement.py
─────────────────────────────────────────
Sub-Plan 2 Phase 3 — Best scheme (Scheme B) refinement + GP integration.

Experiments:
  A3.1: Scheme B + entropy reg λ=0.01 (= A2.3 result, skip / reference only)
  A3.2: Scheme B + entropy reg λ=0.1  (higher regularization)
  A3.4: Scheme B → freeze attention → SVGP head (Step 2 of two-step training)

Usage:
    python scripts/pipeline/s14_phase3_refinement.py \\
        --atom_emb_dir results/atom_embeddings \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/phase3_refinement \\
        --experiment A3.2 A3.4 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from scripts.pipeline.s12_train_attn_pool import (
    AtomEmbeddingDataset,
    collate_atom_emb,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Step 1 training: Scheme B + MLP (reusable for different λ)
# ---------------------------------------------------------------------------

def train_scheme_b_mlp(
    train_loader,
    val_loader,
    test_loader,
    args,
    device,
    entropy_weight: float,
    exp_name: str,
) -> dict:
    """Train SchemeB_SingleBranch + MLPReadout with given entropy_weight."""
    from bayesdiff.attention_pool import MLPReadout, SchemeB_SingleBranch

    d = args.embed_dim
    model = SchemeB_SingleBranch(
        embed_dim=d,
        n_layers=10,
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=entropy_weight,
    ).to(device)
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)

    logger.info(f"=== {exp_name} (λ_ent={entropy_weight}) ===")
    params = list(model.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    best_val_rho = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.n_epochs):
        model.train()
        mlp.train()
        total_loss = 0.0
        n_total = 0

        for batch in train_loader:
            y = batch["pkd"].to(device)
            layers = [l.to(device) for l in batch["layer_embs"]]
            mask = batch["mask"].to(device)

            z_global, info = model(layers, atom_mask=mask)
            pred = mlp(z_global)
            mse_loss = F.mse_loss(pred, y)
            loss = mse_loss + info["entropy_reg"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item() * len(y)
            n_total += len(y)

        # Validation
        val_m = _eval_mlp(model, mlp, val_loader, device)
        if val_m["spearman_rho"] > best_val_rho:
            best_val_rho = val_m["spearman_rho"]
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "mlp": {k: v.cpu().clone() for k, v in mlp.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch+1}: loss={total_loss/n_total:.4f}, "
                f"val_rho={val_m['spearman_rho']:.4f}, val_R2={val_m['R2']:.4f}"
            )

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state["model"])
    mlp.load_state_dict(best_state["mlp"])
    val_m = _eval_mlp(model, mlp, val_loader, device)
    test_m = _eval_mlp(model, mlp, test_loader, device)

    logger.info(
        f"  {exp_name} MLP Test: R²={test_m['R2']:.4f}, ρ={test_m['spearman_rho']:.4f}"
    )

    return {
        "val": val_m,
        "test": test_m,
        "model_state": best_state["model"],
        "mlp_state": best_state["mlp"],
    }


@torch.no_grad()
def _eval_mlp(model, mlp, loader, device) -> dict:
    model.eval()
    mlp.eval()
    all_y, all_pred = [], []
    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        z_global, _ = model(layers, atom_mask=mask)
        pred = mlp(z_global)
        all_y.append(batch["pkd"].numpy())
        all_pred.append(pred.cpu().numpy())
    return compute_metrics(np.concatenate(all_y), np.concatenate(all_pred))


# ---------------------------------------------------------------------------
# Step 2: Extract frozen embeddings + train GP
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_scheme_b_embeddings(
    model: nn.Module,
    loader,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from frozen SchemeB model.

    Returns (X, y) where X is (N, d) and y is (N,).
    """
    model.eval()
    all_z, all_y = [], []
    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        z_global, _ = model(layers, atom_mask=mask)
        all_z.append(z_global.cpu().numpy())
        all_y.append(batch["pkd"].numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


def run_A34_gp(
    model_state: dict,
    train_loader,
    val_loader,
    test_loader,
    args,
    device,
) -> dict:
    """A3.4: Freeze best SchemeB attention → train SVGP head.

    Step 2 of the two-step training strategy (§3.5).
    """
    from bayesdiff.attention_pool import SchemeB_SingleBranch
    from bayesdiff.gp_oracle import GPOracle

    logger.info("=== A3.4: SchemeB (frozen) → SVGP Head ===")

    # 1. Load frozen SchemeB
    d = args.embed_dim
    model = SchemeB_SingleBranch(
        embed_dim=d,
        n_layers=10,
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=args.entropy_weight,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Extract embeddings
    logger.info("  Extracting embeddings from frozen SchemeB...")
    X_train, y_train = extract_scheme_b_embeddings(model, train_loader, device)
    X_val, y_val = extract_scheme_b_embeddings(model, val_loader, device)
    X_test, y_test = extract_scheme_b_embeddings(model, test_loader, device)
    logger.info(
        f"  Embeddings: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    # 3. Train GP
    gp = GPOracle(d=d, n_inducing=args.n_inducing, device=str(device))
    logger.info(f"  Training SVGP (J={args.n_inducing}, {args.gp_epochs} epochs)...")
    history = gp.train(
        X_train, y_train,
        n_epochs=args.gp_epochs,
        batch_size=args.gp_batch_size,
        lr=args.gp_lr,
        verbose=True,
    )

    # 4. Evaluate
    mu_val, var_val = gp.predict(X_val)
    mu_test, var_test = gp.predict(X_test)

    val_m = compute_metrics(y_val, mu_val)
    test_m = compute_metrics(y_test, mu_test)

    # Uncertainty calibration: check correlation between |error| and sqrt(var)
    test_errors = np.abs(y_test - mu_test)
    test_sigma = np.sqrt(var_test)
    unc_rho, unc_p = spearmanr(test_errors, test_sigma)

    logger.info(
        f"  A3.4 GP Test: R²={test_m['R2']:.4f}, ρ={test_m['spearman_rho']:.4f}"
    )
    logger.info(
        f"  Uncertainty calibration: |error|-σ Spearman ρ={unc_rho:.4f} (p={unc_p:.2e})"
    )
    logger.info(f"  Mean σ_test={test_sigma.mean():.4f}, Noise={gp.likelihood.noise.item():.4f}")

    return {
        "val": val_m,
        "test": test_m,
        "uncertainty": {
            "error_sigma_rho": float(unc_rho),
            "error_sigma_p": float(unc_p),
            "mean_sigma_test": float(test_sigma.mean()),
            "mean_sigma_val": float(np.sqrt(var_val).mean()),
            "noise_variance": float(gp.likelihood.noise.item()),
        },
        "gp_final_loss": float(history["loss"][-1]),
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_A32(train_loader, val_loader, test_loader, args, device) -> dict:
    """A3.2: Scheme B + entropy reg λ=0.1."""
    return train_scheme_b_mlp(
        train_loader, val_loader, test_loader, args, device,
        entropy_weight=0.1,
        exp_name="A3.2: SchemeB λ=0.1",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sub-Plan 2 Phase 3: Best scheme refinement + GP"
    )
    parser.add_argument("--atom_emb_dir", type=str, default="results/atom_embeddings")
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--output", type=str, default="results/stage2/phase3_refinement")
    parser.add_argument(
        "--experiment", nargs="+", default=["A3.2", "A3.4"],
        choices=["A3.2", "A3.4"],
    )
    # Model hyperparams
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--attn_hidden_dim", type=int, default=64)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    # GP hyperparams
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--gp_epochs", type=int, default=200)
    parser.add_argument("--gp_batch_size", type=int, default=256)
    parser.add_argument("--gp_lr", type=float, default=0.01)
    # General
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels and splits
    labels_df = pd.read_csv(args.labels)
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))
    with open(args.splits) as f:
        splits = json.load(f)

    # Create datasets (preloaded into RAM)
    train_ds = AtomEmbeddingDataset(args.atom_emb_dir, splits["train"], label_map)
    val_ds = AtomEmbeddingDataset(args.atom_emb_dir, splits["val"], label_map)
    test_ds = AtomEmbeddingDataset(args.atom_emb_dir, splits["test"], label_map)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_atom_emb, num_workers=0, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_atom_emb, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_atom_emb, num_workers=0, pin_memory=True,
    )

    logger.info(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Load Phase 2 reference
    phase2_path = Path("results/stage2/scheme_comparison/scheme_comparison_results.json")
    if phase2_path.exists():
        with open(phase2_path) as f:
            phase2 = json.load(f)
        logger.info(
            f"Phase 2 ref — A2.3 test ρ={phase2['A2.3']['test']['spearman_rho']:.4f}"
        )

    results = {}
    t_start = time.time()

    # --- A3.2: Entropy reg λ=0.1 ---
    if "A3.2" in args.experiment:
        t_exp = time.time()
        res = run_A32(train_loader, val_loader, test_loader, args, device)
        res.pop("model_state", None)
        res.pop("mlp_state", None)
        results["A3.2"] = res
        logger.info(f"  A3.2 took {time.time() - t_exp:.0f}s")

    # --- A3.4: GP integration ---
    if "A3.4" in args.experiment:
        t_exp = time.time()
        # First train SchemeB with best λ (0.01) to get frozen model
        logger.info("=== A3.4 Step 1: Training SchemeB (λ=0.01) for GP feature extraction ===")
        step1_res = train_scheme_b_mlp(
            train_loader, val_loader, test_loader, args, device,
            entropy_weight=args.entropy_weight,
            exp_name="A3.4-Step1",
        )
        model_state = step1_res["model_state"]

        # Step 2: GP on frozen embeddings
        gp_res = run_A34_gp(
            model_state, train_loader, val_loader, test_loader, args, device,
        )
        # Include Step 1 MLP results for comparison
        gp_res["step1_mlp"] = {"val": step1_res["val"], "test": step1_res["test"]}
        results["A3.4"] = gp_res
        logger.info(f"  A3.4 took {time.time() - t_exp:.0f}s")

    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 REFINEMENT RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Exp':<35} {'Val R²':<10} {'Val ρ':<10} {'Test R²':<10} {'Test ρ':<10}")
    logger.info("-" * 75)

    # Phase 2 reference
    if phase2_path.exists():
        for ref_name in ["A2.3"]:
            if ref_name in phase2:
                r = phase2[ref_name]
                logger.info(
                    f"{'(ref) ' + ref_name + ' λ=0.01 MLP':<35} "
                    f"{r['val']['R2']:.4f}     "
                    f"{r['val']['spearman_rho']:.4f}     "
                    f"{r['test']['R2']:.4f}     "
                    f"{r['test']['spearman_rho']:.4f}"
                )
        logger.info("-" * 75)

    for exp_name in sorted(results.keys()):
        r = results[exp_name]
        label = exp_name
        if exp_name == "A3.4":
            # Show both Step 1 MLP and Step 2 GP
            s1 = r.get("step1_mlp", {})
            if s1:
                logger.info(
                    f"{'A3.4-Step1 (SchemeB→MLP)':<35} "
                    f"{s1['val']['R2']:.4f}     "
                    f"{s1['val']['spearman_rho']:.4f}     "
                    f"{s1['test']['R2']:.4f}     "
                    f"{s1['test']['spearman_rho']:.4f}"
                )
            label = "A3.4-Step2 (SchemeB→SVGP)"
        elif exp_name == "A3.2":
            label = "A3.2 (SchemeB λ=0.1 MLP)"

        logger.info(
            f"{label:<35} "
            f"{r['val']['R2']:.4f}     "
            f"{r['val']['spearman_rho']:.4f}     "
            f"{r['test']['R2']:.4f}     "
            f"{r['test']['spearman_rho']:.4f}"
        )

    logger.info("=" * 80)

    if "A3.4" in results:
        unc = results["A3.4"]["uncertainty"]
        logger.info(f"\nUncertainty diagnostics:")
        logger.info(f"  |error|-σ Spearman ρ = {unc['error_sigma_rho']:.4f}")
        logger.info(f"  Mean σ_test = {unc['mean_sigma_test']:.4f}")
        logger.info(f"  GP noise = {unc['noise_variance']:.4f}")

    # Save
    results["args"] = vars(args)
    results["elapsed_seconds"] = time.time() - t_start

    out_file = output_dir / "phase3_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
