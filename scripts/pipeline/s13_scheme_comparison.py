"""
scripts/pipeline/s13_scheme_comparison.py
─────────────────────────────────────────
Sub-Plan 2 Phase 2 — Scheme A (TwoBranch) vs Scheme B (SingleBranch) comparison.

Experiments:
  A2.1: 9-layer MeanPool → AttnFusion → MLP readout (SP3-style baseline)
  A2.2: Scheme A — Last-layer AttnPool (z_atom) + 9-layer MeanPool→AttnFusion (z_global) → MLP
  A2.3: Scheme B — 9-layer shared AttnPool → AttnFusion → MLP

Still Step 1 (MLP readout, no GP).

Usage:
    python scripts/pipeline/s13_scheme_comparison.py \\
        --atom_emb_dir results/atom_embeddings \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/scheme_comparison \\
        --experiment A2.1 A2.2 A2.3 \\
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


# ---------------------------------------------------------------------------
# Reuse dataset/collate from s12
# ---------------------------------------------------------------------------
from scripts.pipeline.s12_train_attn_pool import (
    AtomEmbeddingDataset,
    collate_atom_emb,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model: nn.Module,
    mlp: nn.Module,
    forward_fn,
    train_loader,
    val_loader,
    test_loader,
    args,
    device,
    exp_name: str,
) -> dict:
    """Generic train loop for any scheme + MLP readout.

    Parameters
    ----------
    model : nn.Module
        The aggregation module (SchemeA, SchemeB, or LayerAttentionFusion).
    mlp : nn.Module
        MLP readout head.
    forward_fn : callable(model, mlp, batch, device) -> (pred, reg_loss)
        Defines how to get predictions and optional reg loss from a batch.
    """
    logger.info(f"=== {exp_name} ===")
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
            pred, reg_loss = forward_fn(model, mlp, batch, device)
            mse_loss = F.mse_loss(pred, y)
            loss = mse_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item() * len(y)
            n_total += len(y)

        # Validation
        val_m = _eval_loop(model, mlp, forward_fn, val_loader, device)
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
                f"val_rho={val_m['spearman_rho']:.4f}, "
                f"val_R2={val_m['R2']:.4f}"
            )

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best and evaluate
    model.load_state_dict(best_state["model"])
    mlp.load_state_dict(best_state["mlp"])
    val_m = _eval_loop(model, mlp, forward_fn, val_loader, device)
    test_m = _eval_loop(model, mlp, forward_fn, test_loader, device)

    logger.info(
        f"  {exp_name} Test: R²={test_m['R2']:.4f}, ρ={test_m['spearman_rho']:.4f}"
    )
    return {"val": val_m, "test": test_m, "best_state": best_state}


@torch.no_grad()
def _eval_loop(model, mlp, forward_fn, loader, device) -> dict:
    model.eval()
    mlp.eval()
    all_y, all_pred = [], []
    for batch in loader:
        pred, _ = forward_fn(model, mlp, batch, device)
        all_y.append(batch["pkd"].numpy())
        all_pred.append(pred.cpu().numpy())
    return compute_metrics(np.concatenate(all_y), np.concatenate(all_pred))


# ---------------------------------------------------------------------------
# Forward functions per experiment
# ---------------------------------------------------------------------------

def forward_A21_attnfusion_meanpool(model, mlp, batch, device):
    """A2.1: 9-layer MeanPool → AttnFusion → MLP."""
    layers = batch["layer_embs"]  # list of (B, N, d)
    mask = batch["mask"].to(device)
    mask_f = mask.unsqueeze(-1).float()

    layer_means = []
    for l_emb in layers:
        l_emb = l_emb.to(device)
        z_l = (l_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        layer_means.append(z_l)

    z_fuse, beta = model(layer_means)  # LayerAttentionFusion
    pred = mlp(z_fuse)
    return pred, torch.tensor(0.0, device=device)


def forward_A22_scheme_a(model, mlp, batch, device):
    """A2.2: Scheme A TwoBranch → MLP."""
    layers = [l.to(device) for l in batch["layer_embs"]]
    mask = batch["mask"].to(device)

    z_new, info = model(layers, atom_mask=mask)
    pred = mlp(z_new)
    return pred, info["entropy_reg"]


def forward_A23_scheme_b(model, mlp, batch, device):
    """A2.3: Scheme B SingleBranch → MLP."""
    layers = [l.to(device) for l in batch["layer_embs"]]
    mask = batch["mask"].to(device)

    z_global, info = model(layers, atom_mask=mask)
    pred = mlp(z_global)
    return pred, info["entropy_reg"]


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_A21(train_loader, val_loader, test_loader, args, device) -> dict:
    """A2.1: 9-layer MeanPool → LayerAttentionFusion → MLP."""
    from bayesdiff.attention_pool import MLPReadout
    from bayesdiff.layer_fusion import LayerAttentionFusion

    d = args.embed_dim
    model = LayerAttentionFusion(embed_dim=d, hidden_dim=args.attn_hidden_dim).to(device)
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)

    return train_and_evaluate(
        model, mlp, forward_A21_attnfusion_meanpool,
        train_loader, val_loader, test_loader, args, device,
        exp_name="A2.1: MeanPool→AttnFusion→MLP",
    )


def run_A22(train_loader, val_loader, test_loader, args, device) -> dict:
    """A2.2: Scheme A — TwoBranch → MLP."""
    from bayesdiff.attention_pool import MLPReadout, SchemeA_TwoBranch

    d = args.embed_dim
    model = SchemeA_TwoBranch(
        embed_dim=d,
        n_layers=10,
        fusion_type="concat_mlp",
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=args.entropy_weight,
    ).to(device)
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)

    result = train_and_evaluate(
        model, mlp, forward_A22_scheme_a,
        train_loader, val_loader, test_loader, args, device,
        exp_name="A2.2: SchemeA TwoBranch→MLP",
    )

    # Collect branch-level stats
    model.eval()
    result["branch_stats"] = _collect_scheme_a_stats(model, test_loader, device)
    return result


def run_A23(train_loader, val_loader, test_loader, args, device) -> dict:
    """A2.3: Scheme B — SingleBranch (shared) → MLP."""
    from bayesdiff.attention_pool import MLPReadout, SchemeB_SingleBranch

    d = args.embed_dim
    model = SchemeB_SingleBranch(
        embed_dim=d,
        n_layers=10,
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=args.entropy_weight,
    ).to(device)
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)

    result = train_and_evaluate(
        model, mlp, forward_A23_scheme_b,
        train_loader, val_loader, test_loader, args, device,
        exp_name="A2.3: SchemeB SingleBranch→MLP",
    )

    # Collect per-layer attention stats
    model.eval()
    result["layer_stats"] = _collect_scheme_b_stats(model, test_loader, device)
    return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_scheme_a_stats(model, loader, device) -> dict:
    """Collect branch and attention stats for Scheme A."""
    model.eval()
    atom_entropies = []
    layer_betas = []

    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        _, info = model(layers, atom_mask=mask)

        alpha = info["alpha_atom"]
        log_alpha = torch.log(alpha.clamp(min=1e-12))
        H = -(alpha * log_alpha).sum(dim=-1)
        atom_entropies.append(H.cpu().numpy())
        layer_betas.append(info["beta_layer"].cpu().numpy())

    all_H = np.concatenate(atom_entropies)
    all_beta = np.concatenate(layer_betas, axis=0)  # (N_test, L)
    mean_beta = all_beta.mean(axis=0).tolist()

    return {
        "atom_attn_entropy_mean": float(all_H.mean()),
        "atom_attn_entropy_std": float(all_H.std()),
        "layer_beta_mean": mean_beta,
        "layer_beta_std": all_beta.std(axis=0).tolist(),
    }


@torch.no_grad()
def _collect_scheme_b_stats(model, loader, device) -> dict:
    """Collect per-layer attention stats for Scheme B."""
    model.eval()
    n_layers = model.n_layers
    per_layer_entropy = {l: [] for l in range(n_layers)}
    layer_betas = []

    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        _, info = model(layers, atom_mask=mask)

        for l_idx, alpha_l in enumerate(info["layer_alphas"]):
            log_a = torch.log(alpha_l.clamp(min=1e-12))
            H = -(alpha_l * log_a).sum(dim=-1)
            per_layer_entropy[l_idx].append(H.cpu().numpy())

        layer_betas.append(info["beta_layer"].cpu().numpy())

    stats = {}
    for l_idx in range(n_layers):
        all_H = np.concatenate(per_layer_entropy[l_idx])
        stats[f"layer_{l_idx}_entropy_mean"] = float(all_H.mean())
        stats[f"layer_{l_idx}_entropy_std"] = float(all_H.std())

    all_beta = np.concatenate(layer_betas, axis=0)
    stats["layer_beta_mean"] = all_beta.mean(axis=0).tolist()
    stats["layer_beta_std"] = all_beta.std(axis=0).tolist()

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sub-Plan 2 Phase 2: Scheme A vs Scheme B comparison"
    )
    parser.add_argument(
        "--atom_emb_dir", type=str, default="results/atom_embeddings",
    )
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument(
        "--output", type=str, default="results/stage2/scheme_comparison",
    )
    parser.add_argument(
        "--experiment", nargs="+", default=["A2.1", "A2.2", "A2.3"],
        choices=["A2.1", "A2.2", "A2.3"],
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--attn_hidden_dim", type=int, default=64)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
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

    # Create datasets
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

    # Load Phase 1 results for reference
    phase1_path = Path("results/stage2/attention_pool/preliminary_results.json")
    if phase1_path.exists():
        with open(phase1_path) as f:
            phase1 = json.load(f)
        logger.info(
            f"Phase 1 reference — P0 test ρ={phase1['P0']['test']['spearman_rho']:.4f}, "
            f"P1 test ρ={phase1['P1']['test']['spearman_rho']:.4f}"
        )

    results = {}
    t_start = time.time()

    runners = {"A2.1": run_A21, "A2.2": run_A22, "A2.3": run_A23}

    for exp_name in args.experiment:
        t_exp = time.time()
        res = runners[exp_name](train_loader, val_loader, test_loader, args, device)
        # Don't save state_dict to JSON
        res.pop("best_state", None)
        results[exp_name] = res
        logger.info(f"  {exp_name} took {time.time() - t_exp:.0f}s")

    # Summary
    logger.info("\n" + "=" * 75)
    logger.info("PHASE 2 SCHEME COMPARISON RESULTS")
    logger.info("=" * 75)
    logger.info(f"{'Exp':<30} {'Val R²':<10} {'Val ρ':<10} {'Test R²':<10} {'Test ρ':<10}")
    logger.info("-" * 70)

    # Include Phase 1 baselines for context
    if phase1_path.exists():
        for pname in ["P0", "P1"]:
            if pname in phase1:
                r = phase1[pname]
                logger.info(
                    f"{'(ref) ' + pname:<30} "
                    f"{r['val']['R2']:.4f}     "
                    f"{r['val']['spearman_rho']:.4f}     "
                    f"{r['test']['R2']:.4f}     "
                    f"{r['test']['spearman_rho']:.4f}"
                )
        logger.info("-" * 70)

    for exp_name in sorted(results.keys()):
        r = results[exp_name]
        logger.info(
            f"{exp_name:<30} "
            f"{r['val']['R2']:.4f}     "
            f"{r['val']['spearman_rho']:.4f}     "
            f"{r['test']['R2']:.4f}     "
            f"{r['test']['spearman_rho']:.4f}"
        )
    logger.info("=" * 75)

    # Determine best scheme
    if len(results) >= 2:
        best_exp = max(results.keys(), key=lambda k: results[k]["test"]["spearman_rho"])
        logger.info(f"\nBest scheme: {best_exp} (test ρ={results[best_exp]['test']['spearman_rho']:.4f})")

    # Save
    results["args"] = vars(args)
    results["elapsed_seconds"] = time.time() - t_start

    out_file = output_dir / "scheme_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
