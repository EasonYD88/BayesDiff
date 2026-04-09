"""
scripts/pipeline/s12_train_attn_pool.py
───────────────────────────────────────
Sub-Plan 2 — Train attention pooling with MLP readout (Step 1).

Experiments:
  P0: Last-layer MeanPool → MLP readout (baseline)
  P1: Last-layer SelfAttnPool → MLP readout
  P2: (bonus) Last-layer CrossAttnPool → MLP readout

This is Step 1 of the two-step training strategy (§3.5):
  - Validate representation quality via MLP readout (MSE loss)
  - DO NOT use GP yet — isolate attention learning from GP complexity.

Usage:
    python scripts/pipeline/s12_train_attn_pool.py \\
        --atom_emb_dir results/atom_embeddings \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/attention_pool \\
        --experiment P0 P1 \\
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
# Dataset
# ---------------------------------------------------------------------------

class AtomEmbeddingDataset(torch.utils.data.Dataset):
    """Load pre-extracted atom-level embeddings from per-complex .pt files.

    Each file contains:
        'layer_0'..'layer_9': (N_lig, d) numpy arrays
        'pocket_mean': (d,) numpy array
        'n_ligand_atoms': int
        'n_layers': int

    All data is preloaded into RAM at init to avoid per-batch disk I/O.
    """

    def __init__(
        self,
        atom_emb_dir: str | Path,
        codes: list[str],
        labels: dict[str, float],
        n_layers: int = 10,
    ):
        self.atom_emb_dir = Path(atom_emb_dir)
        self.labels = labels
        self.n_layers = n_layers

        # Preload all data into memory
        self.data_cache = []
        skipped = 0
        for code in codes:
            pt_path = self.atom_emb_dir / f"{code}.pt"
            if not pt_path.exists() or code not in labels:
                skipped += 1
                continue
            raw = torch.load(str(pt_path), weights_only=False)
            layer_embs = []
            for l in range(n_layers):
                layer_embs.append(torch.tensor(raw[f"layer_{l}"], dtype=torch.float32))
            self.data_cache.append({
                "layer_embs": layer_embs,
                "pocket_mean": torch.tensor(raw["pocket_mean"], dtype=torch.float32),
                "pkd": torch.tensor(labels[code], dtype=torch.float32),
                "n_atoms": raw["n_ligand_atoms"],
                "code": code,
            })

        logger.info(
            f"AtomEmbeddingDataset: {len(self.data_cache)}/{len(codes)} "
            f"complexes preloaded into RAM"
        )

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]


def collate_atom_emb(batch: list[dict]) -> dict:
    """Collate variable-length atom embeddings with padding."""
    n_layers = len(batch[0]["layer_embs"])
    max_atoms = max(item["n_atoms"] for item in batch)
    d = batch[0]["layer_embs"][0].shape[-1]
    B = len(batch)

    # Padded tensors per layer: (B, max_atoms, d)
    padded_layers = [torch.zeros(B, max_atoms, d) for _ in range(n_layers)]
    mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    pkd = torch.zeros(B)
    pocket_means = torch.zeros(B, d)

    for i, item in enumerate(batch):
        n = item["n_atoms"]
        mask[i, :n] = True
        pkd[i] = item["pkd"]
        pocket_means[i] = item["pocket_mean"]
        for l in range(n_layers):
            padded_layers[l][i, :n, :] = item["layer_embs"][l]

    return {
        "layer_embs": padded_layers,  # list of (B, max_atoms, d)
        "mask": mask,  # (B, max_atoms)
        "pkd": pkd,  # (B,)
        "pocket_mean": pocket_means,  # (B, d)
        "codes": [item["code"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², Spearman ρ, RMSE."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rho, p_val = spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {
        "R2": float(r2),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "RMSE": float(rmse),
    }


def run_P0_meanpool_baseline(
    train_loader, val_loader, test_loader, args, device
) -> dict:
    """P0: Last-layer MeanPool → MLP readout."""
    from bayesdiff.attention_pool import MLPReadout

    logger.info("=== P0: Last-Layer MeanPool → MLP ===")
    d = args.embed_dim
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_rho = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.n_epochs):
        mlp.train()
        total_loss = 0.0
        n_total = 0

        for batch in train_loader:
            # Last layer mean pool (using mask)
            h_last = batch["layer_embs"][-1].to(device)  # (B, N, d)
            mask = batch["mask"].to(device)  # (B, N)
            mask_f = mask.unsqueeze(-1).float()
            z = (h_last * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

            y = batch["pkd"].to(device)
            pred = mlp(z)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            n_total += len(y)

        # Validation
        val_metrics = _evaluate(mlp, val_loader, device, pooling="mean")
        if val_metrics["spearman_rho"] > best_val_rho:
            best_val_rho = val_metrics["spearman_rho"]
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch+1}: loss={total_loss/n_total:.4f}, "
                f"val_rho={val_metrics['spearman_rho']:.4f}, "
                f"val_R2={val_metrics['R2']:.4f}"
            )

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Test with best model
    mlp.load_state_dict(best_state)
    test_metrics = _evaluate(mlp, test_loader, device, pooling="mean")
    val_metrics = _evaluate(mlp, val_loader, device, pooling="mean")

    logger.info(
        f"  P0 Test: R²={test_metrics['R2']:.4f}, "
        f"ρ={test_metrics['spearman_rho']:.4f}"
    )
    return {"val": val_metrics, "test": test_metrics}


def run_P1_self_attn(
    train_loader, val_loader, test_loader, args, device
) -> dict:
    """P1: Last-layer SelfAttnPool → MLP readout."""
    from bayesdiff.attention_pool import (
        AttentionPoolingWithRegularization,
        MLPReadout,
        SelfAttentionPooling,
    )

    logger.info("=== P1: Last-Layer SelfAttnPool → MLP ===")
    d = args.embed_dim
    pool_raw = SelfAttentionPooling(input_dim=d, hidden_dim=args.attn_hidden_dim)
    pool = AttentionPoolingWithRegularization(
        pool_raw, entropy_weight=args.entropy_weight
    )
    mlp = MLPReadout(input_dim=d, hidden_dim=d)

    pool = pool.to(device)
    mlp = mlp.to(device)

    optimizer = torch.optim.AdamW(
        list(pool.parameters()) + list(mlp.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_rho = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.n_epochs):
        pool.train()
        mlp.train()
        total_loss = 0.0
        n_total = 0

        for batch in train_loader:
            h_last = batch["layer_embs"][-1].to(device)
            mask = batch["mask"].to(device)
            y = batch["pkd"].to(device)

            z, alpha, reg_loss = pool(h_last, mask=mask)
            pred = mlp(z)
            mse_loss = F.mse_loss(pred, y)
            loss = mse_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item() * len(y)
            n_total += len(y)

        val_metrics = _evaluate_attn(pool, mlp, val_loader, device)
        if val_metrics["spearman_rho"] > best_val_rho:
            best_val_rho = val_metrics["spearman_rho"]
            best_state = {
                "pool": {k: v.cpu().clone() for k, v in pool.state_dict().items()},
                "mlp": {k: v.cpu().clone() for k, v in mlp.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch+1}: loss={total_loss/n_total:.4f}, "
                f"val_rho={val_metrics['spearman_rho']:.4f}, "
                f"val_R2={val_metrics['R2']:.4f}"
            )

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    pool.load_state_dict(best_state["pool"])
    mlp.load_state_dict(best_state["mlp"])
    test_metrics = _evaluate_attn(pool, mlp, test_loader, device)
    val_metrics = _evaluate_attn(pool, mlp, val_loader, device)

    # Collect attention statistics
    attn_stats = _collect_attn_stats(pool, test_loader, device)

    logger.info(
        f"  P1 Test: R²={test_metrics['R2']:.4f}, "
        f"ρ={test_metrics['spearman_rho']:.4f}, "
        f"attn_entropy={attn_stats['mean_entropy']:.3f}"
    )
    return {"val": val_metrics, "test": test_metrics, "attn_stats": attn_stats}


def run_P2_cross_attn(
    train_loader, val_loader, test_loader, args, device
) -> dict:
    """P2 (bonus): Last-layer CrossAttnPool → MLP readout."""
    from bayesdiff.attention_pool import (
        AttentionPoolingWithRegularization,
        CrossAttentionPooling,
        MLPReadout,
    )

    logger.info("=== P2: Last-Layer CrossAttnPool (pocket-cond) → MLP ===")
    d = args.embed_dim
    pool_raw = CrossAttentionPooling(
        ligand_dim=d, pocket_dim=d, hidden_dim=args.attn_hidden_dim
    )
    pool = AttentionPoolingWithRegularization(
        pool_raw, entropy_weight=args.entropy_weight
    )
    mlp = MLPReadout(input_dim=d, hidden_dim=d)

    pool = pool.to(device)
    mlp = mlp.to(device)

    optimizer = torch.optim.AdamW(
        list(pool.parameters()) + list(mlp.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_rho = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.n_epochs):
        pool.train()
        mlp.train()
        total_loss = 0.0
        n_total = 0

        for batch in train_loader:
            h_last = batch["layer_embs"][-1].to(device)
            mask = batch["mask"].to(device)
            h_pocket = batch["pocket_mean"].to(device)
            y = batch["pkd"].to(device)

            z, alpha, reg_loss = pool(h_last, h_pocket, ligand_mask=mask)
            pred = mlp(z)
            mse_loss = F.mse_loss(pred, y)
            loss = mse_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item() * len(y)
            n_total += len(y)

        val_metrics = _evaluate_cross_attn(pool, mlp, val_loader, device)
        if val_metrics["spearman_rho"] > best_val_rho:
            best_val_rho = val_metrics["spearman_rho"]
            best_state = {
                "pool": {k: v.cpu().clone() for k, v in pool.state_dict().items()},
                "mlp": {k: v.cpu().clone() for k, v in mlp.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch+1}: loss={total_loss/n_total:.4f}, "
                f"val_rho={val_metrics['spearman_rho']:.4f}"
            )

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    pool.load_state_dict(best_state["pool"])
    mlp.load_state_dict(best_state["mlp"])
    test_metrics = _evaluate_cross_attn(pool, mlp, test_loader, device)
    val_metrics = _evaluate_cross_attn(pool, mlp, val_loader, device)

    logger.info(
        f"  P2 Test: R²={test_metrics['R2']:.4f}, "
        f"ρ={test_metrics['spearman_rho']:.4f}"
    )
    return {"val": val_metrics, "test": test_metrics}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(mlp, loader, device, pooling="mean") -> dict:
    mlp.eval()
    all_y, all_pred = [], []
    for batch in loader:
        h_last = batch["layer_embs"][-1].to(device)
        mask = batch["mask"].to(device)
        mask_f = mask.unsqueeze(-1).float()
        z = (h_last * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        pred = mlp(z)
        all_y.append(batch["pkd"].numpy())
        all_pred.append(pred.cpu().numpy())
    return compute_metrics(np.concatenate(all_y), np.concatenate(all_pred))


@torch.no_grad()
def _evaluate_attn(pool, mlp, loader, device) -> dict:
    pool.eval()
    mlp.eval()
    all_y, all_pred = [], []
    for batch in loader:
        h_last = batch["layer_embs"][-1].to(device)
        mask = batch["mask"].to(device)
        z, alpha, _ = pool(h_last, mask=mask)
        pred = mlp(z)
        all_y.append(batch["pkd"].numpy())
        all_pred.append(pred.cpu().numpy())
    return compute_metrics(np.concatenate(all_y), np.concatenate(all_pred))


@torch.no_grad()
def _evaluate_cross_attn(pool, mlp, loader, device) -> dict:
    pool.eval()
    mlp.eval()
    all_y, all_pred = [], []
    for batch in loader:
        h_last = batch["layer_embs"][-1].to(device)
        mask = batch["mask"].to(device)
        h_pocket = batch["pocket_mean"].to(device)
        z, alpha, _ = pool(h_last, h_pocket, ligand_mask=mask)
        pred = mlp(z)
        all_y.append(batch["pkd"].numpy())
        all_pred.append(pred.cpu().numpy())
    return compute_metrics(np.concatenate(all_y), np.concatenate(all_pred))


@torch.no_grad()
def _collect_attn_stats(pool, loader, device) -> dict:
    """Collect attention weight statistics for analysis."""
    pool.eval()
    entropies = []
    max_weights = []
    for batch in loader:
        h_last = batch["layer_embs"][-1].to(device)
        mask = batch["mask"].to(device)
        _, alpha, _ = pool(h_last, mask=mask)
        # Entropy per sample
        log_alpha = torch.log(alpha.clamp(min=1e-12))
        H = -(alpha * log_alpha).sum(dim=-1)
        entropies.append(H.cpu().numpy())
        max_weights.append(alpha.max(dim=-1).values.cpu().numpy())

    all_H = np.concatenate(entropies)
    all_max = np.concatenate(max_weights)
    return {
        "mean_entropy": float(all_H.mean()),
        "std_entropy": float(all_H.std()),
        "mean_max_weight": float(all_max.mean()),
        "std_max_weight": float(all_max.std()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sub-Plan 2: Attention pooling preliminary experiments (P0/P1/P2)"
    )
    parser.add_argument(
        "--atom_emb_dir", type=str, default="results/atom_embeddings",
        help="Directory with per-complex .pt atom embedding files",
    )
    parser.add_argument(
        "--labels", type=str, default="data/pdbbind_v2020/labels.csv",
    )
    parser.add_argument(
        "--splits", type=str, default="data/pdbbind_v2020/splits.json",
    )
    parser.add_argument(
        "--output", type=str, default="results/stage2/attention_pool",
    )
    parser.add_argument(
        "--experiment", nargs="+", default=["P0", "P1"],
        choices=["P0", "P1", "P2"],
        help="Which experiments to run",
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
    train_ds = AtomEmbeddingDataset(
        args.atom_emb_dir, splits["train"], label_map, n_layers=10
    )
    val_ds = AtomEmbeddingDataset(
        args.atom_emb_dir, splits["val"], label_map, n_layers=10
    )
    test_ds = AtomEmbeddingDataset(
        args.atom_emb_dir, splits["test"], label_map, n_layers=10
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_atom_emb, num_workers=0, pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_atom_emb, num_workers=0, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_atom_emb, num_workers=0, pin_memory=False,
    )

    logger.info(
        f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    results = {}
    t_start = time.time()

    for exp_name in args.experiment:
        t_exp = time.time()
        if exp_name == "P0":
            results["P0"] = run_P0_meanpool_baseline(
                train_loader, val_loader, test_loader, args, device
            )
        elif exp_name == "P1":
            results["P1"] = run_P1_self_attn(
                train_loader, val_loader, test_loader, args, device
            )
        elif exp_name == "P2":
            results["P2"] = run_P2_cross_attn(
                train_loader, val_loader, test_loader, args, device
            )
        logger.info(f"  {exp_name} took {time.time() - t_exp:.0f}s")

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("PRELIMINARY RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Exp':<6} {'Val R²':<10} {'Val ρ':<10} {'Test R²':<10} {'Test ρ':<10}")
    logger.info("-" * 46)
    for exp_name in sorted(results.keys()):
        r = results[exp_name]
        logger.info(
            f"{exp_name:<6} "
            f"{r['val']['R2']:.4f}     "
            f"{r['val']['spearman_rho']:.4f}     "
            f"{r['test']['R2']:.4f}     "
            f"{r['test']['spearman_rho']:.4f}"
        )
    logger.info("=" * 70)

    # Go/No-Go gate
    if "P0" in results and "P1" in results:
        p0_rho = results["P0"]["test"]["spearman_rho"]
        p1_rho = results["P1"]["test"]["spearman_rho"]
        decision = "GO" if p1_rho > p0_rho else "NO-GO"
        logger.info(
            f"\nGo/No-Go: P1 ρ={p1_rho:.4f} vs P0 ρ={p0_rho:.4f} → {decision}"
        )
        results["go_no_go"] = {
            "decision": decision,
            "P0_test_rho": p0_rho,
            "P1_test_rho": p1_rho,
            "delta": p1_rho - p0_rho,
        }

    # Save
    results["args"] = vars(args)
    results["elapsed_seconds"] = time.time() - t_start

    out_file = output_dir / "preliminary_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
