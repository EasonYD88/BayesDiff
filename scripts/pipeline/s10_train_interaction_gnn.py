"""
scripts/pipeline/s10_train_interaction_gnn.py
─────────────────────────────────────────────
Stage 2 Sub-Plan 1 — Train InteractionGNN on PDBbind v2020.

Pre-trains the InteractionGNN to predict pKd from bipartite pocket-ligand
contact graphs at heavy-atom resolution. Uses simple MSE loss + a linear head.

After training, the GNN encoder (without head) can be:
  1. Used to extract z_interaction embeddings for GP training
  2. Combined with z_global in MultiGranularityEncoder

Usage:
    python scripts/pipeline/s10_train_interaction_gnn.py \\
        --data_dir data/pdbbind_v2020 \\
        --output results/stage2/interaction_gnn \\
        --device cuda

See doc/Stage_2/01_multi_granularity_repr.md §3, §8.
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
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bayesdiff.interaction_graph import InteractionGraphBuilder
from bayesdiff.interaction_gnn import InteractionGNN, InteractionGNNPredictor
from bayesdiff.pretrain_dataset import PDBbindPairDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Dataset that wraps PDBbindPairDataset and builds interaction graphs ───


class InteractionGraphDataset(Dataset):
    """Wraps PDBbindPairDataset: loads .pt files and builds PyG interaction graphs.

    Each __getitem__ returns a PyG Data object ready for InteractionGNN.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        cutoff: float = 4.5,
        fold_id: int | None = None,
        shuffle_edges: bool = False,
    ):
        self.pair_dataset = PDBbindPairDataset(data_dir, split=split, fold_id=fold_id)
        self.graph_builder = InteractionGraphBuilder(cutoff=cutoff)
        self.shuffle_edges = shuffle_edges

    def __len__(self) -> int:
        return len(self.pair_dataset)

    def __getitem__(self, idx: int):
        sample = self.pair_dataset[idx]

        build_fn = self.graph_builder.build_graph_shuffled if self.shuffle_edges else self.graph_builder.build_graph
        graph = build_fn(
            ligand_pos=sample["ligand_pos"],
            ligand_element=sample["ligand_element"],
            pocket_pos=sample["protein_pos"],
            pocket_element=sample["protein_element"],
            pocket_aa_type=sample["protein_atom_to_aa_type"],
            pkd=sample["pkd"],
        )
        graph.pdb_code = sample["pdb_code"]
        return graph


def collate_graphs(batch):
    """Collate a list of PyG Data objects into a Batch."""
    return Batch.from_data_list(batch)


# ─── Training loop ─────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.to(device)

        pred, _ = model(batch)
        loss = nn.functional.mse_loss(pred, y.squeeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.shape[0]
        total_n += y.shape[0]

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.to(device)

        pred, _ = model(batch)
        preds.append(pred.cpu())
        targets.append(y.squeeze(-1).cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    # Metrics
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    mae = np.mean(np.abs(targets - preds))

    from scipy.stats import spearmanr, pearsonr
    rho, _ = spearmanr(targets, preds)
    pearson_r, _ = pearsonr(targets, preds)

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "spearman": float(rho),
        "pearson": float(pearson_r),
        "n": len(targets),
    }


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train InteractionGNN on PDBbind")
    parser.add_argument("--data_dir", type=str, default="data/pdbbind_v2020")
    parser.add_argument("--output", type=str, default="results/stage2/interaction_gnn")
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_id", type=int, default=None,
                        help="If set, use splits_5fold.json fold. Default: use splits.json")
    parser.add_argument("--shuffle_edges", action="store_true",
                        help="Use shuffled random edges instead of distance-based contacts (ablation A1.10)")
    parser.add_argument("--readout_mode", type=str, default="node",
                        choices=["edge", "node", "both"],
                        help="GNN readout mode: edge (original), node (ligand mean pool), both (concat)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ───────────────────────────────────────────────────────
    if args.shuffle_edges:
        logger.info("*** SHUFFLED-EDGE MODE (ablation A1.10) ***")
    logger.info("Loading datasets...")
    t0 = time.time()

    train_ds = InteractionGraphDataset(
        args.data_dir, split="train", cutoff=args.cutoff, fold_id=args.fold_id,
        shuffle_edges=args.shuffle_edges,
    )
    val_ds = InteractionGraphDataset(
        args.data_dir, split="val", cutoff=args.cutoff, fold_id=args.fold_id,
        shuffle_edges=args.shuffle_edges,
    )
    test_ds = InteractionGraphDataset(
        args.data_dir, split="test", cutoff=args.cutoff, fold_id=args.fold_id,
        shuffle_edges=args.shuffle_edges,
    )

    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"  Dataset init took {time.time() - t0:.1f}s")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_graphs,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_graphs,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_graphs,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────
    edge_dim = InteractionGraphBuilder(cutoff=args.cutoff).edge_dim

    gnn = InteractionGNN(
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        output_dim=args.output_dim,
        readout_mode=args.readout_mode,
    )
    model = InteractionGNNPredictor(gnn).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters, edge_dim={edge_dim}")

    # ── Optimizer + Scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    # ── Training ───────────────────────────────────────────────────────
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    logger.info(f"Starting training for {args.n_epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, args.n_epochs + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["rmse"])

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_ep

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_spearman": val_metrics["spearman"],
            "val_pearson": val_metrics["pearson"],
            "val_mae": val_metrics["mae"],
            "lr": lr,
            "time_s": elapsed,
        }
        history.append(record)

        # Log
        logger.info(
            f"Ep {epoch:3d}/{args.n_epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_RMSE={val_metrics['rmse']:.3f} R²={val_metrics['r2']:.3f} "
            f"ρ={val_metrics['spearman']:.3f} | "
            f"lr={lr:.1e} | {elapsed:.1f}s"
        )

        # Early stopping
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "gnn_state_dict": model.gnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    total_time = time.time() - t_start
    logger.info(f"Training complete in {total_time:.0f}s. Best epoch: {best_epoch}")

    # ── Evaluate best model on test set ────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    val_metrics_best = ckpt["val_metrics"]

    logger.info(f"Test (CASF-2016): RMSE={test_metrics['rmse']:.3f} "
                f"R²={test_metrics['r2']:.3f} ρ={test_metrics['spearman']:.3f} "
                f"Pearson={test_metrics['pearson']:.3f}")
    logger.info(f"Val (best ep={best_epoch}): RMSE={val_metrics_best['rmse']:.3f} "
                f"R²={val_metrics_best['r2']:.3f} ρ={val_metrics_best['spearman']:.3f}")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "total_time_s": total_time,
        "n_params": n_params,
        "val_metrics_best": val_metrics_best,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training history as CSV for easy plotting
    import csv
    with open(output_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    logger.info(f"Results saved to {output_dir}/")

    # ── Extract and save embeddings for downstream GP training ─────────
    logger.info("Extracting z_interaction embeddings for all splits...")
    model.eval()
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_z, all_y, all_codes = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, z = model(batch)
                all_z.append(z.cpu().numpy())
                all_y.append(batch.y.squeeze(-1).cpu().numpy())
                # Extract pdb_codes from batch
                if hasattr(batch, 'pdb_code'):
                    all_codes.extend(batch.pdb_code)

        all_z = np.concatenate(all_z, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        np.savez(
            output_dir / f"z_interaction_{split_name}.npz",
            embeddings=all_z,
            pkd=all_y,
            pdb_codes=np.array(all_codes) if all_codes else np.array([]),
        )
        logger.info(f"  {split_name}: {all_z.shape[0]} embeddings, dim={all_z.shape[1]}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
