#!/usr/bin/env python3
"""
Retrain A3.6 SchemeB_Independent and save checkpoint.

The original s17 run popped model_state before saving to JSON.
This script reproduces A3.6 with the same hyperparams and saves
the checkpoint for use by FrozenSP2Embedder.

Output: results/stage2/ablation_viz/A36_independent_model.pt  (state_dict)
        results/stage2/ablation_viz/A36_independent_mlp.pt    (MLP state_dict)
"""
from __future__ import annotations

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

from bayesdiff.attention_pool import MLPReadout, SchemeB_Independent
from scripts.pipeline.s12_train_attn_pool import (
    AtomEmbeddingDataset,
    collate_atom_emb,
    compute_metrics,
)


@torch.no_grad()
def _eval(model, mlp, loader, device):
    model.eval(); mlp.eval()
    all_y, all_p = [], []
    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        z, _ = model(layers, atom_mask=mask)
        p = mlp(z)
        all_y.append(batch["pkd"].numpy())
        all_p.append(p.cpu().numpy())
    return compute_metrics(np.concatenate(all_y), np.concatenate(all_p))


def main():
    import pandas as pd

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- data ---
    labels_df = pd.read_csv("data/pdbbind_v2020/labels.csv")
    label_map = dict(zip(labels_df["pdb_code"], labels_df["pkd"]))
    with open("data/pdbbind_v2020/splits.json") as f:
        splits = json.load(f)

    train_ds = AtomEmbeddingDataset("results/atom_embeddings", splits["train"], label_map)
    val_ds   = AtomEmbeddingDataset("results/atom_embeddings", splits["val"],   label_map)
    test_ds  = AtomEmbeddingDataset("results/atom_embeddings", splits["test"],  label_map)

    kw = dict(batch_size=64, collate_fn=collate_atom_emb, num_workers=0, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **kw)
    logger.info(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # --- model ---
    d = 128
    model = SchemeB_Independent(embed_dim=d, n_layers=10, attn_hidden_dim=64, entropy_weight=0.01).to(device)
    mlp   = MLPReadout(input_dim=d, hidden_dim=d).to(device)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in mlp.parameters())
    logger.info(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mlp.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )

    # --- train ---
    best_val_rho = -float("inf")
    best_model_sd = None
    best_mlp_sd = None
    patience_counter = 0
    patience = 30
    t0 = time.time()

    for epoch in range(200):
        model.train(); mlp.train()
        total_loss, n_total = 0.0, 0
        for batch in train_loader:
            y = batch["pkd"].to(device)
            layers = [l.to(device) for l in batch["layer_embs"]]
            mask = batch["mask"].to(device)
            z, info = model(layers, atom_mask=mask)
            pred = mlp(z)
            loss = F.mse_loss(pred, y) + info["entropy_reg"]
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += F.mse_loss(pred, y).item() * len(y)
            n_total += len(y)

        val_m = _eval(model, mlp, val_loader, device)
        if val_m["spearman_rho"] > best_val_rho:
            best_val_rho = val_m["spearman_rho"]
            best_model_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_mlp_sd   = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}: loss={total_loss/n_total:.4f}, val_rho={val_m['spearman_rho']:.4f}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # --- evaluate ---
    model.load_state_dict(best_model_sd)
    mlp.load_state_dict(best_mlp_sd)
    val_m  = _eval(model, mlp, val_loader, device)
    test_m = _eval(model, mlp, test_loader, device)
    elapsed = time.time() - t0

    logger.info(f"A3.6 Val:  R²={val_m['R2']:.4f}, ρ={val_m['spearman_rho']:.4f}")
    logger.info(f"A3.6 Test: R²={test_m['R2']:.4f}, ρ={test_m['spearman_rho']:.4f}")
    logger.info(f"Elapsed: {elapsed:.0f}s")

    # --- save checkpoints ---
    out_dir = Path("results/stage2/ablation_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "A36_independent_model.pt"
    mlp_path   = out_dir / "A36_independent_mlp.pt"
    torch.save(best_model_sd, str(model_path))
    torch.save(best_mlp_sd,   str(mlp_path))
    logger.info(f"Saved model checkpoint: {model_path}")
    logger.info(f"Saved MLP checkpoint:   {mlp_path}")


if __name__ == "__main__":
    main()
