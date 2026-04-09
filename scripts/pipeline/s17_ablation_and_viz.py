"""
scripts/pipeline/s17_ablation_and_viz.py
────────────────────────────────────────
Sub-Plan 2 Phase 3 supplementary ablations + molecular attention visualization.

Experiments:
  A3.5: SchemeB with Multi-Head AttnPool (H=4), shared across layers
  A3.6: SchemeB with independent per-layer AttnPool parameters

Visualization:
  - Per-molecule attention weight heatmaps (top-N test set molecules)
  - Atom-importance 2D scatter for representative complexes
  - Per-layer attention entropy distribution
  - Head diversity analysis (A3.5)

Usage:
    python scripts/pipeline/s17_ablation_and_viz.py \\
        --atom_emb_dir results/atom_embeddings \\
        --labels data/pdbbind_v2020/labels.csv \\
        --splits data/pdbbind_v2020/splits.json \\
        --output results/stage2/ablation_viz \\
        --experiment A3.5 A3.6 VIZ \\
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
# Training: shared infrastructure
# ---------------------------------------------------------------------------

def train_model_mlp(
    model: nn.Module,
    mlp: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    args,
    device,
    exp_name: str,
) -> dict:
    """Generic training loop for SchemeB variants + MLP readout."""
    logger.info(f"=== {exp_name} ===")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {n_params:,}")

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
        f"  {exp_name} Test: R²={test_m['R2']:.4f}, ρ={test_m['spearman_rho']:.4f}"
    )

    return {
        "val": val_m,
        "test": test_m,
        "n_params": n_params,
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
# A3.5: Multi-Head
# ---------------------------------------------------------------------------

def run_A35(train_loader, val_loader, test_loader, args, device) -> dict:
    """A3.5: SchemeB with Multi-Head AttnPool (H=4)."""
    from bayesdiff.attention_pool import MLPReadout, SchemeB_MultiHead

    d = args.embed_dim
    model = SchemeB_MultiHead(
        embed_dim=d,
        n_layers=10,
        n_heads=4,
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=args.entropy_weight,
    ).to(device)
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)

    res = train_model_mlp(
        model, mlp, train_loader, val_loader, test_loader,
        args, device, "A3.5: SchemeB Multi-Head H=4"
    )

    # Head diversity analysis on test set
    model.eval()
    head_div = _compute_head_diversity(model, test_loader, device)
    res["head_diversity"] = head_div
    res.pop("model_state", None)
    res.pop("mlp_state", None)
    return res


@torch.no_grad()
def _compute_head_diversity(model, loader, device) -> dict:
    """Measure how different the H heads' attention patterns are.

    Diversity = 1 - avg pairwise cosine similarity between heads' alpha vectors.
    Also compute per-head entropy stats.
    """
    model.eval()
    all_head_entropies = [[] for _ in range(model.n_heads)]
    pairwise_cos = []

    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        _, info = model(layers, atom_mask=mask)

        # Use last layer's head alphas for analysis
        last_layer_alphas = info["head_alphas"][-1]  # list of H tensors, each (B, N)
        H = len(last_layer_alphas)

        for h_idx in range(H):
            alpha_h = last_layer_alphas[h_idx]
            log_a = torch.log(alpha_h.clamp(min=1e-12))
            ent = -(alpha_h * log_a).sum(dim=-1)  # (B,)
            all_head_entropies[h_idx].append(ent.cpu().numpy())

        # Pairwise cosine similarity between heads
        for i in range(H):
            for j in range(i + 1, H):
                cos = F.cosine_similarity(
                    last_layer_alphas[i], last_layer_alphas[j], dim=-1
                )
                pairwise_cos.append(cos.cpu().numpy())

    head_entropy_stats = {}
    for h_idx in range(len(all_head_entropies)):
        ents = np.concatenate(all_head_entropies[h_idx])
        head_entropy_stats[f"head_{h_idx}_entropy_mean"] = float(ents.mean())
        head_entropy_stats[f"head_{h_idx}_entropy_std"] = float(ents.std())

    avg_cos = float(np.concatenate(pairwise_cos).mean()) if pairwise_cos else 0.0
    diversity = 1.0 - avg_cos

    return {
        **head_entropy_stats,
        "avg_pairwise_cosine": avg_cos,
        "diversity_score": diversity,
    }


# ---------------------------------------------------------------------------
# A3.6: Independent per-layer AttnPool
# ---------------------------------------------------------------------------

def run_A36(train_loader, val_loader, test_loader, args, device) -> dict:
    """A3.6: SchemeB with independent per-layer AttnPool parameters."""
    from bayesdiff.attention_pool import MLPReadout, SchemeB_Independent

    d = args.embed_dim
    model = SchemeB_Independent(
        embed_dim=d,
        n_layers=10,
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=args.entropy_weight,
    ).to(device)
    mlp = MLPReadout(input_dim=d, hidden_dim=d).to(device)

    res = train_model_mlp(
        model, mlp, train_loader, val_loader, test_loader,
        args, device, "A3.6: SchemeB Independent AttnPool"
    )
    res.pop("model_state", None)
    res.pop("mlp_state", None)
    return res


# ---------------------------------------------------------------------------
# Visualization: molecular-level attention analysis
# ---------------------------------------------------------------------------

def run_visualization(test_loader, args, device, output_dir: Path):
    """Generate molecular-level attention visualizations using best SchemeB model.

    Loads the Phase 3 A3.4-Step1 model (best MLP configuration) and
    analyzes attention patterns on the test set.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.error("matplotlib not available. Skipping visualization.")
        return {}

    from bayesdiff.attention_pool import MLPReadout, SchemeB_SingleBranch

    # Load best model from Phase 3
    phase3_dir = Path("results/stage2/phase3_refinement")
    model_path = phase3_dir / "A34_step1_model.pt"
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}. Cannot run visualization.")
        return {}
    else:
        d = args.embed_dim
        model = SchemeB_SingleBranch(
            embed_dim=d, n_layers=10,
            attn_hidden_dim=args.attn_hidden_dim,
            entropy_weight=args.entropy_weight,
        ).to(device)
        checkpoint = torch.load(str(model_path), weights_only=False, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        mlp = None  # We only need attention weights, not predictions
        logger.info(f"Loaded model from {model_path}")

    viz_dir = output_dir / "attention_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect attention data for all test molecules
    logger.info("Collecting attention data from test set...")
    attn_data = _collect_attention_data(model, test_loader, device)

    # 2. Fig: Attention entropy distribution per layer
    logger.info("Generating attention entropy distribution plot...")
    _plot_entropy_distribution(attn_data, viz_dir)

    # 3. Fig: Attention heatmap for top-K molecules
    logger.info("Generating per-molecule attention heatmaps...")
    _plot_molecule_attention_heatmaps(attn_data, viz_dir, n_molecules=6)

    # 4. Fig: Layer beta distribution across test set
    logger.info("Generating layer beta distribution...")
    _plot_layer_beta_distribution(attn_data, viz_dir)

    # 5. Fig: Attention concentration vs molecule size
    logger.info("Generating attention concentration analysis...")
    _plot_attention_vs_size(attn_data, viz_dir)

    logger.info(f"Visualization saved to {viz_dir}")
    return {"viz_dir": str(viz_dir), "n_molecules_analyzed": len(attn_data["codes"])}


@torch.no_grad()
def _collect_attention_data(model, loader, device) -> dict:
    """Collect per-molecule attention weights and metadata from test set."""
    model.eval()
    data = {
        "codes": [],
        "n_atoms": [],
        "pkd": [],
        "layer_alphas": [],   # list of (10,) lists of (N,) arrays
        "beta_layer": [],     # list of (10,) arrays
    }

    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        _, info = model(layers, atom_mask=mask)

        B = mask.shape[0]
        for i in range(B):
            n = mask[i].sum().item()
            data["codes"].append(batch["codes"][i])
            data["n_atoms"].append(n)
            data["pkd"].append(batch["pkd"][i].item())

            mol_alphas = []
            for l_idx in range(len(info["layer_alphas"])):
                alpha = info["layer_alphas"][l_idx][i, :n].cpu().numpy()
                mol_alphas.append(alpha)
            data["layer_alphas"].append(mol_alphas)
            data["beta_layer"].append(info["beta_layer"][i].cpu().numpy())

    return data


def _plot_entropy_distribution(attn_data, viz_dir):
    """Plot attention entropy distribution across test molecules, per layer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_layers = len(attn_data["layer_alphas"][0])
    layer_entropies = [[] for _ in range(n_layers)]

    for mol_alphas in attn_data["layer_alphas"]:
        for l_idx, alpha in enumerate(mol_alphas):
            ent = -np.sum(alpha * np.log(np.clip(alpha, 1e-12, None)))
            layer_entropies[l_idx].append(ent)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(n_layers)
    bp = ax.boxplot(
        [layer_entropies[l] for l in range(n_layers)],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_layers))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Encoder Layer", fontsize=12)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=12)
    ax.set_title("Per-Layer Attention Entropy Distribution (Test Set)", fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])

    # Theoretical max entropy line for reference
    median_n = np.median(attn_data["n_atoms"])
    ax.axhline(y=np.log(median_n), color='red', linestyle='--', alpha=0.5,
               label=f'Max entropy (N={int(median_n)})')
    ax.legend()
    plt.tight_layout()
    fig.savefig(viz_dir / "entropy_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_molecule_attention_heatmaps(attn_data, viz_dir, n_molecules=6):
    """Plot attention heatmaps for representative molecules.

    Select molecules spanning high/low pKd and different sizes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_total = len(attn_data["codes"])
    if n_total == 0:
        return

    # Select representative molecules: sort by pKd, pick evenly spaced
    sorted_idx = np.argsort(attn_data["pkd"])
    pick_idx = np.linspace(0, n_total - 1, n_molecules, dtype=int)
    selected = [sorted_idx[i] for i in pick_idx]

    n_layers = len(attn_data["layer_alphas"][0])
    # Only show layers 0, 3, 6, 9 for clarity
    show_layers = [0, 3, 6, 9] if n_layers >= 10 else list(range(n_layers))

    fig, axes = plt.subplots(
        n_molecules, len(show_layers),
        figsize=(3 * len(show_layers), 2.5 * n_molecules),
    )
    if n_molecules == 1:
        axes = axes[np.newaxis, :]

    for row, mol_idx in enumerate(selected):
        code = attn_data["codes"][mol_idx]
        pkd = attn_data["pkd"][mol_idx]
        n_atoms = attn_data["n_atoms"][mol_idx]
        mol_alphas = attn_data["layer_alphas"][mol_idx]

        for col, l_idx in enumerate(show_layers):
            ax = axes[row, col]
            alpha = mol_alphas[l_idx]

            # Bar plot of attention weights
            bars = ax.bar(range(len(alpha)), alpha,
                          color=plt.cm.Reds(alpha / alpha.max() if alpha.max() > 0 else 0),
                          edgecolor='none')

            if row == 0:
                ax.set_title(f"Layer {l_idx}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{code}\npKd={pkd:.1f}\n({n_atoms} atoms)",
                              fontsize=8, rotation=0, ha='right', va='center')

            ax.set_xlim(-0.5, min(len(alpha), 50) - 0.5)
            ax.set_ylim(0, max(alpha.max() * 1.1, 0.01))
            ax.tick_params(labelsize=6)
            if row < n_molecules - 1:
                ax.set_xticklabels([])

    axes[-1, len(show_layers) // 2].set_xlabel("Atom Index", fontsize=10)
    fig.suptitle("Atom Attention Weights Across Layers\n(Representative Test Molecules)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(viz_dir / "molecule_attention_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_layer_beta_distribution(attn_data, viz_dir):
    """Plot distribution of layer-level attention weights (beta) across test set."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    betas = np.array(attn_data["beta_layer"])  # (N_test, n_layers)
    n_layers = betas.shape[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot of per-layer betas
    bp = ax1.boxplot(
        [betas[:, l] for l in range(n_layers)],
        positions=range(n_layers),
        widths=0.6,
        patch_artist=True,
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_layers))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel("Encoder Layer", fontsize=12)
    ax1.set_ylabel("Layer Attention Weight (β)", fontsize=12)
    ax1.set_title("Layer Beta Distribution (Test Set)", fontsize=12)
    ax1.set_xticks(range(n_layers))
    ax1.set_xticklabels([f"L{i}" for i in range(n_layers)])

    # Right: stacked area — cumulative beta for each molecule
    sorted_mol_idx = np.argsort(attn_data["pkd"])
    betas_sorted = betas[sorted_mol_idx]
    x = np.arange(len(sorted_mol_idx))
    ax2.stackplot(
        x, *[betas_sorted[:, l] for l in range(n_layers)],
        labels=[f"L{l}" for l in range(n_layers)],
        colors=colors, alpha=0.8,
    )
    ax2.set_xlabel("Test Molecules (sorted by pKd)", fontsize=12)
    ax2.set_ylabel("Cumulative β", fontsize=12)
    ax2.set_title("Layer Beta Stacked by Molecule", fontsize=12)
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.set_xlim(0, len(sorted_mol_idx) - 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(viz_dir / "layer_beta_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_attention_vs_size(attn_data, viz_dir):
    """Analyze relationship between molecule size, pKd, and attention concentration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_atoms = np.array(attn_data["n_atoms"])
    pkd_vals = np.array(attn_data["pkd"])

    # Compute attention concentration (Gini coefficient) per molecule
    # Use last layer attention
    gini_vals = []
    max_alpha_vals = []
    for mol_alphas in attn_data["layer_alphas"]:
        alpha = mol_alphas[-1]  # last layer
        sorted_a = np.sort(alpha)
        n = len(sorted_a)
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * sorted_a) / (n * np.sum(sorted_a)) - (n + 1) / n) if np.sum(sorted_a) > 0 else 0
        gini_vals.append(gini)
        max_alpha_vals.append(alpha.max())

    gini_vals = np.array(gini_vals)
    max_alpha_vals = np.array(max_alpha_vals)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1: Gini vs molecule size
    ax = axes[0]
    sc = ax.scatter(n_atoms, gini_vals, c=pkd_vals, cmap='coolwarm', alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label='pKd')
    ax.set_xlabel("Number of Ligand Atoms", fontsize=11)
    ax.set_ylabel("Attention Gini Coefficient", fontsize=11)
    ax.set_title("Attention Concentration vs Size", fontsize=12)

    # 2: Max attention weight vs molecule size
    ax = axes[1]
    ax.scatter(n_atoms, max_alpha_vals, c=pkd_vals, cmap='coolwarm', alpha=0.7, s=20)
    ax.set_xlabel("Number of Ligand Atoms", fontsize=11)
    ax.set_ylabel("Max Attention Weight", fontsize=11)
    ax.set_title("Peak Attention vs Size", fontsize=12)

    # 3: Effective number of atoms (exp(entropy)) vs actual
    eff_atoms = []
    for mol_alphas in attn_data["layer_alphas"]:
        alpha = mol_alphas[-1]
        ent = -np.sum(alpha * np.log(np.clip(alpha, 1e-12, None)))
        eff_atoms.append(np.exp(ent))
    eff_atoms = np.array(eff_atoms)

    ax = axes[2]
    ax.scatter(n_atoms, eff_atoms, c=pkd_vals, cmap='coolwarm', alpha=0.7, s=20)
    ax.plot([0, n_atoms.max()], [0, n_atoms.max()], 'k--', alpha=0.3, label='N_eff = N')
    ax.set_xlabel("Number of Ligand Atoms", fontsize=11)
    ax.set_ylabel("Effective Atoms (exp(H))", fontsize=11)
    ax.set_title("Effective vs Actual Atoms", fontsize=12)
    ax.legend()

    fig.suptitle("Attention Concentration Analysis (Last Layer, Test Set)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(viz_dir / "attention_vs_size.png", dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sub-Plan 2 A3.5/A3.6 ablations + attention visualization"
    )
    parser.add_argument("--atom_emb_dir", type=str, default="results/atom_embeddings")
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--output", type=str, default="results/stage2/ablation_viz")
    parser.add_argument(
        "--experiment", nargs="+", default=["A3.5", "A3.6", "VIZ"],
        choices=["A3.5", "A3.6", "VIZ"],
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

    # Load reference results
    ref_path = Path("results/stage2/scheme_comparison/scheme_comparison_results.json")
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        logger.info(
            f"Reference: A2.3 SchemeB shared → test ρ={ref['A2.3']['test']['spearman_rho']:.4f}"
        )

    results = {}
    t_total = time.time()

    # --- A3.5: Multi-Head H=4 ---
    if "A3.5" in args.experiment:
        t0 = time.time()
        results["A3.5"] = run_A35(
            train_loader, val_loader, test_loader, args, device
        )
        results["A3.5"]["elapsed_s"] = time.time() - t0
        logger.info(f"  A3.5 took {results['A3.5']['elapsed_s']:.0f}s")

    # --- A3.6: Independent per-layer AttnPool ---
    if "A3.6" in args.experiment:
        t0 = time.time()
        results["A3.6"] = run_A36(
            train_loader, val_loader, test_loader, args, device
        )
        results["A3.6"]["elapsed_s"] = time.time() - t0
        logger.info(f"  A3.6 took {results['A3.6']['elapsed_s']:.0f}s")

    # --- VIZ: Attention visualization ---
    if "VIZ" in args.experiment:
        t0 = time.time()
        viz_res = run_visualization(test_loader, args, device, output_dir)
        results["VIZ"] = viz_res
        logger.info(f"  VIZ took {time.time() - t0:.0f}s")

    # ─── Summary Table ────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION + VISUALIZATION RESULTS")
    logger.info("=" * 80)
    header = f"{'Exp':<40} {'Val R²':<10} {'Val ρ':<10} {'Test R²':<10} {'Test ρ':<10} {'#Params':<10}"
    logger.info(header)
    logger.info("-" * len(header))

    # Reference rows
    logger.info(
        f"{'(ref) A2.3 SchemeB shared λ=0.01':<40} {'0.292':<10} {'0.577':<10} {'0.568':<10} {'0.747':<10} {'—':<10}"
    )
    logger.info(
        f"{'(ref) A3.4-S1 SchemeB retrained':<40} {'0.262':<10} {'0.555':<10} {'0.572':<10} {'0.761':<10} {'—':<10}"
    )
    logger.info("-" * len(header))

    for exp_name in ["A3.5", "A3.6"]:
        if exp_name in results:
            res = results[exp_name]
            logger.info(
                f"{exp_name:<40} "
                f"{res['val']['R2']:<10.4f} "
                f"{res['val']['spearman_rho']:<10.4f} "
                f"{res['test']['R2']:<10.4f} "
                f"{res['test']['spearman_rho']:<10.4f} "
                f"{res.get('n_params', '—'):<10}"
            )

    logger.info("=" * 80)

    if "A3.5" in results and "head_diversity" in results["A3.5"]:
        hd = results["A3.5"]["head_diversity"]
        logger.info("\nA3.5 Head Diversity Analysis:")
        logger.info(f"  Diversity score: {hd['diversity_score']:.4f}")
        logger.info(f"  Avg pairwise cosine: {hd['avg_pairwise_cosine']:.4f}")
        for h in range(4):
            k = f"head_{h}_entropy_mean"
            if k in hd:
                logger.info(f"  Head {h} entropy: {hd[k]:.3f} ± {hd[f'head_{h}_entropy_std']:.3f}")

    total_time = time.time() - t_total
    logger.info(f"\nTotal time: {total_time:.0f}s")

    # Save results
    results_path = output_dir / "ablation_results.json"
    # Remove non-serializable items
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, (torch.Tensor, nn.Module))
            }
        else:
            serializable[k] = v

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
