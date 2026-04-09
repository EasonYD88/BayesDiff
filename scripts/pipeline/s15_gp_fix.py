"""
scripts/pipeline/s15_gp_fix.py
───────────────────────────────
Sub-Plan 2 Phase 3 GP fix — address SVGP degradation from s14.

Diagnosis: 128-d embedding → ARD Matérn-5/2 SVGP with J=512 inducing points
could not learn useful structure (noise=1.654, no uncertainty calibration).

Fix strategies:
  A3.4b: PCA (128→32) → SVGP  (reduce GP input dimensionality)
  A3.4c: DKL (128→32 MLP → SVGP, jointly trained)
  A3.4d: PCA (128→16) → SVGP  (even more aggressive reduction)

Also: constrain noise to [0.01, 1.0] to prevent noise explosion.

Usage:
    python scripts/pipeline/s15_gp_fix.py \
        --atom_emb_dir results/atom_embeddings \
        --labels data/pdbbind_v2020/labels.csv \
        --splits data/pdbbind_v2020/splits.json \
        --output results/stage2/gp_fix \
        --phase3_dir results/stage2/phase3_refinement \
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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

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
# SVGP with noise constraint
# ---------------------------------------------------------------------------

class ConstrainedSVGP(ApproximateGP):
    """SVGP with ARD Matérn-5/2 and constrained noise."""

    def __init__(self, inducing_points: torch.Tensor):
        d = inducing_points.shape[1]
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def make_constrained_likelihood(device, noise_lb=0.01, noise_ub=1.0):
    """Gaussian likelihood with noise constrained to [noise_lb, noise_ub]."""
    noise_constraint = gpytorch.constraints.Interval(noise_lb, noise_ub)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint,
    ).to(device)
    # Initialize noise to a reasonable mid-range value
    likelihood.noise = 0.1
    return likelihood


# ---------------------------------------------------------------------------
# DKL Feature Extractor
# ---------------------------------------------------------------------------

class DKLFeatureExtractor(nn.Module):
    """MLP that maps 128-d → reduced_dim before GP."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DKLModel(gpytorch.Module):
    """Deep Kernel Learning: FeatureExtractor + SVGP."""

    def __init__(self, feature_extractor: DKLFeatureExtractor,
                 gp_model: ConstrainedSVGP):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_model = gp_model

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_model(features)


# ---------------------------------------------------------------------------
# Extract frozen SchemeB embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract (X, y) from frozen SchemeB model."""
    model.eval()
    all_z, all_y = [], []
    for batch in loader:
        layers = [l.to(device) for l in batch["layer_embs"]]
        mask = batch["mask"].to(device)
        z_global, _ = model(layers, atom_mask=mask)
        all_z.append(z_global.cpu().numpy())
        all_y.append(batch["pkd"].numpy())
    return np.concatenate(all_z), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# Train SVGP (used by both PCA and DKL paths)
# ---------------------------------------------------------------------------

def train_svgp(
    X_train, y_train, X_val, y_val,
    n_inducing, n_epochs, batch_size, lr, device,
    noise_lb=0.01, noise_ub=1.0,
    patience=50,
):
    """Train SVGP on numpy arrays with constrained noise and early stopping."""
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_v_np = y_val

    N = X.shape[0]
    J = min(n_inducing, N)
    idx = torch.randperm(N)[:J]
    inducing_points = X[idx].clone()

    model = ConstrainedSVGP(inducing_points).to(device)
    likelihood = make_constrained_likelihood(device, noise_lb, noise_ub)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rho = -float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        likelihood.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            output = model(X_b)
            loss = -mll(output, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= N

        # Validation
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(X_v))
            mu_v = pred.mean.cpu().numpy()
        val_m = compute_metrics(y_v_np, mu_v)

        if val_m["spearman_rho"] > best_val_rho:
            best_val_rho = val_m["spearman_rho"]
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "likelihood": {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            noise = likelihood.noise.item()
            logger.info(
                f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                f"val_ρ={val_m['spearman_rho']:.4f}, noise={noise:.4f}"
            )

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state["model"])
    likelihood.load_state_dict(best_state["likelihood"])
    model.eval()
    likelihood.eval()
    return model, likelihood


def train_dkl(
    X_train, y_train, X_val, y_val,
    input_dim, hidden_dim, reduced_dim, n_inducing,
    n_epochs, batch_size, lr, device,
    noise_lb=0.01, noise_ub=1.0,
    patience=50,
):
    """Train DKL model (MLP feature extractor + SVGP) jointly."""
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_v_np = y_val

    N = X.shape[0]

    # Init feature extractor
    feature_extractor = DKLFeatureExtractor(input_dim, hidden_dim, reduced_dim).to(device)

    # Compute initial features for inducing point init
    with torch.no_grad():
        features_init = feature_extractor(X)
    J = min(n_inducing, N)
    idx = torch.randperm(N)[:J]
    inducing_points = features_init[idx].clone()

    gp_model = ConstrainedSVGP(inducing_points).to(device)
    likelihood = make_constrained_likelihood(device, noise_lb, noise_ub)
    dkl = DKLModel(feature_extractor, gp_model).to(device)

    dkl.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": feature_extractor.parameters(), "lr": lr},
        {"params": gp_model.parameters(), "lr": lr},
        {"params": likelihood.parameters(), "lr": lr},
    ])

    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=N)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rho = -float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(n_epochs):
        dkl.train()
        likelihood.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            output = dkl(X_b)
            loss = -mll(output, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= N

        # Validation
        dkl.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(dkl(X_v))
            mu_v = pred.mean.cpu().numpy()
        val_m = compute_metrics(y_v_np, mu_v)

        if val_m["spearman_rho"] > best_val_rho:
            best_val_rho = val_m["spearman_rho"]
            best_state = {
                "dkl": {k: v.cpu().clone() for k, v in dkl.state_dict().items()},
                "likelihood": {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            noise = likelihood.noise.item()
            logger.info(
                f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                f"val_ρ={val_m['spearman_rho']:.4f}, noise={noise:.4f}"
            )

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    dkl.load_state_dict(best_state["dkl"])
    likelihood.load_state_dict(best_state["likelihood"])
    dkl.eval()
    likelihood.eval()
    return dkl, likelihood


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_gp(model, likelihood, X_np, y_np, device, label=""):
    """Evaluate GP model on numpy data. Returns metrics + uncertainty info."""
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
        mu = pred.mean.cpu().numpy()
        var = pred.variance.cpu().numpy()

    metrics = compute_metrics(y_np, mu)
    errors = np.abs(y_np - mu)
    sigma = np.sqrt(var)
    unc_rho, unc_p = spearmanr(errors, sigma)

    if label:
        logger.info(
            f"  {label}: R²={metrics['R2']:.4f}, ρ={metrics['spearman_rho']:.4f}, "
            f"|err|-σ ρ={unc_rho:.4f}, noise={likelihood.noise.item():.4f}"
        )

    return {
        **metrics,
        "uncertainty": {
            "error_sigma_rho": float(unc_rho),
            "error_sigma_p": float(unc_p),
            "mean_sigma": float(sigma.mean()),
            "noise_variance": float(likelihood.noise.item()),
        },
    }


def eval_dkl(dkl, likelihood, X_np, y_np, device, label=""):
    """Evaluate DKL model on numpy data."""
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(dkl(X))
        mu = pred.mean.cpu().numpy()
        var = pred.variance.cpu().numpy()

    metrics = compute_metrics(y_np, mu)
    errors = np.abs(y_np - mu)
    sigma = np.sqrt(var)
    unc_rho, unc_p = spearmanr(errors, sigma)

    if label:
        logger.info(
            f"  {label}: R²={metrics['R2']:.4f}, ρ={metrics['spearman_rho']:.4f}, "
            f"|err|-σ ρ={unc_rho:.4f}, noise={likelihood.noise.item():.4f}"
        )

    return {
        **metrics,
        "uncertainty": {
            "error_sigma_rho": float(unc_rho),
            "error_sigma_p": float(unc_p),
            "mean_sigma": float(sigma.mean()),
            "noise_variance": float(likelihood.noise.item()),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GP fix experiments")
    parser.add_argument("--atom_emb_dir", type=str, default="results/atom_embeddings")
    parser.add_argument("--labels", type=str, default="data/pdbbind_v2020/labels.csv")
    parser.add_argument("--splits", type=str, default="data/pdbbind_v2020/splits.json")
    parser.add_argument("--output", type=str, default="results/stage2/gp_fix")
    parser.add_argument("--phase3_dir", type=str,
                        default="results/stage2/phase3_refinement")
    # SchemeB params (must match s14 training)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--attn_hidden_dim", type=int, default=64)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    # GP params
    parser.add_argument("--n_inducing", type=int, default=256)
    parser.add_argument("--gp_epochs", type=int, default=500)
    parser.add_argument("--gp_batch_size", type=int, default=256)
    parser.add_argument("--gp_lr", type=float, default=0.005)
    parser.add_argument("--gp_patience", type=int, default=80)
    # General
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
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
        train_ds, batch_size=args.batch_size, shuffle=False,
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

    # -----------------------------------------------------------------------
    # Load frozen SchemeB model from Phase 3 A3.4 Step 1
    # -----------------------------------------------------------------------
    from bayesdiff.attention_pool import SchemeB_SingleBranch

    phase3_results_path = Path(args.phase3_dir) / "phase3_results.json"
    assert phase3_results_path.exists(), f"Phase 3 results not found: {phase3_results_path}"

    # Load the Step 1 model checkpoint
    step1_ckpt_path = Path(args.phase3_dir) / "A34_step1_model.pt"

    if step1_ckpt_path.exists():
        logger.info(f"Loading Step 1 model from {step1_ckpt_path}")
        step1_state = torch.load(step1_ckpt_path, map_location="cpu")
    else:
        # Retrain Step 1 if checkpoint not saved
        logger.info("Step 1 checkpoint not found, retraining SchemeB...")
        from scripts.pipeline.s14_phase3_refinement import train_scheme_b_mlp

        class Args:
            pass
        retrain_args = Args()
        retrain_args.embed_dim = args.embed_dim
        retrain_args.attn_hidden_dim = args.attn_hidden_dim
        retrain_args.entropy_weight = args.entropy_weight
        retrain_args.lr = 1e-3
        retrain_args.n_epochs = 200
        retrain_args.patience = 30

        # Shuffle train loader for retraining
        train_loader_shuffle = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_atom_emb, num_workers=0, pin_memory=True,
        )

        res = train_scheme_b_mlp(
            train_loader_shuffle, val_loader, test_loader,
            retrain_args, device,
            entropy_weight=args.entropy_weight,
            exp_name="Step1-retrain",
        )
        step1_state = res["model_state"]
        # Save for future use
        torch.save(step1_state, step1_ckpt_path)
        logger.info(f"Saved Step 1 model to {step1_ckpt_path}")

    # Load SchemeB
    model = SchemeB_SingleBranch(
        embed_dim=args.embed_dim,
        n_layers=10,
        attn_hidden_dim=args.attn_hidden_dim,
        entropy_weight=args.entropy_weight,
    ).to(device)
    model.load_state_dict(step1_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Extract embeddings
    # -----------------------------------------------------------------------
    logger.info("Extracting frozen SchemeB embeddings...")
    X_train, y_train = extract_embeddings(model, train_loader, device)
    X_val, y_val = extract_embeddings(model, val_loader, device)
    X_test, y_test = extract_embeddings(model, test_loader, device)
    logger.info(
        f"Embeddings: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    # Standardize (important for GP kernel)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}
    t_start = time.time()

    # ===================================================================
    # A3.4b: PCA (128 → 32) → SVGP
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("=== A3.4b: PCA(128→32) → SVGP (constrained noise) ===")
    logger.info("=" * 70)

    pca32 = PCA(n_components=32, random_state=42)
    X_tr_pca32 = pca32.fit_transform(X_train_s)
    X_va_pca32 = pca32.transform(X_val_s)
    X_te_pca32 = pca32.transform(X_test_s)
    logger.info(f"PCA 32: explained variance ratio = {pca32.explained_variance_ratio_.sum():.4f}")

    gp32, lik32 = train_svgp(
        X_tr_pca32, y_train, X_va_pca32, y_val,
        n_inducing=args.n_inducing, n_epochs=args.gp_epochs,
        batch_size=args.gp_batch_size, lr=args.gp_lr, device=device,
        patience=args.gp_patience,
    )
    val_m = eval_gp(gp32, lik32, X_va_pca32, y_val, device, "A3.4b Val")
    test_m = eval_gp(gp32, lik32, X_te_pca32, y_test, device, "A3.4b Test")
    results["A3.4b_PCA32"] = {
        "val": val_m, "test": test_m,
        "pca_var_explained": float(pca32.explained_variance_ratio_.sum()),
    }

    # ===================================================================
    # A3.4d: PCA (128 → 16) → SVGP
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("=== A3.4d: PCA(128→16) → SVGP (constrained noise) ===")
    logger.info("=" * 70)

    pca16 = PCA(n_components=16, random_state=42)
    X_tr_pca16 = pca16.fit_transform(X_train_s)
    X_va_pca16 = pca16.transform(X_val_s)
    X_te_pca16 = pca16.transform(X_test_s)
    logger.info(f"PCA 16: explained variance ratio = {pca16.explained_variance_ratio_.sum():.4f}")

    gp16, lik16 = train_svgp(
        X_tr_pca16, y_train, X_va_pca16, y_val,
        n_inducing=args.n_inducing, n_epochs=args.gp_epochs,
        batch_size=args.gp_batch_size, lr=args.gp_lr, device=device,
        patience=args.gp_patience,
    )
    val_m = eval_gp(gp16, lik16, X_va_pca16, y_val, device, "A3.4d Val")
    test_m = eval_gp(gp16, lik16, X_te_pca16, y_test, device, "A3.4d Test")
    results["A3.4d_PCA16"] = {
        "val": val_m, "test": test_m,
        "pca_var_explained": float(pca16.explained_variance_ratio_.sum()),
    }

    # ===================================================================
    # A3.4c: DKL (128 → 32 MLP → SVGP, joint training)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("=== A3.4c: DKL (128→32 MLP + SVGP, jointly trained) ===")
    logger.info("=" * 70)

    dkl, lik_dkl = train_dkl(
        X_train_s, y_train, X_val_s, y_val,
        input_dim=128, hidden_dim=64, reduced_dim=32,
        n_inducing=args.n_inducing, n_epochs=args.gp_epochs,
        batch_size=args.gp_batch_size, lr=args.gp_lr, device=device,
        patience=args.gp_patience,
    )
    val_m = eval_dkl(dkl, lik_dkl, X_val_s, y_val, device, "A3.4c Val")
    test_m = eval_dkl(dkl, lik_dkl, X_test_s, y_test, device, "A3.4c Test")
    results["A3.4c_DKL"] = {"val": val_m, "test": test_m}

    # ===================================================================
    # Summary
    # ===================================================================
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 80)
    logger.info("GP FIX RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"{'Exp':<30} {'Val R²':<10} {'Val ρ':<10} {'Test R²':<10} "
        f"{'Test ρ':<10} {'|err|-σ ρ':<12} {'noise':<8}"
    )
    logger.info("-" * 90)

    for name, r in sorted(results.items()):
        unc = r["test"].get("uncertainty", {})
        logger.info(
            f"{name:<30} "
            f"{r['val']['R2']:.4f}     "
            f"{r['val']['spearman_rho']:.4f}     "
            f"{r['test']['R2']:.4f}     "
            f"{r['test']['spearman_rho']:.4f}     "
            f"{unc.get('error_sigma_rho', 0):.4f}       "
            f"{unc.get('noise_variance', 0):.4f}"
        )

    logger.info("=" * 80)
    logger.info(f"\nTotal time: {elapsed:.0f}s")

    # Comparison with Phase 3 reference
    logger.info("\nReference (from Phase 3):")
    logger.info("  A3.4-Step1 MLP: Test R²=0.572, ρ=0.761")
    logger.info("  A3.4-Step2 SVGP (no fix): Test R²=0.507, ρ=0.719, noise=1.654")

    # Save
    results["args"] = vars(args)
    results["elapsed_seconds"] = elapsed
    out_file = output_dir / "gp_fix_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
