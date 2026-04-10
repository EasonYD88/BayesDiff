"""
bayesdiff/hybrid_oracle.py — Uncertainty-Aware Hybrid Oracle Heads (Sub-Plan 04, §4.3)

Implements the oracle head family:
  - FeatureExtractor:     MLP feature extractor for DKL (shared)
  - DKLSVGPModel:         SVGP in DKL feature space
  - DKLOracle:            Deep Kernel Learning (g_θ MLP + SVGP)
  - DKLEnsembleOracle:    M independent DKL models
  - NNResidualOracle:     MLP readout + GP on residuals
  - PCA_GPOracle:         PCA dimensionality reduction + SVGP
  - SNGPOracle:           Spectral Normalized Neural GP (RFF approximation)
  - EvidentialOracle:     Normal-Inverse-Gamma evidential regression

All classes implement OracleHead ABC from oracle_interface.py.
predict() returns (mu, sigma2) only; predict_for_fusion() adds Jacobian.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from scipy.stats import spearmanr

from bayesdiff.oracle_interface import OracleHead, OracleResult
from bayesdiff.gp_oracle import GPOracle, SVGPModel

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Extractor (shared by DKL and DKL Ensemble)
# ============================================================================


class FeatureExtractor(nn.Module):
    """MLP feature extractor for DKL with optional residual connection.

    Architecture (default, 2-layer):
        z (128) → Linear(128, 256) → ReLU → Dropout(0.1) → Linear(256, 32) → + W_proj·z → u (32)

    Parameter count (default): 128×256 + 256 + 256×32 + 32 + 128×32 + 32 ≈ 45,344
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 32,
        n_layers: int = 2,
        residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.residual = residual

        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        if residual:
            self.proj = nn.Linear(input_dim, output_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        if self.residual:
            nn.init.xavier_normal_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, input_dim) → u: (B, output_dim)"""
        h = self.mlp(z)
        if self.residual:
            h = h + self.proj(z)
        return h


# ============================================================================
# DKL SVGP Model
# ============================================================================


class DKLSVGPModel(gpytorch.models.ApproximateGP):
    """SVGP operating in DKL feature space.

    Identical to bayesdiff.gp_oracle.SVGPModel but constructed with
    feature_dim instead of full embedding dim.
    """

    def __init__(self, inducing_points: torch.Tensor):
        d = inducing_points.shape[1]
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
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


# ============================================================================
# B1: DKL Oracle
# ============================================================================


class DKLOracle(OracleHead):
    """Deep Kernel Learning oracle: FeatureExtractor (MLP) + SVGP.

    Input: z ∈ ℝ^128 (frozen embedding)
    Feature extraction: u = g_θ(z) ∈ ℝ^32
    GP: f(u) ~ GP(m(u), k(u,u')), k = ScaleKernel(Matérn-5/2, ARD)

    Training: Joint ELBO maximization over θ, kernel hyperparams, variational params.
    Prediction: mu = E[f(g_θ(z))], sigma2 = Var[f(g_θ(z))] + noise
    Jacobian: ∂μ/∂z via torch.autograd
    """

    def __init__(
        self,
        input_dim: int = 128,
        feature_dim: int = 32,
        n_inducing: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 2,
        residual: bool = True,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.n_inducing = n_inducing
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.residual = residual
        self.dropout = dropout
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dim, feature_dim, n_layers, residual, dropout
        ).to(self.device)

        self.gp_model: Optional[DKLSVGPModel] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_epochs=300,
        batch_size=256,
        lr_nn=1e-3,
        lr_gp=1e-2,
        weight_decay=1e-4,
        patience=20,
        verbose=True,
    ) -> dict:
        """Joint training of feature extractor + SVGP.

        Optimizer: AdamW with per-parameter-group learning rates.
        Loss: -ELBO from gpytorch.mlls.VariationalELBO.
        Early stopping: on val Spearman ρ (patience epochs).
        """
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        N = len(X_t)

        # Initialize inducing points in feature space
        with torch.no_grad():
            u_init = self.feature_extractor(X_t)
        if N <= self.n_inducing:
            inducing = u_init.clone()
        else:
            idx = torch.randperm(N)[: self.n_inducing]
            inducing = u_init[idx].clone()

        self.gp_model = DKLSVGPModel(inducing).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.feature_extractor.train()
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.feature_extractor.parameters(),
                    "lr": lr_nn,
                    "weight_decay": weight_decay,
                },
                {"params": self.gp_model.hyperparameters(), "lr": lr_gp},
                {"params": self.gp_model.variational_parameters(), "lr": lr_gp},
                {"params": self.likelihood.parameters(), "lr": lr_gp},
            ]
        )

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_model, num_data=N)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        history: dict = {"loss": [], "val_rho": [], "val_nll": []}
        best_val_rho = -float("inf")
        best_state: Optional[dict] = None
        patience_counter = 0

        for epoch in range(n_epochs):
            # --- Train ---
            self.feature_extractor.train()
            self.gp_model.train()
            self.likelihood.train()
            epoch_loss = 0.0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                u = self.feature_extractor(X_batch)
                output = self.gp_model(u)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= N
            history["loss"].append(epoch_loss)

            # --- Validate ---
            val_result = self._predict_internal(X_v)
            val_rho, _ = spearmanr(val_result.mu, y_val)
            val_nll = float(
                np.mean(
                    0.5 * np.log(2 * np.pi * val_result.sigma2)
                    + (y_val - val_result.mu) ** 2 / (2 * val_result.sigma2)
                )
            )
            history["val_rho"].append(val_rho)
            history["val_nll"].append(val_nll)

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_state = {
                    "fe": {
                        k: v.cpu().clone()
                        for k, v in self.feature_extractor.state_dict().items()
                    },
                    "gp": {
                        k: v.cpu().clone()
                        for k, v in self.gp_model.state_dict().items()
                    },
                    "lik": {
                        k: v.cpu().clone()
                        for k, v in self.likelihood.state_dict().items()
                    },
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"  DKL Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                    f"val_ρ={val_rho:.4f}, val_NLL={val_nll:.4f}, "
                    f"noise={self.likelihood.noise.item():.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"  DKL early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state:
            self.feature_extractor.load_state_dict(best_state["fe"])
            self.gp_model.load_state_dict(best_state["gp"])
            self.likelihood.load_state_dict(best_state["lik"])

        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()

        return history

    def _predict_internal(self, X: torch.Tensor | np.ndarray) -> OracleResult:
        """Internal fast-path predict (used by fit() for val monitoring)."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            u = self.feature_extractor(X)
            pred = self.likelihood(self.gp_model(u))
            mu = pred.mean.cpu().numpy()
            var = pred.variance.cpu().numpy()

        return OracleResult(mu=mu, sigma2=var, aux={})

    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2, no Jacobian."""
        return self._predict_internal(X)

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Full prediction with Jacobian ∂μ/∂z for Delta method fusion."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)

        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()

        # Forward
        u = self.feature_extractor(X_t)
        pred = self.likelihood(self.gp_model(u))
        mu_t = pred.mean
        var = pred.variance.detach().cpu().numpy()

        # Jacobian: ∂μ/∂z
        J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            mu_t[i].backward(retain_graph=True)
            J_rows.append(X_t.grad[i].clone().cpu().numpy())

        J_mu = np.stack(J_rows, axis=0)  # (N, d)
        mu = mu_t.detach().cpu().numpy()

        return OracleResult(mu=mu, sigma2=var, jacobian=J_mu, aux={})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "gp_model": self.gp_model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "feature_dim": self.feature_dim,
                    "n_inducing": self.n_inducing,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                    "residual": self.residual,
                    "dropout": self.dropout,
                },
            },
            path / "dkl_model.pt",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path / "dkl_model.pt", map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        # Rebuild feature extractor if config differs
        self.feature_extractor = FeatureExtractor(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["feature_dim"],
            n_layers=cfg["n_layers"],
            residual=cfg["residual"],
            dropout=cfg["dropout"],
        ).to(self.device)

        # Re-initialize GP with dummy inducing points
        dummy_inducing = torch.zeros(cfg["n_inducing"], cfg["feature_dim"])
        self.gp_model = DKLSVGPModel(dummy_inducing).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.feature_extractor.load_state_dict(ckpt["feature_extractor"])
        self.gp_model.load_state_dict(ckpt["gp_model"])
        self.likelihood.load_state_dict(ckpt["likelihood"])
        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()


# ============================================================================
# B2: DKL Ensemble Oracle
# ============================================================================


class DKLEnsembleOracle(OracleHead):
    """Ensemble of M independent DKL models.

    Uncertainty = mean(member variances) + Var(member means).
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_members: int = 5,
        bootstrap: bool = True,
        bootstrap_frac: float = 0.8,
        **dkl_kwargs,
    ):
        self.input_dim = input_dim
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.bootstrap_frac = bootstrap_frac
        self.dkl_kwargs = dkl_kwargs
        self.members = [
            DKLOracle(input_dim=input_dim, **dkl_kwargs) for _ in range(n_members)
        ]

    def fit(self, X_train, y_train, X_val, y_val, seed_base=42, **kwargs) -> dict:
        """Train each member with different seed and optionally bootstrap data."""
        histories = []
        for m, member in enumerate(self.members):
            torch.manual_seed(seed_base + m)
            np.random.seed(seed_base + m)

            if self.bootstrap:
                N = len(X_train)
                idx = np.random.choice(
                    N, size=int(N * self.bootstrap_frac), replace=False
                )
                X_m, y_m = X_train[idx], y_train[idx]
            else:
                X_m, y_m = X_train, y_train

            logger.info(
                f"  Training DKL ensemble member {m+1}/{self.n_members} (N={len(X_m)})"
            )
            h = member.fit(X_m, y_m, X_val, y_val, **kwargs)
            histories.append(h)

        return {"member_histories": histories}

    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast ensemble prediction: mu, sigma2, decomposition — no Jacobian."""
        member_results = [m.predict(X) for m in self.members]

        mus = np.stack([r.mu for r in member_results])  # (M, N)
        sigma2s = np.stack([r.sigma2 for r in member_results])  # (M, N)

        mu_ens = mus.mean(axis=0)  # (N,)
        sigma2_aleatoric = sigma2s.mean(axis=0)  # (N,)
        sigma2_epistemic = mus.var(axis=0)  # (N,)
        sigma2_total = sigma2_aleatoric + sigma2_epistemic  # (N,)

        return OracleResult(
            mu=mu_ens,
            sigma2=sigma2_total,
            aux={
                "sigma2_aleatoric": sigma2_aleatoric,
                "sigma2_epistemic": sigma2_epistemic,
                "member_mus": mus,
                "member_sigma2s": sigma2s,
            },
        )

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Expensive ensemble prediction with Jacobian (for Delta method fusion)."""
        member_results = [m.predict_for_fusion(X) for m in self.members]

        mus = np.stack([r.mu for r in member_results])  # (M, N)
        sigma2s = np.stack([r.sigma2 for r in member_results])  # (M, N)
        jacs = np.stack([r.jacobian for r in member_results])  # (M, N, d)

        mu_ens = mus.mean(axis=0)
        sigma2_aleatoric = sigma2s.mean(axis=0)
        sigma2_epistemic = mus.var(axis=0)
        sigma2_total = sigma2_aleatoric + sigma2_epistemic
        J_ens = jacs.mean(axis=0)  # (N, d)

        return OracleResult(
            mu=mu_ens,
            sigma2=sigma2_total,
            jacobian=J_ens,
            aux={
                "sigma2_aleatoric": sigma2_aleatoric,
                "sigma2_epistemic": sigma2_epistemic,
                "member_mus": mus,
                "member_sigma2s": sigma2s,
            },
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for m, member in enumerate(self.members):
            member.save(path / f"member_{m}")
        with open(path / "ensemble_config.json", "w") as f:
            json.dump(
                {
                    "n_members": self.n_members,
                    "bootstrap": self.bootstrap,
                    "bootstrap_frac": self.bootstrap_frac,
                },
                f,
                indent=2,
            )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path / "ensemble_config.json") as f:
            cfg = json.load(f)
        self.n_members = cfg["n_members"]
        # Rebuild members if count changed
        while len(self.members) < self.n_members:
            self.members.append(DKLOracle(input_dim=self.input_dim, **self.dkl_kwargs))
        for m in range(self.n_members):
            self.members[m].load(path / f"member_{m}")


# ============================================================================
# B3: NN + GP Residual Oracle
# ============================================================================


class NNResidualOracle(OracleHead):
    """Two-stage predictor: MLP for main signal + GP on residuals.

    Stage 1: Train MLP (MSE + early stopping)
    Stage 2: Compute residuals r = y - MLP(z), train GPOracle on (z, r)
    Prediction: mu = MLP(z) + GP_mu(z), sigma2 = GP_sigma2(z) + MC_Dropout_var(z)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        n_inducing: int = 512,
        mc_dropout: bool = True,
        mc_samples: int = 20,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_inducing = n_inducing
        self.mc_dropout = mc_dropout
        self.mc_samples = mc_samples
        self.dropout = dropout
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.gp = GPOracle(d=input_dim, n_inducing=n_inducing, device=str(self.device))

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        nn_epochs=200,
        nn_lr=1e-3,
        nn_patience=30,
        gp_epochs=200,
        gp_batch_size=256,
        gp_lr=0.01,
        batch_size=64,
        verbose=True,
        **kwargs,
    ) -> dict:
        """Two-stage training."""
        history: dict = {"nn_loss": [], "nn_val_rho": [], "gp_loss": []}

        # ---- Stage 1: Train NN ----
        logger.info("  NNResidual Stage 1: Training NN (MSE)...")
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(
            self.nn.parameters(), lr=nn_lr, weight_decay=1e-4
        )
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        best_val_rho = -float("inf")
        best_nn_state: Optional[dict] = None
        patience_counter = 0

        for epoch in range(nn_epochs):
            self.nn.train()
            total_loss, n_total = 0.0, 0
            for xb, yb in loader:
                pred = self.nn(xb).squeeze(-1)
                loss = F.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                n_total += len(yb)

            history["nn_loss"].append(total_loss / n_total)

            # Val
            self.nn.eval()
            with torch.no_grad():
                val_pred = self.nn(X_v).squeeze(-1).cpu().numpy()
            val_rho, _ = spearmanr(val_pred, y_val)
            history["nn_val_rho"].append(val_rho)

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_nn_state = {
                    k: v.cpu().clone() for k, v in self.nn.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 50 == 0:
                logger.info(
                    f"    NN Epoch {epoch+1}: MSE={total_loss/n_total:.4f}, val_ρ={val_rho:.4f}"
                )

            if patience_counter >= nn_patience:
                logger.info(f"    NN early stopping at epoch {epoch+1}")
                break

        if best_nn_state:
            self.nn.load_state_dict(best_nn_state)
        self.nn.eval()

        # ---- Stage 2: Train GP on residuals ----
        logger.info("  NNResidual Stage 2: Training GP on residuals...")
        with torch.no_grad():
            nn_pred_train = self.nn(X_t).squeeze(-1).cpu().numpy()
        r_train = y_train - nn_pred_train

        logger.info(
            f"    Residual stats: mean={r_train.mean():.4f}, std={r_train.std():.4f}"
        )
        logger.info(
            f"    Original label std={y_train.std():.4f} → residual std={r_train.std():.4f}"
        )

        gp_history = self.gp.train(
            X_train,
            r_train,
            n_epochs=gp_epochs,
            batch_size=gp_batch_size,
            lr=gp_lr,
            verbose=verbose,
        )
        history["gp_loss"] = gp_history["loss"]

        return history

    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2, no Jacobian.

        Core uncertainty is from residual GP. MC Dropout is an optional
        auxiliary epistemic term (switchable via mc_dropout flag).
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        # NN prediction (deterministic)
        self.nn.eval()
        with torch.no_grad():
            nn_mu = self.nn(X_t).squeeze(-1).cpu().numpy()

        # GP prediction on residuals (no Jacobian)
        gp_mu, gp_var = self.gp.predict(X)

        # MC Dropout uncertainty (if enabled) — auxiliary epistemic term
        sigma2_mc = np.zeros(len(X))
        if self.mc_dropout:
            self.nn.train()  # enable dropout
            mc_preds = []
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    mc_pred = self.nn(X_t).squeeze(-1).cpu().numpy()
                    mc_preds.append(mc_pred)
            mc_preds_arr = np.stack(mc_preds)  # (T, N)
            sigma2_mc = mc_preds_arr.var(axis=0)
            self.nn.eval()

        mu = nn_mu + gp_mu
        sigma2 = gp_var + sigma2_mc

        return OracleResult(
            mu=mu,
            sigma2=sigma2,
            aux={
                "sigma2_gp": gp_var,
                "sigma2_nn": sigma2_mc,
                "nn_mu": nn_mu,
                "gp_mu": gp_mu,
                "residual_std": float(np.sqrt(gp_var.mean())),
            },
        )

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Full prediction with Jacobian ∂μ/∂z for Delta method fusion."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)

        # NN prediction (with gradient tracking for Jacobian)
        self.nn.eval()
        nn_pred = self.nn(X_t).squeeze(-1)

        # NN Jacobian
        nn_J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            nn_pred[i].backward(retain_graph=True)
            nn_J_rows.append(X_t.grad[i].clone().cpu().numpy())
        nn_J = np.stack(nn_J_rows)  # (N, d)
        nn_mu = nn_pred.detach().cpu().numpy()

        # GP prediction on residuals (with Jacobian)
        gp_mu, gp_var, gp_J = self.gp.predict_with_jacobian(X)

        # MC Dropout uncertainty (if enabled)
        sigma2_mc = np.zeros(len(X))
        if self.mc_dropout:
            self.nn.train()  # enable dropout
            mc_preds = []
            with torch.no_grad():
                X_t_nograd = torch.tensor(X, dtype=torch.float32, device=self.device)
                for _ in range(self.mc_samples):
                    mc_pred = self.nn(X_t_nograd).squeeze(-1).cpu().numpy()
                    mc_preds.append(mc_pred)
            mc_preds_arr = np.stack(mc_preds)  # (T, N)
            sigma2_mc = mc_preds_arr.var(axis=0)
            self.nn.eval()

        mu = nn_mu + gp_mu
        sigma2 = gp_var + sigma2_mc
        J_mu = nn_J + gp_J

        return OracleResult(
            mu=mu,
            sigma2=sigma2,
            jacobian=J_mu,
            aux={
                "sigma2_gp": gp_var,
                "sigma2_nn": sigma2_mc,
                "nn_mu": nn_mu,
                "gp_mu": gp_mu,
                "residual_std": float(np.sqrt(gp_var.mean())),
            },
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.nn.state_dict(), path / "nn_model.pt")
        self.gp.save(path / "gp_model.pt")
        with open(path / "nn_residual_config.json", "w") as f:
            json.dump(
                {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "n_inducing": self.n_inducing,
                    "mc_dropout": self.mc_dropout,
                    "mc_samples": self.mc_samples,
                    "dropout": self.dropout,
                },
                f,
                indent=2,
            )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.nn.load_state_dict(
            torch.load(path / "nn_model.pt", map_location=self.device, weights_only=False)
        )
        self.nn.eval()
        self.gp.load(path / "gp_model.pt")


# ============================================================================
# C.3: PCA + GP Oracle
# ============================================================================


class PCA_GPOracle(OracleHead):
    """PCA dimensionality reduction → existing GPOracle.

    Reduces input from 128d to pca_dim (default 32) via PCA fit on training data,
    then trains a standard SVGP on the reduced features.
    """

    def __init__(
        self,
        input_dim: int = 128,
        pca_dim: int = 32,
        n_inducing: int = 512,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.pca_dim = pca_dim
        self.n_inducing = n_inducing
        self.device_str = device

        self.pca = None  # sklearn PCA, fitted in fit()
        self.gp = GPOracle(
            d=pca_dim, n_inducing=n_inducing, device=device
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA transform."""
        return self.pca.transform(X).astype(np.float32)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_epochs=200,
        batch_size=256,
        lr=0.01,
        verbose=True,
        **kwargs,
    ) -> dict:
        """Fit PCA on training data, then train GP on reduced features."""
        from sklearn.decomposition import PCA

        logger.info(f"  PCA_GP: Fitting PCA({self.pca_dim}) on {X_train.shape}...")
        self.pca = PCA(n_components=self.pca_dim)
        X_train_pca = self.pca.fit_transform(X_train).astype(np.float32)
        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {explained:.4f}")

        logger.info("  PCA_GP: Training SVGP on reduced features...")
        history = self.gp.train(
            X_train_pca,
            y_train,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )
        return history

    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2, no Jacobian."""
        X_pca = self._transform(X)
        mu, var = self.gp.predict(X_pca)
        return OracleResult(mu=mu, sigma2=var, aux={"pca_explained_var": float(self.pca.explained_variance_ratio_.sum())})

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Prediction with Jacobian in original input space.

        J_original = J_pca @ W_pca^T  where W_pca is (pca_dim, input_dim)
        """
        X_pca = self._transform(X)
        mu, var, J_pca = self.gp.predict_with_jacobian(X_pca)

        # Chain rule: ∂μ/∂z = ∂μ/∂u · ∂u/∂z = J_pca @ W_pca
        W_pca = self.pca.components_  # (pca_dim, input_dim)
        J_original = J_pca @ W_pca  # (N, input_dim)

        return OracleResult(
            mu=mu,
            sigma2=var,
            jacobian=J_original,
            aux={"pca_explained_var": float(self.pca.explained_variance_ratio_.sum())},
        )

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "pca.pkl", "wb") as f:
            pickle.dump(self.pca, f)
        self.gp.save(path / "gp_model.pt")
        with open(path / "pca_gp_config.json", "w") as f:
            json.dump(
                {
                    "input_dim": self.input_dim,
                    "pca_dim": self.pca_dim,
                    "n_inducing": self.n_inducing,
                },
                f,
                indent=2,
            )

    def load(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        with open(path / "pca.pkl", "rb") as f:
            self.pca = pickle.load(f)
        self.gp.load(path / "gp_model.pt")


# ============================================================================
# C1: SNGP (Spectral Normalized Neural GP) Oracle
# ============================================================================


class _RandomFourierFeatureLayer(nn.Module):
    """Approximate GP output layer using Random Fourier Features (RFF).

    Implements the Gaussian RBF kernel approximation:
        φ(x) = sqrt(2/D) * cos(Wx + b)
    where W ~ N(0, 1/lengthscale^2), b ~ Uniform(0, 2π).

    The posterior mean and variance are computed via Woodbury identity:
        Σ = (Φ^T Φ + I/β)^{-1}
        μ = Σ Φ^T y
    At test time, for input x:
        μ(x) = φ(x)^T w,  σ²(x) = φ(x)^T Σ φ(x)
    """

    def __init__(self, in_features: int, n_rff: int = 1024,
                 lengthscale: float = 1.0, output_scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.n_rff = n_rff

        # Fixed random weights for RFF (not learned)
        self.register_buffer(
            "W", torch.randn(in_features, n_rff) / lengthscale
        )
        self.register_buffer(
            "b", torch.rand(n_rff) * 2 * np.pi
        )

        # Learnable output weight and precision
        self.beta = nn.Parameter(torch.tensor(1.0))  # precision (1/noise_var)
        self.output_scale = nn.Parameter(torch.tensor(output_scale))

        # Posterior covariance (updated during training via reset_covariance / update_covariance)
        self.register_buffer("precision", torch.eye(n_rff))  # Σ^{-1}
        self.register_buffer("mean_weight", torch.zeros(n_rff))  # posterior mean weight

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RFF features: φ(x) = sqrt(2/D) * cos(Wx + b)."""
        proj = x @ self.W + self.b  # (B, n_rff)
        return (2.0 / self.n_rff) ** 0.5 * torch.cos(proj) * self.output_scale

    def forward(self, x: torch.Tensor):
        """Return (mean, variance) predictions."""
        phi = self._phi(x)  # (B, D)
        mu = phi @ self.mean_weight  # (B,)

        # Variance: φ^T Σ φ, where Σ = precision^{-1}
        # Use solve instead of explicit inverse for numerical stability
        # v = Σ φ^T = precision^{-1} @ φ^T
        v = torch.linalg.solve(self.precision, phi.T)  # (D, B)
        var = (phi.T * v).sum(dim=0)  # (B,)
        var = var.clamp(min=1e-8)

        return mu, var

    def reset_covariance(self):
        """Reset precision matrix to prior: Σ^{-1} = I."""
        self.precision.copy_(torch.eye(self.n_rff, device=self.precision.device))
        self.mean_weight.zero_()

    def update_covariance(self, x: torch.Tensor, y: torch.Tensor):
        """Update posterior precision with a batch of data.

        precision += β * Φ^T Φ
        After all batches, call finalize_covariance() to compute posterior mean.
        """
        phi = self._phi(x)  # (B, D)
        beta = F.softplus(self.beta)
        self.precision.add_(beta * (phi.T @ phi))

    def finalize_covariance(self, dataloader, hidden_fn):
        """Compute posterior mean weight: w = Σ * β * Σ_batches Φ^T y.

        Must be called after all update_covariance() calls.
        """
        beta = F.softplus(self.beta)
        # Accumulate Φ^T y
        phi_ty = torch.zeros(self.n_rff, device=self.precision.device)
        for x_batch, y_batch in dataloader:
            h = hidden_fn(x_batch)
            phi = self._phi(h)
            phi_ty.add_(beta * (phi.T @ y_batch))

        # w = Σ * (Φ^T y) = precision^{-1} @ phi_ty
        self.mean_weight.copy_(
            torch.linalg.solve(self.precision, phi_ty)
        )


class SNGPOracle(OracleHead):
    """Spectral Normalized Neural GP (SNGP) Oracle.

    Architecture:
        z ∈ ℝ^128 → SN-MLP (2 layers, 256 hidden) → RFF-GP (1024 features) → ŷ, σ²

    Spectral normalization constrains hidden-layer Lipschitz constant,
    preserving input-space distances → meaningful GP variance.

    Reference: Liu et al., "Simple and Principled Uncertainty Estimation
    with Deterministic Deep Learning via Distance Awareness", ICML 2020.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_rff: int = 1024,
        lengthscale: float = 1.0,
        spectral_norm_bound: float = 0.95,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_rff = n_rff
        self.lengthscale = lengthscale
        self.spectral_norm_bound = spectral_norm_bound
        self.dropout = dropout
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build spectral-normalized MLP backbone
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            # Apply spectral norm with bound
            lin = torch.nn.utils.parametrizations.spectral_norm(lin)
            layers.append(lin)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers).to(self.device)

        # RFF-GP output layer
        self.rff_layer = _RandomFourierFeatureLayer(
            in_features=hidden_dim, n_rff=n_rff,
            lengthscale=lengthscale,
        ).to(self.device)

    def _hidden(self, X: torch.Tensor) -> torch.Tensor:
        """Forward through SN backbone only."""
        return self.backbone(X)

    def fit(
        self,
        X_train, y_train, X_val, y_val,
        n_epochs=300, batch_size=256,
        lr=1e-3, weight_decay=1e-4,
        patience=20, verbose=True,
        **kwargs,
    ) -> dict:
        """Two-phase training:
        Phase 1: Train SN-MLP backbone with NLL loss (mu from linear readout, fixed sigma2).
        Phase 2: Compute RFF covariance for posterior GP variance.
        """
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        N = len(X_t)

        # Temporary linear head for training backbone
        linear_head = nn.Linear(self.hidden_dim, 1).to(self.device)
        log_noise = nn.Parameter(torch.tensor(0.0, device=self.device))

        all_params = (
            list(self.backbone.parameters())
            + list(linear_head.parameters())
            + [log_noise]
        )
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: dict = {"loss": [], "val_rho": []}
        best_val_rho = -float("inf")
        best_state = None
        patience_counter = 0

        # Phase 1: Train backbone + linear head with Gaussian NLL
        for epoch in range(n_epochs):
            self.backbone.train()
            linear_head.train()
            epoch_loss = 0.0

            for xb, yb in loader:
                h = self.backbone(xb)
                mu = linear_head(h).squeeze(-1)
                noise_var = torch.exp(log_noise).clamp(min=1e-6)
                loss = 0.5 * torch.mean(
                    (yb - mu) ** 2 / noise_var + torch.log(noise_var)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)

            epoch_loss /= N
            history["loss"].append(epoch_loss)

            # Validate
            self.backbone.eval()
            linear_head.eval()
            with torch.no_grad():
                h_v = self.backbone(X_v)
                mu_v = linear_head(h_v).squeeze(-1).cpu().numpy()
            val_rho, _ = spearmanr(mu_v, y_val)
            history["val_rho"].append(val_rho)

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_state = {
                    "backbone": {k: v.cpu().clone() for k, v in self.backbone.state_dict().items()},
                    "log_noise": log_noise.detach().cpu().clone(),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"  SNGP Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, val_ρ={val_rho:.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"  SNGP early stopping at epoch {epoch+1}")
                break

        # Restore best backbone
        if best_state:
            self.backbone.load_state_dict(best_state["backbone"])

        self.backbone.eval()

        # Phase 2: Compute RFF posterior covariance
        logger.info("  SNGP Phase 2: Computing RFF posterior covariance...")
        self.rff_layer.reset_covariance()

        with torch.no_grad():
            for xb, yb in loader:
                h = self.backbone(xb)
                self.rff_layer.update_covariance(h, yb)

        # Finalize: compute posterior mean weight
        self.rff_layer.finalize_covariance(
            loader,
            lambda x: self.backbone(x),
        )

        return history

    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2, no Jacobian."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(X_t)
            mu, var = self.rff_layer(h)
            mu = mu.cpu().numpy()
            var = var.cpu().numpy()

        return OracleResult(mu=mu, sigma2=var, aux={})

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Full prediction with Jacobian ∂μ/∂z for Delta method fusion."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)

        self.backbone.eval()
        h = self.backbone(X_t)
        mu, var = self.rff_layer(h)

        # Jacobian ∂μ/∂z
        J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            mu[i].backward(retain_graph=True)
            J_rows.append(X_t.grad[i].clone().cpu().numpy())

        J_mu = np.stack(J_rows, axis=0)
        mu_np = mu.detach().cpu().numpy()
        var_np = var.detach().cpu().numpy()

        return OracleResult(mu=mu_np, sigma2=var_np, jacobian=J_mu, aux={})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "rff_layer": self.rff_layer.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                    "n_rff": self.n_rff,
                    "lengthscale": self.lengthscale,
                    "spectral_norm_bound": self.spectral_norm_bound,
                    "dropout": self.dropout,
                },
            },
            path / "sngp_model.pt",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path / "sngp_model.pt", map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        # Rebuild backbone
        layers = []
        dims = [cfg["input_dim"]] + [cfg["hidden_dim"]] * cfg["n_layers"]
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            lin = torch.nn.utils.parametrizations.spectral_norm(lin)
            layers.append(lin)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg["dropout"]))
        self.backbone = nn.Sequential(*layers).to(self.device)

        self.rff_layer = _RandomFourierFeatureLayer(
            in_features=cfg["hidden_dim"], n_rff=cfg["n_rff"],
            lengthscale=cfg["lengthscale"],
        ).to(self.device)

        self.backbone.load_state_dict(ckpt["backbone"])
        self.rff_layer.load_state_dict(ckpt["rff_layer"])
        self.backbone.eval()

    def evaluate(self, X: np.ndarray, y: np.ndarray, y_target: float = 7.0) -> dict:
        """Override base evaluate for consistency."""
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
            "R2": float(r2),
            "spearman_rho": float(rho),
            "rmse": float(rmse),
            "nll": float(nll),
            "err_sigma_rho": float(err_sigma_rho),
            "err_sigma_p": float(err_sigma_p),
            "mean_sigma": float(sigma.mean()),
        }


# ============================================================================
# C2: Evidential Regression Oracle
# ============================================================================


class EvidentialOracle(OracleHead):
    """Evidential Regression Oracle — Normal-Inverse-Gamma posterior.

    Architecture:
        z ∈ ℝ^128 → MLP (2 layers, 256→128 hidden) → (γ, ν, α, β) heads

    Learns to output parameters of a NIG distribution over the prediction:
        γ: predicted mean
        ν: evidence for the mean (> 0, via Softplus)
        α: shape of the IG prior (> 1, via Softplus + 1)
        β: scale of the IG prior (> 0, via Softplus)

    Uncertainty decomposition:
        σ²_aleatoric = β / (α - 1)
        σ²_epistemic = β / (ν * (α - 1))
        σ²_total = σ²_aleatoric + σ²_epistemic

    Training: NIG NLL + evidence regularization (Amini et al., NeurIPS 2020).

    Reference: Amini et al., "Deep Evidential Regression", NeurIPS 2020.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        mid_dim: int = 128,
        dropout: float = 0.1,
        lambda_ev: float = 0.1,
        lambda_anneal_epochs: int = 50,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.dropout = dropout
        self.lambda_ev = lambda_ev
        self.lambda_anneal_epochs = lambda_anneal_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ).to(self.device)

        # 4 output heads
        self.gamma_head = nn.Linear(mid_dim, 1).to(self.device)  # mean
        self.nu_head = nn.Linear(mid_dim, 1).to(self.device)     # evidence for mean
        self.alpha_head = nn.Linear(mid_dim, 1).to(self.device)  # IG shape
        self.beta_head = nn.Linear(mid_dim, 1).to(self.device)   # IG scale

        self._init_weights()

    def _init_weights(self):
        for m in [self.backbone, self.gamma_head, self.nu_head, self.alpha_head, self.beta_head]:
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.kaiming_normal_(p.weight, nonlinearity="relu")
                    nn.init.zeros_(p.bias)

    def _all_parameters(self):
        return (
            list(self.backbone.parameters())
            + list(self.gamma_head.parameters())
            + list(self.nu_head.parameters())
            + list(self.alpha_head.parameters())
            + list(self.beta_head.parameters())
        )

    def _forward(self, X: torch.Tensor):
        """Forward pass → (gamma, nu, alpha, beta) all (B,)."""
        h = self.backbone(X)
        gamma = self.gamma_head(h).squeeze(-1)
        nu = F.softplus(self.nu_head(h).squeeze(-1)) + 1e-6       # > 0
        alpha = F.softplus(self.alpha_head(h).squeeze(-1)) + 1.0   # > 1
        beta = F.softplus(self.beta_head(h).squeeze(-1)) + 1e-6    # > 0
        return gamma, nu, alpha, beta

    @staticmethod
    def _nig_nll(y: torch.Tensor, gamma: torch.Tensor,
                 nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """Negative log-likelihood of NIG distribution.

        NLL = 0.5*log(π/ν) - α*log(Ω) + (α+0.5)*log((y-γ)²ν + Ω) + log(Γ(α)/Γ(α+0.5))
        where Ω = 2β(1+ν)
        """
        omega = 2.0 * beta * (1.0 + nu)
        nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        return nll.mean()

    @staticmethod
    def _evidence_reg(y: torch.Tensor, gamma: torch.Tensor,
                      nu: torch.Tensor, alpha: torch.Tensor):
        """Evidence regularization: penalize evidence on incorrect predictions.

        L_reg = |y - γ| * (2ν + α)
        """
        return (torch.abs(y - gamma) * (2.0 * nu + alpha)).mean()

    def fit(
        self,
        X_train, y_train, X_val, y_val,
        n_epochs=300, batch_size=256,
        lr=1e-3, weight_decay=1e-4,
        patience=20, verbose=True,
        **kwargs,
    ) -> dict:
        """Train with NIG NLL + annealed evidence regularization."""
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        N = len(X_t)

        optimizer = torch.optim.AdamW(
            self._all_parameters(), lr=lr, weight_decay=weight_decay
        )

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: dict = {"loss": [], "val_rho": [], "val_nll": []}
        best_val_rho = -float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            self.backbone.train()
            epoch_loss = 0.0

            # Anneal regularization coefficient
            if self.lambda_anneal_epochs > 0:
                coeff = min(1.0, epoch / self.lambda_anneal_epochs) * self.lambda_ev
            else:
                coeff = self.lambda_ev

            for xb, yb in loader:
                gamma, nu, alpha, beta = self._forward(xb)
                nig_loss = self._nig_nll(yb, gamma, nu, alpha, beta)
                reg_loss = self._evidence_reg(yb, gamma, nu, alpha)
                loss = nig_loss + coeff * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)

            epoch_loss /= N
            history["loss"].append(epoch_loss)

            # Validate
            self.backbone.eval()
            with torch.no_grad():
                gamma_v, nu_v, alpha_v, beta_v = self._forward(X_v)
                mu_v = gamma_v.cpu().numpy()
                sigma2_alea = (beta_v / (alpha_v - 1.0)).cpu().numpy()
                sigma2_epi = (beta_v / (nu_v * (alpha_v - 1.0))).cpu().numpy()
                sigma2_v = sigma2_alea + sigma2_epi

            val_rho, _ = spearmanr(mu_v, y_val)
            val_nll = float(
                np.mean(0.5 * np.log(2 * np.pi * np.clip(sigma2_v, 1e-10, None))
                        + (y_val - mu_v) ** 2 / (2 * np.clip(sigma2_v, 1e-10, None)))
            )
            history["val_rho"].append(val_rho)
            history["val_nll"].append(val_nll)

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_state = {
                    "backbone": {k: v.cpu().clone() for k, v in self.backbone.state_dict().items()},
                    "gamma_head": {k: v.cpu().clone() for k, v in self.gamma_head.state_dict().items()},
                    "nu_head": {k: v.cpu().clone() for k, v in self.nu_head.state_dict().items()},
                    "alpha_head": {k: v.cpu().clone() for k, v in self.alpha_head.state_dict().items()},
                    "beta_head": {k: v.cpu().clone() for k, v in self.beta_head.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"  Evidential Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                    f"val_ρ={val_rho:.4f}, val_NLL={val_nll:.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"  Evidential early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state:
            self.backbone.load_state_dict(best_state["backbone"])
            self.gamma_head.load_state_dict(best_state["gamma_head"])
            self.nu_head.load_state_dict(best_state["nu_head"])
            self.alpha_head.load_state_dict(best_state["alpha_head"])
            self.beta_head.load_state_dict(best_state["beta_head"])

        self.backbone.eval()
        return history

    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2 (aleatoric + epistemic), no Jacobian."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.backbone.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = self._forward(X_t)

        gamma_np = gamma.cpu().numpy()
        nu_np = nu.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        beta_np = beta.cpu().numpy()

        alpha_m1 = np.clip(alpha_np - 1.0, 1e-6, None)
        sigma2_alea = beta_np / alpha_m1
        sigma2_epi = beta_np / (nu_np * alpha_m1)
        sigma2_total = sigma2_alea + sigma2_epi

        return OracleResult(
            mu=gamma_np,
            sigma2=np.clip(sigma2_total, 1e-10, None),
            aux={
                "sigma2_aleatoric": sigma2_alea,
                "sigma2_epistemic": sigma2_epi,
                "nu": nu_np,
                "alpha": alpha_np,
                "beta": beta_np,
            },
        )

    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Full prediction with Jacobian ∂γ/∂z for Delta method fusion."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)

        self.backbone.eval()
        gamma, nu, alpha, beta = self._forward(X_t)

        # Jacobian ∂γ/∂z
        J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            gamma[i].backward(retain_graph=True)
            J_rows.append(X_t.grad[i].clone().cpu().numpy())

        J_mu = np.stack(J_rows, axis=0)

        gamma_np = gamma.detach().cpu().numpy()
        nu_np = nu.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()

        alpha_m1 = np.clip(alpha_np - 1.0, 1e-6, None)
        sigma2_alea = beta_np / alpha_m1
        sigma2_epi = beta_np / (nu_np * alpha_m1)
        sigma2_total = sigma2_alea + sigma2_epi

        return OracleResult(
            mu=gamma_np,
            sigma2=np.clip(sigma2_total, 1e-10, None),
            jacobian=J_mu,
            aux={
                "sigma2_aleatoric": sigma2_alea,
                "sigma2_epistemic": sigma2_epi,
                "nu": nu_np,
                "alpha": alpha_np,
                "beta": beta_np,
            },
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "gamma_head": self.gamma_head.state_dict(),
                "nu_head": self.nu_head.state_dict(),
                "alpha_head": self.alpha_head.state_dict(),
                "beta_head": self.beta_head.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "mid_dim": self.mid_dim,
                    "dropout": self.dropout,
                    "lambda_ev": self.lambda_ev,
                    "lambda_anneal_epochs": self.lambda_anneal_epochs,
                },
            },
            path / "evidential_model.pt",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path / "evidential_model.pt", map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        self.backbone = nn.Sequential(
            nn.Linear(cfg["input_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"], cfg["mid_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        ).to(self.device)

        self.gamma_head = nn.Linear(cfg["mid_dim"], 1).to(self.device)
        self.nu_head = nn.Linear(cfg["mid_dim"], 1).to(self.device)
        self.alpha_head = nn.Linear(cfg["mid_dim"], 1).to(self.device)
        self.beta_head = nn.Linear(cfg["mid_dim"], 1).to(self.device)

        self.backbone.load_state_dict(ckpt["backbone"])
        self.gamma_head.load_state_dict(ckpt["gamma_head"])
        self.nu_head.load_state_dict(ckpt["nu_head"])
        self.alpha_head.load_state_dict(ckpt["alpha_head"])
        self.beta_head.load_state_dict(ckpt["beta_head"])

        self.backbone.eval()

    def evaluate(self, X: np.ndarray, y: np.ndarray, y_target: float = 7.0) -> dict:
        """Override base evaluate for consistency."""
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
            "R2": float(r2),
            "spearman_rho": float(rho),
            "rmse": float(rmse),
            "nll": float(nll),
            "err_sigma_rho": float(err_sigma_rho),
            "err_sigma_p": float(err_sigma_p),
            "mean_sigma": float(sigma.mean()),
        }
