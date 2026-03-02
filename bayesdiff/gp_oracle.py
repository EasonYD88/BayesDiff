"""
bayesdiff/gp_oracle.py
──────────────────────
Sparse Variational Gaussian Process (SVGP) for oracle uncertainty.

Predicts μ_oracle(z) and σ²_oracle(z) given encoder embedding z.
Uses GPyTorch with ARD Matérn-5/2 kernel and J=512 inducing points.

Phase 1 implementation (single-fidelity: experimental pKd only).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

logger = logging.getLogger(__name__)


class SVGPModel(ApproximateGP):
    """Sparse Variational GP with ARD Matérn-5/2 kernel.

    Parameters
    ----------
    inducing_points : Tensor
        Shape (J, d) inducing point locations.
    """

    def __init__(self, inducing_points: torch.Tensor):
        d = inducing_points.shape[1]
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPOracle:
    """Wrapper for training/inference with the SVGP oracle.

    Parameters
    ----------
    d : int
        Embedding dimension.
    n_inducing : int
        Number of inducing points.
    device : str
        "cpu", "cuda", or "mps".
    """

    def __init__(self, d: int = 128, n_inducing: int = 512, device: str = "cpu"):
        self.d = d
        self.n_inducing = n_inducing
        self.device = torch.device(device)
        self.model = None
        self.likelihood = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> dict:
        """Train the SVGP model.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, d)
        y_train : np.ndarray, shape (N,)
        n_epochs : int
        batch_size : int
        lr : float
        verbose : bool

        Returns
        -------
        dict with training history
        """
        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        N = X.shape[0]

        # Initialize inducing points via k-means or random subset
        if N <= self.n_inducing:
            inducing_points = X.clone()
        else:
            # Random subset for simplicity; could use k-means
            idx = torch.randperm(N)[: self.n_inducing]
            inducing_points = X[idx].clone()

        self.model = SVGPModel(inducing_points).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lr,
        )

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=N)

        history = {"loss": []}
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= N
            history["loss"].append(epoch_loss)

            if verbose and (epoch + 1) % 20 == 0:
                noise = self.likelihood.noise.item()
                logger.info(
                    f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                    f"noise={noise:.4f}"
                )

        self.model.eval()
        self.likelihood.eval()
        return history

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict μ_oracle and σ²_oracle for input embeddings.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)

        Returns
        -------
        mu : np.ndarray, shape (N,) — predicted pKd
        var : np.ndarray, shape (N,) — predictive variance
        """
        assert self.model is not None, "Model not trained. Call train() first."

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_t))
            mu = pred.mean.cpu().numpy()
            var = pred.variance.cpu().numpy()

        return mu, var

    def predict_with_jacobian(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict μ, σ², and Jacobian ∂μ/∂z for delta method fusion.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)

        Returns
        -------
        mu : np.ndarray, shape (N,)
        var : np.ndarray, shape (N,)
        J_mu : np.ndarray, shape (N, d) — Jacobian of mean w.r.t. input
        """
        assert self.model is not None, "Model not trained."

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)

        # Forward pass with gradient tracking
        self.model.eval()
        self.likelihood.eval()

        pred = self.likelihood(self.model(X_t))
        mu_t = pred.mean

        # Compute Jacobian via backward
        J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            mu_t[i].backward(retain_graph=True)
            J_rows.append(X_t.grad[i].clone().cpu().numpy())

        J_mu = np.stack(J_rows, axis=0)

        mu = mu_t.detach().cpu().numpy()
        var = pred.variance.detach().cpu().numpy()

        return mu, var, J_mu

    def save(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "likelihood_state": self.likelihood.state_dict(),
                "d": self.d,
                "n_inducing": self.n_inducing,
            },
            path,
        )
        logger.info(f"GP model saved to {path}")

    def load(self, path: str | Path, X_dummy: np.ndarray | None = None):
        """Load model checkpoint.

        Parameters
        ----------
        path : path to .pt file
        X_dummy : np.ndarray
            Dummy inducing points for model initialization.
            If None, uses zeros of shape (n_inducing, d).
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.d = ckpt["d"]
        self.n_inducing = ckpt["n_inducing"]

        if X_dummy is None:
            X_dummy = np.zeros((self.n_inducing, self.d), dtype=np.float32)

        inducing = torch.tensor(X_dummy[: self.n_inducing], dtype=torch.float32)
        self.model = SVGPModel(inducing).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.likelihood.load_state_dict(ckpt["likelihood_state"])
        self.model.eval()
        self.likelihood.eval()

        logger.info(f"GP model loaded from {path}")
