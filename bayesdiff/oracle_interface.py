"""
bayesdiff/oracle_interface.py — Unified Oracle Interface (Sub-Plan 04, §4.2)

OracleResult: dataclass for oracle head predictions.
OracleHead: abstract base class for all oracle heads.

All oracle heads must implement:
    - fit(X_train, y_train, X_val, y_val, **kwargs) -> dict
    - predict(X) -> OracleResult  (fast, no Jacobian)
    - predict_for_fusion(X) -> OracleResult  (expensive, with Jacobian)
    - save(path) / load(path)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class OracleResult:
    """Container for oracle head predictions.

    Compatible with:
        bayesdiff/fusion.py: fuse_uncertainties(mu_oracle, sigma2_oracle, J_mu, cov_gen, ...)
        bayesdiff/evaluate.py: evaluate_all(mu_pred, sigma_pred, p_success, y_true, ...)

    jacobian is Optional — only populated by predict_for_fusion(), not by predict().
    This avoids expensive per-sample backward passes during evaluation and diagnostics.
    """

    mu: np.ndarray  # (N,) predicted means (pKd scale)
    sigma2: np.ndarray  # (N,) predicted total variance (must be > 0)
    jacobian: Optional[np.ndarray] = None  # (N, d) ∂μ/∂z — only from predict_for_fusion()
    aux: dict = field(default_factory=dict)
    # Standard aux keys (all optional, all shape (N,)):
    #   'sigma2_aleatoric' : within-model / likelihood variance
    #   'sigma2_epistemic' : between-model / ensemble disagreement variance
    #   'sigma2_gp'        : GP posterior variance component
    #   'sigma2_nn'        : NN epistemic (MC Dropout) variance component
    #   'member_mus'       : (M, N) array of per-member predictions (ensemble only)
    #   'member_sigma2s'   : (M, N) array of per-member variances (ensemble only)


class OracleHead(ABC):
    """Base class for all oracle heads in Sub-Plan 04.

    Subclasses:
        DKLOracle, DKLEnsembleOracle, NNResidualOracle,
        PCA_GPOracle (Tier 1), SNGPOracle, EvidentialOracle (Tier 1b)

    Integration points:
        - predict() → bayesdiff/evaluate.py (fast path, no Jacobian)
        - predict_for_fusion() → bayesdiff/fusion.py (expensive, with Jacobian)
        - bayesdiff/ood.py: MahalanobisOOD.score(z) or score_batch(X)
        - bayesdiff/calibration.py: IsotonicCalibrator.fit(p_raw, y_binary)
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> dict:
        """Train the oracle head on frozen embeddings.

        Parameters
        ----------
        X_train : (N_train, d) float32 embeddings
        y_train : (N_train,) float32 pKd labels
        X_val : (N_val, d) float32 embeddings
        y_val : (N_val,) float32 pKd labels

        Returns
        -------
        history : dict with at least 'loss' key (list of per-epoch values),
                  plus method-specific keys (e.g. 'val_rho', 'val_nll')
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> OracleResult:
        """Predict mu and sigma2 (fast path — no Jacobian).

        Used for evaluation, diagnostics, validation monitoring, and ablation tables.
        Returns OracleResult with jacobian=None.

        Parameters
        ----------
        X : (N, d) float32 embeddings

        Returns
        -------
        OracleResult with mu (N,), sigma2 (N,), jacobian=None, aux dict
        """
        ...

    @abstractmethod
    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Predict mu, sigma2, AND Jacobian ∂μ/∂z (expensive path).

        Only call when entering the generation-uncertainty fusion stage
        (bayesdiff/fusion.py). For all other uses, call predict() instead.

        Parameters
        ----------
        X : (N, d) float32 embeddings

        Returns
        -------
        OracleResult with mu (N,), sigma2 (N,), jacobian (N, d), aux dict
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model checkpoint. Convention: path is a directory."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from checkpoint directory."""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray, y_target: float = 7.0) -> dict:
        """Convenience: predict + compute standard metrics.

        Returns dict with keys: R2, spearman_rho, rmse, nll, err_sigma_rho.
        """
        from bayesdiff.evaluate import gaussian_nll
        from scipy.stats import spearmanr

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
