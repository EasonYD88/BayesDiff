"""
bayesdiff/ood.py
────────────────
Out-of-Distribution detection via Mahalanobis distance.

Flags inputs that are far from the training embedding distribution.
OOD samples should NOT be trusted for GP predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """OOD detection result for a single query."""

    mahalanobis_distance: float
    is_ood: bool
    percentile: float  # percentile in training distribution


class MahalanobisOOD:
    """Mahalanobis distance-based OOD detector.

    Computes d_M(z) = sqrt((z - μ_train)ᵀ Σ_train⁻¹ (z - μ_train))
    and flags OOD if d_M exceeds a threshold derived from the training set.
    """

    def __init__(self):
        self._mu = None
        self._cov_inv = None
        self._train_distances = None
        self._threshold = None

    def fit(
        self,
        X_train: np.ndarray,
        percentile: float = 95.0,
        regularization: float = 1e-5,
    ):
        """Fit OOD detector on training embeddings.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, d)
        percentile : float
            Percentile of training Mahalanobis distances used as threshold.
        regularization : float
            Regularization added to covariance diagonal.
        """
        self._mu = X_train.mean(axis=0)
        cov = np.cov(X_train.T)
        cov += np.eye(cov.shape[0]) * regularization
        self._cov_inv = np.linalg.inv(cov)

        # Compute training distances for threshold selection
        self._train_distances = np.array(
            [self._mahalanobis(x) for x in X_train]
        )
        self._threshold = np.percentile(self._train_distances, percentile)

        logger.info(
            f"OOD detector fitted on {len(X_train)} samples. "
            f"Threshold (p{percentile:.0f}): {self._threshold:.2f}"
        )

    def _mahalanobis(self, z: np.ndarray) -> float:
        """Compute Mahalanobis distance for a single point."""
        diff = z - self._mu
        return float(np.sqrt(diff @ self._cov_inv @ diff))

    def score(self, z: np.ndarray) -> OODResult:
        """Score a single embedding.

        Parameters
        ----------
        z : np.ndarray, shape (d,)

        Returns
        -------
        OODResult
        """
        assert self._mu is not None, "Detector not fitted."
        d_m = self._mahalanobis(z)
        is_ood = d_m > self._threshold
        # Compute percentile
        pct = float((self._train_distances < d_m).mean() * 100)
        return OODResult(
            mahalanobis_distance=d_m,
            is_ood=is_ood,
            percentile=pct,
        )

    def score_batch(self, X: np.ndarray) -> list[OODResult]:
        """Score a batch of embeddings."""
        return [self.score(x) for x in X]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mu=self._mu,
            cov_inv=self._cov_inv,
            train_distances=self._train_distances,
            threshold=self._threshold,
        )

    def load(self, path: str | Path):
        data = np.load(path)
        self._mu = data["mu"]
        self._cov_inv = data["cov_inv"]
        self._train_distances = data["train_distances"]
        self._threshold = float(data["threshold"])
