"""
bayesdiff/ood.py — §4.4 OOD Detection Module
──────────────────────────────────────────────
Out-of-Distribution detection for generated molecules.
Paper reference: §4.4 "Mahalanobis OOD Detection"
Equation reference: doc/Stage_1/03_math_reference.md §7

Implements:
  - Mahalanobis distance (primary, from plan)
  - Relative Mahalanobis (vs uniform background, more robust)
  - Integration with fusion module: OOD flag -> confidence caveat
  - Batch scoring with percentile reporting

Phase 1 implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """OOD detection result for a single query."""
    mahalanobis_distance: float
    relative_mahalanobis: Optional[float]
    is_ood: bool
    percentile: float
    confidence_modifier: float


class MahalanobisOOD:
    """Mahalanobis distance-based OOD detector.

    d_M(z) = sqrt((z - mu)^T Sigma^-1 (z - mu))

    Also computes relative Mahalanobis distance using a background
    distribution (isotropic Gaussian) for better robustness in high dims.
    """

    def __init__(self):
        self._mu = None
        self._cov_inv = None
        self._train_distances = None
        self._threshold = None
        self._bg_mu = None
        self._bg_cov_inv = None
        self._bg_distances = None
        self._percentile_val = 95.0

    def fit(self, X_train, percentile=95.0, regularization=1e-5,
            fit_background=True, background_scale=5.0):
        """Fit OOD detector on training embeddings.

        Parameters
        ----------
        X_train : (N, d)
        percentile : training Mahalanobis percentile for threshold
        regularization : Tikhonov regularization for covariance
        fit_background : fit background distribution for relative Mahalanobis
        background_scale : scale factor for isotropic background
        """
        N, d = X_train.shape
        self._mu = X_train.mean(axis=0)
        self._percentile_val = percentile

        cov = np.cov(X_train.T)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        cov += np.eye(d) * regularization
        self._cov_inv = np.linalg.inv(cov)

        self._train_distances = np.array([self._mahalanobis(x) for x in X_train])
        self._threshold = float(np.percentile(self._train_distances, percentile))

        if fit_background:
            spread = np.trace(cov) / d
            self._bg_mu = self._mu.copy()
            self._bg_cov_inv = np.eye(d) / (background_scale * spread)
            self._bg_distances = np.array(
                [self._mahalanobis_bg(x) for x in X_train]
            )

        logger.info(
            "OOD detector: N=%d, d=%d, threshold(p%.0f)=%.2f, mean_d=%.2f",
            N, d, percentile, self._threshold, self._train_distances.mean()
        )

    def _mahalanobis(self, z):
        diff = z - self._mu
        return float(np.sqrt(max(diff @ self._cov_inv @ diff, 0.0)))

    def _mahalanobis_bg(self, z):
        diff = z - self._bg_mu
        return float(np.sqrt(max(diff @ self._bg_cov_inv @ diff, 0.0)))

    def score(self, z):
        """Score a single embedding z of shape (d,)."""
        assert self._mu is not None, "Not fitted."
        d_m = self._mahalanobis(z)
        is_ood = d_m > self._threshold
        pct = float((self._train_distances < d_m).mean() * 100)

        rel_m = None
        if self._bg_cov_inv is not None:
            d_bg = self._mahalanobis_bg(z)
            rel_m = d_m - d_bg

        if d_m <= self._threshold:
            conf_mod = 1.0
        else:
            excess = (d_m - self._threshold) / max(self._threshold, 1.0)
            conf_mod = float(np.exp(-2.0 * excess))

        return OODResult(
            mahalanobis_distance=d_m,
            relative_mahalanobis=rel_m,
            is_ood=is_ood,
            percentile=pct,
            confidence_modifier=conf_mod,
        )

    def score_batch(self, X):
        """Score batch of embeddings. X : (N, d)"""
        return [self.score(x) for x in X]

    def get_distances(self, X):
        """Return raw Mahalanobis distances for batch."""
        return np.array([self._mahalanobis(x) for x in X])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mu": self._mu,
            "cov_inv": self._cov_inv,
            "train_distances": self._train_distances,
            "threshold": self._threshold,
            "percentile": self._percentile_val,
        }
        if self._bg_cov_inv is not None:
            data["bg_mu"] = self._bg_mu
            data["bg_cov_inv"] = self._bg_cov_inv
            data["bg_distances"] = self._bg_distances
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self._mu = data["mu"]
        self._cov_inv = data["cov_inv"]
        self._train_distances = data["train_distances"]
        self._threshold = float(data["threshold"])
        self._percentile_val = float(data.get("percentile", 95.0))
        if "bg_mu" in data:
            self._bg_mu = data["bg_mu"]
            self._bg_cov_inv = data["bg_cov_inv"]
            self._bg_distances = data["bg_distances"]
