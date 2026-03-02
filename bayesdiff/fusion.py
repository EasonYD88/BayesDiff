"""
bayesdiff/fusion.py
───────────────────
Uncertainty fusion via Delta Method (Law of Total Variance).

Combines generation uncertainty (Σ̂_gen) with oracle uncertainty (σ²_oracle)
to produce total predictive variance:

    σ²_total = σ²_oracle + J_μᵀ Σ̂_gen J_μ
             + (optional 2nd-order Hessian correction)

Then computes P_success = 1 - Φ((y_target - μ) / σ_total).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Container for fused uncertainty prediction."""

    mu: float  # predicted pKd
    sigma2_oracle: float  # oracle variance
    sigma2_gen: float  # propagated generation variance (J^T Σ J)
    sigma2_total: float  # total variance
    sigma_total: float  # sqrt(sigma2_total)
    p_success: float  # P(pKd >= y_target)
    is_ood: bool = False  # OOD flag from Mahalanobis gate


def fuse_uncertainties(
    mu_oracle: float,
    sigma2_oracle: float,
    J_mu: np.ndarray,
    cov_gen: np.ndarray,
    y_target: float = 7.0,
    hessian_correction: bool = False,
    H_mu: np.ndarray | None = None,
) -> FusionResult:
    """Fuse generation and oracle uncertainties for a single pocket.

    Parameters
    ----------
    mu_oracle : float
        GP predicted mean pKd at z̄.
    sigma2_oracle : float
        GP predictive variance at z̄.
    J_mu : np.ndarray, shape (d,)
        Jacobian ∂μ_oracle/∂z evaluated at z̄.
    cov_gen : np.ndarray, shape (d, d)
        Ledoit-Wolf shrunk sample covariance from M diffusion samples.
    y_target : float
        Activity threshold in pKd (default 7.0 = 100 nM Kd).
    hessian_correction : bool
        Whether to include 2nd-order Hessian correction.
    H_mu : np.ndarray | None, shape (d, d)
        Hessian ∂²μ/∂z² (required if hessian_correction=True).

    Returns
    -------
    FusionResult
    """
    # 1st order: propagated generation variance
    sigma2_gen = float(J_mu @ cov_gen @ J_mu)

    # 2nd order Hessian correction (optional)
    correction = 0.0
    if hessian_correction and H_mu is not None:
        # Tr(H Σ) / 2 — bias correction term
        correction = 0.5 * np.trace(H_mu @ cov_gen)
        mu_oracle = mu_oracle + correction

    # Law of Total Variance
    sigma2_total = sigma2_oracle + sigma2_gen

    sigma_total = np.sqrt(max(sigma2_total, 1e-10))

    # P_success = P(pKd >= y_target) assuming Gaussian predictive
    # = 1 - Φ((y_target - μ) / σ)
    z_score = (y_target - mu_oracle) / sigma_total
    p_success = 1.0 - stats.norm.cdf(z_score)

    return FusionResult(
        mu=mu_oracle,
        sigma2_oracle=sigma2_oracle,
        sigma2_gen=sigma2_gen,
        sigma2_total=sigma2_total,
        sigma_total=sigma_total,
        p_success=p_success,
    )


def fuse_batch(
    mu_oracle: np.ndarray,
    sigma2_oracle: np.ndarray,
    J_mu: np.ndarray,
    cov_gen_list: list[np.ndarray],
    y_target: float = 7.0,
) -> list[FusionResult]:
    """Fuse uncertainties for a batch of pockets.

    Parameters
    ----------
    mu_oracle : np.ndarray, shape (N,)
    sigma2_oracle : np.ndarray, shape (N,)
    J_mu : np.ndarray, shape (N, d)
    cov_gen_list : list of (d, d) arrays, length N
    y_target : float

    Returns
    -------
    list[FusionResult]
    """
    results = []
    for i in range(len(mu_oracle)):
        r = fuse_uncertainties(
            mu_oracle=mu_oracle[i],
            sigma2_oracle=sigma2_oracle[i],
            J_mu=J_mu[i],
            cov_gen=cov_gen_list[i],
            y_target=y_target,
        )
        results.append(r)
    return results
