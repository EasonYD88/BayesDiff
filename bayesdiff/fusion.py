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
    p_final: float = 0.0  # P_final = w(z) · P_success (after OOD correction)
    ood_confidence: float = 1.0  # w(z) from OOD module
    is_ood: bool = False  # OOD flag from Mahalanobis gate


def fuse_uncertainties(
    mu_oracle: float,
    sigma2_oracle: float,
    J_mu: np.ndarray,
    cov_gen: np.ndarray,
    y_target: float = 7.0,
    hessian_correction: bool = False,
    H_mu: np.ndarray | None = None,
    ood_confidence: float = 1.0,
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
        Hessian d²μ/dz² (required if hessian_correction=True).
    ood_confidence : float
        OOD confidence weight w(z) from MahalanobisOOD.score().
        P_final = w(z) * P_success (math_explain §7.2).

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

    # Law of Total Variance (math_explain §4):
    #   σ²_total ≈ E_z[σ²_oracle(z)] + J_μᵀ Σ̂_gen J_μ
    # Approximation: E_z[σ²_oracle(z)] ≈ σ²_oracle(z̄), valid when oracle
    # variance varies slowly relative to the generation distribution spread.
    sigma2_total = sigma2_oracle + sigma2_gen

    sigma_total = np.sqrt(max(sigma2_total, 1e-10))

    # P_success = P(pKd >= y_target) assuming Gaussian predictive
    # = 1 - Φ((y_target - μ) / σ)
    z_score = (y_target - mu_oracle) / sigma_total
    p_success = 1.0 - stats.norm.cdf(z_score)

    # P_final = w(z) · P_success (math_explain §7.2, §8)
    p_final = ood_confidence * p_success

    return FusionResult(
        mu=mu_oracle,
        sigma2_oracle=sigma2_oracle,
        sigma2_gen=sigma2_gen,
        sigma2_total=sigma2_total,
        sigma_total=sigma_total,
        p_success=p_success,
        p_final=p_final,
        ood_confidence=ood_confidence,
        is_ood=ood_confidence < 1.0,
    )


def fuse_batch(
    mu_oracle: np.ndarray,
    sigma2_oracle: np.ndarray,
    J_mu: np.ndarray,
    cov_gen_list: list[np.ndarray],
    y_target: float = 7.0,
    ood_confidences: np.ndarray | None = None,
) -> list[FusionResult]:
    """Fuse uncertainties for a batch of pockets.

    Parameters
    ----------
    mu_oracle : np.ndarray, shape (N,)
    sigma2_oracle : np.ndarray, shape (N,)
    J_mu : np.ndarray, shape (N, d)
    cov_gen_list : list of (d, d) arrays, length N
    y_target : float
    ood_confidences : np.ndarray | None, shape (N,)
        Per-pocket OOD confidence weights. Defaults to 1.0 (no OOD correction).

    Returns
    -------
    list[FusionResult]
    """
    results = []
    for i in range(len(mu_oracle)):
        ood_c = float(ood_confidences[i]) if ood_confidences is not None else 1.0
        r = fuse_uncertainties(
            mu_oracle=mu_oracle[i],
            sigma2_oracle=sigma2_oracle[i],
            J_mu=J_mu[i],
            cov_gen=cov_gen_list[i],
            y_target=y_target,
            ood_confidence=ood_c,
        )
        results.append(r)
    return results
