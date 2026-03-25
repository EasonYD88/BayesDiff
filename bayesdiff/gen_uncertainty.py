"""
bayesdiff/gen_uncertainty.py
────────────────────────────
Generation-side uncertainty estimation (U_gen).

Given M samples from a diffusion model for a single pocket:
  1. Compute sample covariance Σ̂_gen with Ledoit-Wolf shrinkage
  2. Detect multimodality with GMM (BIC-based K selection)
  3. Return per-pocket U_gen summary

Phase 1 implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenUncertaintyResult:
    """Container for generation-side uncertainty estimates."""

    z_bar: np.ndarray  # (d,) mean embedding
    cov_gen: np.ndarray  # (d, d) Ledoit-Wolf shrunk covariance
    n_modes: int  # number of GMM modes (1 if unimodal)
    gmm_means: np.ndarray | None = None  # (K, d) GMM cluster centers
    gmm_weights: np.ndarray | None = None  # (K,) mixture weights
    trace_cov: float = 0.0  # scalar summary: Tr(Σ̂_gen)
    raw_embeddings: np.ndarray | None = None  # (M, d) original embeddings


def estimate_gen_uncertainty(
    embeddings: np.ndarray,
    shrinkage: str = "ledoit_wolf",
    detect_modes: bool = True,
    max_modes: int = 5,
    bic_threshold: float = 10.0,
) -> GenUncertaintyResult:
    """Estimate generation uncertainty from M sampled embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (M, d) where M is number of samples per pocket.
    shrinkage : str
        Covariance shrinkage method: "ledoit_wolf", "oas", or "none".
    detect_modes : bool
        Whether to fit GMM for multimodality detection.
    max_modes : int
        Maximum number of GMM components to try.
    bic_threshold : float
        BIC improvement threshold to prefer K > 1.

    Returns
    -------
    GenUncertaintyResult
    """
    M, d = embeddings.shape

    if M < 2:
        logger.warning("Only 1 sample; returning zero covariance")
        return GenUncertaintyResult(
            z_bar=embeddings[0],
            cov_gen=np.zeros((d, d)),
            n_modes=1,
            trace_cov=0.0,
            raw_embeddings=embeddings,
        )

    # ── 1. Compute covariance with shrinkage ─────────────────────────
    z_bar = embeddings.mean(axis=0)

    if shrinkage == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(embeddings)
        cov_gen = lw.covariance_
        logger.info(f"  Ledoit-Wolf shrinkage: α={lw.shrinkage_:.4f}")
    elif shrinkage == "oas":
        from sklearn.covariance import OAS

        oas = OAS().fit(embeddings)
        cov_gen = oas.covariance_
    else:
        cov_gen = np.cov(embeddings.T)

    trace_cov = np.trace(cov_gen)

    # ── 2. Multimodality detection via GMM ───────────────────────────
    n_modes = 1
    gmm_means = None
    gmm_weights = None

    if detect_modes and M >= 2 * max_modes:
        from sklearn.mixture import GaussianMixture

        bics = []
        for k in range(1, max_modes + 1):
            gmm = GaussianMixture(
                n_components=k, covariance_type="full", random_state=42, n_init=3
            ).fit(embeddings)
            bics.append(gmm.bic(embeddings))

        # Select K with best BIC, but prefer K=1 unless improvement is clear
        best_k = np.argmin(bics) + 1
        if best_k > 1 and (bics[0] - bics[best_k - 1]) < bic_threshold:
            best_k = 1  # Not enough evidence for multimodality

        if best_k > 1:
            gmm_final = GaussianMixture(
                n_components=best_k,
                covariance_type="full",
                random_state=42,
                n_init=3,
            ).fit(embeddings)
            n_modes = best_k
            gmm_means = gmm_final.means_  # (K, d)
            gmm_weights = gmm_final.weights_  # (K,)
            gmm_covs = gmm_final.covariances_  # (K, d, d)

            # Aggregate GMM global mean and covariance (math_explain §2.3):
            #   z̄ = Σ π_k μ_k
            #   Σ_gen = Σ π_k [Σ_k + (μ_k - z̄)(μ_k - z̄)ᵀ]
            z_bar = gmm_weights @ gmm_means  # (d,)
            cov_gen = np.zeros((d, d))
            for k in range(best_k):
                diff_k = gmm_means[k] - z_bar
                cov_gen += gmm_weights[k] * (
                    gmm_covs[k] + np.outer(diff_k, diff_k)
                )
            trace_cov = np.trace(cov_gen)

            logger.info(
                f"  GMM detected {n_modes} modes (BIC improvement: "
                f"{bics[0] - bics[best_k - 1]:.1f})"
            )
        else:
            logger.info("  Unimodal (K=1)")

    return GenUncertaintyResult(
        z_bar=z_bar,
        cov_gen=cov_gen,
        n_modes=n_modes,
        gmm_means=gmm_means,
        gmm_weights=gmm_weights,
        trace_cov=trace_cov,
        raw_embeddings=embeddings,
    )
