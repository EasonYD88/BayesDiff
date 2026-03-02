"""
bayesdiff/evaluate.py
─────────────────────
Evaluation metrics for BayesDiff.

Computes all metrics from plan_opendata.md §6.2:
  - ECE (Expected Calibration Error)
  - AUROC (P_success as classifier)
  - EF@1% (Enrichment Factor)
  - Hit Rate @ P >= threshold
  - Spearman ρ (μ_oracle vs pKd_true)
  - RMSE (μ_oracle vs pKd_true)
  - NLL (Gaussian negative log-likelihood)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    """Container for all evaluation metrics."""

    ece: float
    auroc: float
    ef_1pct: float
    hit_rate: float  # at P >= confidence_threshold
    spearman_rho: float
    spearman_pval: float
    rmse: float
    nll: float
    n_samples: int
    confidence_threshold: float = 0.85
    y_target: float = 7.0


def evaluate_all(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    p_success: np.ndarray,
    y_true: np.ndarray,
    y_target: float = 7.0,
    confidence_threshold: float = 0.85,
) -> EvalResults:
    """Compute all evaluation metrics.

    Parameters
    ----------
    mu_pred : np.ndarray, shape (N,)
        Predicted mean pKd.
    sigma_pred : np.ndarray, shape (N,)
        Predicted std dev (σ_total).
    p_success : np.ndarray, shape (N,)
        Calibrated P_success.
    y_true : np.ndarray, shape (N,)
        True experimental pKd.
    y_target : float
        Activity threshold in pKd.
    confidence_threshold : float
        Threshold for "high confidence" hit rate.

    Returns
    -------
    EvalResults
    """
    y_binary = (y_true >= y_target).astype(float)

    # ── ECE ───────────────────────────────────────────────────────────
    from bayesdiff.calibration import compute_ece

    ece = compute_ece(p_success, y_binary)

    # ── AUROC ─────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score

    try:
        auroc = roc_auc_score(y_binary, p_success)
    except ValueError:
        auroc = float("nan")

    # ── EF@1% ─────────────────────────────────────────────────────────
    ef_1pct = enrichment_factor(p_success, y_binary, fraction=0.01)

    # ── Hit Rate @ high confidence ────────────────────────────────────
    high_conf_mask = p_success >= confidence_threshold
    if high_conf_mask.sum() > 0:
        hit_rate = y_binary[high_conf_mask].mean()
    else:
        hit_rate = float("nan")

    # ── Spearman ρ ────────────────────────────────────────────────────
    rho, pval = stats.spearmanr(mu_pred, y_true)

    # ── RMSE ──────────────────────────────────────────────────────────
    rmse = np.sqrt(np.mean((mu_pred - y_true) ** 2))

    # ── NLL ───────────────────────────────────────────────────────────
    nll = gaussian_nll(mu_pred, sigma_pred, y_true)

    return EvalResults(
        ece=ece,
        auroc=auroc,
        ef_1pct=ef_1pct,
        hit_rate=hit_rate,
        spearman_rho=rho,
        spearman_pval=pval,
        rmse=rmse,
        nll=nll,
        n_samples=len(y_true),
        confidence_threshold=confidence_threshold,
        y_target=y_target,
    )


def enrichment_factor(
    scores: np.ndarray,
    labels: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Compute Enrichment Factor at a given fraction.

    EF@f = (hits_in_top_f / n_top_f) / (total_hits / N)
    """
    N = len(scores)
    n_top = max(1, int(N * fraction))
    total_hits = labels.sum()

    if total_hits == 0:
        return 0.0

    # Sort by score descending
    top_idx = np.argsort(scores)[-n_top:]
    hits_in_top = labels[top_idx].sum()

    expected_rate = total_hits / N
    observed_rate = hits_in_top / n_top

    return float(observed_rate / expected_rate) if expected_rate > 0 else 0.0


def gaussian_nll(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
) -> float:
    """Average Gaussian negative log-likelihood.

    NLL = 0.5 * [log(2πσ²) + (y - μ)² / σ²]
    """
    sigma_clipped = np.clip(sigma, 1e-6, None)
    nll_per = 0.5 * (
        np.log(2 * np.pi * sigma_clipped**2)
        + ((y - mu) ** 2) / (sigma_clipped**2)
    )
    return float(nll_per.mean())


def print_results(results: EvalResults):
    """Pretty-print evaluation results."""
    logger.info("=" * 50)
    logger.info(f"Evaluation Results (N={results.n_samples})")
    logger.info(f"  y_target = {results.y_target}, "
                f"conf_thresh = {results.confidence_threshold}")
    logger.info("-" * 50)
    logger.info(f"  ECE:          {results.ece:.4f}")
    logger.info(f"  AUROC:        {results.auroc:.4f}")
    logger.info(f"  EF@1%:        {results.ef_1pct:.2f}")
    logger.info(f"  Hit Rate:     {results.hit_rate:.4f}")
    logger.info(f"  Spearman ρ:   {results.spearman_rho:.4f} "
                f"(p={results.spearman_pval:.2e})")
    logger.info(f"  RMSE:         {results.rmse:.4f}")
    logger.info(f"  NLL:          {results.nll:.4f}")
    logger.info("=" * 50)
