"""
bayesdiff/evaluate.py — §5 Evaluation Module
──────────────────────────────────────────────
Evaluation metrics for BayesDiff.
Paper reference: §5 "Experiments"

Computes all metrics from 04_opendata_plan.md section 6.2:
  - ECE (Expected Calibration Error)
  - AUROC (P_success as classifier)
  - EF@1% (Enrichment Factor)
  - Hit Rate @ P >= threshold
  - Spearman rho (mu_oracle vs pKd_true)
  - RMSE (mu_oracle vs pKd_true)
  - NLL (Gaussian negative log-likelihood)

Phase 1 additions:
  - Multi-threshold evaluation (y=7, y=8)
  - Bootstrap confidence intervals
  - Per-pocket metric breakdown
  - JSON/CSV serialization
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    """Container for all evaluation metrics."""
    ece: float
    auroc: float
    brier: float
    ef_1pct: float
    hit_rate: float
    spearman_rho: float
    spearman_pval: float
    rmse: float
    nll: float
    n_samples: int
    confidence_threshold: float = 0.85
    y_target: float = 7.0
    # Phase 1: optional CI
    ci_auroc: Optional[tuple] = None
    ci_ef_1pct: Optional[tuple] = None
    ci_hit_rate: Optional[tuple] = None


@dataclass
class MultiThresholdResults:
    """Results across multiple y_target thresholds."""
    thresholds: list
    results: list  # list of EvalResults


def evaluate_all(
    mu_pred,
    sigma_pred,
    p_success,
    y_true,
    y_target=7.0,
    confidence_threshold=0.85,
    bootstrap_n=0,
    bootstrap_seed=42,
):
    """Compute all evaluation metrics.

    Parameters
    ----------
    mu_pred : (N,) predicted mean pKd
    sigma_pred : (N,) predicted std dev (sigma_total)
    p_success : (N,) calibrated P_success
    y_true : (N,) true experimental pKd
    y_target : activity threshold in pKd
    confidence_threshold : threshold for high-confidence hit rate
    bootstrap_n : if > 0, compute bootstrap CIs for key metrics
    bootstrap_seed : random seed for bootstrap
    """
    mu_pred = np.asarray(mu_pred, dtype=float)
    sigma_pred = np.asarray(sigma_pred, dtype=float)
    p_success = np.asarray(p_success, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    y_binary = (y_true >= y_target).astype(float)

    # ECE
    ece = compute_ece(p_success, y_binary)

    # Brier Score (03_math_reference §6.3)
    brier = brier_score(p_success, y_binary)

    # AUROC
    auroc = _safe_auroc(y_binary, p_success)

    # EF@1%
    ef_1pct = enrichment_factor(p_success, y_binary, fraction=0.01)

    # Hit Rate @ high confidence
    high_conf_mask = p_success >= confidence_threshold
    if high_conf_mask.sum() > 0:
        hit_rate = float(y_binary[high_conf_mask].mean())
    else:
        hit_rate = float("nan")

    # Spearman rho
    if len(mu_pred) > 2:
        rho, pval = stats.spearmanr(mu_pred, y_true)
    else:
        rho, pval = float("nan"), float("nan")

    # RMSE
    rmse = float(np.sqrt(np.mean((mu_pred - y_true) ** 2)))

    # NLL
    nll = gaussian_nll(mu_pred, sigma_pred, y_true)

    # Bootstrap CIs
    ci_auroc = None
    ci_ef = None
    ci_hr = None
    if bootstrap_n > 0:
        ci_auroc = _bootstrap_ci(
            lambda idx: _safe_auroc(y_binary[idx], p_success[idx]),
            len(y_true), bootstrap_n, bootstrap_seed
        )
        ci_ef = _bootstrap_ci(
            lambda idx: enrichment_factor(p_success[idx], y_binary[idx], 0.01),
            len(y_true), bootstrap_n, bootstrap_seed + 1
        )
        ci_hr = _bootstrap_ci(
            lambda idx: (
                float(y_binary[idx][p_success[idx] >= confidence_threshold].mean())
                if (p_success[idx] >= confidence_threshold).any()
                else float("nan")
            ),
            len(y_true), bootstrap_n, bootstrap_seed + 2
        )

    return EvalResults(
        ece=ece, auroc=auroc, brier=brier, ef_1pct=ef_1pct, hit_rate=hit_rate,
        spearman_rho=float(rho), spearman_pval=float(pval),
        rmse=rmse, nll=nll, n_samples=len(y_true),
        confidence_threshold=confidence_threshold, y_target=y_target,
        ci_auroc=ci_auroc, ci_ef_1pct=ci_ef, ci_hit_rate=ci_hr,
    )


def evaluate_multi_threshold(
    mu_pred, sigma_pred, p_success_dict, y_true,
    thresholds=(7.0, 8.0), confidence_threshold=0.85,
    bootstrap_n=0,
):
    """Evaluate at multiple y_target thresholds.

    Parameters
    ----------
    p_success_dict : dict mapping y_target -> calibrated P_success array
        If a plain ndarray, uses same P_success for all thresholds.
    thresholds : tuple of y_target values
    """
    results = []
    for yt in thresholds:
        if isinstance(p_success_dict, dict):
            ps = p_success_dict.get(yt, p_success_dict.get(str(yt)))
        else:
            ps = p_success_dict
        if ps is None:
            logger.warning("No P_success for threshold %.1f, skipping", yt)
            continue
        r = evaluate_all(
            mu_pred, sigma_pred, ps, y_true,
            y_target=yt, confidence_threshold=confidence_threshold,
            bootstrap_n=bootstrap_n,
        )
        results.append(r)
    return MultiThresholdResults(thresholds=list(thresholds), results=results)


def evaluate_per_pocket(
    mu_pred, sigma_pred, p_success, y_true, pocket_ids,
    y_target=7.0, confidence_threshold=0.85,
):
    """Per-pocket metric breakdown.

    Parameters
    ----------
    pocket_ids : (N,) array of pocket identifiers
    """
    unique_pockets = np.unique(pocket_ids)
    pocket_results = {}
    for pid in unique_pockets:
        mask = pocket_ids == pid
        if mask.sum() < 2:
            continue
        r = evaluate_all(
            mu_pred[mask], sigma_pred[mask], p_success[mask], y_true[mask],
            y_target=y_target, confidence_threshold=confidence_threshold,
        )
        pocket_results[str(pid)] = r
    return pocket_results


# -- Core metric functions --

def brier_score(p_pred, y_binary):
    """Brier Score: mean squared error between predicted probabilities and binary outcomes.

    BS = (1/N) Σ (p_i - y_i)²
    """
    p_pred = np.asarray(p_pred, dtype=float)
    y_binary = np.asarray(y_binary, dtype=float)
    return float(np.mean((p_pred - y_binary) ** 2))


def compute_ece(p_pred, y_binary, n_bins=10):
    """Expected Calibration Error."""
    p_pred = np.asarray(p_pred, dtype=float)
    y_binary = np.asarray(y_binary, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(p_pred)
    if N == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if lo == bin_edges[-2]:
            mask = (p_pred >= lo) & (p_pred <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = p_pred[mask].mean()
        avg_acc = y_binary[mask].mean()
        ece += (n_bin / N) * abs(avg_conf - avg_acc)
    return float(ece)


def enrichment_factor(scores, labels, fraction=0.01):
    """Enrichment Factor at given fraction.

    EF@f = (hits_in_top_f / n_top_f) / (total_hits / N)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    N = len(scores)
    n_top = max(1, int(N * fraction))
    total_hits = labels.sum()
    if total_hits == 0:
        return 0.0
    top_idx = np.argsort(scores)[-n_top:]
    hits_in_top = labels[top_idx].sum()
    expected_rate = total_hits / N
    observed_rate = hits_in_top / n_top
    return float(observed_rate / expected_rate) if expected_rate > 0 else 0.0


def gaussian_nll(mu, sigma, y):
    """Average Gaussian negative log-likelihood."""
    sigma_clipped = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    mu = np.asarray(mu, dtype=float)
    y = np.asarray(y, dtype=float)
    nll_per = 0.5 * (
        np.log(2 * np.pi * sigma_clipped ** 2)
        + ((y - mu) ** 2) / (sigma_clipped ** 2)
    )
    return float(nll_per.mean())


def _safe_auroc(y_binary, scores):
    """AUROC with graceful handling of degenerate cases."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_binary)) < 2:
            return float("nan")
        return float(roc_auc_score(y_binary, scores))
    except Exception:
        return float("nan")


def _bootstrap_ci(metric_fn, n, B=1000, seed=42, alpha=0.05):
    """Bootstrap confidence interval for a metric.

    Parameters
    ----------
    metric_fn : callable(indices) -> float
    n : sample size
    B : number of bootstrap resamples
    seed : random seed
    alpha : significance level (default 95% CI)

    Returns
    -------
    (lower, upper) tuple
    """
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        v = metric_fn(idx)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    vals = np.array(vals)
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return (lo, hi)


# -- Output / Serialization --

def print_results(results, file=None):
    """Pretty-print evaluation results."""
    lines = []
    lines.append("=" * 55)
    lines.append(f"Evaluation Results (N={results.n_samples})")
    lines.append(f"  y_target = {results.y_target}, "
                 f"conf_thresh = {results.confidence_threshold}")
    lines.append("-" * 55)
    lines.append(f"  ECE:          {results.ece:.4f}")
    lines.append(f"  Brier:        {results.brier:.4f}")
    auroc_s = f"{results.auroc:.4f}"
    if results.ci_auroc:
        auroc_s += f"  [{results.ci_auroc[0]:.4f}, {results.ci_auroc[1]:.4f}]"
    lines.append(f"  AUROC:        {auroc_s}")
    ef_s = f"{results.ef_1pct:.2f}"
    if results.ci_ef_1pct:
        ef_s += f"  [{results.ci_ef_1pct[0]:.2f}, {results.ci_ef_1pct[1]:.2f}]"
    lines.append(f"  EF@1%:        {ef_s}")
    hr_s = f"{results.hit_rate:.4f}"
    if results.ci_hit_rate:
        hr_s += f"  [{results.ci_hit_rate[0]:.4f}, {results.ci_hit_rate[1]:.4f}]"
    lines.append(f"  Hit Rate:     {hr_s}")
    lines.append(f"  Spearman rho: {results.spearman_rho:.4f} "
                 f"(p={results.spearman_pval:.2e})")
    lines.append(f"  RMSE:         {results.rmse:.4f}")
    lines.append(f"  NLL:          {results.nll:.4f}")
    lines.append("=" * 55)
    text = "\n".join(lines)
    if file:
        Path(file).write_text(text)
    for line in lines:
        logger.info(line)
    return text


def results_to_dict(results):
    """Convert EvalResults to a JSON-serializable dict."""
    d = asdict(results)
    # Convert tuples to lists for JSON
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


def save_results_json(results, path):
    """Save EvalResults to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(results, MultiThresholdResults):
        data = {
            "thresholds": results.thresholds,
            "results": [results_to_dict(r) for r in results.results],
        }
    elif isinstance(results, dict):
        data = {k: results_to_dict(v) for k, v in results.items()}
    else:
        data = results_to_dict(results)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved results to %s", path)


def save_results_csv(results_list, path, labels=None):
    """Save list of EvalResults to CSV.

    Parameters
    ----------
    results_list : list of EvalResults
    path : output CSV path
    labels : optional list of row labels
    """
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "y_target", "n_samples", "ece", "brier", "auroc",
        "ef_1pct", "hit_rate", "spearman_rho", "rmse", "nll",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(results_list):
            row = {
                "label": labels[i] if labels else f"run_{i}",
                "y_target": r.y_target,
                "n_samples": r.n_samples,
                "ece": f"{r.ece:.4f}",
                "brier": f"{r.brier:.4f}",
                "auroc": f"{r.auroc:.4f}",
                "ef_1pct": f"{r.ef_1pct:.2f}",
                "hit_rate": f"{r.hit_rate:.4f}",
                "spearman_rho": f"{r.spearman_rho:.4f}",
                "rmse": f"{r.rmse:.4f}",
                "nll": f"{r.nll:.4f}",
            }
            writer.writerow(row)
    logger.info("Saved CSV to %s", path)


def comparison_table(results_dict):
    """Format a comparison table from {method_name: EvalResults}.

    Returns a formatted string.
    """
    methods = list(results_dict.keys())
    metrics = ["ECE", "Brier", "AUROC", "EF@1%", "Hit Rate", "Spearman", "RMSE", "NLL"]
    header = f"{'Method':<20}" + "".join(f"{m:>10}" for m in metrics)
    lines = [header, "-" * len(header)]
    for name, r in results_dict.items():
        row = f"{name:<20}"
        row += f"{r.ece:>10.4f}"
        row += f"{r.brier:>10.4f}"
        row += f"{r.auroc:>10.4f}"
        row += f"{r.ef_1pct:>10.2f}"
        row += f"{r.hit_rate:>10.4f}"
        row += f"{r.spearman_rho:>10.4f}"
        row += f"{r.rmse:>10.4f}"
        row += f"{r.nll:>10.4f}"
        lines.append(row)
    return "\n".join(lines)
