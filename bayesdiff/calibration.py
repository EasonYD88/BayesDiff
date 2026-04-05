"""
bayesdiff/calibration.py — §4.4 Calibration Module
────────────────────────────────────────────────────
Probability calibration and ECE evaluation.
Paper reference: §4.4 "Isotonic Regression Calibration"
Equation reference: doc/Stage_1/03_math_reference.md §6

Uses Isotonic Regression on a held-out calibration set to map
raw P_success → calibrated P_success such that ECE is minimized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Isotonic regression calibrator for P_success.

    Fit on calibration set: (P_success_raw, binary_label) pairs.
    Transform at inference: P_calibrated = g(P_raw).
    """

    def __init__(self):
        self._calibrator = None

    def fit(
        self,
        p_raw: np.ndarray,
        y_binary: np.ndarray,
    ):
        """Fit isotonic regression.

        Parameters
        ----------
        p_raw : np.ndarray, shape (N_cal,)
            Raw P_success values on calibration set.
        y_binary : np.ndarray, shape (N_cal,)
            Binary labels (1 if pKd >= y_target, else 0).
        """
        from sklearn.isotonic import IsotonicRegression

        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._calibrator.fit(p_raw, y_binary)
        logger.info(
            f"Isotonic calibrator fitted on {len(p_raw)} samples "
            f"(positive rate: {y_binary.mean():.3f})"
        )

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities.

        Parameters
        ----------
        p_raw : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, shape (N,) — calibrated probabilities
        """
        assert self._calibrator is not None, "Calibrator not fitted."
        return self._calibrator.transform(p_raw)

    def save(self, path: str | Path):
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._calibrator, f)

    def load(self, path: str | Path):
        import pickle

        with open(path, "rb") as f:
            self._calibrator = pickle.load(f)


def compute_ece(
    p_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Parameters
    ----------
    p_pred : np.ndarray, shape (N,)
        Predicted probabilities.
    y_true : np.ndarray, shape (N,)
        Binary ground truth labels.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float : ECE value
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(p_pred)

    for i in range(n_bins):
        mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p_pred >= bin_edges[i]) & (p_pred <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = p_pred[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_conf - avg_acc)

    return ece


def reliability_diagram_data(
    p_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for reliability diagram plotting.

    Returns dict with keys: bin_centers, accuracy, confidence, count.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p_pred >= bin_edges[i]) & (p_pred <= bin_edges[i + 1])

        n_bin = mask.sum()
        centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        counts.append(n_bin)

        if n_bin > 0:
            accuracies.append(y_true[mask].mean())
            confidences.append(p_pred[mask].mean())
        else:
            accuracies.append(0.0)
            confidences.append(0.0)

    return {
        "bin_centers": np.array(centers),
        "accuracy": np.array(accuracies),
        "confidence": np.array(confidences),
        "count": np.array(counts),
    }
