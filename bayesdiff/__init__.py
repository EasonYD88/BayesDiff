"""
BayesDiff: Dual Uncertainty-Aware Confidence Scoring for 3D Molecular Generation.

Architecture (paper §4):
    Pocket PDB → sampling → SE(3) encoding → Σ̂_gen
                                                ↓
                            SVGP Oracle → μ, σ²_oracle, J_μ
                                                ↓
                            Delta Method Fusion → σ²_total
                                                ↓
                            Calibration + OOD → P_success
"""

__version__ = "0.1.0"

# === §4.1 Generation Module ===
from .sampler import TargetDiffSampler
from .gen_uncertainty import estimate_gen_uncertainty

# === §4.2 Oracle Module ===
from .gp_oracle import GPOracle

# === §4.3 Fusion Module ===
from .fusion import fuse_uncertainties

# === §4.4 Calibration & OOD ===
from .calibration import IsotonicCalibrator
from .ood import MahalanobisOOD

# === §5 Evaluation ===
from .evaluate import evaluate_all

# === Data Utilities ===
from .data import parse_pdbbind_index, protein_family_split
