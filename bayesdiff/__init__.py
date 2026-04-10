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
from .oracle_interface import OracleHead, OracleResult

# === §4.3 Fusion Module ===
from .fusion import fuse_uncertainties

# === §4.4 Calibration & OOD ===
from .calibration import IsotonicCalibrator
from .ood import MahalanobisOOD

# === §5 Evaluation ===
from .evaluate import evaluate_all

# === Data Utilities ===
from .data import (
    parse_pdbbind_index,
    parse_casf_coreset,
    load_casf2016_codes,
    protein_family_split,
    cluster_stratified_split,
)

# === Pretrain Dataset (Stage 2) ===
from .pretrain_dataset import PDBbindPairDataset, get_pdbbind_dataloader
