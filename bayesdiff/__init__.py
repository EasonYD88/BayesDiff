"""
BayesDiff: Dual Uncertainty-Aware Confidence Scoring for 3D Molecular Generation.

Modules:
    data             — PDBbind parsing, splits, label transforms
    sampler          — TargetDiff sampling wrapper + embedding extraction
    gen_uncertainty  — Generation-side uncertainty (Σ̂_gen, GMM)
    gp_oracle        — SVGP oracle (μ_oracle, σ²_oracle)
    fusion           — Delta Method uncertainty fusion
    calibration      — Isotonic regression + ECE
    ood              — Mahalanobis OOD detection
    evaluate         — Full evaluation metrics
"""

__version__ = "0.1.0"
