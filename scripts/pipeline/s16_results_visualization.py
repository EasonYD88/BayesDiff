#!/usr/bin/env python3
"""
Sub-Plan 2 — Consolidated Results Visualization

Generates:
1. Bar chart: Test R² and Test ρ across all experiments
2. Phase progression plot: performance across phases
3. GP comparison: MLP vs GP variants
4. Attention weight analysis (layer betas from SchemeB)
"""

import json
import sys
import os
import numpy as np

# Attempt matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

# ─── Results Data ──────────────────────────────────────────────────────
ALL_RESULTS = {
    # Phase 1
    "P0\nMeanPool→MLP": {"test_r2": 0.524, "test_rho": 0.744, "phase": 1},
    "P1\nAttnPool→MLP": {"test_r2": 0.560, "test_rho": 0.753, "phase": 1},
    # Phase 2
    "A2.1\nMeanPool→\nAttnFusion": {"test_r2": 0.564, "test_rho": 0.751, "phase": 2},
    "A2.2\nSchemeA\nTwoBranch": {"test_r2": 0.547, "test_rho": 0.753, "phase": 2},
    "A2.3\nSchemeB\n(λ=0.01)": {"test_r2": 0.568, "test_rho": 0.747, "phase": 2},
    # Phase 3
    "A3.2\nSchemeB\n(λ=0.1)": {"test_r2": 0.544, "test_rho": 0.756, "phase": 3},
    "A3.4-S1\nSchemeB→\nMLP": {"test_r2": 0.572, "test_rho": 0.761, "phase": 3},
    # GP variants
    "A3.4-S2\nSVGP\nraw 128d": {"test_r2": 0.507, "test_rho": 0.719, "phase": "gp"},
    "A3.4b\nPCA32→\nSVGP": {"test_r2": 0.543, "test_rho": 0.746, "phase": "gp"},
    "A3.4c\nDKL\n128→32": {"test_r2": 0.559, "test_rho": 0.760, "phase": "gp"},
    "A3.4d\nPCA16→\nSVGP": {"test_r2": 0.512, "test_rho": 0.726, "phase": "gp"},
    # Ablations
    "A3.5\nMultiHead\nH=4": {"test_r2": 0.565, "test_rho": 0.755, "phase": "abl"},
    "A3.6\nIndependent\nAttnPool": {"test_r2": 0.574, "test_rho": 0.778, "phase": "abl"},
}

PHASE_COLORS = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52", "gp": "#8172B2", "abl": "#DD8452"}
PHASE_LABELS = {1: "Phase 1: Preliminary", 2: "Phase 2: Scheme Comparison",
                3: "Phase 3: Refinement", "gp": "GP Integration", "abl": "Ablation"}


def fig1_bar_chart(outdir):
    """Bar chart: Test R² and Test ρ for all experiments."""
    names = list(ALL_RESULTS.keys())
    r2 = [ALL_RESULTS[n]["test_r2"] for n in names]
    rho = [ALL_RESULTS[n]["test_rho"] for n in names]
    colors = [PHASE_COLORS[ALL_RESULTS[n]["phase"]] for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(len(names))
    width = 0.6

    # Test R²
    bars1 = ax1.bar(x, r2, width, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel("Test R²", fontsize=12)
    ax1.set_ylim(0.45, 0.60)
    ax1.axhline(y=0.524, color='gray', linestyle='--', alpha=0.5, label='P0 baseline')
    ax1.legend(fontsize=9)
    for i, v in enumerate(r2):
        ax1.text(i, v + 0.003, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    # Test ρ
    bars2 = ax2.bar(x, rho, width, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel("Test Spearman ρ", fontsize=12)
    ax2.set_ylim(0.70, 0.77)
    ax2.axhline(y=0.744, color='gray', linestyle='--', alpha=0.5, label='P0 baseline')
    ax2.legend(fontsize=9)
    for i, v in enumerate(rho):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8)

    # Legend for phases
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(PHASE_COLORS.values(), PHASE_LABELS.values())]
    fig.legend(handles=patches, loc='upper center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Sub-Plan 2: Attention-Based Aggregation — All Experiments",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_all_experiments.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig1_all_experiments.png")


def fig2_gp_comparison(outdir):
    """GP variant comparison with MLP baseline."""
    gp_data = {
        "MLP\nbaseline": {"r2": 0.572, "rho": 0.761, "noise": None, "unc_rho": None},
        "SVGP\nraw 128d": {"r2": 0.507, "rho": 0.719, "noise": 1.654, "unc_rho": -0.008},
        "PCA32→\nSVGP": {"r2": 0.543, "rho": 0.746, "noise": 0.903, "unc_rho": 0.042},
        "DKL\n128→32": {"r2": 0.559, "rho": 0.760, "noise": 0.162, "unc_rho": -0.035},
        "PCA16→\nSVGP": {"r2": 0.512, "rho": 0.726, "noise": 0.978, "unc_rho": 0.029},
    }

    names = list(gp_data.keys())
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: R² and ρ
    ax = axes[0]
    r2_vals = [gp_data[n]["r2"] for n in names]
    rho_vals = [gp_data[n]["rho"] for n in names]
    colors_r2 = ['#4C72B0' if n.startswith('MLP') else '#8172B2' for n in names]
    colors_rho = ['#4C72B0' if n.startswith('MLP') else '#C44E52' for n in names]
    ax.bar(x - width/2, r2_vals, width, label='Test R²', color='#4C72B0', alpha=0.8)
    ax.bar(x + width/2, rho_vals, width, label='Test ρ', color='#C44E52', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Prediction Quality")
    ax.legend()
    ax.set_ylim(0.45, 0.80)

    # Panel 2: Noise variance
    ax = axes[1]
    noise_names = [n for n in names if gp_data[n]["noise"] is not None]
    noise_vals = [gp_data[n]["noise"] for n in noise_names]
    noise_colors = ['#C44E52' if v > 1.0 else '#55A868' for v in noise_vals]
    ax.bar(range(len(noise_names)), noise_vals, color=noise_colors, alpha=0.8)
    ax.set_xticks(range(len(noise_names)))
    ax.set_xticklabels(noise_names, fontsize=9)
    ax.set_ylabel("GP Noise Variance")
    ax.set_title("Noise Variance (lower = better signal)")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Panel 3: Uncertainty calibration
    ax = axes[2]
    unc_names = [n for n in names if gp_data[n]["unc_rho"] is not None]
    unc_vals = [gp_data[n]["unc_rho"] for n in unc_names]
    unc_colors = ['#55A868' if abs(v) > 0.05 else '#C44E52' for v in unc_vals]
    ax.bar(range(len(unc_names)), unc_vals, color=unc_colors, alpha=0.8)
    ax.set_xticks(range(len(unc_names)))
    ax.set_xticklabels(unc_names, fontsize=9)
    ax.set_ylabel("|error|-σ Spearman ρ")
    ax.set_title("Uncertainty Calibration (higher = better)")
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylim(-0.1, 0.1)

    fig.suptitle("GP Integration: MLP vs GP Variants", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_gp_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig2_gp_comparison.png")


def fig3_phase_progression(outdir):
    """Line plot: best performance at each phase."""
    phases = ["Baseline\n(L9 Mean)", "Phase 1\n(AttnPool)", "Phase 2\n(SchemeB)",
              "Phase 3\n(Retrained)", "GP Fix\n(DKL)"]
    best_r2 = [0.524, 0.560, 0.568, 0.572, 0.559]
    best_rho = [0.744, 0.753, 0.747, 0.761, 0.760]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(phases))

    ax1.plot(x, best_r2, 'o-', color='#4C72B0', markersize=10, linewidth=2)
    for i, v in enumerate(best_r2):
        ax1.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=10)
    ax1.set_ylabel("Test R²", fontsize=12)
    ax1.set_title("Test R² Progression", fontsize=12)
    ax1.set_ylim(0.50, 0.60)
    ax1.grid(alpha=0.3)

    ax2.plot(x, best_rho, 's-', color='#C44E52', markersize=10, linewidth=2)
    for i, v in enumerate(best_rho):
        ax2.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases, fontsize=10)
    ax2.set_ylabel("Test Spearman ρ", fontsize=12)
    ax2.set_title("Test ρ Progression", fontsize=12)
    ax2.set_ylim(0.73, 0.77)
    ax2.grid(alpha=0.3)

    fig.suptitle("Sub-Plan 2: Performance Progression Across Phases", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_phase_progression.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig3_phase_progression.png")


def fig4_layer_attention(outdir):
    """Layer attention weights (beta) from Phase 2 SchemeB."""
    # From s13_scheme_comparison.py results — A2.3 SchemeB layer betas
    # These are the layer-level attention weights learned by AttnFusion
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "results", "stage2", "scheme_comparison", "scheme_comparison_results.json")
    
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        if "A2.3" in data and "layer_stats" in data["A2.3"]:
            stats = data["A2.3"]["layer_stats"]
            beta = stats.get("layer_beta_mean")
            if beta:
                fig, ax = plt.subplots(figsize=(8, 5))
                layers = [f"Layer {i}" for i in range(len(beta))]
                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(beta)))
                bars = ax.bar(layers, beta, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_ylabel("Attention Weight (β)", fontsize=12)
                ax.set_xlabel("Encoder Layer", fontsize=12)
                ax.set_title("SchemeB: Layer-Level Attention Weights\n"
                             "(deeper layers receive higher attention)",
                             fontsize=12)
                for i, v in enumerate(beta):
                    ax.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
                ax.set_ylim(0, max(beta) * 1.15)
                plt.tight_layout()
                fig.savefig(os.path.join(outdir, "fig4_layer_attention.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved fig4_layer_attention.png")
                return
    
    print("  [SKIP] fig4: scheme_comparison results not found or no layer_beta data")


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outdir = os.path.join(base, "results", "stage2", "figures")
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Saving figures to {outdir}/")
    fig1_bar_chart(outdir)
    fig2_gp_comparison(outdir)
    fig3_phase_progression(outdir)
    fig4_layer_attention(outdir)
    print("Done.")


if __name__ == "__main__":
    main()
