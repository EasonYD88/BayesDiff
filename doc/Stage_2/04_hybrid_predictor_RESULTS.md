# Sub-Plan 04: Hybrid Oracle Predictor — Final Results

**Date**: April 9–10, 2026  
**Author**: Auto-generated from experimental results  
**Status**: Phase G complete. Phase C' (SNGP + Evidential) complete.

---

## 1. Executive Summary

The DKL Ensemble oracle head (M=5, bootstrap, residual MLP) is the clear winner from a systematic 7-method comparison (5 Tier 1 + 2 Tier 1b). It is the **only** method producing statistically significant uncertainty–error correlation on CASF-2016:

| Metric | DKL Ensemble | Runner-up (NN+GP Residual) | Baseline (Raw SVGP) |
|--------|-------------|---------------------------|---------------------|
| ρ (point prediction) | **0.781** | 0.765 | 0.763 |
| R² | **0.607** | 0.582 | 0.573 |
| RMSE | **1.361** | 1.404 | 1.418 |
| NLL | **1.758** | 1.767 | 1.781 |
| ρ_{|err|,σ} | **0.144** (p=0.015) | 0.051 (p=0.39) | 0.020 (p=0.73) |
| Training time | 127s | 145s | 199s |

The stretch target of ρ_{|err|,σ} > 0.15 was not reached, but the achieved 0.144 (Tier 1) / 0.091±0.025 (multi-seed Tier 2) represents a meaningful improvement from ~0 for all non-ensemble methods.

---

## 2. Frozen Representation

- **Model**: A3.6 SchemeB Independent ParamPool (108K params)
- **Embedding**: z ∈ ℝ¹²⁸ per complex
- **MLP ceiling**: ρ = 0.778, R² = 0.574 on CASF-2016
- **Data**: PDBbind v2020 general set (16,108 train / 2,372 val / 285 test=CASF-2016)
- **Checkpoint**: `results/stage2/ablation_viz/A36_independent_model.pt`

---

## 3. Tier 1: Method Family Comparison

Seven oracle head architectures compared on identical frozen embeddings (Tier 1 + Tier 1b):

| Head | Architecture | Params | ρ | R² | RMSE | NLL | ρ_{|err|,σ} | p-value | Time |
|------|-------------|--------|------|------|------|------|-------------|---------|------|
| **DKL Ensemble** | 5× DKL (bootstrap) | ~225K | **0.781** | **0.607** | **1.361** | **1.758** | **0.144** | **0.015** | 127s |
| DKL | ResidualMLP→SVGP | ~45K | 0.775 | 0.589 | 1.391 | 1.818 | 0.001 | 0.99 | 32s |
| Evidential | NIG regression | ~50K | 0.773 | 0.578 | 1.409 | 1.762 | 0.074 | — | 12s |
| SNGP | SN-MLP→RFF-GP | ~80K | 0.768 | 0.584 | 1.401 | 1453† | 0.018 | — | 23s |
| NN+GP Residual | MLP + GP on residuals | ~50K | 0.765 | 0.582 | 1.404 | 1.767 | 0.064 | 0.39 | 145s |
| Raw SVGP | Direct GP on z | ~66K | 0.763 | 0.573 | 1.418 | 1.781 | 0.020 | 0.73 | 199s |
| PCA+SVGP | PCA(32)→SVGP | ~17K | 0.758 | 0.579 | 1.408 | 1.771 | 0.015 | 0.80 | 200s |

† SNGP NLL is mis-calibrated: the RFF posterior variance is orders of magnitude too small, causing extreme NLL. Point prediction is competitive.

**Key findings**:
1. Only DKL Ensemble achieves statistically significant ρ_{|err|,σ} (p=0.015). All single-model methods produce near-zero uncertainty–error correlation.
2. **Evidential** is the runner-up for ρ_{|err|,σ} (0.074) — its learned per-sample uncertainty outperforms GP-based methods, but still falls far short of ensemble disagreement.
3. **SNGP** preserves input-space distances via spectral normalization, but the RFF GP posterior collapses to near-zero variance (NLL=1453), making its uncertainty useless despite competitive point prediction.
4. Ensemble disagreement remains the only reliable mechanism for producing meaningful uncertainty on this dataset.

**SLURM**: Tier 1 — Jobs 5857850 (L40S) + 5857851 (A100). Tier 1b — Job 5881543 (L40S).

---

## 4. Tier 2: DKL Ensemble Ablation Study

### 4.1 Multi-seed Baseline (A4.0)

| Seed | ρ | R² | NLL | ρ_{|err|,σ} |
|------|------|------|------|-------------|
| 42 | 0.789 | 0.620 | 1.743 | 0.100 |
| 123 | 0.774 | 0.599 | 1.807 | 0.109 |
| 777 | 0.776 | 0.604 | 1.809 | 0.063 |
| **Mean±SD** | **0.780±0.008** | **0.608±0.011** | **1.786±0.037** | **0.091±0.025** |

### 4.2 Ablation Configs

| ID | Config | ρ | ρ_{|err|,σ} | Δρ_{|err|,σ} from baseline | Key insight |
|----|--------|------|-------------|---------------------------|-------------|
| A4.0 | Baseline (3-seed avg) | 0.780 | 0.091±0.025 | — | Reference |
| A4.8 | residual=False | 0.781 | 0.017 | −0.074 | Residual connection important for UQ |
| A4.9 | feature_dim=16 | 0.777 | 0.105 | +0.014 | Smaller bottleneck slightly better |
| A4.10 | feature_dim=64 | 0.782 | 0.003 | −0.088 | Larger bottleneck kills UQ |
| A4.11 | M=3 (n_members) | **0.793** | 0.057 | −0.034 | Best point pred, lower UQ |
| A4.12 | bootstrap=False | 0.785 | **−0.056** | −0.147 | **Bootstrap CRITICAL** |
| A4.13 | n_layers=3 | 0.785 | 0.080 | −0.011 | Marginal change |
| A4.14 | n_inducing=1024 | 0.785 | 0.041 | −0.050 | More inducing hurts UQ |
| A4.15 | NN-Residual MC Dropout | 0.772 | 0.034 | −0.057 | Inferior to DKL Ensemble |

### 4.3 Ablation Analysis

1. **Bootstrap is the critical mechanism**: Without bootstrap (A4.12), ρ_{|err|,σ} drops from +0.091 to −0.056. This is the single most important design choice.

2. **Residual connection matters**: Removing residual (A4.8) collapses ρ_{|err|,σ} from 0.091 to 0.017, a 5× reduction.

3. **Feature dimension sweet spot**: d_u=32 (baseline) is near-optimal. d_u=16 is slightly better for UQ (0.105 vs 0.091). d_u=64 catastrophically loses UQ signal (0.003).

4. **Ensemble size M=3 vs M=5**: M=3 gives the best single-seed point prediction (ρ=0.793) but lower UQ (0.057). M=5 provides more stable uncertainty at small cost to ρ.

5. **n_inducing=1024 hurts**: More inducing points do not help and actually degrade UQ (0.041 vs 0.091), likely due to overfitting the GP posterior.

6. **High seed variance**: ρ_{|err|,σ} ranges 0.063–0.109 across seeds, suggesting N=285 is a limiting factor.

**SLURM**: Job 5863003 (L40S), ~30 min total.

---

## 5. Uncertainty Decomposition (DKL Ensemble)

From `uncertainty_diagnostics.json`:

| Component | ρ_{|err|,σ_component} | Fraction of total σ² |
|-----------|----------------------|---------------------|
| Total σ² | 0.144 | 100% |
| Aleatoric (mean GP variance) | 0.011 | 92.4% |
| Epistemic (member disagreement) | **0.180** | 7.6% |

**Key insight**: The epistemic component (inter-member variance) drives essentially all of the uncertainty–error correlation signal (ρ=0.180) despite comprising only 7.6% of total variance. The aleatoric component (intra-member GP variance) contributes no signal (ρ=0.011). This confirms that ensemble disagreement is the primary source of useful uncertainty.

### Ensemble Diversity

From `ensemble_diagnostics.json`:

| Metric | Value |
|--------|-------|
| Members (M) | 5 |
| Mean pairwise ρ | 0.941 |
| Range pairwise ρ | 0.930 – 0.956 |
| Effective ensemble size (M_eff) | 1.05 |

Members are highly correlated (M_eff ≈ 1), explaining why the epistemic fraction is only 7.6%. Despite this, the small disagreement signal is still the primary driver of uncertainty quality, underscoring how difficult it is to obtain calibrated uncertainties from single models.

---

## 6. Calibration Analysis

From the calibration curve (Fig. D.3):
- All heads are **under-confident** (observed fraction below diagonal), especially at higher confidence levels
- DKL and DKL Ensemble are the most under-confident (tightest predicted intervals)
- Raw SVGP and PCA+SVGP are closest to the diagonal (well-calibrated variances) but this is because their variance is nearly constant (no signal)
- Post-hoc isotonic calibration (tested in integration tests T2.3) successfully recalibrates

---

## 7. Final Configuration

**Selected**: DKL Ensemble with default hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M (members) | 5 | Better UQ stability than M=3 |
| d_u (feature_dim) | 32 | Good balance; d_u=16 marginal gain not worth complexity |
| hidden_dim | 256 | Default, not ablated |
| n_layers | 2 | Default; 3 layers marginal change |
| residual | True | Critical for UQ (5× improvement) |
| bootstrap | True | **Critical** for UQ (without: ρ_{|err|,σ} < 0) |
| n_inducing | 512 | 1024 hurts UQ; 512 sufficient |
| kernel | Matérn-5/2 + ARD | Standard choice |

---

## 8. Figures Generated

All in `results/stage2/oracle_heads/figures/`:

| Figure | File | Description |
|--------|------|-------------|
| D.2 | `err_vs_sigma_scatter.{pdf,png}` | |err| vs σ for all 5 heads — DKL Ensemble shows clear positive trend |
| D.3 | `calibration_curve.{pdf,png}` | Expected vs observed confidence — all heads under-confident |
| D.4 | `feature_tsne.{pdf,png}` | t-SNE of raw z vs DKL features u, colored by |err| |
| D.5 | `tier1_comparison_bar.{pdf,png}` | Grouped bar chart: ρ, R², ρ_{|err|,σ} for all heads |
| D.7 | `uncertainty_decomp.{pdf,png}` | Aleatoric vs epistemic σ² sorted by error |
| D.8 | `binned_err_sigma.{pdf,png}` | Mean |err| per σ-quantile bin — monotonicity check |

---

## 9. Test Suite

| Suite | Tests | Status |
|-------|-------|--------|
| Unit (`test_hybrid_oracle.py`) | 34 | ✅ All pass |
| Integration (`test_hybrid_integration.py`) | 5 | ✅ All pass |
| **Total** | **39** | **✅ 39/39 pass** |

Verified on L40S GPU (job 5881541). Includes T1.15–T1.21 for SNGP and Evidential.

---

## 10. Files Created/Modified

### New Files (Sub-Plan 04)

| File | Lines | Purpose |
|------|-------|---------|
| `bayesdiff/oracle_interface.py` | ~160 | OracleResult dataclass + OracleHead ABC |
| `bayesdiff/hybrid_oracle.py` | ~680 | DKL, DKLEnsemble, NNResidual, PCA_GP oracles |
| `scripts/pipeline/s18_train_oracle_heads.py` | ~380 | Training + evaluation pipeline |
| `scripts/pipeline/s19_oracle_diagnostics.py` | ~400 | Diagnostic figures D.2–D.8 |
| `slurm/s18_oracle_heads.sh` | ~40 | Tier 1 SLURM script |
| `slurm/s18_tier2_ablation.sh` | ~80 | Tier 2 SLURM script |
| `slurm/s19_oracle_diagnostics.sh` | ~30 | Diagnostics SLURM script |
| `slurm/run_tests.sh` | ~25 | Test runner SLURM script |
| `slurm/s18_tier1b_baselines.sh` | ~35 | Tier 1b SLURM script (SNGP + Evidential) |
| `tests/stage2/test_hybrid_oracle.py` | ~340 | Unit tests T1.1–T1.21 (incl. SNGP, Evidential) |
| `tests/stage2/test_hybrid_integration.py` | ~150 | Integration tests T2.1–T2.6 |

### Modified Files

| File | Changes |
|------|---------|
| `bayesdiff/__init__.py` | Added OracleHead, OracleResult imports |
| `bayesdiff/hybrid_oracle.py` | Added SNGPOracle + EvidentialOracle (~500 lines) |
| `doc/Stage_2/04_hybrid_predictor.md` | Updated §7.2 results, §8.1 methods, §9 checklist |

---

## 11. Compute Budget

| Phase | GPU | Time | Jobs |
|-------|-----|------|------|
| Tier 1 | L40S + A100 | ~10 min each | 5857850, 5857851 |
| Tier 2 | L40S | ~30 min | 5863003 |
| Tests | L40S | ~1 min | 5865532 |
| Diagnostics | L40S | ~1 min | 5868900 |
| Tier 1b (SNGP+Evid.) | L40S | ~1 min | 5881543 |
| **Total** | | **~53 min GPU** | |

---

## 12. Limitations and Next Steps

### Limitations
1. **ρ_{|err|,σ} below stretch target**: 0.091±0.025 vs target 0.15. CASF-2016 (N=285) may be too small for reliable estimation.
2. **Low effective ensemble size**: M_eff=1.05 despite M=5 members. Bootstrap diversity is limited.
3. **All calibration curves under-confident**: Need post-hoc recalibration before deployment.
4. **Single dataset evaluation**: Results on CASF-2016 only; generalization unknown.

### Recommended Next Steps
1. ~~**Phase C' (deferred)**: Implement SNGP and Evidential baselines~~ **DONE** — see Tier 1b results in §3
2. **Larger test set**: Evaluate on held-out PDBbind subsets for more stable ρ_{|err|,σ}
3. **Calibration integration**: Apply isotonic calibration in the fusion pipeline
4. **Feature diversity**: Explore different random seeds for feature extractor init, or heterogeneous architectures, to increase M_eff
5. **Proceed to Sub-Plan 05**: Delta method fusion with the DKL Ensemble as the oracle head
