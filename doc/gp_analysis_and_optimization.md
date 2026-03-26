# GP Model Analysis & Optimization Plan

> Generated from `scripts/11_gp_training_analysis.py` run on 2026-03-26  
> Results directory: `results/embedding_rdkit/gp_analysis/`

---

## 1. Current Results Summary

### 1.1 Data Setup

| Item | Value |
|------|-------|
| Total pockets with ECFP4 embeddings | 49 (out of 93) |
| Pockets with pKd labels | 24 |
| Embedding dimension | 128 (Morgan fingerprint, radius=2) |
| Train / Val / Test split | 16 / 4 / 4 |
| Augmentation | 16 → 200 samples (Gaussian noise σ_x=0.3, σ_y=0.5) |

### 1.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Sparse Variational GP (SVGP) |
| Kernel | ARD Matérn-5/2 + ScaleKernel |
| Inducing points (J) | 48 |
| Optimizer | Adam, lr=0.01 |
| Epochs | 300 |
| Batch size | 64 |
| Likelihood | Gaussian (learnable noise) |
| Training time | ~30 min (CPU) |

### 1.3 Per-Split Metrics

| Metric | Train (N=16) | Val (N=4) | Test (N=4) | All (N=24) |
|--------|:------------:|:---------:|:----------:|:----------:|
| **RMSE** | 1.275 | 2.740 | 3.337 | 2.047 |
| **R²** | 0.508 | **-1.164** | **-16.503** | -0.068 |
| **Spearman ρ** | 0.715 | **-0.400** | **-0.800** | 0.470 |
| **Pearson r** | 0.867 | -0.626 | -0.862 | 0.489 |
| **NLL** | 1.677 | 3.220 | 4.270 | 2.366 |
| **95% CI coverage** | 100% | 75% | **50%** | 88% |
| **Mean CI width** | 5.41 | 5.43 | 5.81 | 5.48 |

---

## 2. Identified Problems

### 2.1 🔴 P1: Severe Overfitting

**Evidence:**
- Train RMSE = 1.275 vs Val RMSE = 2.740 vs Test RMSE = 3.337 (2.6× gap)
- Train R² = 0.508 but Val/Test R² are **negative** (-1.16, -16.5), meaning predictions are worse than simply predicting the mean
- Train Spearman ρ = 0.715 (good) but Val/Test are **negative** (-0.4, -0.8) — predictions are anti-correlated with truth

**Root cause:**
The model has 128 ARD lengthscale parameters + kernel scale + noise + 48 inducing point locations + variational parameters, totaling **thousands of free parameters** being fit to only **16 real training points** (200 after augmentation, but augmented points are derived from the same 16). This is a classic high-dimensional overfitting scenario where p ≫ n.

### 2.2 🔴 P2: Validation Loss Diverges After Epoch 28

**Evidence:**
- Best val loss = 7.23 at **epoch 28**
- Final val loss = 16.45 at epoch 300 (2.3× worse)
- Train loss decreases monotonically: 25.2 → 2.07

**Root cause:**
No early stopping is implemented. The model continues to memorize the augmented training set long after validation performance has degraded. The 272 additional epochs after the optimum are wasted computation that actively harms generalization.

### 2.3 🟠 P3: Data Augmentation Creates Misleading Signals

**Evidence:**
- 16 real points are expanded to 200 via Gaussian noise (σ_x=0.3 on embeddings, σ_y=0.5 on labels)
- The GP learns local patterns around the 16 cluster centers that don't reflect true data distribution
- ECFP4 fingerprints are **binary** (0/1), but Gaussian noise makes them continuous → destroys the discrete structure

**Root cause:**
Gaussian noise augmentation is inappropriate for binary fingerprint features. Adding continuous noise to {0,1} features creates samples in regions of embedding space that don't correspond to real molecules. The GP fits these artifacts and extrapolates poorly to real val/test molecules.

### 2.4 🟠 P4: Train/Test Distribution Mismatch

**Evidence:**
- Train pKd range: [3.28, 8.96], mean=5.75
- Val pKd range: [1.87, 6.89], mean=4.93
- Test pKd range: [2.11, 3.90], mean=2.97

**Root cause:**
With only 24 samples split randomly, the test set happens to contain only low-affinity pockets (pKd < 4). The model, trained primarily on mid-to-high affinity pockets, extrapolates poorly to this range. The 4-sample test set is too small for any statistical reliability (Spearman ρ with N=4 has p=0.20).

### 2.5 🟡 P5: SVGP is Overkill for This Data Size

**Evidence:**
- SVGP with 48 inducing points for 200 (effective 16) training points
- Variational approximation introduces unnecessary error when exact GP is feasible

**Root cause:**
SVGP is designed for datasets too large for exact GP (N > 1000). For N=24 (or even N=200 augmented), exact GP with direct marginal likelihood optimization is computationally trivial and provides superior hyperparameter estimates via type-II maximum likelihood.

### 2.6 🟡 P6: Uncertainty is Poorly Calibrated

**Evidence:**
- 95% CI width ≈ 5.5 pKd units (very wide) across all splits
- Yet test coverage is only 50% (2 of 4 points fall outside CI)
- The uncertainty is high but not in the right places

**Root cause:**
The GP uncertainty reflects distance in the augmented embedding space, not in the true molecular similarity space. Points that are "far" in ECFP4 space but have similar pKd get high uncertainty, while nearby augmented points get low uncertainty despite being synthetic.

---

## 3. Root Cause Analysis

The core problem is a **data scarcity cascade**:

```
93 pockets sampled by TargetDiff
  → 49 have non-empty SDF files (44 failed molecule reconstruction)
    → 24 match pKd labels in affinity_info.pkl
      → 16 used for training (after 20% val + 20% test split)
        → Augmented to 200 with Gaussian noise on binary features
```

At each stage, information is lost. The final 16 real training points in 128-dimensional space make any ML model prone to overfitting. The augmentation strategy, while well-intentioned, introduces artifacts because ECFP4 fingerprints are binary.

---

## 4. Optimization Plan

### 4.1 Phase 1: Quick Fixes (no code changes to core GP)

#### 4.1.1 Early Stopping
- Stop training at the epoch with minimum validation loss (epoch 28)
- Saves 91% of training time and prevents overfitting
- **Expected impact:** Val RMSE should improve significantly

#### 4.1.2 Reduce Augmentation Noise
- Lower σ_x from 0.3 to 0.05–0.1 (respect binary feature scale)
- Lower σ_y from 0.5 to 0.1–0.2 (pKd std ≈ 1.9, so σ_y=0.5 is ~26% relative noise)
- Or disable augmentation entirely and rely on GP's built-in small-sample capability

#### 4.1.3 Use Stratified Splitting
- Sort pockets by pKd, then assign to train/val/test in a round-robin fashion
- Ensures each split covers the full pKd range
- With only 24 samples, use **Leave-One-Out Cross-Validation (LOOCV)** instead of fixed splits

### 4.2 Phase 2: Model Architecture Changes

#### 4.2.1 Switch to Exact GP
- Replace SVGP with `ExactGP` — only 24 points, no approximation needed
- Use type-II maximum likelihood (marginal likelihood optimization) for hyperparameters
- This is the gold standard for small datasets and eliminates variational approximation error

```python
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        )
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
```

#### 4.2.2 Dimensionality Reduction Before GP
- Apply PCA to reduce 128-dim ECFP4 → 5–10 principal components before GP
- Reduces effective parameters (128 ARD lengthscales → 5–10)
- Avoids curse of dimensionality with 16 training points
- Alternative: Use a fixed Tanimoto kernel (no lengthscale learning) on raw fingerprints

#### 4.2.3 Tanimoto / Jaccard Kernel for Binary Fingerprints
- ECFP4 fingerprints are **binary** → Euclidean distance (used by Matérn/RBF) is suboptimal
- Tanimoto similarity is the standard metric for molecular fingerprints
- Implement a custom Tanimoto kernel:

```python
class TanimotoKernel(gpytorch.kernels.Kernel):
    def forward(self, x1, x2, **params):
        x1x2 = x1 @ x2.T
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)
        return x1x2 / (x1_sq + x2_sq.T - x1x2 + 1e-8)
```

#### 4.2.4 Informative Priors on Hyperparameters
- Place priors on lengthscales to prevent them from collapsing or exploding:
  - `LogNormal(0, 1)` prior on lengthscales
  - `Gamma(2, 0.5)` prior on output variance
- This constrains the model when data is scarce

### 4.3 Phase 3: Data & Evaluation Strategy

#### 4.3.1 Leave-One-Out Cross-Validation (LOOCV)
- With N=24, LOOCV trains 24 models, each on 23 points, predicting the held-out one
- Provides 24 test predictions with minimal data waste
- Computes robust metrics across all folds
- For exact GP, LOOCV can be computed analytically (no retraining needed)

#### 4.3.2 Increase Labeled Data
- 25 of 49 non-empty pockets lack pKd labels — check if labels exist in alternative databases (ChEMBL, PDBbind, BindingDB)
- Some of the 44 empty-SDF pockets might have valid molecules if TargetDiff sampling is re-run with different parameters

#### 4.3.3 Higher-Dimensional Fingerprints
- Current: ECFP4 with 128 bits — very compressed, high collision rate
- Try: 1024 or 2048 bits (standard for ML on fingerprints) + PCA down to 20–50 dims
- Also consider: ECFP6 (radius=3) or FCFP4 (pharmacophore-based)

### 4.4 Phase 4: Advanced Approaches

#### 4.4.1 Multi-Task GP
- Instead of per-pocket mean embedding → per-pocket pKd, model individual molecules
- Use molecule-level embeddings with pocket identity as a task index
- Shares kernel structure across pockets for better generalization

#### 4.4.2 Bayesian Optimization of GP Hyperparameters
- Use a proper Bayesian approach (MCMC or HMC) instead of point-estimate MAP
- GPyTorch supports `pyro` backend for fully Bayesian inference
- Provides posterior uncertainty over hyperparameters, critical for small data

#### 4.4.3 Ensemble GP
- Train K GP models with different random seeds and/or kernel choices
- Average predictions and combine uncertainties
- Provides more robust predictions with small data

---

## 5. Recommended Priority Order

| Priority | Action | Difficulty | Expected Impact |
|----------|--------|:----------:|:---------------:|
| 🥇 1 | Early stopping (epoch ≈28) | Trivial | High — stops overfitting immediately |
| 🥇 2 | LOOCV instead of fixed split | Easy | High — uses all 24 points, robust metrics |
| 🥇 3 | Switch to Exact GP | Easy | High — better hyperparams for small data |
| 🥈 4 | PCA(128→10) before GP | Easy | Medium — reduces curse of dimensionality |
| 🥈 5 | Tanimoto kernel | Medium | Medium — proper similarity for fingerprints |
| 🥈 6 | Remove or reduce augmentation | Trivial | Medium — eliminates augmentation artifacts |
| 🥉 7 | Increase fingerprint bits (1024) | Easy | Low-Medium — less information loss |
| 🥉 8 | Hyperparameter priors | Medium | Low-Medium — regularization |
| 🥉 9 | Additional pKd labels | Research | High — more data always helps |
| 🥉 10 | Fully Bayesian inference | Hard | Medium — better uncertainty |

---

## 6. Generated Figures

All figures are in `results/embedding_rdkit/gp_analysis/figures/`:

| File | Description |
|------|-------------|
| `training_curves.png` | 2×2 panel: ELBO loss, RMSE, NLL, and kernel hyperparameters per epoch. Shows clear train/val divergence after epoch ~28. |
| `pred_vs_true_splits.png` | Predicted vs true pKd scatter for train/val/test with error bars. Train shows good correlation; val/test show anti-correlation. |
| `prediction_analysis.png` | 2×2 panel: residual histograms, uncertainty calibration plot, per-split metric comparison bars, and 95% CI coverage chart. |
| `gp_model_analysis.png` | PCA projection of embeddings colored by pKd, and pKd distribution histograms per split. Shows test set has narrow low-pKd range. |

---

## 7. Key Takeaway

The current GP model achieves good **training** fit (R²=0.51, ρ=0.72) but **fails to generalize** (test R²=-16.5). This is not a fundamental failure of the BayesDiff framework — it's a predictable consequence of fitting a high-parameter model to 16 data points in 128 dimensions. The path forward is clear:

1. **First**: Exact GP + LOOCV + early stopping → should immediately reveal whether ECFP4 embeddings carry genuine predictive signal
2. **Then**: Tanimoto kernel + PCA → domain-appropriate modeling
3. **Finally**: More data (labels, fingerprint bits) → scale up once the model is validated

The fact that train ρ=0.72 suggests the ECFP4 fingerprints **do** contain binding-affinity signal. The challenge is extracting it without overfitting, which the optimizations above should address.
