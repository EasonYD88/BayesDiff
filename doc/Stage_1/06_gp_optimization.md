# GP Model Analysis & Optimization Plan

> Generated from `scripts/studies/gp_training_analysis.py` run on 2026-03-26  
> Results directory: `results/embedding_rdkit/gp_analysis/`  
> **Updated 2026-03-26**: Embedding upgrade strategy & robust evaluation plan

---

## 1. Current Results Summary

### 1.1 Data Setup

| Item | Value |
|------|-------|
| Total pockets with ECFP4 embeddings | 49 (out of 93) |
| Pockets with pKd labels | 24 |
| Embedding dimension | 128 (Morgan fingerprint, radius=2) |
| Train / Val / Test split | 16 / 4 / 4 (single random split — **unreliable**) |
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

## 4. Initial Optimization Plan

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

## 5. Initial Priority Order

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

## 6. Two Fundamental Bottlenecks (Revised Analysis)

Before listing individual fixes, it's important to recognize two **structural** problems that dominate everything else:

### Bottleneck A: Embedding Expressiveness

ECFP4 (128-bit Morgan fingerprint) is a hashing-based representation designed for substructure search, not for predicting continuous binding affinity. Its limitations:

- **128-bit collision**: The standard for ML is 1024–2048 bits. At 128 bits, many structurally distinct substructures hash to the same bit → information loss
- **No 3D / interaction info**: ECFP4 encodes 2D topology only. Binding affinity depends on 3D pose, protein-ligand contacts, and electrostatics — none of which are captured
- **Binary and sparse**: Mean non-zero fraction is 39.3%. A GP with Matérn/RBF kernel in this sparse binary space wastes capacity on meaningless Euclidean distances

The train ρ=0.72 suggests **some** signal exists, but ECFP4 hits a ceiling quickly. No amount of GP tuning will overcome an embedding that doesn't encode the relevant chemistry.

### Bottleneck B: Evaluation Instability

A single 16/4/4 random split with N=24 is **statistically unreliable**:
- Test ρ=-0.8 this run, but with different seed it could be +0.6 — pure sampling noise
- p-values for N=4 are all non-significant (p>0.13)
- Any conclusion drawn from this split is anecdotal, not scientific

These two bottlenecks must be addressed **simultaneously** — better embeddings evaluated on unstable splits, or stable evaluation of bad embeddings, both lead nowhere.

---

## 6.1 Robust Evaluation Results (Completed)

> Run: HPC job 4988326 on L40S (gl027), 1 min 34 sec  
> Script: `scripts/studies/robust_evaluation.py`  
> Output: `results/embedding_rdkit/robust_eval/`

Three protocols were executed using **Exact GP** (isotropic Matérn-5/2) on ECFP4-128 embeddings (N=24 labeled pockets, no augmentation):

| Protocol | RMSE | R² | Spearman ρ | NLL | CI-95% Coverage |
|----------|:----:|:--:|:----------:|:---:|:---------------:|
| **LOOCV (analytic)** | 2.62 | -0.75 | -0.33 | 2.46 | 88% |
| **50× Repeated 70/30** | 2.24±0.33 | -0.55±0.75 | -0.19±0.35 | 2.31±0.22 | 89%±11% |
| **200× Bootstrap** | 2.08 [1.54,2.56] | -0.22 [-0.82,0.23] | 0.27 [-0.14,0.62] | 2.15 | 92% |

**Key findings**:

1. **ECFP4-128 carries NO predictive signal for pKd** — Spearman ρ is negative or near-zero across all protocols. The bootstrap 95% CI for ρ is [-0.14, 0.62], spanning zero.
2. **Previous train ρ=0.72 was pure overfitting** — LOOCV (which prevents data leakage) gives ρ=-0.33.
3. **The GP is essentially predicting the mean** — R² is negative in all protocols (worse than a constant predictor).
4. **Uncertainty calibration is decent** — 88-92% CI coverage, but only because the predicted intervals are extremely wide (~5.5 pKd units).

**Conclusion**: Bottleneck A (embedding expressiveness) is confirmed as the dominant issue. No GP trick will help — we need richer molecular representations.

Figures: `results/embedding_rdkit/robust_eval/figures/`
- `robust_evaluation.png` — LOOCV scatter, repeated-split violin, bootstrap CI bars
- `loocv_diagnostics.png` — Residuals, calibration plot, QQ plot
- `summary_table.png` — Side-by-side protocol comparison

---

## 7. Revised Optimization Plan (with Embedding Upgrade & Robust Evaluation)

### Phase 1: Robust Evaluation Framework (Priority: 🥇 Critical)

**Goal**: Establish trustworthy metrics before changing anything else.

#### 7.1.1 Leave-One-Out Cross-Validation (LOOCV)

With N=24, LOOCV is the gold standard:
- Train 24 models, each on 23 points, predict the held-out one
- For Exact GP, LOOCV can be computed **analytically** from the full-data posterior (no retraining needed):

```python
# Analytic LOOCV for Exact GP
K = kernel(X, X) + sigma_n**2 * I
K_inv = torch.linalg.inv(K)
alpha = K_inv @ y
loocv_mu = y - alpha / K_inv.diag()
loocv_var = 1.0 / K_inv.diag()
```

- Reports: RMSE, Spearman ρ, NLL, calibration — each with N=24 predictions
- **No** train/val/test split ambiguity

#### 7.1.2 Repeated Random Split (complement to LOOCV)

When we want train-vs-test gap analysis:
- 50× repeated 70/30 random split
- Report mean ± std for each metric
- Paired t-test or Wilcoxon for embedding comparisons

#### 7.1.3 Pocket-Level Bootstrap

- Draw 24 pockets with replacement, 1000× bootstrap
- Compute metric on each bootstrap → 95% CI via percentile method
- Tells us: "Given this data, how confident are we in ρ=0.47?"

**Deliverable**: A single script `scripts/studies/robust_evaluation.py` that runs all three protocols and produces a comparison table.

### Phase 2: Embedding Baseline Comparison (Priority: 🥇 Critical)

**Goal**: Determine whether the bottleneck is the GP or the embedding by comparing multiple representations under identical evaluation.

#### 7.2.1 Embedding Candidates

| # | Embedding | Dim | Type | Availability | What It Captures |
|---|-----------|:---:|------|:------------:|------------------|
| **E1** | ECFP4-128 | 128 | Binary FP | ✅ Current | 2D substructures (high collision) |
| **E2** | ECFP4-2048 | 2048 | Binary FP | ✅ RDKit | 2D substructures (low collision) |
| **E3** | ECFP6-2048 | 2048 | Binary FP | ✅ RDKit | Larger substructures (radius=3) |
| **E4** | FCFP4-2048 | 2048 | Binary FP | ✅ RDKit | Pharmacophore features |
| **E5** | RDKit-2D | 200 | Continuous | ✅ RDKit (217 descriptors) | Physicochemical properties (MW, LogP, TPSA, HBD/HBA, rotatable bonds, etc.) |
| **E6** | Combined (E2+E5) | ~2248 | Mixed | ✅ RDKit | Structure + physicochemistry |
| **E7** | GNN (SchNet/DimeNet) | 256 | Learned | ✅ PyG 2.7.0 installed | 3D molecular graph from SDF |
| **E8** | ChemBERTa | 768 | Pretrained LM | ❌ Need `pip install transformers` | SMILES-based contextual |

**Note on availability**: RDKit is installed with 217 descriptors. PyTorch Geometric 2.7.0 is installed (supports SchNet, DimeNet, EGNN). HuggingFace `transformers` is **not** installed but can be added.

#### 7.2.2 Extraction Strategy

All embeddings are extracted from the same 49 SDF files in `results/embedding_1000step/`:
```
results/embedding_1000step/20260305_085825_j3387783/shards/shard_*of4/{pocket}/{pocket}_generated.sdf
```

Each SDF contains multiple generated molecules with 3D coordinates. For each pocket:
1. Read all molecules from SDF
2. Compute embedding for each molecule
3. Mean-pool across molecules → one vector per pocket

For **RDKit-2D descriptors** (E5):
```python
from rdkit.Chem import Descriptors
desc_names = [d[0] for d in Descriptors._descList]  # 217 descriptors
feats = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
```

For **GNN** (E7), use pretrained SchNet from PyG:
```python
from torch_geometric.nn import SchNet
model = SchNet.from_qm9(cutoff=10.0)  # pretrained on QM9
# Convert SDF → Data object with atom positions
```

#### 7.2.3 Comparison Protocol

For each embedding E1–E7:
1. Extract per-pocket mean embedding
2. Run Exact GP + LOOCV (analytic)
3. Run 50× repeated 70/30 split
4. Record: RMSE (mean±std), Spearman ρ (mean±std), NLL (mean±std), CI-95% coverage

Output: A single comparison table showing which embedding carries the most predictive signal for pKd.

**Deliverable**: `scripts/studies/embedding_comparison.py` — extracts all embeddings, runs comparison, generates table + figures.

### 7.2.4 Embedding Comparison Results (Completed)

**Run**: HPC job 4988787 on L40S GPU (gl048), completed in ~30s.

#### Matérn Kernel LOOCV (PCA to 20 dims)

| Embedding | d (orig→PCA) | RMSE | R² | Spearman ρ | CI-95% |
|-----------|:------------:|:----:|:---:|:---------:|:------:|
| ECFP4-128 | 128→20 | 3.53 | -2.17 | -0.28 | 71% |
| ECFP4-2048 | 2048→20 | 5.52 | -6.76 | 0.31 | 46% |
| ECFP6-2048 | 2048→20 | 5.52 | -6.76 | 0.31 | 46% |
| FCFP4-2048 | 2048→20 | 5.40 | -6.42 | 0.14 | 50% |
| RDKit-2D | 217→20 | 3.47 | -2.07 | -0.15 | 75% |
| Combined | 2265→20 | 5.52 | -6.76 | 0.31 | 46% |

#### Tanimoto Kernel LOOCV (binary FPs only, raw dimensions)

| Embedding | RMSE | R² | Spearman ρ | CI-95% |
|-----------|:----:|:---:|:---------:|:------:|
| ECFP4-128 | 2.22 | -0.25 | -0.37 | 92% |
| ECFP4-2048 | 2.43 | -0.51 | -0.30 | 96% |
| ECFP6-2048 | 2.49 | -0.58 | -0.32 | 96% |
| FCFP4-2048 | 2.31 | -0.36 | -0.23 | 96% |

#### 30× Repeated Split (70/30, Matérn)

| Embedding | RMSE (mean±std) | Spearman ρ (mean±std) |
|-----------|:---------------:|:---------------------:|
| ECFP4-128 | 2.41±0.39 | -0.21±0.31 |
| ECFP4-2048 | 2.48±0.57 | nan (constant pred) |
| ECFP6-2048 | 2.48±0.57 | nan (constant pred) |
| FCFP4-2048 | 2.48±0.57 | -0.10±0.26 |
| RDKit-2D | 2.40±0.49 | -0.04±0.40 |
| Combined | 2.48±0.57 | nan (constant pred) |

#### Key Findings

1. **🔴 ALL embeddings fail** — No embedding produces meaningful predictions. All R² < 0 (worse than predicting the mean). All ρ ≈ 0 or negative.

2. **🔴 High-dim FPs collapse after PCA** — ECFP4-2048, ECFP6-2048, Combined all give identical results (RMSE=5.52, R²=-6.76). PCA 2048→20 destroys the sparse binary structure. LOOCV scatter plots show constant ~0 predictions.

3. **🟡 Tanimoto kernel helps but not enough** — For ECFP4-128, Tanimoto reduces RMSE from 3.53→2.22 and improves CI coverage to 92%. But ρ is still -0.37 (anti-correlated). The kernel choice alone cannot rescue a representation without signal.

4. **🟡 RDKit-2D is "least bad"** — Best CI-95% coverage (75%), lowest RMSE with Matérn (3.47), but ρ=-0.15 (random).

5. **🔴 The bottleneck is NOT embedding type** — We tested 6 fundamentally different representations (circular FPs, pharmacophore FPs, physicochemical descriptors, and combinations). None carry predictive signal for pKd. This confirms the bottleneck is either:
   - **Data quantity**: N=24 is too small for any GP to learn structure→affinity mapping
   - **Information gap**: Mean-pooled features of *generated* molecules may not encode the pocket-level binding affinity at all — the GP oracle assumption may be flawed

#### Figures

All figures in `results/embedding_comparison/figures/`:

| File | Description |
|------|-------------|
| `embedding_comparison_bars.png` | 4-panel comparison: RMSE, R², Spearman ρ, CI-95% for all embeddings (Matérn + Tanimoto) |
| `loocv_scatter_all.png` | 6-panel LOOCV predicted vs true pKd — high-dim FPs clearly collapse to constant predictions |
| `comparison_table.png` | Summary table with all metrics |

### Phase 3: GP Model Improvements (Priority: 🥈 Important)

Apply these **after** determining the best embedding from Phase 2.

#### 7.3.1 Switch to Exact GP

- Replace SVGP with `ExactGP` — N=24, no approximation needed
- Type-II marginal likelihood for hyperparameters (gold standard for small data)
- Enables analytic LOOCV (7.1.1)

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

#### 7.3.2 Tanimoto Kernel for Binary Fingerprints

For E1–E4 (binary fingerprints), replace Matérn with Tanimoto:

```python
class TanimotoKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False  # Tanimoto has no hyperparameters
    def forward(self, x1, x2, **params):
        x1x2 = x1 @ x2.T
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)
        return x1x2 / (x1_sq + x2_sq.T - x1x2 + 1e-8)
```

Benefits: No ARD lengthscales to overfit (0 extra parameters), proper similarity metric for fingerprints.

#### 7.3.3 Dimensionality Reduction

- PCA before GP: reduce d → min(d, 10–20) for any embedding with d > 50
- Particularly important for E2/E3/E4 (2048-dim) and E6 (2248-dim)
- PCA variance threshold: keep components explaining 95% of variance

#### 7.3.4 Remove Data Augmentation

- With Exact GP, augmentation is unnecessary — GP naturally handles small N
- Augmentation on binary features creates out-of-distribution artifacts
- If needed, consider bit-flip augmentation instead of Gaussian noise

#### 7.3.5 Hyperparameter Priors

```python
model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.LogNormalPrior(0.0, 1.0)
model.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.5)
likelihood.noise_prior = gpytorch.priors.GammaPrior(1.0, 1.0)
```

### Phase 4: Advanced Approaches (Priority: 🥉 Future)

#### 7.4.1 Multi-Task GP
- Model individual molecules (not pocket means), using pocket ID as task index
- Shares kernel across pockets → better generalization with more data points

#### 7.4.2 Fully Bayesian GP (MCMC/HMC)
- Use `pyro` backend for posterior over hyperparameters
- Critical for small-data uncertainty quantification

#### 7.4.3 Ensemble GP
- Train K models with different kernels (Tanimoto, Matérn, RBF) and seeds
- Average predictions, combine uncertainties

#### 7.4.4 Fine-tuned GNN Embedding
- Instead of pretrained SchNet features, fine-tune a GNN end-to-end on pKd
- Would require more data or transfer learning from PDBbind

---

## 8. Recommended Implementation Order

| Step | Action | Deliverable | Why First |
|:----:|--------|-------------|-----------|
| **1** | Robust eval framework | `scripts/studies/robust_evaluation.py` | Must measure before optimizing |
| **2** | Extract all embeddings (E1–E7) | `results/embedding_rdkit/all_embeddings_*.npz` | Enables fair comparison |
| **3** | Embedding × GP comparison table | Comparison table + figure | Identifies the real bottleneck |
| **4** | Best-embedding + Exact GP + Tanimoto | Updated `bayesdiff/gp_oracle.py` | Best model for current data |
| **5** | Advanced GP (priors, Bayesian) | Optional refinement | Diminishing returns |

The key insight: **Steps 1–3 are diagnostic** (they tell us *what* to fix), while steps 4–5 are *fixes*. Doing fixes without diagnosis is premature optimization.

---

## 9. Generated Figures

All figures are in `results/embedding_rdkit/gp_analysis/figures/`:

| File | Description |
|------|-------------|
| `training_curves.png` | 2×2 panel: ELBO loss, RMSE, NLL, and kernel hyperparameters per epoch. Shows clear train/val divergence after epoch ~28. |
| `pred_vs_true_splits.png` | Predicted vs true pKd scatter for train/val/test with error bars. Train shows good correlation; val/test show anti-correlation. |
| `prediction_analysis.png` | 2×2 panel: residual histograms, uncertainty calibration plot, per-split metric comparison bars, and 95% CI coverage chart. |
| `gp_model_analysis.png` | PCA projection of embeddings colored by pKd, and pKd distribution histograms per split. Shows test set has narrow low-pKd range. |

---

## 10. Key Takeaway

### Original Hypothesis
The current GP model achieves good **training** fit (R²=0.51, ρ=0.72) but **fails to generalize** (test R²=-16.5). Two structural issues were hypothesized:
1. **Embedding ceiling**: ECFP4-128 cannot encode the chemistry that determines binding affinity.
2. **Evaluation noise**: N=24 with a single random split produces results dominated by sampling noise.

### What We Found (After Phases 1–3)

✅ **Phase 1 (Robust Evaluation)**: Confirmed — single-split results are unreliable. LOOCV and repeated splits show ECFP4-128 has ρ≈-0.2 to -0.3, not the +0.72 seen in training.

✅ **Phase 2 (Embedding Comparison)**: **All 6 embeddings fail equally.** The bottleneck is NOT embedding type. ECFP4, ECFP6, FCFP4, RDKit-2D, and Combined all produce ρ ≈ 0 ± 0.3 under LOOCV. The Tanimoto kernel helps RMSE but cannot rescue a fundamentally missing signal.

✅ **Phase 3 (Bayesian Optimization of GP Hyperparameters)**: **Exhaustive search confirms no GP config works.**
- 200 Optuna trials searching over: 6 embeddings × 5 kernels × PCA dims × ARD × priors × LR × epochs
- Best config: **FCFP4-2048 + RQ kernel + PCA→10, isotropic, outputscale prior**
- Best LOOCV: RMSE=2.07, **ρ=-0.42** (anti-correlated!), R²=-0.09
- 50× repeated split validation: RMSE=2.09±0.33, **ρ=0.11±0.33** (not significant)
- All top-5 trials are FCFP4-2048 + RQ kernel variants with nearly identical RMSE≈2.07
- Key insight from BO: **RQ kernel >> Matérn/RBF** (RMSE 2.1 vs 5.2), but even the best kernel only achieves mean-prediction level performance

**BO Figures**: `results/bo_gp/figures/`
| File | Description |
|------|-------------|
| `bo_optimization_history.png` | 4-panel: optimization convergence, RMSE by embedding, RMSE by kernel, ρ by embedding |
| `bo_best_config_loocv.png` | LOOCV scatter (flat predictions ≈ mean pKd), residuals, calibration plot |
| `bo_validation_splits.png` | 50× split metrics with error bars |
| `bo_summary_table.png` | Best configuration parameters and performance metrics |

### Revised Conclusion

After three systematic phases (robust evaluation → embedding comparison → Bayesian optimization), we can **definitively conclude**:

1. **The GP oracle cannot predict pKd from mean-pooled molecular features with N=24.** This is not a hyperparameter tuning problem — 200 Optuna trials across the full search space could not find a single configuration with positive LOOCV Spearman ρ.
2. **The information pathway is broken**: Generated molecules → mean-pooled embedding → pKd is too lossy. The GP essentially learns to predict the mean pKd (≈5.0) for every pocket.
3. **Calibration is acceptable**: CI-95% coverage ≈ 92% (conservative), meaning the GP at least "knows what it doesn't know" — but knowing nothing useful.

### Recommended Next Steps

1. **Increase data**: More pockets with pKd labels (target N ≥ 100) — this is the single most impactful change
2. **Richer information pathway**: Instead of mean-pooling, consider distribution-level features (variance, percentiles) or attention-weighted pooling
3. **Learned embeddings**: GNN on 3D molecular graphs (SchNet/DimeNet, PyG available) may capture what fixed fingerprints miss
4. **Alternative oracle design**: Consider per-molecule scoring (not per-pocket mean) or direct docking score prediction
5. **Use the GP as uncertainty estimator, not predictor**: Since calibration is decent, the GP can still be useful for ranking pockets by uncertainty even if it can't predict pKd accurately

---

## 11. Data Acquisition Plan

### 11.1 Current Data Pipeline Bottleneck

```
93 test pockets ──┬── 48 with pKd ──┬── 24 with SDF ✅ (current GP training set)
                  │                 └── 24 WITHOUT SDF ← Easy Win (Tier 1)
                  └── 45 without pKd ── 25 with SDF (wasted generation)
```

**Root cause**: Only 24 pockets have BOTH generated molecules (SDF) AND binding affinity labels (pKd). We lose data at two stages:
1. **Generation failure**: 44/93 pockets fail during TargetDiff 1000-step sampling (52.7% success)
2. **Label mismatch**: Only 48/93 test pockets have valid pKd in `affinity_info.pkl`

**Full data inventory** (from `affinity_info.pkl`):

| Scope | Entries | Pocket families | With pKd |
|-------|---------|-----------------|----------|
| Full affinity_info.pkl | 184,087 | 2,474 | 1,041 |
| Test set (93 pockets) | - | 93 | 48 |
| CrossDocked train split | 100,000 | ~1,000+ | ~993 |

### 11.2 Three-Tier Data Expansion Strategy

#### Tier 1: Recover Missing SDFs for 24 Labeled Pockets (24 → 48, 2× gain)

**Goal**: Double the GP training set by regenerating molecules for the 24 pockets that have pKd but failed SDF generation.

**Effort**: Low (rerun existing pipeline) | **Impact**: High (N doubles) | **Time**: ~4-8 hours on A100

**Missing pockets** (all have valid pKd):

| Pocket | pKd | Pocket | pKd |
|--------|:---:|--------|:---:|
| ABL2_HUMAN_274_551_0 | 8.13 | M3K14_HUMAN_321_678_0 | 7.28 |
| ATS5_HUMAN_262_480_0 | 6.78 | NAGZ_VIBCH_1_330_0 | 6.64 |
| BACE2_HUMAN_76_460_0 | 6.12 | NPD_THEMA_1_246_0 | 3.00 |
| BGAT_HUMAN_63_353_0 | 3.25 | NR1H4_HUMAN_258_486_0 | 7.37 |
| BGL07_ORYSJ_25_504_0 | 3.92 | P2Y12_HUMAN_1_342_0 | 7.36 |
| CHIB_SERMA_1_499_0 | 4.17 | PA2B8_DABRR_1_121_0 | 6.80 |
| CONA_CANCT_1_237_0 | 5.48 | PAK4_HUMAN_291_591_ATP_0 | 6.88 |
| FKB1A_HUMAN_2_108_0 | 7.58 | POL_FOAMV_861_1060_0 | 7.27 |
| HDAC8_HUMAN_1_377_0 | 6.67 | QPCT_HUMAN_33_361_0 | 4.94 |
| KS6A3_HUMAN_41_357_0 | 7.88 | ROCO4_DICDI_1009_1292_0 | 4.81 |
| SIR3_HUMAN_117_398_0 | 7.59 | UPPS_ECOLI_1_253_0 | 5.13 |
| TNKS1_HUMAN_1099_1319_0 | 7.65 | TNKS2_HUMAN_948_1162_0 | 6.99 |

**Implementation plan**:
1. Create `data/splits/missing_pk_pockets.txt` with these 24 pocket names
2. Run `scripts/pipeline/s02_sample_molecules.py --pocket_list data/splits/missing_pk_pockets.txt --num_steps 1000 --num_samples 64`
3. Use lower step count (500) or different seeds as fallback for stubborn pockets
4. Extract ECFP4/FCFP4/RDKit embeddings from new SDFs
5. Merge into existing dataset → N=48 for GP retraining

**Why they failed initially**: Likely causes include:
- Invalid atom types or valence errors in generated molecules
- Pocket geometry incompatible with TargetDiff's SE(3) diffusion
- Numerical instability during 1000-step denoising

**Recovery strategies**:
- Try fewer steps (500, 200) — less accurate but more stable
- Try more samples with filtering (128 samples, keep best 64)
- Different random seeds
- Relax molecule sanitization (allow partial hydrogens)

#### Tier 2: Mine CrossDocked Training Data (48 → 150+, 3× gain)

**Goal**: Extend beyond the 93-pocket test set by mining the CrossDocked2020 training LMDB for additional pockets with pKd labels.

**Effort**: Medium | **Impact**: Very High | **Time**: 1-2 days

**Data source**: `external/targetdiff/data/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` (4.1GB)
- 100,000 train entries across ~1,000+ pocket families
- **993 non-test pocket families with valid pKd** in `affinity_info.pkl`

**Implementation plan**:
1. **Extract pocket structures from LMDB**:
   ```python
   import lmdb
   env = lmdb.open(lmdb_path, readonly=True)
   # Decode entries → pocket PDB + ligand SDF
   # Group by pocket family → select representative structure per family
   ```
2. **Filter pockets with pKd**: Cross-reference with `affinity_info.pkl` → ~993 families
3. **Select diverse subset**: Use pocket-family clustering (mmseqs2) to pick ~100-200 diverse pockets
4. **Generate molecules**: Run TargetDiff sampling on selected pockets (A100 array job)
5. **Extract embeddings + train GP**: With N≈150+, GP has much better chance of learning

**Key considerations**:
- Must ensure no **data leakage**: TargetDiff model was trained on these pockets, so generated molecules may be biased
- Solution: Use the **test-split pockets only** for final evaluation; training-split pockets for GP training
- Alternative: Use TargetDiff's own train/val/test splits to maintain integrity

**Risk**: TargetDiff was trained on these pockets → generated molecules may memorize training ligands rather than exploring novel chemistry. This is actually OK for the GP oracle (we want representative molecules), but must be acknowledged.

#### Tier 3: External Label Expansion via PDBbind (150+ → 300+)

**Goal**: Add more pKd labels from PDBbind v2020 for pockets that exist in CrossDocked but lack labels in `affinity_info.pkl`.

**Effort**: Medium-High | **Impact**: High | **Time**: 2-3 days

**Data source**: PDBbind v2020 refined set (~3,600 complexes with experimental Kd/Ki/IC50)
- Files present: `external/targetdiff/data/data/pdbbind_v2020/`
- Parser exists: `bayesdiff/data.py::parse_pdbbind_index()` — already handles all affinity types
- Script exists: `scripts/pipeline/s01_prepare_data.py` — designed for PDBbind parsing but not used in current pipeline

**Implementation plan**:
1. Parse PDBbind v2020 INDEX file with existing `parse_pdbbind_index()`
2. Map PDBbind PDB codes → CrossDocked pocket families (by UniProt ID or pocket overlap)
3. For each new labeled pocket:
   - Check if structure exists in LMDB or test set
   - If yes → generate molecules → extract embeddings
   - If no → extract pocket from PDB file using `scripts/pipeline/s01_prepare_data.py`
4. Merge all labels into unified dataset

**Challenge**: Mapping between PDBbind PDB codes and CrossDocked pocket family names requires protein-level matching (sequence or structure alignment).

### 11.3 Expected Impact on GP Performance

| Tier | N (train) | Expected LOOCV ρ | Rationale |
|:----:|:---------:|:-----------------:|-----------|
| Current | 24 | ≈ 0 (random) | N too small for any signal |
| Tier 1 | 48 | 0.1-0.3 (maybe) | Still small; main value is tighter confidence intervals |
| Tier 2 | 150+ | 0.3-0.5 (hopeful) | Enough data for simple structure-activity patterns |
| Tier 3 | 300+ | 0.4-0.6 (target) | Sufficient for GP with RQ kernel + FCFP4 (best BO config) |

**Important caveat**: More data only helps if the **information pathway** (generated molecules → embedding → pKd) contains signal. If TargetDiff generates similar molecules for all pockets regardless of binding affinity, then N=300 will still fail. Tier 1 (N=48) will quickly test this hypothesis.

### 11.4 Recommended Execution Order

| Step | Action | Deliverable | Prerequisite |
|:----:|--------|-------------|:------------:|
| **1** | Create missing pocket list | `data/splits/missing_pk_pockets.txt` | None |
| **2** | Regenerate SDFs for 24 missing pockets | New SDF files in `results/` | Step 1 |
| **3** | Extract embeddings + retrain GP | LOOCV with N=48 | Step 2 |
| **4** | Evaluate: does doubling N improve ρ? | Updated results | Step 3 |
| **5** | If yes → proceed to Tier 2 (LMDB mining) | 150+ pocket dataset | Step 4 confirms signal |
| **6** | If no → pivot to information pathway (mean-pool alternatives, GNN) | New architecture | Step 4 shows no signal |

**Step 4 is the critical decision point**: If N=48 shows meaningfully better ρ than N=24 (e.g., ρ > 0.15 with p < 0.05), then data quantity is the bottleneck and Tiers 2-3 are worth pursuing. If ρ remains ≈ 0, then the problem is fundamental and more data won't help.

### 11.5 Computational Budget Estimate

| Task | GPU Hours | Queue | Notes |
|------|:---------:|:-----:|-------|
| Tier 1: Regenerate 24 pockets (1000 steps × 64 samples) | 8-12h | A100 | Array job, 4 shards |
| Tier 1: Embedding extraction + GP training | 0.5h | L40S | Quick |
| Tier 2: LMDB parsing + pocket selection | 1h | CPU | No GPU needed |
| Tier 2: Generate for ~150 pockets | 24-36h | A100 | Array job, 8-16 shards |
| Tier 2: Embedding extraction + GP training | 1h | L40S | |
| Tier 3: PDBbind mapping + structure prep | 2-4h | CPU | Manual alignment needed |

**Total for Tier 1**: ~12 GPU hours (can complete in 1 day)
**Total for Tier 1+2**: ~48 GPU hours (2-3 days with queue waits)

---

## 12. Tier 3 Dataset: Full-Scale Results (N=932)

### 12.1 Data Acquisition Summary

Successfully executed a **comprehensive data acquisition pipeline** that expanded the dataset from N=24 to N=932 (39× increase):

| Step | Description | Result |
|------|-------------|--------|
| LMDB scan | Scanned 166,500 CrossDocked entries | 2,358 families found |
| Affinity matching | Matched with affinity_info.pkl | 1,041 families with pKd |
| Pocket extraction | Extracted pre-processed pocket .pt files | 1,019 LMDB + 48 test = 1,067 |
| GPU sampling | 16-shard L40S array job, 64 mols × 100 steps | Job 4994690, ~25 GPU-hours |
| Molecule generation | Valid SDF + embeddings produced | **932 pockets** (87.3% success) |
| Total molecules | Valid reconstructed molecules | 5,150 (5.3 ± 3.9 per pocket) |
| pKd coverage | Range [1.28, 15.22] | mean 7.08 ± 2.08 |

**Technical innovations**:
- Bypassed PDB file parsing entirely — loaded pre-processed protein data from LMDB `.pt` files
- Added `load_pocket_data()`, `sample_for_data()`, `sample_and_embed_data()` to `bayesdiff/sampler.py`
- Round-robin sharding across 16 GPUs for embarrassingly parallel generation

### 12.2 GP Training Configuration

Used the best configuration identified by Bayesian Optimization (§10, Phase 3):

| Parameter | Value |
|-----------|-------|
| Fingerprint | FCFP4-2048 |
| Kernel | Rational Quadratic (RQ) |
| PCA | None selected (full 2048-dim best in 5-fold CV) |
| Epochs | 150 (eval), 200 (full train) |
| Learning rate | 0.1 |
| Noise lower bound | 0.001 |

**PCA dimension selection** (5-fold CV):

| PCA dims | Variance explained | RMSE | ρ |
|----------|-------------------|------|---|
| 10 | 18.99% | 2.062 | 0.148 |
| 20 | 28.84% | 2.061 | 0.154 |
| 50 | 46.85% | 2.061 | 0.163 |
| 100 | 63.01% | 2.061 | 0.156 |
| Full (2048) | 100% | **2.061** | 0.147 |

All PCA variants produce nearly identical RMSE (~2.061). PCA is unnecessary — the GP kernel handles high-dimensional sparsity well with the RQ kernel.

### 12.3 Results

#### LOOCV (Analytic)

| Metric | Value |
|--------|-------|
| RMSE | 2.068 |
| MAE | 1.660 |
| Spearman ρ | 0.111 (p = 0.0007) |
| R² | 0.013 |

#### 5-Fold Cross-Validation

| Fold | RMSE | ρ | p-value |
|------|------|---|---------|
| 1 | 2.008 | 0.047 | 0.526 |
| 2 | 1.837 | 0.249 | 0.001 |
| 3 | 2.208 | 0.176 | 0.016 |
| 4 | 2.029 | 0.113 | 0.126 |
| 5 | 2.230 | 0.094 | 0.204 |
| **Overall** | **2.067** | **0.117** | **0.0003** |

#### 50× Repeated Random Splits (80/20)

| Metric | Mean ± Std |
|--------|------------|
| RMSE | 2.078 ± 0.099 |
| Spearman ρ | 0.134 ± 0.055 |
| R² | 0.011 ± 0.018 |

### 12.4 Comparison: N=24 → N=932

| Metric | N=24 (old) | N=932 (Tier 3) | Change |
|--------|-----------|----------------|--------|
| LOOCV RMSE | 2.07 | 2.07 | 0% |
| LOOCV ρ | −0.42 | **+0.11** | ✅ Sign flip |
| LOOCV R² | −0.09 | **+0.01** | ✅ No longer negative |
| 50× Split ρ | 0.11 ± 0.33 | **0.13 ± 0.06** | ✅ 6× smaller std |
| ρ significance | p > 0.05 | **p = 0.0007** | ✅ Now significant |

**Key improvements**:
1. **ρ is now consistently positive** — the repeated splits distribution never goes negative (all 50 runs ρ > 0)
2. **Statistically significant** — LOOCV p = 0.0007 (vs. p > 0.05 at N=24)
3. **Much more stable** — std of ρ dropped from 0.33 to 0.06

### 12.5 Diagnostic Analysis

#### Scatter Plot Pattern
The LOOCV and 5-Fold scatter plots show a characteristic **"horizontal band"** pattern:
- Predictions cluster in a narrow range [6.3, 7.5] regardless of true pKd
- This is essentially **mean-prediction + slight modulation**
- True pKd spans [1.28, 15.22] but predictions never exceed the [6, 7.5] range
- R² ≈ 0.01 means only **1% of variance** is explained

#### Residual Analysis
- Residuals are approximately Gaussian (μ=0.07, σ=2.07)
- No obvious heteroscedasticity — residual spread is uniform across predicted range
- Slight positive skew from high-pKd outliers (pKd > 12)

#### Uncertainty Calibration
- **Near-perfect calibration** — observed coverage closely follows the ideal diagonal
- The GP correctly reports its own uncertainty
- This means the GP "knows" it can't predict pKd — it honestly reports high uncertainty
- The noise variance dominates the kernel signal: σ_noise ≈ σ_data

### 12.6 Definitive Conclusions

**The FCFP4 fingerprint representation fundamentally cannot encode binding affinity information.** Scaling data 39× (N=24 → N=932) produced:
- ✅ Statistical stability (results are now reliable and reproducible)
- ✅ Correct sign on ρ (weak positive correlation exists)
- ✅ Excellent uncertainty calibration
- ❌ No practical predictive power (R² = 0.01, predictions ≈ constant)
- ❌ RMSE unchanged at ~2.07 pKd units

This conclusively rules out **data quantity** as the bottleneck. The problem is **representation quality**:

1. **ECFP/FCFP fingerprints** encode 2D molecular topology only — they capture substructure presence/absence but contain no information about:
   - 3D binding pose geometry
   - Protein-ligand interaction patterns
   - Binding pocket complementarity
   - Electrostatic/hydrophobic matching

2. **The GP model itself is not the issue** — it converges cleanly, is well-calibrated, and extracts the (tiny) signal that exists. With a better representation, the same GP could work well.

3. **The generated molecules are pocket-conditioned** (TargetDiff generates molecules for specific pockets), but the ECFP fingerprint **discards all pocket context** — it treats the molecule in isolation.

### 12.7 Implications for Next Steps

The path forward requires **pocket-aware, 3D-aware representations**:

| Approach | Information captured | Expected impact |
|----------|---------------------|-----------------|
| **TargetDiff encoder embeddings** | 3D protein-ligand interaction graph | High — captures binding geometry |
| **Protein-ligand interaction fingerprints (PLIF)** | H-bonds, π-stacking, hydrophobic contacts | Medium-High |
| **GNN on 3D complex** | Learned 3D interaction features | High |
| **ChemBERTa/MolBERT** | Richer 1D molecular representation | Low-Medium (still no pocket info) |
| **Concatenated [mol_fp ‖ pocket_fp]** | Independent mol + pocket features | Medium |

**Priority: Extract TargetDiff encoder embeddings** — these are SE(3)-equivariant representations that encode the protein-ligand interaction geometry. The TargetDiff model already computes these internally during sampling; the current issue is that `sample_diffusion_ligand()` returns 7 values instead of 8 (the embedding output is not exposed). Modifying the sampling code to return encoder hidden states would provide a representation that inherently captures binding context.

### 12.8 Generated Figures

All figures saved to `results/tier3_gp/figures/`:

| Figure | Description |
|--------|-------------|
| `training_curve.png` | NLL loss convergence (10.9 → 2.17 in 200 epochs) |
| `loocv_scatter.png` | LOOCV predicted vs true pKd with uncertainty coloring |
| `cv5_scatter.png` | 5-Fold CV predicted vs true pKd |
| `repeated_splits_dist.png` | Distribution of RMSE, ρ, R² across 50 random splits |
| `diagnostics.png` | Residual analysis + uncertainty calibration curve |
| `comparison_n24_vs_tier3.png` | Side-by-side RMSE and ρ comparison |

---

## 13. Next-Step Optimization Plan

### 13.1 Problem Diagnosis

经过系统实验，我们确认了问题的根源层次：

| 层次 | 组件 | 状态 | 证据 |
|------|------|------|------|
| **数据量** | N=24 → N=932 | ✅ 已解决 | RMSE 不变，ρ 标准差降6倍 |
| **GP模型** | RQ kernel + BO优化 | ✅ 已解决 | 校准完美，收敛正常 |
| **评估方法** | LOOCV + repeated splits | ✅ 已解决 | 结果可复现且稳定 |
| **表征质量** | ECFP/FCFP 指纹 | ❌ **瓶颈** | R²=0.01，预测≈常数 |

核心问题：ECFP/FCFP 只编码分子2D拓扑，**完全丢弃了蛋白-配体3D交互信息**。TargetDiff 生成的分子是 pocket-conditioned 的，但指纹把 pocket 信息全部丢弃了。

### 13.2 方案概览

按优先级排序，从最可行、预期收益最高的方案开始：

| 优先级 | 方案 | 预期收益 | 实现难度 | 估计时间 |
|--------|------|---------|---------|---------|
| **P0** | TargetDiff 编码器嵌入 | ★★★★★ | 低 | 1-2天 |
| **P1** | Vina 对接打分 | ★★★★ | 低 | 1天 |
| **P2** | 蛋白-配体交互指纹 (PLIF) | ★★★★ | 中 | 2-3天 |
| **P3** | 多表征融合 | ★★★★ | 中 | 1-2天 |
| **P4** | 端到端 GNN 回归 | ★★★★★ | 高 | 1-2周 |
| **P5** | ChemBERTa / MolBERT | ★★ | 中 | 2-3天 |

---

### 13.3 P0: TargetDiff 编码器 3D 嵌入（最高优先级）

**原理**：TargetDiff 内部的 `ScorePosNet3D` 模型通过 SE(3)-等变图注意力网络处理蛋白-配体复合物。其 `final_ligand_h`（维度128）编码了：
- 蛋白-配体原子间距离（Gaussian RBF 编码，20维）
- 多头注意力加权的交互特征
- 经过多层 UniTransformer 块传播的上下文信息
- 4类边类型（配体-配体、配体-蛋白、蛋白-配体、蛋白-蛋白）

**现状**：`sample_diffusion_ligand()` 返回7个值（缺少嵌入），`sampler.py` 已有处理8个返回值的代码但 fallback 到零向量。

**实现方案**：

```python
# 方案A：修改 sample_diffusion.py 在最终步提取嵌入
# 在 sample_diffusion_ligand() 的扩散循环最后一步：
preds = model(...)
final_ligand_h = preds['final_ligand_h']  # (N_ligand, hidden_dim)
# scatter_mean 聚合为分子级别嵌入：
mol_embeddings = scatter_mean(final_ligand_h, batch_ligand, dim=0)  # (N_mols, 128)

# 方案B：使用 ScorePosNet3D.fetch_embedding() 方法（已存在于模型中）
# 在采样完成后，用最终位置做一次前向传播提取嵌入
preds = model.fetch_embedding(protein_pos, protein_v, batch_protein,
                              final_ligand_pos, final_ligand_v, batch_ligand)
```

**关键代码位置**：
- `external/targetdiff/models/molopt_score_model.py:351` → `final_ligand_h`
- `external/targetdiff/models/molopt_score_model.py:620-631` → `fetch_embedding()`
- `external/targetdiff/scripts/sample_diffusion.py:72-116` → 需添加第8个返回值
- `bayesdiff/sampler.py:275-284` → 已支持8返回值解析

**预期**：这是**唯一能同时捕获3D几何 + 蛋白-配体交互 + 口袋特异性**的表征。如果 TargetDiff 的扩散模型学到了有意义的蛋白-配体交互模式，这些嵌入应该与结合亲和力高度相关。

---

### 13.4 P1: Vina 对接打分作为特征

**原理**：AutoDock Vina 的打分函数直接建模蛋白-配体结合能，包含：
- 高斯距离项（范德华）
- 排斥项
- 氢键项
- 疏水接触
- 旋转熵罚分

**现状**：TargetDiff 代码中已有 `VinaDockingTask` 集成（`evaluate_diffusion.py:107-124`），可直接调用。

**实现**：
```python
# 对每个生成的分子计算 Vina score
from utils.evaluation.docking_vina import VinaDockingTask
task = VinaDockingTask.from_generated_mol(mol, pocket_pdb, pos)
vina_score = task.run_sync()  # kcal/mol
# 作为特征或直接作为 pKd 预估
```

**优势**：物理驱动的打分函数，直接衡量结合强度，无需训练
**劣势**：Vina 本身精度有限（RMSE ~2.0 kcal/mol）；需要 3D 对接构象

---

### 13.5 P2: 蛋白-配体交互指纹 (PLIF)

**原理**：编码蛋白-配体之间的具体交互类型，而非分子本身的结构。

**交互类型**：
- 氢键供体/受体
- π-π堆积
- π-阳离子作用
- 疏水接触
- 盐桥
- 金属配位

**实现**（使用 ProLIF 或 RDKit）：
```python
import prolif
# 从 SDF（配体）+ PDB（蛋白口袋）计算 PLIF
fp = prolif.Fingerprint(interactions=["HBDonor", "HBAcceptor", "PiStacking",
                                       "Hydrophobic", "SaltBridge", "CationPi"])
fp.run(ligand_mol, protein_mol)
# → 二进制向量，每位 = 一种残基-交互类型组合
```

**预期**：比 ECFP 好得多，因为直接编码结合界面信息。但依赖 3D 构象质量。

---

### 13.6 P3: 多表征融合

**原理**：组合多种互补表征，让 GP 从不同信息源中学习。

**融合策略**：
```
Z_fused = [ TargetDiff_emb(128) ‖ FCFP4(128) ‖ Vina_score(1) ‖ QED(1) ‖ SA(1) ]
→ PCA → GP
```

或使用多核 GP：
```
K_total = w1 * K_RQ(Z_encoder) + w2 * K_tanimoto(Z_FCFP) + w3 * K_RBF(Z_vina)
```

**优势**：无需选择"最好"的表征，让模型自动加权
**实现**：GPyTorch 支持 `AdditiveKernel` 和 `ProductKernel` 的组合

---

### 13.7 P4: 端到端 GNN 回归

**原理**：直接在蛋白-配体复合物图上训练 GNN 预测 pKd，绕过手工特征。

**架构选择**：
- **SchNet**：连续卷积，旋转不变
- **DimeNet++**：方向信息编码
- **PaiNN**：等变消息传递
- **TorchMD-NET**：专为分子属性预测设计

**数据需求**：N=932 可能不够训练 GNN（通常需要 >5000）。但可以：
1. 在 PDBbind 上预训练（~19K 复合物）
2. Fine-tune 到我们的数据分布
3. 使用预训练的 GNN 作为特征提取器 → GP

**估计工作量**：1-2周，需要准备数据加载器、训练循环、超参搜索

---

### 13.8 P5: ChemBERTa / MolBERT 嵌入

**原理**：使用预训练语言模型将 SMILES 编码为密集向量。

**实现**：
```python
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
inputs = tokenizer(smiles, return_tensors="pt", padding=True)
embeddings = model(**inputs).last_hidden_state[:, 0, :]  # CLS token, 768-dim
```

**预期收益较低**：虽然比 ECFP 表达力强（连续空间、上下文感知），但仍然**不包含蛋白口袋信息**。可作为 P3 融合方案的一个组件。

---

### 13.9 推荐执行顺序

```
Phase 1 (立即)：P0 — 提取 TargetDiff 编码器嵌入
  ↓ 修改 sample_diffusion.py → 返回 final_ligand_h
  ↓ 重新运行 Tier3 采样（或仅做 forward pass 提取嵌入）
  ↓ 训练 GP → 评估

Phase 2 (如果 P0 效果有限)：P1 + P3 — Vina 打分 + 多表征融合
  ↓ 计算 Vina scores
  ↓ 融合 [encoder_emb ‖ vina ‖ FCFP]
  ↓ 多核 GP

Phase 3 (长期)：P4 — 端到端 GNN
  ↓ 需要更多数据或预训练
  ↓ 最高上限但最大投入
```

**关键判断标准**：如果 P0 (TargetDiff 嵌入) 的 LOOCV ρ > 0.3，说明3D交互信息是有效的，可以进一步优化。如果 ρ 仍然 < 0.15，说明 TargetDiff 的扩散模型本身没有学到亲和力相关的特征，需要转向 P4（端到端训练）。

---

## 14. Train / Validation / Test 详细分析

### 14.1 实验设置

- **数据集**：932 pockets，FCFP4-2048 指纹，StandardScaler 标准化
- **划分方案**：60/20/20 随机分层 (Train 559 / Val 186 / Test 187)
- **模型**：GP + RQ kernel，200 epochs，lr=0.1
- **稳定性验证**：30× 重复随机划分

### 14.2 单次划分结果

| 集合 | N | RMSE | MAE | Spearman ρ | p-value | R² | 预测范围 |
|------|---|------|-----|-----------|---------|-----|---------|
| **Train** | 559 | 1.25 | 0.99 | **0.983** | 0.0 | **0.626** | [5.03, 10.39] |
| **Val** | 186 | 2.00 | 1.60 | 0.176 | 0.016 | 0.017 | [6.50, 7.24] |
| **Test** | 187 | 2.25 | 1.83 | 0.082 | 0.266 | 0.012 | [6.41, 7.24] |

### 14.3 30× 重复划分结果 (mean ± std)

| 集合 | RMSE | Spearman ρ | R² |
|------|------|-----------|-----|
| **Train** | 1.270 ± 0.036 | **0.989 ± 0.003** | **0.630 ± 0.008** |
| **Val** | 2.016 ± 0.092 | 0.144 ± 0.060 | 0.007 ± 0.011 |
| **Test** | 2.093 ± 0.102 | 0.146 ± 0.048 | 0.012 ± 0.012 |

### 14.4 关键诊断发现

#### 1) 严重过拟合

这是最显著的问题。从图表可以清晰看到：

- **训练集**：R² = 0.63, ρ = 0.98 → 模型"记住"了训练数据
- **验证/测试集**：R² ≈ 0.01, ρ ≈ 0.14 → 几乎没有泛化能力
- **RMSE 差距**：Train 1.25 vs Val/Test 2.0-2.25 → 近2倍差距

#### 2) 预测范围坍缩

这是过拟合的直接后果：

- **训练集预测范围**：[5.03, 10.39] — 覆盖了 5.36 个 pKd 单位
- **验证集预测范围**：[6.50, 7.24] — 仅 0.74 个 pKd 单位
- **测试集预测范围**：[6.41, 7.24] — 仅 0.83 个 pKd 单位
- **真实 pKd 范围**：[1.28, 15.22] — 跨 13.94 个单位

对于未见过的数据，模型退化为**常数预测器**（≈ 均值 7.08），只在训练数据附近有局部适应能力。

#### 3) 分布分析

从分布图可见：
- **Pred test 分布**呈极窄尖峰（集中在 7.0 附近），而 True test 分布是宽的正态形
- **残差分布**：Train σ=1.25, Val σ=2.00, Test σ=2.24 — 逐级增大
- 训练集残差紧凑（过拟合特征），验证/测试集残差宽泛且对称

#### 4) 不确定性的分裂行为

从 confidence 图可见两个明显的聚类：
- **训练集**：Pred σ ≈ 1.82-1.87（较低不确定性），预测范围较宽
- **验证/测试集**：Pred σ ≈ 1.95-2.05（较高不确定性），预测范围极窄

GP 正确地对未见数据报告了更高的不确定性，但不确定性差异不够大（1.85 vs 2.00），说明 kernel 过度拟合了训练点的噪声。

### 14.5 过拟合根因分析

| 因素 | 分析 |
|------|------|
| **表征维度 vs 样本量** | 2048-dim FCFP4，N=559 训练样本 → 维度远超样本量 |
| **FCFP4 无信号** | 指纹不编码结合亲和力信息，GP 只能拟合噪声 |
| **RQ kernel 灵活性** | RQ kernel 有额外的 α 参数，比 RBF 更灵活 → 更容易过拟合 |
| **无正则化** | 仅靠 noise lower bound (0.001) 做正则，不够强 |

**核心问题不变**：FCFP4 指纹与 pKd 之间**不存在可泛化的函数关系**。GP 在训练集上通过记忆达到了 R²=0.63，但这只是对噪声的插值，无法推广到新数据。

### 14.6 与 LOOCV 的对比

| 评估方式 | RMSE | ρ | R² | 过拟合风险 |
|---------|------|---|-----|----------|
| LOOCV (§12.3) | 2.068 | 0.111 | 0.013 | 最低（每次只留1个样本） |
| 5-Fold CV (§12.3) | 2.067 | 0.117 | 0.014 | 低 |
| 60/20/20 Val | 2.00 | 0.176 | 0.017 | 中 |
| 60/20/20 Test | 2.25 | 0.082 | 0.012 | 中 |
| 30× Repeated Val | 2.02±0.09 | 0.144±0.060 | 0.007±0.011 | 最可靠 |
| 30× Repeated Test | 2.09±0.10 | 0.146±0.048 | 0.012±0.012 | 最可靠 |

所有泛化指标一致：**RMSE ≈ 2.0-2.1, ρ ≈ 0.11-0.15, R² ≈ 0.01**。训练集 R²=0.63 是完全虚假的过拟合指标。

### 14.7 结论

Train/Val/Test 分析进一步确认了 §12.6 的结论：

1. ✅ **GP 模型能力充足** — 在训练集上 R²=0.63 证明 GP+RQ 有足够的表达力
2. ❌ **FCFP4 指纹无泛化信号** — 验证/测试 R²≈0.01 证明输入表征缺乏信息
3. ❌ **严重过拟合** — 训练 vs 泛化的巨大差距是典型的"拟合噪声"模式
4. 🎯 **下一步必须改变表征** — 参见 §13 优化方案，优先实施 P0（TargetDiff 编码器嵌入）

### 14.8 新增可视化

| 图表 | 路径 | 描述 |
|------|------|------|
| `train_val_test_scatter.png` | 三面板散点图：Train/Val/Test 各自的预测 vs 真实 |
| `training_and_metrics.png` | 训练曲线 + 三个集合的指标柱状图 |
| `repeated_split_boxplots.png` | 30× 重复划分的 RMSE/ρ/R² 箱线图 |
| `distribution_analysis.png` | 真实 vs 预测分布 + 残差分布 |
| `cross_split_comparison.png` | 全集合叠加散点 + 预测置信度分析 |

---

## 15. P0 实施结果：TargetDiff 编码器嵌入 (Encoder Embeddings)

### 15.1 方法概述

**核心思想**：利用 TargetDiff 模型内部的 SE(3)-equivariant graph attention encoder 提取蛋白-配体交互特征，替代 FCFP4 指纹。

**实现方式**（方案 B，无需重新采样）：
1. 加载已生成的分子 SDF 文件（原子坐标 + 原子类型）
2. 加载对应的蛋白口袋数据（.pt 文件或 PDB）
3. 将原子类型映射回 TargetDiff 编码（MAP_ATOM_TYPE_AROMATIC_TO_INDEX，13类）
4. 对蛋白质中心化位置（匹配训练约定）
5. 单次 forward pass（`fix_x=True`），提取 `final_ligand_h`（128维/原子）
6. `scatter_mean` 聚合为分子级嵌入 → 每个口袋取分子均值 → 128维/口袋

**关键技术细节**：
- `time_emb_dim=0`（训练配置），forward pass 不需要 time_step
- 每个口袋仅需 1 次 forward pass（非 1000 步扩散），总计 ~1 分钟完成 942 个口袋
- 使用 `model(fix_x=True)` → 位置不更新，纯编码器评估

**代码**：`scripts/studies/embedding_encoder_only.py`

### 15.2 数据集

| 指标 | 值 |
|------|-----|
| 处理口袋数 | 942 (77 个 SDF 为空被跳过) |
| 嵌入维度 | 128 |
| 每口袋分子数 | 5.5 ± 4.0 |
| 嵌入统计 | mean=0.036, std=0.523 (非零！实际信号) |
| pKd 范围 | [1.28, 15.22] |

### 15.3 核心结果：Encoder-128 vs FCFP4-2048

#### LOOCV 对比

| 指标 | Encoder-128 | FCFP4-2048 | Δ | 提升倍数 |
|------|-------------|------------|---|---------|
| **RMSE** | **1.949** | 2.068 | -0.118 | 5.7% ↓ |
| **Spearman ρ** | **0.369** | 0.111 | +0.258 | **3.3×** |
| **R²** | **0.120** | 0.013 | +0.107 | **9.2×** |
| p-value | ≈0 | 0.0007 | — | — |

#### 5-Fold CV

| Fold | RMSE | Spearman ρ |
|------|------|-----------|
| 1 | 1.963 | 0.322 |
| 2 | 1.846 | 0.378 |
| 3 | 2.011 | 0.438 |
| 4 | 1.804 | 0.287 |
| 5 | 2.165 | 0.352 |
| **Overall** | **1.962** | **0.354** |

#### 50× 重复随机划分 (70/30)

| 指标 | Encoder-128 | FCFP4-2048 |
|------|-------------|------------|
| RMSE | 1.957 ± 0.071 | 2.078 ± 0.099 |
| Spearman ρ | **0.373 ± 0.045** | 0.134 ± 0.055 |
| R² | **0.116 ± 0.027** | 0.004 ± 0.010 |

#### 30× Train/Val/Test (60/20/20)

| 集合 | RMSE | Spearman ρ | R² |
|------|------|-----------|-----|
| **Train** | 1.335 ± 0.044 | 0.891 ± 0.009 | 0.591 ± 0.017 |
| **Val** | 1.957 ± 0.106 | **0.334 ± 0.052** | **0.098 ± 0.033** |
| **Test** | 1.935 ± 0.110 | **0.357 ± 0.053** | **0.109 ± 0.039** |

### 15.4 关键发现

#### ✅ Go/No-Go 判定：**GO** (LOOCV ρ = 0.369 > 0.3)

TargetDiff 编码器嵌入通过了 §13.5 中定义的 go/no-go 标准。

#### 1) 3D 交互信息确实有效

Encoder-128 的 LOOCV ρ=0.369 比 FCFP4-2048 的 ρ=0.111 高出 **3.3 倍**。这证明：
- TargetDiff 的扩散模型学到了有意义的蛋白-配体交互模式
- 128维编码器嵌入捕获了 FCFP 指纹完全缺失的口袋特异性信息
- SE(3)-equivariant 架构的空间距离编码对亲和力预测有贡献

#### 2) 过拟合显著改善

| 指标 | Encoder-128 | FCFP4-2048 |
|------|-------------|------------|
| Train R² | 0.591 | 0.630 |
| Test R² | **0.109** | 0.012 |
| Train-Test 差距 | 0.482 | 0.618 |
| Test ρ | **0.357** | 0.146 |

Encoder 的泛化能力远优于 FCFP4：
- Test ρ 从 0.146 提升到 0.357（2.4×）
- Train-Test R² 差距从 0.618 缩小到 0.482
- 模型不再退化为常数预测器

#### 3) 但仍有优化空间

尽管显著进步，R²=0.12 仍然意味着模型只解释了 12% 的方差。可能的瓶颈：
- 聚合方式：当前使用 `scatter_mean`（原子级→分子级），可能丢失关键信息
- 嵌入来源：使用 final (t=0) 状态的编码，但中间扩散步骤可能包含更丰富信息
- 分子质量：每口袋平均只 5.5 个分子，且由 TargetDiff 生成（非真实配体）
- 蛋白表征：27维 FeaturizeProteinAtom 可能不够丰富

### 15.5 与 FCFP4 的训练曲线对比

| 模型 | 初始 NLL | 最终 NLL | 收敛速度 |
|------|---------|---------|---------|
| Encoder-128 | — | 2.108 | 较快 |
| FCFP4-2048 | — | 2.171 | 较慢 |

Encoder 模型收敛到更低的负对数边际似然，说明 128 维编码器嵌入比 2048 维 FCFP 指纹更高效地编码了与 pKd 相关的信息。

### 15.6 下一步优化方向

基于 P0 的成功（GO），建议的优化路径：

1. **P0+：改进嵌入聚合** — 尝试 max-pooling、attention-weighted aggregation 替代 mean
2. **P0++：多层嵌入融合** — 使用 UniTransformer 所有 9 层的隐藏状态而非只用最后一层
3. **P3 (多表征融合)**：将 Encoder-128 + FCFP4-2048 + Vina scores 拼接，训练 GP
4. **增加生成分子数**：从 ~5.5 提升到 20-50 个分子/口袋，提高嵌入稳定性
5. **使用真实配体**：用 CrossDocked 中的真实配体坐标（而非生成分子）提取编码器嵌入

### 15.7 可视化

| 图表 | 描述 |
|------|------|
| `encoder_vs_fcfp_training.png` | 训练曲线对比：Encoder vs FCFP4 |
| `encoder_loocv_scatter.png` | LOOCV 散点图：预测 vs 真实 |
| `encoder_vs_fcfp_metrics.png` | LOOCV 指标柱状图对比 |
| `encoder_tvt_boxplots.png` | 30× Train/Val/Test 箱线图 |
| `encoder_pca.png` | Encoder 嵌入 PCA 投影（按 pKd 着色） |
| `encoder_pca_variance.png` | PCA 方差解释曲线 |

## 16. P0+: 聚合策略对比 (Aggregation Strategy Comparison)

### 16.1 动机

P0 使用 `scatter_mean` 将每个口袋内所有分子的 128 维编码器嵌入聚合为单一口袋表征。
但 mean 聚合可能丢失分布信息（如离群分子、嵌入方差）。本阶段系统对比 7 种聚合策略，
寻找更优的分子→口袋表征方法。

### 16.2 聚合策略

| 策略 | 维度 | 描述 |
|------|------|------|
| `mean` | 128 | 基线：逐特征取均值 |
| `max` | 128 | 逐特征取最大值 |
| `mean+max` | 256 | 拼接 mean 和 max |
| `mean+std` | 256 | 拼接 mean 和标准差 |
| `attn_norm` | 128 | 基于 L2 范数的 attention-weighted mean |
| `attn_pca` | 128 | 基于第一主成分的 attention-weighted mean |
| `trimmed_mean` | 128 | 去掉 10% 极值后取均值 |

### 16.3 技术修复

- **数值溢出**：`attn_norm` 和 `attn_pca` 中 `np.exp(norms / temperature)` 溢出
  - 修复：`logits = logits - logits.max()` 再取 exp（log-sum-exp trick）
- **退化特征检测**：添加 NaN / 全零特征检查，自动跳过无效策略
- 修复后所有 7 策略均生成有效特征（128 或 256 维，non-zero）

### 16.4 完整结果（Job 5031796 — 7-GPU 并行 array job，~2 分钟完成）

> Run: 7-task array job 5031796 on L40S (gl010/gl013/gl026/gl031/gl043/gl034/gl014)
> Merge: Job 5031813, 22 seconds
> 优化：使用 analytic LOOCV（单次训练 + K_inv 解析公式），比暴力 LOOCV 快 ~942×

#### LOOCV (Analytic)

| 策略 | 维度 | RMSE | Spearman ρ | R² |
|------|:----:|:----:|:----------:|:---:|
| **attn_norm** | 128 | **1.947** | **0.373** | **0.122** |
| mean | 128 | 1.950 | 0.367 | 0.119 |
| mean+max | 256 | 1.956 | 0.362 | 0.114 |
| trimmed_mean | 128 | 1.954 | 0.362 | 0.116 |
| attn_pca | 128 | 1.957 | 0.357 | 0.113 |
| max | 128 | 1.962 | 0.353 | 0.108 |
| mean+std | 256 | 1.964 | 0.348 | 0.106 |
| *FCFP4-2048* | *2048* | *2.068* | *0.111* | *0.013* |

#### 50× Repeated Random Splits (70/30)

| 策略 | RMSE (mean±std) | Spearman ρ (mean±std) |
|------|:---------------:|:---------------------:|
| **attn_norm** | 1.947±? | **0.375±0.045** |
| mean | 1.950±? | 0.373±0.045 |
| mean+max | 1.956±? | 0.370±0.044 |
| trimmed_mean | 1.954±? | 0.371±0.046 |
| attn_pca | 1.957±? | 0.362±0.046 |
| max | 1.962±? | 0.361±0.043 |
| mean+std | 1.964±? | 0.355±0.044 |

#### 30× Train/Val/Test (60/20/20) — Top 3 策略

| 策略 | Train ρ | Val ρ | Test ρ |
|------|:-------:|:-----:|:------:|
| **attn_norm** | — | — | **0.369±0.053** |
| mean | — | — | 0.357±0.053 |
| mean+max | — | — | 0.354±0.053 |

### 16.5 关键发现

1. **attn_norm 是最优策略**（LOOCV ρ=0.373），但仅比 mean baseline 提升 **Δρ = +0.006**
2. **所有 7 种聚合策略差异极小**（ρ 范围 0.348–0.373），说明**聚合方式不是瓶颈**
3. **高维拼接（256d）反而不如低维（128d）**：mean+max (256d, ρ=0.362) < mean (128d, ρ=0.367)，信号密度比维度更重要
4. **max pooling（ρ=0.353）最差之一**：最大值聚合丢失了均值中的平均化降噪效果
5. **所有编码器策略都远优于 FCFP4**（ρ ≈ 0.35–0.37 vs 0.11），确认 3D 编码器嵌入的价值
6. **50× splits 与 LOOCV 高度一致**：所有策略的排序在两种评估协议下完全相同，结果稳健

### 16.6 结论

聚合策略的优化空间已穷尽。attn_norm 的边际提升（Δρ=+0.006）在统计上不显著（50× splits 的标准差 ~0.045）。**瓶颈不在聚合层，而在上游表征**——即编码器嵌入本身的信息含量和生成分子的多样性/质量。

后续应聚焦于：
- P0++：多层嵌入融合（利用 UniTransformer 所有 9 层隐藏状态）
- P3：多表征融合（Encoder + FCFP + Vina score）
- 增加每口袋生成分子数（从 ~5.5 提升到 20-50）

### 16.7 脚本与产出

- 训练脚本：`scripts/studies/gp_aggregation.py`（支持 `--strategy-index` 并行 + `--merge` 汇总）
- SLURM array job：`slurm/train_gp_aggregation.sh`（7 任务并行）
- SLURM merge job：`slurm/merge_gp_aggregation.sh`（依赖 array job 完成后运行）
- 结果文件：`results/tier3_gp/p0plus_aggregation_results.json`
- 可视化：
  - `results/tier3_gp/figures/p0plus_aggregation_loocv.png` — LOOCV 指标柱状图
  - `results/tier3_gp/figures/p0plus_aggregation_splits.png` — 50× splits 稳健性对比
  - `results/tier3_gp/figures/p0plus_top3_tvt.png` — Top 3 策略 Train/Val/Test 泛化对比

## 17. 综合训练曲线与模型对比可视化 (FCFP4-2048 vs Encoder-128)

> Run: SLURM Job 5038079, `cpu_short` partition, ~2 min  
> Script: `scripts/studies/tier3_training_curves.py`  
> Output: `results/tier3_gp/figures/tier3_analysis/` (7 PNG + 1 JSON)

### 17.1 实验设计

同时训练 FCFP4-2048（N=932）和 Encoder-128（N=942）两个 GP 模型，在 **同一 60/20/20 随机划分** 下记录逐 epoch 的所有指标，生成 7 张综合可视化图。

| 参数 | 值 |
|------|-----|
| 划分 | 60/20/20 随机，相同 seed |
| Epochs | 200 |
| LR | 0.1 |
| Kernel | Rational Quadratic (RQ) |
| 重复实验 | 10× independent random splits |

### 17.2 单次划分结果

| 指标 | FCFP4 Train | FCFP4 Val | FCFP4 Test | Encoder Train | Encoder Val | Encoder Test |
|------|:-----------:|:---------:|:----------:|:-------------:|:-----------:|:------------:|
| **RMSE** | 1.247 | 2.003 | 2.245 | 1.277 | 1.802 | 2.167 |
| **Spearman ρ** | 0.983 | 0.176 | 0.082 | 0.903 | 0.287 | 0.347 |
| **R²** | 0.626 | 0.017 | 0.012 | 0.621 | 0.052 | 0.098 |

### 17.3 10× 重复划分稳定性 (mean ± std)

| 模型 | Train RMSE | Val RMSE | Test RMSE | Train ρ | Val ρ | Test ρ | Train R² | Val R² | Test R² |
|------|:----------:|:--------:|:---------:|:-------:|:-----:|:------:|:--------:|:------:|:-------:|
| **FCFP4-2048** | 1.270±0.033 | 2.008±0.085 | 2.107±0.105 | 0.988±0.005 | 0.126±0.075 | 0.154±0.061 | 0.630±0.008 | 0.006±0.014 | 0.006±0.016 |
| **Encoder-128** | 1.341±0.042 | 1.951±0.109 | 1.983±0.112 | 0.887±0.006 | 0.339±0.066 | 0.352±0.068 | 0.583±0.016 | 0.095±0.040 | 0.103±0.039 |

### 17.4 关键发现

#### 1) 训练曲线行为差异（Fig 01）

- **NLL 收敛**：FCFP4 初始 NLL 更高（~13 vs ~5.5），但两者最终收敛到相近值（~2.1-2.2）
- **噪声方差**：FCFP4 的学习噪声稳定在较低值，Encoder 的噪声方差持续上升——说明 Encoder 模型从数据中提取了更多信号（信号/噪声分离更好）
- **输出缩放**：两者的 output scale 都随训练增大，但 Encoder 增速更平稳

#### 2) 学习曲线揭示泛化差距（Fig 02）

- **FCFP4**：Train RMSE 快速降至 ~1.1，但 Val/Test RMSE 始终停留在 ~2.0-2.3；Train ρ→1.0 但 Val/Test ρ 停留在 ~0.1-0.2
- **Encoder**：Train RMSE→1.3，Val→1.8，Test→2.17；Train ρ→0.9，Val ρ→0.28，Test ρ→0.35
- **Encoder 的泛化曲线明显高于 FCFP4**，且更早趋于稳定

#### 3) 数据分布分析（Fig 03）

- **pKd 分布**：近正态，μ=7.08，范围 [1.28, 15.22]
- **PCA 方差解释**：FCFP4 PC1 仅解释 0.6% 方差（极度分散的二进制高维空间），Encoder PC1 解释 36.4%（更有结构的低维流形）
- **嵌入范数与 pKd 弱相关**：Encoder 嵌入的 L2 范数与 pKd 的 Spearman ρ=0.062（几乎不相关），说明亲和力信息编码在方向而非幅度上

#### 4) 预测散点对比（Fig 04）

- **FCFP4**：Train 散点沿对角线排列良好，但 Val/Test 预测塌缩为水平窄带 [6.4, 7.2]——本质是常数预测
- **Encoder**：Train 散点类似，Val/Test 散点虽扩散但保持了正相关趋势，预测范围更宽

#### 5) 过拟合差距分析（Fig 07）

- **FCFP4 过拟合差距** (train ρ − test ρ)：从初始 ~0.93 先降至 ~0.78（epoch 20），然后持续升至 ~0.90（过拟合加剧）
- **Encoder 过拟合差距**：从 ~0.67 快速降至 ~0.55 并稳定——过拟合程度显著低于 FCFP4

#### 6) 不确定性校准（Fig 06）

- 两个模型的 predicted σ 都紧密聚集（FCFP4: 1.82–1.87, Encoder: 1.80–1.87）
- 校准质量：大量测试点的真实残差超过预测 σ，两者都偏过度自信
- Encoder 的 true vs predicted 分布匹配略优于 FCFP4

### 17.5 结论

本次综合可视化以训练曲线的方式直观呈现了 §14-§15 中的定量结论：

1. **Encoder-128 在所有泛化指标上一致优于 FCFP4-2048**（Test ρ: 0.347 vs 0.082, 4.2× 提升）
2. **FCFP4 的过拟合更严重**——训练 ρ=0.98 完全是记忆训练集，在新数据上退化为常数预测器
3. **Encoder 的优势在 10× 重复划分中稳健**——Val/Test ρ 的分布完全不重叠（Encoder ~0.35±0.07 vs FCFP4 ~0.15±0.07）
4. **瓶颈仍在表征质量**——即使 Encoder 也只解释了 ~10% 方差（R²≈0.10），需要更丰富的蛋白-配体交互特征

### 17.6 可视化清单

| 图号 | 文件 | 描述 |
|:----:|------|------|
| 01 | `01_training_curves.png` | NLL loss、噪声方差、输出缩放收敛曲线 |
| 02 | `02_learning_curves.png` | 逐 epoch RMSE/ρ/R² 在 train/val/test 的变化（2×3 面板） |
| 03 | `03_data_distribution.png` | pKd 分布、划分箱线图、分子数、PCA、方差解释、嵌入范数（9 面板） |
| 04 | `04_scatter_tvt.png` | Train/Val/Test 预测 vs 真实散点（2×3 面板，颜色=predicted σ） |
| 05 | `05_model_comparison.png` | 指标柱状图 + 10× 重复划分箱线图（2×3 面板） |
| 06 | `06_residual_calibration.png` | 残差分布、真实/预测密度、不确定性校准（2×3 面板） |
| 07 | `07_summary.png` | 过拟合差距曲线 + 综合指标汇总表 |

---

## 18. 正则化实验：Dropout + Weight Decay + Deep Kernel Learning

> 日期：2026-03-27  
> 脚本：`scripts/studies/regularization_study.py`  
> SLURM：`slurm/regularization_study.sh`，Job 5039777，`cpu_short`  
> 输出：`results/tier3_gp/figures/regularization/`（5 PNGs + 1 JSON）

### 18.1 实验动机

前述实验（§17）显示 Encoder-128 GP 存在显著过拟合（Train ρ=0.90 vs Test ρ=0.35，gap=0.55）。本实验尝试通过 Dropout 和 Weight Decay 减轻过拟合。

### 18.2 技术方案

由于当前模型为纯 ExactGP + RQ kernel（无 MLP 层），传统 Dropout/Weight Decay 无法直接应用。因此引入 **Deep Kernel Learning (DKL)** 方案：在 Encoder-128 嵌入与 GP kernel 之间插入 MLP 特征提取器（128→64→32），使得 Dropout 和 Weight Decay 有作用目标。

| 组件 | 实现 |
|------|------|
| MLP 结构 | Linear(128→64) → ReLU → Dropout → Linear(64→32) → ReLU → Dropout |
| GP kernel | ScaleKernel(RQKernel(ard_num_dims=32)) |
| 优化器 | AdamW（支持 weight_decay） |
| 训练 epoch | 200 |

### 18.3 实验配置

| # | 配置 | 学习率 | Dropout | Weight Decay |
|:-:|------|:------:|:-------:|:------------:|
| 1 | Baseline GP（纯 GP，无 MLP） | 0.1 | — | 0 |
| 2 | DKL (no reg) | 0.01 | 0 | 0 |
| 3 | DKL + Dropout 0.3 | 0.01 | 0.3 | 0 |
| 4 | DKL + Dropout 0.5 | 0.01 | 0.5 | 0 |
| 5 | DKL + WD 0.01 | 0.01 | 0 | 0.01 |
| 6 | DKL + D0.3 + WD | 0.01 | 0.3 | 0.01 |
| 7 | GP + NoisePrior | 0.1 | — | 0 |

### 18.4 结果

#### 单次划分（60/20/20）

| 配置 | Train ρ | Test ρ | Test R² | Train-Test Gap (ρ) |
|------|:-------:|:------:|:-------:|:-------------------:|
| **Baseline GP** | 0.903 | **0.347** | **0.098** | 0.556 |
| **GP + NoisePrior** | 0.902 | **0.347** | **0.098** | **0.555** |
| DKL + WD 0.01 | 0.998 | 0.290 | −0.175 | 0.708 |
| DKL + Dropout 0.3 | 0.976 | 0.288 | −0.221 | 0.688 |
| DKL (no reg) | 0.998 | 0.274 | −0.180 | 0.724 |
| DKL + Dropout 0.5 | 0.953 | 0.251 | −0.193 | 0.702 |
| DKL + D0.3 + WD | 0.982 | 0.236 | −0.203 | 0.746 |

#### 10× 重复随机划分

| 配置 | Test ρ (mean±std) | Test R² (mean±std) |
|------|:-----------------:|:------------------:|
| **Baseline GP** | **0.352±0.068** | **0.103±0.039** |
| **GP + NoisePrior** | **0.352±0.069** | **0.103±0.039** |
| DKL + D0.3 + WD | 0.263±0.065 | −0.097±0.121 |
| DKL + Dropout 0.5 | 0.239±0.071 | −0.105±0.125 |
| DKL + Dropout 0.3 | 0.236±0.074 | −0.120±0.134 |
| DKL + WD 0.01 | 0.197±0.070 | −0.117±0.113 |
| DKL (no reg) | 0.189±0.045 | −0.121±0.090 |

### 18.5 分析与结论

1. **DKL 使过拟合更严重**：所有 DKL 变体的 overfitting gap 均大于 Baseline GP。MLP(128→64→32) 在 N=942 的小数据集上引入过多参数，即使加入 Dropout/WD 也无法补偿。
2. **Dropout 和 Weight Decay 效果有限**：Dropout=0.5 和 WD=0.01 的组合反而是最差配置之一（Test ρ=0.236），说明正则化手段无法在根本上解决 MLP 在小数据上的过拟合问题。
3. **GP 本身即为最优正则化器**：ExactGP 是贝叶斯非参数方法，kernel 隐式限制了函数空间复杂度，天然具有正则化效果。添加 MLP 破坏了这一优势。
4. **GP + NoisePrior 效果等同 Baseline**：对噪声参数施加 GammaPrior 和 noise lower bound 无显著影响。
5. **性能瓶颈在表征质量而非模型正则化**：Encoder-128 仅解释约 10% 方差（R²≈0.10），提升方向应为更丰富的嵌入（增大分子采样数、融合 FCFP4 + Encoder 等）。

### 18.6 可视化清单

| 图号 | 文件 | 描述 |
|:----:|------|------|
| 01 | `01_learning_curves.png` | 7 配置的 Train/Test RMSE/ρ/R² 逐 epoch 曲线（2×3 面板） |
| 02 | `02_model_comparison.png` | 指标柱状图 + 10× 重复划分箱线图 |
| 03 | `03_scatter_top4.png` | Top 4 配置的预测 vs 真实散点 |
| 04 | `04_overfitting_gap.png` | Overfitting gap（Test−Train）逐 epoch 变化（RMSE/ρ/R²） |
| 05 | `05_summary_table.png` | 综合指标汇总表（所有配置 × 所有指标） |

