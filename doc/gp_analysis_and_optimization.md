# GP Model Analysis & Optimization Plan

> Generated from `scripts/11_gp_training_analysis.py` run on 2026-03-26  
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
> Script: `scripts/12_robust_evaluation.py`  
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

**Deliverable**: A single script `scripts/12_robust_evaluation.py` that runs all three protocols and produces a comparison table.

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

**Deliverable**: `scripts/13_embedding_comparison.py` — extracts all embeddings, runs comparison, generates table + figures.

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
| **1** | Robust eval framework | `scripts/12_robust_evaluation.py` | Must measure before optimizing |
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
2. Run `scripts/02_sample_molecules.py --pocket_list data/splits/missing_pk_pockets.txt --num_steps 1000 --num_samples 64`
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
- Script exists: `scripts/01_prepare_data.py` — designed for PDBbind parsing but not used in current pipeline

**Implementation plan**:
1. Parse PDBbind v2020 INDEX file with existing `parse_pdbbind_index()`
2. Map PDBbind PDB codes → CrossDocked pocket families (by UniProt ID or pocket overlap)
3. For each new labeled pocket:
   - Check if structure exists in LMDB or test set
   - If yes → generate molecules → extract embeddings
   - If no → extract pocket from PDB file using `scripts/01_prepare_data.py`
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

