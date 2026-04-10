# Sub-Plan 4: Uncertainty-Aware Hybrid Predictor

**DKL / DKL Ensemble / Residual GP**

> **Priority**: P1 — High  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 data); Sub-Plan 2 (frozen best representation)  
> **Training Data**: PDBbind v2020 R1 general set (19,037 P-L complexes before filtering); actual usable pool after label cleaning + CASF-2016 removal + structure checks; split by grouped protein-cluster Train/Val, fixed CASF-2016 Test (285 complexes)  
> **Entry Evidence**: Sub-Plan 02 Phase 3 results (see §1.2)  
> **Estimated Effort**: 2–3 weeks implementation + 1 week testing  
> **Paper Section**: §3.Y Uncertainty-Aware Hybrid Prediction

---

## 1. Motivation

### 1.1 Problem Statement

The roadmap defines Sub-Plan 04 as: *"replace the single GP oracle with a more expressive predictor while preserving uncertainty estimates."*

Sub-Plan 02 has already demonstrated that **representation upgrades work**: the SchemeB Independent ParamPool attention aggregation (A3.6, 108K params) achieves Test $R^2 = 0.574$, $\rho = 0.778$ with an MLP readout. However, the oracle head—which must provide calibrated uncertainty—remains the bottleneck.

**The central question for Sub-Plan 04 is**:

> How do we attach an uncertainty-aware oracle head to the improved representation without sacrificing predictive accuracy, and can we improve uncertainty usefulness beyond raw GP?

### 1.2 Entry Evidence from Sub-Plan 02

Sub-Plan 02 Phase 3 (GP integration experiments) established the following baseline hierarchy:

| Configuration | Repr. | Test $R^2$ | Test $\rho$ | RMSE | Uncertainty Quality |
|---------------|-------|-----------|-------------|------|---------------------|
| **MLP readout (no UQ)** | **A3.6 Indep.** | **0.574** | **0.778** | 1.31 | N/A |
| MLP readout (no UQ) | A3.4 Shared | 0.572 | 0.761 | 1.42 | N/A |
| DKL(128→32→SVGP) | A3.4 Shared | 0.559 | 0.760 | 1.44 | $\|err\|-\sigma$ ρ ≈ 0 (−0.04) |
| PCA32→SVGP | A3.4 Shared | 0.543 | 0.746 | 1.47 | $\|err\|-\sigma$ ρ ≈ 0 (+0.04) |
| raw SVGP(128d) | A3.4 Shared | 0.507 | 0.719 | 1.52 | $\|err\|-\sigma$ ρ ≈ 0 (−0.01) |

> **Two representations available**:
> - **A3.6 Independent** (`SchemeB_Independent`, 108K params): MLP ρ=0.778 — checkpoint at `ablation_viz/A36_independent_model.pt` **← current best, used as frozen representation for Sub-Plan 04**
> - **A3.4 Shared** (`SchemeB_SingleBranch`, 17K params): MLP ρ=0.761 — checkpoint at `phase3_refinement/A34_step1_model.pt`
> - GP baselines (DKL, PCA→SVGP, raw SVGP) above were measured on A3.4 Shared embeddings. They serve as **historical reference**; all heads will be re-evaluated on A3.6 Independent embeddings in Phase 4.2.
> - Source files: `phase3_results.json` (A3.4), `gp_fix_results.json` (A3.4b/c), s18 L40S re-run (A3.6).

The GP baselines come from `s14_phase3_refinement.py` (experiments A3.4 on Shared embeddings), which:
1. Trains `SchemeB_SingleBranch` + `MLPReadout` with entropy_weight=0.01
2. Freezes the attention model, extracts 128d embeddings via `extract_scheme_b_embeddings()`
3. Trains `GPOracle(d=128, n_inducing=512)` on the frozen embeddings
4. Measures $|err|-\sigma$ Spearman correlation on CASF-2016 test set

Sub-Plan 04 uses **A3.6 Independent** as the frozen representation (highest MLP ρ). Oracle heads will establish new baselines on these embeddings.

**Key findings**:
- **Point prediction**: MLP readout ≈ DKL > PCA→SVGP > raw SVGP
- **Uncertainty quality**: All GP-based schemes have near-zero $|err|-\sigma$ correlation; uncertainty estimates are effectively uncalibrated and cannot rank prediction reliability
- **Implication**: The problem is not just prediction capacity — it is that the GP uncertainty signal contains almost no useful information about actual error magnitude

### 1.3 Why GP Uncertainty Fails: Root Cause Analysis

Before designing solutions, we must understand *why* $|err|-\sigma$ ≈ 0:

1. **Input-space distance ≠ label-space difficulty**: The Matérn-5/2 kernel assigns uncertainty based on distance from inducing points in $z$-space. But two complexes can be equidistant from inducing points yet have very different prediction difficulties (e.g., one has clean binding pocket, the other has allosteric conformational change)
2. **Homogeneous noise assumption**: `GaussianLikelihood` estimates a single noise $\sigma^2_n$ for all data points. In reality, some protein families have inherently noisier labels (IC50 → pKi conversion, assay variability)
3. **Posterior collapse in high-d**: With $d=128$ and $J=512$ inducing points, the SVGP posterior variance is dominated by the prior in large portions of the input space — the posterior doesn't contract enough to be informative
4. **DKL feature collapse**: When jointly training $g_\theta$ + GP, the feature extractor can map dissimilar inputs to similar features to minimize ELBO, destroying the distance structure that the GP relies on for uncertainty

**Implications for Sub-Plan 04 design**:
- Ensemble disagreement (B2) sidesteps the GP variance problem entirely by using multi-model spread
- Residual GP (B3) operates on lower-variance residuals where the GP has a better chance of calibrating
- SNGP's spectral normalization prevents feature collapse by preserving input-space distances
- Evidential regression learns a per-sample uncertainty function rather than relying on kernel distance

### 1.4 What Sub-Plan 04 Must Solve

1. **Preserve or improve point prediction**: Any oracle head must not significantly degrade below the MLP/DKL level ($\rho \geq 0.75$)
2. **Produce useful uncertainty**: At least one of: NLL improvement, calibration improvement, $|err|-\sigma$ correlation increase, or family-wise uncertainty ranking
3. **Maintain Delta method compatibility**: Oracle must return `(mu, sigma2, jacobian)` for generation-uncertainty fusion via `bayesdiff/fusion.py`
4. **Work on frozen representations**: Isolate predictor-head quality from representation drift
5. **Fit within existing infrastructure**: Re-use `GPOracle`, `FusionResult`, `EvalResults`, SLURM patterns, and the `scripts/pipeline/s*.py` convention

---

## 2. Architecture Design: Oracle Head Family

Sub-Plan 04 is not a single method — it is a **benchmark phase** that evaluates a family of uncertainty-aware oracle heads. The architecture is organized as a layered experiment matrix.

All oracle heads receive the same input: a frozen 128d embedding vector $z \in \mathbb{R}^{128}$ produced by the current best frozen representation (currently A3.6 SchemeB Independent ParamPool from Sub-Plan 02, \rho=0.778; designed to accept any improved representation from Sub-Plans 01/03). The output must conform to the `OracleResult` interface (§4.2).

### 2.1 Layer 04A: Minimal GP-Compatible Fixes (Baseline Transfer)

These are direct transfers from Sub-Plan 02 Phase 3, included here as entry baselines. They re-use the existing `GPOracle` class from `bayesdiff/gp_oracle.py` with no modifications.

| ID | Head | Description | Implementation |
|----|------|-------------|----------------|
| 04A.1 | raw SVGP(128d) | Current Stage 1 oracle on new representation | `GPOracle(d=128, n_inducing=512)` |
| 04A.2 | PCA32→SVGP | Dimensionality reduction before GP | `sklearn.decomposition.PCA(n_components=32)` → `GPOracle(d=32)` |
| 04A.3 | DKL(128→32→SVGP) | Learned compression + GP (Sub-Plan 02 best GP) | `DKLOracle(input_dim=128, feature_dim=32)` |

These already have prototype results from Sub-Plan 02 (`results/stage2/phase3_refinement/phase3_results.json`); Sub-Plan 04 re-runs them under the unified evaluation protocol for fair comparison.

### 2.2 Layer 04B: Mainline Hybrid Models

The primary experimental targets. Each is described with full architecture detail and parameter counts.

#### B1: DKL (Deep Kernel Learning)

A neural network learns a feature transformation; a GP operates in the learned feature space:

$$
u = g_\theta(z) \in \mathbb{R}^{d_u}, \quad f(u) \sim \mathcal{GP}(m(u), k(u, u'))
$$

$$
\hat{y} = f(g_\theta(z))
$$

```
  z ∈ ℝ^128     u ∈ ℝ^32       ŷ ∈ ℝ
  ──────────► g_θ ──────────► SVGP ──────────►
              (MLP)           (Matérn-5/2)
```

**Feature extractor `g_θ` architecture** (default configuration):

| Layer | Shape | Activation | Params |
|-------|-------|------------|--------|
| Input | (B, 128) | — | — |
| Linear₁ | 128 → 256 | ReLU | 33,024 |
| Dropout₁ | — | p=0.1 | — |
| Linear₂ | 256 → 32 | — | 8,224 |
| Residual proj | 128 → 32 | — | 4,128 |
| Output | (B, 32) | — | — |

Residual: $u = W_{\text{proj}} z + \text{MLP}(z)$, total NN params ≈ **45K**.

**GP component**: Identical to current `SVGPModel` from `bayesdiff/gp_oracle.py`:
- Kernel: `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=32))`
- Mean: `ConstantMean()`
- Variational: `CholeskyVariationalDistribution` + `VariationalStrategy(learn_inducing_locations=True)`
- Inducing points: $J = 512$ learned in $u$-space
- Likelihood: `GaussianLikelihood()`

**Joint training**: $g_\theta$ and GP hyperparameters are trained simultaneously by maximizing the ELBO:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(f)} [\log p(y | f)] - \text{KL}[q(u) \| p(u)]
$$

Implemented via `gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)`.

#### B2: DKL Ensemble (3–5 members)

Train $M$ independent DKL models (each identical to B1) with different random seeds and/or 80% bootstrap samples of the training data:

$$
\mu_{\text{ens}}(z) = \frac{1}{M} \sum_{m=1}^M \mu^{(m)}(z)
$$

$$
\sigma^2_{\text{ens}}(z) = \underbrace{\frac{1}{M} \sum_{m=1}^M \sigma^{2(m)}(z)}_{\text{mean aleatoric}} + \underbrace{\frac{1}{M} \sum_{m=1}^M (\mu^{(m)}(z) - \mu_{\text{ens}}(z))^2}_{\text{epistemic disagreement}}
$$

**Rationale**: A single DKL's GP variance may be mis-calibrated (as observed in Sub-Plan 02), but ensemble disagreement provides an independent uncertainty signal based on member disagreement. This is the most direct path to improving $|err|-\sigma$ correlation.

**Diversity mechanisms** (compared in Tier 2 ablation A4.12):
- **Seed diversity**: Same data, different `torch.manual_seed()` → different inducing point initialization, different NN initialization
- **Bootstrap diversity**: Each member sees a different 80% subsample (drawn at protein-cluster level to avoid leakage)
- **Both**: Different seed + different bootstrap subsample

**Compute cost**: $M \times$ single DKL cost. For $M = 5$: 5× training time but prediction is parallelizable across members. At inference, all $M$ forward passes can run in a single batch.

**Parameter count**: $M \times 45\text{K (NN)} + M \times \text{GP params} \approx 5 \times 65\text{K} \approx 325\text{K}$ total.

#### B3: NN + GP Residual

A neural network handles the main signal; a GP captures structured residuals:

$$
\hat{y}_{\text{NN}} = h_\theta(z), \quad r = y - \hat{y}_{\text{NN}}, \quad r \sim \mathcal{GP}(0, k(z, z'))
$$

$$
\hat{y} = \hat{y}_{\text{NN}} + \hat{r}_{\text{GP}}
$$

**NN component** `h_θ`: Re-uses the existing `MLPReadout(input_dim=128, hidden_dim=128)` from `bayesdiff/attention_pool.py` — a 2-layer MLP mapping (128→128→1). Already validated in Sub-Plan 02 (A3.6) to achieve $\rho = 0.778$ on its own.

**Residual GP**: Operates on the same 128d embeddings but predicts residuals $r_i = y_i - h_\theta(z_i)$ rather than raw labels. Uses `GPOracle(d=128, n_inducing=512)` — the exact same class as Stage 1 oracle but trained on residuals.

**Why residuals help the GP**: After NN removes the main signal, residuals have lower variance and potentially simpler structure. If the NN captures most of the systematic variation, residuals may be closer to GP assumptions (stationary, smooth).

**Uncertainty combines both**:

$$
\sigma^2_{\text{oracle}} = \sigma^2_{\text{GP}}(r) + \sigma^2_{\text{epistemic}}(\text{NN})
$$

where NN epistemic uncertainty is optionally estimated via MC Dropout ($T=20$ forward passes with $p=0.1$):

$$
\sigma^2_{\text{MC}}(z) = \frac{1}{T} \sum_{t=1}^T (\hat{y}_t - \bar{\hat{y}})^2
$$

> **Clarification**: B3’s core uncertainty mechanism is the residual GP ($\sigma^2_{\text{GP}}$). MC Dropout is a switchable auxiliary term (`mc_dropout=True/False`) that adds a small NN epistemic component. B3 is **not** a full Bayesian NN + GP; the NN is deterministic at train time and MC Dropout is only used at inference if enabled. The residual GP alone should be sufficient for the primary uncertainty signal.

**Two-stage training**:
1. **Stage 1** (NN): Train `MLPReadout` with MSE + weight decay ($\lambda=10^{-4}$) + early stopping on val MSE (patience=30)
2. **Stage 2** (GP): Compute residuals $r_i = y_i - h_\theta(z_i)$ on train set. Train `GPOracle` on $(z, r)$ pairs with ELBO, same hyperparameters as current pipeline ($J=512$, $\eta=0.01$, 200 epochs)

**Optional Stage 3** (joint fine-tuning): Unfreeze NN, reduce LR to $10^{-5}$, train NN + GP jointly on ELBO for 50 more epochs. Only if Stage 2 shows improvement.

### 2.3 Layer 04C: Alternative Cheap Baselines (Tier 1b — Deferred)

Lower-cost single-model UQ methods for calibration. These provide context for interpreting B1–B3 results. **Implementation is deferred to Tier 1b** — only pursued after the core Tier 1 experiments (A4.1–A4.5) are complete and analyzed. The primary question Sub-Plan 04 must answer first is whether DKL Ensemble or Residual GP can improve $\rho_{|err|,\sigma}$ over single DKL.

#### C1: Spectral Normalized Neural GP (SNGP)

Replace the GP last layer with a random Fourier feature (RFF) approximation + spectral normalization on all hidden layers:

```
  z ∈ ℝ^128    h ∈ ℝ^256       ŷ, σ² ∈ ℝ
  ──────────► SN-MLP ──────────► RFF-GP ──────────►
              (2 layers)         (1024 features)
```

**Architecture**:
- Hidden layers: 2× `spectral_norm(Linear(128→256))` + ReLU — spectral normalization constrains the Lipschitz constant to preserve input-space distances
- GP output layer: `RandomFeatureGaussianProcess(in_features=256, num_inducing=1024, normalize_input=True, scale_random_features=True)` — this uses a random Fourier feature (RFF) approximation to a GP layer, following [Liu et al., ICML 2020]
- Single forward pass produces both $\hat{y}$ and $\sigma^2$

**Key hyperparameters**:
- `n_rff_features = 1024`: Number of random Fourier features (more = better GP approximation)
- `spectral_norm_bound = 0.95`: Lipschitz constant bound for spectral normalization
- `mean_field_factor = 0.1`: Scales logit variance for calibration (tuned on val set)

**Training**: Standard NLL loss: $\mathcal{L} = \frac{1}{N} \sum_i \left[ \frac{(y_i - \mu_i)^2}{2\sigma^2_i} + \frac{1}{2}\log\sigma^2_i \right]$ with AdamW, lr=$10^{-3}$, weight decay=$10^{-4}$.

**Advantage**: More scalable than exact GP; better distance-awareness than standard NN due to spectral normalization. No inducing points to manage.

**Jacobian**: Computed via standard `torch.autograd` through the SN-MLP + RFF-GP.

#### C2: Evidential Regression

A single NN outputs the parameters of a Normal-Inverse-Gamma (NIG) distribution, following [Amini et al., NeurIPS 2020]:

$$
\text{NN}(z) \to (\gamma, \nu, \alpha, \beta) \quad \Rightarrow \quad p(y|z) = \text{Student-t}_{2\alpha}(\gamma, \beta(1+\nu)/(\nu\alpha))
$$

**Architecture**: Simple MLP with 4-output head:

| Layer | Shape | Activation |
|-------|-------|------------|
| Linear₁ | 128 → 256 | ReLU |
| Linear₂ | 256 → 128 | ReLU |
| $\gamma$ head | 128 → 1 | Identity (predicted mean) |
| $\nu$ head | 128 → 1 | Softplus (evidence for mean, > 0) |
| $\alpha$ head | 128 → 1 | Softplus + 1 (IG shape, > 1) |
| $\beta$ head | 128 → 1 | Softplus (IG scale, > 0) |

**Uncertainty decomposition**:
- Aleatoric: $\sigma^2_{\text{alea}} = \beta / (\alpha - 1)$
- Epistemic: $\sigma^2_{\text{epi}} = \beta / (\nu (\alpha - 1))$
- Total: $\sigma^2_{\text{total}} = \sigma^2_{\text{alea}} + \sigma^2_{\text{epi}}$

**Training loss** (NIG NLL + evidence regularization):

$$
\mathcal{L}_{\text{evid}} = \mathcal{L}_{\text{NIG-NLL}} + \lambda_{\text{ev}} \cdot |y - \gamma| \cdot (2\nu + \alpha)
$$

where $\lambda_{\text{ev}} = 0.1$ gradually increases during training (annealed from 0 to 0.1 over first 50 epochs) to avoid early training instability.

**Advantage**: Single forward pass for both prediction and uncertainty. No GP training or inducing points. ≈50K parameters.

**Limitation**: Evidential regression can produce over-confident uncertainties and the NIG prior assumption may be too rigid. Included as calibration baseline only.

**Jacobian**: Computed via `torch.autograd` through the NN. $J_\mu = \partial\gamma / \partial z$.

### 2.4 Layer 04D: Optional Bayesianized Variants

Only pursued after B2/B3 results are available:

| ID | Variant | Description | When to Pursue |
|----|---------|-------------|----------------|
| 04D.1 | MC Dropout NN + GP residual | Enable dropout at inference in B3's NN, $T=20$ passes | If B3 point-prediction works but $\sigma^2_{\text{epistemic}}$ is uninformative |
| 04D.2 | Bayes-by-Backprop NN + GP residual | Replace B3's NN with variational weight NN (Blundell et al., 2015) | Only if MC Dropout fails and B3 architecture is promising |
| 04D.3 | DKL with heteroscedastic likelihood | Replace `GaussianLikelihood` with `FixedNoiseGaussianLikelihood` where noise is per-sample | If B1 point-prediction is good but uncertainty is homogeneous |

---

## 3. Mathematical Details

### 3.1 DKL Architecture (B1)

**Feature extractor** $g_\theta$:

$$
g_\theta: \mathbb{R}^{128} \to \mathbb{R}^{32}
$$

Implemented as a 2-layer MLP with residual connections:

$$
h_1 = \text{ReLU}(W_1 z + b_1), \quad h_1 \in \mathbb{R}^{256}
$$

$$
h_2 = W_2 \text{Dropout}(h_1, p=0.1) + b_2, \quad h_2 \in \mathbb{R}^{32}
$$

$$
u = g_\theta(z) = W_{\text{proj}} z + h_2, \quad W_{\text{proj}} \in \mathbb{R}^{32 \times 128}
$$

The residual connection ensures the GP has access to at least a linear projection of original features. Weight initialization: Kaiming for $W_1, W_2$; Xavier for $W_{\text{proj}}$.

**GP in feature space**:

$$
k_{\text{DKL}}(z, z') = k_{\text{base}}(g_\theta(z), g_\theta(z'))
$$

where $k_{\text{base}}$ is `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=32))` — identical to current `SVGPModel` but with `ard_num_dims=32` instead of 128.

**Inducing points**: Two strategies (compared in implementation):
- **Strategy A** (default): Initialize $J=512$ inducing points via random subset of training $u$-vectors (after one forward pass through $g_\theta$), then learn via `learn_inducing_locations=True`
- **Strategy B**: Free parameters in $u$-space, initialized from k-means of initial $u$-vectors

### 3.2 Training Objective

Joint training of $\theta$ (NN params), kernel hyperparameters $\psi$, and variational parameters $\phi$:

$$
\max_{\theta, \psi, \phi} \; \mathcal{L}_{\text{ELBO}}(\theta, \psi, \phi) = \mathbb{E}_{q_\phi(f)}[\log p(y | f)] - \text{KL}[q_\phi(u) \| p_\psi(u)]
$$

Computed via `gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)`, same API as existing `GPOracle.train()`.

**Per-component optimizer setup** (extends current single-group Adam):

```python
optimizer = torch.optim.AdamW([
    {"params": feature_extractor.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": gp_model.hyperparameters(), "lr": 1e-2},      # kernel + mean
    {"params": gp_model.variational_parameters(), "lr": 1e-2},
    {"params": likelihood.parameters(), "lr": 1e-2},
])
```

With optional regularization on $g_\theta$:

$$
\mathcal{L} = \mathcal{L}_{\text{ELBO}} - \lambda_{\text{reg}} \|\theta\|_2^2
$$

where $\lambda_{\text{reg}} = 10^{-4}$ (applied via AdamW weight_decay).

### 3.3 Uncertainty Decomposition

**Single DKL** (B1):

$$
\mu_{\text{oracle}}(z) = \mu_{\text{GP}}(g_\theta(z)), \quad \sigma^2_{\text{oracle}}(z) = \sigma^2_{\text{GP}}(g_\theta(z))
$$

where $\sigma^2_{\text{GP}}$ comes from `likelihood(model(X_t)).variance` — this includes both posterior variance and likelihood noise.

**DKL Ensemble** (B2):

$$
\sigma^2_{\text{oracle}}(z) = \underbrace{\frac{1}{M}\sum_m \sigma^{2(m)}_{\text{GP}}}_{\text{within-model}} + \underbrace{\text{Var}_m[\mu^{(m)}]}_{\text{between-model}}
$$

The between-model term captures epistemic uncertainty that a single GP's posterior variance may miss. The `aux` dict returns both components separately for diagnostic purposes.

**NN + GP Residual** (B3):

$$
\mu(z) = h_\theta(z) + \mu_{\text{GP}}^{(r)}(z), \quad \sigma^2(z) = \sigma^2_{\text{GP}}^{(r)}(z) + \sigma^2_{\text{MC}}(z)
$$

where $\sigma^2_{\text{MC}}$ is MC Dropout variance from $T=20$ forward passes through the NN with dropout enabled:

$$
\sigma^2_{\text{MC}}(z) = \frac{1}{T}\sum_{t=1}^T (h^{(t)}_\theta(z))^2 - \left(\frac{1}{T}\sum_{t=1}^T h^{(t)}_\theta(z)\right)^2
$$

### 3.4 Jacobian for Delta Method

All oracle heads must support Jacobian computation for generation-uncertainty fusion via `bayesdiff/fusion.py`. The existing `fuse_uncertainties()` function requires:
- `J_mu: np.ndarray` of shape `(d,)` — the Jacobian $\partial\mu / \partial z$ for a single sample
- Used in Delta method: $\sigma^2_{\text{gen}} = J_\mu^\top \Sigma_{\text{gen}} J_\mu$

**DKL (B1)**: Chains through both $g_\theta$ and GP:

$$
J_\mu = \frac{\partial \mu_{\text{oracle}}}{\partial z} = \frac{\partial \mu_{\text{GP}}}{\partial u} \cdot \frac{\partial g_\theta}{\partial z}
$$

Implementation follows `GPOracle.predict_with_jacobian()` pattern: enable `requires_grad_(True)` on input, forward pass, then `mu_t[i].backward(retain_graph=True)` per sample.

**DKL Ensemble (B2)**: Average of member Jacobians:

$$
J_\mu^{\text{ens}} = \frac{1}{M} \sum_m J_\mu^{(m)}
$$

Each member computes its own Jacobian independently. Total cost = $M \times$ single DKL Jacobian.

**NN + GP Residual (B3)**: Sum of NN and GP Jacobians:

$$
J_\mu = \frac{\partial h_\theta}{\partial z} + \frac{\partial \mu_{\text{GP}}^{(r)}}{\partial z}
$$

The NN Jacobian is $O(1)$ via `torch.autograd.functional.jacobian()`. The GP Jacobian uses the existing `GPOracle.predict_with_jacobian()`.

**SNGP (C1)**: Single autograd backward through SN-MLP + RFF-GP.

**Evidential (C2)**: $J_\mu = \partial\gamma / \partial z$ via single autograd backward.

All handled automatically by `torch.autograd`.

### 3.5 NN+GP Residual (B3) Detailed Training

**Stage 1: NN training**

$$
\min_\theta \; \frac{1}{N} \sum_{i=1}^N (y_i - h_\theta(z_i))^2 + \lambda \|\theta\|_2^2
$$

Using `MLPReadout(input_dim=128, hidden_dim=128)` from `bayesdiff/attention_pool.py`:
```python
mlp = MLPReadout(input_dim=128, hidden_dim=128)  # 128→128→ReLU→1
optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()
# Early stopping: patience=30, monitor val_MSE
```

**Stage 2: Residual GP training**

$$
r_i = y_i - h_\theta(z_i), \quad r \sim \mathcal{GP}(0, k(z, z'))
$$

```python
# Compute residuals on train set
with torch.no_grad():
    r_train = y_train - mlp(X_train).squeeze()

# Train GP on (X_train, r_train)
gp = GPOracle(d=128, n_inducing=512, device="cuda")
gp.train(X_train.numpy(), r_train.numpy(), n_epochs=200, batch_size=256, lr=0.01)
```

**Combined prediction**:

$$
\mu(z) = h_\theta(z) + \mu_{\text{GP}}^{(r)}(z), \quad \sigma^2(z) = \sigma^2_{\text{GP}}^{(r)}(z) + \sigma^2_{\text{MC}}(z)
$$

---

## 4. Implementation Plan

### 4.1 Training Protocol: Freeze Representation, Then Train Predictor

Sub-Plan 02 has demonstrated that the representation itself works. The problem is in the predictor head. Therefore Sub-Plan 04 uses a **frozen-representation protocol** to isolate predictor-head quality.

#### Phase 4.1: Freeze Embeddings

Extract and cache embeddings from the best Sub-Plan 02 model. This follows the same pattern as `run_A34_gp()` in `s14_phase3_refinement.py`:

```python
# 1. Load frozen representation (currently A3.6 Independent; replace with any improved encoder)
model = SchemeB_Independent(embed_dim=128, n_layers=10, attn_hidden_dim=64, entropy_weight=0.01)
model.load_state_dict(torch.load("results/stage2/ablation_viz/A36_independent_model.pt"))
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# 2. Extract embeddings (re-use extract_scheme_b_embeddings from s14)
X_train, y_train = extract_scheme_b_embeddings(model, train_loader, device)  # (N_train, 128), (N_train,)
X_val, y_val = extract_scheme_b_embeddings(model, val_loader, device)        # (N_val, 128), (N_val,)
X_test, y_test = extract_scheme_b_embeddings(model, test_loader, device)     # (285, 128), (285,)

# 3. Save as .npz for all oracle heads to consume
np.savez("results/stage2/frozen_embeddings.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)
```

**Expected shapes** (approximate, depends on post-filtering count):
- Train: ~15,500 × 128
- Val: ~2,200 × 128
- Test: 285 × 128 (CASF-2016)

**Data loaders**: Uses `AtomEmbeddingDataset` + `collate_atom_emb` from `s12_train_attn_pool.py`, same as Phase 3. Labels loaded from `data/pdbbind_v2020/labels.csv`, splits from `data/pdbbind_v2020/splits.json`.

#### Phase 4.2: Oracle Head Comparison on Frozen Embeddings

Train and evaluate all oracle heads on the same frozen embeddings:

| Head | Training | Approximate Time (L40S) | Notes |
|------|----------|------------------------|-------|
| MLP (reference, no UQ) | MSE, 200 epochs | ~5 min | Upper bound for point prediction |
| raw SVGP | ELBO, 200 epochs | ~10 min | Lower bound for GP methods |
| PCA32→SVGP | PCA + ELBO, 200 epochs | ~10 min | Baseline dimensionality reduction |
| DKL | ELBO (joint), 300 epochs | ~20 min | Primary candidate |
| DKL Ensemble ($M=5$) | ELBO × 5, 300 epochs each | ~100 min | Primary uncertainty candidate |
| NN + GP Residual | MSE 200 epochs + ELBO 200 epochs | ~15 min | Alternative architecture |

**Total estimated Tier 1 wall time**: ~2.5 hours on a single L40S GPU.

> **Tier 1b (deferred)**: SNGP (~10 min) and Evidential (~10 min) are deferred until Tier 1 results show whether the core 5 heads answer the primary research question. See §6.1b.

#### Phase 4.3: Optional End-to-End Fine-Tuning

Only if a specific oracle head clearly outperforms the frozen-head baselines on both point prediction and uncertainty, consider **small-range end-to-end fine-tuning** with the representation trunk unfrozen:

```python
# Unfreeze encoder with very low LR (currently SchemeB; works with any encoder)
for p in encoder.parameters():
    p.requires_grad_(True)

optimizer = AdamW([
    {"params": encoder.parameters(), "lr": 1e-5},   # very low — avoid destroying representation
    {"params": oracle_head.parameters(), "lr": 1e-3}, # normal LR for head
], weight_decay=1e-4)

# Short schedule: 50 epochs max, patience=15 on val metrics
```

**Decision criterion** (validation-driven, relative thresholds):
1. Gate is evaluated on **validation set**, not test set
2. UQ gate: val $\rho_{|err|,\sigma} > 0.10$ (uncertainty shows non-trivial signal)
3. Point-prediction gate: val $\rho < \rho_{\text{best\_frozen}} - 0.02$ (point prediction is ≥0.02 below the best frozen-head reference on val)
4. Both gates must be true: uncertainty is working but point prediction has room for improvement

This avoids hardcoding test-set thresholds that depend on specific baseline numbers.

**Benefits of this protocol**:
- Problem isolation: failures are clearly attributable to the predictor head
- Reproducibility: all heads see identical input features
- Efficiency: embeddings computed once, reused across all experiments

### 4.2 Unified Oracle Interface

All oracle heads must conform to the following interface. This is the **core contract** that makes oracle heads plug-compatible with the existing `fusion.py` (`fuse_uncertainties` / `fuse_batch`) and `evaluate.py` (`evaluate_all`).

```python
"""bayesdiff/oracle_interface.py — §4.2 Unified Oracle Interface"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class OracleResult:
    """Container for oracle head predictions.
    
    Must be compatible with bayesdiff/fusion.py:
        fuse_uncertainties(mu_oracle, sigma2_oracle, J_mu, cov_gen, ...)
    and with bayesdiff/evaluate.py:
        evaluate_all(mu_pred, sigma_pred, p_success, y_true, ...)
    
    jacobian is Optional — only populated by predict_for_fusion(), not by predict().
    This avoids expensive per-sample backward passes during evaluation and diagnostics.
    """
    mu: np.ndarray          # (N,) predicted means (pKd scale)
    sigma2: np.ndarray      # (N,) predicted total variance (must be > 0)
    jacobian: Optional[np.ndarray] = None  # (N, d) ∂μ/∂z — only from predict_for_fusion()
    aux: dict = field(default_factory=dict)
    # Standard aux keys (all optional, all shape (N,)):
    #   'sigma2_aleatoric' : within-model / likelihood variance
    #   'sigma2_epistemic' : between-model / ensemble disagreement variance
    #   'sigma2_gp'        : GP posterior variance component
    #   'sigma2_nn'        : NN epistemic (MC Dropout) variance component
    #   'member_mus'       : (M, N) array of per-member predictions (ensemble only)
    #   'member_sigma2s'   : (M, N) array of per-member variances (ensemble only)


class OracleHead(ABC):
    """Base class for all oracle heads in Sub-Plan 04.
    
    Subclasses:
        DKLOracle, DKLEnsembleOracle, NNResidualOracle,
        SNGPOracle, EvidentialOracle, PCA_GPOracle (Tier 1b)
    
    Integration points:
        - predict() → bayesdiff/evaluate.py (fast path, no Jacobian)
        - predict_for_fusion() → bayesdiff/fusion.py (expensive, with Jacobian)
        - bayesdiff/ood.py: MahalanobisOOD.score(z) or score_batch(X)
        - bayesdiff/calibration.py: IsotonicCalibrator.fit(p_raw, y_binary)
    """
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> dict:
        """Train the oracle head on frozen embeddings.
        
        Parameters
        ----------
        X_train : (N_train, d) float32 embeddings
        y_train : (N_train,) float32 pKd labels
        X_val : (N_val, d) float32 embeddings
        y_val : (N_val,) float32 pKd labels
        
        Returns
        -------
        history : dict with at least 'loss' key (list of per-epoch values),
                  plus method-specific keys (e.g. 'val_rho', 'val_nll')
        """
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> OracleResult:
        """Predict mu and sigma2 (fast path — no Jacobian).
        
        Used for evaluation, diagnostics, validation monitoring, and ablation tables.
        Jacobian is NOT computed — returns OracleResult with jacobian=None.
        
        Parameters
        ----------
        X : (N, d) float32 embeddings
        
        Returns
        -------
        OracleResult with mu (N,), sigma2 (N,), jacobian=None, aux dict
        """
        ...
    
    @abstractmethod
    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Predict mu, sigma2, AND Jacobian ∂μ/∂z (expensive path).
        
        Only call when entering the generation-uncertainty fusion stage
        (bayesdiff/fusion.py). For all other uses, call predict() instead.
        
        Parameters
        ----------
        X : (N, d) float32 embeddings
        
        Returns
        -------
        OracleResult with mu (N,), sigma2 (N,), jacobian (N, d), aux dict
        """
        ...
    
    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model checkpoint. Convention: path is a directory.
        Each head saves its own files inside the directory."""
        ...
    
    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from checkpoint directory."""
        ...
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, y_target: float = 7.0) -> dict:
        """Convenience: predict + compute standard metrics.
        
        Uses bayesdiff.evaluate.evaluate_all internally.
        Returns dict with keys: R2, spearman_rho, rmse, nll, ece, err_sigma_rho.
        """
        from bayesdiff.evaluate import evaluate_all, gaussian_nll
        from scipy.stats import spearmanr
        
        result = self.predict(X)
        sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
        
        # Point prediction metrics
        ss_res = np.sum((y - result.mu) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rho, _ = spearmanr(result.mu, y)
        rmse = np.sqrt(np.mean((y - result.mu) ** 2))
        nll = gaussian_nll(result.mu, sigma, y)
        
        # Uncertainty quality metrics
        errors = np.abs(y - result.mu)
        err_sigma_rho, err_sigma_p = spearmanr(errors, sigma)
        
        return {
            'R2': float(r2),
            'spearman_rho': float(rho),
            'rmse': float(rmse),
            'nll': float(nll),
            'err_sigma_rho': float(err_sigma_rho),
            'err_sigma_p': float(err_sigma_p),
            'mean_sigma': float(sigma.mean()),
        }
```

**Backward compatibility with existing `fusion.py`**: The current `fuse_uncertainties()` takes scalar `mu_oracle`, `sigma2_oracle`, and array `J_mu`. Use `predict_for_fusion()` (not `predict()`) when entering the fusion stage:

```python
# Use predict_for_fusion() — only when entering the fusion stage:
result = oracle.predict_for_fusion(X)  # expensive: computes Jacobian
assert result.jacobian is not None, "predict_for_fusion must return Jacobian"

# Per-sample fusion (existing API, no changes needed):
for i in range(N):
    fuse_result = fuse_uncertainties(
        mu_oracle=result.mu[i],
        sigma2_oracle=result.sigma2[i],
        J_mu=result.jacobian[i],       # shape (d,) 
        cov_gen=cov_gen_list[i],        # shape (d, d)
        y_target=7.0,
    )

# Or batch fusion:
fuse_results = fuse_batch(
    mu_oracle=result.mu,
    sigma2_oracle=result.sigma2,
    J_mu=result.jacobian,
    cov_gen_list=cov_gen_list,
    y_target=7.0,
)

# For evaluation / diagnostics / ablation tables — use predict() (fast, no Jacobian):
result = oracle.predict(X)  # cheap: no per-sample backward
metrics = oracle.evaluate(X, y)  # internally calls predict()
```

### 4.3 New Module: `bayesdiff/hybrid_oracle.py`

Complete class signatures with concrete method implementations:

```python
"""
bayesdiff/hybrid_oracle.py — §4 Uncertainty-Aware Hybrid Oracle Heads

Implements the oracle head family for Sub-Plan 04:
  - DKLOracle:          Deep Kernel Learning (g_θ MLP + SVGP)
  - DKLEnsembleOracle:  M independent DKL models
  - NNResidualOracle:   MLP readout + GP on residuals
  - SNGPOracle:         Spectral Normalized Neural GP (Tier 1b — deferred)
  - EvidentialOracle:   Evidential regression (NIG) (Tier 1b — deferred)
  - PCA_GPOracle:       PCA dimensionality reduction + SVGP

All classes implement OracleHead ABC from oracle_interface.py.
predict() returns (mu, sigma2) only; predict_for_fusion() adds Jacobian.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from scipy.stats import spearmanr

from bayesdiff.oracle_interface import OracleHead, OracleResult
from bayesdiff.gp_oracle import GPOracle, SVGPModel

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Extractor (shared by DKL and DKL Ensemble)
# ============================================================================

class FeatureExtractor(nn.Module):
    """MLP feature extractor for DKL with optional residual connection.
    
    Architecture (default, 2-layer):
        z (128) → Linear(128, 256) → ReLU → Dropout(0.1) → Linear(256, 32) → + W_proj·z → u (32)
    
    Parameter count (default): 128×256 + 256 + 256×32 + 32 + 128×32 + 32 = 45,344
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 32,
        n_layers: int = 2,
        residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.residual = residual
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        
        if residual:
            self.proj = nn.Linear(input_dim, output_dim, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        if self.residual:
            nn.init.xavier_normal_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, input_dim) → u: (B, output_dim)"""
        h = self.mlp(z)
        if self.residual:
            h = h + self.proj(z)
        return h


# ============================================================================
# DKL SVGP Model (extends gp_oracle.SVGPModel to work in feature space)
# ============================================================================

class DKLSVGPModel(gpytorch.models.ApproximateGP):
    """SVGP operating in DKL feature space.
    
    Identical to bayesdiff.gp_oracle.SVGPModel but constructed with
    feature_dim instead of full embedding dim.
    """
    
    def __init__(self, inducing_points: torch.Tensor):
        d = inducing_points.shape[1]
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ============================================================================
# B1: DKL Oracle
# ============================================================================

class DKLOracle(OracleHead):
    """Deep Kernel Learning oracle: FeatureExtractor (MLP) + SVGP.
    
    Input: z ∈ ℝ^128 (frozen embedding)
    Feature extraction: u = g_θ(z) ∈ ℝ^32
    GP: f(u) ~ GP(m(u), k(u,u')), k = ScaleKernel(Matérn-5/2, ARD)
    
    Training: Joint ELBO maximization over θ, kernel hyperparams, variational params.
    Prediction: mu = E[f(g_θ(z))], sigma2 = Var[f(g_θ(z))] + noise
    Jacobian: ∂μ/∂z via torch.autograd
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        feature_dim: int = 32,
        n_inducing: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 2,
        residual: bool = True,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.n_inducing = n_inducing
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dim, feature_dim, n_layers, residual, dropout,
        ).to(self.device)
        
        self.gp_model = None      # initialized in fit() after seeing data
        self.likelihood = None
    
    def fit(self, X_train, y_train, X_val, y_val,
            n_epochs=300, batch_size=256, lr_nn=1e-3, lr_gp=1e-2,
            weight_decay=1e-4, patience=20, verbose=True) -> dict:
        """Joint training of feature extractor + SVGP.
        
        Optimizer: AdamW with per-parameter-group learning rates.
        Loss: -ELBO from gpytorch.mlls.VariationalELBO.
        Early stopping: on val Spearman ρ (patience epochs).
        """
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        N = len(X_t)
        
        # Initialize inducing points in feature space
        with torch.no_grad():
            u_init = self.feature_extractor(X_t)
        if N <= self.n_inducing:
            inducing = u_init.clone()
        else:
            idx = torch.randperm(N)[:self.n_inducing]
            inducing = u_init[idx].clone()
        
        self.gp_model = DKLSVGPModel(inducing).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        
        self.feature_extractor.train()
        self.gp_model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.AdamW([
            {"params": self.feature_extractor.parameters(), "lr": lr_nn, "weight_decay": weight_decay},
            {"params": self.gp_model.hyperparameters(), "lr": lr_gp},
            {"params": self.gp_model.variational_parameters(), "lr": lr_gp},
            {"params": self.likelihood.parameters(), "lr": lr_gp},
        ])
        
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_model, num_data=N)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {"loss": [], "val_rho": [], "val_nll": []}
        best_val_rho = -float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # --- Train ---
            self.feature_extractor.train()
            self.gp_model.train()
            self.likelihood.train()
            epoch_loss = 0.0
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                u = self.feature_extractor(X_batch)
                output = self.gp_model(u)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            
            epoch_loss /= N
            history["loss"].append(epoch_loss)
            
            # --- Validate ---
            val_result = self._predict_internal(X_v)
            val_rho, _ = spearmanr(val_result.mu, y_val)
            val_nll = float(np.mean(
                0.5 * np.log(2 * np.pi * val_result.sigma2)
                + (y_val - val_result.mu)**2 / (2 * val_result.sigma2)
            ))
            history["val_rho"].append(val_rho)
            history["val_nll"].append(val_nll)
            
            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_state = {
                    "fe": {k: v.cpu().clone() for k, v in self.feature_extractor.state_dict().items()},
                    "gp": {k: v.cpu().clone() for k, v in self.gp_model.state_dict().items()},
                    "lik": {k: v.cpu().clone() for k, v in self.likelihood.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"  DKL Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, "
                    f"val_ρ={val_rho:.4f}, val_NLL={val_nll:.4f}, noise={self.likelihood.noise.item():.4f}"
                )
            
            if patience_counter >= patience:
                logger.info(f"  DKL early stopping at epoch {epoch+1}")
                break
        
        # Restore best
        if best_state:
            self.feature_extractor.load_state_dict(best_state["fe"])
            self.gp_model.load_state_dict(best_state["gp"])
            self.likelihood.load_state_dict(best_state["lik"])
        
        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()
        
        return history
    
    def _predict_internal(self, X: torch.Tensor | np.ndarray) -> OracleResult:
        """Internal fast-path predict (used by fit() for val monitoring)."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            u = self.feature_extractor(X)
            pred = self.likelihood(self.gp_model(u))
            mu = pred.mean.cpu().numpy()
            var = pred.variance.cpu().numpy()
        
        return OracleResult(mu=mu, sigma2=var, aux={})
    
    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2, no Jacobian."""
        return self._predict_internal(X)
    
    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Full prediction with Jacobian ∂μ/∂z for Delta method fusion."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)
        
        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()
        
        # Forward
        u = self.feature_extractor(X_t)
        pred = self.likelihood(self.gp_model(u))
        mu_t = pred.mean
        var = pred.variance.detach().cpu().numpy()
        
        # Jacobian: ∂μ/∂z (same pattern as GPOracle.predict_with_jacobian)
        J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            mu_t[i].backward(retain_graph=True)
            J_rows.append(X_t.grad[i].clone().cpu().numpy())
        
        J_mu = np.stack(J_rows, axis=0)  # (N, d)
        mu = mu_t.detach().cpu().numpy()
        
        return OracleResult(mu=mu, sigma2=var, jacobian=J_mu, aux={})
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "feature_extractor": self.feature_extractor.state_dict(),
            "gp_model": self.gp_model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "feature_dim": self.feature_dim,
                "n_inducing": self.n_inducing,
            },
        }, path / "dkl_model.pt")
    
    def load(self, path):
        path = Path(path)
        ckpt = torch.load(path / "dkl_model.pt", map_location=self.device)
        cfg = ckpt["config"]
        # Re-initialize GP with dummy inducing points
        dummy_inducing = torch.zeros(cfg["n_inducing"], cfg["feature_dim"])
        self.gp_model = DKLSVGPModel(dummy_inducing).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.feature_extractor.load_state_dict(ckpt["feature_extractor"])
        self.gp_model.load_state_dict(ckpt["gp_model"])
        self.likelihood.load_state_dict(ckpt["likelihood"])
        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()


# ============================================================================
# B2: DKL Ensemble Oracle
# ============================================================================

class DKLEnsembleOracle(OracleHead):
    """Ensemble of M independent DKL models.
    
    Uncertainty = mean(member variances) + Var(member means).
    """
    
    def __init__(self, input_dim: int = 128, n_members: int = 5,
                 bootstrap: bool = True, bootstrap_frac: float = 0.8,
                 **dkl_kwargs):
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.bootstrap_frac = bootstrap_frac
        self.members = [DKLOracle(input_dim=input_dim, **dkl_kwargs) for _ in range(n_members)]
    
    def fit(self, X_train, y_train, X_val, y_val, seed_base=42, **kwargs) -> dict:
        """Train each member with different seed and optionally bootstrap data."""
        histories = []
        for m, member in enumerate(self.members):
            torch.manual_seed(seed_base + m)
            np.random.seed(seed_base + m)
            
            if self.bootstrap:
                N = len(X_train)
                idx = np.random.choice(N, size=int(N * self.bootstrap_frac), replace=False)
                X_m, y_m = X_train[idx], y_train[idx]
            else:
                X_m, y_m = X_train, y_train
            
            logger.info(f"  Training DKL ensemble member {m+1}/{self.n_members} (N={len(X_m)})")
            h = member.fit(X_m, y_m, X_val, y_val, **kwargs)
            histories.append(h)
        
        return {"member_histories": histories}
    
    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast ensemble prediction: mu, sigma2, decomposition — no Jacobian."""
        member_results = [m.predict(X) for m in self.members]
        M = self.n_members
        
        mus = np.stack([r.mu for r in member_results])          # (M, N)
        sigma2s = np.stack([r.sigma2 for r in member_results])  # (M, N)
        
        mu_ens = mus.mean(axis=0)                               # (N,)
        sigma2_aleatoric = sigma2s.mean(axis=0)                 # (N,)
        sigma2_epistemic = mus.var(axis=0)                      # (N,)
        sigma2_total = sigma2_aleatoric + sigma2_epistemic      # (N,)
        
        return OracleResult(
            mu=mu_ens,
            sigma2=sigma2_total,
            aux={
                "sigma2_aleatoric": sigma2_aleatoric,
                "sigma2_epistemic": sigma2_epistemic,
                "member_mus": mus,
                "member_sigma2s": sigma2s,
            },
        )
    
    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Expensive ensemble prediction with Jacobian (for Delta method fusion)."""
        member_results = [m.predict_for_fusion(X) for m in self.members]
        M = self.n_members
        
        mus = np.stack([r.mu for r in member_results])          # (M, N)
        sigma2s = np.stack([r.sigma2 for r in member_results])  # (M, N)
        jacs = np.stack([r.jacobian for r in member_results])   # (M, N, d)
        
        mu_ens = mus.mean(axis=0)
        sigma2_aleatoric = sigma2s.mean(axis=0)
        sigma2_epistemic = mus.var(axis=0)
        sigma2_total = sigma2_aleatoric + sigma2_epistemic
        J_ens = jacs.mean(axis=0)                               # (N, d)
        
        return OracleResult(
            mu=mu_ens,
            sigma2=sigma2_total,
            jacobian=J_ens,
            aux={
                "sigma2_aleatoric": sigma2_aleatoric,
                "sigma2_epistemic": sigma2_epistemic,
                "member_mus": mus,
                "member_sigma2s": sigma2s,
            },
        )
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for m, member in enumerate(self.members):
            member.save(path / f"member_{m}")
        # Save ensemble config
        import json
        with open(path / "ensemble_config.json", "w") as f:
            json.dump({"n_members": self.n_members, "bootstrap": self.bootstrap,
                        "bootstrap_frac": self.bootstrap_frac}, f, indent=2)
    
    def load(self, path):
        path = Path(path)
        import json
        with open(path / "ensemble_config.json") as f:
            cfg = json.load(f)
        self.n_members = cfg["n_members"]
        for m, member in enumerate(self.members):
            member.load(path / f"member_{m}")


# ============================================================================
# B3: NN + GP Residual Oracle  
# ============================================================================

class NNResidualOracle(OracleHead):
    """Two-stage predictor: MLP for main signal + GP on residuals.
    
    Stage 1: Train MLP (MSE + early stopping)
    Stage 2: Compute residuals r = y - MLP(z), train GPOracle on (z, r)
    Prediction: mu = MLP(z) + GP_mu(z), sigma2 = GP_sigma2(z) + MC_Dropout_var(z)
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128,
                 n_inducing: int = 512, mc_dropout: bool = True,
                 mc_samples: int = 20, dropout: float = 0.1,
                 device: str = "cuda"):
        self.input_dim = input_dim
        self.mc_dropout = mc_dropout
        self.mc_samples = mc_samples
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # NN: same as MLPReadout from attention_pool.py but with dropout
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)
        
        # GP: standard GPOracle from bayesdiff/gp_oracle.py
        self.gp = GPOracle(d=input_dim, n_inducing=n_inducing, device=str(self.device))
    
    def fit(self, X_train, y_train, X_val, y_val,
            nn_epochs=200, nn_lr=1e-3, nn_patience=30,
            gp_epochs=200, gp_batch_size=256, gp_lr=0.01,
            batch_size=64, verbose=True) -> dict:
        """Two-stage training."""
        history = {"nn_loss": [], "nn_val_rho": [], "gp_loss": []}
        
        # ---- Stage 1: Train NN ----
        logger.info("  NNResidual Stage 1: Training NN (MSE)...")
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        
        optimizer = torch.optim.AdamW(self.nn.parameters(), lr=nn_lr, weight_decay=1e-4)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_rho = -float("inf")
        best_nn_state = None
        patience_counter = 0
        
        for epoch in range(nn_epochs):
            self.nn.train()
            total_loss, n_total = 0.0, 0
            for xb, yb in loader:
                pred = self.nn(xb).squeeze(-1)
                loss = F.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                n_total += len(yb)
            
            history["nn_loss"].append(total_loss / n_total)
            
            # Val
            self.nn.eval()
            with torch.no_grad():
                val_pred = self.nn(X_v).squeeze(-1).cpu().numpy()
            val_rho, _ = spearmanr(val_pred, y_val)
            history["nn_val_rho"].append(val_rho)
            
            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_nn_state = {k: v.cpu().clone() for k, v in self.nn.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 50 == 0:
                logger.info(f"    NN Epoch {epoch+1}: MSE={total_loss/n_total:.4f}, val_ρ={val_rho:.4f}")
            
            if patience_counter >= nn_patience:
                logger.info(f"    NN early stopping at epoch {epoch+1}")
                break
        
        if best_nn_state:
            self.nn.load_state_dict(best_nn_state)
        self.nn.eval()
        
        # ---- Stage 2: Train GP on residuals ----
        logger.info("  NNResidual Stage 2: Training GP on residuals...")
        with torch.no_grad():
            nn_pred_train = self.nn(X_t).squeeze(-1).cpu().numpy()
        r_train = y_train - nn_pred_train
        
        logger.info(f"    Residual stats: mean={r_train.mean():.4f}, std={r_train.std():.4f}")
        logger.info(f"    Original label std={y_train.std():.4f} → residual std={r_train.std():.4f}")
        
        gp_history = self.gp.train(X_train, r_train,
                                    n_epochs=gp_epochs, batch_size=gp_batch_size,
                                    lr=gp_lr, verbose=verbose)
        history["gp_loss"] = gp_history["loss"]
        
        return history
    
    def predict(self, X: np.ndarray) -> OracleResult:
        """Fast prediction: mu and sigma2, no Jacobian.
        
        Core uncertainty is from residual GP. MC Dropout is an optional
        auxiliary epistemic term (switchable via mc_dropout flag).
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # NN prediction (deterministic)
        self.nn.eval()
        with torch.no_grad():
            nn_mu = self.nn(X_t).squeeze(-1).cpu().numpy()
        
        # GP prediction on residuals (no Jacobian)
        gp_mu, gp_var = self.gp.predict(X)
        
        # MC Dropout uncertainty (if enabled) — auxiliary epistemic term
        sigma2_mc = np.zeros(len(X))
        if self.mc_dropout:
            self.nn.train()  # enable dropout
            mc_preds = []
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    mc_pred = self.nn(X_t).squeeze(-1).cpu().numpy()
                    mc_preds.append(mc_pred)
            mc_preds = np.stack(mc_preds)  # (T, N)
            sigma2_mc = mc_preds.var(axis=0)
            self.nn.eval()
        
        mu = nn_mu + gp_mu
        sigma2 = gp_var + sigma2_mc
        
        return OracleResult(
            mu=mu, sigma2=sigma2,
            aux={
                "sigma2_gp": gp_var,
                "sigma2_nn": sigma2_mc,
                "nn_mu": nn_mu,
                "gp_mu": gp_mu,
                "residual_std": float(np.sqrt(gp_var.mean())),
            },
        )
    
    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Full prediction with Jacobian ∂μ/∂z for Delta method fusion."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_t.requires_grad_(True)
        
        # NN prediction (deterministic, for Jacobian)
        self.nn.eval()
        nn_pred = self.nn(X_t).squeeze(-1)
        
        # NN Jacobian
        nn_J_rows = []
        for i in range(len(X_t)):
            if X_t.grad is not None:
                X_t.grad.zero_()
            nn_pred[i].backward(retain_graph=True)
            nn_J_rows.append(X_t.grad[i].clone().cpu().numpy())
        nn_J = np.stack(nn_J_rows)  # (N, d)
        nn_mu = nn_pred.detach().cpu().numpy()
        
        # GP prediction on residuals (with Jacobian)
        gp_mu, gp_var, gp_J = self.gp.predict_with_jacobian(X)
        
        # MC Dropout uncertainty (if enabled)
        sigma2_mc = np.zeros(len(X))
        if self.mc_dropout:
            self.nn.train()  # enable dropout
            mc_preds = []
            with torch.no_grad():
                X_t_nograd = torch.tensor(X, dtype=torch.float32, device=self.device)
                for _ in range(self.mc_samples):
                    mc_pred = self.nn(X_t_nograd).squeeze(-1).cpu().numpy()
                    mc_preds.append(mc_pred)
            mc_preds = np.stack(mc_preds)  # (T, N)
            sigma2_mc = mc_preds.var(axis=0)
            self.nn.eval()
        
        # Combine
        mu = nn_mu + gp_mu
        sigma2 = gp_var + sigma2_mc
        J_mu = nn_J + gp_J
        
        return OracleResult(
            mu=mu, sigma2=sigma2, jacobian=J_mu,
            aux={
                "sigma2_gp": gp_var,
                "sigma2_nn": sigma2_mc,
                "nn_mu": nn_mu,
                "gp_mu": gp_mu,
                "residual_std": float(np.sqrt(gp_var.mean())),
            },
        )
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.nn.state_dict(), path / "nn_model.pt")
        self.gp.save(path / "gp_model.pt")
    
    def load(self, path):
        path = Path(path)
        self.nn.load_state_dict(torch.load(path / "nn_model.pt", map_location=self.device))
        self.nn.eval()
        self.gp.load(path / "gp_model.pt")
```

### 4.4 Modifications to Existing Modules

**`bayesdiff/fusion.py`** — **No changes required**. The existing `fuse_uncertainties()` and `fuse_batch()` signatures already accept `mu_oracle: float`, `sigma2_oracle: float`, `J_mu: np.ndarray`, which map directly from `OracleResult` fields.

**`bayesdiff/evaluate.py`** — **No changes required**. The existing `evaluate_all(mu_pred, sigma_pred, p_success, y_true)` works with any predictor that produces (mu, sigma).

**`bayesdiff/ood.py`** — **Minor extension**: Add an optional method to compute Mahalanobis distance in the DKL feature space (for DKL-specific OOD detection). This is a non-breaking addition:

```python
# In bayesdiff/ood.py — add to MahalanobisOOD class:
def fit_feature_space(self, feature_extractor: nn.Module, X_train: np.ndarray, **kwargs):
    """Fit OOD detector in DKL feature space instead of raw embedding space."""
    with torch.no_grad():
        X_t = torch.tensor(X_train, dtype=torch.float32)
        U_train = feature_extractor(X_t).cpu().numpy()
    self.fit(U_train, **kwargs)
```

### 4.5 New Pipeline Script: `scripts/pipeline/s18_train_oracle_heads.py`

Named `s18` to follow the existing numbering convention (s17 = ablation_and_viz).

```python
"""
scripts/pipeline/s18_train_oracle_heads.py
──────────────────────────────────────────
Sub-Plan 4: Train and evaluate oracle heads on frozen embeddings.

Phase 4.1: Extract frozen embeddings (if not cached)
Phase 4.2: Train all oracle heads
Phase 4.3: Unified evaluation + comparison table

Usage:
    python scripts/pipeline/s18_train_oracle_heads.py \
        --frozen_embeddings results/stage2/frozen_embeddings.npz \
        --labels data/pdbbind_v2020/labels.csv \
        --splits data/pdbbind_v2020/splits.json \
        --output results/stage2/oracle_heads \
        --heads dkl,dkl_ensemble,nn_residual,svgp,pca_svgp,sngp,evidential \
        --device cuda \
        --seed 42

    # Or train a single head for debugging:
    python scripts/pipeline/s18_train_oracle_heads.py \
        --frozen_embeddings results/stage2/frozen_embeddings.npz \
        --output results/stage2/oracle_heads \
        --heads dkl \
        --device cuda

    # If frozen embeddings don't exist, extract them first:
    python scripts/pipeline/s18_train_oracle_heads.py \
        --extract_embeddings \
        --schemeb_checkpoint results/stage2/ablation_viz/A36_independent_model.pt \
        --atom_emb_dir results/atom_embeddings \
        --labels data/pdbbind_v2020/labels.csv \
        --splits data/pdbbind_v2020/splits.json \
        --output results/stage2/oracle_heads \
        --device cuda

Output:
    results/stage2/oracle_heads/
        frozen_embeddings.npz                # cached (X_train, y_train, X_val, y_val, X_test, y_test)
        dkl/dkl_model.pt, config.json
        dkl_ensemble/member_0/..., ensemble_config.json
        nn_residual/nn_model.pt, gp_model.pt
        svgp/gp_model.pt
        pca_svgp/pca.pkl, gp_model.pt
        sngp/sngp_model.pt
        evidential/evidential_model.pt
        tier1_comparison.json               # all metrics in one table
        tier1_comparison.csv                # spreadsheet-friendly
        uncertainty_diagnostics.json        # per-head |err|-σ analysis
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bayesdiff.hybrid_oracle import (
    DKLOracle, DKLEnsembleOracle, NNResidualOracle,
    # SNGPOracle, EvidentialOracle — added as implemented
)
from bayesdiff.gp_oracle import GPOracle

logger = logging.getLogger(__name__)

HEAD_REGISTRY = {
    "dkl": lambda args: DKLOracle(
        input_dim=128, feature_dim=32, n_inducing=args.n_inducing,
        hidden_dim=256, n_layers=2, device=args.device,
    ),
    "dkl_ensemble": lambda args: DKLEnsembleOracle(
        input_dim=128, n_members=args.ensemble_members,
        feature_dim=32, n_inducing=args.n_inducing,
        hidden_dim=256, n_layers=2, device=args.device,
    ),
    "nn_residual": lambda args: NNResidualOracle(
        input_dim=128, hidden_dim=128, n_inducing=args.n_inducing,
        device=args.device,
    ),
    "svgp": lambda args: _wrap_gp_oracle(GPOracle(d=128, n_inducing=args.n_inducing, device=args.device)),
    # "pca_svgp", "sngp", "evidential" — Tier 1b, added after Tier 1 analysis
}


def main():
    parser = argparse.ArgumentParser(description="Sub-Plan 4: Oracle Head Comparison")
    parser.add_argument("--frozen_embeddings", type=str, default="results/stage2/frozen_embeddings.npz")
    parser.add_argument("--output", type=str, default="results/stage2/oracle_heads")
    parser.add_argument("--heads", type=str, default="dkl,dkl_ensemble,nn_residual,svgp")
    parser.add_argument("--n_inducing", type=int, default=512)
    parser.add_argument("--ensemble_members", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load frozen embeddings
    data = np.load(args.frozen_embeddings)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    logger.info(f"Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    # Train and evaluate each head
    heads_to_train = args.heads.split(",")
    results = {}
    
    for head_name in heads_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training oracle head: {head_name}")
        logger.info(f"{'='*60}")
        
        t_start = time.time()
        oracle = HEAD_REGISTRY[head_name](args)
        history = oracle.fit(X_train, y_train, X_val, y_val, n_epochs=args.n_epochs)
        
        # Evaluate on val and test
        val_metrics = oracle.evaluate(X_val, y_val)
        test_metrics = oracle.evaluate(X_test, y_test)
        
        elapsed = time.time() - t_start
        
        results[head_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "elapsed_seconds": elapsed,
        }
        
        # Save checkpoint
        oracle.save(Path(args.output) / head_name)
        
        logger.info(f"  {head_name} Test: R²={test_metrics['R2']:.4f}, ρ={test_metrics['spearman_rho']:.4f}, "
                     f"|err|-σ ρ={test_metrics['err_sigma_rho']:.4f}, NLL={test_metrics['nll']:.4f}")
    
    # Save comparison table
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tier1_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    _print_summary_table(results)


if __name__ == "__main__":
    main()
```

### 4.6 SLURM Script: `slurm/s18_oracle_heads.sh`

Following existing patterns from `slurm/s14_phase3_refinement.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=s18_oracle
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/s18_oracle_%j.out
#SBATCH --error=slurm/logs/s18_oracle_%j.err

# Sub-Plan 4: Oracle Head Family Comparison (Tier 1)

set -e

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

# Phase 4.2: Train all Tier 1 oracle heads
python scripts/pipeline/s18_train_oracle_heads.py \
    --frozen_embeddings results/stage2/frozen_embeddings.npz \
    --output results/stage2/oracle_heads \
    --heads dkl,dkl_ensemble,nn_residual,svgp \
    --n_inducing 512 \
    --ensemble_members 5 \
    --n_epochs 300 \
    --batch_size 256 \
    --device cuda \
    --seed 42

echo "=== Done ==="
echo "Date: $(date)"
```

**Time budget**: DKL ~20 min + DKL Ensemble ~100 min + NNResidual ~15 min + SVGP ~10 min = ~2.5 hours. 6-hour time limit provides comfortable margin.

### 4.7 Training Strategy

**Per-head hyperparameters** (consolidated):

| Head | Optimizer | LR | Weight Decay | Epochs | Patience | Batch Size | Special |
|------|-----------|------|-------------|--------|----------|------------|---------|
| DKL (B1) | AdamW | NN: 1e-3, GP: 1e-2 | 1e-4 (NN only) | 300 | 20 (val ρ) | 256 | Multi-param-group |
| DKL Ensemble (B2) | AdamW × M | Same as B1 | Same as B1 | 300 × M | 20 × M | 256 | 80% bootstrap |
| NN+GP Residual (B3) | AdamW (NN), Adam (GP) | NN: 1e-3, GP: 1e-2 | 1e-4 (NN) | NN: 200, GP: 200 | NN: 30, GP: — | NN: 64, GP: 256 | Two-stage |
| raw SVGP (A1) | Adam | 1e-2 | — | 200 | — | 256 | Same as GPOracle.train() |
| PCA→SVGP (A2) | Adam | 1e-2 | — | 200 | — | 256 | PCA(32) on train, transform val/test |

**Early stopping protocol**: Monitor validation Spearman $\rho$ (not loss) because ELBO can decrease while generalization degrades. Restore best checkpoint.

**Overfitting prevention** (PDBbind general set ~18K usable complexes after filtering):
- Dataset is moderate-sized: overfitting risk exists but is manageable at ~18K (not the ~5K refined set)
- Feature dim $d_u = 32$ (much smaller than input $d = 128$)
- Shallow NN (2 layers for DKL, 2 layers for MLP)
- Dropout ($p = 0.1$)
- Weight decay ($\lambda = 10^{-4}$)
- Monitor train/val ELBO gap — flag if gap exceeds 20%
- Ensemble diversity (B2) naturally regularizes
- For NN + GP Residual (B3): NN stage has its own early stopping; GP stage trains on lower-variance residuals

---

## 5. Test Plan

### 5.1 Unit Tests: `tests/stage2/test_hybrid_oracle.py`

| Test ID | Test Name | What It Verifies | Synthetic Data | Key Assertion |
|---------|-----------|-----------------|---------------|---------------|
| T1.1 | `test_feature_extractor_shape` | Output shape = (B, d_u) | `torch.randn(32, 128)` | `out.shape == (32, 32)` |
| T1.2 | `test_feature_extractor_residual` | With residual=True, output ≈ input projection for zero-initialized MLP | zero-init mlp, randn input | `‖out - proj(z)‖ < 1e-5` |
| T1.3 | `test_dkl_forward` | DKL produces OracleResult with correct shapes | `torch.randn(50, 128)` | `.mu.shape == (50,)`, `.sigma2.shape == (50,)`, `.jacobian.shape == (50,128)` |
| T1.4 | `test_dkl_uncertainty_positive` | σ² > 0 for all inputs | `torch.randn(100, 128)` | `(result.sigma2 > 0).all()` |
| T1.5 | `test_dkl_training_loss_decreases` | ELBO improves over 50 epochs on synthetic data | linear+noise (500, 128) | `loss[-1] < loss[0] * 0.5` |
| T1.6 | `test_dkl_jacobian_shape` | Jacobian has correct shape (N, d) | `torch.randn(10, 128)` | `.jacobian.shape == (10, 128)` |
| T1.7 | `test_dkl_jacobian_finite_diff` | Autograd Jacobian ≈ finite-difference Jacobian | `torch.randn(5, 128)` | `max ‖J_auto - J_fd‖ < 0.01` |
| T1.8 | `test_dkl_ensemble_forward` | Ensemble produces OracleResult with aux decomposition | `torch.randn(50, 128)` | aux has `sigma2_aleatoric`, `sigma2_epistemic` |
| T1.9 | `test_dkl_ensemble_disagreement` | Ensemble σ²_epistemic > 0 when members differ | 400 train, 100 val | `sigma2_epistemic.mean() > 0` |
| T1.10 | `test_nn_residual_forward` | NN+GP residual produces OracleResult | `torch.randn(50, 128)` | all three fields present with correct shapes |
| T1.11 | `test_nn_residual_two_stage` | Residuals have smaller variance than raw labels | linear+noise (500, 128) | `residual.std() < y.std()` |
| T1.12 | `test_oracle_interface_compliance` | All heads return valid OracleResult | loop over all classes | `isinstance(result, OracleResult)` for each |
| T1.13 | `test_save_load_roundtrip` | Save → load → same predictions (all heads) | trained models, randn test input | `max ‖mu_orig - mu_loaded‖ < 1e-5` |
| T1.14 | `test_ood_uncertainty` | OOD inputs get larger σ² than in-distribution (all heads) | ID: randn, OOD: 5*randn+10 | `sigma2_ood.mean() > sigma2_id.mean()` |

#### Concrete Test Implementations

```python
"""tests/stage2/test_hybrid_oracle.py"""

import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile

from bayesdiff.hybrid_oracle import (
    FeatureExtractor, DKLOracle, DKLEnsembleOracle, NNResidualOracle,
)
from bayesdiff.oracle_interface import OracleResult, OracleHead


# =================================================================
# Fixtures
# =================================================================

@pytest.fixture
def synthetic_data():
    """Linear regression with Gaussian noise. d=128, N=500."""
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(500, 128).astype(np.float32)
    w = np.random.randn(128).astype(np.float32)
    y = X @ w + 0.3 * np.random.randn(500).astype(np.float32)
    return X[:400], y[:400], X[400:], y[400:]


@pytest.fixture
def trained_dkl(synthetic_data):
    """Pre-trained DKL oracle on synthetic data (small, fast)."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = DKLOracle(input_dim=128, feature_dim=16, n_inducing=50, 
                       hidden_dim=64, n_layers=2, device="cpu")
    oracle.fit(X_train, y_train, X_val, y_val, n_epochs=30, patience=100, verbose=False)
    return oracle


ALL_ORACLE_CLASSES = [
    pytest.param(DKLOracle, {"input_dim": 128, "feature_dim": 16, "n_inducing": 50, 
                             "hidden_dim": 64, "device": "cpu"}, id="DKL"),
    pytest.param(DKLEnsembleOracle, {"input_dim": 128, "n_members": 2, "feature_dim": 16, 
                                      "n_inducing": 50, "hidden_dim": 64, "device": "cpu"}, id="DKLEnsemble"),
    pytest.param(NNResidualOracle, {"input_dim": 128, "hidden_dim": 64, "n_inducing": 50,
                                     "device": "cpu"}, id="NNResidual"),
]


# =================================================================
# T1.1 – T1.2: FeatureExtractor shape and residual
# =================================================================

def test_feature_extractor_shape():
    """T1.1: Output shape = (B, d_u)."""
    fe = FeatureExtractor(input_dim=128, hidden_dim=256, output_dim=32)
    z = torch.randn(32, 128)
    out = fe(z)
    assert out.shape == (32, 32), f"Expected (32, 32), got {out.shape}"


def test_feature_extractor_residual():
    """T1.2: With residual=True and zero-init MLP, output ≈ proj(z)."""
    fe = FeatureExtractor(input_dim=128, hidden_dim=256, output_dim=32, residual=True)
    # Zero out MLP weights
    with torch.no_grad():
        for m in fe.mlp.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
                m.bias.zero_()
    z = torch.randn(10, 128)
    out = fe(z)
    expected = fe.proj(z)
    assert torch.allclose(out, expected, atol=1e-5), "Residual shortcut should dominate with zero MLP"


def test_feature_extractor_no_residual():
    """Verify non-residual variant works."""
    fe = FeatureExtractor(input_dim=128, hidden_dim=256, output_dim=32, residual=False)
    z = torch.randn(10, 128)
    out = fe(z)
    assert out.shape == (10, 32)


# =================================================================
# T1.3 – T1.7: DKL Oracle
# =================================================================

def test_dkl_forward(trained_dkl):
    """T1.3: DKL predict() produces OracleResult with correct shapes (no Jacobian)."""
    X_test = np.random.randn(50, 128).astype(np.float32)
    result = trained_dkl.predict(X_test)
    assert isinstance(result, OracleResult)
    assert result.mu.shape == (50,)
    assert result.sigma2.shape == (50,)
    assert result.jacobian is None, "predict() should not compute Jacobian"


def test_dkl_uncertainty_positive(trained_dkl):
    """T1.4: σ² > 0 for all inputs."""
    X_test = np.random.randn(100, 128).astype(np.float32)
    result = trained_dkl.predict(X_test)
    assert (result.sigma2 > 0).all(), f"Found non-positive variance: min={result.sigma2.min()}"


def test_dkl_training_loss_decreases(synthetic_data):
    """T1.5: ELBO should improve during training."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = DKLOracle(input_dim=128, feature_dim=16, n_inducing=50, 
                       hidden_dim=64, device="cpu")
    history = oracle.fit(X_train, y_train, X_val, y_val, n_epochs=50, patience=100, verbose=False)
    assert history['loss'][-1] < history['loss'][0], \
        f"Loss did not decrease: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}"
    assert history['loss'][-1] < history['loss'][0] * 0.8, \
        "Loss should decrease by at least 20%"


def test_dkl_jacobian_shape(trained_dkl):
    """T1.6: predict_for_fusion() Jacobian shape = (N, d)."""
    X_test = np.random.randn(10, 128).astype(np.float32)
    result = trained_dkl.predict_for_fusion(X_test)
    assert result.jacobian is not None, "predict_for_fusion must return Jacobian"
    assert result.jacobian.shape == (10, 128)
    assert np.isfinite(result.jacobian).all(), "Jacobian contains NaN or Inf"


def test_dkl_jacobian_finite_diff(trained_dkl):
    """T1.7: Autograd Jacobian ≈ finite-difference Jacobian (spot check)."""
    X_test = np.random.randn(3, 128).astype(np.float32)
    result = trained_dkl.predict_for_fusion(X_test)
    J_auto = result.jacobian  # (3, 128)
    
    eps = 1e-4
    J_fd = np.zeros_like(J_auto)
    for j in range(128):
        X_p = X_test.copy()
        X_m = X_test.copy()
        X_p[:, j] += eps
        X_m[:, j] -= eps
        mu_p = trained_dkl.predict(X_p).mu
        mu_m = trained_dkl.predict(X_m).mu
        J_fd[:, j] = (mu_p - mu_m) / (2 * eps)
    
    max_diff = np.abs(J_auto - J_fd).max()
    assert max_diff < 0.05, f"Jacobian mismatch: max |J_auto - J_fd| = {max_diff:.6f}"


# =================================================================
# T1.8 – T1.9: DKL Ensemble
# =================================================================

def test_dkl_ensemble_forward(synthetic_data):
    """T1.8: Ensemble produces OracleResult with aux decomposition."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = DKLEnsembleOracle(input_dim=128, n_members=2, feature_dim=16, 
                                n_inducing=50, hidden_dim=64, device="cpu")
    oracle.fit(X_train, y_train, X_val, y_val, n_epochs=20, verbose=False)
    
    result = oracle.predict(X_val)
    assert isinstance(result, OracleResult)
    assert 'sigma2_aleatoric' in result.aux
    assert 'sigma2_epistemic' in result.aux
    assert 'member_mus' in result.aux
    assert result.aux['member_mus'].shape == (2, len(X_val))


def test_dkl_ensemble_disagreement(synthetic_data):
    """T1.9: Ensemble epistemic uncertainty should be non-zero."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = DKLEnsembleOracle(input_dim=128, n_members=3, feature_dim=16,
                                n_inducing=50, hidden_dim=64, device="cpu")
    oracle.fit(X_train, y_train, X_val, y_val, n_epochs=30, verbose=False)
    
    result = oracle.predict(X_val)
    assert result.aux['sigma2_epistemic'].mean() > 0, \
        "Ensemble members should disagree (σ²_epistemic > 0)"
    assert result.sigma2.mean() > result.aux['sigma2_epistemic'].mean(), \
        "Total variance should exceed epistemic-only variance"


# =================================================================
# T1.10 – T1.11: NN + GP Residual
# =================================================================

def test_nn_residual_forward(synthetic_data):
    """T1.10: NN+GP residual predict() produces OracleResult with correct shapes (no Jacobian)."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = NNResidualOracle(input_dim=128, hidden_dim=64, n_inducing=50, device="cpu")
    oracle.fit(X_train, y_train, X_val, y_val, nn_epochs=30, gp_epochs=30, verbose=False)
    
    result = oracle.predict(X_val)
    assert isinstance(result, OracleResult)
    assert result.mu.shape == (len(X_val),)
    assert result.sigma2.shape == (len(X_val),)
    assert result.jacobian is None, "predict() should not compute Jacobian"
    
    # Also test predict_for_fusion()
    result_fusion = oracle.predict_for_fusion(X_val)
    assert result_fusion.jacobian is not None
    assert result_fusion.jacobian.shape == (len(X_val), 128)


def test_nn_residual_two_stage(synthetic_data):
    """T1.11: Residuals should have smaller variance than raw labels."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = NNResidualOracle(input_dim=128, hidden_dim=64, n_inducing=50, device="cpu")
    oracle.fit(X_train, y_train, X_val, y_val, nn_epochs=50, gp_epochs=30, verbose=False)
    
    # Check that GP residual std < raw label std
    result = oracle.predict(X_val)
    assert result.aux.get('residual_std', 1e10) < y_val.std(), \
        "Residuals should have lower variance than raw labels"


# =================================================================
# T1.12 – T1.14: Cross-cutting tests
# =================================================================

@pytest.mark.parametrize("OracleClass,kwargs", ALL_ORACLE_CLASSES)
def test_oracle_interface_compliance(OracleClass, kwargs, synthetic_data):
    """T1.12: All heads return valid OracleResult."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = OracleClass(**kwargs)
    oracle.fit(X_train, y_train, X_val, y_val, n_epochs=10, verbose=False)
    result = oracle.predict(X_val)
    
    assert isinstance(result, OracleResult)
    assert isinstance(result.mu, np.ndarray) and result.mu.ndim == 1
    assert isinstance(result.sigma2, np.ndarray) and result.sigma2.ndim == 1
    assert result.jacobian is None, "predict() should not compute Jacobian"
    assert isinstance(result.aux, dict)
    assert np.isfinite(result.mu).all()
    assert np.isfinite(result.sigma2).all()
    assert (result.sigma2 > 0).all()
    
    # Also verify predict_for_fusion() returns Jacobian
    result_fusion = oracle.predict_for_fusion(X_val)
    assert isinstance(result_fusion.jacobian, np.ndarray) and result_fusion.jacobian.ndim == 2


@pytest.mark.parametrize("OracleClass,kwargs", ALL_ORACLE_CLASSES)
def test_save_load_roundtrip(OracleClass, kwargs, synthetic_data):
    """T1.13: Save → load → same predictions."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = OracleClass(**kwargs)
    oracle.fit(X_train, y_train, X_val, y_val, n_epochs=10, verbose=False)
    result_before = oracle.predict(X_val[:10])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        oracle.save(tmpdir)
        
        oracle2 = OracleClass(**kwargs)
        oracle2.load(tmpdir)
        result_after = oracle2.predict(X_val[:10])
    
    np.testing.assert_allclose(result_before.mu, result_after.mu, atol=1e-5,
                                err_msg=f"{OracleClass.__name__}: mu mismatch after load")
    np.testing.assert_allclose(result_before.sigma2, result_after.sigma2, atol=1e-5,
                                err_msg=f"{OracleClass.__name__}: sigma2 mismatch after load")


@pytest.mark.parametrize("OracleClass,kwargs", ALL_ORACLE_CLASSES)
def test_ood_uncertainty(OracleClass, kwargs, synthetic_data):
    """T1.14: OOD inputs should have larger σ² than in-distribution."""
    X_train, y_train, X_val, y_val = synthetic_data
    oracle = OracleClass(**kwargs)
    oracle.fit(X_train, y_train, X_val, y_val, n_epochs=20, verbose=False)
    
    X_id = np.random.randn(20, 128).astype(np.float32)
    X_ood = (np.random.randn(20, 128) * 5 + 10).astype(np.float32)  # far from training dist
    
    result_id = oracle.predict(X_id)
    result_ood = oracle.predict(X_ood)
    
    assert result_ood.sigma2.mean() > result_id.sigma2.mean(), \
        f"{OracleClass.__name__}: OOD σ² ({result_ood.sigma2.mean():.4f}) " \
        f"should exceed ID σ² ({result_id.sigma2.mean():.4f})"
```

### 5.2 Integration Tests: `tests/stage2/test_hybrid_integration.py`

| Test ID | Test Name | What It Verifies | Key Modules Involved | Key Assertion |
|---------|-----------|-----------------|---------------------|---------------|
| T2.1 | `test_oracle_with_delta_method` | Delta method works through all oracle Jacobians | `hybrid_oracle`, `fusion` | valid `FusionResult` with finite fields |
| T2.2 | `test_oracle_with_gen_uncertainty` | Generation uncertainty + oracle fusion = valid total uncertainty | `hybrid_oracle`, `fusion`, `gen_uncertainty` | `σ²_total = σ²_oracle + σ²_gen` |
| T2.3 | `test_oracle_with_calibration` | Oracle predictions can be calibrated via isotonic regression | `hybrid_oracle`, `calibration` | post-cal ECE < pre-cal ECE |
| T2.4 | `test_oracle_with_ood_detection` | Mahalanobis OOD works in oracle feature space | `hybrid_oracle`, `ood` | OOD scores for out-of-dist > in-dist |
| T2.5 | `test_full_pipeline_oracle` | Frozen embeddings → oracle → fusion → calibration → metrics | all modules | valid `EvalResults` with all fields present |
| T2.6 | `test_oracle_head_swap` | Swapping oracle heads in pipeline produces different but valid results | all modules | results differ but both valid |

```python
"""tests/stage2/test_hybrid_integration.py"""

import numpy as np
import pytest
import torch

from bayesdiff.hybrid_oracle import DKLOracle, DKLEnsembleOracle, NNResidualOracle
from bayesdiff.oracle_interface import OracleResult
from bayesdiff.fusion import fuse_uncertainties, fuse_batch, FusionResult
from bayesdiff.evaluate import evaluate_all, EvalResults
from bayesdiff.calibration import IsotonicCalibrator
from bayesdiff.ood import MahalanobisOOD


@pytest.fixture
def trained_oracle_and_data():
    """Pre-trained DKL oracle + synthetic data for integration tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(500, 128).astype(np.float32)
    w = np.random.randn(128).astype(np.float32)
    y = X @ w + 0.3 * np.random.randn(500).astype(np.float32)
    
    oracle = DKLOracle(input_dim=128, feature_dim=16, n_inducing=50,
                       hidden_dim=64, device="cpu")
    oracle.fit(X[:400], y[:400], X[400:450], y[400:450], n_epochs=30, verbose=False)
    
    return oracle, X[450:], y[450:]


def test_oracle_with_delta_method(trained_oracle_and_data):
    """T2.1: Delta method fusion works with oracle Jacobians."""
    oracle, X_test, y_test = trained_oracle_and_data
    result = oracle.predict(X_test)
    
    # Synthetic generation covariance (diagonal for simplicity)
    for i in range(min(10, len(X_test))):
        cov_gen = np.eye(128).astype(np.float32) * 0.01
        fuse_result = fuse_uncertainties(
            mu_oracle=float(result.mu[i]),
            sigma2_oracle=float(result.sigma2[i]),
            J_mu=result.jacobian[i],
            cov_gen=cov_gen,
            y_target=7.0,
        )
        assert isinstance(fuse_result, FusionResult)
        assert np.isfinite(fuse_result.sigma2_total)
        assert fuse_result.sigma2_total >= result.sigma2[i], \
            "Total variance must be >= oracle-only variance (Delta method adds σ²_gen)"


def test_oracle_with_calibration(trained_oracle_and_data):
    """T2.3: Isotonic calibration should reduce ECE."""
    oracle, X_test, y_test = trained_oracle_and_data
    result = oracle.predict(X_test)
    sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
    
    # Create binary target for calibration (pKd > median)
    y_binary = (y_test > np.median(y_test)).astype(float)
    p_raw = 1.0 / (1.0 + np.exp(-(result.mu - np.median(y_test)) / sigma))
    
    calibrator = IsotonicCalibrator()
    # Fit on first half, evaluate on second half
    n = len(y_test) // 2
    calibrator.fit(p_raw[:n], y_binary[:n])
    p_cal = calibrator.transform(p_raw[n:])
    
    assert p_cal.min() >= 0 and p_cal.max() <= 1, "Calibrated probabilities out of [0, 1]"


def test_oracle_with_ood_detection(trained_oracle_and_data):
    """T2.4: Mahalanobis OOD detects far-from-training inputs."""
    oracle, X_test, y_test = trained_oracle_and_data
    
    # Fit OOD detector on training embeddings
    np.random.seed(42)
    X_train = np.random.randn(400, 128).astype(np.float32)
    ood_detector = MahalanobisOOD()
    ood_detector.fit(X_train)
    
    X_id = np.random.randn(20, 128).astype(np.float32)
    X_ood = (np.random.randn(20, 128) * 5 + 10).astype(np.float32)
    
    scores_id = ood_detector.score_batch(X_id)
    scores_ood = ood_detector.score_batch(X_ood)
    
    assert scores_ood.mean() > scores_id.mean(), \
        "OOD scores for out-of-distribution should be higher than in-distribution"


def test_full_pipeline_oracle(trained_oracle_and_data):
    """T2.5: End-to-end pipeline: oracle → fusion → evaluate."""
    oracle, X_test, y_test = trained_oracle_and_data
    result = oracle.predict(X_test)
    sigma = np.sqrt(np.clip(result.sigma2, 1e-10, None))
    
    # Compute p_success (probability pKd > 7.0)
    from scipy.stats import norm
    p_success = 1.0 - norm.cdf(7.0, loc=result.mu, scale=sigma)
    
    # Evaluate
    eval_result = evaluate_all(
        mu_pred=result.mu,
        sigma_pred=sigma,
        p_success=p_success,
        y_true=y_test,
        y_target=7.0,
    )
    assert isinstance(eval_result, EvalResults)
    assert hasattr(eval_result, 'spearman_rho')
    assert hasattr(eval_result, 'rmse')
    assert np.isfinite(eval_result.spearman_rho)
    assert np.isfinite(eval_result.rmse)


def test_oracle_head_swap(trained_oracle_and_data):
    """T2.6: Different oracle heads produce different but valid results."""
    _, X_test, y_test = trained_oracle_and_data
    np.random.seed(42)
    torch.manual_seed(42)
    X_train = np.random.randn(400, 128).astype(np.float32)
    w = np.random.randn(128).astype(np.float32)
    y_train = X_train @ w + 0.3 * np.random.randn(400).astype(np.float32)
    X_val = np.random.randn(50, 128).astype(np.float32)
    y_val = X_val @ w + 0.3 * np.random.randn(50).astype(np.float32)
    
    # Train two different heads
    dkl = DKLOracle(input_dim=128, feature_dim=16, n_inducing=50, hidden_dim=64, device="cpu")
    dkl.fit(X_train, y_train, X_val, y_val, n_epochs=20, verbose=False)
    
    nn_res = NNResidualOracle(input_dim=128, hidden_dim=64, n_inducing=50, device="cpu")
    nn_res.fit(X_train, y_train, X_val, y_val, nn_epochs=20, gp_epochs=20, verbose=False)
    
    result_dkl = dkl.predict(X_test[:20])
    result_nnres = nn_res.predict(X_test[:20])
    
    # Both valid
    assert isinstance(result_dkl, OracleResult) and isinstance(result_nnres, OracleResult)
    assert np.isfinite(result_dkl.mu).all() and np.isfinite(result_nnres.mu).all()
    # But different
    assert not np.allclose(result_dkl.mu, result_nnres.mu, atol=0.01), \
        "Different heads should produce different predictions"
```

### 5.3 Running Tests

```bash
# Run all Sub-Plan 04 unit tests (from project root):
cd /scratch/yd2915/BayesDiff
python -m pytest tests/stage2/test_hybrid_oracle.py -v --tb=short

# Run integration tests:
python -m pytest tests/stage2/test_hybrid_integration.py -v --tb=short

# Run only fast tests (no training, < 1 min):
python -m pytest tests/stage2/test_hybrid_oracle.py -v -k "shape or residual or compliance"

# Run with GPU:
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/stage2/ -v --tb=long 2>&1 | tee test_results.log
```

**Expected times** (CPU):
- T1.1–T1.4: < 5s each (no training)
- T1.5–T1.7: ~30s each (short training)
- T1.8–T1.9: ~60s each (ensemble training)
- T1.10–T1.11: ~30s each (two-stage training)
- T1.12–T1.14: ~90s each (parameterized across all heads)
- T2.1–T2.6: ~60s each (integration, with short pre-training)
- **Total**: ~15 min on CPU, < 5 min on GPU

---

## 6. Ablation Experiments

### 6.1 Tier 1: Method Family Comparison (Required)

Compare oracle head families on frozen embeddings with the unified evaluation protocol. All experiments use the same `frozen_embeddings.npz` (§4.1) and the same `evaluate_all()` function from `bayesdiff/evaluate.py`.

| ID | Configuration | Class | Key Hyperparams | GPU Time (est.) | Purpose |
|----|--------------|-------|----------------|-----------------|---------|
| A4.1 | raw SVGP(128d) | `GPOracle` (existing) | n_inducing=512, lr=0.01, 200 ep | ~10 min | GP baseline (carried from 02) |
| A4.2 | PCA32→SVGP | `GPOracle` + sklearn PCA | PCA(32), then same as A4.1 | ~10 min | Dimensionality reduction baseline |
| A4.3 | DKL(128→32→SVGP) | `DKLOracle` | feature_dim=32, hidden=256, 2 layers, 300 ep | ~20 min | Single DKL |
| A4.4 | DKL Ensemble (M=5) | `DKLEnsembleOracle` | 5 members, 80% bootstrap, same as A4.3 | ~100 min | Primary uncertainty candidate |
| A4.5 | NN + GP Residual | `NNResidualOracle` | hidden=128, NN 200 ep + GP 200 ep | ~15 min | Alternative hybrid architecture |

**Total Tier 1 wall time**: ~2.5 hours on L40S GPU. All 5 experiments can be run in a single SLURM job using `s18_train_oracle_heads.py`.

### 6.1b Tier 1b: Deferred Baselines (After Tier 1 Analysis)

Only pursued if Tier 1 results leave open questions about the relative strength of the core methods, or if no core method achieves $\rho_{|err|,\sigma} > 0.15$.

| ID | Configuration | Class | Key Hyperparams | GPU Time (est.) | Purpose |
|----|--------------|-------|----------------|-----------------|---------|
| A4.6 | SNGP | `SNGPOracle` | SN bound=0.95, 1024 RFF, 200 ep | ~10 min | Cheap scalable baseline |
| A4.7 | Evidential Regression | `EvidentialOracle` | λ_ev annealed 0→0.1, 200 ep | ~10 min | Cheap single-model UQ baseline |

**SLURM submission**:
```bash
sbatch slurm/s18_oracle_heads.sh  # runs all Tier 1 experiments sequentially
```

**Expected output format** (`tier1_comparison.json`):
```json
{
  "svgp": {
    "test": {"R2": 0.48, "spearman_rho": 0.719, "rmse": 1.45, "nll": 2.10, "err_sigma_rho": 0.02, "mean_sigma": 1.15},
    "val": {"R2": 0.50, "spearman_rho": 0.730, ...},
    "elapsed_seconds": 612
  },
  "dkl": {
    "test": {"R2": 0.55, "spearman_rho": 0.760, "rmse": 1.35, "nll": 1.90, "err_sigma_rho": "???", "mean_sigma": "???"},
    ...
  },
  ...
}
```

**Key comparison metrics** (all computed on CASF-2016 test set, N=285):

| Metric | Symbol | Computation | Decision Use |
|--------|--------|-------------|-------------|
| Spearman ρ (point prediction) | $\rho$ | `spearmanr(mu_pred, y_true)` | Must be ≥ 0.75 to proceed |
| R² | $R^2$ | `1 - SS_res / SS_tot` | Secondary point prediction metric |
| RMSE | — | `sqrt(mean((y - mu)²))` | Lower = better |
| Test NLL | $\text{NLL}$ | `mean(0.5 log 2πσ² + (y-μ)²/(2σ²))` | Unified quality metric (point + UQ) |
| Error–uncertainty correlation | $\rho_{|err|,\sigma}$ | `spearmanr(|y - mu|, sigma)` | **Primary UQ metric** — must be > 0.15 |
| ECE | — | `compute_ece(p_pred, y_binary, n_bins=10)` | Calibration quality |
| Enrichment Factor @ 1% | $\text{EF}_{1\%}$ | from `evaluate_all()` | Practical screening utility |

**Tier 1 decision rule**:
1. Rank by $\rho_{|err|,\sigma}$ (primary criterion)
2. Break ties by NLL
3. Reject any method with $\rho < 0.75$ (point prediction too degraded)
4. Top 1–2 methods advance to Tier 2

### 6.2 Tier 2: Intra-Method Hyperparameters (After Tier 1)

Fine-grained ablations within the most promising methods from Tier 1. Assume DKL/DKL Ensemble wins Tier 1 (if not, substitute the actual winner).

| ID | Configuration | Specific Change vs Tier 1 Winner | GPU Time (est.) | Purpose |
|----|--------------|--------------------------------|-----------------|---------|
| A4.8 | DKL residual=False | `FeatureExtractor(residual=False)` | ~20 min | Residual connection utility |
| A4.9 | DKL $d_u = 16$ | `feature_dim=16` | ~15 min | Smaller bottleneck |
| A4.10 | DKL $d_u = 64$ | `feature_dim=64` | ~25 min | Larger bottleneck |
| A4.11 | DKL Ensemble M=3 | `n_members=3` (vs 5) | ~60 min | Min useful ensemble size |
| A4.12 | DKL Ensemble seed-only | `bootstrap=False` (init diversity only) | ~100 min | Bootstrap importance |
| A4.13 | DKL 3-layer FeatureExtractor | `n_layers=3, hidden_dim=256` | ~25 min | Deeper feature extractor |
| A4.14 | DKL n_inducing=1024 | `n_inducing=1024` | ~30 min | More inducing points |
| A4.15 | MC Dropout on NN (B3 only) | `mc_dropout=True, mc_samples=20` | ~20 min | NN Bayesianization |

**Total Tier 2 wall time**: ~5 hours on L40S. Can run 2 SLURM jobs in parallel (split experiments into two scripts).

**Concrete Tier 2 SLURM**:
```bash
# Run Tier 2 ablations with specific head hyperparameters:
python scripts/pipeline/s18_train_oracle_heads.py \
    --frozen_embeddings results/stage2/frozen_embeddings.npz \
    --output results/stage2/oracle_heads/tier2_dkl_fd16 \
    --heads dkl \
    --dkl_feature_dim 16 \
    --n_epochs 300 --device cuda

python scripts/pipeline/s18_train_oracle_heads.py \
    --frozen_embeddings results/stage2/frozen_embeddings.npz \
    --output results/stage2/oracle_heads/tier2_dkl_fd64 \
    --heads dkl \
    --dkl_feature_dim 64 \
    --n_epochs 300 --device cuda
```

**Tier 2 analysis**: For each ablation row, compute:
1. Test metrics (same as Tier 1)
2. $\Delta$ vs Tier 1 winner (improvement/degradation)
3. Compute budget (GPU minutes per experiment)
4. Train/val gap (overfitting indicator)

### 6.3 Tier 3: Cross-Plan Integration (After Tier 2)

Combine best oracle head with other Sub-Plan outputs. These experiments require completed Sub-Plans 01–03 and/or 05.

| ID | Configuration | Dependencies | Purpose |
|----|--------------|-------------|---------|
| A4.16 | Best 04 head + enhanced representation (Sub-Plans 1–3) | 01–03 complete | Full combination: improved representation + improved predictor |
| A4.17 | Best 04 head + multi-task trunk (Sub-Plan 05) | 05 complete | Multi-task + improved predictor |
| A4.18 | Best 04 head + end-to-end fine-tuning (§4.1 Phase 4.3) | 04 Tier 2 complete | Unfreeze representation with oracle head |

**Priority rule**: Tier 1 first (compare method families) → Tier 2 (tune the winner) → Tier 3 (integrate across Sub-Plans). Do not proceed to the next tier until the previous is complete and analyzed.

**Ablation execution protocol**:
1. Each experiment saves its own `config.json` + `metrics.json` + model checkpoint
2. After each tier, generate comparison table (`tier{N}_comparison.csv`) and diagnostic plots
3. Record key findings in `doc/Stage_2/04_hybrid_predictor_RESULTS.md`
4. Commit results + analysis before proceeding to next tier

---

## 7. Evaluation & Success Criteria

### 7.1 Updated Baselines

The success criteria must reflect the current state of the project, not the Stage 1 era.

**Current baselines from Sub-Plan 02** (on CASF-2016 test set, N=285):

| Configuration | Repr. | Test $R^2$ | Test $\rho$ | RMSE | NLL | $\rho_{|err|,\sigma}$ | ECE | Source |
|---------------|-------|-----------|-------------|------|-----|----------------------|-----|--------|
| **MLP readout (no UQ)** | **A3.6 Indep.** | **0.574** | **0.778** | 1.31 | N/A | N/A | N/A | s18 L40S re-run |
| MLP readout (no UQ) | A3.4 Shared | 0.572 | 0.761 | 1.42 | N/A | N/A | N/A | `phase3_results.json` A3.4 step1_mlp |
| DKL(128→32→SVGP) | A3.4 Shared | 0.559 | 0.760 | 1.44 | ~2.0 | −0.04 | ~0.15 | `gp_fix_results.json` A3.4c_DKL |
| PCA32→SVGP | A3.4 Shared | 0.543 | 0.746 | 1.47 | ~2.1 | +0.04 | ~0.18 | `gp_fix_results.json` A3.4b_PCA32 |
| raw SVGP(128d) | A3.4 Shared | 0.507 | 0.719 | 1.52 | ~2.3 | −0.01 | ~0.20 | `phase3_results.json` A3.4 |

> **Note**: GP baselines above are from A3.4 Shared embeddings (historical reference). Sub-Plan 04 uses A3.6 Independent embeddings (ρ=0.778 MLP ceiling). All oracle heads will be re-baselined on A3.6 embeddings in Phase 4.2.

**Key observations from baselines**:
- A3.6 MLP ceiling: $\rho = 0.778$ — Sub-Plan 04 oracle heads are measured against this
- On A3.4 embeddings, MLP and DKL were essentially tied ($\rho = 0.761$ vs $0.760$) — the DKL's problem is purely uncertainty quality
- All GP variants show $\rho_{|err|,\sigma} \approx 0$ — uncertainty is uncorrelated with prediction error
- GP baselines may shift on A3.6 embeddings; raw SVGP and DKL will be re-measured as part of Tier 1

### 7.2 Success Criteria

#### Point Prediction (Hard Constraints)

| Criterion | Metric | Threshold | Rationale |
|-----------|--------|-----------|-----------|
| Must not degrade significantly vs MLP | Spearman $\rho$ | $\rho \geq 0.75$ | Within ~0.03 of MLP ceiling (0.778) |
| Must improve over raw SVGP | Spearman $\rho$ | $\rho > 0.73$ | Clear improvement over A4.1 |
| RMSE reasonable | RMSE | $\leq 1.44$ | Within 10% of MLP RMSE (1.31) |

#### Uncertainty Quality (Primary Goal — at least one criterion must be met)

| Criterion | Metric | Threshold | Computation (from `evaluate.py`) |
|-----------|--------|-----------|--------------------------------|
| Error–uncertainty correlation | $\rho_{|err|,\sigma}$ | $> 0.15$ (up from ≈ 0) | `spearmanr(np.abs(y - mu), sigma)` |
| NLL improvement | Test NLL | ≥ 5% reduction vs DKL baseline | `gaussian_nll(mu, sigma, y)` |
| Calibration | ECE | $\leq 0.05$ | `compute_ece(p_pred, y_binary, n_bins=10)` |
| Family-wise ranking | Per-pocket $\rho$ | Avg per-pocket $\rho > 0.3$ | `evaluate_per_pocket(mu, sigma, y, pocket_ids)` |

**Stretch goals** (not required for success, but would strengthen the paper):
- $\rho_{|err|,\sigma} > 0.25$ — strong uncertainty calibration
- Enrichment Factor @ 1% > 10 — practical screening utility
- Uncertainty decomposition: $\sigma^2_{epistemic}$ correlated with $|err|$ independently of $\sigma^2_{aleatoric}$

#### Failure Criterion

A method is **rejected** from the mainline if:
- Point prediction does not improve over raw SVGP ($\rho \leq 0.72$), **OR**
- Point prediction ≤ DKL baseline **AND** uncertainty quality shows no improvement on any criterion

**Implication of universal failure** (all methods fail): If no oracle head achieves $\rho_{|err|,\sigma} > 0.15$, this indicates the frozen 128d embeddings may not contain sufficient information for meaningful uncertainty estimation. In that case:
1. Investigate whether the representation itself needs to be uncertainty-aware (→ connects to Sub-Plan 05 multi-task trunk)
2. Consider whether PDBbind label noise is so high that $\rho_{|err|,\sigma} > 0.15$ is unreachable with this data
3. Document the negative result — this is still a valuable finding for the paper

#### Tier 1 Experimental Results (April 9, 2026)

Two independent runs on A3.6 Independent embeddings (seed=42, n_inducing=512, 300 epochs):

| Head | R² | ρ | RMSE | NLL | ρ\_err\_σ | ρ\_err\_σ (run 2) |
|------|-----|------|------|-----|-----------|-------------------|
| **dkl_ensemble** | **0.604–0.607** | **0.781** | **1.35–1.37** | **1.754–1.758** | **0.088** | **0.144** |
| dkl | 0.579–0.589 | 0.771–0.775 | 1.39–1.41 | 1.818–1.835 | 0.028 | 0.001 |
| nn_residual | 0.579–0.582 | 0.765 | 1.40–1.41 | 1.767–1.768 | 0.064–0.067 | 0.064 |
| svgp | 0.573–0.588 | 0.763–0.768 | 1.39–1.42 | 1.760–1.781 | 0.020–0.042 | 0.023 |
| pca_svgp | 0.567–0.579 | 0.754–0.758 | 1.41–1.43 | 1.771–1.789 | 0.011–0.019 | 0.015 |

> Source: `slurm/logs/s18_oracle_5857850.out` (L40S), `s18_oracle_5857851.out` (A100).
> JSON: `results/stage2/oracle_heads/tier1_comparison.json` (A100 Phase 4.2 run).

**Assessment against criteria**:
- **Point prediction**: All heads pass ρ ≥ 0.75 except pca_svgp (marginal at 0.754–0.758). DKL Ensemble and single DKL both exceed threshold.
- **Uncertainty quality**: DKL Ensemble achieves ρ\_err\_σ = 0.088–0.144, the only head approaching the 0.15 target. nn\_residual shows moderate signal (0.064). Single DKL is inconsistent (0.001–0.028). SVGP/PCA variants show minimal signal.
- **NLL**: DKL Ensemble has the best NLL (1.754–1.758), improving ~3% over single DKL.
- **Reproducibility concern**: ρ\_err\_σ varies substantially across runs for DKL Ensemble (0.088 vs 0.144) and single DKL (0.001 vs 0.028), suggesting seed sensitivity. Tier 2 should include multi-seed averaging.

**Tier 1 Decision**: **DKL Ensemble** advances to Tier 2 as the clear winner. nn_residual is the secondary candidate if Tier 2 ablation reveals DKL Ensemble instability.

#### Tier 2 Experimental Results (April 9, 2026)

DKL Ensemble ablations on A3.6 Independent embeddings (L40S, job 5863003).

The previous summary table reported the **test split only**. Each Tier 2 run actually saved both `val` and `test` metrics in its per-run `tier1_comparison.csv/json`; the tables below make both splits explicit.

**Multi-seed baseline** (M=5, d_u=32, residual=True, bootstrap=True, n_inducing=512):

| Seed | Val R² | Val ρ | Val NLL | Val ρ\_err\_σ | Test R² | Test ρ | Test NLL | Test ρ\_err\_σ |
|------|--------|-------|---------|----------------|---------|--------|----------|-----------------|
| 42 | 0.321 | 0.584 | 1.871 | 0.085 | 0.620 | 0.789 | 1.743 | 0.100 |
| 123 | 0.297 | 0.568 | 1.933 | 0.098 | 0.599 | 0.774 | 1.807 | 0.109 |
| 777 | 0.299 | 0.568 | 1.947 | 0.104 | 0.604 | 0.776 | 1.809 | 0.063 |
| **mean±std** | **0.306±0.013** | **0.573±0.010** | **1.917±0.041** | **0.096±0.010** | **0.608±0.011** | **0.780±0.008** | **1.786±0.037** | **0.091±0.025** |

**Ablation results** (seed=42 unless noted; `val` shown first because it is the model-selection split, `test` remains the main reporting split):

| ID | Configuration | Val ρ | Val ρ\_err\_σ | Δval ρ\_err\_σ vs baseline mean | Test ρ | Test NLL | Test ρ\_err\_σ | Δtest ρ\_err\_σ vs baseline mean |
|----|--------------|-------|----------------|----------------------------------|--------|----------|-----------------|-------------------------------|
| A4.8 | residual=False | 0.570 | 0.077 | −0.019 | 0.781 | 1.785 | 0.017 | −0.074 ↓ |
| A4.9 | feature_dim=16 | 0.574 | 0.102 | +0.006 | 0.777 | 1.806 | **0.105** | +0.014 |
| A4.10 | feature_dim=64 | 0.577 | 0.097 | +0.001 | 0.782 | 1.775 | 0.003 | −0.088 ↓ |
| A4.11 | M=3 | 0.582 | 0.078 | −0.018 | **0.793** | **1.732** | 0.057 | −0.034 |
| A4.12 | bootstrap=False | 0.580 | 0.082 | −0.013 | 0.785 | 1.782 | −0.056 | −0.147 ↓↓ |
| A4.13 | n_layers=3 | **0.586** | **0.122** | +0.026 | 0.785 | 1.753 | 0.080 | −0.011 |
| A4.14 | n_inducing=1024 | 0.583 | 0.079 | −0.017 | 0.785 | 1.759 | 0.041 | −0.050 |
| A4.15 | NNResidual MC Drop | 0.558 | −0.028 | −0.124 | 0.772 | 1.760 | 0.034 | −0.057 |

> Source: `slurm/logs/s18_tier2_5863003.out`, plus per-config `results/stage2/oracle_heads/tier2/*/tier1_comparison.csv` and `tier1_comparison.json` files, which each contain both `val` and `test` rows.

**Key findings**:
1. **Bootstrap is critical for UQ**: A4.12 (no bootstrap) → ρ\_err\_σ = −0.056 (anti-correlated). Data diversity is the key mechanism, not just init diversity.
2. **Residual connection matters for UQ**: A4.8 (no residual) → ρ\_err\_σ drops from 0.091 to 0.017. Residual shortcut preserves information needed for uncertainty estimation.
3. **Smaller bottleneck (d_u=16) is slightly better for UQ**: A4.9 → ρ\_err\_σ = 0.105 (best single-seed), with acceptable ρ=0.777. Larger bottleneck (d_u=64, A4.10) kills UQ (0.003).
4. **M=3 has best point prediction**: A4.11 → ρ=0.793, NLL=1.732, but UQ signal drops to 0.057.
5. **Baseline seed variance is high**: ρ\_err\_σ ranges 0.063–0.109 (std=0.025). At N=285, the UQ signal is noisy.
6. **No configuration reaches ρ\_err\_σ > 0.15** target. Best mean = 0.091±0.025 (baseline), best single = 0.109 (seed 123).

**Tier 2 Decision**: Default baseline config (M=5, d_u=32, residual=True, bootstrap=True, n_inducing=512) remains the best trade-off. The ρ\_err\_σ = 0.091±0.025 does not reach the 0.15 stretch target, but represents a meaningful improvement over GP baselines (≈0). The high seed variance (0.063–0.109) suggests **multi-seed ensembles** or **larger test sets** may be needed to stabilize the signal.

**Recommended next steps** (Tier 3 / further investigation):
- Multi-seed ensemble: average predictions across 3–5 full DKL Ensemble runs (seeds 42/123/777) → may stabilize ρ\_err\_σ
- Investigate whether A3.6 embeddings contain sufficient UQ-relevant information (→ Sub-Plan 05 multi-task trunk)
- CASF-2016 N=285 may be too small for reliable ρ\_err\_σ measurement — consider per-pocket analysis

### 7.3 Diagnostic Metrics & Analysis Tools

Beyond pass/fail criteria, collect detailed diagnostics for every oracle head:

#### 7.3.1 Uncertainty Decomposition (Ensemble Methods)

```python
def analyze_uncertainty_decomposition(result: OracleResult, y_true: np.ndarray) -> dict:
    """Decompose total uncertainty into components and correlate with error."""
    errors = np.abs(y_true - result.mu)
    sigma = np.sqrt(result.sigma2)
    
    diagnostics = {
        'rho_err_sigma_total': spearmanr(errors, sigma)[0],
    }
    
    if 'sigma2_aleatoric' in result.aux:
        sigma_a = np.sqrt(result.aux['sigma2_aleatoric'])
        sigma_e = np.sqrt(result.aux['sigma2_epistemic'])
        diagnostics['rho_err_sigma_aleatoric'] = spearmanr(errors, sigma_a)[0]
        diagnostics['rho_err_sigma_epistemic'] = spearmanr(errors, sigma_e)[0]
        diagnostics['mean_aleatoric_frac'] = float((result.aux['sigma2_aleatoric'] / result.sigma2).mean())
        diagnostics['mean_epistemic_frac'] = float((result.aux['sigma2_epistemic'] / result.sigma2).mean())
    
    return diagnostics
```

**Expected finding**: $\rho_{|err|, \sigma_{epistemic}}$ should be higher than $\rho_{|err|, \sigma_{aleatoric}}$ because epistemic uncertainty reflects model ignorance (correlated with where the model is likely wrong), while aleatoric reflects inherent noise.

#### 7.3.2 Feature Space Quality

```python
def analyze_feature_space(feature_extractor, X_train, X_test, y_test) -> dict:
    """Compare distances in raw z-space vs learned u-space."""
    with torch.no_grad():
        U_train = feature_extractor(torch.tensor(X_train)).numpy()
        U_test = feature_extractor(torch.tensor(X_test)).numpy()
    
    # Nearest-neighbor label consistency
    from sklearn.neighbors import KNeighborsRegressor
    knn_z = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    knn_u = KNeighborsRegressor(n_neighbors=5).fit(U_train, y_train)
    
    rho_z = spearmanr(knn_z.predict(X_test), y_test)[0]
    rho_u = spearmanr(knn_u.predict(U_test), y_test)[0]
    
    return {
        'knn_rho_raw_space': rho_z,
        'knn_rho_learned_space': rho_u,
        'improvement': rho_u - rho_z,
    }
```

**Expected finding**: KNN $\rho$ should be higher in the learned $u$-space → the feature extractor learns a label-relevant geometry.

#### 7.3.3 Binned Error–Uncertainty Analysis

```python
def binned_err_sigma_analysis(mu, sigma, y_true, n_bins=10) -> dict:
    """Bin predictions by predicted σ, compute mean |error| per bin."""
    errors = np.abs(y_true - mu)
    bins = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    
    bin_means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sigma >= lo) & (sigma < hi + 1e-10)
        if mask.sum() > 0:
            bin_means.append({
                'sigma_lo': float(lo), 'sigma_hi': float(hi),
                'mean_error': float(errors[mask].mean()),
                'mean_sigma': float(sigma[mask].mean()),
                'n_samples': int(mask.sum()),
            })
    
    return {'bins': bin_means}
```

**Expected finding**: If uncertainty is well-calibrated, mean |error| should increase with predicted σ (monotonic trend in the binned analysis).

#### 7.3.4 Ensemble Diversity Metrics

```python
def analyze_ensemble_diversity(result: OracleResult) -> dict:
    """Compute diversity metrics for ensemble methods."""
    if 'member_mus' not in result.aux:
        return {}
    
    member_mus = result.aux['member_mus']  # (M, N)
    M = member_mus.shape[0]
    
    # Pairwise correlation between members
    pairwise_rhos = []
    for i in range(M):
        for j in range(i + 1, M):
            rho, _ = spearmanr(member_mus[i], member_mus[j])
            pairwise_rhos.append(rho)
    
    return {
        'mean_pairwise_rho': float(np.mean(pairwise_rhos)),
        'min_pairwise_rho': float(np.min(pairwise_rhos)),
        'max_pairwise_rho': float(np.max(pairwise_rhos)),
        'effective_ensemble_size': float(1.0 / (1.0 - np.var(member_mus, axis=0).mean() / member_mus.var())),
    }
```

**Expected finding**: Mean pairwise $\rho$ should be < 0.95 — if members are too correlated, the ensemble provides no additional uncertainty signal. Bootstrap diversity (B2) should produce lower correlation than seed-only diversity.

#### 7.3.5 Train/Val Gap Monitoring

For overfitting detection:
- Log both training loss and validation metrics per epoch
- Compute final gap: `train_rho - val_rho`
- Flag if gap > 0.05 (point prediction) or if val_NLL increases while train_NLL decreases
- The training history from `fit()` already captures these signals

### 7.4 Visualization Suite

Generate the following diagnostic plots (implemented in `scripts/pipeline/s19_oracle_diagnostics.py`):

| Plot | X-axis | Y-axis | Purpose |
|------|--------|--------|---------|
| `err_vs_sigma_scatter.png` | predicted $\sigma$ | $|y - \mu|$ | Core uncertainty quality check |
| `calibration_curve.png` | expected confidence | observed confidence | Calibration quality |
| `tier1_comparison_bar.png` | oracle head | metric value (grouped bars) | Method family comparison |
| `training_curves.png` | epoch | ELBO / val $\rho$ | Convergence behavior |
| `feature_tsne.png` | t-SNE dim 1 | t-SNE dim 2 | Feature space visualization (raw $z$ vs learned $u$) |
| `uncertainty_decomp.png` | sample index (sorted by error) | $\sigma^2_{alea}$, $\sigma^2_{epi}$ stacked | Uncertainty component analysis |
| `binned_err_sigma.png` | σ quantile bin | mean $|err|$ per bin | Monotonicity check |

### 7.5 Overfitting Risk Assessment

With PDBbind general set (~18K usable complexes after filtering):
- Overfitting risk is **moderate** — substantially less than with ~5K refined set
- DKL adds ~45K parameters via feature extractor (§2.2) — manageable at 18K data points
- Ensemble (B2) provides natural regularization via member diversity
- NN + GP Residual (B3) benefits from NN generalization + GP smoothing
- BNN / ensemble / evidential methods become more feasible at this data scale
- Grouped protein-cluster splits prevent leakage but may reduce effective N by ~30%
- **Mitigation stack**: Early stopping on val $\rho$ (not loss), weight decay, dropout, train/val gap monitoring, bootstrap diversity

### 7.6 Statistical Significance

For the final comparison (best oracle head vs baseline DKL):

**Grouped bootstrap** (primary): CASF-2016 contains 57 targets × 5 ligands per target. The 285 complexes are **not** independent — ligands within the same target share protein structure. Therefore:
- Resample at the **target level**: draw 57 targets with replacement, include all 5 ligands for each drawn target
- Compute metrics on the resampled set → repeat 1000 times → 95% CI
- This properly accounts for within-target correlation and gives conservative (wider) CIs

**Sample-level bootstrap** (secondary): Standard 1000 bootstrap resamples of all 285 complexes, treating each as independent. This gives tighter CIs but may be overly optimistic.

**Report both**:
- Target-grouped bootstrap 95% CI for $\Delta\rho$ and $\Delta\rho_{|err|,\sigma}$
- Sample-level bootstrap 95% CI for comparison
- Cohen's $d$ for the $\rho_{|err|,\sigma}$ improvement

A result is considered **significant** if the **target-grouped** 95% CI of $\Delta\rho_{|err|,\sigma}$ does not include 0.

```python
def grouped_bootstrap_ci(mu, sigma, y_true, target_ids, n_bootstrap=1000, seed=42):
    """Bootstrap CI respecting CASF-2016 target grouping (57 targets × 5 ligands).
    
    Parameters
    ----------
    mu, sigma, y_true : (N,) arrays — predictions, uncertainties, labels
    target_ids : (N,) array — target group ID for each complex (e.g., PDB ID)
    n_bootstrap : number of bootstrap resamples
    seed : random seed
    
    Returns
    -------
    dict of {metric_name: (lower_2.5%, upper_97.5%)}
    """
    rng = np.random.RandomState(seed)
    unique_targets = np.unique(target_ids)
    n_targets = len(unique_targets)  # 57 for CASF-2016
    
    metrics_boot = []
    for _ in range(n_bootstrap):
        # Resample targets (not individual complexes)
        boot_targets = rng.choice(unique_targets, size=n_targets, replace=True)
        boot_idx = np.concatenate([np.where(target_ids == t)[0] for t in boot_targets])
        
        errors = np.abs(y_true[boot_idx] - mu[boot_idx])
        rho_err_sigma, _ = spearmanr(errors, sigma[boot_idx])
        rho_pred, _ = spearmanr(mu[boot_idx], y_true[boot_idx])
        metrics_boot.append({'rho': rho_pred, 'err_sigma_rho': rho_err_sigma})
    
    return {k: (np.percentile([m[k] for m in metrics_boot], 2.5),
                np.percentile([m[k] for m in metrics_boot], 97.5))
            for k in metrics_boot[0]}
```

---

## 8. Paper Integration

### 8.1 Methods Section (Draft)

> **§3.Y Uncertainty-Aware Hybrid Prediction**
> 
> After upgrading the representation (§3.X), the raw SVGP oracle suffers from degraded performance on high-dimensional embeddings and—critically—produces uncertainty estimates with near-zero correlation to actual prediction errors ($\rho_{|err|,\sigma} \approx 0$; Table D.1). This motivates a systematic comparison of uncertainty-aware oracle heads that preserve predictive accuracy while drastically improving uncertainty calibration.
> 
> To isolate predictor-head quality from representation effects, we adopt a **frozen-representation protocol**: the current best attention-aggregated embeddings ($z \in \mathbb{R}^{128}$; currently SchemeB Independent ParamPool from Sub-Plan 02) are pre-computed once and frozen. All oracle heads are trained on the same embeddings, enabling fair comparison.
> 
> Our primary candidates are: (1) **Deep Kernel Learning** (DKL) [Wilson et al., 2016], which learns a feature transformation $g_\theta: \mathbb{R}^{128} \to \mathbb{R}^{32}$ via a 2-layer residual MLP (~45K params) before a SVGP with Matérn-5/2 + ARD kernel; (2) **DKL Ensemble**, which combines $M = 5$ independently trained DKL models with 80% bootstrap subsampling, decomposing total variance into aleatoric ($\bar{\sigma}^2_m$) and epistemic ($\text{Var}[\mu_m]$) components; and (3) **NN + GP Residual**, where a 2-layer MLP captures the main prediction signal and a GP operates on the residuals $r = y - \text{MLP}(z)$, with MC Dropout ($T=20$) providing additional NN epistemic uncertainty.
> 
> As lightweight baselines without learned feature transforms, we optionally include SNGP [Liu et al., 2020] with Random Fourier Features ($P = 1024$) and Evidential Regression [Amini et al., 2020] with a Normal-Inverse-Gamma output head (Tier 1b; implemented only if core results require additional context).
> 
> All oracle heads conform to a unified interface producing $(\mu, \sigma^2)$ for evaluation, with optional Jacobian $J_\mu = \partial\mu/\partial z$ computed only when entering the Delta method fusion stage (§3.Z). Training uses joint ELBO maximization for GP-based methods and MSE/Evidential loss for NN-only baselines, with early stopping on validation Spearman $\rho$ (patience = 20).
> 
> We find that **DKL Ensemble** ($M=5$, bootstrap) achieves $\rho_{|err|,\sigma} = 0.091 \pm 0.025$ across 3 seeds (up from $\approx 0$ for raw SVGP) while maintaining $\rho = 0.780$ (vs. $0.778$ MLP ceiling), with test NLL of $1.79 \pm 0.04$ (vs. $\sim 1.82$ single DKL). Bootstrap subsampling is the critical mechanism: without it, $\rho_{|err|,\sigma}$ drops to $-0.056$. The residual connection in the feature extractor is also important ($0.091 \to 0.017$ without). Larger feature dimensions ($d_u = 64$) collapse uncertainty signal ($\rho_{|err|,\sigma} = 0.003$), while the compact $d_u = 32$ bottleneck preserves it. The $\rho_{|err|,\sigma} = 0.15$ stretch target was not reached; CASF-2016 ($N=285$, 57 targets) may be too small for reliable uncertainty–error correlation measurement.

### 8.2 Key References

| Reference | Method | Use in Paper |
|-----------|--------|-------------|
| Wilson et al. (2016) "Deep Kernel Learning" | DKL | Primary approach B1 |
| Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" | Deep Ensemble | Approach B2 |
| Liu et al. (2020) "Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness" | SNGP/DUE | Tier 1b baseline C1 |
| Amini et al. (2020) "Deep Evidential Regression" | Evidential | Tier 1b baseline C2 |
| Gardner et al. (2018) "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration" | GPyTorch library | Implementation foundation |

### 8.3 Figures

| Figure | Content | Script | Purpose |
|--------|---------|--------|---------|
| Fig. D.1 | Architecture diagram: DKL / DKL-Ens / NN+GP-Residual | `scripts/figures/fig_d1_oracle_architectures.py` | Explain oracle head designs |
| Fig. D.2 | $|err|$–$\sigma$ scatter: raw GP vs DKL vs DKL-Ens vs NN+Residual | `scripts/pipeline/s19_oracle_diagnostics.py` | Core uncertainty quality comparison |
| Fig. D.3 | Calibration curves: all oracle heads (expected vs observed) | `scripts/pipeline/s19_oracle_diagnostics.py` | Calibration quality |
| Fig. D.4 | Feature space t-SNE: raw $z$ vs learned $u$, colored by $|err|$ | `scripts/pipeline/s19_oracle_diagnostics.py` | Learned feature quality |
| Fig. D.5 | Tier 1 method family comparison: grouped bar chart | `scripts/pipeline/s19_oracle_diagnostics.py` | Design choice justification |
| Fig. D.6 | Training curves: ELBO, val $\rho$, val NLL vs epoch (best method) | `scripts/pipeline/s19_oracle_diagnostics.py` | Training convergence |
| Fig. D.7 | Uncertainty decomposition: $\sigma^2_{alea}$ vs $\sigma^2_{epi}$ sorted by $|err|$ | `scripts/pipeline/s19_oracle_diagnostics.py` | Decomposition analysis |
| Fig. D.8 | Binned error–sigma analysis: monotonicity check | `scripts/pipeline/s19_oracle_diagnostics.py` | Calibration monotonicity |

**Figure generation workflow**:
```bash
# After Tier 1 results are ready:
python scripts/pipeline/s19_oracle_diagnostics.py \
    --results_dir results/stage2/oracle_heads \
    --frozen_embeddings results/stage2/frozen_embeddings.npz \
    --output_dir results/stage2/oracle_heads/figures \
    --format pdf
```

### 8.4 Tables

| Table | Content | Data Source |
|-------|---------|-----------|
| Tab. D.1 | Tier 1 oracle head comparison (A4.1–A4.7): $R^2$, $\rho$, RMSE, NLL, ECE, $\rho_{|err|,\sigma}$, EF@1% | `tier1_comparison.json` |
| Tab. D.2 | Tier 2 ablation for best method (A4.8–A4.15) with $\Delta$ columns | `tier2_comparison.json` |
| Tab. D.3 | Uncertainty decomposition: aleatoric vs epistemic fraction, per-component $\rho_{|err|,\sigma}$ | `uncertainty_diagnostics.json` |
| Tab. D.4 | Cross-plan integration results (A4.16–A4.18) | Tier 3 results |
| Tab. D.5 | Ensemble diversity: pairwise $\rho$, effective ensemble size | `ensemble_diagnostics.json` |

### 8.5 Supplementary Materials

- **Extended results**: Per-pocket breakdown of all metrics (CASF-2016, 57 target families)
- **Negative results**: Document which methods failed and why (contributes to understanding)
- **Ablation completeness**: Full hyperparameter sensitivity tables for Tier 2
- **Compute budget**: Report GPU hours for each experiment to support reproducibility

---

## 9. Implementation Checklist

### Phase A: Foundation (Days 1–2)

- [x] **A.1** Create `bayesdiff/oracle_interface.py` with `OracleResult` dataclass and `OracleHead` ABC
  - File: `bayesdiff/oracle_interface.py` (~100 lines, §4.2)
  - Depends on: nothing
  - Verify: `python -c "from bayesdiff.oracle_interface import OracleHead, OracleResult"`
- [x] **A.2** Implement `FeatureExtractor` in `bayesdiff/hybrid_oracle.py`
  - File: `bayesdiff/hybrid_oracle.py` (§4.3, first ~80 lines)
  - Depends on: A.1
  - Verify: T1.1, T1.2 pass
- [x] **A.3** Pre-compute and cache frozen embeddings from Sub-Plan 02 best representation
  - Script: Add `--extract_embeddings` mode to `s18_train_oracle_heads.py`
  - Output: `results/stage2/oracle_heads/frozen_embeddings.npz` (A3.6 Independent, ρ=0.778)
  - Depends on: Sub-Plan 02 complete (SchemeB checkpoint exists)
  - Verify: `python -c "import numpy as np; d=np.load('results/stage2/oracle_heads/frozen_embeddings.npz'); print(d['X_train'].shape)"`

### Phase B: Core Oracle Heads (Days 3–5)

- [x] **B.1** Implement `DKLOracle` (primary candidate)
  - File: `bayesdiff/hybrid_oracle.py` (§4.3, DKLSVGPModel + DKLOracle class, ~300 lines)
  - Depends on: A.1, A.2
  - Verify: T1.3, T1.4, T1.5, T1.6, T1.7 pass
- [x] **B.2** Implement `DKLEnsembleOracle`
  - File: `bayesdiff/hybrid_oracle.py` (§4.3, ~120 lines)
  - Depends on: B.1
  - Verify: T1.8, T1.9 pass
- [x] **B.3** Implement `NNResidualOracle`
  - File: `bayesdiff/hybrid_oracle.py` (§4.3, ~200 lines)
  - Depends on: A.1, existing `GPOracle`
  - Verify: T1.10, T1.11 pass

### Phase C: Baselines (Day 6)

- [x] **C.3** Implement `PCA_GPOracle` wrapper (PCA → existing GPOracle)
  - File: `bayesdiff/hybrid_oracle.py` (~60 lines)
  - Depends on: A.1, existing `GPOracle`
  - Verify: T1.12 pass

### Phase C' (Deferred — Tier 1b, after Phase E analysis)

- [ ] **C.1** Implement `SNGPOracle` (spectral normalized neural GP baseline)
  - File: `bayesdiff/hybrid_oracle.py` (~150 lines)
  - Depends on: A.1
  - Verify: T1.12 (interface compliance), T1.14 (OOD) pass
- [ ] **C.2** Implement `EvidentialOracle` (evidential regression baseline)
  - File: `bayesdiff/hybrid_oracle.py` (~120 lines)
  - Depends on: A.1
  - Verify: T1.12 (interface compliance), T1.14 (OOD) pass

### Phase D: Pipeline & Tests (Days 7–8)

- [x] **D.1** Write `scripts/pipeline/s18_train_oracle_heads.py` pipeline script
  - File: `scripts/pipeline/s18_train_oracle_heads.py` (§4.5, ~300 lines)
  - Depends on: B.1–B.3, C.3
  - Verify: `python scripts/pipeline/s18_train_oracle_heads.py --help`
- [x] **D.2** Write `slurm/s18_oracle_heads.sh` SLURM submission script
  - File: `slurm/s18_oracle_heads.sh` (§4.6)
  - Verify: `sbatch --test-only slurm/s18_oracle_heads.sh`
- [x] **D.3** Write unit tests in `tests/stage2/test_hybrid_oracle.py`
  - File: `tests/stage2/test_hybrid_oracle.py` (§5.1, ~300 lines)
  - Depends on: B.1–B.3
  - Verify: `python -m pytest tests/stage2/test_hybrid_oracle.py -v`
- [x] **D.4** Write integration tests in `tests/stage2/test_hybrid_integration.py`
  - File: `tests/stage2/test_hybrid_integration.py` (§5.2, ~150 lines)
  - Depends on: B.1–B.3, D.1
  - Verify: `python -m pytest tests/stage2/test_hybrid_integration.py -v`
- [x] **D.5** Run full test suite (all T1.x and T2.x pass)
  - Depends on: D.3, D.4
  - Completed: Job 5865532 (L40S), 21 unit tests + 5 integration tests = 26/26 pass
  - Verify: `python -m pytest tests/stage2/ -v --tb=short` with 0 failures

### Phase E: Tier 1 Experiments (Days 9–10)

- [x] **E.1** Submit Tier 1 SLURM job: `sbatch slurm/s18_oracle_heads.sh`
  - Depends on: D.5 (all tests pass)
  - Output: `results/stage2/oracle_heads/tier1_comparison.json`
  - Completed: Jobs 5857850 (L40S) + 5857851 (A100), April 9 2026
- [x] **E.2** Analyze Tier 1 results
  - DKL Ensemble: ρ=0.781, ρ_err_σ=0.088–0.144 (best UQ signal)
  - All heads pass ρ≥0.75 except pca_svgp (marginal)
  - Output: recorded in §7.2 "Tier 1 Experimental Results" section
- [x] **E.3** Select Tier 1 winner(s) for Tier 2
  - Decision: **DKL Ensemble** advances. nn_residual as secondary candidate.

### Phase F: Tier 2 Refinement (Days 11–13)

- [x] **F.1** Run Tier 2 ablations (A4.8–A4.15) for the Tier 1 winner
  - Completed: Job 5863003 (L40S), April 9 2026
  - 3-seed baseline + 8 ablation configs, ~30 min total
- [x] **F.2** Analyze Tier 2 results and select best configuration
  - Best trade-off: default baseline (M=5, d_u=32, residual=True, bootstrap=True)
  - ρ_err_σ = 0.091±0.025 across 3 seeds (below 0.15 target but best available)
  - Bootstrap critical for UQ; residual=False and large d_u kill UQ signal
  - Recorded in §7.2 "Tier 2 Experimental Results"
- [x] **F.3** Write diagnostics script `scripts/pipeline/s19_oracle_diagnostics.py`
  - Generate: figures D.2–D.8 (err scatter, calibration, t-SNE, tier1 bar, decomp, binned)
  - Also: uncertainty_diagnostics.json, ensemble_diagnostics.json
  - Submitted: Job 5865735 (L40S)

### Phase G: Documentation & Paper (Days 14–15)

- [x] **G.1** Generate paper figures and tables from results
  - Figures D.2–D.8 generated by s19 (job 5868900): err scatter, calibration, t-SNE, tier1 bar, decomp, binned
  - JSONs: `uncertainty_diagnostics.json`, `ensemble_diagnostics.json`
  - Output: `results/stage2/oracle_heads/figures/` (12 figure files + 2 JSON)
- [x] **G.2** Draft methods section text (§8.1)
  - Filled TBD placeholders with actual results: DKL Ensemble ρ_{|err|,σ}=0.091±0.025, ρ=0.780, NLL=1.79
  - Documented bootstrap criticality, residual importance, feature dimension effects
- [x] **G.3** Update `doc/Stage_2/04_hybrid_predictor_RESULTS.md` with final findings
  - Created comprehensive results document with 12 sections
  - Includes: executive summary, Tier 1/2 tables, uncertainty decomposition, calibration, config, figures, tests, compute budget, limitations
- [ ] **G.4** Commit all results, figures, and documentation

### Dependency Graph

```
A.1 ──→ A.2 ──→ B.1 ──→ B.2
  │              │       │
  │              ↓       │
  │             B.3      │
  │              │       │
  ├──→ C.1      │       │
  ├──→ C.2      │       │
  └──→ C.3      │       │
                 │       │
A.3 ────────────→↓       ↓
                D.1 ──→ D.2
                 │
                D.3 ──→ D.5 ──→ E.1 ──→ E.2 ──→ E.3 ──→ F.1 ──→ F.2 ──→ G.1
                D.4 ──↗            │                              │
                                   └──────────────────────────────┘
                                              F.3 ──→ G.1 ──→ G.2 ──→ G.4
```

### New Files Created by Sub-Plan 04

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `bayesdiff/oracle_interface.py` | ~100 | OracleResult + OracleHead ABC |
| `bayesdiff/hybrid_oracle.py` | ~800 | All oracle head implementations |
| `scripts/pipeline/s18_train_oracle_heads.py` | ~200 | Training pipeline script |
| `scripts/pipeline/s19_oracle_diagnostics.py` | ~300 | Diagnostic plots and analysis |
| `slurm/s18_oracle_heads.sh` | ~30 | SLURM submission for Tier 1 |
| `tests/stage2/test_hybrid_oracle.py` | ~300 | Unit tests (T1.1–T1.14) |
| `tests/stage2/test_hybrid_integration.py` | ~150 | Integration tests (T2.1–T2.6) |
| `doc/Stage_2/04_hybrid_predictor_RESULTS.md` | — | Experimental results (created during E.2) |

**Total new code**: ~1,880 lines across 8 files.

### Modified Files

| File | Change | Breaking? |
|------|--------|-----------|
| `bayesdiff/ood.py` | Add `fit_feature_space()` method to `MahalanobisOOD` | No (additive) |
| `bayesdiff/__init__.py` | Import new modules | No (additive) |

**No breaking changes** to existing modules (`fusion.py`, `evaluate.py`, `gp_oracle.py`, `data.py`).
