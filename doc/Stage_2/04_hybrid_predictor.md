# Sub-Plan 4: Hybrid Predictor (Deep Kernel Learning)

> **Priority**: P1 — High  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); Sub-Plans 1–3 (benefits from richer representations)  
> **Training Data**: PDBbind v2020 refined set (~5,316 complexes), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 2–3 weeks implementation + 1 week testing  
> **Paper Section**: §3.Y Hybrid Prediction with Deep Kernel Learning

---

## 1. Motivation

The current BayesDiff oracle is a **pure Sparse Variational GP (SVGP)** with ARD Matérn-5/2 kernel:

$$
f(z) \sim \mathcal{GP}(m(z), k_{\text{Matérn}}(z, z'))
$$

While GPs provide well-calibrated uncertainty estimates, their capacity is limited:
- **Kernel expressiveness**: the Matérn kernel operates on raw embedding distances, which may not capture the relevant similarity for binding affinity
- **Scalability**: SVGP with $J=512$ inducing points struggles with high-dimensional inputs
- **Non-linearity**: GPs with stationary kernels cannot model complex, non-stationary input-output relationships

**The core tension**: Neural networks have strong prediction capacity but poor uncertainty; GPs have principled uncertainty but limited expressiveness. **Deep Kernel Learning (DKL) combines both**.

---

## 2. Architecture Design

### 2.1 Approach A: Deep Kernel Learning (DKL)

A neural network learns a feature transformation; a GP operates in the learned feature space:

$$
u = g_\theta(z) \in \mathbb{R}^{d_u}, \quad f(u) \sim \mathcal{GP}(m(u), k(u, u'))
$$

$$
\hat{y} = f(g_\theta(z))
$$

**Interpretation**: The neural network $g_\theta$ learns a "better kernel" by mapping inputs into a space where the GP's stationary kernel is more appropriate.

```
  z ∈ ℝ^d       u ∈ ℝ^d_u      ŷ ∈ ℝ
  ──────► g_θ ──────► GP ──────►
          (MLP)      (SVGP)
```

**Joint training**: $g_\theta$ and GP hyperparameters are trained simultaneously by maximizing the ELBO:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(f)} [\log p(y | f)] - \text{KL}[q(u) \| p(u)]
$$

### 2.2 Approach B: Neural Network + GP Residual

A neural network handles the main signal; a GP captures structured residuals:

$$
\hat{y}_{\text{NN}} = h_\theta(z)
$$

$$
r = y - \hat{y}_{\text{NN}}, \quad r \sim \mathcal{GP}(0, k(z, z'))
$$

$$
\hat{y} = \hat{y}_{\text{NN}} + \hat{r}_{\text{GP}}
$$

**Uncertainty combines both**:

$$
\sigma^2_{\text{oracle}} = \sigma^2_{\text{GP}}(r) + \sigma^2_{\text{epistemic}}(\text{NN})
$$

where NN epistemic uncertainty can be estimated via MC Dropout or ensemble.

### 2.3 Approach C: Spectral Normalized Neural GP (SNGP)

Replace the GP last layer with a random feature approximation + spectral normalization:

$$
\hat{y}, \sigma^2 = \text{SNGP}(g_\theta(z))
$$

**Advantage**: More scalable than exact GP; better distance-awareness than standard NN.

### 2.4 Recommended Progression

1. **Start with Approach A (DKL)** — cleanest integration with existing BayesDiff
2. **Compare with Approach B** as ablation
3. **Approach C** only if scalability becomes an issue

---

## 3. Mathematical Details

### 3.1 DKL Architecture

**Feature extractor** $g_\theta$:

$$
g_\theta: \mathbb{R}^d \to \mathbb{R}^{d_u}
$$

Implemented as a 2–3 layer MLP with residual connections:

$$
u = g_\theta(z) = z + \text{MLP}(z)
$$

(residual connection ensures the GP has access to at least the original features)

**GP in feature space**:

$$
k_{\text{DKL}}(z, z') = k_{\text{base}}(g_\theta(z), g_\theta(z'))
$$

where $k_{\text{base}}$ is the original ScaleKernel(Matérn-5/2) with ARD.

**Inducing points**: Learned in feature space $u$, not in original space $z$:

$$
\tilde{u}_j = g_\theta(\tilde{z}_j), \quad j = 1, \dots, J
$$

Or inducing points can be free parameters in $u$-space (more flexible).

### 3.2 Training Objective

Joint training of $\theta$ (NN params), kernel hyperparameters $\psi$, and variational parameters $\phi$:

$$
\max_{\theta, \psi, \phi} \; \mathcal{L}_{\text{ELBO}}(\theta, \psi, \phi) = \mathbb{E}_{q_\phi(f)}[\log p(y | f)] - \text{KL}[q_\phi(u) \| p_\psi(u)]
$$

With optional regularization on $g_\theta$:

$$
\mathcal{L} = \mathcal{L}_{\text{ELBO}} - \lambda_{\text{reg}} \|\theta\|_2^2
$$

### 3.3 Uncertainty Decomposition

The DKL hybrid provides the same uncertainty outputs as the current GP:

$$
\mu_{\text{oracle}}(z) = \mu_{\text{GP}}(g_\theta(z))
$$

$$
\sigma^2_{\text{oracle}}(z) = \sigma^2_{\text{GP}}(g_\theta(z))
$$

**Jacobian for Delta method** now chains through both the NN and GP:

$$
J_\mu = \frac{\partial \mu_{\text{oracle}}}{\partial z} = \frac{\partial \mu_{\text{GP}}}{\partial u} \cdot \frac{\partial g_\theta}{\partial z}
$$

This is handled automatically by `torch.autograd`.

### 3.4 NN+GP Residual (Approach B) Details

**NN training** (first stage):

$$
\min_\theta \; \frac{1}{N} \sum_{i=1}^N (y_i - h_\theta(z_i))^2 + \lambda \|\theta\|_2^2
$$

**Residual GP** (second stage):

$$
r_i = y_i - h_\theta(z_i), \quad r \sim \mathcal{GP}(0, k(z, z'))
$$

**Combined prediction**:

$$
\mu(z) = h_\theta(z) + \mu_{\text{GP}}^{(r)}(z)
$$

$$
\sigma^2(z) = \sigma^2_{\text{GP}}^{(r)}(z)
$$

---

## 4. Implementation Plan

### 4.1 New Module: `bayesdiff/hybrid_oracle.py`

```python
class FeatureExtractor(nn.Module):
    """MLP feature extractor for DKL."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64,
                 n_layers: int = 2, residual: bool = True, dropout: float = 0.1):
        ...
    
    def forward(self, z):
        """z: (B, d) → u: (B, d_u)"""
        ...


class DKLOracle:
    """Deep Kernel Learning oracle — feature extractor + SVGP."""
    
    def __init__(self, input_dim: int, feature_dim: int = 64, n_inducing: int = 512,
                 kernel: str = 'matern', hidden_dim: int = 256, n_layers: int = 2):
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, feature_dim, n_layers)
        self.gp = SVGPModel(feature_dim, n_inducing, kernel)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    def train(self, X_train, y_train, n_epochs=300, lr=0.01, batch_size=256):
        """
        Joint training of feature extractor + GP.
        
        Returns:
            history: dict with loss, lr, etc.
        """
        ...
    
    def predict(self, X_test):
        """
        Returns:
            mu: (N,) predicted means
            sigma2: (N,) predicted variances
        """
        ...
    
    def jacobian(self, z):
        """
        Compute ∂μ/∂z through feature extractor + GP.
        Required for Delta method uncertainty propagation.
        """
        ...
    
    def save(self, path): ...
    def load(self, path): ...


class NNResidualOracle:
    """Approach B: Neural network + GP residual predictor."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        self.nn = FeatureExtractor(input_dim, hidden_dim, output_dim=1, n_layers=n_layers)
        self.gp = GPOracle(input_dim)
    
    def train(self, X_train, y_train, nn_epochs=100, gp_epochs=200):
        """Two-stage training: NN first, then GP on residuals."""
        ...
    
    def predict(self, X_test):
        """mu = nn(X) + gp_mu(X), sigma2 = gp_sigma2(X)."""
        ...
```

### 4.2 Modifications to Existing Modules

**`bayesdiff/fusion.py`**: Accept any oracle that provides `(mu, sigma2, jacobian)`:

```python
def fuse_uncertainties(oracle_result, gen_result, y_target=7.0):
    """
    Args:
        oracle_result: NamedTuple with .mu, .sigma2, .jacobian
        gen_result: GenUncertaintyResult with .cov_gen
    """
    # Existing logic unchanged — just interface generalization
    ...
```

**`bayesdiff/evaluate.py`**: No changes needed (operates on predictions, not on oracle internals).

### 4.3 New Pipeline Script: `scripts/pipeline/s10_train_dkl.py`

```python
"""
Train DKL hybrid oracle on embeddings.

Usage:
    python scripts/pipeline/s10_train_dkl.py \
        --embeddings data/pdbbind_v2020/embeddings.npz \
        --labels data/pdbbind_v2020/labels.csv \
        --output results/dkl_model/ \
        --feature_dim 64 \
        --hidden_dim 256 \
        --n_layers 2 \
        --n_inducing 512 \
        --n_epochs 300 \
        --lr 0.01 \
        --batch_size 256 \
        --device cuda

Output:
    results/dkl_model/
        model.pt            # Feature extractor + GP checkpoint
        training_history.json
        config.json
"""
```

### 4.4 Training Strategy

**Learning rate schedule**: Different components need different LR:

| Component | Learning Rate | Rationale |
|-----------|--------------|-----------|
| Feature extractor $g_\theta$ | $10^{-3}$ | Needs to learn useful features |
| GP kernel hyperparameters | $10^{-2}$ | Standard GPyTorch range |
| GP variational parameters | $10^{-2}$ | Standard GPyTorch range |
| Inducing point locations | $10^{-2}$ | Should adapt to feature space |

**Regularization**:
- Weight decay on $g_\theta$: $\lambda = 10^{-4}$
- Spectral normalization on $g_\theta$ layers (optional, improves distance-awareness)
- Early stopping on validation ELBO (patience=20)

**Overfitting prevention** (critical with ~5,316 complexes, still moderate-sized):
- Feature dim $d_u = 32$ or $64$ (much smaller than input $d = 128$)
- Shallow NN (2 layers max)
- Strong dropout ($p = 0.1$–$0.3$)
- Monitor train/val ELBO gap

---

## 5. Test Plan

### 5.1 Unit Tests: `tests/stage2/test_hybrid_oracle.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_feature_extractor_shape` | Output shape = (B, d_u) |
| T1.2 | `test_feature_extractor_residual` | With residual=True, output ≈ input for zero-initialized MLP |
| T1.3 | `test_dkl_forward` | DKL produces (mu, sigma2) with correct shapes |
| T1.4 | `test_dkl_uncertainty_positive` | σ² > 0 for all inputs |
| T1.5 | `test_dkl_training_loss_decreases` | ELBO improves over 50 epochs on synthetic data |
| T1.6 | `test_dkl_jacobian_shape` | Jacobian has correct shape (d,) |
| T1.7 | `test_dkl_jacobian_finite_diff` | Autograd Jacobian ≈ finite-difference Jacobian |
| T1.8 | `test_nn_residual_forward` | NN+GP residual produces (mu, sigma2) |
| T1.9 | `test_nn_residual_improves_baseline` | NN+GP outperforms pure GP on synthetic nonlinear data |
| T1.10 | `test_save_load_roundtrip` | Save → load → same predictions |
| T1.11 | `test_dkl_ood_uncertainty` | OOD inputs get larger σ² than in-distribution |
| T1.12 | `test_feature_space_clustering` | After training, $g_\theta$ maps similar-pKd molecules closer |

```python
def test_dkl_training_loss_decreases():
    """DKL ELBO should improve during training on synthetic regression data."""
    X = torch.randn(200, 128)
    w = torch.randn(128)
    y = X @ w + 0.3 * torch.randn(200)
    
    oracle = DKLOracle(input_dim=128, feature_dim=32, n_inducing=50)
    history = oracle.train(X, y, n_epochs=50)
    
    # Loss should decrease
    assert history['loss'][-1] < history['loss'][0]
    # Final loss should be reasonable
    assert history['loss'][-1] < history['loss'][0] * 0.5


def test_dkl_ood_uncertainty():
    """Out-of-distribution inputs should have higher predictive variance."""
    X_train = torch.randn(200, 128)  # In-distribution: N(0, 1)
    y_train = torch.randn(200)
    
    oracle = DKLOracle(input_dim=128, feature_dim=32, n_inducing=50)
    oracle.train(X_train, y_train, n_epochs=50)
    
    X_id = torch.randn(20, 128)           # In-distribution
    X_ood = torch.randn(20, 128) * 5 + 10  # Out-of-distribution
    
    _, sigma2_id = oracle.predict(X_id)
    _, sigma2_ood = oracle.predict(X_ood)
    
    assert sigma2_ood.mean() > sigma2_id.mean()
```

### 5.2 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_dkl_with_delta_method` | Delta method works through DKL Jacobian |
| T2.2 | `test_dkl_with_gen_uncertainty` | Generation uncertainty + DKL fusion = valid total uncertainty |
| T2.3 | `test_dkl_with_calibration` | DKL predictions can be calibrated via isotonic regression |
| T2.4 | `test_dkl_with_ood_detection` | Mahalanobis OOD works in DKL feature space |
| T2.5 | `test_full_pipeline_dkl` | End-to-end: embeddings → DKL → fusion → calibration → metrics |
| T2.6 | `test_dkl_vs_gp_comparison` | DKL achieves ≥ same metrics as pure GP on same data |

### 5.3 Ablation Experiments

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A4.1 | Pure SVGP (baseline) | Reference |
| A4.2 | DKL (d_u=32, 1 layer) | Minimal DKL |
| A4.3 | DKL (d_u=64, 2 layers) | Standard DKL |
| A4.4 | DKL (d_u=128, 3 layers) | Deep DKL |
| A4.5 | DKL with residual connection | Test residual importance |
| A4.6 | DKL without residual connection | Control |
| A4.7 | DKL with spectral normalization | Distance-aware variant |
| A4.8 | NN + GP residual (Approach B) | Alternative architecture |
| A4.9 | DKL with enhanced repr (Sub-Plans 1–3) | Full combination |
| A4.10 | DKL + multi-task (Sub-Plan 5) | Two upgrades combined |

---

## 6. Evaluation & Success Criteria

### 6.1 Quantitative Metrics

| Metric | Pure GP (Stage 1) | Success | Stretch |
|--------|-------------------|---------|---------|
| $R^2$ | 0.120 | ≥ 0.18 | ≥ 0.28 |
| Spearman $\rho$ | 0.369 | ≥ 0.45 | ≥ 0.55 |
| NLL | baseline | ≥ 5% reduction | ≥ 15% reduction |
| ECE | 0.034 | ≤ 0.05 | ≤ 0.03 |
| AUROC | 1.000 | ≥ 0.95 | ≥ 0.98 |

### 6.2 Diagnostic Metrics

- **Feature space quality**: Spearman ρ in $u$-space vs $z$-space → should improve
- **Uncertainty calibration**: Expected vs observed confidence → should remain well-calibrated
- **Train/val gap**: Monitor for overfitting given small dataset
- **Inducing point utilization**: Fraction of inducing points actively used

### 6.3 Critical Risk: Overfitting

With ~5,316 PDBbind complexes:
- Total unique labeled data points: ~5,316
- DKL adds $O(10^4)$ parameters via feature extractor
- Use proper train/val/test split with protein family clustering
- If overfitting persists: fall back to pure GP or use Approach B with very small NN

---

## 7. Paper Integration

### 7.1 Methods Section (Draft)

> **§3.Y Hybrid Oracle with Deep Kernel Learning**
> 
> While Gaussian processes provide well-calibrated uncertainty estimates, the expressiveness of stationary kernels is limited when operating on high-dimensional molecular embeddings. We adopt Deep Kernel Learning (DKL) [Wilson et al., 2016] to combine the representation learning capacity of neural networks with the principled uncertainty quantification of GPs.
> 
> Specifically, we introduce a learned feature transformation $g_\theta: \mathbb{R}^d \to \mathbb{R}^{d_u}$ parameterized as a residual MLP:
> 
> $$u = g_\theta(z) = z_{\text{proj}} + \text{MLP}_\theta(z)$$
> 
> where $z_{\text{proj}} = W_{\text{proj}} z$ projects to dimension $d_u$. The SVGP then operates in the transformed feature space:
> 
> $$f(z) \sim \mathcal{GP}(m(g_\theta(z)), k(g_\theta(z), g_\theta(z')))$$
> 
> The feature extractor and GP are trained jointly by maximizing the ELBO. Crucially, the Delta method uncertainty propagation (Eq. X) remains valid because the Jacobian $\partial \mu / \partial z$ is computed automatically via backpropagation through both $g_\theta$ and the GP posterior.

### 7.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. D.1 | Architecture: z → g_θ → GP → (μ, σ²) | Explain DKL design |
| Fig. D.2 | Feature space t-SNE: before/after $g_\theta$ | Show learned feature quality |
| Fig. D.3 | Calibration: DKL vs pure GP | Uncertainty quality comparison |
| Fig. D.4 | Training curves: ELBO + metrics over epochs | Training behavior |
| Fig. D.5 | Ablation chart (A4.1–A4.10) | Design choice justification |

### 7.3 Tables

| Table | Content |
|-------|---------|
| Tab. D.1 | DKL architecture ablation (A4.2–A4.7) |
| Tab. D.2 | DKL vs NN+GP residual (A4.8) |
| Tab. D.3 | DKL + representation upgrades (A4.9–A4.10) |

---

## 8. Implementation Checklist

- [ ] Implement `FeatureExtractor` in `hybrid_oracle.py`
- [ ] Implement `DKLOracle` with joint training
- [ ] Implement `NNResidualOracle` with two-stage training
- [ ] Add Jacobian computation through DKL
- [ ] Generalize `fusion.py` oracle interface
- [ ] Write `s10_train_dkl.py` pipeline script
- [ ] Write unit tests (T1.1–T1.12)
- [ ] Write integration tests (T2.1–T2.6)
- [ ] Run ablation experiments (A4.1–A4.10)
- [ ] Monitor and mitigate overfitting
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
