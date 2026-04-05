# Sub-Plan 7: Uncertainty-Guided Generation (Closed-Loop Design)

> **Priority**: P3 — Future  
> **Dependency**: Sub-Plans 0–6 (requires PDBbind dataset + mature oracle + uncertainty pipeline)  
> **Estimated Effort**: 3–4 weeks implementation + 2 weeks testing  
> **Paper Section**: §4/§5 or Separate Follow-up Paper

---

## 1. Motivation

The current BayesDiff pipeline is **open-loop**: generate molecules first (TargetDiff), then evaluate (GP oracle + uncertainty). The generator has no knowledge of what makes a molecule likely to succeed.

**Closed-loop design** feeds the oracle's uncertainty estimates back into the generation process, enabling:

1. **Post-generation reranking** — select the best molecules from a large candidate pool
2. **Guided sampling** — bias the diffusion process toward high-$P_{\text{success}}$ molecules
3. **Active retraining** — iteratively improve the oracle with informative new data

This transforms BayesDiff from a passive evaluator into an **active molecular design system**.

---

## 2. Architecture Design

### 2.1 Three Layers of Closed-Loop Integration

```
Layer 1: Post-Generation Reranking (easiest)
    Generate M molecules → Score all → Select top-K by S(m)

Layer 2: Guided Sampling (moderate)
    During diffusion: x_{t-1} = denoise(x_t) + η ∇_{x_t} log P_success(x_t)

Layer 3: Active Retraining Loop (hardest)
    Generate → Score → Select uncertain → Oracle retrain → Repeat
```

### 2.2 Composite Scoring Function

For any closed-loop strategy, a composite score balances multiple objectives:

$$
S(m) = \lambda_1 P_{\text{success}}(m) - \lambda_2 \sigma^2_{\text{total}}(m) - \lambda_3 \text{OOD}(m) + \lambda_4 \text{Diversity}(m)
$$

| Term | Meaning | Encourages |
|------|---------|-----------|
| $P_{\text{success}}(m)$ | Predicted probability of activity | High predicted affinity |
| $\sigma^2_{\text{total}}(m)$ | Total predictive uncertainty | Confident predictions |
| $\text{OOD}(m)$ | Mahalanobis OOD score | In-distribution molecules |
| $\text{Diversity}(m)$ | Tanimoto distance to selected set | Chemical diversity |

---

## 3. Layer 1: Post-Generation Reranking

### 3.1 Algorithm

```
Input: Pocket x, oracle model, OOD detector, calibrator
Parameters: M (sample size), K (selection size), λ₁-λ₄ (weights)

1. Sample M molecules: {m₁, ..., m_M} ~ TargetDiff(x)
2. For each mᵢ:
   a. Extract embedding zᵢ
   b. Predict μᵢ, σ²ᵢ = Oracle(zᵢ)
   c. Compute P_success,ᵢ = 1 - Φ((y_target - μᵢ) / σᵢ)
   d. Compute OOD score
   e. Calibrate P_success,ᵢ
3. Compute composite score S(mᵢ) for each molecule
4. Select top-K molecules by S(mᵢ)
5. Optional: apply diversity filter (Tanimoto clustering)
```

### 3.2 Diversity Filter

To avoid selecting K near-identical molecules:

1. Compute pairwise Tanimoto similarities between top-K candidates
2. Apply greedy maximum-diversity selection:
   - Start with highest-scoring molecule
   - Iteratively add molecule that maximizes: $S(m) - \gamma \max_{m' \in \text{selected}} \text{Sim}(m, m')$

### 3.3 Implementation

```python
class MoleculeReranker:
    """Post-generation reranking with composite scoring."""
    
    def __init__(self, oracle, calibrator, ood_detector,
                 lambda_success=1.0, lambda_uncertainty=0.3,
                 lambda_ood=0.2, lambda_diversity=0.1):
        ...
    
    def score(self, embeddings, molecules=None):
        """
        Score M molecules for a single pocket.
        
        Args:
            embeddings: (M, d) molecular embeddings
            molecules: optional list of RDKit Mol objects (for diversity)
        
        Returns:
            scores: (M,) composite scores
            details: dict with per-component scores
        """
        ...
    
    def select_top_k(self, embeddings, k, molecules=None, diversity_threshold=0.7):
        """
        Select top-K molecules with diversity constraint.
        
        Returns:
            indices: (K,) selected indices
            scores: (K,) scores of selected molecules
        """
        ...
```

---

## 4. Layer 2: Guided Sampling

### 4.1 Theory: Classifier-Guided Diffusion

In the diffusion reverse process, at each step $t$:

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Guided sampling** adds a gradient signal from the oracle:

$$
\tilde{\mu}_\theta(x_t, t) = \mu_\theta(x_t, t) + \eta_t \sigma_t^2 \nabla_{x_t} \log P_{\text{success}}(x_t)
$$

where $\eta_t$ is a guidance scale that can vary with timestep.

### 4.2 Practical Challenges

1. **Oracle operates on embeddings, not on atom positions**: Need to backpropagate through the encoder
2. **Noisy intermediate states**: At high $t$, $x_t$ is very noisy; oracle predictions may be unreliable
3. **Gradient computation**: Requires differentiable path from $x_t$ through encoder to oracle

### 4.3 Approximate Guidance Strategies

#### Strategy A: Score-Based Guidance (Requires Differentiable Oracle)

Full backpropagation through encoder + oracle:

$$
\nabla_{x_t} \log P_{\text{success}}(x_t) = \nabla_{x_t} z \cdot \nabla_z \log P_{\text{success}}(z)
$$

**Pros**: Exact gradient. **Cons**: Expensive; requires differentiable encoder.

#### Strategy B: Embedding-Space Guidance (Decoupled)

Guide in embedding space, then map back:

1. At each step $t$, extract embedding $z_t$ from current $x_t$
2. Compute guidance in embedding space: $\Delta z = \eta \nabla_z \log P_{\text{success}}(z_t)$
3. Project guidance back to position space (approximately): $\Delta x_t \approx J_{\text{enc}}^+ \Delta z$ (pseudoinverse of encoder Jacobian)

**Pros**: Modular; uses existing oracle. **Cons**: Approximate; projection may be noisy.

#### Strategy C: Rejection-Based Guidance (Simplest)

Don't modify the diffusion process; instead:

1. Run multiple diffusion trajectories in parallel
2. At intermediate checkpoints (e.g., $t = T/4, T/2, 3T/4$), extract embeddings and score
3. Kill trajectories with low predicted $P_{\text{success}}$
4. Continue remaining trajectories

**Pros**: No gradient computation needed; uses oracle as black box. **Cons**: Wastes compute on killed trajectories.

### 4.4 Recommended Approach

**Start with Strategy C** (rejection-based) as it requires no modification to the diffusion model. Then explore Strategy A if compute allows.

### 4.5 Implementation

```python
class GuidedSampler:
    """Uncertainty-guided molecular generation."""
    
    def __init__(self, base_sampler, oracle, guidance_strategy='rejection',
                 guidance_scale=1.0, checkpoint_steps=None):
        """
        Args:
            base_sampler: TargetDiffSampler
            oracle: trained oracle (GP, DKL, or multi-task)
            guidance_strategy: 'rejection' | 'score_based' | 'embedding_space'
            guidance_scale: η parameter
            checkpoint_steps: list of diffusion steps to evaluate (for rejection)
        """
        ...
    
    def sample_guided(self, pocket_pdb, num_samples=64, num_initial=256):
        """
        Generate molecules with guidance.
        
        For rejection strategy:
            Start with num_initial trajectories,
            prune at checkpoints,
            return num_samples final molecules.
        """
        ...


class RejectionGuidedSampler(GuidedSampler):
    """Strategy C: Reject low-scoring trajectories at checkpoints."""
    
    def __init__(self, base_sampler, oracle, survival_fraction=0.5,
                 checkpoints=[25, 50, 75]):
        ...
    
    def sample_guided(self, pocket_pdb, num_final=64):
        """
        1. Start 256 trajectories
        2. At step 25: score, keep top 75% (192)
        3. At step 50: score, keep top 66% (128) 
        4. At step 75: score, keep top 50% (64)
        5. Complete remaining 64 trajectories
        """
        ...
```

---

## 5. Layer 3: Active Retraining Loop

### 5.1 Algorithm

```
Input: Initial oracle model, pocket set, TargetDiff

For iteration k = 1, 2, ..., K:
    1. Generate N_k molecules per pocket using guided sampling
    2. Score molecules with current oracle → μ_k, σ²_k, P_success,k
    3. Select most informative molecules:
       - High uncertainty: argmax σ²_total (exploration)
       - Near decision boundary: |P_success - 0.5| < ε (boundary)
       - High predicted activity: argmax μ (exploitation)
    4. Compute "ground truth" labels for selected molecules:
       - Vina re-scoring (fast, low-fidelity)
       - MM-GBSA (moderate, medium-fidelity)
       - Experimental (slow, high-fidelity) — if available
    5. Augment training set: D_{k+1} = D_k ∪ {(z_selected, y_selected)}
    6. Retrain oracle on D_{k+1}
    7. Evaluate: check if metrics improve on held-out test set
```

### 5.2 Acquisition Functions

For selecting which molecules to label next:

| Acquisition | Formula | Type |
|-------------|---------|------|
| **Maximum Uncertainty** | $a(z) = \sigma^2_{\text{total}}(z)$ | Exploration |
| **Expected Improvement** | $a(z) = \mathbb{E}[\max(0, f(z) - f^*)]$ | Exploitation |
| **Upper Confidence Bound** | $a(z) = \mu(z) + \beta \sigma(z)$ | Balanced |
| **BALD** (Bayesian Active Learning by Disagreement) | $a(z) = H[y|z] - \mathbb{E}[H[y|z, \theta]]$ | Information gain |

### 5.3 Practical Considerations

- **Label budget**: How many molecules can we label per iteration?
  - Vina re-scoring: ~1000/iteration (fast)
  - MM-GBSA: ~100/iteration (moderate compute)
  - Experimental: ~10/iteration (expensive; likely out of scope)
- **Convergence criteria**: Stop when $R^2$ improvement < threshold on validation set
- **Cold start**: First iteration uses existing training data + random selection

### 5.4 Implementation

```python
class ActiveLearningLoop:
    """Iterative active learning with oracle retraining."""
    
    def __init__(self, oracle, sampler, labeler, acquisition='uncertainty',
                 budget_per_iteration=100, max_iterations=10):
        ...
    
    def run(self, pocket_set, X_train, y_train, X_val, y_val):
        """
        Execute active learning loop.
        
        Returns:
            history: list of per-iteration metrics
            final_oracle: retrained oracle
        """
        ...
    
    def select_candidates(self, X_pool, n_select):
        """Select most informative candidates from pool."""
        ...
    
    def label_candidates(self, candidates):
        """Label selected candidates using the labeler (Vina, MM-GBSA, etc.)."""
        ...


class VinaLabeler:
    """Label molecules using AutoDock Vina re-scoring."""
    def __init__(self, vina_path):
        ...
    def label(self, molecules, pocket_pdb):
        ...
```

---

## 6. Test Plan

### 6.1 Unit Tests: Reranking

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_composite_scoring` | Score computation correct for known inputs |
| T1.2 | `test_top_k_selection` | Correct top-K selected by score |
| T1.3 | `test_diversity_filter` | Diversity constraint removes redundant molecules |
| T1.4 | `test_score_components` | Each component (P_success, σ², OOD, diversity) computed correctly |
| T1.5 | `test_lambda_sensitivity` | Different λ weights change rankings appropriately |

### 6.2 Unit Tests: Guided Sampling

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_rejection_sampler_output_count` | Correct number of final molecules |
| T2.2 | `test_rejection_improves_mean_score` | Guided molecules have higher mean P_success |
| T2.3 | `test_checkpoint_pruning` | Correct fraction survives at each checkpoint |
| T2.4 | `test_guidance_scale_effect` | Higher η → stronger selection pressure |

### 6.3 Unit Tests: Active Learning

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T3.1 | `test_uncertainty_acquisition` | Most uncertain points selected |
| T3.2 | `test_ucb_acquisition` | UCB balances exploration/exploitation |
| T3.3 | `test_retraining_improves_metrics` | Oracle metrics improve after adding informative data |
| T3.4 | `test_convergence_detection` | Loop stops when improvement stalls |

### 6.4 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T4.1 | `test_reranking_end_to_end` | Generate → score → select pipeline works |
| T4.2 | `test_guided_vs_unguided` | Guided sampling produces higher-quality molecules |
| T4.3 | `test_active_loop_synthetic` | Active learning on synthetic data shows improvement |
| T4.4 | `test_full_closed_loop` | All three layers combined |

### 6.5 Ablation Experiments

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A7.1 | Unguided generation + random selection | Baseline |
| A7.2 | Unguided + composite reranking | Layer 1 only |
| A7.3 | Unguided + reranking + diversity | Layer 1 with diversity |
| A7.4 | Rejection-guided sampling | Layer 2 |
| A7.5 | Rejection-guided + reranking | Layer 1 + 2 |
| A7.6 | Active learning (1 iteration) | Layer 3 minimal |
| A7.7 | Active learning (5 iterations) | Layer 3 extended |
| A7.8 | Full closed-loop (all layers) | Maximum integration |
| A7.9 | λ weight optimization (grid search) | Scoring function tuning |
| A7.10 | Acquisition function comparison | UCB vs. uncertainty vs. EI |

---

## 7. Evaluation & Success Criteria

### 7.1 Quantitative Metrics

| Metric | Unguided Baseline | Success | Stretch |
|--------|-------------------|---------|---------|
| Mean $P_{\text{success}}$ of top-K | baseline | ≥ 1.5× | ≥ 2× |
| Hit rate (pKd ≥ 7.0) in top-K | baseline | ≥ 1.3× | ≥ 2× |
| EF@1% | baseline | ≥ 1.5× | ≥ 3× |
| Diversity (avg Tanimoto distance) | baseline | ≥ 0.8× (no collapse) | ≥ 1.0× |
| Oracle $R^2$ after active learning | 0.120 | ≥ 0.20 | ≥ 0.30 |

### 7.2 Diagnostic Metrics

- **Compute efficiency**: How many total molecules generated to find K good ones?
  - Unguided: M molecules, select K
  - Rejection: M₀ > M molecules started, K survive
  - Metric: good-molecule yield = K_good / M_total
- **Active learning curve**: Oracle performance vs. number of labeled data points
- **Diversity-quality tradeoff**: Pareto frontier of quality vs. diversity

### 7.3 Failure Criteria

- Guided sampling produces **lower** diversity → mode collapse → reduce guidance scale
- Active learning oracle $R^2$ **decreases** → new data introduces noise → check label quality
- Compute cost > 5× unguided for marginal improvement → not cost-effective

---

## 8. Paper Integration

### 8.1 Methods/Results Section (Draft)

> **§4.X / §5.X Uncertainty-Guided Molecular Generation**
> 
> We demonstrate the utility of BayesDiff's uncertainty estimates as a feedback signal for molecular generation through three levels of integration.
> 
> *Post-generation reranking* (Layer 1): Given $M$ candidate molecules, we compute a composite score $S(m) = \lambda_1 P_{\text{success}}(m) - \lambda_2 \sigma^2_{\text{total}}(m) - \lambda_3 \text{OOD}(m) + \lambda_4 \text{Diversity}(m)$ and select the top-$K$ molecules subject to a diversity constraint. This achieves [X×] improvement in hit rate compared to random selection (Table X).
> 
> *Rejection-guided sampling* (Layer 2): We prune low-scoring diffusion trajectories at intermediate checkpoints during the reverse process, focusing computational resources on promising molecules. This improves the yield of high-quality molecules by [Y%] while reducing computational cost by [Z%] (Figure X).
> 
> *Active retraining* (Layer 3): We iteratively select the most informative molecules (highest predictive uncertainty) for oracle retraining, improving $R^2$ from $X$ to $Y$ over $N$ iterations with a label budget of $B$ molecules per iteration (Figure X).

### 8.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. G.1 | Closed-loop architecture diagram | System overview |
| Fig. G.2 | Reranking: score distribution before/after | Show selection quality |
| Fig. G.3 | Rejection sampling: survival curves | Show pruning efficiency |
| Fig. G.4 | Active learning curve: $R^2$ vs. iterations | Show improvement |
| Fig. G.5 | Diversity-quality Pareto frontier | Show tradeoff |
| Fig. G.6 | Compute efficiency: yield vs. cost | Practical value |

### 8.3 Tables

| Table | Content |
|-------|---------|
| Tab. G.1 | Reranking ablation (A7.1–A7.3) |
| Tab. G.2 | Guided sampling comparison (A7.4–A7.5) |
| Tab. G.3 | Active learning results (A7.6–A7.8) |
| Tab. G.4 | Acquisition function comparison (A7.10) |

---

## 9. Technical Considerations

### 9.1 Compute Requirements

| Operation | Cost per Pocket | Hardware |
|-----------|----------------|----------|
| Generate 256 molecules | ~30 min | A100 |
| Score 256 molecules | ~1 min | A100 |
| Rejection pruning (3 checkpoints) | ~0 extra (piggybacks on generation) | — |
| Vina re-scoring 100 molecules | ~10 min | CPU |
| Oracle retraining | ~5 min | A100 |

### 9.2 Modifying TargetDiff for Checkpointing

Rejection-guided sampling requires access to intermediate diffusion states:

```python
# In TargetDiff reverse process:
for t in reversed(range(T)):
    x_t = denoise_step(x_{t+1}, t)
    
    if t in checkpoint_steps:
        # Extract embeddings and score
        z_t = encoder(x_t)
        scores = oracle.predict(z_t)
        # Prune low-scoring trajectories
        mask = scores > threshold_t
        x_t = x_t[mask]
```

### 9.3 Practical Deployment Considerations

- **Reranking (Layer 1)** can be deployed immediately with current infrastructure
- **Guided sampling (Layer 2)** requires modification to TargetDiff sampling loop
- **Active learning (Layer 3)** requires a labeling oracle (Vina available; MM-GBSA needs setup)

---

## 10. Implementation Checklist

- [ ] Implement `MoleculeReranker` with composite scoring
- [ ] Implement diversity filter (Tanimoto-based)
- [ ] Implement `RejectionGuidedSampler`
- [ ] Modify TargetDiff sampling loop to support checkpointing
- [ ] Implement `ActiveLearningLoop`
- [ ] Implement acquisition functions (uncertainty, UCB, EI)
- [ ] Implement `VinaLabeler` for automated re-scoring
- [ ] Write unit tests (T1.1–T3.4)
- [ ] Write integration tests (T4.1–T4.4)
- [ ] Run ablation experiments (A7.1–A7.10)
- [ ] Generate paper figures and tables
- [ ] Draft results section text
