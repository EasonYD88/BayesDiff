# Sub-Plan 5: Multi-Task Learning (Regression + Ranking + Classification)

> **Priority**: P2 — Medium  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); Sub-Plans 1–3 (benefits from richer backbone)  
> **Training Data**: PDBbind v2020 refined set (~5,316 complexes), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 2–3 weeks implementation + 1 week testing  
> **Paper Section**: §3.Z Multi-Task Oracle Training

---

## 1. Motivation

BayesDiff simultaneously serves multiple downstream decision tasks:

| Decision Task | Current Metric | Objective Function Used |
|--------------|----------------|------------------------|
| Predict pKd value | $R^2 = 0.12$, RMSE = 1.87 | MSE regression |
| Rank molecules by affinity | Spearman $\rho = 0.369$ | (implicit from regression) |
| Classify active vs. inactive | AUROC = 1.0 | (implicit from $P_{\text{success}}$ threshold) |
| Estimate success probability | ECE = 0.034 | (calibration post-hoc) |

**Problem**: Training *only* on MSE regression does not directly optimize for ranking or classification performance. The loss landscape for "get the pKd number right" is different from "correctly order molecules" or "correctly identify active binders."

**Solution**: Multi-task learning with three complementary heads sharing a common backbone, so the representation is shaped by all three objectives simultaneously.

---

## 2. Architecture Design

### 2.1 Overall Structure

```
        z (from encoder + attention/fusion)
                    │
            ┌───────┴───────┐
            │  Shared Trunk  │
            │  (2-layer MLP) │
            └───────┬───────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Reg Head │ │Cls Head │ │Rank Head│
  │  ŷ_pkd  │ │  p_act  │ │  s_rank │
  └─────────┘ └─────────┘ └─────────┘
       │            │            │
       ▼            ▼            ▼
    L_reg        L_cls        L_rank
       │            │            │
       └────────────┼────────────┘
                    ▼
          L_total = λ₁L_reg + λ₂L_cls + λ₃L_rank
```

### 2.2 Task Definitions

#### Task 1: Regression (pKd Prediction)

$$
\hat{y}_{\text{reg}} = W_{\text{reg}} h_{\text{trunk}} + b_{\text{reg}} \in \mathbb{R}
$$

$$
\mathcal{L}_{\text{reg}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_{\text{reg},i})^2
$$

#### Task 2: Binary Classification (Active Binder)

$$
p_{\text{act}} = \sigma(W_{\text{cls}} h_{\text{trunk}} + b_{\text{cls}}) \in [0, 1]
$$

$$
\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} [c_i \log p_{\text{act},i} + (1 - c_i) \log(1 - p_{\text{act},i})]
$$

where $c_i = \mathbb{1}[\text{pKd}_i \geq \tau]$ and $\tau$ is the activity threshold (default: $\tau = 7.0$, corresponding to 100 nM).

#### Task 3: Pairwise Ranking

For pairs $(i, j)$ where $y_i > y_j$:

$$
\mathcal{L}_{\text{rank}} = \frac{1}{|P|} \sum_{(i,j) \in P} \max(0, \delta - (s_i - s_j))
$$

where $s_i = W_{\text{rank}} h_{\text{trunk},i} + b_{\text{rank}}$ is a ranking score and $\delta$ is a margin (default: 0.5).

Alternative: **BPR (Bayesian Personalized Ranking) loss**:

$$
\mathcal{L}_{\text{BPR}} = -\frac{1}{|P|} \sum_{(i,j) \in P} \log \sigma(s_i - s_j)
$$

### 2.3 Joint Objective

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{cls}} + \lambda_3 \mathcal{L}_{\text{rank}}
$$

**Default**: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0.3$

**Advanced**: Use **uncertainty-based weighting** (Kendall et al., 2018):

$$
\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{reg}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{cls}} + \frac{1}{2\sigma_3^2} \mathcal{L}_{\text{rank}} + \log \sigma_1 \sigma_2 \sigma_3
$$

where $\sigma_1, \sigma_2, \sigma_3$ are learnable task-specific uncertainty parameters.

### 2.4 Pair Sampling Strategy for Ranking Loss

Sampling all pairs is $O(N^2)$. Efficient strategies:

1. **In-batch sampling**: For each minibatch of $B$ samples, form $B(B-1)/2$ pairs
2. **Hard negative mining**: Prioritize pairs where the model currently ranks incorrectly
3. **Within-pocket pairs**: Only compare molecules generated for the same pocket

---

## 3. Implementation Plan

### 3.1 New Module: `bayesdiff/multi_task.py`

```python
class SharedTrunk(nn.Module):
    """Shared feature extractor for multi-task learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        ...
    
    def forward(self, z):
        """z: (B, d) → h: (B, d_trunk)"""
        ...


class RegressionHead(nn.Module):
    """Linear head for pKd regression."""
    def __init__(self, input_dim: int):
        ...
    def forward(self, h):
        """h: (B, d_trunk) → ŷ: (B,)"""
        ...


class ClassificationHead(nn.Module):
    """Linear head for active/inactive classification."""
    def __init__(self, input_dim: int):
        ...
    def forward(self, h):
        """h: (B, d_trunk) → p: (B,) in [0, 1]"""
        ...


class RankingHead(nn.Module):
    """Linear head for pairwise ranking scores."""
    def __init__(self, input_dim: int):
        ...
    def forward(self, h):
        """h: (B, d_trunk) → s: (B,) ranking scores"""
        ...


class MultiTaskOracle(nn.Module):
    """Multi-task predictor with regression, classification, and ranking heads."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, trunk_dim: int = 128,
                 activity_threshold: float = 7.0, learned_weights: bool = False):
        self.trunk = SharedTrunk(input_dim, hidden_dim, trunk_dim)
        self.reg_head = RegressionHead(trunk_dim)
        self.cls_head = ClassificationHead(trunk_dim)
        self.rank_head = RankingHead(trunk_dim)
        
        if learned_weights:
            self.log_sigma = nn.Parameter(torch.zeros(3))  # Uncertainty weighting
        
        self.threshold = activity_threshold
    
    def forward(self, z):
        """
        Returns:
            reg_out: (B,) pKd predictions
            cls_out: (B,) activity probabilities
            rank_out: (B,) ranking scores
            h_trunk: (B, d_trunk) trunk features (for GP input)
        """
        h = self.trunk(z)
        return self.reg_head(h), self.cls_head(h), self.rank_head(h), h
    
    def compute_loss(self, z, y, lambda_reg=1.0, lambda_cls=0.5, lambda_rank=0.3):
        """
        Compute weighted multi-task loss.
        
        Args:
            z: (B, d) input embeddings
            y: (B,) pKd labels
        
        Returns:
            total_loss, loss_dict
        """
        ...
    
    def _make_pairs(self, y):
        """Generate positive pairs (i, j) where y_i > y_j from batch."""
        ...


class MultiTaskGPOracle:
    """Multi-task backbone + GP for uncertainty-aware prediction."""
    
    def __init__(self, multi_task: MultiTaskOracle, gp_input_dim: int = 128):
        self.multi_task = multi_task
        self.gp = GPOracle(gp_input_dim)
    
    def train(self, X_train, y_train, mt_epochs=100, gp_epochs=200):
        """
        Two-stage training:
        1. Train multi-task backbone end-to-end
        2. Extract trunk features, train GP on top
        """
        ...
    
    def predict(self, X_test):
        """
        Returns:
            mu: (N,) from GP on trunk features
            sigma2: (N,) from GP
            cls_prob: (N,) from classification head
            rank_score: (N,) from ranking head
        """
        ...
```

### 3.2 Integration with GP Uncertainty

The multi-task backbone produces enriched trunk features $h \in \mathbb{R}^{d_{\text{trunk}}}$. The GP oracle then operates on these features:

$$
\mu_{\text{oracle}}, \sigma^2_{\text{oracle}} = \text{GP}(h_{\text{trunk}})
$$

**Two options**:
1. **Sequential**: Train multi-task → freeze trunk → extract $h$ → train GP
2. **Joint (DKL-style)**: Train multi-task + GP simultaneously

Option 1 is simpler and recommended first.

### 3.3 Enhanced $P_{\text{success}}$ Computation

The classification head provides an independent estimate of activity probability:

$$
P_{\text{success}}^{\text{final}} = \alpha \cdot P_{\text{success}}^{\text{GP}} + (1 - \alpha) \cdot p_{\text{cls}}
$$

where $\alpha$ is optimized on a calibration set. Or use the classification probability as a prior in the GP-based estimation.

### 3.4 New Pipeline Script: `scripts/pipeline/s10b_train_multitask.py`

```python
"""
Train multi-task oracle (regression + classification + ranking).

Usage:
    python scripts/pipeline/s10b_train_multitask.py \
        --embeddings data/pdbbind_v2020/embeddings.npz \
        --labels data/pdbbind_v2020/labels.csv \
        --output results/multitask_model/ \
        --hidden_dim 256 \
        --trunk_dim 128 \
        --lambda_reg 1.0 \
        --lambda_cls 0.5 \
        --lambda_rank 0.3 \
        --threshold 7.0 \
        --n_epochs 200 \
        --device cuda
"""
```

---

## 4. Test Plan

### 4.1 Unit Tests: `tests/stage2/test_multi_task.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_trunk_output_shape` | Trunk: (B, d) → (B, d_trunk) |
| T1.2 | `test_reg_head_shape` | Regression head: (B, d_trunk) → (B,) |
| T1.3 | `test_cls_head_range` | Classification output ∈ [0, 1] |
| T1.4 | `test_rank_head_shape` | Ranking head: (B, d_trunk) → (B,) |
| T1.5 | `test_multitask_forward` | All three heads produce valid outputs simultaneously |
| T1.6 | `test_pair_generation` | Pair sampling produces correct (i, j) pairs from labels |
| T1.7 | `test_ranking_loss_decreases` | Ranking loss is reduced when correct pairs are scored higher |
| T1.8 | `test_cls_loss_correct_labels` | Low loss when predictions match true active/inactive labels |
| T1.9 | `test_joint_loss_computation` | Total loss = weighted sum of three subloss terms |
| T1.10 | `test_learned_task_weights` | Uncertainty-based weights are learnable and positive |
| T1.11 | `test_gradient_flow_all_heads` | All parameters in all heads receive gradients |
| T1.12 | `test_training_convergence` | Joint loss decreases over 50 epochs on synthetic data |

```python
def test_pair_generation():
    """Verify that pair generation creates correct (better, worse) pairs."""
    y = torch.tensor([8.0, 6.0, 7.5, 5.0, 9.0])
    oracle = MultiTaskOracle(input_dim=128)
    pairs = oracle._make_pairs(y)
    
    for i, j in pairs:
        assert y[i] > y[j], f"Pair ({i},{j}): y[{i}]={y[i]} should > y[{j}]={y[j]}"
    
    # Total pairs: C(5,2) = 10 ordered pairs
    assert len(pairs) == 10


def test_training_convergence():
    """Multi-task oracle should converge on synthetic data."""
    torch.manual_seed(42)
    X = torch.randn(200, 128)
    w = torch.randn(128)
    y = X @ w / 128 * 3 + 7  # pKd centered around 7
    
    oracle = MultiTaskOracle(input_dim=128, activity_threshold=7.0)
    optimizer = torch.optim.Adam(oracle.parameters(), lr=1e-3)
    
    losses = []
    for epoch in range(50):
        total_loss, _ = oracle.compute_loss(X, y)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    
    assert losses[-1] < losses[0] * 0.5  # At least 50% reduction
```

### 4.2 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_multitask_to_gp` | Trunk features → GP training succeeds |
| T2.2 | `test_multitask_with_delta` | Jacobian through trunk+GP works for Delta method |
| T2.3 | `test_cls_improves_p_success` | Combined P_success is better calibrated |
| T2.4 | `test_rank_improves_spearman` | Ranking head improves Spearman ρ |
| T2.5 | `test_full_pipeline_multitask` | End-to-end with multi-task oracle |
| T2.6 | `test_multitask_vs_singletask` | Multi-task ≥ single-task on at least 2 of 3 metrics |

### 4.3 Ablation Experiments

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A5.1 | Regression only (baseline GP) | Reference |
| A5.2 | Regression + Classification | Two-task benefit |
| A5.3 | Regression + Ranking | Two-task benefit |
| A5.4 | All three tasks (fixed λ) | Full multi-task |
| A5.5 | All three tasks (learned λ) | Uncertainty weighting |
| A5.6 | $\lambda_{\text{cls}}$ sensitivity: 0.1, 0.3, 0.5, 1.0 | Classification weight |
| A5.7 | $\lambda_{\text{rank}}$ sensitivity: 0.1, 0.3, 0.5, 1.0 | Ranking weight |
| A5.8 | Threshold sensitivity: $\tau$ = 6.0, 6.5, 7.0, 7.5, 8.0 | Activity cutoff |
| A5.9 | Margin loss vs. BPR loss for ranking | Ranking loss variant |
| A5.10 | Multi-task + DKL (Sub-Plan 4) | Combined predictors |

---

## 5. Evaluation & Success Criteria

### 5.1 Quantitative Metrics

| Metric | Baseline (GP) | Success | Stretch |
|--------|---------------|---------|---------|
| $R^2$ | 0.120 | ≥ 0.15 | ≥ 0.25 |
| Spearman $\rho$ | 0.369 | ≥ 0.45 | ≥ 0.55 |
| AUROC (classification) | 1.000 | ≥ 0.95 | ≥ 0.99 |
| ECE | 0.034 | ≤ 0.05 | ≤ 0.03 |
| EF@1% | baseline | ≥ 1.3× | ≥ 2.0× |
| **NDCG@10** (new) | N/A | ≥ 0.7 | ≥ 0.85 |

### 5.2 Diagnostic Metrics

- **Per-task loss curves**: Monitor convergence of each head independently
- **Task gradient conflict**: Measure cosine similarity between task gradients
- **Trunk representation quality**: t-SNE/UMAP of $h_{\text{trunk}}$ colored by pKd
- **Learned $\lambda$ values**: Which task gets most weight in uncertainty-based weighting?

### 5.3 Risk: Negative Transfer

Multi-task learning can *hurt* performance if tasks conflict (negative transfer). Mitigation:

1. **Monitor per-task metrics**: If any metric degrades > 5% vs. single-task, investigate
2. **Gradient surgery** (Yu et al., 2020): Project conflicting gradients to prevent negative transfer
3. **Fallback**: If multi-task hurts, use single-task backbone but extract classification features as auxiliary inputs

---

## 6. Paper Integration

### 6.1 Methods Section (Draft)

> **§3.Z Multi-Task Oracle Training**
> 
> To align the learned representation with the multiple decision modes of BayesDiff — point prediction, binary activity classification, and molecule ranking — we train the oracle backbone with a joint multi-task objective:
> 
> $$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{cls}} + \lambda_3 \mathcal{L}_{\text{rank}}$$
> 
> where $\mathcal{L}_{\text{reg}}$ is MSE for pKd regression, $\mathcal{L}_{\text{cls}}$ is binary cross-entropy for activity classification (with threshold $\tau = 7.0$, corresponding to 100 nM affinity), and $\mathcal{L}_{\text{rank}}$ is a margin-based pairwise ranking loss.
> 
> A shared trunk network maps the molecular embedding $z$ to a task-general representation $h$, from which three lightweight heads produce task-specific outputs. The trunk representation $h$ is then used as input to the GP oracle for uncertainty estimation, ensuring that the learned features simultaneously support accurate prediction, reliable ranking, and calibrated classification.
> 
> Task weights $\lambda_1, \lambda_2, \lambda_3$ are determined via [grid search / uncertainty-based weighting].

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. M.1 | Architecture diagram with three heads | Explain multi-task design |
| Fig. M.2 | Per-task training curves | Show convergence behavior |
| Fig. M.3 | Task gradient cosine similarity | Analyze task relationships |
| Fig. M.4 | Learned task weights (uncertainty-based) | Interpretability |
| Fig. M.5 | Ranking quality: NDCG@k curve | Show ranking improvement |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. M.1 | Task combination ablation (A5.1–A5.5) |
| Tab. M.2 | Weight sensitivity (A5.6–A5.7) |
| Tab. M.3 | Activity threshold sensitivity (A5.8) |

---

## 7. Implementation Checklist

- [ ] Implement `SharedTrunk` in `multi_task.py`
- [ ] Implement `RegressionHead`, `ClassificationHead`, `RankingHead`
- [ ] Implement `MultiTaskOracle` with joint loss computation
- [ ] Implement pair sampling (in-batch, hard negative mining)
- [ ] Implement uncertainty-based task weighting
- [ ] Implement `MultiTaskGPOracle` (trunk → GP pipeline)
- [ ] Write `s10b_train_multitask.py` pipeline script
- [ ] Write unit tests (T1.1–T1.12)
- [ ] Write integration tests (T2.1–T2.6)
- [ ] Run ablation experiments (A5.1–A5.10)
- [ ] Add NDCG metric to evaluation module
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
