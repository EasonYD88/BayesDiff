# Sub-Plan 5: Multi-Task Trunk Shaping for the Best Hybrid Oracle

**Regression + Classification + Group-Aware Ranking**

> **Priority**: P2 — Medium  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 dataset); Sub-Plan 4 (best oracle head = DKL Ensemble)  
> **Training Data**: PDBbind v2020 R1 general set (19,037 before filtering); grouped protein-cluster Train/Val; CASF-2016 fixed Test (285). See [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 2–3 weeks implementation + 1 week testing  
> **Paper Section**: §3.Z Multi-Task Trunk Shaping  
> **SP4 Baseline**: DKL Ensemble (M=5, d_u=32, residual+bootstrap) — $\rho = 0.781$, $R^2 = 0.607$, $\rho_{|err|,\sigma} \approx 0.09$, NLL $\approx 1.76$

---

## 1. Motivation

Sub-Plan 04 showed that oracle-head upgrades can preserve strong point prediction ($\rho \approx 0.78$) and moderately improve uncertainty quality, but the error–uncertainty correlation remains weak ($\rho_{|err|,\sigma} \approx 0.09$). The DKL Ensemble is the confirmed best oracle head; single-model GP and SNGP produced unreliable uncertainty signals. Ensemble disagreement is the only reliable UQ mechanism identified so far.

This suggests that the bottleneck may no longer lie purely in the oracle head, but in whether the **trunk representation** is explicitly shaped for the downstream decision tasks of BayesDiff: numerical affinity prediction, active/inactive discrimination, and within-target prioritization.

**Goal of Sub-Plan 05**: Learn a trunk representation that is better aligned with the three downstream decision modes — regression, classification, and ranking — and then feed this trunk into the best oracle head selected in Sub-Plan 04. This is not an independent oracle; it is an upstream trunk module whose output connects to the SP4 winner.

**Why this matters at 19k scale**: The PDBbind v2020 R1 general set provides ~19,037 complexes (vs. the 5,316 refined subset). Multi-task trunk shaping is substantially more justified at this scale — classification and ranking signals from 19k samples provide richer supervisory gradients than they would from 5k.

---

## 2. Architecture Design

### 2.1 Overall Structure

Sub-Plan 05 produces a shaped trunk $h_{\text{trunk}}$. The oracle head comes from Sub-Plan 04.

```
        z (from encoder + attention/fusion)
                    │
            ┌───────┴───────┐
            │  Shared Trunk  │   ← trained by SP05 multi-task loss
            │  (2-layer MLP) │
            └───────┬───────┘
                    │
                    ├─── h_trunk ───► best SP4 oracle head (DKL Ensemble)
                    │                     → μ, σ², Jacobian
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Reg Head │ │Cls Head │ │Rank Head│   ← auxiliary shaping heads
  │  ŷ_pkd  │ │  p_act  │ │  s_rank │      (active during training only)
  └─────────┘ └─────────┘ └─────────┘
       │            │            │
       ▼            ▼            ▼
    L_reg        L_cls        L_rank
       │            │            │
       └────────────┼────────────┘
                    ▼
          L_total = λ₁L_reg + λ₂L_cls  [+ λ₃L_rank in v2]
```

### 2.2 Phased Task Introduction

#### Phase 05-v1: Regression + Classification (first version)

Classification is prioritized because:
- It most directly helps $P_{\text{success}}$ estimation
- It integrates cleanly with post-hoc calibration
- It provides more stable trunk shaping than ranking

$$
\mathcal{L}_{\text{v1}} = \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{cls}}
$$

**Default**: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$

#### Phase 05-v2: Add Group-Aware Ranking

Ranking is introduced only after 05-v1 is validated:

$$
\mathcal{L}_{\text{v2}} = \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{cls}} + \lambda_3 \mathcal{L}_{\text{rank}}
$$

**Default**: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0.3$

### 2.3 Task Definitions

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

#### Task 3: Group-Aware Pairwise Ranking (05-v2 only)

Ranking loss is computed **only within biologically coherent groups**. Global cross-target pair construction is disallowed by default.

**Default and only grouping for Tier 1–2**: protein cluster (mmseqs2 `cluster_id` from SP0). Same-target and same-pocket-family are deferred variants for Tier 3+ exploration only — mixing group definitions in the first pass would confound ranking-head effectiveness with group-definition effects.

For pairs $(i, j)$ where $y_i > y_j$ and $\text{group}(i) = \text{group}(j)$:

$$
\mathcal{L}_{\text{rank}} = -\frac{1}{|P|} \sum_{(i,j) \in P} \log \sigma(s_i - s_j) \quad \text{(BPR loss)}
$$

where $s_i = W_{\text{rank}} h_{\text{trunk},i} + b_{\text{rank}}$ is a ranking score.

**BPR is preferred over margin hinge** because it is smoother and more robust under noisy biological labels. Margin loss may cause the trunk to learn overly sharp decision boundaries.

### 2.4 Group-Aware Pair Sampling Strategy

Sampling all pairs is $O(N^2)$. Pairs must respect biological grouping:

1. **Within-cluster pairs** (**default, locked for Tier 1–2**): Only construct pairs from the same protein cluster (as defined by mmseqs2 clustering in Sub-Plan 0, stored in `data/pdbbind_v2020/cluster_assignments.csv` with columns `pdb_code, cluster_id, cluster_median_pkd, pkd_bin`)
2. **Within-target pairs** (deferred to Tier 3+): Stricter variant — only compare ligands bound to the exact same target protein
3. **Hard negative mining** (deferred to Tier 3+): Within each group, prioritize pairs where the model currently ranks incorrectly
4. **Batch construction**: Use a GroupedBatchSampler that fills each minibatch by sampling entire groups, so each batch contains multiple samples from the same cluster

**Explicitly disallowed**: Naive in-batch pair construction across arbitrary samples from different targets.

#### Efficient Grouped Pair Construction Algorithm

```python
def make_grouped_pairs(y: Tensor, groups: Tensor, max_pairs_per_group: int = 50) -> Tensor:
    """
    Build (i, j) pairs where y[i] > y[j] and groups[i] == groups[j].
    
    For groups with many members (e.g. > 10), randomly subsample to cap at
    max_pairs_per_group to prevent quadratic blow-up.
    
    Args:
        y: (B,) pKd labels
        groups: (B,) integer group IDs (protein cluster)
        max_pairs_per_group: max ordered pairs per group per batch
    
    Returns:
        pairs: (P, 2) tensor of (better_idx, worse_idx) pairs
    """
    unique_groups = groups.unique()
    all_pairs = []
    for g in unique_groups:
        mask = (groups == g)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        y_g = y[idx]
        # All ordered pairs within group
        i_idx, j_idx = torch.meshgrid(torch.arange(len(idx)), torch.arange(len(idx)), indexing='ij')
        valid = y_g[i_idx] > y_g[j_idx]
        pairs_local = torch.stack([idx[i_idx[valid]], idx[j_idx[valid]]], dim=1)
        # Subsample if too many
        if len(pairs_local) > max_pairs_per_group:
            perm = torch.randperm(len(pairs_local))[:max_pairs_per_group]
            pairs_local = pairs_local[perm]
        all_pairs.append(pairs_local)
    if not all_pairs:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.cat(all_pairs, dim=0)
```

#### GroupedBatchSampler for Ranking-Enabled Training

```python
class GroupedBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches where each batch is filled by sampling entire protein clusters.
    
    Strategy: shuffle clusters, then greedily fill batches by adding whole clusters
    until batch_size is reached. Ensures each batch has multiple same-group samples
    for pair construction.
    """
    def __init__(self, group_ids: np.ndarray, batch_size: int = 128,
                 min_group_size: int = 2, drop_last: bool = False):
        # group_ids: (N,) cluster assignments
        # Build group → indices mapping
        self.groups = {}
        for i, g in enumerate(group_ids):
            self.groups.setdefault(int(g), []).append(i)
        # Filter groups with < min_group_size (useless for ranking)
        self.groups = {g: idx for g, idx in self.groups.items() if len(idx) >= min_group_size}
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        group_keys = list(self.groups.keys())
        random.shuffle(group_keys)
        batch = []
        for g in group_keys:
            batch.extend(self.groups[g])
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch and not self.drop_last:
            yield batch
```

### 2.5 Class Imbalance Considerations for Classification

At $\tau = 7.0$ (100 nM) on the PDBbind v2020 R1 general set (~19k complexes), the positive (active) rate is roughly 30–40%. This is moderately imbalanced but not extreme. Mitigation strategies:

1. **No special handling** (default): BCE loss handles moderate imbalance adequately for trunk shaping purposes
2. **Pos-weight in BCE**: If positive rate < 25%, use `pos_weight = N_neg / N_pos` in `F.binary_cross_entropy_with_logits`
3. **Monitor**: Log per-class recall during training to detect collapse to majority-class prediction

The exact positive rate must be computed from the cleaned general set after CASF-2016 removal and label filtering. Add a diagnostic check in the pipeline script.

### 2.5 Advanced Task Weighting (deferred to Tier 3)

**Uncertainty-based weighting** (Kendall et al., 2018) — only if fixed weights are insufficient:

$$
\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{reg}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{cls}} + \frac{1}{2\sigma_3^2} \mathcal{L}_{\text{rank}} + \log \sigma_1 \sigma_2 \sigma_3
$$

where $\sigma_1, \sigma_2, \sigma_3$ are learnable task-specific uncertainty parameters.

---

## 3. Implementation Plan

### 3.0 Data Pipeline: Loading and Preparation

The trunk training requires frozen embeddings (from SP4) joined with cluster metadata (from SP0).

```python
# ── Data loading ────────────────────────────────────────────────
import numpy as np
import pandas as pd

# 1. Load frozen embeddings (same file used by s18_train_oracle_heads.py)
emb = np.load("results/stage2/oracle_heads/frozen_embeddings.npz")
X_train = emb["X_train"].astype(np.float32)  # (N_train, 128)
y_train = emb["y_train"].astype(np.float32)  # (N_train,)
X_val   = emb["X_val"].astype(np.float32)    # (N_val, 128)
y_val   = emb["y_val"].astype(np.float32)    # (N_val,)
X_test  = emb["X_test"].astype(np.float32)   # (N_test, 128)  CASF-2016
y_test  = emb["y_test"].astype(np.float32)   # (N_test,)

# 2. Load cluster assignments for group-aware ranking (v2 only)
clusters = pd.read_csv("data/pdbbind_v2020/cluster_assignments.csv")
# columns: pdb_code, cluster_id, cluster_median_pkd, pkd_bin

# 3. Load split info to map indices → PDB codes → cluster IDs
split = pd.read_csv("data/pdbbind_v2020/splits/split_s0_seed42.csv")
# join on pdb_code to get group IDs for each training sample

# 4. Build group_ids arrays aligned with X_train/X_val indices
train_codes = split[split["split"] == "train"]["pdb_code"].values
code_to_cluster = dict(zip(clusters["pdb_code"], clusters["cluster_id"]))
groups_train = np.array([code_to_cluster.get(c, -1) for c in train_codes])

# 5. Classification labels
c_train = (y_train >= 7.0).astype(np.float32)  # active/inactive
c_val   = (y_val >= 7.0).astype(np.float32)

# Log class balance
pos_rate_train = c_train.mean()
print(f"Classification positive rate (train): {pos_rate_train:.3f}")
print(f"  Active: {int(c_train.sum())}, Inactive: {int(len(c_train) - c_train.sum())}")
```

**HARD REQUIREMENT**: The frozen embeddings NPZ does not currently store PDB codes or cluster IDs.  Before any SP05 experiments, the NPZ **must** be augmented with `codes_train`, `codes_val`, `codes_test`, `groups_train`, and `groups_val` arrays so that sample-to-cluster alignment is guaranteed.  Alternatively, a separate alignment script must produce these arrays and verify index ordering against the split CSV.  Without this, all ranking (v2) experiments and any group-aware evaluation will produce incorrect results.

### 3.1 New Module: `bayesdiff/multi_task.py`

```python
"""Multi-task trunk shaping for the best hybrid oracle (Sub-Plan 05).

Trains a shared trunk representation with auxiliary regression, classification,
and ranking heads. The shaped trunk is then consumed by the SP4 oracle head
(DKL Ensemble) for uncertainty-aware prediction.

Usage:
    from bayesdiff.multi_task import MultiTaskTrunk, MultiTaskHybridOracle
    from bayesdiff.hybrid_oracle import DKLEnsembleOracle
    
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=128)
    oracle_head = DKLEnsembleOracle(input_dim=128, n_members=5)
    hybrid = MultiTaskHybridOracle(trunk, oracle_head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from bayesdiff.oracle_interface import OracleHead, OracleResult


class SharedTrunk(nn.Module):
    """Shared feature extractor for multi-task learning.
    
    Architecture: input_dim → hidden_dim → hidden_dim → output_dim
    with ReLU activations, dropout, and optional residual connection.
    Matches the FeatureExtractor pattern from DKLOracle but with
    multi-task-specific output.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        layers = []
        d_in = input_dim
        for i in range(n_layers):
            d_out = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            d_in = d_out
        self.mlp = nn.Sequential(*layers)
        
        # Residual: project input to output_dim if dims differ
        self.residual = residual
        if residual:
            self.proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, input_dim) → h: (B, output_dim)"""
        h = self.mlp(z)
        if self.residual:
            h = h + self.proj(z)
        return h


class RegressionHead(nn.Module):
    """Single linear layer for pKd regression."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_trunk) → ŷ: (B,)"""
        return self.linear(h).squeeze(-1)


class ClassificationHead(nn.Module):
    """Single linear layer for active/inactive classification.
    
    Outputs logits (not probabilities) — apply sigmoid externally or
    use F.binary_cross_entropy_with_logits for numerically stable training.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_trunk) → logits: (B,)"""
        return self.linear(h).squeeze(-1)
    
    def predict_prob(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_trunk) → p: (B,) in [0, 1]"""
        return torch.sigmoid(self.forward(h))


class RankingHead(nn.Module):
    """Single linear layer for pairwise ranking scores."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_trunk) → s: (B,) ranking scores"""
        return self.linear(h).squeeze(-1)


class MultiTaskTrunk(nn.Module):
    """Multi-task trunk shaping module.
    
    Trains a shared trunk with auxiliary task heads (reg, cls, [rank]).
    The trunk features h_trunk are the primary output, intended to be
    fed into the best SP4 oracle head (DKL Ensemble).
    
    Two phases:
      v1 (enable_ranking=False): L = λ₁ L_reg + λ₂ L_cls
      v2 (enable_ranking=True):  L = λ₁ L_reg + λ₂ L_cls + λ₃ L_rank
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, trunk_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1, residual: bool = True,
                 activity_threshold: float = 7.0, enable_ranking: bool = False,
                 learned_weights: bool = False, cls_pos_weight: Optional[float] = None):
        super().__init__()
        self.trunk = SharedTrunk(input_dim, hidden_dim, trunk_dim, n_layers, dropout, residual)
        self.reg_head = RegressionHead(trunk_dim)
        self.cls_head = ClassificationHead(trunk_dim)
        self.rank_head = RankingHead(trunk_dim) if enable_ranking else None
        
        self.enable_ranking = enable_ranking
        self.threshold = activity_threshold
        
        # Optional: uncertainty-based learned task weights (Kendall 2018)
        self.learned_weights = learned_weights
        if learned_weights:
            n_tasks = 3 if enable_ranking else 2
            self.log_sigma = nn.Parameter(torch.zeros(n_tasks))
        
        # Optional: positive class weight for imbalanced classification
        self.cls_pos_weight = cls_pos_weight
    
    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, input_dim) frozen embeddings
        
        Returns:
            h_trunk: (B, trunk_dim) shaped features for downstream oracle
            reg_out: (B,) pKd predictions
            cls_logits: (B,) classification logits (not probabilities)
            rank_out: (B,) ranking scores, or None if ranking disabled
        """
        h = self.trunk(z)
        reg = self.reg_head(h)
        cls_logits = self.cls_head(h)
        rank = self.rank_head(h) if self.rank_head is not None else None
        return h, reg, cls_logits, rank
    
    def compute_loss(self, z: torch.Tensor, y: torch.Tensor,
                     groups: Optional[torch.Tensor] = None,
                     lambda_reg: float = 1.0, lambda_cls: float = 0.5,
                     lambda_rank: float = 0.3,
                     max_pairs_per_group: int = 50):
        """
        Compute weighted multi-task loss.
        
        Args:
            z: (B, input_dim) frozen embeddings
            y: (B,) pKd labels
            groups: (B,) integer group IDs for within-group pair construction
                    (required if enable_ranking=True)
            lambda_reg, lambda_cls, lambda_rank: fixed task weights
            max_pairs_per_group: cap on pairs per protein cluster per batch
        
        Returns:
            total_loss: scalar tensor
            loss_dict: {'L_reg': float, 'L_cls': float, 'L_rank': float,
                        'L_total': float, 'n_pairs': int}
        """
        h, reg_out, cls_logits, rank_out = self.forward(z)
        
        # ── Task 1: Regression (MSE) ──
        L_reg = F.mse_loss(reg_out, y)
        
        # ── Task 2: Classification (BCE with logits) ──
        c = (y >= self.threshold).float()
        pw = torch.tensor([self.cls_pos_weight], device=z.device, dtype=z.dtype) if self.cls_pos_weight else None
        L_cls = F.binary_cross_entropy_with_logits(cls_logits, c, pos_weight=pw)
        
        # ── Task 3: Ranking (BPR, optional) ──
        L_rank = torch.tensor(0.0, device=z.device)
        n_pairs = 0
        if self.enable_ranking and rank_out is not None:
            assert groups is not None, "groups required when ranking is enabled"
            pairs = self._make_grouped_pairs(y, groups, max_pairs_per_group)
            n_pairs = len(pairs)
            if n_pairs > 0:
                s_better = rank_out[pairs[:, 0]]
                s_worse  = rank_out[pairs[:, 1]]
                L_rank = -F.logsigmoid(s_better - s_worse).mean()
        
        # ── Combine ──
        if self.learned_weights:
            # Kendall (2018) homoscedastic uncertainty weighting
            precisions = torch.exp(-2 * self.log_sigma)  # 1 / σ²
            losses = [L_reg, L_cls] + ([L_rank] if self.enable_ranking else [])
            total = sum(p * l for p, l in zip(precisions, losses))
            total = total + self.log_sigma.sum()  # regularizer: log σ₁σ₂σ₃
        else:
            total = lambda_reg * L_reg + lambda_cls * L_cls
            if self.enable_ranking:
                total = total + lambda_rank * L_rank
        
        loss_dict = {
            'L_reg': L_reg.item(),
            'L_cls': L_cls.item(),
            'L_rank': L_rank.item(),
            'L_total': total.item(),
            'n_pairs': n_pairs,
        }
        if self.learned_weights:
            for i, name in enumerate(['sigma_reg', 'sigma_cls', 'sigma_rank'][:len(self.log_sigma)]):
                loss_dict[name] = torch.exp(self.log_sigma[i]).item()
        
        return total, loss_dict
    
    def _make_grouped_pairs(self, y: torch.Tensor, groups: torch.Tensor,
                            max_pairs_per_group: int = 50) -> torch.Tensor:
        """Generate (better, worse) index pairs within each group.
        
        Args:
            y: (B,) pKd labels
            groups: (B,) integer group IDs
            max_pairs_per_group: subsample cap per group
        
        Returns:
            pairs: (P, 2) long tensor, pairs[:, 0] = better, pairs[:, 1] = worse
        """
        device = y.device
        unique_groups = groups.unique()
        all_pairs = []
        
        for g in unique_groups:
            mask = (groups == g)
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue
            y_g = y[idx]
            # Build all ordered pairs (i, j) where y_g[i] > y_g[j]
            n = len(idx)
            ii, jj = torch.meshgrid(torch.arange(n, device=device),
                                     torch.arange(n, device=device), indexing='ij')
            valid = y_g[ii] > y_g[jj]
            local_pairs = torch.stack([ii[valid], jj[valid]], dim=1)
            # Map back to global indices
            global_pairs = torch.stack([idx[local_pairs[:, 0]], idx[local_pairs[:, 1]]], dim=1)
            # Subsample if too many
            if len(global_pairs) > max_pairs_per_group:
                perm = torch.randperm(len(global_pairs), device=device)[:max_pairs_per_group]
                global_pairs = global_pairs[perm]
            all_pairs.append(global_pairs)
        
        if not all_pairs:
            return torch.zeros(0, 2, dtype=torch.long, device=device)
        return torch.cat(all_pairs, dim=0)
    
    def extract_trunk_features(self, z: torch.Tensor) -> np.ndarray:
        """Extract h_trunk as numpy array (for feeding to SP4 oracle head).
        
        Args:
            z: (N, input_dim) frozen embeddings (numpy or tensor)
        
        Returns:
            h: (N, trunk_dim) numpy float32 array
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        self.eval()
        with torch.no_grad():
            h = self.trunk(z.to(next(self.parameters()).device))
        return h.cpu().numpy()


class MultiTaskHybridOracle:
    """Multi-task shaped trunk + best SP4 oracle head.
    
    Complete two-stage pipeline:
      Stage 1: Train MultiTaskTrunk on frozen embeddings with multi-task loss
      Stage 2: Freeze trunk, extract h_trunk, fit SP4 OracleHead on top
    
    At inference time:
      z → trunk → h_trunk → oracle_head.predict() → OracleResult
      (optionally also cls_head and rank_head outputs in aux dict)
    """
    
    def __init__(self, multi_task: MultiTaskTrunk, oracle_head: OracleHead):
        self.multi_task = multi_task
        self.oracle_head = oracle_head  # e.g. DKLEnsembleOracle from SP4
    
    def train_trunk(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    groups_train: Optional[np.ndarray] = None,
                    groups_val: Optional[np.ndarray] = None,
                    n_epochs: int = 200, batch_size: int = 256,
                    lr: float = 1e-3, weight_decay: float = 1e-4,
                    lambda_reg: float = 1.0, lambda_cls: float = 0.5,
                    lambda_rank: float = 0.3,
                    patience: int = 20,
                    device: str = "cuda") -> dict:
        """Stage 1: Train multi-task trunk end-to-end.
        
        Args:
            X_train: (N_train, 128) frozen embeddings
            y_train: (N_train,) pKd labels
            X_val: (N_val, 128) validation embeddings
            y_val: (N_val,) validation labels
            groups_train: (N_train,) cluster IDs (required if ranking enabled)
            groups_val: (N_val,) cluster IDs for validation (required if ranking enabled)
            n_epochs: max training epochs
            batch_size: minibatch size (or group-filled batch size for v2)
            lr: AdamW learning rate
            weight_decay: L2 regularization
            lambda_reg, lambda_cls, lambda_rank: fixed task weights
            patience: early stopping patience on val total loss
            device: 'cuda' or 'cpu'
        
        Returns:
            history: dict with keys 'train_loss', 'val_loss', 'train_L_reg',
                     'train_L_cls', 'train_L_rank', 'val_L_reg', 'val_L_cls',
                     'val_L_rank', 'best_epoch'
        """
        self.multi_task = self.multi_task.to(device)
        optimizer = torch.optim.AdamW(self.multi_task.parameters(), lr=lr,
                                       weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # Convert to tensors
        Xt = torch.from_numpy(X_train).float().to(device)
        yt = torch.from_numpy(y_train).float().to(device)
        Xv = torch.from_numpy(X_val).float().to(device)
        yv = torch.from_numpy(y_val).float().to(device)
        gt = torch.from_numpy(groups_train).long().to(device) if groups_train is not None else None
        gv = torch.from_numpy(groups_val).long().to(device) if groups_val is not None else None
        
        history = {k: [] for k in ['train_loss', 'val_loss', 'train_L_reg',
                                     'train_L_cls', 'train_L_rank',
                                     'val_L_reg', 'val_L_cls', 'val_L_rank']}
        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        
        N = len(Xt)
        for epoch in range(n_epochs):
            # ── Train ──
            self.multi_task.train()
            
            # v2 (ranking enabled): use GroupedBatchSampler so each batch
            # contains whole protein clusters for within-group pair construction.
            # v1: standard random-permutation batching.
            if self.multi_task.enable_ranking and groups_train is not None:
                sampler = GroupedBatchSampler(groups_train, batch_size=batch_size,
                                              min_group_size=2, drop_last=False)
                batches = list(sampler)
            else:
                perm = torch.randperm(N, device=device)
                batches = [perm[start:start + batch_size] for start in range(0, N, batch_size)]
            
            epoch_losses = []
            for idx in batches:
                if isinstance(idx, list):
                    idx = torch.tensor(idx, dtype=torch.long, device=device)
                z_b, y_b = Xt[idx], yt[idx]
                g_b = gt[idx] if gt is not None else None
                loss, ld = self.multi_task.compute_loss(
                    z_b, y_b, groups=g_b,
                    lambda_reg=lambda_reg, lambda_cls=lambda_cls,
                    lambda_rank=lambda_rank)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multi_task.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(ld)
            scheduler.step()
            
            # Aggregate epoch train metrics
            for key in ['L_reg', 'L_cls', 'L_rank', 'L_total']:
                avg = np.mean([d[key] for d in epoch_losses])
                if key == 'L_total':
                    history['train_loss'].append(avg)
                else:
                    history[f'train_{key}'].append(avg)
            
            # ── Validate ──
            self.multi_task.eval()
            with torch.no_grad():
                val_loss, vld = self.multi_task.compute_loss(
                    Xv, yv, groups=gv,
                    lambda_reg=lambda_reg, lambda_cls=lambda_cls,
                    lambda_rank=lambda_rank)
            history['val_loss'].append(vld['L_total'])
            history['val_L_reg'].append(vld['L_reg'])
            history['val_L_cls'].append(vld['L_cls'])
            history['val_L_rank'].append(vld['L_rank'])
            
            # Early stopping
            if vld['L_total'] < best_val_loss:
                best_val_loss = vld['L_total']
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in
                              self.multi_task.state_dict().items()}
            elif epoch - best_epoch >= patience:
                break
        
        # Restore best checkpoint
        if best_state is not None:
            self.multi_task.load_state_dict(best_state)
        self.multi_task = self.multi_task.to(device)
        history['best_epoch'] = best_epoch
        return history
    
    def train_oracle(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> dict:
        """Stage 2: Freeze trunk, extract h_trunk, fit SP4 oracle head.
        
        Args:
            X_train, y_train: raw embeddings and labels (pre-trunk)
            X_val, y_val: validation set
            **kwargs: passed to oracle_head.fit()
        
        Returns:
            oracle_history: dict from OracleHead.fit()
        """
        self.multi_task.eval()
        h_train = self.multi_task.extract_trunk_features(X_train)
        h_val   = self.multi_task.extract_trunk_features(X_val)
        return self.oracle_head.fit(h_train, y_train, h_val, y_val, **kwargs)
    
    def predict(self, X: np.ndarray) -> OracleResult:
        """Full inference: z → trunk → oracle_head → OracleResult.
        
        Also attaches cls_prob and rank_score in result.aux.
        """
        self.multi_task.eval()
        h = self.multi_task.extract_trunk_features(X)
        result = self.oracle_head.predict(h)
        
        # Attach auxiliary head outputs
        z_t = torch.from_numpy(X).float()
        with torch.no_grad():
            device = next(self.multi_task.parameters()).device
            _, _, cls_logits, rank_out = self.multi_task(z_t.to(device))
            result.aux['cls_prob'] = torch.sigmoid(cls_logits).cpu().numpy()
            if rank_out is not None:
                result.aux['rank_score'] = rank_out.cpu().numpy()
        
        return result
    
    def predict_for_fusion(self, X: np.ndarray) -> OracleResult:
        """Expensive path with full Jacobian ∂μ/∂z for fusion.py.
        
        fusion.fuse_uncertainties() requires the Jacobian in the **original
        embedding space z**, not in the shaped trunk space h.  We therefore
        use autograd to differentiate through the entire trunk + oracle
        forward graph so that:
        
            J_fusion = ∂μ/∂z = (∂μ/∂h) · (∂h/∂z)   (chain rule)
        
        Approach: enable requires_grad on z, run trunk → h → oracle
        forward, then torch.autograd.functional.jacobian or row-by-row
        backward to collect ∂μ_i/∂z_i.
        """
        self.multi_task.eval()
        device = next(self.multi_task.parameters()).device
        z_t = torch.from_numpy(X).float().to(device).requires_grad_(True)
        
        # Forward through trunk (keep graph)
        h = self.multi_task.trunk(z_t)          # (N, trunk_dim), differentiable
        
        # Forward through oracle head to get μ
        # oracle_head.predict_for_fusion expects numpy → internally builds its
        # own torch graph.  Instead, call the oracle's internal forward so we
        # can chain autograd.  If the oracle head does not expose a torch
        # forward, fall back to the two-stage Jacobian product:
        #   J_oracle = oracle_head.predict_for_fusion(h).jacobian   # ∂μ/∂h
        #   J_trunk  = autograd over trunk                          # ∂h/∂z
        #   J_full   = J_oracle @ J_trunk                           # ∂μ/∂z
        h_np = h.detach().cpu().numpy()
        oracle_result = self.oracle_head.predict_for_fusion(h_np)
        
        # J_oracle: (N, trunk_dim) = ∂μ/∂h  (from oracle head)
        J_oracle = oracle_result.jacobian  # (N, d_h)
        
        if J_oracle is not None:
            # Compute J_trunk = ∂h/∂z row-by-row via backward passes
            N, d_z = z_t.shape
            d_h = h.shape[1]
            J_trunk = torch.zeros(N, d_h, d_z, device=device)
            for k in range(d_h):
                grad_outputs = torch.zeros_like(h)
                grad_outputs[:, k] = 1.0
                g = torch.autograd.grad(h, z_t, grad_outputs=grad_outputs,
                                         retain_graph=True, create_graph=False)[0]
                J_trunk[:, k, :] = g  # (N, d_z)
            
            # J_full[i] = J_oracle[i] @ J_trunk[i]  →  ∂μ_i/∂z_i  (N, d_z)
            J_oracle_t = torch.from_numpy(J_oracle).float().to(device)  # (N, d_h)
            J_full = torch.einsum('nh,nhz->nz', J_oracle_t, J_trunk)   # (N, d_z)
            oracle_result.jacobian = J_full.detach().cpu().numpy()
        
        return oracle_result
    
    # ── Engineering note on predict_for_fusion ──
    # The first version uses two-stage Jacobian composition:
    #   J_oracle (∂μ/∂h from oracle, numpy) × J_trunk (∂h/∂z from autograd, torch)
    # This is a "half-graph" approach — mathematically correct but more fragile
    # than a single end-to-end autograd graph, since the oracle's Jacobian is
    # detached numpy that we re-attach via einsum.  This is acceptable for v1
    # because predict_for_fusion is only called on the **fusion path**, never
    # during standard evaluation (predict() has no Jacobian).
    #
    # The O(d_h) backward passes (one per trunk output dim) are also non-trivial
    # when trunk_dim=128.  If later profiling shows this is too slow, the
    # recommended fallback is to propagate Σ_gen into trunk space instead:
    #   Σ_gen_h = J_trunk @ Σ_gen @ J_trunk^T   (only needs J_trunk once)
    # and call fuse_uncertainties in h-space directly, avoiding the full
    # chain-rule product entirely.

    def save(self, path: str):
        """Save trunk + oracle head to directory."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.multi_task.state_dict(), os.path.join(path, "trunk.pt"))
        self.oracle_head.save(os.path.join(path, "oracle_head"))
    
    def load(self, path: str):
        """Load trunk + oracle head from directory."""
        import os
        self.multi_task.load_state_dict(
            torch.load(os.path.join(path, "trunk.pt"), weights_only=False))
        self.oracle_head.load(os.path.join(path, "oracle_head"))
```

### 3.2 Integration with SP4 Oracle Head

The multi-task trunk produces shaped features $h \in \mathbb{R}^{d_{\text{trunk}}}$. The best SP4 oracle head then operates on these features:

$$
\mu_{\text{oracle}}, \sigma^2_{\text{oracle}} = \text{DKLEnsemble}(h_{\text{trunk}})
$$

**Training protocol** (sequential, recommended):

| Step | Action | Input | Output |
|------|--------|-------|--------|
| 1 | Load frozen embeddings | `frozen_embeddings.npz` | `X_train (N, 128)`, `y_train (N,)` |
| 2 | Load cluster assignments | `cluster_assignments.csv` + split CSV | `groups_train (N,)` |
| 3 | Train MultiTaskTrunk | `X_train, y_train, [groups_train]` | `trunk.pt` checkpoint |
| 4 | Extract shaped features | `trunk(X_train) → h_train` | `h_train (N, trunk_dim)` |
| 5 | Fit DKL Ensemble on shaped features | `h_train, y_train, h_val, y_val` | `oracle_head/` checkpoint |
| 6 | Evaluate | `trunk(X_test) → h_test → oracle.predict(h_test)` | EvalResults on CASF-2016 |

**Key constraint**: The DKL Ensemble's `input_dim` must match `trunk_dim`. If `trunk_dim = 128` (default), no change needed. If `trunk_dim ≠ 128`, create the DKL Ensemble with `input_dim=trunk_dim`.

**Compatibility**: This reuses the existing `OracleHead` interface from `bayesdiff/oracle_interface.py` — the oracle head sees $h_{\text{trunk}}$ exactly as it currently sees raw embeddings. No changes to `hybrid_oracle.py` required.

### 3.3 Training Hyperparameters

| Parameter | Default | Search Range | Notes |
|-----------|---------|-------------|-------|
| `trunk_dim` | 128 | {64, 128, 256} | Must match oracle input_dim |
| `hidden_dim` | 256 | {128, 256, 512} | Trunk MLP width |
| `n_layers` | 2 | {1, 2, 3} | Trunk depth |
| `dropout` | 0.1 | {0.0, 0.1, 0.2} | |
| `residual` | True | {True, False} | Skip connection in trunk |
| `lr` | 1e-3 | {1e-4, 5e-4, 1e-3} | AdamW |
| `weight_decay` | 1e-4 | {0, 1e-5, 1e-4} | L2 regularization |
| `batch_size` | 256 | {128, 256, 512} | Larger for v1, smaller for v2 (group-filled) |
| `n_epochs` | 200 | — | With early stopping |
| `patience` | 20 | — | On val total loss |
| `lambda_reg` | 1.0 | fixed | Regression always weight 1 |
| `lambda_cls` | 0.5 | {0.1, 0.3, 0.5, 1.0} | Tier 3 search |
| `lambda_rank` | 0.3 | {0.1, 0.3, 0.5} | Tier 3 search, v2 only |
| `threshold` | 7.0 | {6.0, 6.5, 7.0, 7.5, 8.0} | Tier 4 search |
| `max_pairs_per_group` | 50 | — | Cap on ranking pairs per cluster |

**Optimizer**: AdamW with cosine annealing schedule (T_max = n_epochs)  
**Gradient clipping**: max_norm = 1.0  
**Early stopping**: based on val total loss (reg + cls [+ rank]), patience = 20 epochs  
**Best model selection**: restore best val-loss checkpoint after training

### 3.4 Enhanced $P_{\text{success}}$ Computation

The classification head and the oracle head both produce probability-like outputs from the same trunk. Direct linear mixing would double-count shared information. Instead, use a held-out calibration combiner:

$$
P_{\text{success}}^{\text{final}} = g(P_{\text{oracle}},\; p_{\text{cls}},\; s_{\text{rank}})
$$

where:
- $P_{\text{oracle}}$: calibrated success probability from the best SP4 oracle head (via existing `IsotonicCalibrator` in `bayesdiff/calibration.py`)
- $p_{\text{cls}}$: classification head output (sigmoid of logits)
- $s_{\text{rank}}$: ranking score (optional, from 05-v2)
- $g$: a monotonic calibration combiner trained on the **calibration split only** — not part of the main model

#### Calibration Combiner Implementation

```python
class CalibrationCombiner:
    """Combines oracle P_success with auxiliary head outputs.
    
    Trained on a held-out calibration subset (e.g., 20% of validation).
    Two variants:
      - 'isotonic': Isotonic regression on the average of inputs
      - 'logistic': Logistic regression on stacked features
    """
    
    def __init__(self, method: str = 'logistic'):
        self.method = method
        if method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1.0, max_iter=1000)
        elif method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, P_oracle: np.ndarray, p_cls: np.ndarray,
            y_true: np.ndarray, threshold: float = 7.0,
            s_rank: Optional[np.ndarray] = None):
        """Fit on calibration split.
        
        Args:
            P_oracle: (N_cal,) oracle success probability
            p_cls: (N_cal,) classification head probability
            y_true: (N_cal,) true pKd labels
            threshold: activity threshold for binary target
            s_rank: (N_cal,) optional ranking scores
        """
        y_binary = (y_true >= threshold).astype(np.float32)
        if self.method == 'logistic':
            features = [P_oracle.reshape(-1, 1), p_cls.reshape(-1, 1)]
            if s_rank is not None:
                features.append(s_rank.reshape(-1, 1))
            X = np.hstack(features)
            self.model.fit(X, y_binary)
        elif self.method == 'isotonic':
            # Simple average → isotonic
            avg = (P_oracle + p_cls) / 2
            self.model.fit(avg, y_binary)
    
    def predict(self, P_oracle: np.ndarray, p_cls: np.ndarray,
                s_rank: Optional[np.ndarray] = None) -> np.ndarray:
        """Return calibrated combined P_success."""
        if self.method == 'logistic':
            features = [P_oracle.reshape(-1, 1), p_cls.reshape(-1, 1)]
            if s_rank is not None:
                features.append(s_rank.reshape(-1, 1))
            X = np.hstack(features)
            return self.model.predict_proba(X)[:, 1]
        elif self.method == 'isotonic':
            avg = (P_oracle + p_cls) / 2
            return self.model.predict(avg)
```

**Calibration split strategy**: Carve 20% of the validation set as a calibration-only subset. This subset is not used for early stopping or hyperparameter selection — only for fitting $g$.

### 3.5 Within-Group NDCG Metric

Standard NDCG computed globally is meaningless for BayesDiff because it mixes cross-target rankings. Within-group NDCG evaluates ranking quality only within biologically meaningful groups.

```python
def within_group_ndcg(y_true: np.ndarray, y_pred: np.ndarray,
                       groups: np.ndarray, k: int = 10) -> dict:
    """Compute NDCG@k averaged over protein clusters.
    
    Args:
        y_true: (N,) true pKd values
        y_pred: (N,) predicted pKd or ranking scores
        groups: (N,) protein cluster IDs
        k: cutoff for NDCG
    
    Returns:
        dict with 'ndcg_mean', 'ndcg_std', 'n_groups_evaluated',
              'ndcg_per_group' (dict: group_id → ndcg)
    """
    from sklearn.metrics import ndcg_score
    unique_groups = np.unique(groups)
    ndcgs = {}
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2:
            continue  # need at least 2 samples for ranking
        yt = y_true[mask]
        yp = y_pred[mask]
        # ndcg_score expects 2D arrays
        cutoff = min(k, len(yt))
        ndcgs[int(g)] = ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=cutoff)
    
    vals = list(ndcgs.values())
    return {
        'ndcg_mean': np.mean(vals) if vals else 0.0,
        'ndcg_std': np.std(vals) if vals else 0.0,
        'n_groups_evaluated': len(vals),
        'ndcg_per_group': ndcgs,
    }
```

### 3.6 New Pipeline Script: `scripts/pipeline/s20_train_multitask_trunk.py`

```python
"""
Train multi-task trunk shaping module and connect to best SP4 oracle head.

Two-stage pipeline:
  Stage 1: Train MultiTaskTrunk (reg + cls [+ rank]) on frozen embeddings
  Stage 2: Freeze trunk, extract h_trunk, fit DKL Ensemble on top
  Stage 3: Evaluate on CASF-2016 test set
  Stage 4: (Optional) Fit calibration combiner

Usage:
    # Phase v1 (reg + cls only):
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --clusters data/pdbbind_v2020/cluster_assignments.csv \
        --split data/pdbbind_v2020/splits/split_s0_seed42.csv \
        --output results/stage2/multitask_trunk/v1 \
        --phase v1 \
        --hidden_dim 256 --trunk_dim 128 \
        --lambda_reg 1.0 --lambda_cls 0.5 \
        --threshold 7.0 --n_epochs 200 \
        --oracle_head dkl_ensemble \
        --device cuda

    # Phase v2 (reg + cls + group-aware ranking):
    python scripts/pipeline/s20_train_multitask_trunk.py \
        --embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --clusters data/pdbbind_v2020/cluster_assignments.csv \
        --split data/pdbbind_v2020/splits/split_s0_seed42.csv \
        --output results/stage2/multitask_trunk/v2 \
        --phase v2 \
        --hidden_dim 256 --trunk_dim 128 \
        --lambda_reg 1.0 --lambda_cls 0.5 --lambda_rank 0.3 \
        --threshold 7.0 --n_epochs 200 \
        --oracle_head dkl_ensemble \
        --device cuda
"""
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from bayesdiff.multi_task import MultiTaskTrunk, MultiTaskHybridOracle
from bayesdiff.hybrid_oracle import DKLEnsembleOracle
from bayesdiff.oracle_interface import OracleResult
from bayesdiff.evaluate import evaluate_all
from bayesdiff.calibration import IsotonicCalibrator
from bayesdiff.fusion import fuse_uncertainties

log = logging.getLogger(__name__)


def main(args):
    # ── Seed everything for reproducibility ───────────────────────
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info(f"Seed: {args.seed}")
    
    # ── 0. Load data ──────────────────────────────────────────────
    log.info(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.float32)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.float32)
    log.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Load cluster assignments (for v2 ranking)
    groups_train = None
    groups_val = None
    if args.phase == 'v2':
        log.info(f"Loading clusters from {args.clusters}")
        clusters = pd.read_csv(args.clusters)
        split_df = pd.read_csv(args.split)
        code_to_cluster = dict(zip(clusters["pdb_code"], clusters["cluster_id"]))
        
        train_codes = split_df[split_df["split"] == "train"]["pdb_code"].values
        groups_train = np.array([code_to_cluster.get(c, -1) for c in train_codes])
        valid_mask = groups_train >= 0
        log.info(f"  Cluster coverage (train): {valid_mask.mean():.1%} of train samples")
        
        val_codes = split_df[split_df["split"] == "val"]["pdb_code"].values
        groups_val = np.array([code_to_cluster.get(c, -1) for c in val_codes])
        valid_mask_v = groups_val >= 0
        log.info(f"  Cluster coverage (val): {valid_mask_v.mean():.1%} of val samples")
    
    # Class balance check
    pos_rate = (y_train >= args.threshold).mean()
    log.info(f"  Classification positive rate (tau={args.threshold}): {pos_rate:.3f}")
    cls_pos_weight = None
    if pos_rate < 0.25:
        cls_pos_weight = (1 - pos_rate) / pos_rate
        log.info(f"  Using pos_weight={cls_pos_weight:.2f} for imbalanced BCE")
    
    # ── 1. Create trunk ───────────────────────────────────────────
    trunk = MultiTaskTrunk(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        trunk_dim=args.trunk_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        residual=True,
        activity_threshold=args.threshold,
        enable_ranking=(args.phase == 'v2'),
        learned_weights=args.learned_weights,
        cls_pos_weight=cls_pos_weight,
    )
    log.info(f"MultiTaskTrunk: {sum(p.numel() for p in trunk.parameters())} params")
    
    # ── 2. Create oracle head ─────────────────────────────────────
    oracle_head = DKLEnsembleOracle(
        input_dim=args.trunk_dim,
        n_members=5,
        bootstrap=True,
        feature_dim=32,
        n_inducing=512,
        residual=True,
        device=args.device,
    )
    
    hybrid = MultiTaskHybridOracle(trunk, oracle_head)
    
    # ── A5.0 shortcut: skip trunk, fit oracle directly ─────────
    if args.no_trunk:
        log.info("--no_trunk: skipping Stage 1, fitting oracle directly on frozen embeddings")
        history = {}  # no trunk training history
        oracle_head.fit(X_train, y_train, X_val, y_val)
        
        log.info("Evaluating on CASF-2016 test set (no trunk)...")
        result = oracle_head.predict(X_test)
        eval_result = evaluate_all(
            mu_pred=result.mu,
            sigma_pred=np.sqrt(result.sigma2),
            y_true=y_test,
        )
        log.info(f"  Spearman rho: {eval_result.spearman_rho:.3f}")
        log.info(f"  NLL: {eval_result.nll:.3f}")
        log.info(f"  RMSE: {eval_result.rmse:.3f}")
        
        # Save directly and return (no trunk checkpoint)
        output_dir = Path(args.output) / f"seed{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        oracle_head.save(str(output_dir / "oracle_head"))
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(vars(eval_result), f, indent=2, default=str)
        log.info(f"Saved to {output_dir}")
        return
    
    # ── 3. Stage 1: Train trunk ───────────────────────────────────
    log.info("Stage 1: Training multi-task trunk...")
    history = hybrid.train_trunk(
        X_train, y_train, X_val, y_val,
        groups_train=groups_train,
        groups_val=groups_val,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
        lambda_rank=args.lambda_rank,
        patience=args.patience,
        device=args.device,
    )
    log.info(f"  Best epoch: {history['best_epoch']}")
    log.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    log.info(f"  Best val loss: {min(history['val_loss']):.4f}")
    
    # ── 4. Stage 2: Fit oracle head on shaped features ────────────
    log.info("Stage 2: Fitting DKL Ensemble on shaped trunk features...")
    oracle_history = hybrid.train_oracle(X_train, y_train, X_val, y_val)
    
    # ── 5. Evaluate on CASF-2016 ──────────────────────────────────
    # Minimal evaluation loop: only metrics needed for Tier 1 go/no-go.
    # Calibrated P_success, CalibrationCombiner, reliability diagrams,
    # and EF@1% are deferred — they are NOT required for the Tier 1 gate,
    # which depends solely on ρ, R², RMSE, NLL, ECE, and ρ_{|err|,σ}.
    log.info("Evaluating on CASF-2016 test set...")
    result = hybrid.predict(X_test)
    
    eval_result = evaluate_all(
        mu_pred=result.mu,
        sigma_pred=np.sqrt(result.sigma2),
        y_true=y_test,
    )
    log.info(f"  Spearman rho: {eval_result.spearman_rho:.3f}")
    log.info(f"  NLL: {eval_result.nll:.3f}")
    log.info(f"  RMSE: {eval_result.rmse:.3f}")
    
    # TODO (post-Tier 1): Calibrated P_success + CalibrationCombiner
    # Implement after Tier 1 passes the go/no-go gate. Requires:
    #   - Gaussian tail P(pKd >= τ | μ, σ²) → raw P_success
    #   - IsotonicCalibrator fitted on val split
    #   - (optional) CalibrationCombiner(P_oracle, p_cls, [s_rank])
    #   - EF@1%, reliability diagram, AUPRC
    # These are needed for Tier 4 (A5.11) and paper figures, not for
    # the Tier 1 decision.
    
    # ── 6. Save ───────────────────────────────────────────────────
    output_dir = Path(args.output) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid.save(str(output_dir))
    
    with open(output_dir / "history.json", "w") as f:
        json.dump({k: v if not isinstance(v, list) else [float(x) for x in v]
                    for k, v in history.items()}, f, indent=2)
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(vars(eval_result), f, indent=2, default=str)
    
    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--clusters", default="data/pdbbind_v2020/cluster_assignments.csv")
    parser.add_argument("--split", default="data/pdbbind_v2020/splits/split_s0_seed42.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["v1", "v2"], default="v1")
    parser.add_argument("--oracle_head", default="dkl_ensemble")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--trunk_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lambda_rank", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learned_weights", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (3-seed protocol: 42, 123, 777)")
    parser.add_argument("--no_trunk", action="store_true",
                        help="Skip trunk training; fit oracle directly on frozen embeddings (A5.0)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(args)
```

### 3.7 SLURM Job Scripts

#### `slurm/s20_multitask_tier1.sh` — Tier 1 ablation (A5.1 vs A5.2)

```bash
#!/bin/bash
#SBATCH --job-name=sp05-tier1
#SBATCH --output=slurm_logs/sp05_tier1_%j.out
#SBATCH --error=slurm_logs/sp05_tier1_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

EMB="results/stage2/oracle_heads/frozen_embeddings.npz"
OUT="results/stage2/multitask_trunk"

echo "=== A5.0: True SP4 baseline (no trunk, frozen embedding → DKL Ensemble) ==="
python scripts/pipeline/s20_train_multitask_trunk.py \
    --embeddings $EMB --output $OUT/A5.0_no_trunk \
    --phase v1 --no_trunk --n_epochs 0 --device cuda

echo "=== A5.1: Regression-only trunk (baseline) ==="
python scripts/pipeline/s20_train_multitask_trunk.py \
    --embeddings $EMB --output $OUT/A5.1_reg_only \
    --phase v1 --lambda_cls 0.0 --n_epochs 200 --device cuda

echo "=== A5.2: Reg + Cls trunk (primary hypothesis) ==="
python scripts/pipeline/s20_train_multitask_trunk.py \
    --embeddings $EMB --output $OUT/A5.2_reg_cls \
    --phase v1 --lambda_cls 0.5 --n_epochs 200 --device cuda

echo "=== Compare results ==="
python -c "
import json
for name in ['A5.0_no_trunk', 'A5.1_reg_only', 'A5.2_reg_cls']:
    with open(f'$OUT/{name}/eval_results.json') as f:
        r = json.load(f)
    print(f'{name}: rho={r[\"spearman_rho\"]:.3f} NLL={r[\"nll\"]:.3f} RMSE={r[\"rmse\"]:.3f}')
"
```

#### `slurm/s20_multitask_tier2.sh` — Tier 2 ranking ablation (A5.3–A5.6)

```bash
#!/bin/bash
#SBATCH --job-name=sp05-tier2
#SBATCH --output=slurm_logs/sp05_tier2_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

EMB="results/stage2/oracle_heads/frozen_embeddings.npz"
CLU="data/pdbbind_v2020/cluster_assignments.csv"
OUT="results/stage2/multitask_trunk"

echo "=== A5.3: Reg + Rank (no cls) ==="
python scripts/pipeline/s20_train_multitask_trunk.py \
    --embeddings $EMB --clusters $CLU --output $OUT/A5.3_reg_rank \
    --phase v2 --lambda_cls 0.0 --lambda_rank 0.3 --device cuda

echo "=== A5.4: Reg + Cls + Rank (full three-task) ==="
python scripts/pipeline/s20_train_multitask_trunk.py \
    --embeddings $EMB --clusters $CLU --output $OUT/A5.4_reg_cls_rank \
    --phase v2 --lambda_cls 0.5 --lambda_rank 0.3 --device cuda

echo "=== A5.6: Naive in-batch ranking (ablation — should be worse) ==="
python scripts/pipeline/s20_train_multitask_trunk.py \
    --embeddings $EMB --clusters $CLU --output $OUT/A5.6_naive_ranking \
    --phase v2 --lambda_cls 0.5 --lambda_rank 0.3 \
    --naive_ranking --device cuda
```

---

## 4. Test Plan

### 4.1 Unit Tests: `tests/stage2/test_multi_task.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_trunk_output_shape` | Trunk: (B, d) → (B, d_trunk) |
| T1.2 | `test_trunk_residual_connection` | With residual=True, output ≠ MLP-only output |
| T1.3 | `test_reg_head_shape` | Regression head: (B, d_trunk) → (B,) |
| T1.4 | `test_cls_head_range` | ClassificationHead.predict_prob() ∈ [0, 1] |
| T1.5 | `test_cls_head_logits_unbounded` | ClassificationHead.forward() produces raw logits |
| T1.6 | `test_rank_head_shape` | Ranking head: (B, d_trunk) → (B,) |
| T1.7 | `test_multitask_forward_v1` | Reg + cls heads produce valid outputs; rank is None |
| T1.8 | `test_multitask_forward_v2` | All three heads produce valid outputs simultaneously |
| T1.9 | `test_grouped_pair_generation` | Pair sampling only produces (i, j) pairs within the same group |
| T1.10 | `test_no_cross_group_pairs` | No pairs exist where group(i) ≠ group(j) |
| T1.11 | `test_pair_generation_empty_group` | Groups with < 2 members produce no pairs |
| T1.12 | `test_pair_subsample_cap` | max_pairs_per_group is respected |
| T1.13 | `test_bpr_loss_correct_gradient` | BPR loss is lower when correct pairs are scored higher |
| T1.14 | `test_cls_loss_correct_labels` | Low loss when predictions match true active/inactive labels |
| T1.15 | `test_joint_loss_v1` | Total loss = λ₁L_reg + λ₂L_cls (no ranking term) |
| T1.16 | `test_joint_loss_v2` | Total loss = λ₁L_reg + λ₂L_cls + λ₃L_rank |
| T1.17 | `test_joint_loss_ranking_requires_groups` | AssertionError if ranking enabled but groups=None |
| T1.18 | `test_learned_task_weights` | Uncertainty-based weights are learnable and positive |
| T1.19 | `test_learned_weights_gradient` | log_sigma receives gradients during backward |
| T1.20 | `test_gradient_flow_all_heads` | All parameters in all heads receive non-zero gradients |
| T1.21 | `test_training_convergence` | Joint loss decreases > 50% over 50 epochs on synthetic data |
| T1.22 | `test_trunk_features_extractable` | h_trunk can be detached as numpy and shape matches |
| T1.23 | `test_cls_pos_weight` | pos_weight changes loss value for imbalanced labels |
| T1.24 | `test_loss_dict_keys` | loss_dict contains all expected keys |

```python
import pytest
import torch
import numpy as np

from bayesdiff.multi_task import (
    SharedTrunk, RegressionHead, ClassificationHead, RankingHead,
    MultiTaskTrunk, MultiTaskHybridOracle,
)

# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def trunk_v1():
    return MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=False)

@pytest.fixture
def trunk_v2():
    return MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)

@pytest.fixture
def synthetic_data():
    torch.manual_seed(42)
    B = 64
    z = torch.randn(B, 128)
    w = torch.randn(128)
    y = z @ w / 128 * 3 + 7  # pKd centered around 7
    groups = torch.tensor([i % 8 for i in range(B)])  # 8 groups of 8
    return z, y, groups


# ── T1.1: Trunk output shape ─────────────────────────────────────

def test_trunk_output_shape():
    trunk = SharedTrunk(input_dim=128, hidden_dim=256, output_dim=64)
    z = torch.randn(32, 128)
    h = trunk(z)
    assert h.shape == (32, 64)


# ── T1.2: Residual connection ────────────────────────────────────

def test_trunk_residual_connection():
    z = torch.randn(16, 128)
    trunk_res = SharedTrunk(128, 256, 64, residual=True)
    trunk_nores = SharedTrunk(128, 256, 64, residual=False)
    # Copy MLP weights
    trunk_nores.mlp.load_state_dict(trunk_res.mlp.state_dict())
    h_res = trunk_res(z)
    h_nores = trunk_nores(z)
    assert not torch.allclose(h_res, h_nores, atol=1e-6)


# ── T1.4: Classification head range ──────────────────────────────

def test_cls_head_range():
    head = ClassificationHead(64)
    h = torch.randn(100, 64) * 10  # large inputs
    probs = head.predict_prob(h)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert probs.shape == (100,)


# ── T1.7: v1 forward (no ranking) ────────────────────────────────

def test_multitask_forward_v1(trunk_v1):
    z = torch.randn(32, 128)
    h, reg, cls, rank = trunk_v1(z)
    assert h.shape == (32, 64)
    assert reg.shape == (32,)
    assert cls.shape == (32,)
    assert rank is None


# ── T1.8: v2 forward (with ranking) ──────────────────────────────

def test_multitask_forward_v2(trunk_v2):
    z = torch.randn(32, 128)
    h, reg, cls, rank = trunk_v2(z)
    assert h.shape == (32, 64)
    assert reg.shape == (32,)
    assert cls.shape == (32,)
    assert rank is not None
    assert rank.shape == (32,)


# ── T1.9: Grouped pair generation ────────────────────────────────

def test_grouped_pair_generation():
    y = torch.tensor([8.0, 6.0, 7.5, 5.0, 9.0, 4.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups)
    
    for i, j in pairs:
        assert y[i] > y[j], f"Pair ({i},{j}): y[{i}]={y[i]} should > y[{j}]={y[j]}"
        assert groups[i] == groups[j], (
            f"Cross-group pair: group[{i}]={groups[i]} != group[{j}]={groups[j]}"
        )
    
    # Group 0 has 3 members → C(3,2)=3 ordered pairs with distinct values
    # Group 1 has 3 members → 3 ordered pairs
    # Total = 6
    assert len(pairs) == 6


# ── T1.10: No cross-group pairs ──────────────────────────────────

def test_no_cross_group_pairs():
    torch.manual_seed(0)
    y = torch.randn(100) * 3 + 7
    groups = torch.randint(0, 10, (100,))
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups)
    
    for i, j in pairs:
        assert groups[i] == groups[j]


# ── T1.11: Empty groups produce no pairs ──────────────────────────

def test_pair_generation_empty_group():
    y = torch.tensor([7.0, 6.0, 8.0])
    groups = torch.tensor([0, 1, 2])  # each group has 1 member
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups)
    assert len(pairs) == 0


# ── T1.12: Pair subsample cap ────────────────────────────────────

def test_pair_subsample_cap():
    y = torch.arange(20, dtype=torch.float)  # 20 distinct values
    groups = torch.zeros(20, dtype=torch.long)  # all same group → C(20,2)=190 pairs
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    pairs = trunk._make_grouped_pairs(y, groups, max_pairs_per_group=30)
    assert len(pairs) <= 30


# ── T1.13: BPR loss gradient direction ───────────────────────────

def test_bpr_loss_correct_gradient():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    z = torch.randn(10, 128)
    y = torch.arange(10, dtype=torch.float)
    groups = torch.zeros(10, dtype=torch.long)
    
    # Compute loss, verify it's a valid scalar
    loss, ld = trunk.compute_loss(z, y, groups=groups)
    assert loss.requires_grad
    assert ld['n_pairs'] > 0
    assert ld['L_rank'] > 0


# ── T1.15: v1 joint loss composition ─────────────────────────────

def test_joint_loss_v1():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=False)
    z = torch.randn(32, 128)
    y = torch.randn(32) * 2 + 7
    
    loss, ld = trunk.compute_loss(z, y, lambda_reg=1.0, lambda_cls=0.5)
    
    # L_rank should be 0 in v1
    assert ld['L_rank'] == 0.0
    assert ld['n_pairs'] == 0
    # Total should be λ₁*L_reg + λ₂*L_cls
    expected = 1.0 * ld['L_reg'] + 0.5 * ld['L_cls']
    assert abs(ld['L_total'] - expected) < 1e-4


# ── T1.17: Ranking requires groups ───────────────────────────────

def test_joint_loss_ranking_requires_groups():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, enable_ranking=True)
    z = torch.randn(32, 128)
    y = torch.randn(32) * 2 + 7
    
    with pytest.raises(AssertionError):
        trunk.compute_loss(z, y, groups=None)


# ── T1.20: Gradient flow to all heads ─────────────────────────────

def test_gradient_flow_all_heads(trunk_v2, synthetic_data):
    z, y, groups = synthetic_data
    loss, _ = trunk_v2.compute_loss(z, y, groups=groups)
    loss.backward()
    
    for name, param in trunk_v2.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ── T1.21: Training convergence ──────────────────────────────────

def test_training_convergence():
    torch.manual_seed(42)
    X = torch.randn(200, 128)
    w = torch.randn(128)
    y = X @ w / 128 * 3 + 7  # pKd centered around 7
    
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64, activity_threshold=7.0)
    optimizer = torch.optim.Adam(trunk.parameters(), lr=1e-3)
    
    losses = []
    for epoch in range(50):
        total_loss, _ = trunk.compute_loss(X, y)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    
    assert losses[-1] < losses[0] * 0.5, (
        f"Loss did not decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


# ── T1.22: Feature extraction ────────────────────────────────────

def test_trunk_features_extractable(trunk_v1):
    z_np = np.random.randn(50, 128).astype(np.float32)
    h = trunk_v1.extract_trunk_features(z_np)
    assert isinstance(h, np.ndarray)
    assert h.shape == (50, 64)
    assert h.dtype == np.float32


# ── T1.24: Loss dict keys ────────────────────────────────────────

def test_loss_dict_keys(trunk_v2, synthetic_data):
    z, y, groups = synthetic_data
    _, ld = trunk_v2.compute_loss(z, y, groups=groups)
    expected_keys = {'L_reg', 'L_cls', 'L_rank', 'L_total', 'n_pairs'}
    assert expected_keys.issubset(set(ld.keys()))
```

### 4.2 Integration Tests: `tests/stage2/test_multi_task_integration.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_trunk_to_dkl_ensemble` | Trunk features → DKL Ensemble training succeeds |
| T2.2 | `test_trunk_to_oracle_interface` | h_trunk satisfies OracleHead.fit() input contract |
| T2.3 | `test_jacobian_through_trunk` | Jacobian ∂μ/∂z through trunk+oracle works for Delta method |
| T2.4 | `test_calibration_combiner` | Combiner g(P_oracle, p_cls) produces valid probabilities |
| T2.5 | `test_full_pipeline_v1` | End-to-end: trunk(reg+cls) → DKL Ensemble → evaluation |
| T2.6 | `test_shaped_vs_unshaped` | Shaped trunk ≥ unshaped trunk on at least one UQ metric |
| T2.7 | `test_save_load_roundtrip` | MultiTaskHybridOracle save → load → predict matches |
| T2.8 | `test_within_group_ndcg` | within_group_ndcg produces valid [0,1] scores per group |

```python
import pytest
import torch
import numpy as np
import tempfile

from bayesdiff.multi_task import MultiTaskTrunk, MultiTaskHybridOracle
from bayesdiff.hybrid_oracle import DKLEnsembleOracle
from bayesdiff.oracle_interface import OracleResult


# ── T2.1: Trunk → DKL Ensemble ───────────────────────────────────

def test_trunk_to_dkl_ensemble():
    """Full two-stage training on tiny synthetic data."""
    torch.manual_seed(42)
    np.random.seed(42)
    N_train, N_val = 200, 50
    d = 128
    
    X_train = np.random.randn(N_train, d).astype(np.float32)
    y_train = np.random.randn(N_train).astype(np.float32) * 2 + 7
    X_val = np.random.randn(N_val, d).astype(np.float32)
    y_val = np.random.randn(N_val).astype(np.float32) * 2 + 7
    
    trunk = MultiTaskTrunk(input_dim=d, trunk_dim=64, hidden_dim=128)
    oracle = DKLEnsembleOracle(input_dim=64, n_members=2, feature_dim=16,
                                n_inducing=32, device="cpu")
    hybrid = MultiTaskHybridOracle(trunk, oracle)
    
    # Stage 1
    history = hybrid.train_trunk(X_train, y_train, X_val, y_val,
                                  n_epochs=10, device="cpu")
    assert len(history['train_loss']) > 0
    
    # Stage 2
    oracle_hist = hybrid.train_oracle(X_train, y_train, X_val, y_val)
    assert 'member_histories' in oracle_hist
    
    # Predict
    result = hybrid.predict(X_val)
    assert isinstance(result, OracleResult)
    assert result.mu.shape == (N_val,)
    assert result.sigma2.shape == (N_val,)
    assert np.all(result.sigma2 > 0)
    assert 'cls_prob' in result.aux


# ── T2.2: h_trunk satisfies OracleHead contract ──────────────────

def test_trunk_to_oracle_interface():
    trunk = MultiTaskTrunk(input_dim=128, trunk_dim=64)
    X = np.random.randn(100, 128).astype(np.float32)
    h = trunk.extract_trunk_features(X)
    
    # h must be (N, trunk_dim) float32 — same as what OracleHead.fit() expects
    assert h.dtype == np.float32
    assert h.shape == (100, 64)
    assert np.isfinite(h).all()


# ── T2.7: Save/load roundtrip ────────────────────────────────────

def test_save_load_roundtrip():
    torch.manual_seed(42)
    trunk = MultiTaskTrunk(input_dim=32, trunk_dim=16, hidden_dim=32)
    oracle = DKLEnsembleOracle(input_dim=16, n_members=2, feature_dim=8,
                                n_inducing=16, device="cpu")
    hybrid = MultiTaskHybridOracle(trunk, oracle)
    
    X = np.random.randn(50, 32).astype(np.float32)
    y = np.random.randn(50).astype(np.float32) * 2 + 7
    
    hybrid.train_trunk(X, y, X, y, n_epochs=5, device="cpu")
    hybrid.train_oracle(X, y, X, y)
    
    result_before = hybrid.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hybrid.save(tmpdir)
        
        trunk2 = MultiTaskTrunk(input_dim=32, trunk_dim=16, hidden_dim=32)
        oracle2 = DKLEnsembleOracle(input_dim=16, n_members=2, feature_dim=8,
                                     n_inducing=16, device="cpu")
        hybrid2 = MultiTaskHybridOracle(trunk2, oracle2)
        hybrid2.load(tmpdir)
        
        result_after = hybrid2.predict(X)
    
    np.testing.assert_allclose(result_before.mu, result_after.mu, atol=1e-5)
    np.testing.assert_allclose(result_before.sigma2, result_after.sigma2, atol=1e-5)


# ── T2.8: Within-group NDCG ──────────────────────────────────────

def test_within_group_ndcg():
    from bayesdiff.multi_task import within_group_ndcg  # or wherever it's placed
    
    y_true = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    y_pred = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.float32)  # perfect
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    result = within_group_ndcg(y_true, y_pred, groups, k=5)
    assert result['ndcg_mean'] == pytest.approx(1.0, abs=1e-6)
    assert result['n_groups_evaluated'] == 2
    
    # Scrambled predictions should have lower NDCG
    y_pred_bad = np.array([0, 1, 2, 3, 4, 9, 8, 7, 6, 5], dtype=np.float32)
    result_bad = within_group_ndcg(y_true, y_pred_bad, groups, k=5)
    assert result_bad['ndcg_mean'] < result['ndcg_mean']
```

### 4.3 Ablation Experiments

#### Tier 1: Does trunk shaping help the SP4 winner?

This is the **make-or-break** tier. SP05 is terminated if A5.2 cannot pass the formal decision gate below.

| Ablation ID | Configuration | Purpose | Expected Outcome |
|-------------|--------------|---------|-----------------|
| A5.0 | Frozen embedding → DKL Ensemble directly (**no trunk**) | True SP4 baseline — no extra parameters | $\rho \approx 0.78$, $\rho_{|err|,\sigma} \approx 0.09$ |
| A5.1 | Regression-only trunk + DKL Ensemble | Does an extra trunk layer help even without multi-task? | Near A5.0; measures trunk overhead |
| A5.2 | Reg + Cls trunk + DKL Ensemble | **Primary hypothesis test** | $\rho \geq 0.76$, $\rho_{|err|,\sigma} \geq 0.12$ |

**Protocol for A5.0 (true SP4 baseline — no trunk)**:
- Skip trunk training entirely; fit DKL Ensemble directly on the frozen embeddings (`frozen_embeddings.npz`)
- This is the exact SP4 configuration and serves as the ground-truth baseline
- If the SP4 checkpoint already exists, reuse its results (job 5863003)

**Protocol for A5.1 (trunk overhead test)**:
- Train a SharedTrunk with only regression head (λ_cls = 0.0)
- Extract h_trunk, fit DKL Ensemble, evaluate on CASF-2016
- Tells us whether the additional trunk parameters help or hurt vs. the no-trunk baseline (A5.0)

**Protocol for A5.2 (hypothesis)**:
- Same trunk architecture, but λ_cls = 0.5
- Compare all metrics head-to-head against both A5.0 and A5.1

**Formal Tier 1 → Tier 2 Decision Gate** (all conditions evaluated on 3-seed mean):

| Condition | Comparison | Threshold | Rationale |
|-----------|------------|-----------|----------|
| A5.2 vs. **A5.0**: UQ improvement | $\rho_{|err|,\sigma}$ mean increase ≥ 0.02 **or** NLL mean decrease ≥ 0.05 | Hard gate | "Is the shaped trunk better than no trunk at all?" |
| A5.2 vs. **A5.1**: Classification value | At least one UQ metric ($\rho_{|err|,\sigma}$, NLL, ECE) also improves over A5.1 | Soft gate | "Does the cls head add value beyond the trunk alone?" |
| A5.2: Point prediction floor | Test $\rho$ ≥ 0.76 (mean across seeds) | Hard gate | "No unacceptable regression degradation" |

If condition 1 fails → terminate SP05 entirely (trunk itself has no value).  
If condition 1 passes but condition 2 fails → proceed to Tier 2 but reconsider cls weight (try lower $\lambda_2$).  
If all three pass → proceed to Tier 2 with full confidence.

**Seeds**: Run each configuration with 3 seeds (42, 123, 777) and report mean ± std, as established in SP4 Tier 2.

#### Tier 2: Add ranking (only if Tier 1 passes decision gate)

| Ablation ID | Configuration | Purpose | Expected Outcome |
|-------------|--------------|---------|-----------------|
| A5.3 | Reg + Rank trunk (grouped BPR) + DKL Ensemble | Ranking alone vs. classification | Ranking alone may help ρ but not ECE |
| A5.4 | Reg + Cls + Rank trunk (grouped BPR) + DKL Ensemble | Full three-task | Best overall if no negative transfer |
| A5.5 | BPR vs. margin hinge (δ=0.5) for ranking loss | Ranking loss variant | BPR expected smoother/better |
| A5.6 | Grouped ranking vs. naive in-batch ranking | Pair construction strategy | Grouped expected strictly better |

**A5.6 is a critical sanity check**: Naive in-batch ranking should produce worse or equal within-group NDCG, validating the grouped-pair design.

#### Tier 3: Task weight optimization (only if multi-task shows benefit)

| Ablation ID | Configuration | Purpose | Expected Outcome |
|-------------|--------------|---------|-----------------|
| A5.7 | Grid search: λ₂ ∈ {0.1, 0.3, 0.5, 1.0}, λ₃ ∈ {0.1, 0.3, 0.5} | Weight sensitivity | Identify optimal weights |
| A5.8 | Learned λ (uncertainty-based weighting, Kendall 2018) | Automatic balancing | May match or beat grid search |
| A5.9 | Gradient surgery (PCGrad, Yu et al. 2020) | Conflict resolution | Only if gradient cosine < -0.1 |

**A5.9 gate**: Before running PCGrad, log per-step gradient cosine similarity between task losses during A5.4 training. Only implement PCGrad if the mean cosine similarity is consistently negative (< -0.1 over > 50% of training steps).

#### Tier 4: Threshold and fusion

| Ablation ID | Configuration | Purpose | Expected Outcome |
|-------------|--------------|---------|-----------------|
| A5.10 | Threshold: $\tau$ = 6.0, 6.5, 7.0, 7.5, 8.0 | Activity cutoff sensitivity | τ=7.0 likely optimal; τ=8.0 creates severe imbalance |
| A5.11 | Combiner: isotonic vs. logistic vs. no-combiner | P_success fusion method | Logistic expected best (more features) |

#### Result Reporting Template

Each ablation produces a JSON file with:
```json
{
    "ablation_id": "A5.2",
    "config": {"phase": "v1", "lambda_cls": 0.5, "threshold": 7.0},
    "seed": 42,
    "trunk_training": {"best_epoch": 85, "train_loss_final": 0.42, "val_loss_final": 0.51},
    "point_prediction": {"spearman_rho": 0.773, "r2": 0.594, "rmse": 1.35},
    "uncertainty": {"nll": 1.72, "ece": 0.031, "rho_err_sigma": 0.128},
    "classification": {"auroc": 0.95, "auprc": 0.88, "ef_1pct": 4.2},
    "ranking": {"within_group_ndcg_5": 0.82, "within_group_ndcg_10": 0.78}
}
```

---

## 5. Evaluation & Success Criteria

### 5.1 Baseline

All metrics are compared against the **SP4 DKL Ensemble with unshaped trunk** (the current best configuration):

| Metric | SP4 Baseline | Source |
|--------|-------------|--------|
| Spearman $\rho$ | 0.781 | SP4 Tier 1 (job 5863003) |
| $R^2$ | 0.607 | SP4 Tier 1 |
| RMSE | ~1.35 | SP4 Tier 1 |
| $\rho_{|err|,\sigma}$ | 0.091 ± 0.025 | SP4 Tier 2 multi-seed (seeds 42/123/777) |
| NLL | 1.756 | SP4 Tier 1 |
| ECE | ~0.03 | SP4 Tier 1 |
| Epistemic $\rho_{|err|,\sigma}$ | 0.180 | SP4 diagnostics (d_5 epistemic only) |

**Important context from SP4**: Ensemble M_eff = 1.05 (member correlation ρ = 0.941), meaning ensemble diversity is very low. The multi-task trunk might help by creating representations where ensemble members disagree more.

### 5.2 Success Criteria

| Category | Metric | Success Condition | Measurement Method |
|----------|--------|-------------------|--------------------|
| **Point prediction** | Test $\rho$ | Drop ≤ 0.02 vs. SP4 baseline (i.e., $\rho \geq 0.76$) | `scipy.stats.spearmanr(y_test, mu_pred)` |
| **Point prediction** | Test $R^2$ | Drop ≤ 0.03 vs. SP4 baseline | `1 - MSE / Var(y_test)` |
| **Uncertainty (primary)** | $\rho_{|err|,\sigma}$ | Improve by ≥ 0.03 (i.e., $\geq 0.12$) | `spearmanr(|y - mu|, sqrt(sigma2))` |
| **Uncertainty** | NLL | Improve or match ($\leq 1.76$) | `gaussian_nll(y, mu, sigma2)` from evaluate.py |
| **Uncertainty** | ECE | Improve or match ($\leq 0.035$) | 10-bin expected calibration error |
| **Ensemble diversity** | M_eff | Increase from 1.05 | `N / (1 + (N-1) * mean_corr)` on member predictions |
| **Classification** | AUPRC | Report; primary over AUROC | `sklearn.metrics.average_precision_score` |
| **Classification** | EF@1% | Report improvement factor over random | Top-1% enrichment |
| **Ranking** | Within-group NDCG@5 | Must be computed within clusters | `within_group_ndcg(...)` |
| **Ranking** | Within-group NDCG@10 | Same, broader cutoff | |

### 5.3 Failure Criterion

Sub-Plan 05 v1 is declared **failed** and does not enter the main pipeline if:

1. Multi-task trunk **does not improve** any of: NLL, ECE, $\rho_{|err|,\sigma}$ over SP4 baseline, **AND**
2. Point prediction $\rho$ degrades by > 0.02

In this case, the unshaped trunk + DKL Ensemble from SP4 remains the production configuration.

**Partial success**: If classification or ranking metrics improve but UQ metrics do not, the trunk shaping is informative but does not enter the main oracle pipeline. The classification head may still be used independently in the P_success combiner.

### 5.4 Evaluation Script: `scripts/pipeline/s21_evaluate_multitask.py`

```python
"""
Evaluate multi-task trunk + oracle head against SP4 baseline.

Generates:
  - Comparison table (JSON + stdout)
  - Per-task loss curves (PDF)
  - t-SNE: unshaped vs. shaped trunk (PDF)
  - Reliability diagrams (PDF)
  - Within-group NDCG curves (PDF)

Usage:
    python scripts/pipeline/s21_evaluate_multitask.py \
        --shaped_model results/stage2/multitask_trunk/v1 \
        --baseline_model results/stage2/oracle_heads/dkl_ensemble \
        --embeddings results/stage2/oracle_heads/frozen_embeddings.npz \
        --clusters data/pdbbind_v2020/cluster_assignments.csv \
        --output results/stage2/multitask_trunk/v1/figures
"""
```

**Diagnostic outputs**:

| Output file | Content | Decision it informs |
|------------|---------|---------------------|
| `comparison_table.json` | Side-by-side metrics: shaped vs. unshaped | Go/no-go for mainline |
| `training_curves.pdf` | L_reg, L_cls, L_rank vs. epoch (train + val) | Convergence quality |
| `tsne_trunk.pdf` | h_trunk colored by pKd: before vs. after shaping | Representation quality |
| `gradient_cosine.pdf` | cos(∇L_reg, ∇L_cls) per training step | Task conflict detection |
| `reliability_diagram.pdf` | P_success calibration (oracle vs. combiner) | Combiner benefit |
| `ndcg_k_curve.pdf` | Within-group NDCG@k for k = 1, 3, 5, 10, 20 | Ranking quality |
| `ensemble_diversity.json` | M_eff, member correlation for shaped vs. unshaped trunk | Diversity gain |

### 5.5 Diagnostic Metrics (Computed During Training)

- **Per-task loss curves**: Monitor convergence of each head independently; watch for one head dominating
- **Task gradient conflict**: Log `cos_sim(∂L_reg/∂θ_trunk, ∂L_cls/∂θ_trunk)` every 10 steps; if consistently < -0.1, flag for PCGrad (Tier 3)
- **Trunk representation quality**: t-SNE/UMAP of $h_{\text{trunk}}$ colored by pKd, before vs. after multi-task shaping
- **Classification calibration**: Reliability diagram for $p_{\text{cls}}$ vs. actual active rate
- **Learned $\lambda$ values** (if Tier 3 reached): Which task gets most weight? Log $\sigma_1, \sigma_2, \sigma_3$ vs. epoch

### 5.6 Risk: Negative Transfer

Multi-task learning can *hurt* performance if tasks conflict (negative transfer). Mitigation:

1. **Monitor per-task metrics**: If any metric degrades > 5% vs. single-task, investigate
2. **Gradient surgery** (Yu et al., 2020): Only apply if gradient cosine similarity is consistently negative (< -0.1 over 50% of steps)
3. **Early detection**: If val loss stops decreasing before epoch 30 but individual L_reg continues to decrease, the classification head may be interfering — try reducing λ_cls
4. **Fallback**: If multi-task trunk hurts the SP4 oracle head, revert to unshaped trunk

---

## 6. Paper Integration

### 6.1 Methods Section (Draft)

> **§3.Z Multi-Task Trunk Shaping**
> 
> Sub-Plan 04 established the DKL Ensemble as the best oracle head, achieving strong point prediction ($\rho = 0.78$) but limited error–uncertainty correlation ($\rho_{|err|,\sigma} \approx 0.09$). To improve the quality of the learned representation feeding the oracle, we train the trunk network with a joint multi-task objective that aligns it with the three decision modes of BayesDiff:
> 
> $$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{cls}} + \lambda_3 \mathcal{L}_{\text{rank}}$$
> 
> where $\mathcal{L}_{\text{reg}}$ is MSE for pKd regression, $\mathcal{L}_{\text{cls}}$ is binary cross-entropy for activity classification (with threshold $\tau = 7.0$, corresponding to 100 nM affinity), and $\mathcal{L}_{\text{rank}}$ is a BPR pairwise ranking loss computed only within biologically coherent groups (same protein cluster).
> 
> A shared trunk maps the molecular embedding $z$ to a task-shaped representation $h$, which is then consumed by the DKL Ensemble oracle head for uncertainty-aware prediction. The auxiliary classification and ranking heads are active only during trunk training and discarded at inference, except that their outputs may be combined with the oracle's calibrated probability via a held-out calibration combiner.
> 
> Task weights $\lambda_1, \lambda_2, \lambda_3$ are determined via [grid search / uncertainty-based weighting].

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. M.1 | Architecture: trunk → SP4 oracle + auxiliary heads | Explain trunk-shaping design |
| Fig. M.2 | Per-task training curves (reg, cls, [rank]) | Show convergence behavior |
| Fig. M.3 | t-SNE of h_trunk: unshaped vs. shaped, colored by pKd | Representation quality |
| Fig. M.4 | Within-group NDCG@k curve (if v2 tested) | Ranking improvement |
| Fig. M.5 | Calibration combiner: reliability diagram | P_success quality |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. M.1 | Tier 1–2 task combination ablation (A5.1–A5.6) |
| Tab. M.2 | Weight sensitivity and learned weights (A5.7–A5.9) |
| Tab. M.3 | Activity threshold sensitivity (A5.10) |
| Tab. M.4 | Calibration combiner comparison (A5.11) |

---

## 7. Implementation Checklist

### Phase A: Core Module (prerequisite for everything)

- [ ] **A.1** Create `bayesdiff/multi_task.py` with `SharedTrunk`, `RegressionHead`, `ClassificationHead`, `RankingHead`
- [ ] **A.2** Implement `MultiTaskTrunk` with phased loss (v1: reg+cls, v2: +rank)
- [ ] **A.3** Implement `_make_grouped_pairs()` with `max_pairs_per_group` cap
- [ ] **A.4** Implement BPR loss in `compute_loss()` 
- [ ] **A.5** Implement `extract_trunk_features()` (numpy ↔ torch bridge)
- [ ] **A.6** Implement `MultiTaskHybridOracle` with `train_trunk()`, `train_oracle()`, `predict()`, `save()`, `load()`
- [ ] **A.7** Add `within_group_ndcg()` metric function

### Phase B: Tests (before any experiments)

- [ ] **B.1** Write `tests/stage2/test_multi_task.py` — unit tests T1.1–T1.24
- [ ] **B.2** Write `tests/stage2/test_multi_task_integration.py` — integration tests T2.1–T2.8
- [ ] **B.3** Run all tests locally: `pytest tests/stage2/test_multi_task*.py -v`

### Phase C: Pipeline Scripts

- [ ] **C.1** Write `scripts/pipeline/s20_train_multitask_trunk.py` (main training pipeline)
- [ ] **C.2** Write `scripts/pipeline/s21_evaluate_multitask.py` (diagnostics + comparison)
- [ ] **C.3** Implement `CalibrationCombiner` class (isotonic + logistic variants)
- [ ] **C.4** Write `slurm/s20_multitask_tier1.sh` (Tier 1 SLURM job)
- [ ] **C.5** **[BLOCKER]** Augment frozen embeddings NPZ with `codes_train/val/test` + `groups_train/val` arrays (or write and validate an alignment script). **Must be done before any experiments.**

### Phase D: Tier 1 Experiments (go/no-go gate)

- [ ] **D.0** Run A5.0 (true SP4 baseline: frozen embedding → DKL Ensemble, no trunk) — or reuse SP4 results
- [ ] **D.1** Run A5.1 (regression-only trunk + DKL Ensemble) × 3 seeds
- [ ] **D.2** Run A5.2 (reg + cls trunk + DKL Ensemble) × 3 seeds
- [ ] **D.3** Compare results; write Tier 1 results summary
- [ ] **D.4** **Decision gate**: if no UQ improvement, stop here

### Phase E: Tier 2 Experiments (conditional on D.4 passing)

- [ ] **E.1** Write `slurm/s20_multitask_tier2.sh`
- [ ] **E.2** Run A5.3–A5.4 (ranking ablations) × 3 seeds
- [ ] **E.3** Run A5.5 (BPR vs. hinge)
- [ ] **E.4** Run A5.6 (grouped vs. naive ranking)
- [ ] **E.5** Write Tier 2 results summary

### Phase F: Tier 3–4 and Analysis (conditional)

- [ ] **F.1** Analyze gradient conflict from D/E training logs
- [ ] **F.2** Conditional: Implement PCGrad and run A5.9
- [ ] **F.3** Run A5.7–A5.8 (weight sensitivity + learned weights)
- [ ] **F.4** Run A5.10 (threshold sensitivity)
- [ ] **F.5** Run A5.11 (calibration combiner comparison)
- [ ] **F.6** Generate all diagnostic figures (s21)

### Phase G: Paper Integration

- [ ] **G.1** Generate paper figures (Fig. M.1–M.5)
- [ ] **G.2** Generate paper tables (Tab. M.1–M.4)
- [ ] **G.3** Finalize methods section text
- [ ] **G.4** Update `doc/progress_log.md` with SP05 results

### Dependencies

```
Phase A ──► Phase B ──► Phase C ──► Phase D ──►─┬──► Phase E ──► Phase F
                                                 │
                                                 └──► Phase G (can start after D)
```

### Key Files Summary

| File | Type | Status |
|------|------|--------|
| `bayesdiff/multi_task.py` | Module | To create |
| `scripts/pipeline/s20_train_multitask_trunk.py` | Pipeline | To create |
| `scripts/pipeline/s21_evaluate_multitask.py` | Diagnostics | To create |
| `slurm/s20_multitask_tier1.sh` | SLURM | To create |
| `slurm/s20_multitask_tier2.sh` | SLURM | To create |
| `tests/stage2/test_multi_task.py` | Unit tests | To create |
| `tests/stage2/test_multi_task_integration.py` | Integration tests | To create |
| `results/stage2/multitask_trunk/` | Output dir | Created at runtime |
| `data/pdbbind_v2020/cluster_assignments.csv` | Data | Exists (from SP0) |
| `results/stage2/oracle_heads/frozen_embeddings.npz` | Data | Exists (from SP4) |
| `bayesdiff/oracle_interface.py` | Dependency | Exists (from SP4) |
| `bayesdiff/hybrid_oracle.py` | Dependency | Exists (from SP4) |
| `bayesdiff/evaluate.py` | Dependency | Exists |
| `bayesdiff/calibration.py` | Dependency | Exists |
