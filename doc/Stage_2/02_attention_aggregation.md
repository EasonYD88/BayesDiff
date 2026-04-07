# Sub-Plan 2: Attention-Based Aggregation

> **角色**: Phase A **Step 2** — 串行链的中间环节，负责从融合后的原子表示产出 $z_{\text{atom}}$  
> **Priority**: P0 — Critical  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); **Sub-Plan 3** (Token-Level Layer Fusion → $\tilde{h}_i$)  
> **下游消费者**: Sub-Plan 1 (Multi-Granularity 主干框架，接收 $z_{\text{atom}}$ 作为三路之一)  
> **Training Data**: PDBbind v2020 refined set (~5,316 complexes), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 1–2 weeks implementation + 1 week testing  
> **Paper Section**: §3.X.2 Attention-Based Pooling

---

## 0. 在串行链中的位置

```
  TargetDiff Frozen Encoder
         │
         ▼
  ┌──────────────────────────┐
  │ Step 1 (SP3):            │
  │ Token-Level Layer Fusion │
  │ → {h̃_i}                 │
  └────────────┬─────────────┘
               │
               ▼
  ╔════════════════════════════╗
  ║ 【本 Sub-Plan: Attention  ║
  ║   Pooling on {h̃_i}】     ║
  ║                            ║
  ║ z_atom = AttnPool({h̃_i})  ║
  ╚════════════╤═══════════════╝
               │  z_atom
               ▼
  ┌──────────────────────────┐
  │ Step 3 (SP1):            │
  │ Multi-Granularity Fusion │
  │ z_atom + z_inter + z_glob│
  │ → z_new                  │
  └──────────────────────────┘
```

**核心变化**：
- 本 Sub-Plan **不再**是独立的 pooling 替代方案（独立对比 mean pool vs attention pool）
- 而是作为串行链的 Step 2，在 Sub-Plan 3 输出的 $\tilde{h}_i$ 上进行 attention pooling
- 其输出 $z_{\text{atom}}$ 是 Sub-Plan 1 三路融合的其中一路
- 输入从原始 $h_i^{(L)}$（最后一层）变为 $\tilde{h}_i$（layer-fused 后的原子表示）

---

## 1. Motivation

Mean pooling assigns equal weight to every atom in the ligand:

$$
z = \frac{1}{N} \sum_{i=1}^{N} h_i
$$

This is biologically unrealistic. In protein-ligand binding:
- A few pharmacophore atoms (H-bond donors/acceptors, aromatic centers) dominate binding energy
- Solvent-exposed atoms contribute minimally
- The pocket context determines which atoms are important

**Goal**: Replace mean pooling with learnable, pocket-conditioned attention pooling that produces a more informative atom-level summary embedding $z_{\text{atom}}$.

**在串行链中的具体目标**：
- 输入：Sub-Plan 3 输出的 layer-fused 原子表示 $\{\tilde{h}_i\}_{i=1}^{N}$
- 输出：$z_{\text{atom}} = \text{AttentionPool}(\{\tilde{h}_i\})$
- 该 $z_{\text{atom}}$ 作为 Sub-Plan 1 三路融合的一路参与最终表示构建
- 不再单独作为系统的完整 pooling 方案（那是旧架构的定位）

---

## 2. Architecture Design

### 2.1 Three Attention Variants (Progressive Complexity)

#### Variant A: Self-Attention Pooling (Ligand-Only)

The ligand decides which of its own atoms matter most:

$$
s_i = w^\top \tanh(W h_i + b)
$$

$$
\alpha_i = \frac{\exp(s_i)}{\sum_{k=1}^{N} \exp(s_k)}
$$

$$
z_{\text{attn}} = \sum_{i=1}^{N} \alpha_i h_i
$$

**Parameters**: $w \in \mathbb{R}^{d_h}$, $W \in \mathbb{R}^{d_h \times d}$, $b \in \mathbb{R}^{d_h}$ — very lightweight.

#### Variant B: Pocket-Conditioned Cross-Attention

The pocket context modulates which ligand atoms are important:

$$
q = W_q \bar{h}^{(P)}, \quad k_i = W_k h_i^{(L)}, \quad v_i = W_v h_i^{(L)}
$$

$$
\alpha_i = \text{softmax}\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)
$$

$$
z_{\text{cross}} = \sum_{i=1}^{N} \alpha_i v_i
$$

where $\bar{h}^{(P)}$ is the mean pocket embedding (or a learned pocket summary token).

#### Variant C: Multi-Head Attention Pooling

Multiple attention heads capture different aspects of binding:

$$
z_{\text{multi}} = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
$$

$$
\text{head}_h = \sum_i \alpha_i^{(h)} W_V^{(h)} h_i
$$

Using $H = 4$ heads (pharmacophoric, geometric, electronic, steric).

### 2.2 Regularization

To prevent attention collapse (all weight on one atom):

1. **Entropy regularization**:
$$
\mathcal{L}_{\text{ent}} = -\lambda_{\text{ent}} \sum_i \alpha_i \log \alpha_i
$$
Encourages non-degenerate distributions (not too peaked).

2. **Sparsity regularization** (optional):
$$
\mathcal{L}_{\text{sparse}} = \lambda_{\text{sp}} \|\alpha\|_1
$$
Encourages focusing on a few important atoms.

3. These two can be balanced: entropy prevents collapse, sparsity prevents uniformity.

---

## 3. Implementation Plan

### 3.1 New Module: `bayesdiff/attention_pool.py`

```python
class SelfAttentionPooling(nn.Module):
    """Variant A: Ligand self-attention pooling."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h: Tensor, mask: Optional[Tensor] = None):
        """
        Args:
            h: (B, N, d) atom embeddings, B=batch, N=max atoms, d=embed dim
            mask: (B, N) boolean mask for valid atoms (padding=False)
        
        Returns:
            z: (B, d) attention-pooled embedding
            alpha: (B, N) attention weights (for visualization)
        """
        ...


class CrossAttentionPooling(nn.Module):
    """Variant B: Pocket-conditioned cross-attention pooling."""
    
    def __init__(self, ligand_dim: int, pocket_dim: int, hidden_dim: int = 128):
        ...
    
    def forward(self, h_ligand, h_pocket, ligand_mask=None):
        """
        Args:
            h_ligand: (B, N_L, d_L)
            h_pocket: (B, N_P, d_P) or (B, d_P) if pre-pooled
        
        Returns:
            z: (B, d_L) cross-attention pooled embedding
            alpha: (B, N_L) attention weights
        """
        ...


class MultiHeadAttentionPooling(nn.Module):
    """Variant C: Multi-head attention pooling."""
    
    def __init__(self, input_dim: int, n_heads: int = 4, hidden_dim: int = 128):
        ...
    
    def forward(self, h, mask=None, context=None):
        """
        Args:
            h: (B, N, d) atom embeddings
            mask: (B, N) boolean mask
            context: (B, d_ctx) optional pocket context
        
        Returns:
            z: (B, d_out) multi-head pooled embedding
            alpha: (B, H, N) per-head attention weights
        """
        ...


class AttentionPoolingWithRegularization(nn.Module):
    """Wrapper that adds entropy and sparsity regularization losses."""
    
    def __init__(self, pooling_module, entropy_weight=0.01, sparsity_weight=0.0):
        ...
    
    def forward(self, *args, **kwargs):
        """Returns (z, alpha, reg_loss)."""
        ...
```

### 3.2 Modifications to `bayesdiff/sampler.py`

Ensure `sample_and_embed()` returns padded atom embeddings suitable for batched attention:

```python
def _pad_atom_embeddings(self, atom_embs_list, max_atoms=None):
    """
    Pad variable-length atom embeddings to uniform length.
    
    Args:
        atom_embs_list: list of M tensors, each (N_i, d)
    
    Returns:
        padded: (M, N_max, d) zero-padded tensor
        mask: (M, N_max) boolean mask (True = real atom)
    """
    ...
```

### 3.3 Integration with Multi-Granularity Framework (Sub-Plan 1)

In the serial chain, attention-pooled embedding $z_{\text{atom}}$ is one of three inputs to the final fusion:

```python
# In the full chain (SP3 → SP2 → SP1):
# Step 1 (SP3): token-level layer fusion
h_tilde = token_level_fusion(layer_atom_embs)  # (N, d) — fused per-atom repr

# Step 2 (SP2): attention pooling on fused atoms → z_atom
attn_pool = SelfAttentionPooling(input_dim=d)  # or CrossAttention
z_atom, alpha = attn_pool(h_tilde.unsqueeze(0))  # (1, d_atom)

# Step 3 (SP1): interaction graph + global pool + fusion
z_interaction = interaction_gnn(build_graph(h_tilde, pocket_feats))
z_global = h_tilde.mean(dim=0)  # simple mean pool on fused atoms
z_new = fusion_mlp(torch.cat([z_atom, z_interaction, z_global], dim=-1))
```

**Critical**: Attention weights are computed per-molecule, so generation uncertainty estimation over M samples naturally works — each sample $m_i$ gets its own attention-pooled $z_i$, then statistics are computed over $\{z_1, \dots, z_M\}$.

### 3.4 Training Strategy

The attention module parameters need to be trained **jointly** with SP3 (token-level fusion) and SP1 (interaction GNN + fusion MLP) parameters:

**Recommended — End-to-end joint training**:
- Freeze TargetDiff encoder (too expensive to retrain)
- Train SP3 token-level fusion + SP2 attention pooling + SP1 interaction GNN + fusion MLP + GP/DKL jointly
- Loss = GP marginal likelihood + attention entropy regularization
- 所有可训练参数统一优化，避免 stage-wise 训练的 distribution shift

**Fallback — Stage-wise training** (if joint training unstable):
- Phase 1: 固定 SP3 (e.g., uniform weights) → 训练 SP2 attention + GP
- Phase 2: 解冻 SP3 → fine-tune SP3 + SP2 jointly
- Phase 3: 加入 SP1 interaction GNN

---

## 4. Test Plan

### 4.1 Unit Tests: `tests/stage2/test_attention_pool.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_self_attn_output_shape` | Output shape = (B, d) for various B, N, d |
| T1.2 | `test_self_attn_weights_sum_to_one` | $\sum_i \alpha_i = 1$ for each sample |
| T1.3 | `test_self_attn_mask_handling` | Padded positions get zero attention weight |
| T1.4 | `test_cross_attn_output_shape` | Correct shape with pocket context |
| T1.5 | `test_cross_attn_pocket_influence` | Different pocket contexts → different outputs |
| T1.6 | `test_multi_head_output_shape` | Output dim = n_heads × head_dim or projected |
| T1.7 | `test_multi_head_different_heads` | Each head produces different attention patterns |
| T1.8 | `test_entropy_regularization` | Uniform attention → max entropy; peaked → low entropy |
| T1.9 | `test_gradient_flow` | All parameters receive gradients |
| T1.10 | `test_determinism` | Same input + seed → same output |
| T1.11 | `test_numerical_stability` | No NaN with very large/small embeddings |
| T1.12 | `test_single_atom_molecule` | Works when N=1 (trivial case) |

```python
def test_self_attn_mask_handling():
    """Masked (padded) atoms must receive zero attention."""
    pool = SelfAttentionPooling(input_dim=32)
    h = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False  # First sample: only 5 real atoms
    mask[1, 8:] = False  # Second sample: 8 real atoms
    
    z, alpha = pool(h, mask=mask)
    
    assert alpha[0, 5:].abs().max() < 1e-6  # Padded atoms: zero weight
    assert alpha[1, 8:].abs().max() < 1e-6
    assert torch.allclose(alpha[0, :5].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(alpha[1, :8].sum(), torch.tensor(1.0), atol=1e-5)
```

### 4.2 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_attn_pool_with_gp` | Attention-pooled embeddings → GP training converges |
| T2.2 | `test_gen_uncertainty_with_attn` | M attention-pooled embeddings → valid covariance |
| T2.3 | `test_delta_method_through_attn` | Jacobian computes correctly through attention layer |
| T2.4 | `test_full_pipeline_attn_pooling` | End-to-end on synthetic data with attention replacing mean |

### 4.3 Ablation Experiments

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A2.1 | Mean pooling (baseline) | Reference |
| A2.2 | Self-attention pooling | Isolate self-attention benefit |
| A2.3 | Cross-attention pooling (pocket-conditioned) | Test pocket influence |
| A2.4 | Multi-head attention (H=2) | Multi-head effect |
| A2.5 | Multi-head attention (H=4) | More heads |
| A2.6 | Multi-head attention (H=8) | Diminishing returns? |
| A2.7 | Self-attn + entropy reg (λ=0.01) | Regularization effect |
| A2.8 | Self-attn + entropy reg (λ=0.1) | Higher regularization |
| A2.9 | Self-attn + sparsity reg | Sparse attention |
| A2.10 | Cross-attn + multi-head (best combo) | Combined approach |

---

## 5. Evaluation & Success Criteria

### 5.1 Quantitative Metrics

| Metric | Stage 1 Baseline | Success | Stretch |
|--------|------------------|---------|---------|
| $R^2$ | 0.120 | ≥ 0.15 | ≥ 0.22 |
| Spearman $\rho$ | 0.369 | ≥ 0.42 | ≥ 0.50 |
| Attention entropy | N/A | 1.0–3.0 (neither collapsed nor uniform) | — |

### 5.2 Qualitative Analysis

- **Attention weight visualization**: For known binder examples, do high-attention atoms correspond to pharmacophore positions?
- **Pocket-conditioned shift**: For the same ligand in different pockets, does cross-attention shift to different atoms?
- **Per-head specialization**: In multi-head variant, do different heads attend to chemically distinct atom groups?

### 5.3 Failure Criteria

- Attention collapses to uniform → equivalent to mean pooling → check regularization
- Attention collapses to single atom → overfitting → increase entropy regularization
- $R^2$ decreases → attention layer adds noise → simplify architecture

---

## 6. Paper Integration

### 6.1 Methods Section (Draft)

> **§3.X.2 Attention-Based Aggregation**
> 
> We replace mean pooling with a learned attention mechanism that assigns adaptive importance weights to individual atoms. Given atom embeddings $\{h_i\}_{i=1}^{N}$ from the SE(3)-equivariant encoder, we compute:
> 
> $$\alpha_i = \text{softmax}(w^\top \tanh(W h_i + b))$$
> 
> $$z_{\text{attn}} = \sum_{i=1}^{N} \alpha_i h_i$$
> 
> [If pocket-conditioned:] When pocket context $\bar{h}^{(P)}$ is available, we additionally employ cross-attention where the pocket representation serves as the query, and ligand atom representations serve as keys and values.
> 
> To prevent attention collapse, we add an entropy regularization term $\mathcal{L}_{\text{ent}} = -\lambda \sum_i \alpha_i \log \alpha_i$ to the training objective.

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. A.1 | Attention weight heatmap on 3–5 example molecules | Show learned importance |
| Fig. A.2 | 3D molecular visualization with atoms colored by attention weight | Intuitive illustration |
| Fig. A.3 | Attention entropy distribution across molecules | Validate regularization |
| Fig. A.4 | Per-head attention patterns (multi-head variant) | Head specialization |
| Fig. A.5 | Ablation bar chart (A2.1–A2.10) | Justify design choices |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. A.1 | Attention variant comparison (A2.1–A2.6): $R^2$, $\rho$, ECE |
| Tab. A.2 | Regularization sensitivity (A2.7–A2.9) |

---

## 7. Compatibility Notes

### 7.1 With Sub-Plan 3 (Token-Level Layer Fusion) — 上游

本 Sub-Plan 的输入不再是原始编码器最后一层的 $h_i^{(L)}$，而是 SP3 输出的 $\tilde{h}_i$：

$$
z_{\text{atom}} = \text{AttentionPool}(\{\tilde{h}_i\})
$$

其中 $\tilde{h}_i = \text{TokenLevelFusion}(h_i^{(l_1)}, \dots, h_i^{(l_k)})$。

**维度兼容**：如果 SP3 的 token-level fusion 输出维度 $d_{\text{fused}}$ 与原始 $d=128$ 不同，则 attention pooling 的 `input_dim` 需要相应调整。推荐 SP3 输出维度与原始相同（$d_{\text{fused}} = d$），避免额外复杂度。

### 7.2 With Sub-Plan 1 (Multi-Granularity) — 下游

Attention pooling 的输出 $z_{\text{atom}}$ 是 SP1 三路融合的其中一路：

$$
z_{\text{new}} = \text{FusionMLP}([z_{\text{atom}}; z_{\text{interaction}}; z_{\text{global}}])
$$

### 7.3 With Generation Uncertainty

Each of $M$ generated molecules $m_1, \dots, m_M$ gets independently processed through the full chain:

$$
\tilde{h}_i^{(j)} = \text{TokenFusion}(h_i^{(l_1, j)}, \dots) \quad \text{(SP3)}
$$
$$
z_{\text{atom}}^{(j)} = \text{AttentionPool}(\{\tilde{h}_i^{(j)}\}) \quad \text{(SP2)}
$$

Then $\hat{\Sigma}_{\text{gen}}$ is estimated over $\{z_{\text{new}}^{(1)}, \dots, z_{\text{new}}^{(M)}\}$. The attention weights may vary across molecules, which is correct behavior.

### 7.4 With Delta Method

The Jacobian $J_\mu = \partial \mu / \partial z_{\text{new}}$ chains through SP3 → SP2 → SP1. Since all modules are differentiable, `autograd` handles this automatically.

---

## 8. Implementation Checklist

- [ ] Implement `SelfAttentionPooling` in `attention_pool.py`
- [ ] Implement `CrossAttentionPooling` in `attention_pool.py`
- [ ] Implement `MultiHeadAttentionPooling` in `attention_pool.py`
- [ ] Implement `AttentionPoolingWithRegularization` wrapper
- [ ] Add `_pad_atom_embeddings()` to `sampler.py`
- [ ] Write unit tests (T1.1–T1.12)
- [ ] Write integration tests (T2.1–T2.4)
- [ ] Run ablation experiments (A2.1–A2.10)
- [ ] Generate attention visualization figures
- [ ] Draft methods section text
