# Sub-Plan 3: Multi-Layer Fusion

> **Priority**: P1 — High  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); independent of Sub-Plans 1–2, combinable  
> **Training Data**: PDBbind v2020 R1 set, 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 1–2 weeks implementation + 1 week testing  
> **Paper Section**: §3.X.3 Multi-Scale Feature Fusion

---

## 1. Motivation

The current pipeline extracts embeddings only from the **last layer** ($L$-th) of the SE(3)-equivariant encoder:

$$
z = \text{Pool}(H^{(L)})
$$

However, different layers of a deep equivariant network learn representations at different spatial resolutions:

| Layer Depth | Information Captured | Analogy |
|-------------|---------------------|---------|
| Shallow (1–2) | Local bond geometry, atom neighborhoods | "Where are the atoms?" |
| Middle (3–4) | Functional group patterns, local contacts | "What chemical groups are present?" |
| Deep (5–6) | Global molecular shape, long-range interactions | "What does the whole molecule look like?" |

Using only the last layer forces all information through a single bottleneck. Multi-layer fusion preserves multi-scale features that are all relevant for binding affinity prediction.

**Evidence**: In NLP, models like BERT achieve best performance by combining multiple Transformer layers rather than using only the last. Similar benefits have been observed in molecular GNNs.

---

## 2. Architecture Design

### 2.1 Layer Selection Strategy

For a TargetDiff encoder with $L$ layers, select a subset $\mathcal{S} \subseteq \{1, \dots, L\}$:

| Strategy | Layers Selected | Rationale |
|----------|----------------|-----------|
| **Sparse uniform** | $\{2, 4, 6, L\}$ | Even coverage |
| **Progressive** | $\{L/4, L/2, 3L/4, L\}$ | Four spatial scales |
| **All layers** | $\{1, \dots, L\}$ | Maximum information (expensive) |
| **Learned selection** | Top-$k$ by importance | Data-driven |

**Recommended**: Start with all layers for probing; then narrow to informative subset for fusion.

### 2.2 Per-Layer Pooling

For each selected layer $l \in \mathcal{S}$:

$$
z^{(l)} = \text{Pool}(H^{(l)}) \in \mathbb{R}^d
$$

where $\text{Pool}$ can be mean pooling (baseline) or attention pooling (from Sub-Plan 2).

### 2.3 Progressive Fusion Strategy

The fusion methods are ordered by complexity. Each stage has a **go/no-go gate** — proceed to the next stage only if the current stage demonstrates that multi-layer information is beneficial.

#### Stage 1: Single-Layer Probing (Diagnostic — No Fusion)

Train an independent GP on $z^{(l)}$ for **each** layer $l \in \{1, \dots, L\}$ separately:

$$
\text{GP}_l: z^{(l)} \mapsto \hat{y}
$$

**Purpose**: Establish per-layer predictive quality. Answers:
- Is the last layer actually the best single layer?
- Do middle layers carry stronger signal than the last layer?
- Are shallow layers near-useless (can be excluded from fusion)?

**Go/No-Go**: If all layers perform roughly the same as the last layer (within noise), multi-layer fusion is unlikely to help → reconsider plan. If some non-final layers match or exceed the last layer → proceed.

#### Stage 2: Learned Weighted Sum (Minimal Multi-Layer Baseline)

$$
\beta_l = \frac{\exp(w_l)}{\sum_{k \in \mathcal{S}} \exp(w_k)}, \quad z_{\text{fuse}} = \sum_{l \in \mathcal{S}} \beta_l z^{(l)}
$$

where $\{w_l\}$ are learnable scalar parameters. Very lightweight; $|\mathcal{S}|$ extra parameters.

**Purpose**: The cleanest possible multi-layer combination. If this doesn't improve over the best single layer, then multi-layer information may not be complementary.

**Go/No-Go**: If weighted sum $\leq$ best single layer → stop here (multi-layer fusion not justified for this architecture). If weighted sum > best single layer → proceed.

#### Stage 3: Layer-Attention Fusion (Input-Dependent Weighting)

$$
\beta_l = \frac{\exp(u^\top \tanh(W z^{(l)}))}{\sum_{k \in \mathcal{S}} \exp(u^\top \tanh(W z^{(k)}))}
$$

$$
z_{\text{fuse}} = \sum_{l \in \mathcal{S}} \beta_l z^{(l)}
$$

**Purpose**: Test whether different molecules benefit from different layer combinations. If layer attention > weighted sum, then input-dependent weighting provides value.

**Go/No-Go**: If layer attention ≈ weighted sum → the optimal layer mixture is molecule-independent; use weighted sum for simplicity. If layer attention > weighted sum → proceed to test if stronger nonlinear fusion helps further.

#### Stage 4: Concatenation + Bottleneck MLP (Nonlinear Fusion)

$$
z_{\text{concat}} = [z^{(l_1)}; z^{(l_2)}; \dots; z^{(l_m)}] \in \mathbb{R}^{m \cdot d}
$$

$$
z_{\text{fuse}} = \text{MLP}_{\text{bottleneck}}(z_{\text{concat}}) \in \mathbb{R}^{d_{\text{out}}}
$$

**Bottleneck MLP**: $\mathbb{R}^{m \cdot d} \xrightarrow{\text{Linear}} \mathbb{R}^{2d_{\text{out}}} \xrightarrow{\text{ReLU+LN}} \mathbb{R}^{d_{\text{out}}}$

**Purpose**: Only justified when Stages 2–3 confirm multi-layer information is effective. Tests whether cross-layer nonlinear interactions provide additional signal.

**Go/No-Go**: If concat+MLP ≈ layer attention → stick with layer attention (fewer parameters). If concat+MLP > layer attention but unstable → proceed to Stage 5.

#### Stage 5: Concatenation + Layer Dropout (Regularized Fusion)

Same as Stage 4, but during training, randomly drop entire layers with probability $p_{\text{drop}}$:

$$
z_{\text{concat}} = [m_1 z^{(l_1)}; m_2 z^{(l_2)}; \dots; m_m z^{(l_m)}], \quad m_i \sim \text{Bernoulli}(1 - p_{\text{drop}})
$$

**Purpose**: Only needed if concat+MLP shows benefit but exhibits instability or overfitting. Prevents co-adaptation between layers.

### 2.4 Decision Flow Summary

```
Stage 1: Single-layer probing
    │
    ├─ All layers ≈ last layer ──→ STOP (multi-layer not useful)
    │
    └─ Some layers ≥ last layer ──→ Stage 2: Weighted Sum
                                        │
                                        ├─ ≤ best single layer ──→ STOP
                                        │
                                        └─ > best single layer ──→ Stage 3: Layer Attention
                                                                      │
                                                                      ├─ ≈ weighted sum ──→ USE weighted sum
                                                                      │
                                                                      └─ > weighted sum ──→ Stage 4: Concat+MLP
                                                                                              │
                                                                                              ├─ stable ──→ USE concat+MLP
                                                                                              │
                                                                                              └─ unstable ──→ Stage 5: +Dropout
```

---

## 3. Implementation Plan

### 3.0 Stage 0: Hook Infrastructure (Shared by All Stages)

#### 3.0.1 Modifications to `bayesdiff/sampler.py`

The key change is extracting hidden states from **multiple layers** of the TargetDiff encoder, not just the final layer.

```python
class TargetDiffSampler:
    def sample_and_embed(self, pocket_pdb, num_samples=64, 
                          extract_layers=None):
        """
        Args:
            extract_layers: list of int, e.g. [2, 4, 6, 8].
                           If None, only extract last layer (backward compatible).
        
        Returns dict with:
            'z_global': (M, d)  — last layer mean-pooled (as before)
            'z_per_layer': dict mapping layer_idx → (M, d) mean-pooled embeddings
            'h_per_layer': dict mapping layer_idx → list of M × (N_i, d) atom embeddings
        """
        ...
    
    def _register_hooks(self, layer_indices):
        """Register forward hooks on specified encoder layers to capture hidden states."""
        ...
    
    def _remove_hooks(self):
        """Clean up hooks after embedding extraction."""
        ...
```

**Technical detail**: Use PyTorch forward hooks on the encoder's message-passing layers to capture intermediate hidden states without modifying the TargetDiff source code:

```python
def _register_hooks(self, layer_indices):
    self._hooks = []
    self._layer_outputs = {}
    
    for idx in layer_indices:
        layer = self.model.encoder.layers[idx]  # Adjust for actual architecture
        
        def hook_fn(module, input, output, layer_idx=idx):
            self._layer_outputs[layer_idx] = output
        
        self._hooks.append(layer.register_forward_hook(hook_fn))
```

#### 3.0.2 New Pipeline Script: `scripts/pipeline/s08b_extract_multilayer.py`

```python
"""
Extract multi-layer embeddings from TargetDiff encoder for all pockets.

Usage:
    python scripts/pipeline/s08b_extract_multilayer.py \
        --data_dir data/pdbbind_v2020/processed \
        --layers all \
        --output_dir data/multilayer_embeddings/ \
        --device cuda

Output:
    data/multilayer_embeddings/{pdb_code}.npz
        - 'layer_1': (M, d) embeddings from layer 1
        - 'layer_2': (M, d) embeddings from layer 2
        - ...
        - 'layer_L': (M, d) embeddings from layer L
        - 'z_global': (M, d) last layer mean-pooled (for backward compatibility)
"""
```

Extract **all** layers upfront — this avoids re-running the expensive generative model later when testing different layer subsets.

### 3.1 Stage 1: Single-Layer Probing

#### 3.1.1 New Script: `scripts/pipeline/s09a_single_layer_probe.py`

```python
"""
Train an independent GP on each layer's embeddings separately.
Produces a table of per-layer metrics (R², Spearman ρ, NLL).

Usage:
    python scripts/pipeline/s09a_single_layer_probe.py \
        --embedding_dir data/multilayer_embeddings/ \
        --label_file data/pdbbind_v2020/index/INDEX_refined_data.2020 \
        --output results/stage2/layer_probing.csv

Output:
    results/stage2/layer_probing.csv  — columns: layer_idx, R2, spearman, NLL
    results/stage2/layer_probing.png  — bar chart of per-layer performance
"""
```

No new modules needed — reuse existing GP pipeline with different input embeddings.

**Decision point**: Review `layer_probing.csv`. If no non-final layer matches the last layer → stop multi-layer investigation.

### 3.2 Stage 2: Weighted Sum Fusion

#### 3.2.1 New Module: `bayesdiff/layer_fusion.py` (partial)

```python
class WeightedSumFusion(nn.Module):
    """Learned scalar weights per layer (softmax-normalized)."""
    def __init__(self, n_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))
    
    def forward(self, layer_embeddings: List[Tensor]):
        """
        Args:
            layer_embeddings: list of (B, d) tensors, one per layer
        Returns:
            z_fuse: (B, d) fused representation
            layer_weights: (n_layers,) softmax weights for interpretability
        """
        weights = F.softmax(self.logits, dim=0)
        z_fuse = sum(w * z for w, z in zip(weights, layer_embeddings))
        return z_fuse, weights
```

#### 3.2.2 New Script: `scripts/pipeline/s09b_weighted_sum_fusion.py`

Train GP with weighted-sum fused embeddings. Compare vs. best single layer from Stage 1.

**Decision point**: If weighted sum ≤ best single layer → multi-layer fusion not justified. Report negative result and stop.

### 3.3 Stage 3: Layer Attention Fusion

#### 3.3.1 Extend `bayesdiff/layer_fusion.py`

```python
class LayerAttentionFusion(nn.Module):
    """Input-dependent layer attention."""
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.W = nn.Linear(embed_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, layer_embeddings: List[Tensor]):
        """
        Returns:
            z_fuse: (B, d) fused representation
            layer_weights: (B, n_layers) per-sample layer weights
        """
        scores = [self.u(torch.tanh(self.W(z))) for z in layer_embeddings]
        scores = torch.cat(scores, dim=-1)  # (B, n_layers)
        weights = F.softmax(scores, dim=-1)
        z_fuse = sum(w.unsqueeze(-1) * z for w, z in zip(weights.unbind(-1), layer_embeddings))
        return z_fuse, weights
```

**Decision point**: If layer attention ≈ weighted sum → use weighted sum (simpler). If layer attention > weighted sum → proceed.

### 3.4 Stage 4: Concat + MLP Fusion

#### 3.4.1 Extend `bayesdiff/layer_fusion.py`

```python
class ConcatMLPFusion(nn.Module):
    """Concatenation + bottleneck MLP."""
    def __init__(self, embed_dim, n_layers, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * n_layers, 2 * output_dim),
            nn.LayerNorm(2 * output_dim),
            nn.ReLU(),
            nn.Linear(2 * output_dim, output_dim),
        )
    
    def forward(self, layer_embeddings: List[Tensor]):
        z_concat = torch.cat(layer_embeddings, dim=-1)
        return self.mlp(z_concat), None  # No interpretable weights
```

### 3.5 Stage 5: Concat + Layer Dropout

#### 3.5.1 Extend `bayesdiff/layer_fusion.py`

```python
class ConcatDropoutFusion(nn.Module):
    """Concatenation with layer dropout regularization."""
    def __init__(self, embed_dim, n_layers, output_dim, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * n_layers, 2 * output_dim),
            nn.LayerNorm(2 * output_dim),
            nn.ReLU(),
            nn.Linear(2 * output_dim, output_dim),
        )
    
    def forward(self, layer_embeddings: List[Tensor]):
        if self.training:
            masks = [torch.bernoulli(torch.tensor(1.0 - self.drop_prob)) for _ in range(self.n_layers)]
            layer_embeddings = [m * z for m, z in zip(masks, layer_embeddings)]
        z_concat = torch.cat(layer_embeddings, dim=-1)
        return self.mlp(z_concat), None
```

### 3.6 Factory + Downstream Module Changes

```python
class LayerFusion(nn.Module):
    """Factory for all fusion methods."""
    
    def __init__(self, embed_dim: int, n_layers: int, output_dim: int,
                 method: str = 'weighted_sum'):
        """
        Args:
            method: 'weighted_sum' | 'layer_attention' | 'concat_mlp' | 'concat_dropout'
        """
        ...
```

**`bayesdiff/gen_uncertainty.py`**:
- Accept fused embeddings of dimension $d_{\text{out}}$ (may differ from original 128)
- No algorithmic changes needed; Ledoit-Wolf works for any $d$

**`bayesdiff/gp_oracle.py`**:
- Accept input dimension as parameter (already parameterized)
- May need to increase number of inducing points if $d_{\text{out}} > 128$

**`bayesdiff/fusion.py`**:
- Jacobian dimension changes; handled automatically by `autograd`

---

## 4. Test Plan

### 4.1 Unit Tests: `tests/stage2/test_layer_fusion.py`

| Test ID | Test Name | What It Verifies | Stage |
|---------|-----------|-----------------|-------|
| T1.1 | `test_weighted_sum_weights` | Weights sum to 1; positive; gradients flow | 2 |
| T1.2 | `test_learned_weights_initialization` | Initial weights are roughly uniform | 2 |
| T1.3 | `test_layer_attention_input_dependent` | Different inputs → different layer weights | 3 |
| T1.4 | `test_concat_mlp_shape` | Output shape = (B, d_out) for various inputs | 4 |
| T1.5 | `test_concat_dropout_train_vs_eval` | Dropout active in train, inactive in eval | 5 |
| T1.6 | `test_gradient_flow_all_methods` | All fusion method parameters receive gradients | 2–5 |
| T1.7 | `test_single_layer_input` | When n_layers=1, output ≈ identity transform | 2–3 |
| T1.8 | `test_numerical_stability` | No NaN/Inf with extreme values | 2–5 |

```python
def test_layer_attention_input_dependent():
    """Different molecules should produce different layer importance weights."""
    fusion = LayerAttentionFusion(embed_dim=128, hidden_dim=64)
    
    # Two different "molecules" with 4 layer embeddings each
    layers_a = [torch.randn(1, 128) for _ in range(4)]
    layers_b = [torch.randn(1, 128) * 3 for _ in range(4)]  # Different distribution
    
    _, weights_a = fusion(layers_a)
    _, weights_b = fusion(layers_b)
    
    # Weights should differ
    assert not torch.allclose(weights_a, weights_b, atol=0.01)
```

### 4.2 Hook Extraction Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_hook_registration` | Hooks register on correct layers |
| T2.2 | `test_hook_output_shapes` | Each hooked layer produces (N, d) output |
| T2.3 | `test_hook_cleanup` | Hooks are properly removed after extraction |
| T2.4 | `test_hook_no_side_effects` | Model output unchanged when hooks are active |

### 4.3 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T3.1 | `test_multilayer_to_gp` | Fused embeddings → GP training converges |
| T3.2 | `test_multilayer_gen_uncertainty` | Ledoit-Wolf on fused embeddings → valid covariance |
| T3.3 | `test_multilayer_full_pipeline` | End-to-end pipeline with layer fusion |
| T3.4 | `test_backward_compatibility` | Single-layer mode produces identical results to baseline |

### 4.4 Staged Experiments (with Go/No-Go Gates)

#### Stage 1: Single-Layer Probing

| Exp ID | Configuration | Purpose |
|--------|--------------|---------|
| E1.1 | GP on $z^{(l)}$ for each $l \in \{1, \dots, L\}$ | Per-layer predictive quality |
| E1.2 | CKA similarity matrix across all layers | Layer redundancy analysis |

**Gate 1**: Compare per-layer $R^2$ and Spearman $\rho$.
- If best non-final layer $R^2$ $\geq$ $0.9 \times$ last layer $R^2$ → **proceed** (complementary information likely exists)
- If all non-final layers $\ll$ last layer → **stop** (multi-layer fusion unlikely to help)

#### Stage 2: Weighted Sum

| Exp ID | Configuration | Purpose |
|--------|--------------|---------|
| E2.1 | Weighted sum over best-$k$ layers (from E1.1), $k \in \{2, 4, L\}$ | Multi-layer baseline |
| E2.2 | Inspect learned $\beta_l$ weights | Confirm non-trivial layer usage |

**Gate 2**: Compare weighted sum $R^2$ vs. best single layer $R^2$.
- If weighted sum $>$ best single layer (significance test) → **proceed**
- If weighted sum $\leq$ best single layer → **stop** (negative result; report and conclude)

#### Stage 3: Layer Attention

| Exp ID | Configuration | Purpose |
|--------|--------------|---------|
| E3.1 | Layer attention over same layer set as E2.1 | Input-dependent weighting |
| E3.2 | Analyze per-sample layer weight variance | Do weights actually vary across molecules? |

**Gate 3**: Compare layer attention vs. weighted sum.
- If layer attention > weighted sum → **proceed** (input-dependent weighting matters)
- If layer attention ≈ weighted sum → **use weighted sum** (simpler; stop here)

#### Stage 4: Concat + MLP

| Exp ID | Configuration | Purpose |
|--------|--------------|---------|
| E4.1 | Concat+MLP over same layer set | Nonlinear cross-layer interactions |
| E4.2 | Output dim sensitivity: 64, 128, 256 | Capacity analysis |

**Gate 4**: Compare concat+MLP vs. layer attention.
- If concat+MLP > layer attention, stable → **use concat+MLP** (stop)
- If concat+MLP > layer attention, unstable → **proceed** to Stage 5
- If concat+MLP ≈ layer attention → **use layer attention** (stop)

#### Stage 5: Concat + Dropout (conditional)

| Exp ID | Configuration | Purpose |
|--------|--------------|---------|
| E5.1 | Concat+MLP + layer dropout ($p=0.1$) | Regularized fusion |
| E5.2 | Dropout rate sweep: $p \in \{0.05, 0.1, 0.2\}$ | Sensitivity analysis |

#### Cross-Cutting Ablations (run after best method is selected)

| Exp ID | Configuration | Purpose |
|--------|--------------|---------|
| E6.1 | Best fusion + attention pooling (Sub-Plan 2) | Combined benefit |
| E6.2 | Layer subset ablation: remove one layer at a time | Measure each layer's marginal contribution |

---

## 5. Evaluation & Success Criteria

### 5.1 Quantitative Metrics

| Metric | Stage 1 Baseline | Success | Stretch |
|--------|------------------|---------|---------|
| $R^2$ | 0.120 | ≥ 0.16 | ≥ 0.22 |
| Spearman $\rho$ | 0.369 | ≥ 0.43 | ≥ 0.50 |
| NLL | baseline | ≥ 3% reduction | ≥ 10% reduction |

### 5.2 Diagnostic Metrics

- **Per-layer GP performance** (Stage 1): $R^2$, Spearman $\rho$ for each layer independently
- **Learned layer weights** (Stages 2–3): Which layers contribute most?
- **Per-sample weight variance** (Stage 3): Do different molecules use different layer mixtures?
- **Representation similarity** (CKA): How redundant are different layers' representations?

### 5.3 Expected Findings (Hypotheses)

1. **Stage 1**: Middle layers are competitive with or exceed the last layer for affinity prediction; shallow layers carry less signal
2. **Stage 2**: Weighted sum over $\geq 2$ layers outperforms any single layer (complementary information exists)
3. **Stage 3**: Layer attention assigns non-uniform, input-dependent weights (different molecules benefit from different layers)
4. **Stage 4**: Nonlinear fusion provides modest additional gain over attention (diminishing returns)
5. **Stage 1 × Sub-Plan 2**: Multi-layer fusion benefits are amplified when combined with attention pooling

---

## 6. Paper Integration

### 6.1 Methods Section (Draft)

> **§3.X.3 Multi-Scale Feature Fusion**
> 
> Different layers of the SE(3)-equivariant encoder capture information at different spatial scales. We first evaluated whether exploiting multi-scale representations is beneficial at all by probing each encoder layer independently (training a separate GP on each layer's pooled embeddings). This revealed that [intermediate layers carry competitive / complementary signal to the final layer] (Fig. L.1).
> 
> Motivated by this finding, we fuse pooled representations from multiple encoder layers $\mathcal{S} = \{l_1, \dots, l_m\}$ via a learned aggregation function:
> 
> $$z_{\text{fuse}} = \Psi(z^{(l_1)}, \dots, z^{(l_m)})$$
> 
> where $z^{(l)} = \text{Pool}(H^{(l)})$ is the pooled embedding from the $l$-th encoder layer.
> 
> We progressively evaluated fusion strategies of increasing complexity: (1) learned weighted sum, (2) input-dependent layer attention, and (3) concatenation with bottleneck MLP (Table X). [Weighted sum / Layer attention] achieves the best performance–complexity trade-off, with learned weights indicating that [intermediate / deep] layers contribute most to binding affinity prediction.

### 6.2 Figures

| Figure | Content | Purpose | Stage |
|--------|---------|---------|-------|
| Fig. L.1 | Per-layer GP performance (bar chart) | Each layer's individual quality | 1 |
| Fig. L.2 | CKA similarity matrix across layers | Layer redundancy analysis | 1 |
| Fig. L.3 | Learned layer weights (bar/violin plot) | Which layers matter most | 2–3 |
| Fig. L.4 | Progressive method comparison (cascade plot) | Justify chosen method | 2–4 |
| Fig. L.5 | Fused vs. last-layer t-SNE | Visual quality improvement | Best |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. L.1 | Per-layer probing results (E1.1) |
| Tab. L.2 | Progressive fusion comparison: single-layer → weighted sum → attention → concat (E2.1–E4.1) |
| Tab. L.3 | Output dimension sensitivity (E4.2, if reached) |

---

## 7. Technical Considerations

### 7.1 Memory Overhead

Storing embeddings from $m$ layers instead of 1:
- Per molecule: $m \times d$ floats → $4 \times 128 = 512$ dims vs. 128
- For ~5,316 complexes: $5316 \times 512 \times 4$ bytes ≈ 10.9 MB (negligible)
- GPU memory during hook extraction: need to keep $m$ intermediate tensors → ~2× peak memory

### 7.2 Compatibility with TargetDiff Architecture

Need to verify the number and structure of encoder layers in the TargetDiff model:

```python
# Inspect model architecture
for name, module in model.named_modules():
    if 'layer' in name.lower() or 'block' in name.lower():
        print(name, type(module))
```

Expected: 6–9 message-passing layers in the SchNet/EGNN backbone.

### 7.3 Frozen vs Trainable

The encoder layers themselves remain **frozen** (pretrained TargetDiff weights). Only the fusion module parameters are trained.

---

## 8. Implementation Checklist

### Stage 0: Infrastructure
- [x] Inspect TargetDiff encoder to identify hookable layers
- [x] Implement multi-layer extraction in `sampler.py` (used `return_layer_h=True`, no hooks needed)
- [x] Write extraction tests (T2.1–T2.5)
- [x] Write `s08b_extract_multilayer.py` — extract all layers for all complexes

### Stage 1: Single-Layer Probing
- [x] Write `s09a_single_layer_probe.py` — GP per layer
- [x] Run E1.1: per-layer GP metrics
- [x] Run E1.2: CKA similarity matrix
- [x] **Gate 1 decision**: ✅ PROCEED — L8 val R²=0.250 > L9 val R²=0.232, ratio=1.077

### Stage 2: Weighted Sum (proceed only if Gate 1 passes)
- [x] Implement `WeightedSumFusion` in `layer_fusion.py`
- [x] Write unit tests T1.1–T1.2
- [x] Write `s09b_weighted_sum_fusion.py`
- [x] Run E2.1: weighted sum vs. best single layer
- [x] Run E2.2: inspect learned weights
- [x] **Gate 2 decision**: ❌ STOP — weighted sum val R²=0.239 < best single L8 R²=0.250 (−4.5%)
  - top2 (L8,L6): weights collapsed to L8=0.999, val R²=0.217
  - top4 (L8,L6,L9,L5): L9=0.653, L8=0.346, val R²=0.231
  - all (L0–L9): L9=0.592, L8=0.407, all others→0, val R²=0.239
  - Conclusion: weighted sum degenerates to L8+L9 blend, no improvement over single layer

### Stage 3: Layer Attention (proceeded despite Gate 2 STOP)
- [x] Implement `LayerAttentionFusion` in `layer_fusion.py`
- [x] Write unit test T1.3
- [x] Run E3.1: layer attention vs. weighted sum
- [x] Run E3.2: per-sample weight variance analysis
- [x] **Gate 3 decision**: ❌ STOP — attention val R²=0.203 < best single L8 R²=0.250 (−19.0%)
  - top2 (L8,L6): val R²=0.203, test R²=0.494, entropy=0.82, CV=0.57
  - top4 (L8,L6,L9,L5): val R²=0.118, test R²=0.491, CV=1.31
  - all (L0–L9): val R²=0.120, test R²=0.513, CV=4.79
  - E3.2: Weights DO vary across samples (high CV), but extra parameters cause overfitting
  - Notable: test R² improves (0.513 > 0.449) despite poor val R² — train/val overfit

### Stage 4: Concat + MLP (proceed only if Gate 3 passes)
- [ ] Implement `ConcatMLPFusion` in `layer_fusion.py`
- [ ] Write unit test T1.4
- [ ] Run E4.1: concat+MLP vs. layer attention
- [ ] Run E4.2: output dimension sensitivity
- [ ] **Gate 4 decision**: Is nonlinear fusion worth the complexity?

### Stage 5: Concat + Dropout (proceed only if Gate 4 shows instability)
- [ ] Implement `ConcatDropoutFusion` in `layer_fusion.py`
- [ ] Write unit test T1.5
- [ ] Run E5.1–E5.2: dropout rate sweep

### Wrap-Up
- [ ] Write integration tests (T3.1–T3.4) for selected method
- [ ] Run cross-cutting ablations (E6.1–E6.2)
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
