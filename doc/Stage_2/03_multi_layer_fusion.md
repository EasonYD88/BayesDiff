# Sub-Plan 3: Multi-Layer Fusion

> **Priority**: P1 — High  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); independent of Sub-Plans 1–2, combinable  
> **Training Data**: PDBbind v2020 refined set (~5,316 complexes), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
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

**Recommended**: Start with sparse uniform; use layer-attention to learn importance.

### 2.2 Per-Layer Pooling

For each selected layer $l \in \mathcal{S}$:

$$
z^{(l)} = \text{Pool}(H^{(l)}) \in \mathbb{R}^d
$$

where $\text{Pool}$ can be mean pooling (baseline) or attention pooling (from Sub-Plan 2).

### 2.3 Fusion Methods

#### Method A: Concatenation + Bottleneck MLP

$$
z_{\text{concat}} = [z^{(l_1)}; z^{(l_2)}; \dots; z^{(l_m)}] \in \mathbb{R}^{m \cdot d}
$$

$$
z_{\text{fuse}} = \text{MLP}_{\text{bottleneck}}(z_{\text{concat}}) \in \mathbb{R}^{d_{\text{out}}}
$$

**Bottleneck MLP**: $\mathbb{R}^{m \cdot d} \xrightarrow{\text{Linear}} \mathbb{R}^{2d_{\text{out}}} \xrightarrow{\text{ReLU+LN}} \mathbb{R}^{d_{\text{out}}}$

#### Method B: Learned Weighted Sum

$$
\beta_l = \frac{\exp(w_l)}{\sum_{k \in \mathcal{S}} \exp(w_k)}, \quad z_{\text{fuse}} = \sum_{l \in \mathcal{S}} \beta_l z^{(l)}
$$

where $\{w_l\}$ are learnable scalar parameters. Very lightweight; $|\mathcal{S}|$ extra parameters.

#### Method C: Layer-Attention Fusion

$$
\beta_l = \frac{\exp(u^\top \tanh(W z^{(l)}))}{\sum_{k \in \mathcal{S}} \exp(u^\top \tanh(W z^{(k)}))}
$$

$$
z_{\text{fuse}} = \sum_{l \in \mathcal{S}} \beta_l z^{(l)}
$$

**Advantage**: Input-dependent layer weighting (different molecules may benefit from different layers).

#### Method D: Concatenation + Layer Dropout

Same as Method A, but during training, randomly drop entire layers with probability $p_{\text{drop}}$:

$$
z_{\text{concat}} = [m_1 z^{(l_1)}; m_2 z^{(l_2)}; \dots; m_m z^{(l_m)}], \quad m_i \sim \text{Bernoulli}(1 - p_{\text{drop}})
$$

Prevents co-adaptation between layers; improves robustness.

---

## 3. Implementation Plan

### 3.1 Modifications to `bayesdiff/sampler.py`

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

### 3.2 New Module: `bayesdiff/layer_fusion.py`

```python
class LayerFusion(nn.Module):
    """Multi-layer feature fusion module."""
    
    def __init__(self, embed_dim: int, n_layers: int, output_dim: int,
                 method: str = 'concat_mlp'):
        """
        Args:
            embed_dim: dimension of each layer's pooled embedding
            n_layers: number of layers to fuse
            output_dim: final output dimension
            method: 'concat_mlp' | 'weighted_sum' | 'layer_attention' | 'concat_dropout'
        """
        ...
    
    def forward(self, layer_embeddings: List[Tensor]):
        """
        Args:
            layer_embeddings: list of (B, d) tensors, one per layer
        
        Returns:
            z_fuse: (B, d_out) fused representation
            layer_weights: (B, n_layers) or (n_layers,) — for interpretability
        """
        ...


class ConcatMLPFusion(nn.Module):
    """Method A: Concatenation + bottleneck MLP."""
    def __init__(self, embed_dim, n_layers, output_dim):
        ...

class WeightedSumFusion(nn.Module):
    """Method B: Learned scalar weights per layer."""
    def __init__(self, n_layers):
        ...

class LayerAttentionFusion(nn.Module):
    """Method C: Input-dependent layer attention."""
    def __init__(self, embed_dim, hidden_dim=64):
        ...

class ConcatDropoutFusion(nn.Module):
    """Method D: Concatenation with layer dropout."""
    def __init__(self, embed_dim, n_layers, output_dim, drop_prob=0.1):
        ...
```

### 3.3 New Pipeline Script: `scripts/pipeline/s08b_extract_multilayer.py`

```python
"""
Extract multi-layer embeddings from TargetDiff encoder for all pockets.

Usage:
    python scripts/pipeline/s08b_extract_multilayer.py \
        --data_dir data/pdbbind_v2020/processed \
        --layers 2 4 6 8 \
        --output_dir data/multilayer_embeddings/ \
        --device cuda

Output:
    data/multilayer_embeddings/{pdb_code}.npz
        - 'layer_2': (M, d) embeddings from layer 2
        - 'layer_4': (M, d) embeddings from layer 4
        - ...
        - 'z_global': (M, d) last layer mean-pooled (for backward compatibility)
"""
```

### 3.4 Modifications to Downstream Modules

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

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_concat_mlp_shape` | Output shape = (B, d_out) for various inputs |
| T1.2 | `test_weighted_sum_weights` | Weights sum to 1; positive; gradients flow |
| T1.3 | `test_layer_attention_input_dependent` | Different inputs → different layer weights |
| T1.4 | `test_concat_dropout_train_vs_eval` | Dropout active in train, inactive in eval |
| T1.5 | `test_gradient_flow_all_methods` | All fusion method parameters receive gradients |
| T1.6 | `test_single_layer_input` | When n_layers=1, output ≈ identity transform |
| T1.7 | `test_numerical_stability` | No NaN/Inf with extreme values |
| T1.8 | `test_learned_weights_initialization` | Initial weights are roughly uniform |

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

### 4.4 Ablation Experiments

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A3.1 | Last layer only (baseline) | Reference |
| A3.2 | Layers {L/2, L} + concat MLP | Minimal multi-layer |
| A3.3 | Layers {L/4, L/2, 3L/4, L} + concat MLP | Four-scale |
| A3.4 | All layers + concat MLP | Maximum information |
| A3.5 | Layers {L/4, L/2, 3L/4, L} + weighted sum | Lightweight fusion |
| A3.6 | Layers {L/4, L/2, 3L/4, L} + layer attention | Data-driven fusion |
| A3.7 | Layers {L/4, L/2, 3L/4, L} + concat dropout | Regularized fusion |
| A3.8 | Layer attention + attention pooling (Sub-Plan 2) | Combined approach |
| A3.9 | Output dim sensitivity: 64, 128, 256, 512 | Capacity analysis |

---

## 5. Evaluation & Success Criteria

### 5.1 Quantitative Metrics

| Metric | Stage 1 Baseline | Success | Stretch |
|--------|------------------|---------|---------|
| $R^2$ | 0.120 | ≥ 0.16 | ≥ 0.22 |
| Spearman $\rho$ | 0.369 | ≥ 0.43 | ≥ 0.50 |
| NLL | baseline | ≥ 3% reduction | ≥ 10% reduction |

### 5.2 Diagnostic Metrics

- **Learned layer weights** (weighted sum / layer attention): Which layers contribute most?
- **Per-layer representation quality**: Train GP on each layer separately → which layers are most informative?
- **Representation similarity** (CKA): How similar are different layers' representations?

### 5.3 Expected Findings (Hypotheses)

1. Middle layers contribute more than shallow layers for affinity prediction
2. Combining shallow + deep layers outperforms deep-only
3. Layer attention assigns non-uniform weights (i.e., not all layers equally useful)
4. Multi-layer fusion has larger benefit when combined with attention pooling (Sub-Plan 2)

---

## 6. Paper Integration

### 6.1 Methods Section (Draft)

> **§3.X.3 Multi-Scale Feature Fusion**
> 
> Different layers of the SE(3)-equivariant encoder capture information at different spatial scales. To preserve multi-scale features relevant to binding affinity, we extract pooled representations from multiple encoder layers $\mathcal{S} = \{l_1, \dots, l_m\}$ and fuse them via a learned aggregation function:
> 
> $$z_{\text{fuse}} = \Psi(z^{(l_1)}, \dots, z^{(l_m)})$$
> 
> where $z^{(l)} = \text{Pool}(H^{(l)})$ is the pooled embedding from the $l$-th encoder layer.
> 
> We compare four fusion strategies: (A) concatenation with bottleneck MLP, (B) learned weighted sum, (C) input-dependent layer attention, and (D) concatenation with layer dropout regularization (Table X). Layer attention fusion achieves the best performance, with learned weights indicating that [intermediate / deep] layers contribute most to binding affinity prediction.

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. L.1 | Layer weight distribution (violin/box plot) | Which layers matter most |
| Fig. L.2 | CKA similarity matrix across layers | Layer redundancy analysis |
| Fig. L.3 | Per-layer GP performance (bar chart) | Each layer's individual quality |
| Fig. L.4 | Ablation: fusion method comparison | Justify chosen method |
| Fig. L.5 | Fused vs. last-layer t-SNE | Visual quality improvement |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. L.1 | Layer selection ablation (A3.1–A3.4) |
| Tab. L.2 | Fusion method comparison (A3.5–A3.7) |
| Tab. L.3 | Output dimension sensitivity (A3.9) |

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

- [ ] Inspect TargetDiff encoder to identify hookable layers
- [ ] Implement forward hook mechanism in `sampler.py`
- [ ] Implement `ConcatMLPFusion` in `layer_fusion.py`
- [ ] Implement `WeightedSumFusion` in `layer_fusion.py`
- [ ] Implement `LayerAttentionFusion` in `layer_fusion.py`
- [ ] Implement `ConcatDropoutFusion` in `layer_fusion.py`
- [ ] Write `s08b_extract_multilayer.py` pipeline script
- [ ] Write unit tests (T1.1–T1.8)
- [ ] Write hook tests (T2.1–T2.4)
- [ ] Write integration tests (T3.1–T3.4)
- [ ] Run ablation experiments (A3.1–A3.9)
- [ ] Analyze learned layer weights
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
