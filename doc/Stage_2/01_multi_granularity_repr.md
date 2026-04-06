# Sub-Plan 1: Multi-Granularity Representation

> **Priority**: P0 — Critical  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集)  
> **Training Data**: PDBbind v2020 R1 general set (19,037), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 2–3 weeks implementation + 1 week testing  
> **Paper Section**: §3.X Enhanced Representation Learning

---

## 1. Motivation

The current BayesDiff pipeline compresses all atomic information into a single mean-pooled vector:

$$
z_{\text{global}} = \frac{1}{|A|} \sum_{i \in A} h_i^{(L)} \in \mathbb{R}^{128}
$$

This discards three categories of information critical for binding affinity prediction:

| Information Lost | Why It Matters | Example |
|-----------------|----------------|---------|
| **Atom-level features** | Key pharmacophore atoms dominate binding | H-bond donor/acceptor positions |
| **Interaction-level features** | Binding is a *pairwise* phenomenon | Protein-ligand contact patterns |
| **Structural hierarchy** | Global shape ≠ local fit quality | A molecule can have correct size but wrong local geometry |

**Goal**: Replace the single $z_{\text{global}} \in \mathbb{R}^{128}$ with a multi-granularity representation:

$$
z_{\text{new}} = \phi(z_{\text{atom}}, z_{\text{interaction}}, z_{\text{global}}) \in \mathbb{R}^{d_{\text{new}}}
$$

---

## 2. Architecture Design

### 2.1 Three-Level Representation

```
  TargetDiff Encoder
         │
         ├──→ Level 1: Atom-Level Tokens
         │       h_i^(L) for selected atoms
         │       → Set-Transformer / attention pooling
         │       → z_atom ∈ ℝ^d₁
         │
         ├──→ Level 2: Interaction Graph
         │       Bipartite graph G = (V_L, V_P, E)
         │       → Interaction-GNN (2-3 layers)
         │       → z_interaction ∈ ℝ^d₂
         │
         └──→ Level 3: Global Embedding
                 Mean/attention pool over all atoms
                 → z_global ∈ ℝ^d₃ (current method, retained)
```

### 2.2 Interaction Graph Construction

Given ligand atom positions $\{r_i^{(L)}\}$ and pocket residue positions $\{r_j^{(P)}\}$:

1. **Contact edges**: $(i, j)$ if $\|r_i^{(L)} - r_j^{(P)}\| \leq d_{\text{cutoff}}$ (default: 4.5 Å)
2. **Edge features** for each contact $(i, j)$:
   - Distance: $d_{ij} = \|r_i^{(L)} - r_j^{(P)}\|$
   - Radial basis: $\text{RBF}(d_{ij}) \in \mathbb{R}^{16}$ (Gaussian expansion, 0–8 Å, 16 centers)
   - Atom type pair: one-hot encoding of (ligand atom type, residue type)
   - Interaction type indicator: hydrogen bond / hydrophobic / π-π / salt bridge / van der Waals
3. **Node features**:
   - Ligand atoms: $h_i^{(L)}$ from TargetDiff encoder
   - Pocket residues: $h_j^{(P)}$ from pocket encoder or pre-computed (ESM-2 / per-residue features)

### 2.3 Interaction GNN

A lightweight message-passing network operating on the bipartite graph:

$$
m_{ij}^{(k+1)} = \text{MLP}_{\text{msg}}([h_i^{(k)}; h_j^{(k)}; e_{ij}])
$$

$$
h_i^{(k+1)} = h_i^{(k)} + \text{MLP}_{\text{upd}}\left(\sum_{j \in \mathcal{N}(i)} m_{ij}^{(k+1)}\right)
$$

After $K=2$ layers:

$$
z_{\text{interaction}} = \frac{1}{|E|} \sum_{(i,j) \in E} \text{MLP}_{\text{edge}}([h_i^{(K)}; h_j^{(K)}; e_{ij}])
$$

### 2.4 Fusion Function $\phi$

**Option A — Concatenation + MLP** (simple, recommended for first iteration):

$$
z_{\text{new}} = \text{MLP}_{\text{fuse}}([z_{\text{atom}}; z_{\text{interaction}}; z_{\text{global}}])
$$

where $\text{MLP}_{\text{fuse}}: \mathbb{R}^{d_1+d_2+d_3} \to \mathbb{R}^{d_{\text{out}}}$ with one hidden layer + LayerNorm + ReLU.

**Option B — Gated fusion** (for ablation):

$$
g = \sigma(W_g [z_{\text{atom}}; z_{\text{interaction}}; z_{\text{global}}] + b_g) \in \mathbb{R}^3
$$

$$
z_{\text{new}} = g_1 \cdot W_1 z_{\text{atom}} + g_2 \cdot W_2 z_{\text{interaction}} + g_3 \cdot W_3 z_{\text{global}}
$$

---

## 3. Implementation Plan

### 3.1 New Module: `bayesdiff/interaction_graph.py`

```python
class InteractionGraphBuilder:
    """Constructs pocket-ligand bipartite interaction graphs."""
    
    def __init__(self, cutoff: float = 4.5, rbf_centers: int = 16, max_rbf_dist: float = 8.0):
        ...
    
    def build_graph(self, ligand_pos, ligand_features, pocket_pos, pocket_features):
        """
        Args:
            ligand_pos: (N_L, 3) ligand atom coordinates
            ligand_features: (N_L, d) ligand atom embeddings from encoder
            pocket_pos: (N_P, 3) pocket residue Cα coordinates
            pocket_features: (N_P, d_p) pocket residue features
        
        Returns:
            PyG Data object with:
              - x: (N_L + N_P, d_max) node features
              - edge_index: (2, E) bipartite edges
              - edge_attr: (E, d_edge) edge features
              - batch: (N_L + N_P,) batch indices
              - node_type: (N_L + N_P,) 0=ligand, 1=pocket
        """
        ...
    
    def _compute_edge_features(self, dist, ligand_types, pocket_types):
        """Compute RBF distance + atom pair + interaction type features."""
        ...
    
    def _gaussian_rbf(self, dist):
        """Gaussian radial basis function expansion."""
        ...
```

### 3.2 New Module: `bayesdiff/interaction_gnn.py`

```python
class InteractionGNN(nn.Module):
    """Lightweight GNN for pocket-ligand interaction encoding."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128, n_layers=2, output_dim=128):
        ...
    
    def forward(self, x, edge_index, edge_attr, node_type):
        """
        Returns:
            z_interaction: (batch_size, output_dim) interaction-level embedding
        """
        ...

class BipartiteMessagePassing(nn.Module):
    """Message passing layer for bipartite graphs."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        ...
    
    def forward(self, x, edge_index, edge_attr):
        ...
```

### 3.3 New Module: `bayesdiff/multi_granularity.py`

```python
class MultiGranularityEncoder(nn.Module):
    """Combines atom-level, interaction-level, and global-level representations."""
    
    def __init__(self, atom_dim, interaction_dim, global_dim, output_dim=256, fusion='concat_mlp'):
        ...
    
    def forward(self, atom_embeddings, interaction_graph, global_embedding):
        """
        Args:
            atom_embeddings: (N_atoms, d) per-atom features from encoder
            interaction_graph: PyG Data from InteractionGraphBuilder
            global_embedding: (d,) current mean-pooled embedding
        
        Returns:
            z_new: (d_out,) multi-granularity representation
        """
        z_atom = self.atom_encoder(atom_embeddings)        # (d₁,)
        z_interaction = self.interaction_gnn(interaction_graph)  # (d₂,)
        z_global = self.global_proj(global_embedding)       # (d₃,)
        return self.fuse(z_atom, z_interaction, z_global)   # (d_out,)
```

### 3.4 Modifications to `bayesdiff/sampler.py`

**Change**: Expose atom-level embeddings and positions alongside the current global embedding.

```python
# Current: returns z_global only
# New: returns dict with multiple levels

class TargetDiffSampler:
    def sample_and_embed(self, pocket_pdb, num_samples=64):
        ...
        return {
            'z_global': z_global,          # (M, d) — as before
            'atom_embeddings': atom_embs,  # list of M tensors, each (N_i, d)
            'atom_positions': atom_pos,    # list of M tensors, each (N_i, 3)
            'atom_types': atom_types,      # list of M tensors, each (N_i,)
            'pocket_positions': pocket_pos,  # (N_P, 3)
            'pocket_features': pocket_feat,  # (N_P, d_p)
        }
```

### 3.5 New Pipeline Script: `scripts/pipeline/s08_extract_atom_embeddings.py`

Extract and store atom-level data for all pockets:

```python
"""
Usage:
    python scripts/pipeline/s08_extract_atom_embeddings.py \
        --data_dir data/pdbbind_v2020/processed \
        --output_dir data/atom_embeddings/ \
        --device cuda

Output per pocket:
    data/atom_embeddings/{pdb_code}/
        mol_{i}_atoms.npz   # atom_embeddings, atom_positions, atom_types
        pocket.npz           # pocket_positions, pocket_features
"""
```

### 3.6 New Pipeline Script: `scripts/pipeline/s09_build_interaction_graphs.py`

```python
"""
Usage:
    python scripts/pipeline/s09_build_interaction_graphs.py \
        --atom_dir data/atom_embeddings/ \
        --output_dir data/interaction_graphs/ \
        --cutoff 4.5

Output:
    data/interaction_graphs/{pdb_code}/
        mol_{i}_graph.pt   # PyG Data object
"""
```

### 3.7 Modifications to `bayesdiff/gen_uncertainty.py`

Extend `estimate_gen_uncertainty()` to accept higher-dimensional multi-granularity embeddings:

```python
def estimate_gen_uncertainty(embeddings, shrinkage="ledoit_wolf", ...):
    """
    Now accepts embeddings of any dimension d (was fixed at 128).
    Ledoit-Wolf shrinkage is even more important at higher d.
    """
    ...
```

---

## 4. Test Plan

### 4.1 Unit Tests: `tests/stage2/test_interaction_graph.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_graph_construction_basic` | Graph builds from synthetic positions; correct node/edge counts |
| T1.2 | `test_cutoff_filtering` | Only edges with distance ≤ cutoff are included |
| T1.3 | `test_rbf_expansion` | RBF output shape = (E, n_centers); values in [0, 1] |
| T1.4 | `test_edge_features_shape` | Edge features have correct dimensionality |
| T1.5 | `test_no_contacts` | Handles case where no atom pairs are within cutoff |
| T1.6 | `test_batch_construction` | Multiple molecules produce correct batched graph |
| T1.7 | `test_determinism` | Same input → same graph (seed-independent for geometry) |

```python
def test_graph_construction_basic():
    """Synthetic: 10 ligand atoms, 20 pocket residues, random positions, cutoff=5.0."""
    builder = InteractionGraphBuilder(cutoff=5.0)
    lig_pos = torch.randn(10, 3)
    lig_feat = torch.randn(10, 128)
    pkt_pos = torch.randn(20, 3)
    pkt_feat = torch.randn(20, 64)
    
    graph = builder.build_graph(lig_pos, lig_feat, pkt_pos, pkt_feat)
    
    assert graph.x.shape[0] == 30  # 10 + 20 nodes
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    assert (graph.node_type[:10] == 0).all()
    assert (graph.node_type[10:] == 1).all()
```

### 4.2 Unit Tests: `tests/stage2/test_interaction_gnn.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_forward_shape` | Output shape = (batch_size, output_dim) |
| T2.2 | `test_gradient_flow` | Gradients flow through all parameters |
| T2.3 | `test_permutation_equivariance` | Reordering atom indices doesn't change output (up to reorder) |
| T2.4 | `test_empty_graph` | Graceful handling of graph with no edges |
| T2.5 | `test_variable_size` | Different-sized molecules in same batch |

### 4.3 Unit Tests: `tests/stage2/test_multi_granularity.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T3.1 | `test_fusion_concat_mlp` | Output dim correct for concat+MLP mode |
| T3.2 | `test_fusion_gated` | Gate values sum to ~1; output shape correct |
| T3.3 | `test_end_to_end_forward` | Full flow from atom embs → z_new |
| T3.4 | `test_backward_pass` | Loss.backward() succeeds; all params have gradients |
| T3.5 | `test_output_scale` | z_new values are in reasonable range (no NaN/Inf) |
| T3.6 | `test_compatibility_with_gp` | z_new can be fed to GPOracle without errors |

### 4.4 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T4.1 | `test_full_pipeline_synthetic` | Synthetic data → interaction graph → GNN → fusion → GP → metrics |
| T4.2 | `test_gen_uncertainty_higher_dim` | gen_uncertainty works with d=256 embeddings |
| T4.3 | `test_delta_method_compatibility` | Jacobian computation works with new representation |
| T4.4 | `test_metric_improvement_synthetic` | On constructed data, multi-gran outperforms mean-pool |

### 4.5 Ablation Tests (for Paper)

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A1.1 | Baseline (mean pool, d=128) | Reference |
| A1.2 | Atom-level only (z_atom) | Isolate atom-level contribution |
| A1.3 | Interaction-level only (z_interaction) | Isolate interaction contribution |
| A1.4 | Global + Atom | Two-level combination |
| A1.5 | Global + Interaction | Two-level combination |
| A1.6 | Full multi-granularity (concat MLP) | Main proposed method |
| A1.7 | Full multi-granularity (gated) | Alternative fusion |
| A1.8 | Cutoff sensitivity: 3.5 / 4.5 / 6.0 / 8.0 Å | Hyperparameter sensitivity |
| A1.9 | GNN layers: 1 / 2 / 3 | Architecture sensitivity |

---

## 5. Evaluation & Success Criteria

### 5.1 Primary Metrics

| Metric | Stage 1 Baseline | Success Threshold | Stretch Goal |
|--------|------------------|-------------------|--------------|
| $R^2$ | 0.120 | ≥ 0.18 (+50%) | ≥ 0.25 |
| Spearman $\rho$ | 0.369 | ≥ 0.45 | ≥ 0.55 |
| NLL | baseline | ≥ 5% reduction | ≥ 15% reduction |
| ECE | 0.034 | ≤ 0.05 (no degradation) | ≤ 0.03 |

### 5.2 Diagnostic Metrics

- **Interaction graph statistics**: avg edges/molecule, avg distance, contact type distribution
- **Representation analysis**: t-SNE/UMAP of z_new colored by pKd → expect better separation
- **Learned gate values** (gated fusion): which level contributes most across pockets

### 5.3 Failure Criteria

- $R^2$ or $\rho$ *decreases* compared to baseline → representation is noisy, not informative
- NLL increases > 10% → uncertainty calibration damaged
- Training diverges → reduce learning rate / add regularization

---

## 6. Paper Integration

### 6.1 Methods Section Text (Draft Outline)

> **§3.X.1 Multi-Granularity Molecular Representation**
> 
> To address the representation bottleneck identified in our Stage 1 analysis, we replace the single mean-pooled embedding with a three-level representation that captures complementary aspects of protein-ligand binding:
> 
> 1. *Atom-level representation* $z_{\text{atom}}$: preserves individual pharmacophore features via attention-weighted pooling over ligand atom embeddings $\{h_i^{(L)}\}$ from the SE(3)-equivariant encoder.
> 
> 2. *Interaction-level representation* $z_{\text{interaction}}$: encodes the binding interface via a lightweight message-passing GNN operating on a pocket-ligand contact graph, where edges connect atom pairs within a distance cutoff of $d_c$ = 4.5 Å.
> 
> 3. *Global representation* $z_{\text{global}}$: the original mean-pooled embedding, capturing overall molecular shape and property distributions.
> 
> These three representations are fused via [concat+MLP / gated fusion] to produce $z_{\text{new}} \in \mathbb{R}^{d_{\text{new}}}$, which replaces the original embedding in all downstream uncertainty computations.

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. X.1 | Architecture diagram with three levels | Explain the method |
| Fig. X.2 | t-SNE: z_global vs z_new, colored by pKd | Show improved representation |
| Fig. X.3 | Ablation bar chart (A1.1–A1.7) | Justify design choices |
| Fig. X.4 | Interaction graph example visualization | Intuitive illustration |
| Fig. X.5 | Gate values heatmap (gated fusion) | Interpretability |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. X.1 | Ablation: representation level vs. metrics |
| Tab. X.2 | Cutoff sensitivity (A1.8) |
| Tab. X.3 | Interaction graph statistics (avg edges, contact types) |

---

## 7. Data Requirements

| Data | Source | Size | Status |
|------|--------|------|--------|
| Ligand atom positions | PDBbind v2020 crystal structures | ~5,316 complexes | ✅ Available |
| Ligand atom embeddings | SE(3) encoder hidden states | ~5,316 × (N_atoms, 128) | ⚠️ Requires forward pass |
| Pocket residue positions | PDBbind v2020 structures | ~5,316 complexes | ✅ Available |
| Pocket residue features | Per-residue encoding | ~5,316 × (N_res, d_p) | ⚠️ Need to extract |
| pKd labels | PDBbind v2020 INDEX | ~5,316 complexes | ✅ Available |

---

## 8. Implementation Checklist

- [ ] Modify `sampler.py` to expose atom-level embeddings and positions
- [ ] Implement `interaction_graph.py` with `InteractionGraphBuilder`
- [ ] Implement `interaction_gnn.py` with `InteractionGNN`
- [ ] Implement `multi_granularity.py` with `MultiGranularityEncoder`
- [ ] Write `s08_extract_atom_embeddings.py` pipeline script
- [ ] Write `s09_build_interaction_graphs.py` pipeline script
- [ ] Update `gen_uncertainty.py` for variable-dim embeddings
- [ ] Update `fusion.py` for variable-dim Jacobians
- [ ] Write all unit tests (T1.1–T3.6)
- [ ] Write integration tests (T4.1–T4.4)
- [ ] Run ablation experiments (A1.1–A1.9)
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
