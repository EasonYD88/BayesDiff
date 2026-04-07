# Sub-Plan 1: Multi-Granularity Representation (主干整合框架)

> **角色**: Phase A **Step 3** — 串行链的最下游，负责把 atom / interaction / global 三种信息统一成最终表示 $z_{\text{new}}$  
> **Priority**: P0 — Critical  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); **Sub-Plan 3** (Token-Level Layer Fusion → $\tilde{h}_i$); **Sub-Plan 2** (Attention Pooling → $z_{\text{atom}}$)  
> **Training Data**: PDBbind v2020 R1 general set (19,037), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Data Split**: 默认使用 `splits.json`（= fold 0 单次划分: Train / Val / CASF-2016 Test）。5-fold 仅在需要稳健性评估或超参选择时启用（见 §8）。  
> **Estimated Effort**: 1–2 weeks MVP implementation + testing; 1 week v2 extensions (conditional on MVP gain)  
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

### 在串行链中的位置

本 Sub-Plan 是串行链 (SP3 → SP2 → SP1) 的最终环节，也是**主干整合框架**。它消费上游产出：
- $\tilde{h}_i$ from SP3 (token-level layer fusion) — 用于构建交互图和全局池化
- $z_{\text{atom}}$ from SP2 (attention pooling on $\tilde{h}_i$) — 三路融合的其中一路

```
  ┌─────────────────────────────────────────┐
  │  SP3: Token-Level Layer Fusion          │
  │  → {h̃_i} (per-atom fused repr)         │
  └──────────────────┬──────────────────────┘
                     │
          ┌──────────┼──────────────────┐
          ▼          │                  │
  ┌──────────────┐   │                  │
  │ SP2: Attn    │   │                  │
  │ Pool({h̃_i}) │   │                  │
  │ → z_atom     │   │                  │
  └──────┬───────┘   │                  │
         │           ▼                  ▼
         │   ╔══════════════╗   ╔══════════════╗
         │   ║ Interaction  ║   ║ Global Pool  ║
         │   ║ Graph + GNN  ║   ║ mean({h̃_i}) ║
         │   ║ → z_inter    ║   ║ → z_global   ║
         │   ╚══════╤═══════╝   ╚══════╤═══════╝
         │          │                  │
         └──────────┼──────────────────┘
                    ▼
  ╔═════════════════════════════════════════╗
  ║  【本 Sub-Plan: Multi-Granularity      ║
  ║    Fusion】                            ║
  ║  z_new = MLP([z_atom; z_inter; z_glob])║
  ╚═════════════════╤═══════════════════════╝
                    ▼
              GP / DKL / Ranking Head
```

---

## 2. Architecture Design

### 2.1 Three-Level Representation

> **架构变更 (2026-04-06)**：三层表示不再从原始编码器最后一层 $h_i^{(L)}$ 直接派生，
> 而是基于 SP3 token-level layer fusion 输出的 $\tilde{h}_i$。

```
  TargetDiff Encoder (Frozen)
         │
         │ 提取多层 hidden states: H^(1), ..., H^(L)
         ▼
  SP3: Token-Level Layer Fusion
         │
         │ {h̃_i} = fused per-atom representations
         ▼
         ├──→ Level 1: Atom-Level Summary (SP2)
         │       attention pooling over {h̃_i}
         │       → z_atom ∈ ℝ^d₁
         │
         ├──→ Level 2: Interaction Graph (本 SP1)
         │       Bipartite graph G = (V_L, V_P, E)
         │       ligand nodes use {h̃_i} as features
         │       → Interaction-GNN (2-3 layers)
         │       → z_interaction ∈ ℝ^d₂
         │
         └──→ Level 3: Global Embedding (本 SP1)
                 Mean pool over {h̃_i}
                 → z_global ∈ ℝ^d₃
```

**与旧架构的区别**：
- 旧：三路各自独立从 $h_i^{(L)}$ 出发
- 新：三路共享同一个 $\tilde{h}_i$（经过 SP3 多层融合后的更优表示）
- SP2 attention pooling 的输入从 $h_i^{(L)}$ 变为 $\tilde{h}_i$
- 交互图的 ligand node features 从 $h_i^{(L)}$ 变为 $\tilde{h}_i$

### 2.2 Interaction Graph Construction

> **Design rationale (MVP-first)**: The pocket side uses **heavy atoms** (all non-hydrogen atoms), not Cα.  
> Cα coordinates are too coarse — they sit at the backbone and are often >5 Å from the true contact surface, making any distance cutoff unreliable. Heavy atoms faithfully represent the side-chain geometry that actually contacts the ligand.  
> If graph size becomes a bottleneck, the fallback is **side-chain centroid / functional-atom proxy** per residue (e.g., Nζ for Lys, Oδ for Asp), NOT Cα.

Given ligand atom positions $\{r_i^{(L)}\}$ and pocket heavy-atom positions $\{r_j^{(P)}\}$:

1. **Pocket node extraction**: For each pocket residue within 10 Å of the ligand center-of-mass, extract all heavy atoms (C, N, O, S, etc.). Typical pocket yields 200–500 heavy atoms.
2. **Contact edges**: $(i, j)$ if $\|r_i^{(L)} - r_j^{(P)}\| \leq d_{\text{cutoff}}$ (default: 4.5 Å). At heavy-atom resolution this cutoff is geometrically faithful.
3. **Edge features (MVP)** for each contact $(i, j)$:
   - Distance: $d_{ij} = \|r_i^{(L)} - r_j^{(P)}\|$
   - Radial basis: $\text{RBF}(d_{ij}) \in \mathbb{R}^{16}$ (Gaussian expansion, 0–8 Å, 16 centers)
   - Ligand atom type: one-hot encoding of element (C/N/O/S/F/Cl/Br/P/other)
   - Pocket atom type: one-hot encoding of element + parent residue type (20 amino acids)
4. **Edge features (deferred to v2, after MVP shows stable gain)**:
   - Interaction type indicator: hydrogen bond / hydrophobic / π-π / salt bridge / van der Waals
   - Rationale for deferral: rule-based interaction labels on generated (noisy) poses inject hard-coded chemical priors that are difficult to validate and may introduce systematic bias. Adding them only after the geometry-only branch shows improvement isolates the contribution of each component.
5. **Node features**:
   - Ligand atoms: $\tilde{h}_i$ from SP3 token-level layer fusion (replaces $h_i^{(L)}$ from original encoder)
   - Pocket heavy atoms: element embedding + parent residue embedding (learned, d=32 each → concatenated to d=64). No external pretrained encoder in MVP.

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
    """Constructs pocket-ligand bipartite interaction graphs at heavy-atom resolution."""
    
    def __init__(self, cutoff: float = 4.5, rbf_centers: int = 16, max_rbf_dist: float = 8.0,
                 pocket_radius: float = 10.0, use_heavy_atoms: bool = True):
        ...
    
    def extract_pocket_heavy_atoms(self, protein_atoms, ligand_center, radius=10.0):
        """
        Extract heavy atoms within `radius` of ligand center-of-mass.
        
        Args:
            protein_atoms: dict with 'positions' (N_all, 3), 'elements' (N_all,),
                          'residue_types' (N_all,), 'is_hydrogen' (N_all,) bool mask
            ligand_center: (3,) ligand center-of-mass
            radius: pocket extraction radius (default 10.0 Å)
        
        Returns:
            pocket_pos: (N_P, 3) heavy atom coordinates
            pocket_elements: (N_P,) element indices
            pocket_residue_types: (N_P,) residue type indices
        """
        ...
    
    def build_graph(self, ligand_pos, ligand_features, pocket_pos, pocket_features,
                    ligand_elements=None, pocket_elements=None, pocket_residue_types=None):
        """
        Args:
            ligand_pos: (N_L, 3) ligand atom coordinates
            ligand_features: (N_L, d) ligand atom embeddings from encoder
            pocket_pos: (N_P, 3) pocket heavy-atom coordinates
            pocket_features: (N_P, d_p) pocket atom features (element + residue embeddings)
            ligand_elements: (N_L,) ligand element type indices
            pocket_elements: (N_P,) pocket element type indices
            pocket_residue_types: (N_P,) residue type indices for each pocket atom
        
        Returns:
            PyG Data object with:
              - x: (N_L + N_P, d_max) node features
              - edge_index: (2, E) bipartite edges
              - edge_attr: (E, d_edge) edge features (RBF + atom types, NO interaction type)
              - batch: (N_L + N_P,) batch indices
              - node_type: (N_L + N_P,) 0=ligand, 1=pocket
        """
        ...
    
    def _compute_edge_features(self, dist, ligand_elements, pocket_elements, pocket_residue_types):
        """Compute RBF distance + ligand atom type + pocket atom/residue type (MVP: no interaction type)."""
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
    """
    主干整合框架：combines atom-level, interaction-level, and global-level representations.
    
    在串行链中的位置：
    - 接收 SP3 输出的 {h̃_i} (layer-fused per-atom repr)
    - 接收 SP2 输出的 z_atom (attention-pooled)
    - 自身负责：interaction graph → z_interaction, mean pool → z_global
    - 最终融合三路 → z_new
    """
    
    def __init__(self, atom_dim, interaction_dim, global_dim, output_dim=128, fusion='concat_mlp'):
        ...
    
    def forward(self, h_tilde, z_atom, pocket_data):
        """
        Args:
            h_tilde: (N_atoms, d) per-atom layer-fused features from SP3
            z_atom: (d₁,) attention-pooled embedding from SP2
            pocket_data: dict with pocket positions, elements, residue types
        
        Returns:
            z_new: (d_out,) multi-granularity representation
        """
        # Build interaction graph from h_tilde + pocket
        interaction_graph = self.graph_builder(h_tilde, pocket_data)
        z_interaction = self.interaction_gnn(interaction_graph)  # (d₂,)
        z_global = h_tilde.mean(dim=0)                          # (d₃,)
        return self.fuse(z_atom, z_interaction, z_global)        # (d_out,)
```

### 3.4 Modifications to `bayesdiff/sampler.py`

**Change**: Expose per-layer, per-atom embeddings (token-level) alongside positions and pocket data. This supports the full serial chain: SP3 token-level fusion → SP2 attention pool → SP1 interaction graph.

```python
# Current: returns z_global only
# New: returns dict with multi-layer atom-level data for the serial chain

class TargetDiffSampler:
    def sample_and_embed(self, pocket_pdb, num_samples=64, extract_layers=None):
        ...
        return {
            'z_global': z_global,                 # (M, d) — as before (backward compat)
            'atom_embeddings_per_layer': layer_h,  # dict: layer_idx → list of M × (N_i, d) 
                                                   # ← SP3 token-level fusion input
            'atom_positions': atom_pos,            # list of M tensors, each (N_i, 3)
            'atom_types': atom_types,              # list of M tensors, each (N_i,)
            'pocket_positions': pocket_pos,        # (N_P, 3) — heavy-atom coords
            'pocket_elements': pocket_elem,        # (N_P,) element type indices
            'pocket_residue_types': pocket_res,    # (N_P,) amino acid type indices
            'pocket_features': pocket_feat,        # (N_P, d_p) — element + residue embeddings
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
        pocket.npz           # pocket_heavy_atom_positions, pocket_elements, pocket_residue_types
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
| T4.2 | `test_gen_uncertainty_higher_dim` | gen_uncertainty works with d=128 (or 192) embeddings |
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
| A1.10 | z_interaction with **shuffled contact edges** (random bipartite edges, same edge count) | Sanity check: does contact topology matter, or is it just node features? If z_interaction doesn’t beat the shuffled-edge control, the interaction branch is not learning real structure. |
| A1.11 | output_dim: 128 / 192 / 256 | Embedding dimension vs GP sample efficiency |

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

- **Interaction graph statistics**: avg edges/molecule, avg distance, heavy-atom count per pocket
- **Representation analysis**: t-SNE/UMAP of z_new colored by pKd → expect better separation
- **Shuffled-edge Δ**: performance gap between real-topology and shuffled-edge z_interaction (must be significant)

### 5.3 Failure Criteria

- $R^2$ or $\rho$ *decreases* compared to baseline → representation is noisy, not informative
- NLL increases > 10% → uncertainty calibration damaged
- Training diverges → reduce learning rate / add regularization

---

## 6. Paper Integration

### 6.1 Methods Section Text (Draft Outline)

> **§3.X.1 Multi-Granularity Molecular Representation**
> 
> To address the representation bottleneck identified in our Stage 1 analysis, we propose a serial representation pipeline that systematically enriches the embedding at three stages:
> 
> **Token-level multi-layer fusion.** Rather than using only the final layer's atom embeddings $\{h_i^{(L)}\}$, we fuse representations from multiple encoder layers at the atom level. For each atom $i$, we compute $\tilde{h}_i = \sum_{l \in \mathcal{S}} \beta_{l,i} h_i^{(l)}$, where $\beta_{l,i}$ are input-dependent layer attention weights, and $\mathcal{S}$ is a selected subset of encoder layers. This preserves multi-scale geometric and chemical information per atom that would be lost by extracting only the final layer.
> 
> **Attention-based atom summary.** We apply learned attention pooling over the fused atom representations $\{\tilde{h}_i\}$ to obtain $z_{\text{atom}} = \sum_i \alpha_i \tilde{h}_i$, where $\alpha_i$ reflects each atom's importance for binding affinity prediction.
> 
> **Interaction-level and global-level embedding.** Using the same $\{\tilde{h}_i\}$, we construct a bipartite pocket-ligand contact graph at **heavy-atom resolution** (not Cα), with edges connecting atom pairs within a distance cutoff of $d_c$ = 4.5 Å. A lightweight message-passing GNN encodes this graph into $z_{\text{interaction}}$, capturing the binding interface geometry. A simple mean pool over $\{\tilde{h}_i\}$ yields $z_{\text{global}}$, retaining overall molecular shape information.
> 
> The three representations are fused via [concat+MLP / gated fusion] to produce $z_{\text{new}} \in \mathbb{R}^{d_{\text{new}}}$, which replaces the original embedding in all downstream uncertainty computations.

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. X.1 | Architecture diagram with three levels | Explain the method |
| Fig. X.2 | t-SNE: z_global vs z_new, colored by pKd | Show improved representation |
| Fig. X.3 | Ablation bar chart (A1.1–A1.7, **A1.10 shuffled-edge**) | Justify design choices |
| Fig. X.4 | Interaction graph example visualization | Intuitive illustration |
| Fig. X.5 | Gate values heatmap (gated fusion) | Interpretability |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. X.1 | Ablation: representation level vs. metrics |
| Tab. X.2 | Cutoff sensitivity (A1.8) |
| Tab. X.3 | Interaction graph statistics (avg edges, heavy-atom count, pocket size) |
| Tab. X.4 | Shuffled-edge control: real topology vs random edges (A1.10) |

---

## 7. Data Requirements

| Data | Source | Size | Status |
|------|--------|------|--------|
| Ligand atom positions | PDBbind v2020 crystal structures | ~5,316 complexes | ✅ Available |
| Ligand **multi-layer** atom embeddings | SE(3) encoder hidden states, all layers | ~5,316 × L × (N_atoms, 128) | ✅ Available (SP3 已提取 `all_multilayer_embeddings.npz`) |
| Layer-fused atom embeddings ($\tilde{h}_i$) | SP3 token-level fusion output | ~5,316 × (N_atoms, d) | ⚠️ 需 SP3 token-level fusion 模块 |
| $z_{\text{atom}}$ | SP2 attention pooling on $\tilde{h}_i$ | ~5,316 × (d₁,) | ⚠️ 需 SP2 attention pooling 模块 |
| Pocket heavy-atom positions | PDBbind v2020 structures (all non-H atoms within 10 Å of ligand) | ~5,316 complexes × 200–500 atoms/pocket | ✅ Available (extract from PDB) |
| Pocket atom element types | Parsed from PDB ATOM records | ~5,316 complexes | ✅ Available |
| Pocket residue types (per atom) | Parsed from PDB ATOM records | ~5,316 complexes | ✅ Available |
| pKd labels | PDBbind v2020 INDEX | ~5,316 complexes | ✅ Available |

---

## 8. Implementation Phasing

### Phase 1 — MVP (must ship first, gate for Phase 2)

> **前置条件**：SP3 token-level fusion 和 SP2 attention pooling 须先实现并验证。

| Component | MVP Scope | Deferred to v2 |
|-----------|-----------|----------------|
| **上游输入** | SP3 TokenLevelAttention + SP2 SelfAttentionPooling | 更复杂的 SP3/SP2 变体 |
| **Data split** | `splits.json`（fold 0 单次 Train/Val/Test） | 5-fold（`splits_5fold.json`）仅在超参选择或稳健性评估时启用 |
| Pocket graph nodes | Heavy atoms (all non-H) | — |
| Edge features | distance + RBF + ligand element + pocket element/residue type | H-bond / hydrophobic / π-π / salt bridge / vdW interaction labels |
| Ligand node features | $\tilde{h}_i$ from SP3 (不再是 $h_i^{(L)}$) | — |
| Fusion | concat + MLP | Gated fusion |
| Output dim | 128 | 192 / 256 (ablation A1.11) |
| Ablation | A1.1–A1.6, **A1.10 (shuffled-edge control)** | A1.7 (gated), A1.8–A1.9, A1.11 |

**Data split 策略**：
- MVP 全程使用 `splits.json`（= fold 0），快速迭代，避免 5× 训练开销。
- 在 CASF-2016 test set 上报告最终指标。
- 只有在 MVP 通过 gate criterion 后、准备写论文或对比超参时，才切换到 `splits_5fold.json` 做 5-fold 评估（报告 Val 均值 ± std）。

**Gate criterion**: MVP must show $\Delta R^2 \geq +0.03$ AND $\Delta \rho \geq +0.04$ over baseline（baseline = SP3+SP2 无交互图）AND z_interaction must beat shuffled-edge control (p < 0.05). Only then proceed to Phase 2.

### Phase 2 — Extensions (conditional)

- 5-fold 稳健性评估（`splits_5fold.json`），报告 Val 指标均值 ± std
- Chemistry-aware edge features (interaction type indicators)
- Gated fusion (A1.7)
- Higher output dimensions (A1.11)
- Cutoff / layer-count sweeps (A1.8–A1.9)
- ESM-2 pocket features (replaces learned embeddings)

---

## 9. Implementation Checklist

> **注意**：本 checklist 假设 SP3 (token-level fusion) 和 SP2 (attention pooling) 已在各自的 Sub-Plan 中实现。

### 前置 (SP3 + SP2)
- [ ] 确认 SP3 `TokenLevelAttention` 模块已实现并通过测试
- [ ] 确认 SP2 `SelfAttentionPooling` / `CrossAttentionPooling` 模块已实现并通过测试
- [ ] 确认 `sampler.py` 已暴露 per-layer, per-atom hidden states

### 本 Sub-Plan (SP1: 交互级 + 全局级 + 融合)
- [ ] Implement `interaction_graph.py` with `InteractionGraphBuilder` (heavy-atom resolution, MVP edge features only; ligand node features = $\tilde{h}_i$)
- [ ] Implement `interaction_gnn.py` with `InteractionGNN`
- [ ] Implement `multi_granularity.py` with `MultiGranularityEncoder` (接收 $\tilde{h}_i$, $z_{\text{atom}}$, pocket data; output_dim=128, concat+MLP)
- [ ] Write `s09_build_interaction_graphs.py` pipeline script (从 $\tilde{h}_i$ 构建图)
- [ ] Update `gen_uncertainty.py` for variable-dim embeddings
- [ ] Update `fusion.py` for variable-dim Jacobians
- [ ] Write all unit tests (T1.1–T3.6)
- [ ] Write integration tests (T4.1–T4.4) — 覆盖完整 SP3→SP2→SP1 链

### 端到端验证
- [ ] Implement shuffled-edge graph builder for ablation A1.10
- [ ] Run MVP ablation experiments (A1.1–A1.6, A1.10) — 所有配置均使用 $\tilde{h}_i$ 作为输入
- [ ] **Decision gate**: evaluate MVP results against gate criteria
- [ ] (Phase 2) Add chemistry-aware edge features, gated fusion, extended ablations
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
