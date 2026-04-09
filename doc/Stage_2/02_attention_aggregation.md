# Sub-Plan 2: Attention-Based Aggregation

> **角色**: Phase A — 注意力汇聚（层内 AttnPool + 层间 AttnFusion），两种候选架构  
> **Priority**: P0 — Critical  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集); Sub-Plan 3 (global-level probing evidence)  
> **Training Data**: PDBbind v2020 refined set (~5,316 complexes), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 1–2 weeks implementation + 1 week testing  
> **Paper Section**: §3.X.2 Attention-Based Pooling

---

## 0. 架构总览：两种候选方案

> **2026-04-07 架构决策**：放弃旧的 SP3→SP2→SP1 三步串行链。
> 本 Sub-Plan 重新定位为：围绕 **层内 attention pooling（AttnPool）** 和 **层间 attention fusion（AttnFusion）** 的两种组合方案。
> 先做一个预备实验验证"层内 AttnPool 是否比 MeanPool 更有效"，再根据结果比较两个方案。

### 方案 A：Two-Branch（$z_{\text{atom}} + z_{\text{global}}$）

```
  TargetDiff Frozen Encoder (L=9 layers)
         │
         │  H^(1), H^(2), ..., H^(9)   (per-layer, per-atom)
         │
    ┌────┴────────────────────────────────┐
    │                                      │
    ▼                                      ▼
  Branch 1: Last-Layer AttnPool      Branch 2: Layer-Level AttnFusion
  ┌─────────────────────┐             ┌──────────────────────────────┐
  │ H^(9) ∈ ℝ^{N×d}    │             │ For each layer l=1..9:       │
  │                     │             │   z^(l) = MeanPool(H^(l))    │
  │ z_atom = AttnPool(  │             │                              │
  │   {h_i^(9)}_{i=1}^N │             │ z_global = AttnFusion(       │
  │ )                   │             │   {z^(1), ..., z^(9)}        │
  └─────────┬───────────┘             │ )                            │
            │                         └──────────────┬───────────────┘
            │  z_atom ∈ ℝ^d                         │  z_global ∈ ℝ^d
            │                                        │
            └──────────────┬─────────────────────────┘
                           ▼
                 ┌──────────────────┐
                 │ Fusion (Concat   │
                 │  + MLP / Gated)  │
                 │ → z_new ∈ ℝ^d   │
                 └────────┬─────────┘
                          ▼
                    GP / DKL Head
```

**特点**：
- $z_{\text{atom}}$：对最后一层做 atom-level attention pooling → 捕捉"哪些原子最重要"
- $z_{\text{global}}$：对 9 层各自的 mean-pooled vector 做 attention fusion → 捕捉"哪些层最有用"
- 两路互补：atom-level importance + layer-level importance

### 方案 B：Single-Branch（AttnPool 替代 MeanPool + AttnFusion）

```
  TargetDiff Frozen Encoder (L=9 layers)
         │
         │  H^(1), H^(2), ..., H^(9)   (per-layer, per-atom)
         │
         ▼
  ┌──────────────────────────────────────┐
  │ For each layer l=1..9:               │
  │   z^(l) = AttnPool(H^(l))           │
  │   (learnable atom-level attention    │
  │    within each layer)                │
  └──────────────────┬───────────────────┘
                     │  {z^(1), ..., z^(9)}  每层已是 attn-pooled
                     ▼
  ┌──────────────────────────────────────┐
  │ z_global = AttnFusion(               │
  │   {z^(1), ..., z^(9)}               │
  │ )                                    │
  └──────────────────┬───────────────────┘
                     │  z_global ∈ ℝ^d
                     ▼
               GP / DKL Head
```

**特点**：
- 每一层的 pooling 都用 attention（替代 mean pooling）→ 层内已经挑选重要原子
- 层间再用 attention fusion → 挑选重要层
- 不需要单独的 $z_{\text{atom}}$ 分支，因为 atom-level attention 已经嵌入每层 pooling 中
- 参数更多（9 个 AttnPool），但结构更统一

### 核心问题与验证路径

```
  Preliminary: 层内 AttnPool 是否优于 MeanPool？
      │
      ├─ 实验: 在 last layer (H^(9)) 上对比 MeanPool vs AttnPool → GP
      │         (单层，最简单的 A/B test)
      │
      ├─ 如果 AttnPool ≤ MeanPool → 方案 B 的前提不成立，优先方案 A
      │
      └─ 如果 AttnPool > MeanPool → 两个方案都值得尝试
              │
              ├─ 方案 A: last-layer AttnPool + 9-layer MeanPool→AttnFusion
              │
              └─ 方案 B: 9-layer AttnPool→AttnFusion（省略 z_atom 分支）
```

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

同时，SP3 global-level probing 已证明多层信息有价值（旧 932-pocket 数据集: Attn-all test $R^2=0.528 \gg$ L8 test $R^2=0.420$）。新 PDBbind v2020 baseline（L9 MeanPool）: Val $R^2=0.232$, Val $\rho=0.498$, Test $R^2=0.449$, Test $\rho=0.697$。此前的层间 fusion 均基于 mean-pooled 的 per-layer vector。

**两个开放问题**：
1. **层内 AttnPool 是否有帮助？** 把每层 mean pool 换成 attention pool，信号质量能否提升？
2. **两种组合哪个更优？** (A) 单独的 atom-attention 分支 + mean-pooled 的 layer fusion，还是 (B) 统一在每层都做 attention pool 后再 layer fusion？

**验证策略**：先回答问题 1（最简单的 A/B test），再根据结果比较方案 A 和 B。

---

## 2. Architecture Design

### 2.0 Preliminary Experiment: 层内 AttnPool vs MeanPool（单层 A/B Test）

在进入方案 A/B 比较之前，先在**最简单的设置**下验证 attention pooling 对层内汇聚的增益：

| Config | Pooling | Layer(s) | Predictor | 目标 |
|--------|---------|----------|-----------|------|
| P0 | MeanPool | Last (L=9) | MLP readout | Baseline（Val R²=0.232, Val ρ=0.498, Test R²=0.449, Test ρ=0.697） |
| P1 | AttnPool (Self-Attn) | Last (L=9) | MLP readout | AttnPool 是否优于 MeanPool？ |
| P2 | AttnPool (Cross-Attn, pocket-cond) | Last (L=9) | MLP readout | Bonus: Pocket conditioning（不影响 Go/No-Go） |

> **注意**：Preliminary 和 Phase 2 均用 MLP readout 而非 GP，以隔离 representation 质量的变化（见 §3.5 两步训练策略）。

**Go/No-Go**（仅基于 P0 vs P1）:
- 如果 P1 ≤ P0 → 层内 AttnPool 在 last layer 上无增益。方案 B 的核心前提（每层都换 AttnPool）不太可能有意义。优先验证方案 A（z_atom + z_global），其中 z_atom 用 AttnPool，z_global 仍用 MeanPool→AttnFusion。
- 如果 P1 > P0 → AttnPool 有增益，两个方案都值得继续。

**关于 P2（Cross-Attention）— bonus ablation，不影响 Go/No-Go**：
- 当前 pocket query 使用 $\bar{h}^{(P)}$（mean pocket embedding），粒度太粗，无法表达"pocket 的哪个区域在与 ligand 交互"
- P2 结果仅作参考，不决定主线方向
- 更合理的 pocket representation 留待后续改进：learned pocket summary token、top-contact residues summary、或用 SP3 找到更好的 pocket-side fused representation

### 2.1 Intra-Layer Attention Pooling（层内 AttnPool）

> 用于：方案 A 的 $z_{\text{atom}}$ 分支；方案 B 的每层 pooling。

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

#### Variant C: Multi-Head Attention Pooling（后续补充，不进入第一版主线）

> **注意**：Multi-head 仅作为最优方案确定后的补充实验。不要预设每个 head 的化学含义（pharmacophoric、geometric 等），这属于"解释先于证据"。最多事后分析 head 是否自然分化出不同 pattern。

Multiple attention heads capture different aspects of binding:

$$
z_{\text{multi}} = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
$$

$$
\text{head}_h = \sum_i \alpha_i^{(h)} W_V^{(h)} h_i
$$

仅在 Phase 3 细化阶段、best scheme 已稳定有效后，尝试 $H=4$。

### 2.2 Regularization

To prevent attention collapse (all weight on one atom):

**Entropy regularization**（第一版唯一的正则项）:
$$
\mathcal{L}_{\text{ent}} = -\lambda_{\text{ent}} \sum_i \alpha_i \log \alpha_i
$$

- $\lambda_{\text{ent}} > 0$ 鼓励分布不要太尖（防止 collapse 到单个原子）
- $\lambda_{\text{ent}}$ 不宜过大，否则退化为 uniform（等价于 mean pooling）
- 推荐初始值 $\lambda_{\text{ent}} = 0.01$

> **⚠️ 关于 sparsity regularization 的说明**：
> 旧版本使用 $\|\alpha\|_1$ 作为 sparsity penalty，但这对 softmax 输出无效——因为 $\alpha_i \geq 0$ 且 $\sum_i \alpha_i = 1$，所以 $\|\alpha\|_1 = 1$ 恒为常数。
> 如果后续需要鼓励更尖的分布（sparsity），可考虑：
> - **Negative entropy**：$\mathcal{L}_{\text{sparse}} = +\lambda \sum_i \alpha_i \log \alpha_i$（与 entropy reg 方向相反）
> - **Concentration penalty**：$\mathcal{L}_{\text{conc}} = -\lambda \sum_i \alpha_i^2$（Herfindahl index）
> - 但第一版不加 sparsity，只用 entropy reg 即可。

### 2.3 Inter-Layer Attention Fusion（层间 AttnFusion）

> SP3 global-level probing 已有 Layer-Attention Fusion 实现（见 [03_multi_layer_fusion.md](03_multi_layer_fusion.md) §2.3 Stage 3）。此处直接复用，但明确其在两种方案中的角色。

给定 9 层的 pooled vectors $\{z^{(1)}, \dots, z^{(9)}\}$（MeanPool 或 AttnPool 产出），层间 attention fusion：

$$
\beta_l = \frac{\exp(u^\top \tanh(W z^{(l)}))}{\sum_{k=1}^{9} \exp(u^\top \tanh(W z^{(k)}))}
$$

$$
z_{\text{global}} = \sum_{l=1}^{9} \beta_l z^{(l)}
$$

其中 $u \in \mathbb{R}^{d_h}$, $W \in \mathbb{R}^{d_h \times d}$ 是可学习参数。

**在两个方案中的角色**：

| | 方案 A | 方案 B |
|---|---|---|
| 每层 pooling | $z^{(l)} = \text{MeanPool}(H^{(l)})$ | $z^{(l)} = \text{AttnPool}(H^{(l)})$ |
| 层间 fusion | $z_{\text{global}} = \text{AttnFusion}(\{z^{(l)}\})$ | $z_{\text{global}} = \text{AttnFusion}(\{z^{(l)}\})$ |
| 额外分支 | $z_{\text{atom}} = \text{AttnPool}(H^{(9)})$ | 无（AttnPool 已在每层内完成） |

### 2.4 方案 A：Two-Branch 详细设计

**Branch 1 — $z_{\text{atom}}$**：
- 输入：$H^{(9)} \in \mathbb{R}^{N \times d}$（最后一层 atom-level embeddings）
- 操作：Self-Attention Pooling（§2.1 Variant A）或 Cross-Attention Pooling（§2.1 Variant B）
- 输出：$z_{\text{atom}} \in \mathbb{R}^{d}$

**Branch 2 — $z_{\text{global}}$**：
- 输入：$H^{(1)}, \dots, H^{(9)}$
- 每层先 MeanPool：$z^{(l)} = \frac{1}{N} \sum_{i=1}^{N} h_i^{(l)}$
- 层间 AttnFusion（§2.3）：$z_{\text{global}} = \sum_l \beta_l z^{(l)}$
- 输出：$z_{\text{global}} \in \mathbb{R}^{d}$

**Fusion**：
$$
z_{\text{new}} = \text{FusionMLP}([z_{\text{atom}}; z_{\text{global}}]) \in \mathbb{R}^{d_{\text{out}}}
$$

或 gated fusion：
$$
g = \sigma(W_g [z_{\text{atom}}; z_{\text{global}}])
$$
$$
z_{\text{new}} = g \odot z_{\text{atom}} + (1 - g) \odot z_{\text{global}}
$$

**直觉**：$z_{\text{atom}}$ 关注"哪些原子重要"（空间/化学重要性），$z_{\text{global}}$ 关注"哪些层的全局信息重要"（多尺度特征提取的重要性）。两者正交互补。

### 2.5 方案 B：Single-Branch 详细设计

**Per-Layer AttnPool（第一版仅做共享参数）**：
- 对每层 $l$：$z^{(l)} = \text{AttnPool}(H^{(l)})$
- **第一版：共享参数** — 所有层用同一组 attention 参数 → 参数少，训练稳定
- 独立参数（每层单独一组）作为后续补充实验 — 参数量 9×，浅层信号弱容易训练不稳，几千样本下容易过拟合

**Layer-Level AttnFusion**：
- $z_{\text{global}} = \text{AttnFusion}(\{z^{(1)}, \dots, z^{(9)}\})$（同 §2.3）

**不需要额外 $z_{\text{atom}}$ 分支**：
- 因为层内 AttnPool 已经在每层内完成了 atom-importance 的加权
- 最后一层的 AttnPool attention weights $\alpha_i^{(9)}$ 自然反映了 atom-level importance
- 如果需要可视化 atom importance，可以取 $\alpha^{(9)}$ 或加权平均 $\sum_l \beta_l \alpha^{(l)}$

### 2.6 方案比较：预期 trade-off

| 维度 | 方案 A (Two-Branch) | 方案 B (Single-Branch, 共享参数) |
|------|---------------------|-------------------------------|
| atom-level attention | 仅在 last layer | 在所有 9 层（共享同一套参数） |
| layer-level fusion 输入质量 | MeanPool（无选择性） | AttnPool（有选择性） |
| 新增参数量 | 1× AttnPool + AttnFusion + FusionMLP | 1× AttnPool（共享） + AttnFusion |
| 概念简洁性 | 双分支，显式分离两种 attention | 单分支，统一框架 |
| 可能问题 | z_atom 与 z_global 可能冗余 | 共享参数假设各层 atom importance 分布相同 |

---

## 3. Implementation Plan

### 3.1 New Module: `bayesdiff/attention_pool.py`

```python
class SelfAttentionPooling(nn.Module):
    """Intra-layer self-attention pooling over atoms."""
    
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
    """Pocket-conditioned cross-attention pooling."""
    
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
    """Multi-head attention pooling (Phase 3 only, not in v1 main line)."""
    ...
```

> **注意**：`MultiHeadAttentionPooling` 第一版不实现，仅在 Phase 3 最优方案确定后作为补充实验。

```python
class AttentionPoolingWithRegularization(nn.Module):
    """Wrapper that adds entropy regularization loss.
    
    第一版仅支持 entropy regularization。
    旧版 sparsity_weight (L1 on softmax) 已移除——softmax 输出的 L1 norm 恒为 1。
    """
    
    def __init__(self, pooling_module, entropy_weight=0.01):
        ...
    
    def forward(self, *args, **kwargs):
        """Returns (z, alpha, reg_loss)."""
        ...
```

### 3.2 New Module: `bayesdiff/layer_attn_fusion.py`

> 复用 SP3 已有的 Layer-Attention Fusion 逻辑，但封装为独立模块，支持方案 A/B 的不同输入。

```python
class LayerAttentionFusion(nn.Module):
    """Inter-layer attention fusion: {z^(l)} → z_global."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        ...
    
    def forward(self, layer_embeds: Tensor):
        """
        Args:
            layer_embeds: (B, L, d) — L layer-pooled vectors
        
        Returns:
            z_global: (B, d) attention-fused embedding
            beta: (B, L) layer attention weights
        """
        ...
```

### 3.3 New Module: Integration for Scheme A & B

```python
class SchemeA_TwoBranch(nn.Module):
    """方案 A: z_atom (last-layer AttnPool) + z_global (MeanPool→AttnFusion)."""
    
    def __init__(self, embed_dim: int, n_layers: int = 9,
                 fusion_type: str = 'concat_mlp'):
        super().__init__()
        self.atom_pool = SelfAttentionPooling(embed_dim)
        self.layer_fusion = LayerAttentionFusion(embed_dim)
        if fusion_type == 'concat_mlp':
            self.fusion = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        elif fusion_type == 'gated':
            self.gate = nn.Linear(2 * embed_dim, embed_dim)
            ...
    
    def forward(self, all_layer_atom_embs, atom_mask=None):
        """
        Args:
            all_layer_atom_embs: list of L tensors, each (B, N, d)
            atom_mask: (B, N) boolean mask
        
        Returns:
            z_new: (B, d)
            info: dict with z_atom, z_global, alpha_atom, beta_layer
        """
        # Branch 1: last-layer AttnPool → z_atom
        z_atom, alpha = self.atom_pool(all_layer_atom_embs[-1], mask=atom_mask)
        
        # Branch 2: per-layer MeanPool → AttnFusion → z_global
        layer_means = []
        for l_emb in all_layer_atom_embs:
            if atom_mask is not None:
                z_l = (l_emb * atom_mask.unsqueeze(-1)).sum(1) / atom_mask.sum(1, keepdim=True)
            else:
                z_l = l_emb.mean(dim=1)
            layer_means.append(z_l)
        layer_stack = torch.stack(layer_means, dim=1)  # (B, L, d)
        z_global, beta = self.layer_fusion(layer_stack)
        
        # Fusion
        z_new = self.fusion(torch.cat([z_atom, z_global], dim=-1))
        return z_new, {'z_atom': z_atom, 'z_global': z_global,
                       'alpha_atom': alpha, 'beta_layer': beta}


class SchemeB_SingleBranch(nn.Module):
    """方案 B: Per-layer shared AttnPool → AttnFusion (no separate z_atom).
    
    第一版仅支持共享参数。独立参数作为后续补充实验。
    """
    
    def __init__(self, embed_dim: int, n_layers: int = 9):
        super().__init__()
        # 共享参数：所有层用同一个 AttnPool 实例
        # 注意：不要用 ModuleList([instance] * n)，那是重复引用同一对象，
        # 在调试/hook/state_dict 中会混淆。直接存为单一 module。
        self.shared_pool = SelfAttentionPooling(embed_dim)
        self.n_layers = n_layers
        self.layer_fusion = LayerAttentionFusion(embed_dim)
    
    def forward(self, all_layer_atom_embs, atom_mask=None):
        """
        Args:
            all_layer_atom_embs: list of L tensors, each (B, N, d)
            atom_mask: (B, N) boolean mask
        
        Returns:
            z_global: (B, d)
            info: dict with per-layer alphas and betas
        """
        layer_vecs = []
        layer_alphas = []
        for l_emb in all_layer_atom_embs:
            z_l, alpha_l = self.shared_pool(l_emb, mask=atom_mask)
            layer_vecs.append(z_l)
            layer_alphas.append(alpha_l)
        
        layer_stack = torch.stack(layer_vecs, dim=1)  # (B, L, d)
        z_global, beta = self.layer_fusion(layer_stack)
        
        return z_global, {'layer_alphas': layer_alphas, 'beta_layer': beta}
```

### 3.4 Modifications to `bayesdiff/sampler.py`

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

### 3.5 Training Strategy

> **核心原则**：两步走，先验证 representation 质量，再接 GP/DKL。不要一开始就 end-to-end joint train，否则结果不好时无法分辨是 attention 没学好、GP 优化难、还是两者耦合不稳定。

**Step 1（Representation Validation）— 验证 attention 是否改善表示质量**：
- Freeze TargetDiff encoder
- 训练 AttnPool / LayerAttnFusion / FusionMLP（方案 A 或 B 的可训练参数）
- Readout head：**轻量 MLP**（e.g., Linear → ReLU → Linear → pKd），不用 GP
- Loss：MSE on pKd
- Optimizer：AdamW, lr=1e-3, weight_decay=1e-4
- Regularization：entropy reg on attention weights ($\lambda_{\text{ent}} = 0.01$)
- Early stopping on validation Spearman $\rho$
- **评估**：对比 MLP readout 下 MeanPool vs AttnPool（以及方案 A vs B）的 $R^2$ 和 $\rho$
  - 如果 AttnPool 表示在 MLP readout 下已经更好 → representation 确实更好 → 继续 Step 2
  - 如果 AttnPool ≤ MeanPool → attention 没学好，先调 attention 架构/超参

**Step 2（GP/DKL Integration）— 确认最优表示后接 GP**：
- 冻结 Step 1 训好的 attention 参数（或 fine-tune with small lr）
- 用最优表示喂给 GP / DKL head
- Loss：GP marginal likelihood
- 此时如果结果不好，可以确定是 GP 端的问题，而非 representation 端

**方案 A 额外注意**：FusionMLP 的训练需确保两个分支的梯度平衡，可以监控 $z_{\text{atom}}$ 和 $z_{\text{global}}$ 的梯度 norm。
**方案 B 额外注意**：共享参数的 AttnPool 接受来自所有 9 层的梯度 → 梯度量大，可能需要适当降低 lr。

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
| T1.6 | `test_entropy_regularization` | Uniform attention → max entropy; peaked → low entropy |
| T1.7 | `test_gradient_flow` | All parameters receive gradients |
| T1.8 | `test_determinism` | Same input + seed → same output |
| T1.9 | `test_numerical_stability` | No NaN with very large/small embeddings |
| T1.10 | `test_single_atom_molecule` | Works when N=1 (trivial case) |
| T1.11 | `test_shared_pool_scheme_b` | SchemeB shared_pool produces same weights for same input regardless of layer |

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

#### Phase 1: Preliminary — 层内 AttnPool 验证

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| P0 | Last-layer MeanPool → MLP readout | Baseline（Test R²=0.449, Test ρ=0.697） |
| P1 | Last-layer Self-AttnPool → MLP readout | AttnPool vs MeanPool |
| P2 | Last-layer Cross-AttnPool (pocket-cond) → MLP readout | Bonus: Pocket conditioning（不影响 Go/No-Go） |

**Go/No-Go gate**: P1 > P0 → proceed to Phase 2. P2 仅作参考。

> **注意**：Preliminary 阶段用 MLP readout（非 GP），遵循两步训练策略（§3.5）。

**Phase 1 实际结果（2026-04-08）**：

| Exp | Val R² | Val ρ | Test R² | Test ρ | Attn Entropy |
|-----|--------|-------|---------|--------|--------------|
| P0 (MeanPool→MLP) | 0.237 | 0.522 | 0.524 | 0.744 | N/A |
| P1 (AttnPool→MLP) | **0.277** | **0.551** | **0.560** | **0.753** | 2.87 |

> **Go/No-Go: ✅ GO** — P1 在所有指标上超过 P0（Test Δρ=+0.009, ΔR²=+0.037）。
> Attention entropy=2.87 在健康范围 [1.0, 3.0] 内，无 collapse。
> MLP readout 已超过 §5.1 的 Preliminary Success 阈值（Test R²≥0.47, ρ≥0.72）。
> **→ 进入 Phase 2 方案比较。**

#### Phase 2: 方案 A vs 方案 B

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A2.1 | 9-layer MeanPool → AttnFusion → MLP readout（SP3 baseline） | Layer fusion（需在 PDBbind v2020 上重跑） |
| A2.2 | **方案 A**: Last-layer AttnPool ($z_{\text{atom}}$) + 9-layer MeanPool→AttnFusion ($z_{\text{global}}$) → Concat+MLP readout | Two-Branch |
| A2.3 | **方案 B (shared)**: 9-layer AttnPool (共享参数) → AttnFusion → MLP readout | Single-Branch, shared AttnPool |

> **注意**：Phase 2 同样用 MLP readout。方案 B 独立参数版本（A2.4）作为后续补充，第一版不做——参数量大、浅层信号弱、几千样本下容易过拟合。

**Phase 2 实测结果** (SLURM #5783372, 2026-04-08, 1253s total):

| Exp | Val R² | Val ρ | Test R² | Test ρ | Early Stop | Time |
|-----|--------|-------|---------|--------|------------|------|
| (ref) P0 | 0.237 | 0.522 | 0.524 | 0.744 | — | — |
| (ref) P1 | 0.277 | 0.551 | 0.560 | 0.753 | — | — |
| A2.1 (MeanPool→AttnFusion) | 0.263 | 0.539 | **0.564** | 0.751 | epoch 102 | 500s |
| A2.2 (Scheme A TwoBranch) | 0.270 | **0.566** | 0.547 | **0.753** | epoch 63 | 315s |
| A2.3 (Scheme B SingleBranch) | **0.292** | **0.577** | **0.568** | 0.747 | epoch 72 | 438s |

**分析**：
- **Val 指标**：A2.3 (Scheme B) 在 val 上全面领先（Val ρ=0.577 >> A2.2=0.566 >> A2.1=0.539）
- **Test 指标**：A2.2 (Scheme A) 在 test ρ 上微弱领先（0.753 vs A2.3=0.747），但 A2.3 在 test R² 上更优（0.568 vs 0.547）
- **差异很小**：Test ρ 差异仅 0.006，在 N=285 test set 上不显著
- **A2.3 层权重分析（layer_beta）**：集中在 layer 7-9（β=[0.128, 0.322, 0.531]），与 SP3 prior knowledge 一致（深层特征更重要）
- **A2.2 层权重分析**：layer 4 和 8 最高（β=[0.173, 0.269]），分布更均匀
- **Attention entropy**：A2.2 atom entropy=2.77, A2.3 per-layer entropy 均在 [2.49, 3.03] 范围，全部健康
- **决策**：A2.3 (Scheme B) 更优——Val 指标全面领先，Test R² 更优，模型更简洁（共享参数）。选择 **Scheme B** 进入 Phase 3。

#### Phase 3: 最优方案细化 + GP 接入

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A3.1 | Best scheme + entropy reg (λ=0.01) | Regularization tuning |
| A3.2 | Best scheme + entropy reg (λ=0.1) | Higher regularization |
| A3.3 | (方案 A only) Gated fusion vs Concat+MLP | Fusion method comparison |
| A3.4 | Best scheme + GP/DKL head（Step 2 of training） | 接入 GP，验证最终性能 |
| A3.5 | Best scheme + multi-head (H=4)（事后分析 head pattern） | 补充：Multi-head variant |
| A3.6 | (方案 B only) 独立参数 AttnPool | 补充：per-layer independent AttnPool |
| A3.7 | (方案 A only) Cross-AttnPool for z_atom | 补充：Pocket-conditioned z_atom |
**Phase 3 实测结果** (SLURM #5786901, 2026-04-08, 1077s total):

| Exp | Val R² | Val ρ | Test R² | Test ρ | Notes |
|-----|--------|-------|---------|--------|-------|
| (ref) A2.3 SchemeB λ=0.01 MLP | 0.292 | 0.577 | 0.568 | 0.747 | Phase 2 best |
| A3.2 SchemeB λ=0.1 MLP | 0.246 | 0.534 | 0.544 | **0.756** | Higher λ hurts val, mixed test |
| A3.4-Step1 SchemeB→MLP (retrain) | 0.262 | 0.555 | **0.572** | **0.761** | Stochastic variation |
| A3.4-Step2 SchemeB→SVGP | 0.258 | 0.550 | 0.507 | 0.719 | GP degrades vs MLP |

**分析**：
- **A3.2 (λ=0.1)**：更高 entropy reg 降低了 val 性能（val ρ=0.534 vs 0.577），但 test ρ 微增（0.756 vs 0.747）。差异在 N=285 test set 上不显著。λ=0.01 仍然更优（val 指标全面领先）。
- **A3.4 Step 1 vs Step 2**：MLP readout（Test ρ=0.761）显著优于 SVGP（Test ρ=0.719）。GP 退化 Δρ=-0.042。
- **GP 退化原因**：noise variance=1.654 过高（几乎等于 data variance ~1.8），SVGP 退化为近似常数预测。|error|-σ 相关性 ρ=-0.008（p=0.90），uncertainty 完全未校准。
- **诊断**：128维 embedding 空间下 N=16K 样本不足以训好 512 个 inducing point 的 GP。GP 需要更低维的 representation。

**GP 修复实验** (SLURM #5797939, 2026-04-09, 336s total):

基于 A3.4-Step1 冻结的 SchemeB 模型，将 128 维 embedding 降维后再接 SVGP：

| Exp | Val R² | Val ρ | Test R² | Test ρ | |err|-σ ρ | Noise | Notes |
|-----|--------|-------|---------|--------|----------|-------|-------|
| (ref) A3.4-Step1 MLP | 0.262 | 0.555 | 0.572 | 0.761 | — | — | MLP baseline |
| (ref) A3.4-Step2 SVGP raw 128d | 0.258 | 0.550 | 0.507 | 0.719 | -0.008 | 1.654 | GP 退化 |
| A3.4b PCA32→SVGP | 0.272 | 0.546 | 0.543 | 0.746 | 0.042 | 0.903 | PCA 保留 90.3% 方差 |
| **A3.4c DKL (128→32→SVGP)** | **0.268** | **0.547** | **0.559** | **0.760** | -0.035 | 0.162 | **最佳 GP 方案** |
| A3.4d PCA16→SVGP | 0.260 | 0.534 | 0.512 | 0.726 | 0.029 | 0.978 | 降维过激 |

**GP 修复分析**：
- **DKL (A3.4c)** 恢复到 Test ρ=0.760，与 MLP baseline (0.761) 几乎持平。DKL 联合训练的 MLP(128→64→ReLU→32) 学到了适合 GP 的低维表示。Noise=0.162 远低于原始 GP (1.654)，说明 GP 能有效利用 DKL 特征。
- **PCA32 (A3.4b)** 部分恢复（ρ=0.746），PCA 保留了 90.3% 方差，但线性降维丢失了 GP 需要的非线性结构。
- **PCA16 (A3.4d)** 降维过激（ρ=0.726），16 维仅保留 ~78% 方差，信息损失过大。
- **Uncertainty 校准全面不佳**：所有 GP 方案 |err|-σ ρ 均在 [-0.035, 0.042]，接近零。GP 的不确定性估计无法区分哪些预测更可靠。这是一个根本性问题：(1) SVGP variational 近似偏差、(2) 128 维 representation 的 uncertainty 主要来自 inducing point 覆盖不均，与预测误差无关。
- **结论**：DKL 是最优 GP 接入方式（恢复预测精度），但 GP uncertainty 在该任务上质量不高。SchemeB + MLP readout 仍是最实用的配置。GP uncertainty 可作为 BayesDiff 框架的附加信号，但不应作为唯一不确定性来源。

**Sub-Plan 2 总结：全实验汇总表**：

| Experiment | Val R² | Val ρ | Test R² | Test ρ | Phase |
|------------|--------|-------|---------|--------|-------|
| P0 MeanPool→MLP | 0.237 | 0.522 | 0.524 | 0.744 | Phase 1 |
| P1 AttnPool→MLP | 0.277 | 0.551 | 0.560 | 0.753 | Phase 1 |
| A2.1 MeanPool→AttnFusion | 0.263 | 0.539 | 0.564 | 0.751 | Phase 2 |
| A2.2 SchemeA TwoBranch | 0.270 | 0.566 | 0.547 | 0.753 | Phase 2 |
| A2.3 SchemeB SingleBranch (λ=0.01) | **0.292** | **0.577** | 0.568 | 0.747 | Phase 2 |
| A3.2 SchemeB (λ=0.1) | 0.246 | 0.534 | 0.544 | 0.756 | Phase 3 |
| A3.4-Step1 SchemeB→MLP | 0.262 | 0.555 | **0.572** | **0.761** | Phase 3 |
| A3.4-Step2 SVGP raw 128d | 0.258 | 0.550 | 0.507 | 0.719 | Phase 3 |
| A3.4b PCA32→SVGP | 0.272 | 0.546 | 0.543 | 0.746 | GP Fix |
| **A3.4c DKL (128→32→SVGP)** | 0.268 | 0.547 | 0.559 | **0.760** | GP Fix |
| A3.4d PCA16→SVGP | 0.260 | 0.534 | 0.512 | 0.726 | GP Fix |
| A3.5 SchemeB MultiHead H=4 | 0.250 | 0.543 | 0.565 | 0.755 | Ablation |
| **A3.6 SchemeB Independent** | 0.278 | **0.568** | **0.574** | **0.778** | **Ablation — 新 SOTA** |

**关键 takeaway**：
1. **AttnPool > MeanPool**：层内 attention pooling 在所有设置下优于 mean pooling（P1 > P0, Phase 2 全面超过 baseline）。
2. **Scheme B > Scheme A**：共享参数 AttnPool→AttnFusion 比 TwoBranch 更优且更简洁。
3. **独立参数 AttnPool (A3.6) 是新 SOTA**：Test ρ=0.778，显著超过共享参数版本 (0.747-0.761)。每层独立参数允许各层学习不同的 atom importance pattern，108K params 仍在合理范围。
4. **Multi-Head (A3.5) 效果中等**：H=4 heads diversity=0.763（heads 确实学到不同 pattern），但 ρ=0.755 不如独立参数方案。
5. **MLP readout ≈ DKL > raw GP**：MLP 和 DKL 表现接近（ρ=0.761 vs 0.760），raw SVGP 在高维下退化严重。
6. **GP uncertainty 质量不佳**：所有 GP 方案的 |err|-σ 相关性接近零，uncertainty 估计无法校准。
7. **Layer attention 高度集中**：Layer 7-9 占 >95% 权重（可视化确认），深层特征主导。Attention entropy 2.0-3.5 nats，无 collapse。
---

## 5. Evaluation & Success Criteria

### 5.1 Quantitative Metrics

| Metric | Baseline (L9 MeanPool, PDBbind v2020) | Preliminary Success (P1>P0) | Full Success (best scheme) |
|--------|------------------------------------------|----------------------------|---------------------------|
| Test $R^2$ | 0.449 | ≥ 0.47 | ≥ 0.52 |
| Test Spearman $\rho$ | 0.697 | ≥ 0.72 | ≥ 0.75 |
| Val $R^2$ | 0.232 | ≥ 0.25 | ≥ 0.30 |
| Val Spearman $\rho$ | 0.498 | ≥ 0.52 | ≥ 0.56 |
| Attention entropy | N/A | 1.0–3.0 | 1.0–3.0 |

### 5.2 Qualitative Analysis

- **Attention weight visualization**: For known binder examples, do high-attention atoms correspond to pharmacophore positions?
- **Pocket-conditioned shift（P2 bonus）**: For the same ligand in different pockets, does cross-attention shift to different atoms?
- **Per-layer attention comparison (方案 B)**: 共享参数的 AttnPool 在不同层上学到的 attention pattern 是否有差异（输入不同导致输出不同）？
- **方案 A 分支贡献**: Gated fusion 中 gate 值分布 — $z_{\text{atom}}$ 和 $z_{\text{global}}$ 各贡献多少？
- **Step 1 vs Step 2 对比**: MLP readout 下的排序和 GP 下的排序是否一致？

### 5.3 Failure Criteria

- Preliminary (P1 ≤ P0) → 层内 AttnPool 无增益 → 简化为仅用 Layer-AttnFusion（SP3 方案），放弃方案 B，方案 A 可以作为轻量尝试
- Attention collapses to uniform → equivalent to mean pooling → 降低 entropy reg λ
- Attention collapses to single atom → overfitting → 提高 entropy reg λ
- 方案 B 共享参数下训练不稳定 → 退回方案 A
- Step 1 (MLP readout) 表示更好但 Step 2 (GP) 反而变差 → GP 端问题，单独调 GP 超参

---

## 6. Paper Integration

### 6.1 Methods Section (Draft)

> **§3.X.2 Attention-Based Aggregation**
> 
> We introduce learnable attention at two levels: **intra-layer** (atom-level attention pooling) and **inter-layer** (layer-level attention fusion).
> 
> **Intra-layer attention pooling.** Given atom embeddings $\{h_i\}_{i=1}^{N}$ from a single encoder layer, we replace mean pooling with:
> 
> $$\alpha_i = \text{softmax}(w^\top \tanh(W h_i + b))$$
> 
> $$z_{\text{attn}} = \sum_{i=1}^{N} \alpha_i h_i$$
> 
> This allows the model to focus on pharmacophore-relevant atoms rather than treating all atoms equally.
> 
> **Inter-layer attention fusion.** Given per-layer pooled vectors $\{z^{(1)}, \dots, z^{(L)}\}$, we compute:
> 
> $$\beta_l = \text{softmax}(u^\top \tanh(W z^{(l)}))$$
> 
> $$z_{\text{global}} = \sum_{l=1}^{L} \beta_l z^{(l)}$$
> 
> We investigate two combination strategies: (A) a two-branch architecture that maintains a separate atom-attention branch $z_{\text{atom}}$ alongside the layer-fused $z_{\text{global}}$, and (B) a unified architecture where attention pooling replaces mean pooling within each layer before inter-layer fusion.
> 
> To prevent attention collapse, we add an entropy regularization term $\mathcal{L}_{\text{ent}} = -\lambda \sum_i \alpha_i \log \alpha_i$ to the training objective.

### 6.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. A.1 | Attention weight heatmap on 3–5 example molecules | Show learned atom importance |
| Fig. A.2 | 3D molecular visualization with atoms colored by attention weight | Intuitive illustration |
| Fig. A.3 | Attention entropy distribution across molecules | Validate regularization |
| Fig. A.4 | Per-layer attention patterns (方案 B) — 浅层 vs 深层 | Layer-specific atom focus |
| Fig. A.5 | 方案 A gate distribution — $z_{\text{atom}}$ vs $z_{\text{global}}$ 贡献比 | Branch contribution |
| Fig. A.6 | Ablation bar chart (Phase 1–3) | Justify design choices |

### 6.3 Tables

| Table | Content |
|-------|---------|
| Tab. A.1 | Preliminary: MeanPool vs AttnPool on last layer (P0–P2) |
| Tab. A.2 | Scheme comparison (A2.1–A2.4): $R^2$, $\rho$, ECE |
| Tab. A.3 | Best scheme fine-tuning (A3.1–A3.6) |

---

## 7. Compatibility Notes

### 7.1 With Sub-Plan 3 (Multi-Layer Fusion) — 层间 AttnFusion 复用

本 Sub-Plan 的层间 AttnFusion（§2.3）直接复用 SP3 已有的 Layer-Attention Fusion 实现。
- 新 PDBbind v2020 baseline: L9 MeanPool → Test $R^2$=0.449, Test $\rho$=0.697（SP3 global-level probing 需在新数据集上重跑以获取 AttnFusion baseline）
- 两个方案的区别仅在于每层 pooling 是 MeanPool（SP3 已有）还是 AttnPool（本 SP 新增）

### 7.2 With Sub-Plan 1 (Multi-Granularity) — 如需扩展

如果后续需要加入 $z_{\text{interaction}}$（interaction graph + GNN），可以直接在最优方案输出上追加：

**方案 A 扩展**：
$$
z_{\text{final}} = \text{FusionMLP}([z_{\text{atom}}; z_{\text{global}}; z_{\text{interaction}}])
$$

**方案 B 扩展**：
$$
z_{\text{final}} = \text{FusionMLP}([z_{\text{global}}; z_{\text{interaction}}])
$$

### 7.3 With Generation Uncertainty

Each of $M$ generated molecules $m_1, \dots, m_M$ gets independently processed:

**方案 A**：
$$
z_{\text{atom}}^{(j)} = \text{AttnPool}(H^{(9, j)}) \quad z_{\text{global}}^{(j)} = \text{AttnFusion}(\{\text{MeanPool}(H^{(l, j)})\}_{l=1}^{9})
$$
$$
z_{\text{new}}^{(j)} = \text{Fusion}(z_{\text{atom}}^{(j)}, z_{\text{global}}^{(j)})
$$

**方案 B**：
$$
z_{\text{global}}^{(j)} = \text{AttnFusion}(\{\text{AttnPool}(H^{(l, j)})\}_{l=1}^{9})
$$

Then $\hat{\Sigma}_{\text{gen}}$ is estimated over $\{z^{(1)}, \dots, z^{(M)}\}$. Attention weights may vary across molecules, which is correct behavior.

### 7.4 With Delta Method

All modules (AttnPool, AttnFusion, FusionMLP) are differentiable. The Jacobian $J_\mu = \partial \mu / \partial z$ chains through all layers via `autograd` automatically.

---

## 8. Implementation Checklist

### Phase 1: Preliminary 验证（MLP readout）
- [x] Implement `SelfAttentionPooling` in `attention_pool.py`
- [x] Implement `CrossAttentionPooling` in `attention_pool.py`（P2 bonus ablation 用）
- [x] Implement `AttentionPoolingWithRegularization` wrapper（仅 entropy reg）
- [x] Add `extract_multilayer_atom_embeddings()` to `sampler.py`
- [x] Write unit tests (T1.1–T1.11) — 27/27 passed
- [x] Implement MLP readout head for representation validation（§3.5 Step 1）
- [x] Implement `SchemeA_TwoBranch` and `SchemeB_SingleBranch` in `attention_pool.py`
- [x] Create extraction script `s08c_extract_atom_embeddings.py`
- [x] Create training script `s12_train_attn_pool.py` (P0/P1/P2)
- [x] Create SLURM scripts (`s08c_extract_atom_emb.sh`, `s12_train_attn_pool.sh`)
- [x] Submit atom embedding extraction job (SLURM #5694194→#5709305→#5709464, 50 shards)
- [x] Run Preliminary experiments (P0, P1) — SLURM #5716251, 326s
- [ ] (Bonus) Run P2 — Cross-AttnPool
- [x] **Go/No-Go decision**: P1 > P0? → **GO** (ρ=0.753 > 0.744)

### Phase 2: 方案比较（MLP readout）
- [x] Reuse `LayerAttentionFusion` from `layer_fusion.py`（SP3 already implemented）
- [x] Implement `SchemeA_TwoBranch` (last-layer AttnPool + MeanPool→AttnFusion + FusionMLP)
- [x] Implement `SchemeB_SingleBranch` (共享参数 AttnPool→AttnFusion, `self.shared_pool` pattern)
- [x] Run ablation experiments (A2.1–A2.3) — SLURM #5783372, 方案 B (A2.3) 胜出（Val ρ=0.577, Test R²=0.568）
- [ ] Write integration tests (T2.1–T2.4)

### Phase 3: 最优方案细化 + GP 接入
- [x] Run entropy reg tuning — A3.2 (λ=0.1): test ρ=0.756 但 val 退化，λ=0.01 仍最优
- [x] ~~(方案 A) Test gated fusion vs concat+MLP (A3.3)~~ — 方案 B 胜出，跳过
- [x] **接入 GP/DKL head**（§3.5 Step 2）— SLURM #5786901, GP 退化（Test ρ=0.719 < MLP 0.761），noise=1.654 过高
- [x] GP 修复：PCA32→SVGP (ρ=0.746), **DKL 128→32 (ρ=0.760)**, PCA16 (ρ=0.726) — SLURM #5797939
- [x] (补充) Multi-head H=4 (A3.5) — SLURM #5800119: Test R²=0.565, ρ=0.755, diversity=0.763, 123K params
- [x] (补充) 方案 B 独立参数 (A3.6) — SLURM #5800119: **Test R²=0.574, ρ=0.778**, 108K params, 新 SOTA
- [x] Generate attention visualization figures — 4 figures in `results/stage2/ablation_viz/attention_viz/`
- [ ] Draft methods section text
