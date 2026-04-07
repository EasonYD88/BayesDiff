# Stage 2 Implementation Roadmap

## Overview

This document provides the master plan for BayesDiff Stage 2: upgrading the system from a **calibrated uncertainty evaluator** (Stage 1) to a **high-performance uncertainty-aware molecular ranking system** with significantly improved representation and prediction quality.

Stage 2 addresses the three core bottlenecks identified in [problem_and_solution.md](problem_and_solution.md):

1. **Representation bottleneck** — current mean-pooled SE(3) embeddings capture only ~12% of variance
2. **Information compression** — mean pooling destroys atom-level and interaction-level signals
3. **System positioning** — BayesDiff should be a *ranking/prioritization* system, not a pure regression system

---

## Sub-Plan Index

> **架构决策 (2026-04-06)**：Sub-Plans 1–3 不再是独立并列的模块，而是一条**串行链**：
> 
> **3 → 2 → 1**（多层提纯 → 原子级注意力汇总 → 交互级+全局级融合）
> 
> Sub-Plan 3 负责"哪几层的表示最有用"，输出 token-level 融合后的 $\tilde{h}_i$；
> Sub-Plan 2 不再是独立方法，而是在 $\tilde{h}_i$ 上做 attention pooling 得到 $z_{\text{atom}}$；
> Sub-Plan 1 是主干框架，把 atom / interaction / global 三种信息统一起来。

| # | Sub-Plan | File | 角色 | Priority | Dependency |
|---|----------|------|------|----------|------------|
| **0** | **PDBbind v2020 R1 + CASF-2016 数据集准备** | [00a_supervised_pretraining.md](00a_supervised_pretraining.md) | 数据前置 | **P0 — Critical** | None |
| 3 | Multi-Layer Fusion (Token-Level) | [03_multi_layer_fusion.md](03_multi_layer_fusion.md) | **Step 1**: 多层表示提纯 → $\tilde{h}_i$ | **P0 — Critical** | Sub-Plan 0 |
| 2 | Attention-Based Aggregation | [02_attention_aggregation.md](02_attention_aggregation.md) | **Step 2**: $z_{\text{atom}}$ 实现 | **P0 — Critical** | Sub-Plans 0, 3 |
| 1 | Multi-Granularity Representation | [01_multi_granularity_repr.md](01_multi_granularity_repr.md) | **Step 3**: 主干整合框架 | **P0 — Critical** | Sub-Plans 0, 3, 2 |
| 4 | Hybrid Predictor (DKL) | [04_hybrid_predictor.md](04_hybrid_predictor.md) | 预测器升级 | **P1 — High** | Sub-Plans 0–3 |
| 5 | Multi-Task Learning | [05_multi_task_learning.md](05_multi_task_learning.md) | 多任务学习 | **P2 — Medium** | Sub-Plans 0–3 |
| 6 | Physics-Aware Features | [06_physics_aware_features.md](06_physics_aware_features.md) | 物理特征增强 | **P2 — Medium** | Sub-Plan 0 |
| 7 | Uncertainty-Guided Generation | [07_uncertainty_guided_generation.md](07_uncertainty_guided_generation.md) | 闭环生成 | **P3 — Future** | Sub-Plans 0–6 |

---

## Implementation Phases

### Phase 0: PDBbind v2020 R1 + CASF-2016 数据集准备 (Sub-Plan 0)

**Goal**: 整理 PDBbind v2020 R1 general set (19,037 P-L complexes) 为训练/验证集，使用 CASF-2016 (285 complexes) 作为标准化测试集，供后续所有 Sub-Plan 使用。

**内容**：
- 解析 `INDEX_general_PL.2020R1.lst`，提取 pdb_code → pKd 映射，过滤不精确标签
- 从有效集中**剔除 CASF-2016 的 285 个复合物**
- 提取蛋白序列 → mmseqs2 30% identity 聚类 → 按蛋白簇分层抽样划分 Train/Val
- CASF-2016 core set (285 complexes, 57 targets × 5 ligands) 作为独立 Test set
- Featurize（与 TargetDiff 同 featurizer），构建 `PDBbindPairDataset` DataLoader

**数据划分策略**：
1. 剔除 CASF-2016 后，提取每个复合物蛋白序列，用 mmseqs2 按 30% seq identity 聚类
2. 按蛋白簇（而非单个样本）划分：同一 cluster 的所有样本必须全部进 Train 或全部进 Val
3. 分层抽样：按 cluster median pKd 分位数分箱，在每个 bin 内抽 ~10–15% 的 clusters 作为 Val
4. **Test**: CASF-2016 core set（285 complexes）— 固定 benchmark，不参与训练

**为什么必须前置**：
- 后续所有 Sub-Plan 的训练和评估都基于此数据集
- 使用实验数据（晶体结构 + 实验 pKd）而非生成分子，大幅提升信号质量
- CASF-2016 是 scoring function 领域公认的标准 benchmark，便于与其他方法（OnionNet-2、PIGNet 等）直接对比

```
  PDBbind v2020 R1                data/pdbbind_v2020/
  19,037 P-L complexes    ────►   ├── processed/*.pt
  + INDEX_general_PL.2020R1.lst   ├── labels.csv
  (过滤后取有效 binding data)   ├── clusters.json
                                  └── splits.json  (train / val / test)
  CASF-2016 core set
  285 complexes           ────►   test split (独立 benchmark)
  + CoreSet.dat
```

**详细设计**：见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)

**验收标准**：DataLoader 能正确 batch 不等长蛋白-配体 pair；Train/Val 中不含 CASF-2016 PDB code；同一 protein cluster 不跨 Train/Val；Val pKd 分布与 Train 接近

---

### Phase A: Representation Upgrade (Sub-Plans 3 → 2 → 1, 串行链)

**Goal**: Replace the current mean-pooled 128-dim embedding with a richer, multi-granularity representation. **Builds on the pretrained encoder from Phase 0.**

> **架构决策**：Sub-Plans 1–3 不是并列独立的候选方案，而是一条串行处理链。
> Sub-Plan 3 先做"多层信息提纯"，Sub-Plan 2 再做"原子级重要性汇总"，
> Sub-Plan 1 最后做"交互级 + 全局级融合"。

```
  ┌─────────────────────────────────────────────┐
  │          TargetDiff Frozen Encoder           │
  │          (SE(3)-Equivariant, L layers)       │
  └──────────────────┬──────────────────────────┘
                     │  H^(1), H^(2), ..., H^(L)
                     │  (per-layer atom-level hidden states)
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Step 1: Token-Level Layer Fusion (SP3)     │
  │  对每个原子 i，融合 top-k 层的表示：         │
  │  h̃_i = Fuse(h_i^(l₁), h_i^(l₂), ..., h_i^(lₖ)) │
  └──────────────────┬──────────────────────────┘
                     │  {h̃_i} — 融合后的原子表示
                     │
          ┌──────────┼──────────────────┐
          ▼          ▼                  ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ Step 2 (SP2) │ │ Step 3a(SP1) │ │ Step 3b      │
  │ Attn Pool    │ │ Interaction  │ │ Global Pool  │
  │ on {h̃_i}    │ │ Graph + GNN  │ │ mean({h̃_i}) │
  │              │ │ on {h̃_i} +  │ │              │
  │              │ │ pocket feats │ │              │
  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                │                │
         ▼                ▼                ▼
      z_atom        z_interaction       z_global
         │                │                │
         └────────────────┼────────────────┘
                          ▼
               ┌──────────────────┐
               │ Fusion (SP1)     │
               │ MLP / Gated      │
               │ → z_new ∈ ℝ^d   │
               └────────┬─────────┘
                        ▼
              ┌──────────────────┐
              │   GP / DKL /     │
              │   Ranking Head   │
              │   (Sub-Plan 4)   │
              └──────────────────┘
```

**串行依赖关系**：
- Sub-Plan 3 是最上游，输出 token-level 融合后的原子表示 $\tilde{h}_i$
- Sub-Plan 2 消费 $\tilde{h}_i$，输出 $z_{\text{atom}}$（不再是独立方法）
- Sub-Plan 1 是主干框架，消费 $\tilde{h}_i$（建交互图 + 全局池化）和 $z_{\text{atom}}$（来自 SP2），最终融合三路信息

**每个 Sub-Plan 仍可独立测试**（用于 ablation），但最终系统是这条完整链。

### Phase B: Predictor Upgrade (Sub-Plans 4–5)

**Goal**: Replace the single GP oracle with a more expressive predictor while preserving uncertainty estimates.

- Sub-Plan 4 (DKL / Hybrid Predictor): Wraps a neural feature extractor around the GP
- Sub-Plan 5 (Multi-Task Learning): Adds ranking and classification heads alongside regression

### Phase C: Feature Augmentation (Sub-Plan 6)

**Goal**: Inject domain knowledge via physics-aware features as auxiliary inputs.

### Phase D: Closed-Loop Generation (Sub-Plan 7)

**Goal**: Feed uncertainty signals back into the generative model for guided sampling.

---

## Shared Infrastructure

### New Modules to Create

| Module | Location | Purpose | Sub-Plan (链路位置) |
|--------|----------|---------|----------|
| `bayesdiff/pretrain_dataset.py` | New | PDBbind pair-level dataset & dataloader | 0 |
| `bayesdiff/layer_fusion.py` | New | Token-level multi-layer embedding fusion: $h_i^{(l)} \to \tilde{h}_i$ | 3 (Step 1) |
| `bayesdiff/attention_pool.py` | New | Attention-based aggregation: $\tilde{h}_i \to z_{\text{atom}}$ | 2 (Step 2) |
| `bayesdiff/interaction_graph.py` | New | Build pocket-ligand interaction graphs from $\tilde{h}_i$ | 1 (Step 3) |
| `bayesdiff/interaction_gnn.py` | New | Lightweight GNN for interaction encoding: graph → $z_{\text{interaction}}$ | 1 (Step 3) |
| `bayesdiff/multi_granularity.py` | New | 主干整合: fuse $z_{\text{atom}} + z_{\text{interaction}} + z_{\text{global}} \to z_{\text{new}}$ | 1 (Step 3) |
| `bayesdiff/pair_model.py` | New | Pair-level encoder + aggregation + predictor | 1–4 |
| `bayesdiff/hybrid_oracle.py` | New | DKL and NN+GP hybrid predictors | 4 |
| `bayesdiff/multi_task.py` | New | Multi-task heads (regression + rank + classification) | 5 |
| `bayesdiff/physics_features.py` | New | Physics-aware feature extraction | 6 |
| `bayesdiff/guided_sampling.py` | New | Uncertainty-guided generation | 7 |

### Modified Modules

| Module | Changes |
|--------|---------|
| `bayesdiff/sampler.py` | Expose per-layer atom-level embeddings (token-level, not pooled); return atom positions and pocket data |
| `bayesdiff/gen_uncertainty.py` | Support higher-dim inputs from multi-granularity repr |
| `bayesdiff/gp_oracle.py` | Accept variable-dim inputs; add DKL variant |
| `bayesdiff/fusion.py` | Generalize Delta method for new representations |
| `bayesdiff/evaluate.py` | Add ranking metrics (NDCG, MRR); per-family breakdown |

### New Pipeline Scripts

| Script | Purpose | Sub-Plan |
|--------|---------|----------|
| `scripts/pipeline/s00_prepare_pdbbind.py` | Prepare PDBbind v2020 R1 train/val + CASF-2016 test (pocket extraction, splits) | 0 |
| `scripts/pipeline/s08_extract_atom_embeddings.py` | Extract per-layer atom-level embeddings (token-level) from encoder | 3 (Step 1) |
| `scripts/pipeline/s08b_extract_multilayer.py` | Extract multi-layer embeddings (already completed, used in SP3 probing) | 3 |
| `scripts/pipeline/s09_build_interaction_graphs.py` | Construct pocket-ligand interaction graphs from $\tilde{h}_i$ | 1 (Step 3) |
| `scripts/pipeline/s10_train_enhanced.py` | Train full chain (SP3→SP2→SP1) + predictor | 1–4 |
| `scripts/pipeline/s11_ablation_stage2.py` | Stage 2 ablation studies (串行链逐步添加) | All |

### New Test Files

| Test File | Coverage | Sub-Plan |
|-----------|----------|----------|
| `tests/stage2/test_pretrain_dataset.py` | PDBbind 数据加载 | 0 |
| `tests/stage2/test_interaction_graph.py` | Interaction graph construction | 1 |
| `tests/stage2/test_attention_pool.py` | Attention pooling module | 2 |
| `tests/stage2/test_layer_fusion.py` | Multi-layer fusion module | 3 |
| `tests/stage2/test_hybrid_oracle.py` | DKL / hybrid predictor | 4 |
| `tests/stage2/test_multi_task.py` | Multi-task heads | 5 |
| `tests/stage2/test_physics_features.py` | Physics feature extraction | 6 |
| `tests/stage2/test_stage2_integration.py` | End-to-end Stage 2 pipeline | All |

---

## Evaluation Framework

### Primary Metrics (Paper Table)

| Metric | Current (Stage 1) | Target (Stage 2) | Measure |
|--------|-------------------|-------------------|---------|
| $R^2$ (regression) | 0.120 | ≥ 0.30 | Prediction quality |
| Spearman $\rho$ | 0.369 | ≥ 0.55 | Ranking quality |
| AUROC (success) | 1.000 | ≥ 0.95 | Decision quality |
| ECE | 0.034 | ≤ 0.05 | Calibration |
| NLL | baseline | ≥ 10% reduction | Uncertainty quality |
| EF@1% | baseline | ≥ 1.5× | Enrichment |

### Ablation Matrix (Paper Table)

Each row = one configuration; columns = all metrics above.
Ablation 沿串行链逐步添加组件，验证每一步的增量贡献。

| Configuration | 对应链路 | Sub-Plans Active |
|---------------|---------|-----------------|
| Baseline (Stage 1: last-layer mean pool) | — | None |
| + Token-Level Layer Fusion (best method) → mean pool | SP3 only | 3 |
| + Token-Level Layer Fusion → Attention Pool | SP3 → SP2 | 3 + 2 |
| + Token-Level Layer Fusion → Attn Pool + Interaction Graph | SP3 → SP2 + SP1(interaction) | 3 + 2 + 1(partial) |
| + Full Chain (SP3 → SP2 → SP1: z_atom + z_interaction + z_global) | 完整串行链 | 3 + 2 + 1 |
| + Full Chain + DKL Predictor | 完整链 + 预测器升级 | 3 + 2 + 1 + 4 |
| + Full Chain + Multi-Task | 完整链 + 多任务 | 3 + 2 + 1 + 5 |
| + Full Chain + Physics Features | 完整链 + 物理特征 | 3 + 2 + 1 + 6 |
| Full Stage 2 | 全部 | 3 + 2 + 1 + 4 + 5 + 6 |
| (Ablation) SP2 only: Attn Pool on last-layer $h_i^{(L)}$ | SP2 单独 | 2 |
| (Ablation) SP1 only: Interaction Graph on last-layer $h_i^{(L)}$ | SP1 单独 | 1 |
| (Ablation) Shuffled-edge control on full chain | 拓扑 sanity check | 3 + 2 + 1(shuffled) |

---

## Paper Integration Plan

### Manuscript Sections Affected

| Section | Changes |
|---------|---------|
| §3 Methods | Add §3.X: Enhanced Representation Learning |
| §3 Methods | Add §3.Y: Hybrid Predictor Architecture |
| §4 Results | Add Table: Stage 2 Ablation Results |
| §4 Results | Add Figure: Representation quality comparison |
| §4 Results | Add Figure: Attention weight visualization |
| §5 Discussion | Update: representation bottleneck → partially resolved |
| Supplement | Full ablation details, per-family breakdown |

### Key Figures to Generate

1. **Architecture diagram** — Updated pipeline with multi-granularity representation
2. **Representation quality heatmap** — $R^2$ and $\rho$ across encoder variants
3. **Attention weight visualization** — Which atoms/interactions receive high attention
4. **Ablation bar chart** — Incremental gains from each sub-plan
5. **Calibration plot** — Before/after Stage 2 reliability diagrams
6. **Per-family scatter** — Performance across protein families

---

## Compute Budget

| Component | Estimated GPU-hours | Hardware |
|-----------|-------------------|----------|
| **Phase 0: 数据集准备** | | |
| PDBbind data preparation (INDEX + pocket + featurize + split) | 7–14h | CPU |
| **Phase A: Representation Upgrade** | | |
| Atom-level embedding extraction | 20–40h | A100 |
| Interaction graph construction | 2–4h | CPU |
| Attention pooling training | 10–20h | A100 |
| Multi-layer fusion training | 10–20h | A100 |
| DKL training | 20–40h | A100 |
| Multi-task training | 20–40h | A100 |
| Full ablation matrix (10 configs) | 100–200h | A100 |
| **Total** | **~180–360h** | **A100** |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Atom embeddings not available from TargetDiff | Blocks Sub-Plans 1, 2 | Pre-check encoder API; fallback to SchNet re-encoding |
| DKL overfits on small dataset | Blocks Sub-Plan 4 | Use SVGP backbone; strong regularization; early stopping |
| Multi-task objectives conflict | Degrades Sub-Plan 5 | Pareto-optimal λ search; gradient surgery |
| Physics features noisy | No improvement | Use as auxiliary input only; gate mechanism |
| Compute budget exceeded | Delays timeline | Prioritize P0 sub-plans; downsample ablation matrix |
