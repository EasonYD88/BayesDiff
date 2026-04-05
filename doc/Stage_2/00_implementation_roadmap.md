# Stage 2 Implementation Roadmap

## Overview

This document provides the master plan for BayesDiff Stage 2: upgrading the system from a **calibrated uncertainty evaluator** (Stage 1) to a **high-performance uncertainty-aware molecular ranking system** with significantly improved representation and prediction quality.

Stage 2 addresses the three core bottlenecks identified in [problem_and_solution.md](problem_and_solution.md):

1. **Representation bottleneck** — current mean-pooled SE(3) embeddings capture only ~12% of variance
2. **Information compression** — mean pooling destroys atom-level and interaction-level signals
3. **System positioning** — BayesDiff should be a *ranking/prioritization* system, not a pure regression system

---

## Sub-Plan Index

| # | Sub-Plan | File | Priority | Dependency |
|---|----------|------|----------|------------|
| **0** | **PDBbind v2020 数据集准备** | [00a_supervised_pretraining.md](00a_supervised_pretraining.md) | **P0 — Critical (前置)** | None |
| 1 | Multi-Granularity Representation | [01_multi_granularity_repr.md](01_multi_granularity_repr.md) | **P0 — Critical** | Sub-Plan 0 |
| 2 | Attention-Based Aggregation | [02_attention_aggregation.md](02_attention_aggregation.md) | **P0 — Critical** | Sub-Plan 0 |
| 3 | Multi-Layer Fusion | [03_multi_layer_fusion.md](03_multi_layer_fusion.md) | **P1 — High** | Sub-Plan 0 |
| 4 | Hybrid Predictor (DKL) | [04_hybrid_predictor.md](04_hybrid_predictor.md) | **P1 — High** | Sub-Plans 0–3 |
| 5 | Multi-Task Learning | [05_multi_task_learning.md](05_multi_task_learning.md) | **P2 — Medium** | Sub-Plans 0–3 |
| 6 | Physics-Aware Features | [06_physics_aware_features.md](06_physics_aware_features.md) | **P2 — Medium** | Sub-Plan 0 |
| 7 | Uncertainty-Guided Generation | [07_uncertainty_guided_generation.md](07_uncertainty_guided_generation.md) | **P3 — Future** | Sub-Plans 0–6 |

---

## Implementation Phases

### Phase 0: PDBbind v2020 数据集准备 (Sub-Plan 0)

**Goal**: 整理 PDBbind v2020 refined set (~5,316 complexes) 为统一的 pair-level 训练数据集，供后续所有 Sub-Plan 使用。

**内容**：
- 解析 INDEX 文件，提取 pdb_code → pKd 映射
- 提取 10Å pocket，featurize（与 TargetDiff 同 featurizer）
- 按蛋白家族 30% identity 聚类划分 Train / Val / Cal / Test
- 构建 `PDBbindPairDataset` DataLoader

**为什么必须前置**：
- 后续所有 Sub-Plan 的训练和评估都基于此数据集
- 使用实验数据（晶体结构 + 实验 pKd）而非生成分子，大幅提升信号质量

```
  PDBbind v2020 refined set       data/pdbbind_v2020/
  ~5,316 complexes        ────►   ├── processed/*.pt
  + INDEX_refined_data.2020       ├── labels.csv
                                  └── splits.json
```

**详细设计**：见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)

**验收标准**：DataLoader 能正确 batch 不等长蛋白-配体 pair，split 无家族泄漏

---

### Phase A: Representation Upgrade (Sub-Plans 1–3)

**Goal**: Replace the current mean-pooled 128-dim embedding with a richer, multi-granularity representation. **Builds on the pretrained encoder from Phase 0.**

```
                      ┌─────────────────────┐
                      │  TargetDiff Encoder  │
                      │  (SE(3)-Equivariant) │
                      └─────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ Sub-Plan 1  │  │ Sub-Plan 2  │  │ Sub-Plan 3  │
     │ Multi-Gran  │  │ Attn Pool   │  │ Layer Fuse  │
     │ Repr        │  │             │  │             │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                   ┌─────────────────┐
                   │ z_new ∈ ℝ^d_new │
                   │  (richer repr)  │
                   └────────┬────────┘
                            ▼
                 ┌──────────────────┐
                 │   GP / Hybrid    │
                 │   (Sub-Plan 4)   │
                 └──────────────────┘
```

**Sub-Plans 1, 2, 3 are independent** — can be implemented and tested in parallel.
Each produces a candidate embedding; final system combines the best choices.

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

| Module | Location | Purpose | Sub-Plan |
|--------|----------|---------|----------|
| `bayesdiff/pretrain_dataset.py` | New | PDBbind pair-level dataset & dataloader | 0 |
| `bayesdiff/pair_model.py` | New | Pair-level encoder + aggregation + predictor | 1–4 |
| `bayesdiff/interaction_graph.py` | New | Build pocket-ligand interaction graphs | 1 |
| `bayesdiff/attention_pool.py` | New | Attention-based aggregation module | 2 |
| `bayesdiff/layer_fusion.py` | New | Multi-layer embedding extraction & fusion | 3 |
| `bayesdiff/hybrid_oracle.py` | New | DKL and NN+GP hybrid predictors | 4 |
| `bayesdiff/multi_task.py` | New | Multi-task heads (regression + rank + classification) | 5 |
| `bayesdiff/physics_features.py` | New | Physics-aware feature extraction | 6 |
| `bayesdiff/guided_sampling.py` | New | Uncertainty-guided generation | 7 |

### Modified Modules

| Module | Changes |
|--------|---------|
| `bayesdiff/sampler.py` | Expose per-layer embeddings; return atom-level features |
| `bayesdiff/gen_uncertainty.py` | Support higher-dim inputs from multi-granularity repr |
| `bayesdiff/gp_oracle.py` | Accept variable-dim inputs; add DKL variant |
| `bayesdiff/fusion.py` | Generalize Delta method for new representations |
| `bayesdiff/evaluate.py` | Add ranking metrics (NDCG, MRR); per-family breakdown |

### New Pipeline Scripts

| Script | Purpose | Sub-Plan |
|--------|---------|----------|
| `scripts/pipeline/s00_prepare_pdbbind.py` | Prepare PDBbind v2020 refined set (pocket extraction, splits) | 0 |
| `scripts/pipeline/s08_extract_atom_embeddings.py` | Extract atom-level embeddings from encoder | 1 |
| `scripts/pipeline/s09_build_interaction_graphs.py` | Construct pocket-ligand interaction graphs | 1 |
| `scripts/pipeline/s10_train_enhanced.py` | Train with new representation + predictor | 1–4 |
| `scripts/pipeline/s11_ablation_stage2.py` | Stage 2 ablation studies | All |

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

| Configuration | Sub-Plans Active |
|---------------|-----------------|
| Baseline (Stage 1) | None |
| + Attention Pooling | 2 |
| + Multi-Layer Fusion | 3 |
| + Interaction Graph | 1 |
| + Attention + Multi-Layer | 2 + 3 |
| + Full Repr (1+2+3) | 1 + 2 + 3 |
| + DKL Predictor | 1 + 2 + 3 + 4 |
| + Multi-Task | 1 + 2 + 3 + 5 |
| + Physics Features | 1 + 2 + 3 + 6 |
| Full Stage 2 | 1 + 2 + 3 + 4 + 5 + 6 |

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
