# 预训练嵌入提升方案 (Pre-trained Embedding Enhancement Plan)

> 日期：2026-03-27
> 目标：利用在大规模分子库上预训练的 3D GNN 权重提取更丰富的嵌入，替代/增强当前 TargetDiff 编码器嵌入，降低过拟合、提升 GP 泛化性能。

---

## 0. 问题诊断

当前 Encoder-128 嵌入来源于 TargetDiff 扩散模型的编码器（UniTransformerO2, 9 层, hidden_dim=128）。该编码器是在 CrossDocked2020 上作为扩散模型的一部分训练的——其目标是**生成分子**，而非**预测活性**。

| 问题 | 数据 |
|------|------|
| 过拟合 | Train ρ=0.90, Test ρ=0.35, gap=0.55 |
| 方差解释力低 | Test R²≈0.10（仅解释 10% 方差） |
| 每口袋分子数不足 | 平均 5.3 个生成分子/pocket |
| 正则化无效 | DKL+Dropout+WD 反而恶化性能（§18） |

**根本原因**：嵌入质量不足，而非模型正则化不足。

---

## 1. 可用预训练模型评估

### 1.1 方案排序

| 优先级 | 方案 | 预训练规模 | 领域匹配度 | 实施难度 | 预期收益 |
|:------:|------|:---------:|:---------:|:-------:|:-------:|
| **1** | **Uni-Mol** (unimol_tools) | 209M 构象 | 分子（好） | Easy (pip) | **高** |
| **2** | **Multi-layer TargetDiff** (重跑全量) | CrossDocked2020 | 蛋白-配体（完美） | None（已有） | **中** |
| **3** | **SchNet** (pyg QM9 预训练) | QM9 130K | 小分子（一般） | Easy（已安装） | **低-中** |
| **4** | **融合方案** | 组合多模型 | 互补 | Medium | **高** |

### 1.2 Uni-Mol 详细评估

- **预训练数据**：209M 分子构象（来自 ZINC + ChEMBL 等）
- **架构**：Transformer with 3D positional encoding
- **嵌入维度**：512-dim（CLS token）+ 原子级嵌入
- **安装**：`pip install unimol_tools`
- **权重**：自动从 HuggingFace 下载（`mol_pre_all_h_220816.pt`）
- **输入**：SMILES 字符串 或 原子坐标 + 类型
- **API**：`UniMolRepr(data_type='molecule').get_repr(smiles_list)` → 512-dim

**关键优势**：在 209M 构象上预训练，规模远超 TargetDiff（~100K 复合物），且专门优化分子表征。

### 1.3 Multi-layer TargetDiff

当前仅在 17 个 pocket 上测试过（结果不可靠）。需要在全部 942 个 pocket 上重新提取 10 层嵌入并评估。

已有脚本：`scripts/22_extract_multilayer_embeddings.py`（提取）、`scripts/23_train_gp_multilayer.py`（GP 评估）。

### 1.4 SchNet (PyG QM9 预训练)

- 已在 `torch_geometric 2.7.0` 中可用
- `SchNet.from_qm9_pretrained()` 提供 12 个 QM9 target 的预训练权重
- 128 hidden channels，6 interaction blocks
- 局限：QM9 仅含 130K 小分子（≤9 重原子），领域差距大

---

## 2. 实施计划

### Phase A：Multi-layer TargetDiff 全量嵌入提取（最快验证）

1. 在 GPU 节点上将 `scripts/22_extract_multilayer_embeddings.py` 跑满 942 个 pocket
2. 构建多层组合策略（concat, weighted avg, PCA 降维）
3. 用 GP 评估所有策略（LOOCV + 50× random split）

### Phase B：Uni-Mol 嵌入提取

1. `pip install unimol_tools`
2. 从 SDF 文件读取 SMILES → 提取 512-dim Uni-Mol 嵌入
3. 逐 pocket 平均 → 全局 `X_unimol_512.npy`
4. GP 评估

### Phase C：SchNet 嵌入提取

1. 加载 QM9 预训练 SchNet
2. 从 SDF 读取 3D 坐标 → 提取中间层嵌入（128-dim）
3. 逐 pocket 平均 → 全局 `X_schnet_128.npy`
4. GP 评估

### Phase D：嵌入融合

1. 最佳单模型嵌入 + Encoder-128 拼接
2. PCA 降维到最佳维度
3. GP 最终评估

### Phase E：结果可视化 & 文档

---

## 3. 评估指标

- LOOCV RMSE / Spearman ρ / R²
- 50× repeated random 80/20 split（mean ± std）
- Overfitting gap: Train ρ − Test ρ
- PCA 方差解释率

## 4. 成功标准

- Test ρ > 0.40（当前 0.35）
- Overfitting gap < 0.50（当前 0.55）
- Test R² > 0.15（当前 0.10）

---

## 5. 实验结果

### 各嵌入方案 GP 性能对比

| Embedding | Dim | N | LOOCV ρ | Test ρ (10×) | Overfit Gap | 状态 |
|-----------|-----|---|---------|--------------|-------------|------|
| **Encoder-128** (baseline) | 128 | 942 | **0.367** | **0.362±0.054** | 0.525 | ✅ |
| Uni-Mol-512 | 512 | 932 | 0.087 | 0.111±0.067 | 0.852 | ✅ |
| Uni-Mol→PCA-64 | 64 | 932 | 0.080 | 0.102±0.055 | — | ✅ |
| Encoder+UniMol→PCA-64 | 64 | 932 | 0.264 | 0.286±0.058 | 0.641 | ✅ |
| Encoder+UniMol→PCA-128 | 128 | 932 | 0.264 | 0.246±0.057 | — | ✅ |
| FCFP4-2048 | 2048 | 932 | 0.114 | 0.106±0.066 | — | ✅ |
| SchNet-128 | 128 | 942 | 0.096 | 0.118±0.063 | 0.561 | ✅ |
| Encoder+SchNet→256d | 256 | 942 | 0.356 | 0.355±0.060 | 0.543 | ✅ |

> **注意**: FCFP4 此处仅用 tier3 ~5mol/pocket SDF 样本。在 50mol 研究中 FCFP4-2048 LOOCV ρ=0.749。

### 关键发现

1. **预训练嵌入未能提升性能**：Uni-Mol (209M 分子预训练) LOOCV ρ=0.087，远低于 Encoder-128 baseline ρ=0.367
2. **SchNet 同样失败**: SchNet (QM9 130K 预训练) LOOCV ρ=0.096，与 Uni-Mol 同级别
3. **融合部分保留性能**: Encoder+SchNet→256d Test ρ=0.355 接近 baseline，但未超越
4. **原因分析**:
   - Uni-Mol 在通用分子构象上预训练，学到的是化学空间的通用表征
   - TargetDiff Encoder 虽然是为生成任务训练的，但它专门处理蛋白-配体界面的 SE(3) 几何特征
   - binding affinity 更依赖于**界面相互作用**而非**分子的内在化学性质**
4. **FCFP4 启示**：化学指纹在 50mol 数据上 ρ=0.749 远超所有 3D 嵌入，说明 2D 拓扑/药效团特征对 affinity 的预测力远超 3D 几何特征

### 可视化产物

`results/tier3_gp/` 目录：
- `01_comparison_bars.png` — LOOCV ρ + Test ρ 柱状图
- `02_boxplots.png` — Test ρ 分布箱线图
- `03_overfit_analysis.png` — Train vs Test 过拟合分析图
- `04_summary_table.png` — 完整结果表格
- `05_dim_vs_performance.png` — 维度 vs 性能散点图
- `06_pca_comparison.png` — PCA 降维效果对比
- `pretrained_comparison_results.json` — 完整 JSON 结果

---

## 6. 执行状态

- [x] Phase A: Multi-layer TargetDiff — 仅 17/1019 pockets（LMDB 已删除），结果不可靠
- [x] Phase B: Uni-Mol 提取 — 932 pockets, 512-dim (Job 5123719)
- [x] Phase C: SchNet 提取 — 942 pockets, 128-dim (Job 5130267, schnetpack stub modules fix)
- [x] Phase D: 嵌入融合 + 比较 — 17 configs × GP 评估，6 figures
- [x] Phase E: 文档更新

### SchNet 调试历史

1. ✅ `extract_zip` import → stdlib `zipfile`
2. ✅ 目录名 `qm9_U0` → `qm9_energy_U0`
3. ✅ torchvision 循环导入 → `sys.modules` mock
4. ✅ `schnetpack.atomistic.model` 缺失 → 注册 stub modules (Job 5130267 ✅ 成功)

### 总结论

**预训练嵌入实验结果为负面** — 不满足成功标准 (Test ρ > 0.40, Gap < 0.50)。
当前最优路线是 **FCFP4 化学指纹** (ρ=0.749)，而非 3D 嵌入增强。
