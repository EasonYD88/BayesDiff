# BayesDiff Stage 2 — Progress Log

## Sub-Plan 0: PDBbind v2020 数据集准备

**Status**: ✅ 数据准备完成  
**Commit**: `9129938` — `feat: PDBbind v2020 dataset preparation pipeline (Sub-Plan 0)`

### 数据下载（2026-04-05 完成 ✅）

已下载至 `data/PDBbind_data_set/`：

| 文件 | 大小 | 内容 |
|------|------|------|
| `PDBbind_v2020_R1_index.tar.gz` | 498K | 索引文件（`INDEX_general_PL.2020R1.lst`，19,037 P-L complexes） |
| `PDBbind_v2020_R1_ligand_protein_data.tar.gz` | 1.2G | 结构文件（`P-L/YEAR_RANGE/XXXX/`，含 protein, pocket, ligand） |
| `CASF-2016.tar.gz` | 1.5G | CASF-2016 benchmark（`CoreSet.dat` + structures） |

> **与原计划差异**：下载到的是 PDBbind v2020 R1 **general set**（19,037 complexes），
> 而非 refined set（~5,316）。R1 版本使用 v2024 re-processed structures，质量更高。

### 数据处理结果（2026-04-05 完成 ✅）

#### Stage 1: INDEX 解析

| 指标 | 数值 |
|------|------|
| INDEX 总条目 | 19,037 |
| 不精确标签剔除（`<`, `>`, `~`） | 270 |
| 有效条目 | 18,767 |
| 匹配结构文件 | 18,767/18,767 (100%) |
| CASF-2016 PDB codes | 285 (57 targets) |
| 亲和力类型 | Kd: 6,935 / IC50: 6,932 / Ki: 4,900 |
| pKd 范围 | [0.40, 15.22], mean=6.39±1.83 |

#### Stage 2: Featurize（100-shard SLURM array）

| 指标 | 数值 |
|------|------|
| 成功 featurize | 18,765 |
| 失败 | 2 (3vjs, 3vjt — 配体 SDF 无法读取) |
| 输出 | `data/pdbbind_v2020/processed/*.pt` |
| 并行策略 | cpu_short + l40s_public 双分区100-shard array job |

#### Stage 3-4: Merge + Cluster-Stratified Split

| 指标 | 数值 |
|------|------|
| mmseqs2 聚类阈值 | 30% sequence identity |
| 蛋白簇总数 | 3,250 |
| Val 簇数 | 386 (11.9%) |

**最终数据划分：**

| Split | Complexes | pKd 范围 | mean±std |
|-------|-----------|----------|----------|
| Train | 16,232 | [0.45, 15.22] | 6.40±1.83 |
| Val | 2,248 (12.2%) | [0.40, 12.00] | 6.31±1.79 |
| Test (CASF-2016) | 285 | [2.07, 11.82] | 6.49±2.17 |

- ✅ 三个 split 之间零重叠
- ✅ 同蛋白簇样本全部进同一 split（无信息泄漏）
- ✅ pKd 分布在 Train/Val 间保持一致（分层抽样）
- 输出文件：`data/pdbbind_v2020/splits.json`, `data/pdbbind_v2020/labels.csv`

### 代码更新

| 文件 | 修改 |
|------|------|
| `bayesdiff/data.py` | `parse_pdbbind_index()` 支持 R1/refined 自动检测；新增 `parse_casf_coreset()`, `cluster_stratified_split()`, `_find_protein_pdb()` |
| `bayesdiff/__init__.py` | 新增导出 `parse_casf_coreset`, `load_casf2016_codes`, `cluster_stratified_split` |
| `scripts/pipeline/s00_prepare_pdbbind.py` | 重写：支持 R1 `P-L/YEAR_RANGE/` 目录结构、CASF-2016 test split、`code_to_dir.json` 映射、`shard_status_{:04d}of{:04d}` 格式 |
| `slurm/s00_featurize_cpu.sh` | 新增 100-shard CPU array job (cpu_short partition) |
| `slurm/s00_featurize_array.sh` | 新增 100-shard GPU array job (l40s_public partition) |
| `slurm/s00_merge_split.sh` | 新增 merge + split SLURM script |

### 新增文件

| 文件 | 用途 |
|------|------|
| `bayesdiff/pretrain_dataset.py` | `PDBbindPairDataset` + `collate_pair_data`（PyG 风格变长 batch） |
| `scripts/pipeline/s00_prepare_pdbbind.py` | 4 阶段处理流水线（parse → featurize → merge → split），支持 `--shard_index/--num_shards` |
| `scripts/pipeline/s00b_pdbbind_eda.py` | EDA 可视化（pKd 分布、split 质量、配体理化性质、数据质量报告） |
| `scripts/utils/download_pdbbind.sh` | PDBbind 下载辅助脚本 |
| `slurm/pipeline/s00a_parse.sh` | SLURM: 解析 INDEX 文件 |
| `slurm/pipeline/s00b_featurize_array.sh` | SLURM: 50 分片并行 featurize（array job） |
| `slurm/pipeline/s00c_merge_split.sh` | SLURM: 合并分片 + 蛋白家族 split |
| `slurm/pipeline/s00d_eda.sh` | SLURM: EDA 可视化 |
| `slurm/pipeline/s00_launch_all.sh` | 一键启动全部，自动设置 job 依赖 |
| `tests/test_pretrain_dataset.py` | 7 个测试覆盖 M0.1–M0.4 |

### 修改文件

| 文件 | 修改 |
|------|------|
| `bayesdiff/__init__.py` | 新增 `PDBbindPairDataset`, `get_pdbbind_dataloader` 导出 |
| `bayesdiff/data.py` | 无需修改 — `parse_pdbbind_index()` 已完整支持 v2020 refined set |

### 里程碑验证

| 里程碑 | 验证结果 |
|--------|----------|
| M0.1 INDEX 解析 | ✅ 18,767 条目解析成功，自动检测 R1 格式 |
| M0.2 CASF-2016 解析 + 剔除 | ✅ 285 PDB codes 全部找到并标记为 test |
| M0.3 Pocket 提取 | ✅ R1 数据已含预提取 `_pocket.pdb`，缺失时自动提取 10Å pocket |
| M0.4 蛋白序列聚类 | ✅ mmseqs2 30% identity → 3,250 clusters from 18,480 sequences |
| M0.5 数据集划分 | ✅ Cluster-stratified split: 16,232 train / 2,248 val / 285 test |
| M0.6 DataLoader | ✅ 变长 batch 正确（protein_pos, ligand_pos 正确拼接，batch index 正确） |
| 单复合物 featurize | ✅ 18,765/18,767 成功（2 失败：ligand SDF 不可读） |
| 全流水线 | ✅ parse → featurize (100-shard) → merge → split 完成 |
| 分片并行 | ✅ 100 分片 × cpu_short+l40s_public 双分区并行 |

### HPC 并行策略

- 100 个 SLURM array task 在 `cpu_short` + `l40s_public` 双分区提交
- cpu_short: 2 concurrent tasks × 16 CPU workers
- l40s_public: 4+ concurrent tasks × 16 CPU workers
- 18,767 complexes / 100 shards ≈ 188/shard
- 实际 wall time: ~8 min (parse) + ~10 min (featurize, 并行) + ~8 min (split with mmseqs2)

### 阻塞项

全部已解决：
- ~~⚠️ PDBbind v2020 raw data 未下载~~ ✅ 已下载
- ~~⚠️ 代码适配~~ ✅ `s00_prepare_pdbbind.py` + `data.py` 重写完成
- ~~⚠️ CASF-2016 处理~~ ✅ CoreSet.dat 解析 + test split
- ~~⚠️ 蛋白序列聚类~~ ✅ mmseqs2 安装 + 3,250 clusters
- ~~⚠️ Split 重写~~ ✅ cluster-stratified split (12.2% val)

---

## Sub-Plan 0: 5-Fold Grouped Train/Val Splits

**Status**: ✅ 完成  
**Date**: 2026-04-06

### 动机

单一 Train/Val 划分可能引入划分偏差。为保证模型选择和超参调优的稳健性，在同一蛋白聚类结果上执行 5 次不同随机种子的 grouped + stratified Train/Val 划分。CASF-2016 永远固定为独立 test benchmark。

### 5-Fold 划分结果

| Fold | Seed | Train | Val | Val% | Train pKd mean | Val pKd mean | KS p-value | Pass |
|------|------|-------|-----|------|----------------|--------------|------------|------|
| 0 | 42 | 16,108 | 2,372 | 12.8% | 6.383 | 6.393 | 0.244 | ✅ |
| 1 | 43 | 16,253 | 2,227 | 12.1% | 6.383 | 6.392 | 0.379 | ✅ |
| 2 | 44 | 16,159 | 2,321 | 12.6% | 6.376 | 6.437 | 0.142 | ✅ |
| 3 | 45 | 16,211 | 2,269 | 12.3% | 6.381 | 6.406 | 0.089 | ✅ |
| 4 | 50 | 16,102 | 2,378 | 12.9% | 6.386 | 6.373 | 0.090 | ✅ |

- Test (CASF-2016): 285 complexes（5 个 fold 共享）
- 蛋白簇: 3,250（mmseqs2 30% seq identity，共享同一聚类结果）
- 所有 fold KS test $p > 0.05$ ✅
- Fold 间 Val Jaccard 最大重叠率: 0.162（远低于 0.8 退化阈值）✅
- 同 cluster 不跨 Train/Val ✅

> **注**：默认 seed 规则为 `42 + fold_id`，但 fold 4 的原始 seed=46 未通过 KS test（$p = 2.4 \times 10^{-6}$），替换为 seed=50（$p = 0.090$，PASS）。

### 代码更新

| 文件 | 修改 | 说明 |
|------|------|------|
| `bayesdiff/data.py` | 新增 `cluster_stratified_split_nfold()` | 5-fold grouped stratified split 核心函数 |
| `scripts/pipeline/s00_prepare_pdbbind.py` | 新增 `--n_folds` 参数 | `stage_split()` 中调用 nfold 函数，生成 `splits_5fold.json` |
| `bayesdiff/pretrain_dataset.py` | 新增 `fold_id` 参数 | `PDBbindPairDataset` 支持从 `splits_5fold.json` 加载指定 fold |
| `scripts/pipeline/s00b_pdbbind_eda.py` | 新增 `--only_5fold` 参数 | 5-fold 质量检查输出至 `results/pdbbind_eda/5fold/` |
| `doc/Stage_2/00a_supervised_pretraining.md` | 新增 §1.3.2, §2.3.1 等 | 5-fold 方案文档化 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `data/pdbbind_v2020/splits_5fold.json` | 5-fold splits（含 seed、cluster_info） |
| `data/pdbbind_v2020/splits.json` | 默认划分 = fold 0（向后兼容） |
| `results/pdbbind_eda/5fold/5fold_val_sizes.png` | 5 fold Val 样本数对比 |
| `results/pdbbind_eda/5fold/5fold_val_pkd_kde.png` | 5 fold Val pKd 分布叠加 |
| `results/pdbbind_eda/5fold/5fold_ks_test_summary.csv` | KS test 汇总 |
| `results/pdbbind_eda/5fold/5fold_val_overlap_heatmap.png` | Fold 间 Val Jaccard 热图 |
| `results/pdbbind_eda/5fold/eda_summary.json` | 5-fold EDA 汇总 |

### 里程碑验证

| 里程碑 | 验证结果 |
|--------|----------|
| M0.5 5-fold grouped split | ✅ 5 组 Train/Val + 固定 Test；所有 KS test p > 0.05；max Jaccard = 0.162 < 0.8 |

---

## Sub-Plan 1: Multi-Layer Fusion — Stage 0: Infrastructure

**Status**: ✅ 完成  
**Date**: 2026-04-06  
**Plan**: `doc/Stage_2/03_multi_layer_fusion.md`

### 架构分析

| 属性 | 值 |
|------|------|
| 模型 | `ScorePosNet3D` → `UniTransformerO2TwoUpdateGeneral` |
| `num_blocks` | 1 |
| `num_layers` (base_block) | 9 |
| 总隐藏层数 | 10（1 init_h_emb_layer + 9 AttentionLayerO2TwoUpdateNodeGeneral） |
| `hidden_dim` | 128 |
| `return_layer_h` | 模型已原生支持，无需 forward hooks |

### 核心实现

**Embedding 提取方法**：单次 forward pass + crystal ligand（`fix_x=True`），不走 diffusion sampling，速度远快于 93-pocket 方法。

| 文件 | 修改/新增 | 说明 |
|------|-----------|------|
| `bayesdiff/sampler.py` | 修改 | 新增 3 个方法：`load_complex_data(pt_path)`, `extract_multilayer_embeddings(pt_path)`, `num_encoder_layers` property |
| `scripts/pipeline/s08b_extract_multilayer.py` | 新增 | 多层 embedding 提取脚本，支持 `--shard_index/--num_shards` 分片并行 + `--stage merge` 合并 |
| `slurm/s08b_extract_multilayer.sh` | 新增 | 50-shard SLURM array job（a100_chemistry partition） |
| `slurm/s08b_merge.sh` | 新增 | 合并脚本，依赖 array job 完成后运行 |
| `tests/stage2/__init__.py` | 新增 | Stage 2 测试目录 |
| `tests/stage2/test_multilayer_extraction.py` | 新增 | 5 个测试：T2.1 层数=10, T2.2 shape=(128,), T2.3 无NaN/Inf, T2.4 层间差异, T2.5 load_complex_data |

### 方法说明

- `load_complex_data(pt_path)`: 加载 `.pt` 文件，保留蛋白质和配体数据（区别于 `load_pocket_data` 会丢弃配体）
- `extract_multilayer_embeddings(pt_path)`: 构建 batch → protein_atom_emb + ligand_atom_emb → `compose_context()` → `refine_net(return_layer_h=True)` → 对每层 ligand hidden states 取 mean pooling → 返回 `{'layer_0': (128,), ..., 'layer_9': (128,), 'z_global': (128,), 'n_layers': 10}`
- `num_encoder_layers`: 返回 `1 + len(self._model.refine_net.base_block)` = 10

### 下一步

- ~~提交 SLURM extraction job（18,765 complexes × 50 shards）~~ ✅ 完成
- ~~合并 embeddings → `results/multilayer_embeddings/all_multilayer_embeddings.npz`~~ ✅ 完成
- ~~开始 Stage 1: Single-Layer Probing（s09a_single_layer_probe.py）~~ ✅ 完成

---

## Sub-Plan 1: Multi-Layer Fusion — Stage 0b: Extraction Run

**Status**: ✅ 完成  
**Date**: 2026-04-06

### 提取结果

| 指标 | 数值 |
|------|------|
| 总复合物 | 18,765 |
| 成功提取 | 18,765 (100%) |
| 失败 | 0 |
| 50-shard 并行耗时 | ~2 min/shard |
| 合并文件大小 | 155 MB |
| 合并 keys | 225,180 (18,765 × 12: 10 layers + z_global + n_layers) |
| 输出 | `results/multilayer_embeddings/all_multilayer_embeddings.npz` |

### Bugfix

- `protein_atom_to_aa_type >= 20`（非标准氨基酸）→ clamp 到 [0,19]
- `.pt` 文件缺少 `ligand_hybridization` → 手动从 `ligand_atom_feature` 的 aromatic 列计算 `ligand_atom_feature_full`（`add_aromatic` mode 只需 element + aromatic）
- SLURM 脚本 conda activation 修复

---

## Sub-Plan 1: Multi-Layer Fusion — Stage 1: Single-Layer Probing

**Status**: ✅ 完成  
**Date**: 2026-04-06  
**Commit**: (pending)

### E1.1 — Per-Layer GP Metrics

| Layer | Val R² | Val ρ | Val RMSE | Val NLL | Test R² | Test ρ | Test RMSE | Test NLL |
|-------|--------|-------|----------|---------|---------|--------|-----------|----------|
| L0 | 0.114 | 0.356 | 1.670 | 1.932 | 0.129 | 0.434 | 2.026 | 2.158 |
| L1 | 0.213 | 0.458 | 1.574 | 1.872 | 0.309 | 0.577 | 1.804 | 2.033 |
| L2 | 0.216 | 0.474 | 1.572 | 1.870 | 0.348 | 0.617 | 1.753 | 1.996 |
| L3 | 0.205 | 0.462 | 1.582 | 1.876 | 0.359 | 0.640 | 1.738 | 1.987 |
| L4 | 0.219 | 0.465 | 1.569 | 1.868 | 0.381 | 0.659 | 1.708 | 1.966 |
| L5 | 0.224 | 0.479 | 1.564 | 1.865 | 0.427 | 0.683 | 1.643 | 1.923 |
| L6 | 0.233 | 0.495 | 1.554 | 1.859 | 0.429 | 0.681 | 1.639 | 1.923 |
| L7 | 0.221 | 0.487 | 1.566 | 1.867 | 0.418 | 0.674 | 1.656 | 1.936 |
| **L8** | **0.250** | **0.512** | **1.537** | **1.847** | 0.432 | 0.686 | 1.636 | 1.921 |
| L9 | 0.232 | 0.498 | 1.555 | 1.860 | **0.449** | **0.697** | **1.612** | **1.903** |

**关键发现：**
- L0（init embedding）信号最弱（Val R²=0.114），与预期一致
- 性能随层数单调递增后在 L5-L6 趋于平台
- **L8 在验证集上超过 L9**（Val R²=0.250 vs 0.232），说明倒数第二层可能更适合
- L9 在测试集上略优（Test R²=0.449 vs 0.432），可能due to CASF-2016 的 distribution shift
- 所有层表现远超之前 50mol GP（ρ≈0.4），得益于 16K 训练样本

### E1.2 — CKA Similarity Matrix

- 保存为 `results/stage2/layer_probing/cka_matrix.npy`
- 可视化保存为 `results/stage2/layer_probing/cka_heatmap.png`

### Gate 1 Decision

| 指标 | 值 |
|------|------|
| Last layer (L9) val R² | 0.2323 |
| Best non-final (L8) val R² | 0.2501 |
| Ratio | 1.077 |
| Threshold | 0.90 |
| **Decision** | ✅ **PROCEED** to Stage 2 |

> L8 在验证集上 R² 比 L9 高 7.7%，且 L5-L8 都与 L9 可比，
> 说明不同层包含互补信息，multi-layer fusion 有潜力。

### 代码更新

| 文件 | 修改/新增 | 说明 |
|------|-----------|------|
| `scripts/pipeline/s09a_single_layer_probe.py` | 新增 | Per-layer GP + CKA + Gate 1 决策脚本 |
| `slurm/s09a_layer_probe.sh` | 新增 | SLURM job script |

### 输出文件

| 文件 | 说明 |
|------|------|
| `results/stage2/layer_probing/layer_probing.csv` | 10 层 × 8 metrics |
| `results/stage2/layer_probing/layer_probing.png` | Bar chart (Fig L.1) |
| `results/stage2/layer_probing/cka_heatmap.png` | CKA 矩阵热图 (Fig L.2) |
| `results/stage2/layer_probing/gate1_decision.json` | Gate 1 决策记录 |
| `results/stage2/layer_probing/stage1_summary.json` | 完整汇总 |
| `results/stage2/layer_probing/gp_layer_*.pt` | 10 个 per-layer GP 模型 |

### 下一步

- Stage 2: Weighted Sum Fusion — 实现 `WeightedSumFusion` 可学习层权重
- 在已有 multi-layer embeddings 上训练 weighted sum GP
- Gate 2: 多层融合是否超过最佳单层？
