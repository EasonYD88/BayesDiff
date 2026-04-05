# Sub-Plan 0: PDBbind v2020 数据集准备

> **Priority**: P0 — Critical（前置于所有其他 Sub-Plan）  
> **Dependency**: None  
> **Estimated Effort**: 1–2 weeks  
> **Paper Section**: §3.X Data Preparation

---

## 0. 背景与动机

当前 BayesDiff 的 predictor 训练数据来自 TargetDiff 生成的分子——这些分子本身质量参差不齐，embedding 信号弱（$R^2 = 0.12$）。核心问题在于：**我们试图用一个弱表征（mean-pooled SE(3) embedding）+ 弱数据（生成分子 + 估算的 pKd label）来训练 oracle，两头都弱。**

**解决思路**：先整理一个高质量的 PDBbind v2020 refined set 数据集（实验解析的蛋白-配体复合物 + 实验测定的 pKd），作为后续所有 Sub-Plan 的**统一训练数据源**。后续 Sub-Plan 1–7 的 representation、predictor、multi-task 等模块均在此数据集上训练和评估。

**为什么用 PDBbind 而非生成分子？**
1. **数据质量高**：配体构象来自晶体解析，pKd 来自实验测量，远比生成分子 + 估算标签可靠
2. **Pair-level 表征**：每条样本天然是 (ligand, pocket, pKd) 三元组，适合训练 pair-level encoder
3. **Encoder 共享**：使用与 TargetDiff 相同的 featurizer，确保特征维度一致，训练好的 encoder 权重可直接迁移到生成分子评估阶段

---

## 1. 数据集：PDBbind v2020 Refined Set

### 1.1 数据概览

| 属性 | 值 |
|------|-----|
| 来源 | PDBbind v2020 refined set |
| 复合物数量 | ~5,316 |
| 标签 | 实验 pKd/pKi（转换为统一 pKd 尺度） |
| 配体格式 | SDF（晶体构象） |
| 蛋白格式 | PDB（全蛋白 → 提取 pocket） |
| Pocket 提取 | 配体周围 10Å 球形截取 |

### 1.2 原始数据结构

```
PDBbind v2020 refined/
├── XXXX/                      # PDB code
│   ├── XXXX_protein.pdb       # 全蛋白结构
│   ├── XXXX_ligand.sdf        # 配体（晶体构象）
│   └── XXXX_pocket.pdb        # 10Å pocket（需提取）
└── INDEX_refined_data.2020    # pKd labels
```

### 1.3 数据处理流程

1. **解析 INDEX 文件**：复用 `bayesdiff/data.py` 中的 `parse_pdbbind_index()` 获取 pdb_code → pKd 映射
2. **提取 pocket**：复用 `extract_pocket_from_protein()` 对每个复合物提取 10Å pocket
3. **构建 pair 数据**：每条样本 = (pocket_atoms, ligand_atoms, pKd)
4. **Featurize**：使用与 TargetDiff 相同的 featurizer（`FeaturizeProteinAtom`, `FeaturizeLigandAtom`），确保特征维度与生成阶段完全一致
5. **划分数据集**：
   - 使用 `protein_family_split()` 基于 mmseqs2 30% sequence identity 聚类
   - Train / Val / Cal / Test = 70% / 10% / 10% / 10%
   - **关键**：确保同一蛋白家族不跨集，避免数据泄漏

### 1.4 输出格式

```
data/pdbbind_v2020/
├── processed/                 # 处理后的 pair 数据
│   ├── XXXX.pt                # 每个复合物的 PyTorch 数据
│   └── ...
├── labels.csv                 # pdb_code, pKd, affinity_type
└── splits.json                # {train: [...], val: [...], cal: [...], test: [...]}
```

### 1.5 数据加载器

```python
class PDBbindPairDataset(Dataset):
    """
    每个样本返回：
    - protein_pos:    (N_pocket, 3)    pocket 原子坐标
    - protein_feat:   (N_pocket, d_p)  pocket 原子特征
    - ligand_pos:     (N_ligand, 3)    配体原子坐标  
    - ligand_feat:    (N_ligand, d_l)  配体原子特征
    - pkd:            scalar           实验 pKd 标签
    
    使用与 TargetDiff 相同的 featurizer：
    - FeaturizeProteinAtom
    - FeaturizeLigandAtom
    确保特征维度与生成阶段完全一致。
    """
```

---

## 2. 数据可视化与分析

数据准备完成后，需对数据集进行系统性的 EDA（Exploratory Data Analysis），确保数据质量并为后续建模提供依据。

### 2.1 标签分布

| 可视化 | 内容 | 目的 |
|--------|------|------|
| pKd 直方图 | 全数据集 pKd 分布（bin=0.5） | 检查是否近正态、有无严重偏斜 |
| pKd 按 split 对比 | Train / Val / Cal / Test 各自的 pKd 分布 | 确认 split 后标签分布无明显偏移 |
| Affinity type 饼图 | Kd / Ki / IC50 各占比 | 了解标签来源异质性 |
| 分辨率 vs pKd 散点图 | X 轴晶体分辨率，Y 轴 pKd | 检查低分辨率结构是否系统偏离 |

### 2.2 蛋白与配体统计

| 可视化 | 内容 | 目的 |
|--------|------|------|
| Pocket 大小分布 | 原子数直方图 + 残基数直方图 | 了解 pocket 尺寸范围，指导 batch padding |
| 配体大小分布 | 重原子数直方图 | 检查配体复杂度分布 |
| 配体理化性质 | MW、logP、TPSA、HBD/HBA、可旋转键数 | 确认类药性（Rule of 5 覆盖率） |
| 蛋白家族分布 | Top-20 家族的样本数柱状图 | 检查家族是否极端不平衡 |

### 2.3 Split 质量检查

| 可视化 | 内容 | 目的 |
|--------|------|------|
| 家族-split 交叉表 | 热力图：行=家族，列=split | 验证同家族不跨集（对角线结构） |
| 化学空间 t-SNE/UMAP | 配体 fingerprint 降维，按 split 着色 | 检查 split 间化学空间是否有重叠/泄漏 |
| Split 样本数汇总 | 柱状图：各 split 样本数 | 确认比例接近 70/10/10/10 |

### 2.4 数据质量排查

| 检查项 | 方法 | 处理 |
|--------|------|------|
| 缺失配体 | 检查 SDF 文件是否可被 RDKit 解析 | 记录失败 PDB code，排除或修复 |
| 异常 pKd | pKd < 1 或 pKd > 14 的样本 | 标记为 outlier，分析是否保留 |
| 重复配体 | InChI 去重，检查同一配体出现在不同 pocket 的情况 | 记录但不删除（不同 pocket 是合理的） |
| 极小/极大 pocket | 原子数 < 50 或 > 2000 | 检查提取参数是否合理 |

### 2.5 输出

```
results/pdbbind_eda/
├── pKd_distribution.png           # 标签分布
├── pKd_by_split.png               # 按 split 对比
├── affinity_type_pie.png          # 亲和力类型占比
├── resolution_vs_pkd.png          # 分辨率 vs pKd
├── pocket_size_hist.png           # Pocket 大小
├── ligand_size_hist.png           # 配体大小
├── ligand_properties.png          # 理化性质 4 panel
├── family_distribution.png        # 蛋白家族 top-20
├── family_split_heatmap.png       # 家族×split 检查
├── chemical_space_tsne.png        # 化学空间降维
├── split_summary.png              # Split 样本数
├── data_quality_report.csv        # 异常样本汇总
└── eda_summary.json               # 关键统计数字（供下游引用）
```

### 2.6 实现

| 文件 | 用途 |
|------|------|
| `scripts/pipeline/s00b_pdbbind_eda.py` | EDA 主脚本：生成所有可视化和统计报告 |

---

## 3. 下游 Sub-Plan 如何使用此数据集

本数据集是后续所有 Sub-Plan 的**统一训练基础**。各 Sub-Plan 在此数据上训练各自的模块：

| Sub-Plan | 使用方式 |
|----------|----------|
| **01 Multi-Granularity Repr** | 在 pair 数据上构建 interaction graph，训练 interaction GNN |
| **02 Attention Aggregation** | 在 pair 数据上训练 attention pooling 模块 |
| **03 Multi-Layer Fusion** | 在 pair 数据上提取多层 embedding，训练 fusion 权重 |
| **04 Hybrid Predictor (DKL)** | 在 pair 数据上端到端训练 encoder + DKL predictor |
| **05 Multi-Task Learning** | 在 pair 数据上训练 regression + ranking + classification 多头 |
| **06 Physics-Aware Features** | 从 pair 数据中提取物理特征，作为辅助输入 |
| **07 Uncertainty-Guided Gen** | 使用在 pair 数据上训练好的 predictor，迁移到生成分子 |

### 迁移到生成分子

所有模块在 PDBbind 上训练完成后，将 encoder 权重迁移到 TargetDiff 生成分子的评估任务上。迁移策略详见 [07_uncertainty_guided_generation.md](07_uncertainty_guided_generation.md)。

---

## 3. 实现清单

### 3.1 新增文件

| 文件 | 用途 |
|------|------|
| `bayesdiff/pretrain_dataset.py` | PDBbind pair-level 数据集类与 DataLoader |
| `scripts/pipeline/s00_prepare_pdbbind.py` | 数据处理脚本：解析 INDEX、提取 pocket、featurize、划分 |
| `scripts/pipeline/s00b_pdbbind_eda.py` | EDA 脚本：数据可视化、分布分析、质量检查 |

### 3.2 修改文件

| 文件 | 修改 |
|------|------|
| `bayesdiff/data.py` | 扩展 `parse_pdbbind_index()` 支持 v2020 refined set 完整解析 |

### 3.3 里程碑

| 里程碑 | 内容 | 验证标准 |
|--------|------|----------|
| M0.1 | INDEX 解析完成 | ~5,316 条记录，pKd 分布合理（范围 ~2–12） |
| M0.2 | Pocket 提取完成 | 每个复合物有对应 pocket 文件 |
| M0.3 | 数据集划分完成 | 4 个 split，同家族不跨集 |
| M0.4 | DataLoader 可运行 | 能正确 batch 不等长蛋白-配体 pair |
| M0.5 | EDA 完成 | 所有可视化生成，无严重数据质量问题 |

---

## 4. 计算资源估算

| 步骤 | 时间 | 硬件 |
|------|------|------|
| INDEX 解析 + pocket 提取 | 2–4h | CPU |
| Featurize 全部复合物 | 4–8h | CPU |
| 数据集划分（mmseqs2 聚类） | 1–2h | CPU |
| EDA 可视化与质量检查 | 0.5–1h | CPU |
| **合计** | **~7.5–15h** | **CPU** |
