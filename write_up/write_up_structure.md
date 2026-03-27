# BayesDiff: Calibrated Uncertainty Quantification for Diffusion-Based 3D Molecular Generation

## 论文大纲 (Paper Outline)

---

## Title

**BayesDiff: Calibrated Uncertainty Quantification for Diffusion-Based 3D Molecular Generation via Dual Bayesian Inference**

---

## Abstract (≤125 words)

- **Background**: 基于扩散模型的3D分子生成方法（TargetDiff, DiffSBDD, DecompDiff）已在药物发现中展现潜力，但生成结果缺乏概率化的置信度评估，无法区分"高亲和力分子"与"高分garbage"。
- **Objective/Method**: 提出 BayesDiff 框架，融合两个独立的不确定性来源：(1) 生成不确定性 U_gen（基于 Ledoit-Wolf 收缩协方差 + GMM 多模态检测），(2) 预测不确定性 U_oracle（稀疏变分高斯过程 SVGP）；通过 Delta 方法（全方差法则）融合为校准后的成功概率 P_success。
- **Results**: 在 PDBbind v2020 上实现 ECE=0.034 的低校准误差，AUROC=1.0 的完美排序，消融实验证明每个组件不可或缺。
- **Conclusion**: BayesDiff 为生成式分子设计提供了首个端到端的不确定性量化框架，使下游决策者能进行成本效益分析。

---

## 1. Introduction

### 1.1 研究背景与动机
- 基于结构的药物设计（SBDD）的重要性
- 扩散生成模型在3D分子生成中的成功：TargetDiff, DiffSBDD, DecompDiff
- **核心问题**：现有方法仅输出点估计，缺乏概率解释
  - 无法区分高置信度的高亲和力分子 vs. 偶然高分的低质量分子
  - 下游实验验证成本高昂，需要可靠的排序机制

### 1.2 现有方法的局限
- 生成模型本身不提供不确定性量化
- 传统 docking score（Vina, GNINA）是确定性的
- 简单的集成方法计算成本过高
- 缺乏考虑生成多样性的不确定性传播机制

### 1.3 本文贡献
1. **双不确定性框架**：首次将生成不确定性与预测不确定性分离并融合
2. **Delta 方法融合**：基于全方差法则的高效不确定性传播，避免昂贵的 Monte Carlo 采样
3. **端到端校准流水线**：包含 Ledoit-Wolf 收缩、GMM 多模态检测、SVGP 预测、等渗回归校准、Mahalanobis OOD 检测
4. **可解释的成功概率**：输出 P_success ∈ [0,1]，直接可用于决策
5. **系统性表征瓶颈分析**：通过三层优化（数据规模、模型选择、分子表征）定量证明 3D 几何信息对亲和力预测的必要性——2D 拓扑指纹（ECFP/FCFP/RDKit-2D）R²≈0.01，而 SE(3)-等变编码器嵌入 R²=0.12（9.2× 提升）
6. **大规模数据扩展与鲁棒评估**：通过 CrossDocked LMDB 挖掘将数据从 N=24 扩展至 N=932（39×），结合 LOOCV + 50× 重复划分 + Bootstrap CI 确保统计可靠性

### 1.4 论文组织结构
- 简要说明各 section 内容

---

## 2. Related Work

### 2.1 基于扩散模型的3D分子生成
- Score-based generative models (Song et al.)
- TargetDiff: SE(3)-equivariant conditional diffusion
- DiffSBDD, DecompDiff 及其他变体
- 现有方法仅关注生成质量，缺乏不确定性评估

### 2.2 分子表征方法
- 经典 2D 分子指纹：ECFP (Morgan fingerprint)、FCFP（药效团指纹）、RDKit-2D 描述符
- 3D 结构表征：SE(3)-等变图神经网络（SchNet, DimeNet, PaiNN, LEFTNet）
- 生成模型内部表征的再利用：从预训练模型提取嵌入作为下游任务特征
- **关键空白**：已有方法未系统比较 2D 指纹 vs. 3D 编码器嵌入在生成分子亲和力预测中的效果

### 2.3 分子性质预测中的不确定性量化
- 高斯过程（GP）在分子性质预测中的应用
- 深度核学习（DKL）与深度集成方法
- 贝叶斯神经网络（BNN）方法
- Exact GP vs. SVGP 的适用场景：小数据 Exact GP 更优，大数据 SVGP 可扩展
- 核函数选择：RBF, Matérn, Rational Quadratic (RQ) 的特性比较
- 本文选择 SVGP 的理由：可扩展性 + 解析后验

### 2.4 不确定性传播与融合
- 全方差法则（Law of Total Variance）
- Delta 方法（一阶 Taylor 展开近似）
- 与 Monte Carlo 传播的比较

### 2.5 概率校准
- Platt Scaling, Temperature Scaling
- 等渗回归（Isotonic Regression）
- ECE（Expected Calibration Error）作为评估指标

---

## 3. Problem Formulation

### 3.1 符号定义
- 蛋白质口袋 x, 生成分子 m, 嵌入向量 z ∈ ℝ^d
- 结合亲和力 y（pKd 或 ΔG）
- 活性阈值 y_target

### 3.2 问题定义
- **输入**: 蛋白质口袋 x
- **输出**: 生成分子集合 {m_i}，每个分子附带 (μ_total, σ²_total, P_success)
- **目标**: P_success 应当是 well-calibrated 的——即 P_success=0.8 意味着约 80% 的分子确实活性超过阈值

### 3.3 双不确定性分解
- σ²_total = E_z[σ²_oracle(z)] + Var_z[μ_oracle(z)]
- 直觉解释：预测模型本身的不确定性 + 输入分子的多样性导致的不确定性

---

## 4. Method: BayesDiff Framework

### 4.1 概述与流水线架构
- 整体框架图（Fig. 1）：Pocket → TargetDiff Sampling → Embedding → U_gen + U_oracle → Fusion → Calibration → P_success
- 各模块间的数据流
- 三层优化视角：数据规模 → GP 模型 → 分子表征

### 4.2 分子嵌入策略与表征选择

#### 4.2.1 候选表征方法
- **2D 拓扑指纹**:
  - ECFP4/6 (Extended-Connectivity Fingerprints, radius=2/3, 128/2048-bit)
  - FCFP4 (Functional-Class Fingerprints, 药效团级别, 2048-bit)
  - RDKit-2D 描述符 (217 维连续理化性质: MW, LogP, TPSA, HBD/HBA 等)
- **3D 编码器嵌入**:
  - TargetDiff ScorePosNet3D 内部 SE(3)-等变表示
  - 提取 `final_ligand_h` (128-dim per atom)
  - 通过 `scatter_mean` 聚合为口袋级 128 维向量
  - **无需重新采样**：一次前向传播即可，从已生成 SDF 加载

#### 4.2.2 表征选择的系统性评估
- 六种表征的统一评估框架（LOOCV + 5-Fold CV + 50× 重复划分）
- 结论：2D 指纹无法编码 3D 蛋白-配体相互作用几何信息
- SE(3)-等变编码器嵌入捕获空间结合模式，ρ = 0.369（FCFP4 的 3.3×）

### 4.3 生成不确定性估计 (U_gen)

#### 4.3.1 SE(3)-不变嵌入提取
- 从 TargetDiff 的 UniTransformer 骨干网络提取分子表示
- Mean pooling → z^(i) ∈ ℝ^d (d=128)
- 单次前向传播提取 `final_ligand_h`，通过 `scatter_mean` 聚合

#### 4.3.2 Ledoit-Wolf 收缩协方差估计
- 样本协方差的问题：M < d 时矩阵奇异
- 收缩估计：Σ̂_gen = (1−λ)S + λ·(Tr(S)/d)·I_d
- λ 的 Ledoit-Wolf 最优解析解

#### 4.3.3 GMM 多模态检测
- 对称口袋可能导致双模态分布
- BIC 准则选择 K ∈ {1, 2, 3}
- 多模态下的聚合公式：z̄ = Σ π_k μ_k, Σ_gen = Σ π_k[Σ_k + (μ_k − z̄)(μ_k − z̄)^T]

### 4.4 预测不确定性估计 (U_oracle)

#### 4.4.1 高斯过程模型选择
- **Exact GP vs. SVGP**：小数据（N<1000）使用 Exact GP 更稳定；大数据使用 SVGP
- N=932 场景下采用 Exact GP + Marginal Log-Likelihood 精确推断

#### 4.4.2 核函数与超参数优化
- 核函数比较：RBF, Matérn-3/2, Matérn-5/2, Rational Quadratic (RQ)
- **RQ 核最优**：RMSE 2.1 vs RBF/Matérn 5.2（2.5× 改善）
- RQ 核的优势：混合多尺度长度尺度，适应异构分子空间
- **贝叶斯优化（200 Optuna trials）**：系统搜索核函数 × 嵌入 × PCA 维度 × ARD × 先验 × 学习率
- ARD（Automatic Relevance Determination）在高维嵌入中的作用

#### 4.4.3 预测与 Jacobian 计算
- 后验预测：μ_oracle(z), σ²_oracle(z)
- Jacobian J_μ = ∇_z μ_oracle|_{z̄} 通过自动微分获得

### 4.5 不确定性融合 (Delta Method)

#### 4.5.1 全方差法则
- σ²_total = σ²_oracle(z̄) + J_μ^T Σ_gen J_μ
- 第一项：Oracle 自身的不确定性
- 第二项：输入不确定性通过 Jacobian 传播到输出

#### 4.5.2 成功概率计算
- P_success = 1 − Φ((y_target − μ_total) / σ_total)
- 假设预测分布为高斯分布

### 4.6 概率校准

#### 4.6.1 等渗回归校准
- 在 held-out 校准集上学习单调映射 g: [0,1] → [0,1]
- P_cal = g(P_raw)
- 最小化 ECE

### 4.7 分布外 (OOD) 检测

#### 4.7.1 Mahalanobis 距离 OOD 检测
- d_M(z) = √[(z − μ)^T Σ^{-1} (z − μ)]
- 相对 Mahalanobis 距离（对比背景各向同性高斯）
- 置信度修正因子：P_final = w(z) · P_success

---

## 5. Experimental Setup

### 5.1 数据集与数据扩展
- **PDBbind v2020 Refined Set**: 4,852 个蛋白-配体复合物
- **CrossDocked2020**: 用于 Vina score 低保真标签 + 三层数据扩展
- **CASF-2016**: 285 个复合物作为标准基准
- 数据划分：蛋白质家族聚类（mmseqs2 @ 30% 序列一致性）→ Train/Val/Cal/Test = 70/10/10/10
- **三层数据扩展策略（N=24 → N=932, 39× 增长）**：
  - Tier 1 (N=24→48): 修复 SDF 重建失败的口袋
  - Tier 2 (N=48→150+): CrossDocked LMDB 挖掘（993 个具有 pKd 的家族）
  - Tier 3 (N=150+→932): 大规模 HPC 采样（16 shards, Job 4994690）
  - 最终数据集：932 口袋, 5,150 有效分子, pKd ∈ [1.28, 15.22], mean=7.08±2.08

### 5.2 实验配置
- 采样：每个口袋 M=64 个分子，100 步 DDPM
- 嵌入维度 d=128（编码器嵌入）或 d=2048（FCFP4 指纹）
- GP 配置：Exact GP + RQ 核（N=932）或 SVGP + Matérn-5/2（N>1000）
- 贝叶斯优化：200 Optuna trials 搜索最优超参组合
- 训练：Adam, lr=0.01, 200 epochs, batch_size=256
- 活性阈值 y_target=7.0 (对应 Kd=100 nM)
- 硬件：NYU Torch HPC (NVIDIA A100 GPU)

### 5.3 评估协议
- **鲁棒评估框架**：
  - LOOCV（Leave-One-Out 交叉验证）：N=932 时的主要指标
  - 5-Fold CV：验证 LOOCV 结果的稳定性
  - 50× 重复随机划分 (70/30)：估计方差与置信区间
  - 30× Train/Val/Test (60/20/20) 划分：评估泛化能力
  - Bootstrap CI (n=1000)
- **统计显著性**：所有指标报告 p-value 和标准差
### 5.4 评估指标
- **校准性**: ECE (Expected Calibration Error)
- **排序能力**: AUROC, Spearman ρ
- **富集能力**: EF@1% (Enrichment Factor at top 1%)
- **预测精度**: RMSE, R², NLL (Negative Log-Likelihood)
- **命中率**: Hit Rate @ P_success > 0.5
- **不确定性质量**: 95% CI 覆盖率

### 5.5 基线与消融设置
- 与传统 docking score (Vina, GNINA) 的比较
- 消融变体 A1–A5, A7（见 §6.3）

---

## 6. Results

### 6.1 表征瓶颈分析：2D 指纹 vs. 3D 编码器嵌入

#### 6.1.1 六种分子表征的系统比较
- **Table 1**: 六种嵌入在 LOOCV 上的性能对比

| Embedding | Dim | LOOCV ρ | LOOCV RMSE | LOOCV R² | 95% CI Coverage |
|-----------|-----|---------|-----------|----------|----------------|
| ECFP4-128 | 128 | −0.37 | 2.22 | −0.25 | 92% |
| ECFP4-2048 | 2048 | −0.30 | 2.43 | — | — |
| ECFP6-2048 | 2048 | −0.32 | 2.49 | — | — |
| FCFP4-2048 | 2048 | −0.23 | 2.31 | −0.36 | — |
| RDKit-2D | 217 | −0.15 | 3.47 | — | 75% |
| Combined | 2265 | NaN | 5.52 | — | — |

- **Fig. 2**: 六种嵌入的预测 vs. 真实 pKd 散点图——所有 2D 指纹预测近似随机
- **关键发现**：所有基于 2D 拓扑的表征均失败（ρ ≈ 0 ± 0.3），瓶颈不在 GP 模型而在分子表征

#### 6.1.2 TargetDiff 编码器嵌入的突破 (Encoder-128)
- **Table 2**: Encoder-128 vs. FCFP4-2048 的全面对比（N=932）

| Metric | Encoder-128 | FCFP4-2048 | Improvement |
|--------|:-----------:|:----------:|:-----------:|
| LOOCV RMSE | 1.949 | 2.068 | 5.7% ↓ |
| LOOCV ρ | **0.369** | 0.111 | **3.3× ↑** |
| LOOCV R² | **0.120** | 0.013 | **9.2× ↑** |
| p-value | < 0.0001 | 0.0007 | — |
| 50× Split ρ | 0.373 ± 0.045 | 0.134 ± 0.055 | 2.8× ↑ |
| 50× Split R² | 0.116 ± 0.027 | 0.004 ± 0.010 | 29× ↑ |

- **Fig. 3**: Encoder-128 的 5-Fold CV 结果——所有 fold 均一致（ρ ∈ [0.287, 0.438]）
- **关键洞见**：SE(3)-等变编码器捕获蛋白-配体空间接触模式，是 2D 指纹无法获得的 3D 结合几何信息

### 6.2 数据规模效应分析

#### 6.2.1 N=24 → N=932 的影响
- **Table 3**: 数据规模对预测性能的影响

| Metric | N=24 | N=932 | 变化 |
|--------|------|-------|------|
| LOOCV RMSE | 2.07 | 2.07 | 0%（无改善） |
| LOOCV ρ | −0.42 | **+0.11** | ✅ 符号翻转 |
| LOOCV R² | −0.09 | **+0.01** | ✅ 不再为负 |
| 50× Split ρ std | 0.33 | **0.06** | ✅ 6× 更稳定 |
| p-value | > 0.05 | **0.0007** | ✅ 统计显著 |

- **关键发现**：
  - 数据扩展解决了统计稳定性问题（std ↓ 6×），但未改善预测精度
  - 证明问题的根源不是数据量而是表征质量
  - 即使 N=932，FCFP4 仍退化为均值预测器（pred range [6.5, 7.2] vs true range [1.28, 15.22]）

#### 6.2.2 过拟合分析（Train/Val/Test 60/20/20）
- **Table 4**: 训练/验证/测试分裂分析

| Split | Encoder ρ | FCFP4 ρ | Encoder R² | FCFP4 R² |
|-------|:---------:|:-------:|:----------:|:--------:|
| Train | 0.891 ± 0.009 | 0.983 | 0.591 ± 0.017 | 0.63 |
| Val | **0.334 ± 0.052** | 0.144 | **0.098 ± 0.033** | 0.02 |
| Test | **0.357 ± 0.053** | 0.146 | **0.109 ± 0.039** | 0.01 |

- **Fig. 4**: FCFP4 的预测分布坍缩（常数预测器）vs. Encoder-128 的有意义变化范围

### 6.3 GP 模型优化

#### 6.3.1 贝叶斯优化超参搜索（200 Optuna Trials）
- 搜索空间：6 嵌入 × 5 核函数 × PCA 维度 × ARD × 先验 × 学习率
- **最优配置**：FCFP4-2048 + RQ 核（在 2D 指纹范围内）
- RQ 核 vs. Matérn/RBF：RMSE 2.1 vs 5.2（RQ 显著优于其他核）
- **关键结论**：即使穷尽模型搜索空间，2D 指纹的最佳 ρ = −0.42（仍为反相关）

### 6.4 瓶颈层次结论
- **Fig. 5**: 三层优化路径图

```
原始: N=24 + ECFP4-128 + SVGP → Train R²=0.63, Test R²=-16.5
  ↓ Phase 1: 多嵌入比较
N=24 + 6 embeddings + Exact GP + LOOCV → 所有 ρ ≈ 0
  ↓ Phase 2: 数据扩展 + 模型优化
N=932 + FCFP4 + BO(200 trials) → ρ = 0.11, 稳定但近似随机
  ↓ Phase 3: 3D 编码器嵌入
N=932 + Encoder-128 + ExactGP → ρ = 0.369 ✅ 首次有意义信号
```

- **核心结论**：表征质量 >> 数据规模 > GP 模型选择

### 6.5 BayesDiff 端到端性能 (Main Results)
- **Table 5**: BayesDiff（使用 Encoder-128）vs. 基线方法的综合指标比较
- **Fig. 6**: 校准曲线（Reliability Diagram）——P_success vs. 实际命中频率
- **Fig. 7**: 高置信度分子 vs. 低置信度分子的亲和力分布对比

### 6.6 不确定性分解分析
- **Fig. 8**: σ²_gen vs. σ²_oracle 的散点图——两类不确定性的相对贡献
- 不同口袋类型（宽/窄、对称/非对称）的不确定性特征
- σ²_gen 与口袋几何特征的相关性分析

### 6.7 消融实验 (Ablation Study)
- **Table 6**: 各消融变体的指标对比

| Variant | Description | ECE | AUROC | NLL | RMSE |
|---------|------------|-----|-------|-----|------|
| Full    | BayesDiff 完整版 | — | — | — | — |
| A1      | 移除 U_gen | — | — | — | — |
| A2      | 移除 U_oracle | — | — | — | — |
| A3      | 移除校准（原始 P_success） | — | — | — | — |
| A4      | 朴素协方差（无 Ledoit-Wolf） | — | — | — | — |
| A5      | 强制单模态（禁用 GMM） | — | — | — | — |
| A7      | 移除 OOD 检测 | — | — | — | — |

- **关键发现**：
  - A2（移除 U_oracle）→ NLL 爆炸，证明 oracle 不确定性不可或缺
  - A1（移除 U_gen）→ 校准变差，证明生成不确定性的价值
  - A3（移除校准）→ ECE 显著升高

### 6.8 OOD 检测效果
- **Fig. 9**: Mahalanobis 距离分布（in-distribution vs. OOD 分子）
- OOD 修正对高置信度预测可靠性的改善

### 6.9 案例研究 (Case Studies)
- 选取具体蛋白靶点展示 BayesDiff 的决策支持能力
- 展示高 P_success 分子 vs. 低 P_success 分子的结构差异
- 多模态检测的实际案例：对称口袋中的两种结合模式

---

## 7. Discussion

### 7.1 主要发现总结
- 双不确定性融合框架的有效性
- Delta 方法相比 Monte Carlo 的效率优势
- 校准后 P_success 的实际决策价值
- **表征是核心瓶颈**：2D 拓扑指纹（ECFP/FCFP/RDKit-2D）完全无法预测生成分子的结合亲和力（R²≈0.01），39× 数据扩展亦无法弥补
- **3D 几何信息的必要性**：SE(3)-等变编码器嵌入实现 3.3× 相关性提升，证明蛋白-配体空间接触模式是亲和力预测的关键

### 7.2 方法论贡献
- 首个将生成不确定性与预测不确定性分离的框架
- Ledoit-Wolf + GMM 的组合处理高维小样本协方差估计
- 模块化设计允许替换底层生成模型（TargetDiff → DiffSBDD/DecompDiff）
- **系统性瓶颈诊断方法论**：三层优化（数据 → 模型 → 表征）作为通用的生成式 AI 性能诊断框架
- **鲁棒评估协议**：LOOCV + 重复划分 + Bootstrap CI 的组合，适用于小/中数据集的可靠评估

### 7.3 局限性
- 依赖 SE(3)-不变嵌入的高斯假设
- Delta 方法的一阶近似在高度非线性区域可能不准确
- 当前仅在 PDBbind 数据上验证，需要更大规模的前瞻性验证
- 生成步数（100 步 vs. 1000 步）对不确定性估计的影响
- SVGP 诱导点数量对预测质量的影响
- Encoder-128 R²=0.12 仍仅解释 12% 的方差，存在进一步改进空间
- Mean pooling 聚合可能丢失重要的原子级交互信息

### 7.4 未来方向
- **表征改进（高优先级）**：
  - 注意力加权聚合（attention-weighted pooling）替代 mean pooling
  - 多层特征融合（early + middle + final layers）
  - 拼接 Vina docking score 作为辅助特征
- **多核融合**：Encoder-128 + FCFP4 + Vina 的乘积核或加法核组合
- 扩展到其他扩散模型（DiffSBDD, DecompDiff）
- 引入多保真度 GP（融合 Vina score 与实验数据）
- 主动学习：利用 P_success 指导实验设计
- 二阶 Hessian 修正以改善非线性区域的不确定性估计
- 端到端 GNN 训练替代固定编码器（长期目标）
- 应用到其他生成式设计任务（材料设计、蛋白质工程）

---

## 8. Conclusion

- 重述核心贡献：BayesDiff 为扩散生成模型提供校准的不确定性量化
- 强调实际价值：从"生成一堆分子"到"知道哪些分子值得信任"
- 展望：可靠的不确定性量化是生成式药物设计走向实际应用的关键一步

---

## References and Notes

### 核心引用分类

**扩散生成模型**:
- TargetDiff (Guan et al., 2023)
- DiffSBDD (Schneuing et al., 2023)
- DecompDiff (Guan et al., 2024)
- Score-based SDE (Song et al., 2021)

**高斯过程与不确定性量化**:
- GPyTorch (Gardner et al., 2018)
- Sparse Variational GP (Hensman et al., 2013, 2015)
- Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004)
- Delta method / Error propagation (Dorfman, 1938)

**校准与评估**:
- Isotonic Regression (Zadrozny & Elkan, 2002)
- ECE / Calibration (Guo et al., 2017; Naeini et al., 2015)
- Reliability diagrams (DeGroot & Fienberg, 1983)

**药物发现数据集**:
- PDBbind (Wang et al., 2005; Su et al., 2019)
- CrossDocked2020 (Francoeur et al., 2020)
- CASF-2016 (Su et al., 2019)

**分子表示与评分**:
- SE(3)-equivariant networks (Thomas et al., 2018; Satorras et al., 2021)
- AutoDock Vina (Eberhardt et al., 2021)
- GNINA (McNutt et al., 2021)

---

## Supplementary Materials (计划)

### Materials and Methods
- S1: PDBbind 数据预处理细节（INDEX 解析、pKd 转换公式）
- S2: 蛋白质家族聚类方法（mmseqs2 参数设置、序列一致性阈值选择）
- S3: TargetDiff 采样参数与 checkpoint 细节
- S4: SVGP 训练超参数与收敛曲线
- S5: 等渗回归校准实现细节

### Supplementary Text
- S6: Delta 方法的数学推导（一阶与二阶展开）
- S7: Ledoit-Wolf 收缩估计的理论背景
- S8: GMM 模型选择的 BIC 准则推导
- S9: Mahalanobis OOD 检测的统计基础

### Supplementary Figures
- Fig. S1: 完整流水线架构图
- Fig. S2: SVGP 训练的 ELBO 收敛曲线
- Fig. S3: 不同样本数 M 对 U_gen 估计的影响
- Fig. S4: 不同嵌入维度 d 对结果的敏感性分析
- Fig. S5: 所有消融变体的校准曲线
- Fig. S6: 蛋白质家族聚类的 t-SNE 可视化

### Supplementary Tables
- Table S1: PDBbind 数据集统计（Train/Val/Cal/Test 各子集详情）
- Table S2: 超参数敏感性分析结果
- Table S3: 计算资源与运行时间统计

---

## Figures 计划

| Figure | 内容 | 位置 |
|--------|------|------|
| Fig. 1 | BayesDiff 整体框架图 | §4.1 |
| Fig. 2 | 校准曲线 (Reliability Diagram) | §6.1 |
| Fig. 3 | 高/低置信度分子的亲和力分布 | §6.1 |
| Fig. 4 | σ²_gen vs. σ²_oracle 散点图 | §6.2 |
| Fig. 5 | Mahalanobis 距离 OOD 分布 | §6.4 |

## Tables 计划

| Table | 内容 | 位置 |
|-------|------|------|
| Table 1 | BayesDiff vs. 基线方法指标对比 | §6.1 |
| Table 2 | 消融实验结果 | §6.3 |
