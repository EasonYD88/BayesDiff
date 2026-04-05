# BayesDiff: 面向 3D 分子生成的双重不确定性感知框架

> **核心命题**：现有基于扩散模型的 3D 分子生成方法（TargetDiff, DiffSBDD, DecompDiff 等）输出的对接分数缺乏概率语义——它们无法区分"高置信高亲和力"与"恰好落在评分函数盲区的高分垃圾"。BayesDiff 通过一套端到端的双重不确定性量化框架，为每个生成分子赋予经校准的成功概率 $P_{success}$，从而在生成阶段就完成可靠的风险分层。

---

## 0. 问题定义与现有方法的缺陷

**现状**：条件扩散模型 $p_\theta(x | c)$ 可以在蛋白口袋 $c$ 中生成 3D 配体 $x$，随后用 Vina / GNINA 等打分函数做 post-hoc 排序。这一流程存在三层系统性风险：

| 风险类型 | 描述 | 后果 |
|---------|------|------|
| **生成多样性幻觉** | 同一条件下多次采样结构差异极大，但打分函数只取最优 | 优选出物理上不稳定的低概率构象 |
| **打分外推偏差** | 候选分子落在训练集化学空间之外，打分函数外推失效 | "高分垃圾"——分数高但实验验证失败 |
| **缺乏置信度量化** | 最终输出只有点估计，无概率区间 | 下游决策者无法做成本-收益分析 |

**BayesDiff 的回答**：将生成不确定性（$U_{gen}$）与预测不确定性（$U_{oracle}$）在同一概率框架下严格融合，输出经校准的 $P_{success}$。

---

## 1. 系统架构

### 1.1 数据集定义

| 数据集 | 用途 | 来源 | 规模 |
|--------|------|------|------|
| $\mathcal{D}_{diff}$ | 训练条件扩散模型 | CrossDocked2020（~22.5M poses）或 PDBbind v2020 refined set（~5K complexes） | 大规模 |
| $\mathcal{D}_{oracle}^{(l)}$ | GP 初始训练（低精度层） | AutoDock Vina batch rescoring | ~10K-50K |
| $\mathcal{D}_{oracle}^{(h)}$ | GP 精调（高精度层） | MM-GBSA 或实验 $K_d$ / $\text{IC}_{50}$ | ~500-2K |
| $\mathcal{D}_{cal}$ | 置信度校准（held-out） | 上述混合，按 scaffold split | ~1K |

> **关键设计**：采用 **多精度层级 (Multi-fidelity)** oracle 数据，而非单一 Vina 评分。低精度数据提供覆盖度，高精度数据修正系统偏差。GP 多精度核可自然处理此结构（详见 §3）。

### 1.2 输入

| 符号 | 含义 | 表示 |
|------|------|------|
| $c_{pocket}$ | 靶点结合口袋 | 残基原子 3D 坐标 + 元素类型 |
| $c_{scaffold}$ | （可选）核心骨架约束 | SMILES 或 3D 子结构坐标 |
| $y_{target}$ | 目标属性阈值 | 例如 $\Delta G_{bind} \le -9.0$ kcal/mol |

### 1.3 输出

对每个候选分子 $x_{gen}$，输出一个三元组：

$$\text{Output} = \left( x_{gen},\; \mu_{total},\; P_{success} \right)$$

其中 $\mu_{total}$ 为期望属性值，$P_{success} = P(y \le y_{target} \mid x_{gen}, c)$ 为经校准的成功概率。

---

## 2. 生成不确定性量化 ($U_{gen}$)

### 2.1 蒙特卡洛后验采样

扩散模型的逆过程本质是一个条件随机微分方程。给定 $c = (c_{pocket}, c_{scaffold})$，执行 $M$ 次独立采样：

$$\{x^{(1)}, x^{(2)}, \dots, x^{(M)}\} \sim p_\theta(x \mid c), \quad M \ge 32$$

> **$M$ 的选取**：$M$ 需足够大以稳定估计协方差矩阵。对 $d$ 维潜空间，经验法则要求 $M \gg d$；实践中 $M \in [32, 128]$，配合 shrinkage 估计器（见 §2.3）即可。

### 2.2 SE(3)-不变潜空间映射

3D 分子坐标存在旋转、平移、原子排列的不变性，无法直接做统计。我们使用预训练的 **SE(3)-等变图神经网络** $E_\phi$ 将其映射到几何不变的潜空间 $\mathcal{Z} \subseteq \mathbb{R}^d$：

$$z^{(m)} = E_\phi(x^{(m)}), \quad m = 1, \dots, M$$

**编码器选型**（按推荐优先级）：

| 模型 | 等变类型 | 优势 | 参考 |
|------|---------|------|------|
| PaiNN | SE(3)-等变 | 标量/向量双通道，表达力强，效率高 | Schütt et al., ICML 2021 |
| EGNN | E(n)-等变 | 实现简洁，无球谐计算 | Satorras et al., ICML 2021 |
| LEFTNet | SE(3)-等变 | 专为分子属性预测设计，frame 机制 | Du et al., NeurIPS 2023 |

**潜空间维度 $d$ 的选取**：GP 在高维空间中性能急剧退化（核函数趋于常数）。实践中：
- $d \in [32, 64]$：适合标准 GP，推荐作为默认值。
- $d > 128$：必须使用 Deep Kernel Learning (DKL) 或先做非线性降维。

### 2.3 潜空间统计量

**质心（生成期望）**：

$$\bar{z} = \frac{1}{M} \sum_{m=1}^M z^{(m)}$$

**协方差矩阵（生成散度）**——使用 **Ledoit-Wolf shrinkage 估计器** 替代朴素样本协方差，避免小样本下矩阵病态：

$$\hat{\Sigma}_{gen} = (1 - \alpha) \hat{\Sigma}_{sample} + \alpha \cdot \frac{\text{tr}(\hat{\Sigma}_{sample})}{d} \cdot I_d$$

其中 $\alpha \in [0, 1]$ 由 Ledoit-Wolf 准则自动确定，$\hat{\Sigma}_{sample}$ 为标准无偏样本协方差。

**生成不确定性标量化**（用于快速筛选）：

$$U_{gen} = \text{tr}(\hat{\Sigma}_{gen}) = \sum_{i=1}^{d} \lambda_i$$

> **物理意义**：$U_{gen}$ 大 → 模型对该口袋条件下的生成"犹豫不决"，多次采样结构散布广泛 → 可能处于自由能面的平坦区或多极小值区。

### 2.4 多模态检测（超越高斯假设）

**原始方案的关键缺陷**：直接假设 $\{z^{(m)}\}$ 服从单一高斯分布。但扩散模型在某些口袋中可能生成双模态或多模态分布（例如对称口袋中配体的两种等价朝向），此时单一高斯假设严重失效，会高估 $U_{gen}$。

**修正方案**：在计算统计量之前，先用 Gaussian Mixture Model (GMM) 做模态检测：

1. 用 BIC 准则在 $K \in \{1, 2, 3\}$ 间选择最优混合分量数。
2. 若 $K > 1$，对每个模态 $k$ 单独计算 $(\bar{z}_k, \hat{\Sigma}_{gen,k})$。
3. 下游 GP 评估对每个模态独立进行，最终取**模态加权的置信度**：

$$P_{success} = \sum_{k=1}^{K} \pi_k \cdot P_{success}^{(k)}$$

其中 $\pi_k$ 为 GMM 混合权重。

---

## 3. 预测不确定性量化 ($U_{oracle}$)

### 3.1 稀疏变分高斯过程 (SVGP)

标准 GP 的 $O(N^3)$ 复杂度在 $|\mathcal{D}_{oracle}| > 5\text{K}$ 时不可接受。采用 **Sparse Variational GP (SVGP)** with $J$ 个诱导点：

$$f(z) \sim \mathcal{GP}\left(0, k_\theta(z, z')\right), \quad \text{with inducing variables } \mathbf{u} = f(\mathbf{Z}_J)$$

训练目标为 ELBO：

$$\mathcal{L}_{SVGP} = \sum_{i=1}^{N} \mathbb{E}_{q(f_i)} \left[ \log p(y_i | f_i) \right] - \text{KL}\left[ q(\mathbf{u}) \| p(\mathbf{u}) \right]$$

诱导点数 $J \in [256, 1024]$，复杂度降至 $O(NJ^2)$。使用 GPyTorch 的 `VariationalStrategy` 实现。

### 3.2 多精度核 (Multi-fidelity Kernel)

为联合建模低精度（Vina）和高精度（MM-GBSA / 实验）数据，采用 **Intrinsic Coregionalization Model (ICM)**：

$$k_{MF}((z, t), (z', t')) = k_{base}(z, z') \cdot B_{tt'}$$

其中 $t \in \{0, 1\}$ 是精度层级指示变量，$B \in \mathbb{R}^{2 \times 2}$ 为可学习的共区域化矩阵。这允许低精度数据的信号自动传递到高精度预测中，同时学习两者间的系统偏差。

### 3.3 核函数设计

基础核 $k_{base}$ 采用 **Matérn-5/2 + Automatic Relevance Determination (ARD)**：

$$k_{M52}(z, z') = \sigma_f^2 \left(1 + \sqrt{5}r + \frac{5}{3}r^2\right) \exp(-\sqrt{5}r), \quad r = \sqrt{\sum_{i=1}^d \frac{(z_i - z_i')^2}{\ell_i^2}}$$

ARD 中每维有独立长度尺度 $\ell_i$，可自动下压无信息维度。

### 3.4 后验预测

SVGP 的近似后验分布为：

$$q(f(\bar{z})) = \mathcal{N}\left(\mu_{oracle}(\bar{z}),\; \sigma^2_{oracle}(\bar{z})\right)$$

- $\mu_{oracle}(\bar{z})$：在生成质心处的属性期望值
- $\sigma^2_{oracle}(\bar{z})$：在该区域的**认知不确定性 (epistemic uncertainty)**

> **物理意义**：$\sigma^2_{oracle}$ 大 → 该分子落在 GP 训练数据稀疏的化学空间区域 → 预测不可信，属于**外推**。

### 3.5 显式 OOD 检测层

GP 方差在极端外推时增长缓慢（核函数趋于常数）。增加一层**显式分布外 (OOD) 检测**作为 hard gate：

$$d_{Maha}(\bar{z}) = \sqrt{(\bar{z} - \mu_{train})^T \Sigma_{train}^{-1} (\bar{z} - \mu_{train})}$$

其中 $\mu_{train}$, $\Sigma_{train}$ 为训练集在潜空间的均值和协方差。若 $d_{Maha}(\bar{z}) > \tau_{OOD}$（阈值由 $\chi^2_d$ 分布的 99% 分位数确定），直接标记为 `UNRELIABLE`，不输出 $P_{success}$。

---

## 4. 双重不确定性融合

### 4.1 全方差公式 + Delta Method（一阶近似）

目标：求输入不确定 $z \sim q(z)$ 时，$f(z)$ 的总方差。由全方差公式：

$$\text{Var}[y] = \underbrace{\mathbb{E}_z[\text{Var}[y|z]]}_{\text{平均认知不确定性}} + \underbrace{\text{Var}_z[\mathbb{E}[y|z]]}_{\text{生成不确定性传播}}$$

对 GP 后验均值 $\mu_{oracle}(z)$ 在 $\bar{z}$ 处做一阶 Taylor 展开（Delta Method）：

$$\mu_{oracle}(z) \approx \mu_{oracle}(\bar{z}) + J_\mu \cdot (z - \bar{z}), \quad J_\mu = \nabla_z \mu_{oracle}(\bar{z}) \in \mathbb{R}^d$$

代入得一阶融合公式：

$$\boxed{\sigma^2_{total} \approx \underbrace{\sigma^2_{oracle}(\bar{z})}_{\text{GP 认知不确定性}} + \underbrace{J_\mu^T \hat{\Sigma}_{gen} J_\mu}_{\text{生成不确定性经梯度传播}}}$$

**公式解读**：
- 第一项：代理模型在该区域的知识盲区。
- 第二项：生成构象散布（$\hat{\Sigma}_{gen}$）经属性敏感度梯度（$J_\mu$）放大后的贡献。即使 GP 自身很确定，如果生成结构不稳定且属性对该不稳定性敏感，总置信度仍会崩溃 → **"高分垃圾"的精确数学杀手**。

### 4.2 二阶修正项（当 $U_{gen}$ 较大时）

一阶 Delta Method 在 $\hat{\Sigma}_{gen}$ 较大时（$\text{tr}(\hat{\Sigma}_{gen}) > \tau_{nonlinear}$）精度不够。补充二阶修正：

$$\sigma^2_{total,2nd} = \sigma^2_{total,1st} + \frac{1}{2} \text{tr}\left( H_\mu \hat{\Sigma}_{gen} H_\mu \hat{\Sigma}_{gen} \right)$$

其中 $H_\mu = \nabla^2_z \mu_{oracle}(\bar{z}) \in \mathbb{R}^{d \times d}$ 为 GP 均值函数的 Hessian 矩阵。

> **实践建议**：Hessian 计算成本为 $O(d^2)$，在 $d=64$ 时完全可行。通过 `torch.autograd.functional.hessian` 或 Hessian-vector product 高效实现。

### 4.3 蒙特卡洛积分回退方案

当分布高度非高斯（GMM 检测到 $K > 1$）或 $U_{gen}$ 极大时，Delta Method 近似不再可靠。退化到全蒙特卡洛积分：

$$\sigma^2_{total,MC} = \frac{1}{M} \sum_{m=1}^M \left[ \sigma^2_{oracle}(z^{(m)}) + \left(\mu_{oracle}(z^{(m)}) - \bar{\mu}\right)^2 \right]$$

其中 $\bar{\mu} = \frac{1}{M} \sum_{m} \mu_{oracle}(z^{(m)})$。

这是全方差公式的无偏 MC 估计，无需任何分布假设，但计算量为 $M$ 次 GP 推理。

**自适应策略**：

$$\sigma^2_{total} = \begin{cases} \sigma^2_{total,1st} & \text{if } U_{gen} < \tau_{low} \text{ and } K=1 \\ \sigma^2_{total,2nd} & \text{if } \tau_{low} \le U_{gen} < \tau_{high} \text{ and } K=1 \\ \sigma^2_{total,MC} & \text{if } U_{gen} \ge \tau_{high} \text{ or } K > 1 \end{cases}$$

---

## 5. 置信度输出与校准

### 5.1 原始置信度

属性服从近似正态分布 $y \sim \mathcal{N}(\mu_{oracle}(\bar{z}),\; \sigma^2_{total})$，原始成功概率为：

$$P_{success}^{raw} = \Phi\left(\frac{y_{target} - \mu_{oracle}(\bar{z})}{\sigma_{total}}\right)$$

其中 $\Phi$ 为标准正态 CDF。

### 5.2 置信度校准（关键步骤！）

**原始方案的严重缺陷**：直接输出 $P_{success}^{raw}$ 隐含假设高斯近似完美且 GP 后验完美校准——这几乎不可能成立。未校准的概率在实际决策中比点估计更危险（虚假的精确感）。

**校准方案**：在 held-out 校准集 $\mathcal{D}_{cal}$ 上做 **Isotonic Regression** 后校准：

1. 对 $\mathcal{D}_{cal}$ 中每个分子计算 $P_{success}^{raw}$。
2. 获取真实标签 $y_i^{true}$（来自高精度计算/实验），判断 $s_i = \mathbb{1}[y_i^{true} \le y_{target}]$。
3. 拟合单调非降函数 $g: [0,1] \to [0,1]$，使得 $g(P_{success}^{raw}) \approx \mathbb{E}[s \mid P_{success}^{raw}]$。
4. 最终输出：

$$P_{success} = g(P_{success}^{raw})$$

**校准质量评估**：Expected Calibration Error (ECE)：

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

目标：$\text{ECE} < 0.05$。

---

## 6. 主动学习闭环 (Active Learning Loop)

GP 不应止步于初始训练。通过主动学习 (Active Learning)，系统可以自适应地选择最有信息量的分子进行高精度评估，迭代提升 GP 在关键化学空间区域的精度：

### 6.1 Acquisition Function

采用 **Uncertainty-Biased Expected Improvement**：

$$\alpha(z) = \text{EI}(z) + \lambda \cdot \sigma_{total}(z)$$

其中 $\text{EI}(z) = \mathbb{E}[\max(0, y_{best} - f(z))]$ 是标准期望改进，$\lambda > 0$ 控制 exploration-exploitation 平衡。

### 6.2 闭环流程

```
Repeat:
  1. 扩散模型生成 N_batch 个候选分子
  2. 对每个候选计算 P_success, sigma_total
  3. 选 top-K 个 alpha(z) 最高的分子
  4. 提交至高精度 oracle（MM-GBSA / FEP / 湿实验）
  5. 将新数据追加至 D_oracle，重训练 GP
  6. 重新校准 g(.) on updated D_cal
```

收敛判据：当连续 $T$ 轮迭代中 $\text{max}_k\; \sigma_{total}(z_k) < \epsilon$ 时停止。

---

## 7. 评估方案

### 7.1 指标体系

| 指标 | 定义 | 目标 |
|------|------|------|
| ECE | 校准误差 | < 0.05 |
| AUROC | 以 $P_{success}$ 区分真实 hit/non-hit 的 ROC 曲线下面积 | > 0.85 |
| EF @ 1% | 取 $P_{success}$ top 1% 的命中率 vs 随机选取的富集倍数 | > 20 |
| Hit Rate @ 0.85 | 所有 $P_{success} \ge 0.85$ 分子中真实 hit 的比例 | > 80% |
| Spearman rho | $\mu_{total}$ 与高精度 oracle 排序的秩相关 | > 0.6 |
| 生成有效率 | 化学有效（通过 RDKit sanitize）的比例 | > 95% |
| 平均推理延迟 | 单分子 $P_{success}$ 计算时间（$M=64$） | < 30s on A100 |

### 7.2 消融实验设计

| 消融 | 移除/替换 | 预期影响 |
|------|----------|----------|
| A1: 无 $U_{gen}$ | $\sigma^2_{total} = \sigma^2_{oracle}$ only | ECE 显著恶化，高分垃圾漏过 |
| A2: 无 $U_{oracle}$ | $\sigma^2_{total} = J_\mu^T \Sigma_{gen} J_\mu$ only | 对 OOD 分子无检测能力 |
| A3: 无校准 | 直接输出 $P_{success}^{raw}$ | ECE > 0.15，概率不可靠 |
| A4: 朴素协方差 | 用样本协方差替代 Ledoit-Wolf | 小 $M$ 时矩阵病态，数值不稳定 |
| A5: 无多模态检测 | 强制单高斯 | 对称口袋场景 $U_{gen}$ 虚高 |
| A6: 单精度 oracle | 只用 Vina | Spearman rho 下降，校准变差 |
| A7: 无 OOD 检测 | 移除 Mahalanobis gate | 极端外推分子获得虚假高 $P_{success}$ |
| A8: 无主动学习 | GP 只训练一次 | 在数据稀疏区域 $\sigma^2_{oracle}$ 持续偏高 |

### 7.3 基线比较

| 方法 | 描述 |
|------|------|
| TargetDiff + Vina rescore | 标准生成+打分流程，无不确定性 |
| DiffSBDD + Vina rescore | 同上 |
| MC Dropout Ensemble | 用 dropout 做近似贝叶斯推断 |
| Deep Ensemble (5 models) | 5 个独立打分网络的方差作为不确定性 |
| BayesDiff (Ours) | 完整框架 |

---

## 8. 执行清单

### Phase 1: 基础设施（Week 1-2）

- [ ] 搭建项目骨架：`bayesdiff/gen_uncertainty.py`, `bayesdiff/gp_oracle.py`, `bayesdiff/fusion.py`, `bayesdiff/calibration.py`
- [ ] 实现 SE(3)-等变编码器（基于 PaiNN），或复用预训练 Uni-Mol / GemNet 的表征层
- [ ] 实现 MC 采样 + Ledoit-Wolf 协方差估计 + GMM 模态检测
- [ ] 验证：对 CrossDocked2020 测试集做 MC 采样，可视化潜空间分布

### Phase 2: GP Oracle（Week 3-4）

- [ ] 基于 GPyTorch 实现 SVGP + ARD Matern-5/2 核
- [ ] 实现多精度 ICM 核，联合训练 Vina + MM-GBSA 数据
- [ ] 实现 Mahalanobis OOD 检测层
- [ ] 验证：在 PDBbind 测试集上评估 GP 的 RMSE 和校准

### Phase 3: 融合与校准（Week 5-6）

- [ ] 实现一阶/二阶 Delta Method 融合（`torch.autograd.grad` + `torch.autograd.functional.hessian`）
- [ ] 实现 MC 积分回退方案 + 自适应策略切换
- [ ] 实现 Isotonic Regression 校准 + ECE 评估
- [ ] 验证：在 held-out 集上绘制 reliability diagram

### Phase 4: 主动学习与端到端评估（Week 7-8）

- [ ] 实现 acquisition function 和 active learning loop
- [ ] 跑全消融实验和基线比较
- [ ] 计算全部评估指标
- [ ] 写论文

---

## 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $x$ | 3D 分子构象（原子坐标 + 类型） |
| $c$ | 条件（口袋 + 可选骨架） |
| $z = E_\phi(x)$ | 潜空间嵌入向量 |
| $\bar{z}$ | MC 采样潜空间质心 |
| $\hat{\Sigma}_{gen}$ | Shrinkage 生成协方差矩阵 |
| $U_{gen} = \text{tr}(\hat{\Sigma}_{gen})$ | 标量化生成不确定性 |
| $\mu_{oracle}(z)$ | GP 后验均值 |
| $\sigma^2_{oracle}(z)$ | GP 后验方差（认知不确定性） |
| $J_\mu$ | GP 均值对 $z$ 的雅可比向量 |
| $H_\mu$ | GP 均值对 $z$ 的 Hessian 矩阵 |
| $\sigma^2_{total}$ | 融合后总方差 |
| $P_{success}$ | 校准后的成功概率 |
| $d_{Maha}$ | Mahalanobis OOD距离 |
