
# 分子生成框架：融合生成不确定性、Oracle 不确定性、概率校准与 OOD 修正

## 1. 整体框架

**输入**：蛋白口袋或条件输入 \( x \)

**输出**：最终用于排序/决策的成功概率 \( P_{\text{final}} \)

### 核心变量
- \( x \): 蛋白口袋 / 条件输入
- \( m^{(i)} \): 第 \( i \) 个生成的分子
- \( z^{(i)} \in \mathbb{R}^d \): 分子 \( m^{(i)} \) 的图级 embedding
- \( y \): 待预测的性质（如 binding affinity / activity score）
- \( \mathcal{D} = \{(z_n, y_n)\}_{n=1}^N \): 训练数据集
- \( p(z \mid x) \): 生成模型在 latent space 诱导的条件分布
- \( \mu_{\text{oracle}}(z) \), \( \sigma^2_{\text{oracle}}(z) \): GP 对 \( y \) 的预测均值与方差
- \( P_{\text{success}} \): 原始成功概率
- \( P_{\text{success}}^{\text{cal}} \): 校准后的成功概率
- \( d_{\text{OOD}}(z) \): OOD 距离
- \( w(z) \): OOD 置信度修正权重
- \( P_{\text{final}} \): 最终决策概率

### 完整流程
1. 生成多个候选分子 \( m^{(1)}, \dots, m^{(M)} \)
2. 编码为 embeddings \( z^{(1)}, \dots, z^{(M)} \)
3. 估计生成分布 \( p(z \mid x) \)
4. 使用 Gaussian Process 建模 \( z \mapsto y \)
5. 融合生成不确定性与 Oracle 不确定性
6. 将连续预测分布转换为原始成功概率 \( P_{\text{success}} \)
7. 对 \( P_{\text{success}} \) 进行**概率校准**
8. 对校准概率进行**OOD 修正**
9. 输出最终概率 \( P_{\text{final}} \)

## 2. 生成侧：分子分布与生成不确定性

### 2.1 生成样本
对同一口袋 \( x \) 生成 \( M \) 个分子，得到 embedding 集合：
\[
\mathcal{Z}(x) = \{z^{(1)}, z^{(2)}, \dots, z^{(M)}\}, \quad z^{(i)} \in \mathbb{R}^d
\]

### 2.2 单峰高斯近似（推荐 + Ledoit–Wolf 收缩）

样本均值：
\[
\bar{z} = \frac{1}{M} \sum_{i=1}^M z^{(i)}
\]

样本协方差：
\[
S = \frac{1}{M-1} \sum_{i=1}^M (z^{(i)} - \bar{z})(z^{(i)} - \bar{z})^\top
\]

Ledoit–Wolf 收缩估计：
\[
\hat{\Sigma}_{\text{gen}} = (1 - \lambda) S + \lambda T
\]
其中 \( \lambda \in [0,1] \) 为收缩系数，\( T \) 通常为对角矩阵或球形矩阵。

生成分布近似：
\[
p(z \mid x) \approx \mathcal{N}(\bar{z}, \hat{\Sigma}_{\text{gen}})
\]

### 2.3 多峰近似：Gaussian Mixture Model (GMM)

\[
p(z \mid x) \approx \sum_{k=1}^K \pi_k \mathcal{N}(z \mid \mu_k, \Sigma_k)
\]

全局均值与协方差：
\[
\bar{z} = \sum_{k=1}^K \pi_k \mu_k, \quad
\Sigma_{\text{gen}} = \sum_{k=1}^K \pi_k \left[ \Sigma_k + (\mu_k - \bar{z})(\mu_k - \bar{z})^\top \right]
\]

## 3. Oracle：高斯过程预测分子性质

预测分布：
\[
y \mid z, \mathcal{D} \approx \mathcal{N}\big(\mu_{\text{oracle}}(z), \sigma^2_{\text{oracle}}(z)\big)
\]

### 3.1 GP 模型
潜函数先验：
\[
f(z) \sim \mathcal{GP}(m(z), k(z, z'))
\]
常用均值函数：\( m(z) = 0 \)

观测模型：
\[
y_n = f(z_n) + \varepsilon_n, \quad \varepsilon_n \sim \mathcal{N}(0, \sigma_\varepsilon^2)
\]

### 3.2 常用核函数
- **RBF Kernel**：
\[
k_{\text{RBF}}(z,z') = \sigma_f^2 \exp\left( -\frac{\|z - z'\|^2}{2\ell^2} \right)
\]

- **Matérn-5/2 Kernel**：
\[
k_{\text{Matérn-5/2}}(z,z') = \sigma_f^2 \left(1 + \frac{\sqrt{5} r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left( -\frac{\sqrt{5} r}{\ell} \right), \quad r = \|z - z'\|
\]

- **ARD 形式**：每维独立长度尺度 \( \ell_j \)

## 4. 不确定性融合（Law of Total Variance）

总预测方差：
\[
\sigma^2_{\text{total}}(x) \approx
\underbrace{\mathbb{E}_{z \mid x} \big[ \sigma^2_{\text{oracle}}(z) \big]}_{\text{Oracle uncertainty}} +
\underbrace{\mathrm{Var}_{z \mid x} \big[ \mu_{\text{oracle}}(z) \big]}_{\text{Generated-input uncertainty}}
\]

### 一阶 Delta Method 近似
\[
\mu_{\text{oracle}}(z) \approx \mu_{\text{oracle}}(\bar{z}) + J_\mu^\top (z - \bar{z}), \quad J_\mu = \nabla_z \mu_{\text{oracle}}(z) \big|_{\bar{z}}
\]

\[
\sigma^2_{\text{total}}(x) \approx \mathbb{E}_{z \mid x}[\sigma^2_{\text{oracle}}(z)] + J_\mu^\top \hat{\Sigma}_{\text{gen}} J_\mu
\]

## 5. 从连续预测到原始成功概率

假设 \( y \mid x \sim \mathcal{N}(\mu, \sigma^2_{\text{total}}) \)，成功定义为 \( y \ge \tau \)：

\[
P_{\text{success}} = \Pr(y \ge \tau \mid x) = \Phi\left( \frac{\mu - \tau}{\sigma_{\text{total}}} \right)
\]

其中 \( \Phi(\cdot) \) 为标准正态分布累积分布函数（CDF）。

## 6. 概率校准（Probability Calibration）

目标：学习映射 \( g: [0,1] \to [0,1] \)，使 \( P_{\text{success}}^{\text{cal}} = g(P_{\text{success}}) \) 更接近真实成功频率。

### 推荐方法：Isotonic Regression（等张回归）
约束：\( g \) 单调不减

优化：
\[
\min_{g \in \mathcal{M}} \sum_{i=1}^n (t_i - g(p_i))^2
\]
其中 \( t_i = \mathbf{1}(y_i \ge \tau) \)，\( \mathcal{M} \) 为单调函数族。

输出：
\[
P_{\text{success}}^{\text{cal}} = g_{\text{iso}}(P_{\text{success}})
\]

### 其他常用校准方法
- **Platt Scaling**：\( P^{\text{cal}} = \sigma(A \cdot \text{logit}(p) + B) \)
- **Temperature Scaling**：\( P^{\text{cal}} = \sigma(s / T) \)
- **Beta Calibration**：\( P^{\text{cal}} = \sigma(a \log p + b \log(1-p) + c) \)
- **Histogram Binning**
- **Bayesian Binning**（Beta-Binomial 平滑）

### 校准评价指标
- Brier Score
- Negative Log Likelihood (NLL)
- Expected Calibration Error (ECE)

## 7. OOD 修正（Out-of-Distribution Correction）

### 7.1 Mahalanobis 距离（核心方法）
训练集统计量：
\[
\mu_{\text{train}} = \frac{1}{N} \sum_{n=1}^N z_n, \quad
\Sigma_{\text{train}} = \frac{1}{N-1} \sum_{n=1}^N (z_n - \mu_{\text{train}})(z_n - \mu_{\text{train}})^\top
\]

OOD 距离：
\[
d_{\text{OOD}}(z) = \sqrt{ (z - \mu_{\text{train}})^\top \Sigma_{\text{train}}^{-1} (z - \mu_{\text{train}}) }
\]

### 7.2 软惩罚权重（推荐）
\[
w(z) = \exp(-\alpha \, d_{\text{OOD}}(z)), \quad \alpha > 0
\]

最终概率：
\[
P_{\text{final}} = w(z) \cdot P_{\text{success}}^{\text{cal}}
\]

### 其他 OOD 修正方式
- 硬阈值拒绝：\( d_{\text{OOD}}(z) > \delta \) 时置 0
- 方差膨胀：\( \sigma_{\text{final}}^2 = \sigma^2_{\text{total}} + \beta \, d_{\text{OOD}}(z)^2 \)
- 均值惩罚：\( \mu_{\text{final}} = \mu - \gamma \, d_{\text{OOD}}(z) \)
- 联合修正（均值 + 方差 + 权重）

## 8. 最推荐的落地完整公式链

\[
\begin{aligned}
p(z \mid x) &\approx \mathcal{N}(\bar{z}, \hat{\Sigma}_{\text{gen}}) \quad \text{或 GMM} \\
\sigma^2_{\text{total}} &\approx \mathbb{E}_{z\mid x}[\sigma^2_{\text{oracle}}(z)] + J_\mu^\top \hat{\Sigma}_{\text{gen}} J_\mu \\
P_{\text{success}} &= \Phi\left( \frac{\mu - \tau}{\sigma_{\text{total}}} \right) \\
P_{\text{success}}^{\text{cal}} &= g_{\text{iso}}(P_{\text{success}}) \quad \text{(Isotonic Regression)} \\
d_{\text{OOD}}(z) &= \sqrt{(z - \mu_{\text{train}})^\top \Sigma_{\text{train}}^{-1} (z - \mu_{\text{train}})} \\
w(z) &= \exp(-\alpha \, d_{\text{OOD}}(z)) \\
P_{\text{final}} &= w(z) \cdot P_{\text{success}}^{\text{cal}}
\end{aligned}
\]

## 9. 方法总表

### 概率校准方法
| 方法                  | 公式                                      | 特点                     |
|-----------------------|-------------------------------------------|--------------------------|
| Isotonic Regression   | \( P^{\text{cal}} = g_{\text{iso}}(P) \) | 灵活、无分布假设        |
| Platt Scaling         | Sigmoid 参数化                            | 平滑、参数少            |
| Temperature Scaling   | \( \sigma(s/T) \)                         | 只调一个参数，保序      |
| Beta Calibration      | 更灵活的 Beta 变换                        | 适应偏态分布            |
| Histogram Binning     | 分箱经验频率                              | 简单但不连续            |

### OOD 修正方法
| 方法             | 核心公式                              | 特点               |
|------------------|---------------------------------------|--------------------|
| Mahalanobis 距离 | \( d_{\text{OOD}}(z) = \sqrt{\cdots} \) | 考虑协方差         |
| 软指数惩罚       | \( w(z) = \exp(-\alpha d) \)          | 连续降权（推荐）   |
| 硬阈值拒绝       | \( d > \delta \to 0 \)                | 严格过滤           |
| 方差膨胀         | \( \sigma^2 + \beta d^2 \)            | 增加不确定性       |
| 均值惩罚         | \( \mu - \gamma d \)                  | 直接降低预测值     |

此版本已高度结构化、公式统一、逻辑完整，可直接用于 LaTeX 转 Markdown 或论文写作。如需进一步添加伪代码、算法框、实验细节或 LaTeX 源码版本，请随时告诉我！