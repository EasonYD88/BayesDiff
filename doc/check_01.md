---

# 分子生成框架调试指南：定位常数预测 Bug（mu_pred / sigma 全恒定问题）

**问题现象总结**  
当前最明确的异常有三点：  
1. 所有 pocket 的 `mu_pred` 完全一样（均为 5.83134）。  
2. 所有 pocket 的 `sigma2_oracle`、`sigma2_total`、`p_success` 也几乎完全一样。  
3. 生成不确定性完全失效（`sigma2_gen = 0`、`trace_cov_gen = 0`、`n_modes = 1` 对所有样本一致）。  
4. 消融实验中除 A2（去掉 U_oracle）外，其他版本与 Full 几乎一模一样，说明大部分模块实际未影响最终输出。

这类问题**通常不是模型能力不足**，而是典型的实现 Bug（输入塌缩、维度混淆、merge 错位、保存广播、梯度断链等）。  
下面按**最推荐的检查顺序**给出完整排查流程。请**严格按顺序执行**，前一步 Bug 存在时，后面的检查均无意义。

## 一、先确认是不是“输入已经塌了”（最高优先级）

数据流预期：  
采样分子 → 提取每个样本的 \( z^{(m)} \) → 计算每个 pocket 的 \( \bar{z} \) 和 \( \hat{\Sigma}_{\text{gen}} \) → GP 推理 → 融合 → 校准 → 评估。

**最关键检查**：不同 pocket 的 \( \bar{z} \) 是否真的不同。

### 1.1 检查 test 集每个 pocket 的 \( \bar{z} \)
加载评估阶段的中间缓存（通常为 `.npz`）：

```python
import numpy as np

data = np.load("your_test_embeddings.npz", allow_pickle=True)

targets = data["targets"]
zbar = data["zbar"]          # shape: [N_test, d]

print("targets shape:", targets.shape)
print("zbar shape:", zbar.shape)
print("first 5 targets:", targets[:5])
print("first 5 zbar norms:", np.linalg.norm(zbar[:5], axis=1))
print("unique rows:", np.unique(np.round(zbar, 6), axis=0).shape[0])
print("std over samples, mean:", zbar.std(axis=0).mean())
print("max pairwise diff from row0:", np.abs(zbar - zbar[0]).max())
```

**关键判断**：  
- `unique rows == 1` → 所有 pocket 的 \( \bar{z} \) 完全相同 → Bug 在 GP 之前。  
- `std over samples, mean ≈ 0` 或 `max diff ≈ 0` → 输入已塌缩。

### 1.2 检查是否发生了错误的广播或全局平均
重点检查以下脚本中的聚合操作：  
`03_extract_embeddings.py`、`gen_uncertainty.py`、`05_evaluate.py`

搜索关键词：`mean(`、`average(`、`reshape(`、`squeeze(`、`repeat(`、`broadcast_to(`、`tile(`

**常见错误写法**：
```python
# 错误：全局平均
zbar = z_samples.mean(axis=(0, 1))
# 错误：广播同一个向量
zbar = np.repeat(global_mean[None, :], N_test, axis=0)
```

**正确写法**：
```python
zbar = z_samples.mean(axis=1)   # [N_test, M, d] → [N_test, d]
```

## 二、检查 64 个生成样本是否其实都一样

`sigma2_gen = 0`、`trace_cov_gen = 0` 的最常见原因：虽然 `n_samples = 64`，但 64 个 embedding 实际上完全相同。

### 2.1 对单个 pocket 检查样本离散程度

```python
data = np.load("your_test_embeddings.npz", allow_pickle=True)
zs = data["z_samples"]   # shape [N_test, M, d]

i = 0
print("target:", targets[i])
print("zs[i].shape:", zs[i].shape)
print("per-sample norms (first 10):", np.linalg.norm(zs[i][:10], axis=1))
print("mean std across dims:", zs[i].std(axis=0).mean())
print("max abs diff from sample0:", np.abs(zs[i] - zs[i][0]).max())
print("unique sample rows:", np.unique(np.round(zs[i], 6), axis=0).shape[0])
```

**正常情况**：`unique sample rows > 1`、`mean std > 0`。

### 2.2 检查采样分子文件是否重复
前往 `results/generated_molecules/` 或并行采样目录，检查某个 pocket 的 64 个样本文件（SMILES、坐标是否不同）。

```python
from collections import Counter
smiles_list = [...]   # 某个 pocket 的 64 个 smiles
cnt = Counter(smiles_list)
print("n_total:", len(smiles_list))
print("n_unique:", len(cnt))
print("top duplicates:", cnt.most_common(10))
```

## 三、检查 merge/shard 后 pocket 与 embedding 是否错位

重点文件：`08_sample_molecules_shard.py`、`07_merge_sampling_shards.py`。

### 3.1 检查 merge 前后 target 列表

```python
print("before merge targets (first 20):", targets_before[:20])
print("after merge targets  (first 20):", targets_after[:20])
print("n unique before:", len(set(targets_before)))
print("n unique after:", len(set(targets_after)))
from collections import Counter
print(Counter(targets_after).most_common(10))
```

**正确**：每个 pocket 应各自对应 64 个样本。

**常见错误**：`merged[target] = shard_results[0]`（永远只取第一个）。

## 四、检查 GP 输入是否真的随 pocket 变化

### 4.1 检查 GP 测试输入

```python
print("X_test unique rows:", np.unique(np.round(X_test, 6), axis=0).shape[0])
```

### 4.2 直接打印 GP raw posterior

```python
model.eval()
likelihood.eval()
with torch.no_grad():
    pred = likelihood(model(X_test_tensor))
    mu = pred.mean.cpu().numpy()
    var = pred.variance.cpu().numpy()

print("mu first 10:", mu[:10])
print("var first 10:", var[:10])
print("mu std:", mu.std())
print("unique mu:", np.unique(np.round(mu, 6)).shape[0])
```

### 4.3 检查 GP 训练是否塌缩

```python
print("mean constant:", model.mean_module.constant.item())
print("outputscale:", model.covar_module.outputscale.item())
print("lengthscale:", model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
print("noise:", likelihood.noise.item())
```

## 五、检查 sigma2_gen 为什么恒为 0

### 5.1 打印 Σ_gen

```python
print("cov shape:", cov.shape)
print("trace(cov):", np.trace(cov))
print("diag mean:", np.diag(cov).mean())
print("max abs cov entry:", np.abs(cov).max())
```

### 5.2 检查 Jacobian（Delta Method）

```python
x = x.clone().detach().requires_grad_(True)
mu = model(x).mean
grad = torch.autograd.grad(mu.sum(), x)[0]
print("grad norm:", grad.norm().item())
```

**手工验证**：
```python
sigma2_gen_manual = (grad @ cov @ grad.T).item()
```

## 六、检查 A2（仅 U_gen）为何数值爆炸

说明当前 `sigma2_gen ≈ 0`，NLL 爆炸是正常现象。检查是否缺少 variance floor。

## 七、检查 calibration 和 OOD 为什么完全无影响

```python
print("raw p std:", p_raw.std())
print("cal p std:", p_cal.std())
print("ood distance std:", distances.std())
print("n flagged:", ood_flags.sum())
```

## 八、检查评估脚本是否把单个预测复制给了所有样本（极高概率）

**最常见保存错误**：

```python
# 错误写法
"mu_pred": float(mu_pred[0])   # 永远取第一个
```

**正确写法**：
```python
for i, target in enumerate(targets):
    result = {
        "target": target,
        "mu_pred": float(mu_pred[i]),
        ...
    }
```

## 九、最有效的一次性定位办法：生成 Debug DataFrame

```python
import hashlib
def vec_hash(x):
    return hashlib.md5(np.round(x, 6).tobytes()).hexdigest()[:8]

# 构建包含以下列的 DataFrame：
# target, pkd_true, zbar_norm, zbar_hash, trace_cov_gen, grad_norm,
# mu_gp_raw, var_gp_raw, sigma2_gen, sigma2_total, p_raw, p_cal,
# ood_distance, ood_flag
```

通过观察各列是否“全一样”即可瞬间定位 Bug 发生在哪一层。

## 十、今天建议的实际排查顺序（8 条最小检查清单）

直接复制运行以下代码（放在评估脚本开头）：

```python
# 1-2 zbar 检查
print("unique zbar rows:", np.unique(np.round(zbar, 6), axis=0).shape[0])
print("zbar std mean:", zbar.std(axis=0).mean())

# 3-4 单 pocket samples 检查
print("single pocket unique z_samples:", np.unique(np.round(zs[0], 6), axis=0).shape[0])
print("single pocket trace cov:", np.trace(np.cov(zs[0].T)))

# 5 X_test 检查
print("X_test unique rows:", np.unique(np.round(X_test, 6), axis=0).shape[0])

# 6 GP raw 输出
print("GP mu std:", mu.std(), "GP var std:", var.std())

# 7-8 Jacobian & sigma2_gen
print("grad norm:", grad.norm().item())
print("sigma2_gen manual:", float(grad @ cov @ grad.T))
```

---

**当前最可能的 5 个 Bug（概率从高到低）**：  
1. 平均维度写错（pocket 维 vs sample 维混淆）  
2. merge shard 时覆盖/错位  
3. 保存 json 时索引写死成 `[0]`  
4. z_samples 实际重复（采样或提取阶段）  
5. Jacobian 在 `no_grad()` 下求导导致 `sigma2_gen = 0`

按上面顺序跑完 8 条打印结果后，把输出贴给我，我可以立刻帮你定位具体是哪一行代码出了问题。  
需要我帮你写完整的 Debug 脚本或修改某个具体文件的补丁，也随时说！