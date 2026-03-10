# BayesDiff 实操方案：纯公开数据 + Mac 开发 + HPC 扩展

> **核心策略**：不跑任何 oracle 计算。所有"真值"来自 PDBbind 的实验 $K_d/K_i/\text{IC}_{50}$，所有 docking 分数来自 CrossDocked2020 已有标注或本地 Quick Vina 2 批量打分（CPU 可跑）。扩散模型直接用 **TargetDiff 官方预训练权重**，不从头训练。

---

## 0. 方案与 overall_plan 的关系

| overall_plan 中的设计 | 本方案的落地策略 |
|----------------------|---------------|
| $\mathcal{D}_{diff}$：训练扩散模型 | **不训练**，直接用 TargetDiff 预训练 checkpoint |
| $\mathcal{D}_{oracle}^{(h)}$：MM-GBSA / 实验 | PDBbind v2020 refined set 实验亲和力（公开） |
| $\mathcal{D}_{oracle}^{(l)}$：Vina batch rescore | CrossDocked2020 已有 Vina 分数 / 本地 Quick Vina 2 |
| 多精度 ICM 核 | Phase 1 先跑单精度（仅实验标签），Phase 2 加 Vina 做多精度 |
| 主动学习闭环 | **离线模拟**：按 acquisition function 排序，模拟逐批揭示标签 |

---

## 1. 数据准备

### 1.1 PDBbind v2020（实验真值）

| 项目 | 内容 |
|------|------|
| 下载 | http://www.pdbbind.org.cn/ （需注册，免费） |
| Refined set | ~4,852 complexes with 实验 $K_d/K_i/\text{IC}_{50}$ |
| General set | ~19,443 complexes（质量较低，可做低精度层） |
| 格式 | PDB + SDF + `INDEX_refined_data.2020` 文本文件 |
| 标签转换 | $pK = -\log_{10}(K_d)$，统一为 $pK_d$ 或 $\Delta G = RT \ln K_d$ |

**Mac 端操作**：

```bash
# 解压后目录结构
pdbbind_v2020/
  refined-set/
    1a1e/  # PDB code
      1a1e_protein.pdb
      1a1e_ligand.sdf
      1a1e_pocket.pdb   # 10Å 口袋
    ...
  INDEX_refined_data.2020  # PDB_code, resolution, -logKd, Kd_value
```

### 1.2 CrossDocked2020（扩散模型训练集 + Vina 分数）

| 项目 | 内容 |
|------|------|
| 下载 | https://bits.csb.pitt.edu/files/crossdocked2020/ |
| 内容 | ~22.5M cross-docked poses，每个 pose 带 Vina score |
| 用途 | TargetDiff 的训练来源；我们只用其预存 Vina 分数做低精度层 |

> 如果不想下全量（~30GB），可以只用 TargetDiff 预处理后的子集（~100K，见其 GitHub repo）。

### 1.3 CASF-2016（标准基准测试集）

| 项目 | 内容 |
|------|------|
| 下载 | 包含在 PDBbind 包中，`CASF-2016/` 文件夹 |
| 核心集 | 285 complexes，实验 $K_d$ 已知 |
| 用途 | 标准评估打分能力、排序、对接成功率的基准 |
| 优势 | 所有对比方法（Vina, GNINA, RTMScore 等）都有公开成绩 |

### 1.4 数据划分方案

```
PDBbind refined set (4,852 complexes)
│
├── 按 protein family (UniProt cluster @ 30% seqID) 做 split
│   ├── Train:  70% (~3,396) → GP 训练
│   ├── Val:    10% (~485)   → 超参调优
│   ├── Cal:    10% (~485)   → Isotonic Regression 校准
│   └── Test:   10% (~486)   → 最终评估
│
└── 额外 held-out: CASF-2016 core set (285)
    → 与上述 zero overlap，作为独立验证
```

> **为什么用 protein family split**：按 PDB code 随机 split 会导致同源蛋白泄漏。按 UniProt 30% sequence identity clustering 是该领域金标准（同 TankBind, DiffDock 论文）。可用 `mmseqs2 easy-cluster` 一条命令完成。

---

## 2. 扩散模型：直接复用预训练

### 2.1 TargetDiff

| 项目 | 内容 |
|------|------|
| Repo | https://github.com/guanjq/targetdiff |
| Checkpoint | `pretrained_model.pt`（~200MB） |
| 依赖 | PyTorch 1.13+, PyG, RDKit |
| 采样 | 每个口袋生成 $M$ 个分子，100 步 DDPM，~30s/分子 on GPU |

**Mac 端可行性**：

| 操作 | Mac (M1/M2/M3 CPU) | Mac (MPS) | HPC (A100) |
|------|-------|---------|------|
| 加载模型 | OK | OK | OK |
| 采样 1 分子 | ~10 min | ~2 min | ~30s |
| 采样 $M=64$ per pocket | 不现实 | 勉强可（~2h/pocket） | **推荐**（~30 min/pocket） |
| 采样 $M=8$ per pocket（调试） | ~80 min | ~16 min | - |

**策略**：Mac 上用 $M=4 \sim 8$ 做代码调试和 pipeline 验证；正式实验在 HPC 上用 $M=64$。

### 2.2 备选：DiffSBDD / DecompDiff

如果需要第二个扩散模型做对比：

| 模型 | Repo | 预训练 |
|------|------|--------|
| DiffSBDD | https://github.com/arneschneuing/DiffSBDD | 有 |
| DecompDiff | https://github.com/bytedance/DecompDiff | 有 |

---

## 3. SE(3)-等变编码器：用 SchNet / PaiNN 的预训练表征

### 3.1 选型

不从头训练编码器。用 **TorchMD-NET** 提供的预训练 PaiNN（在 QM9/OC20 上预训练），或直接用 TargetDiff 内部的 SchNet encoder 最后一层 embedding 作为 $z$。

**最省力方案**：直接提取 TargetDiff encoder 的中间表征：

```python
# 伪代码—从 TargetDiff 模型提取 ligand embedding
with torch.no_grad():
    # TargetDiff 的 encoder 输出 node-level features
    h_ligand = model.encoder(ligand_pos, ligand_atom_type, pocket_pos, pocket_atom_type)
    # 对 ligand 节点做 mean pooling → graph-level embedding
    z = h_ligand.mean(dim=0)  # shape: (d,)
```

**潜空间维度**：TargetDiff SchNet 默认 $d=128$。可选：
- 直接用 $d=128$，配合 DKL 或 PCA 降维
- 加一层可训练的 MLP projector $128 \to 64$（用回归损失在 PDBbind 上微调）

### 3.2 Mac 端可行性

编码器推理非常轻量（纯 forward pass），Mac CPU 上 ~50ms/分子，**完全没问题**。

---

## 4. GP Oracle：单精度快速启动

### 4.1 Phase 1：单精度 GP（仅实验标签）

**数据**：PDBbind refined train set（~3,396 complexes），标签为实验 $pK_d$。

**实现**：

```python
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)  # d=64
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

**Mac 端资源估算**：

| 参数 | 值 |
|------|-----|
| $N_{train}$ | 3,396 |
| $d$ (潜空间维度) | 64 |
| $J$ (诱导点) | 512 |
| 训练时间 (Mac M2 CPU) | ~5-15 min (200 epochs, batch=256) |
| 内存 | < 4 GB |

**结论**：GP 训练完全可以在 Mac 上完成，不需要 HPC。

### 4.2 Phase 2：多精度扩展（可选）

如果 Phase 1 效果好，进一步加入 Vina 分数做低精度层：
- 用 Quick Vina 2（CPU 可跑）对 PDBbind 全集重新打分
- 或直接用 CrossDocked2020 中已有的 Vina 分数
- ICM 核实现参考 GPyTorch `MultitaskKernel`

---

## 5. 完整 Pipeline 与计算资源分配

### 5.1 数据流

```
[PDBbind PDB files]
        │
        ▼
[TargetDiff 采样 M 个分子 per pocket]  ← HPC (GPU)
        │
        ▼
[SE(3) 编码器提取 z^(m)]  ← Mac OK
        │
        ▼
[计算 z̄, Σ̂_gen, GMM 模态检测]  ← Mac OK (numpy/sklearn)
        │
        ▼
[GP 推理: μ_oracle(z̄), σ²_oracle(z̄)]  ← Mac OK (GPyTorch CPU)
        │
        ▼
[融合: σ²_total = σ²_oracle + J_μᵀ Σ̂_gen J_μ]  ← Mac OK (torch.autograd)
        │
        ▼
[校准: P_success = g(Φ(...))]  ← Mac OK (sklearn IsotonicRegression)
        │
        ▼
[评估: ECE, AUROC, EF@1%, ...]  ← Mac OK
```

### 5.2 什么必须上 HPC，什么 Mac 就够

| 步骤 | Mac (M1/M2/M3) | HPC (GPU) |
|------|:---:|:---:|
| 数据预处理 (PDB → PyG graphs) | ✅ | - |
| TargetDiff 采样 ($M=64$, 全测试集) | ❌ 太慢 | ✅ **必须** |
| TargetDiff 采样 ($M=4$, 调试 5 个口袋) | ✅ ~3h | - |
| SE(3) 编码器 forward pass | ✅ | - |
| GP 训练 (SVGP, $N=3.4$K, $J=512$) | ✅ ~15 min | - |
| GP 推理 + autograd 求 $J_\mu$ | ✅ | - |
| 不确定性融合 + 校准 | ✅ | - |
| 全消融实验 ($8 \times$ full eval) | ✅ ~1h | - |
| 基线对比 (Deep Ensemble × 5) | ⚠️ 慢 | ✅ 推荐 |

### 5.3 HPC 任务清单

只需要提交 **一个 batch job**：

```bash
#!/bin/bash
#SBATCH --job-name=bayesdiff_sample
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 对测试集每个口袋生成 M=64 个分子
python sample_all_pockets.py \
    --checkpoint pretrained_model.pt \
    --test_pockets data/pdbbind_test_pockets.pkl \
    --num_samples 64 \
    --output_dir results/generated_molecules/

# 提取 embedding
python extract_embeddings.py \
    --input_dir results/generated_molecules/ \
    --encoder targetdiff \
    --output results/embeddings.npz
```

**预估时间**：486 个测试口袋 × 64 samples × 30s/sample ≈ **~260 GPU-hours**。单卡 A100 约 11 天；4 卡并行约 3 天。

> **节省方案**：先只跑 CASF-2016 core set（285 口袋），约 **6 天单卡 / 1.5 天 4 卡**。

---

## 6. 评估协议

### 6.1 任务定义

| 任务 | 标签 | 阈值 |
|------|------|------|
| 回归 | $pK_d$（连续值） | - |
| 分类（活性） | $\mathbb{1}[pK_d \ge 7]$（对应 $K_d \le 100$ nM） | $y_{target} = 7$ |
| 分类（高亲和） | $\mathbb{1}[pK_d \ge 8]$（对应 $K_d \le 10$ nM） | $y_{target} = 8$ |

> **注意**：overall_plan 里用 $\Delta G$，这里统一改用 $pK_d$（越大越好）。对应的 $P_{success} = 1 - \Phi\left(\frac{y_{target} - \mu}{\sigma}\right)$（取右尾）。

### 6.2 指标

| 指标 | 说明 | 计算工具 |
|------|------|---------|
| **ECE** | 10-bin Expected Calibration Error | 手写 / `netcal` |
| **AUROC** | $P_{success}$ 作为分类器的 ROC-AUC | `sklearn.metrics.roc_auc_score` |
| **EF@1%** | Enrichment Factor at top 1% | 手写 |
| **Hit Rate @ $P \ge 0.85$** | 高置信分子中的真实命中率 | 手写 |
| **Spearman $\rho$** | $\mu_{oracle}$ vs $pK_d^{true}$ 的秩相关 | `scipy.stats.spearmanr` |
| **RMSE** | $\mu_{oracle}$ vs $pK_d^{true}$ | `sklearn` |
| **NLL** | 负对数似然（衡量概率预测质量） | 手写 |

### 6.3 消融（与 overall_plan 一致，但标签来源不同）

| ID | 消融 | 做法 |
|----|------|------|
| A1 | 无 $U_{gen}$ | $\sigma^2_{total} = \sigma^2_{oracle}$ |
| A2 | 无 $U_{oracle}$ | $\sigma^2_{total} = J_\mu^T \Sigma_{gen} J_\mu$ |
| A3 | 无校准 | 输出 $P_{success}^{raw}$ |
| A4 | 朴素协方差 | 无 Ledoit-Wolf shrinkage |
| A5 | 无多模态检测 | 强制 $K=1$ |
| A7 | 无 OOD 检测 | 移除 Mahalanobis gate |

> A6（多精度）和 A8（主动学习）留到 Phase 2。

### 6.4 基线

| 基线 | 不确定性来源 | 实现 |
|------|-----------|------|
| Vina rescore（无 UQ） | 无 | Quick Vina 2 on Mac |
| GNINA rescore（无 UQ） | 无 | GNINA （Linux/HPC） |
| MC Dropout | TargetDiff encoder 开 dropout，跑 $T=20$ 次 | 改 1 行代码 |
| Deep Ensemble | 训练 5 个独立 MLP: $z \to pK_d$ | Mac OK， ~5 min |
| GP only（无 $U_{gen}$） | $\sigma^2_{oracle}$ | 即消融 A1 |
| **BayesDiff (Ours)** | $\sigma^2_{total}$ (calibrated) | 完整 pipeline |

---

## 7. 项目结构

```
BayesDiff/
├── doc/
│   ├── overall_plan.md         # 理论全景（已有）
│   ├── plan_opendata.md        # 本文件：实操方案
│   └── progress_log.md         # 开发日志
├── data/
│   ├── pdbbind/                # PDBbind v2020 (手动下载，.gitignore)
│   │   ├── refined-set/
│   │   ├── INDEX_refined_data.2020
│   │   └── CASF-2016/
│   ├── splits/                 # 划分文件 (train/val/cal/test PDB lists)
│   └── embeddings/             # 预计算的 z 向量 (.npz)
├── bayesdiff/
│   ├── __init__.py
│   ├── data.py                 # 数据加载、split、标签转换
│   ├── sampler.py              # 调用 TargetDiff 采样 + 提取 embedding
│   ├── gen_uncertainty.py      # Σ̂_gen, GMM 模态检测
│   ├── gp_oracle.py            # SVGP 训练/推理
│   ├── fusion.py               # Delta method 融合
│   ├── calibration.py          # Isotonic regression + ECE
│   ├── ood.py                  # Mahalanobis OOD 检测
│   └── evaluate.py             # 全部指标计算
├── scripts/
│   ├── 01_prepare_data.py      # Mac: 预处理 PDBbind
│   ├── 02_sample_molecules.py  # HPC: TargetDiff 批量采样
│   ├── 03_extract_embeddings.py# Mac/HPC: 提取 z
│   ├── 04_train_gp.py          # Mac: 训练 SVGP
│   ├── 05_evaluate.py          # Mac: 融合 + 校准 + 评估
│   ├── 06_ablation.py          # Mac: 消融实验
│   └── run_full_pipeline.py    # Mac: 一键端到端 debug pipeline
├── external/
│   └── targetdiff/             # TargetDiff clone (.gitignore)
├── results/                    # 运行输出 (.gitignore)
│   ├── figures/
│   ├── generated_molecules/
│   ├── gp_model/
│   ├── evaluation/
│   └── ablation/
├── notebooks/
│   ├── debug_pipeline.py       # Mac 端调试用
│   └── validate_phase1.py      # Phase 1 验证测试 (41 checks)
├── slurm/
│   └── sample_job.sh           # HPC SLURM 脚本
├── requirements.txt
└── README.md
```

---

## 8. 依赖与环境

### 8.1 Mac 本地环境

```bash
conda create -n bayesdiff python=3.10
conda activate bayesdiff

# 核心
pip install torch torchvision  # CPU 版; MPS 自动启用 on Apple Silicon
pip install torch-geometric     # PyG CPU
pip install gpytorch            # GP
pip install rdkit               # 化学
pip install scikit-learn scipy numpy pandas
pip install matplotlib seaborn

# TargetDiff 依赖
pip install easydict pyyaml tqdm
```

### 8.2 HPC 环境（补充 GPU 支持）

```bash
# 在 Mac 环境基础上加 CUDA 版本
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric  # 按 PyG 官方指引装 CUDA 版
```

---

## 9. 分阶段执行计划

### Phase 0: 数据 + 环境（Day 1-2）— Mac ✅ COMPLETE

- [x] 下载 PDBbind v2020 refined set + CASF-2016
- [x] 创建 conda 环境，安装依赖
- [x] 运行 `01_prepare_data.py`：解析 INDEX 文件，生成 split，转 $pK_d$
- [x] 克隆 TargetDiff repo，下载预训练权重
- [x] 在 Mac 上对 **3 个口袋** 各采样 $M=2$ 个分子，验证 pipeline 通路

**验收标准**：能拿到 3 个口袋的分子 + 对应 SE(3) embedding $z$（128-dim）。✅ 已验收。

> **实际情况**：使用 TargetDiff CrossDocked2020 test set（93 targets）作为数据源。
> affinity_info.pkl 提供 pKd 标签（184K entries, 76K with pK）。
> Mac CPU 上 3 pockets × 2 samples × 20 steps ≈ 7 min。20 步不足以产生有效分子，
> 但 SE(3) embedding 已从 TargetDiff encoder 最后一层提取（scatter_mean over ligand atoms）。

### Phase 1: 核心模块实现（Day 3-7）— Mac ✅ COMPLETE

- [x] `gen_uncertainty.py`：Ledoit-Wolf 协方差 + GMM 模态检测
- [x] `gp_oracle.py`：SVGP 训练/推理（PCA + k-means inducing + early stopping）
- [x] `fusion.py`：一阶 Delta method + MC 回退 + 多模态融合
- [x] `calibration.py`：Isotonic + Platt + Temperature + cross-validated + ECE/ACE
- [x] `ood.py`：Mahalanobis 距离 + relative distance + confidence modifier
- [x] `evaluate.py`：全部指标 + bootstrap CI + multi-threshold + per-pocket
- [x] `data.py`：PDBbind INDEX 解析 + protein family split + CASF-2016
- [x] `sampler.py`：TargetDiff wrapper + SE(3) embedding extraction

**验收标准**：用 toy data 跑通完整 pipeline，输出 $P_{success}$。✅ 已验收（41/41 validation checks pass）。

> **SE(3) Embeddings**：从 TargetDiff UniTransformer 9 层 equivariant backbone 提取
> final_ligand_h（N_lig × 128），scatter_mean 得到 per-molecule 128-dim invariant embedding。
> 替代了 Phase 0 的 27-dim hand-crafted placeholder。

> **独立脚本**：`scripts/04_train_gp.py`、`05_evaluate.py`、`06_ablation.py` 已创建，
> 可独立于 `run_full_pipeline.py` 单独调用。

### Phase 2: HPC 批量采样（Day 5-12，与 Phase 1 并行）— HPC

- [ ] 准备 SLURM 脚本 `sample_job.sh`
- [ ] 先跑 CASF-2016 core set（285 口袋 × $M=64$）
- [ ] 提取全部 embedding，打包下载回 Mac
- [ ] （可选）跑 PDBbind 全测试集（486 口袋 × $M=64$）

**验收标准**：得到 `embeddings/casf_test.npz`，shape `(285, 64, d)`。

### Phase 3: 正式实验 + 消融（Day 10-14）— Mac

- [ ] 在 CASF-2016 上训练 GP → 评估全部指标
- [ ] 跑 6 项消融实验
- [ ] 跑 4 个基线
- [ ] 绘制 reliability diagram, ROC curve, enrichment curve
- [ ] 统计各方法 ECE / AUROC / EF@1% 对比表

**验收标准**：完整的 Table 1（方法对比）+ Table 2（消融）+ Figure 2（reliability diagram）。

### Phase 4: 写作（Day 14-21）

- [ ] Introduction + Related Work
- [ ] Method（从 overall_plan 精简）
- [ ] Experiments（从 Phase 3 结果整理）
- [ ] Conclusion

---

## 10. 风险与备选

| 风险 | 应对 |
|------|------|
| PDBbind 下载需审批，耗时 | 备选：用 BindingDB 公开子集（~30K with $K_d$） |
| TargetDiff 在 PDBbind 口袋上生成质量差 | 备选：用 DiffSBDD 或 DecompDiff |
| $M=64$ 采样 HPC 时间不够 | 先用 $M=32$；或只做 CASF-2016 子集 |
| GP 在 $d=128$ 上表现差 | 加 PCA 降到 $d=32 \sim 64$；或用 DKL |
| 校准集太小（485 个） | 用 5-fold 交叉校准 |
| Mac MPS 对 PyG 不稳定 | 全部用 CPU，采样部分上 HPC |

---

## 2026-03-05 并行执行更新（不改原脚本，新增并行脚本）

为满足“多卡并行 + 批量 sample + 不覆盖已有结果”，新增并行入口：

- `slurm/sample_array_job.sh`：Slurm array 多卡并行采样（每个 task 占 1 张 GPU，按 shard 切分 pocket）
- `scripts/07_merge_sampling_shards.py`：合并分片产物
- `slurm/merge_sample_shards_job.sh`：CPU 合并作业入口

### 并行提交示例（4 卡）

```bash
cd /scratch/$USER/BayesDiff

ARRAY_JOB_ID=$(sbatch --parsable \
  --account=torch_pr_281_chemistry \
  --array=0-3 \
  --export=ALL,POCKET_LIST=data/splits/test_pockets.txt,NUM_SAMPLES=64,NUM_STEPS=100,OUTPUT_ROOT=results/generated_molecules_parallel \
  slurm/sample_array_job.sh)

echo "ARRAY_JOB_ID=${ARRAY_JOB_ID}"
```

### 合并示例

```bash
# run_tag 默认格式: <timestamp>_j<ARRAY_JOB_ID>
RUN_TAG="<your_run_tag>"

sbatch \
  --account=torch_pr_281_chemistry \
  --dependency=afterok:${ARRAY_JOB_ID} \
  --export=ALL,RUN_DIR=results/generated_molecules_parallel/${RUN_TAG},EXPECTED_SHARDS=4 \
  slurm/merge_sample_shards_job.sh
```

### 防覆盖策略

默认输出到：

- `results/generated_molecules_parallel/<RUN_TAG>/shards/`（每 shard 独立）
- `results/generated_molecules_parallel/<RUN_TAG>/all_embeddings.npz`（合并后）

`RUN_TAG` 默认包含时间戳 + job id，因此不会覆盖当前已有 `results/generated_molecules/` 或历史并行结果。

补充：并行执行使用 `scripts/08_sample_molecules_shard.py` 作为分片包装器，内部调用原 `scripts/02_sample_molecules.py`，因此原采样代码保持不变。

---

## 2026-03-05 项目结构更新

> 本次更新聚焦“并行代码另写，不修改原采样脚本”。

当前与并行采样直接相关的结构如下：

```text
BayesDiff/
├── scripts/
│   ├── 02_sample_molecules.py        # 原单卡采样入口（保持不变）
│   ├── 07_merge_sampling_shards.py   # 新增：合并分片结果
│   └── 08_sample_molecules_shard.py  # 新增：分片包装器（调用 02）
├── slurm/
│   ├── sample_job.sh                 # 原单卡作业脚本（保持不变）
│   ├── sample_array_job.sh           # 新增：多卡 array 并行采样
│   └── merge_sample_shards_job.sh    # 新增：CPU 合并作业
└── results/
    ├── generated_molecules/          # 既有单卡结果目录
    └── generated_molecules_parallel/ # 新增并行输出根目录（按 run_tag 隔离）
```

## 2026-03-05 项目进度更新

- [x] 并行采样脚本新增完成（`sample_array_job.sh`）
- [x] 分片包装器新增完成（`08_sample_molecules_shard.py`）
- [x] 分片合并工具新增完成（`07_merge_sampling_shards.py` + `merge_sample_shards_job.sh`）
- [x] 文档并行流程补充完成（含防覆盖输出策略）
- [x] 100-step 全 pipeline 完成（S0-S8），结果回收到 GitHub
- [x] 1000-step 采样: 88/93 pockets 完成（jobs 3387783 + 3546121）
- [ ] 1000-step 最终 5 pockets + merge + GP + eval + ablation（job 3902319 已提交）
- [ ] 100-step vs 1000-step 对比分析

当前状态：**100-step pipeline 完整完成。1000-step 采样 88/93 完成，job 3902319 将完成剩余 5 pockets 并自动执行 merge + GP + eval + ablation。**
