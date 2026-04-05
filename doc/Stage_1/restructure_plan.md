# BayesDiff 项目结构重组计划

> **目标**: 在不改变任何功能代码的前提下，重新梳理项目结构、优化文件命名，使项目逻辑清晰，方便写文章时快速定位代码、实验结果和图表。

---

## 一、当前问题诊断

### 1.1 scripts/ 目录混乱

当前 `scripts/` 下有 **35 个脚本**，编号 01–29（含子编号如 26a–d、28c），存在以下问题：

| 问题 | 具体表现 |
|------|----------|
| **编号冲突** | `11_gp_training_analysis.py` 和 `11_reextract_embeddings.py` 共用编号 11 |
| **编号断裂** | 没有编号 `15` 之后直接跳到 `18`（14→15→16→17→18 虽然连续但与前半段的主线脱节） |
| **主线 vs 辅助不分** | 核心 pipeline（01→09）和后续探索实验（10→29）混在一起，不区分主实验与辅助分析 |
| **命名不一致** | 有的用 `_shard`、有的用 `_array`；`26a/b/c/d` 这种子编号难以维护 |
| **shim/工具散落** | `torch_scatter_shim.py`、`_check_deps.py` 与实验脚本混在一起 |

### 1.2 bayesdiff/ 模块扁平

所有 8 个核心模块平铺在一个目录下，没有子包结构。虽然模块数量不多，但从论文角度看缺少逻辑分组（生成端 vs 预言端 vs 融合端 vs 评估端）。

### 1.3 slurm/ 缺乏组织

46 个 slurm 脚本平铺，无法快速找到"跑采样用哪个"、"跑评估用哪个"。

### 1.4 results/ 结构与论文脱节

结果目录按"技术产出"组织（如 `gp_model/`、`generated_molecules/`），而非按"论文中的实验"组织。写文章时需要在多个目录间跳转才能找到一个实验的完整结果。

### 1.5 doc/ 无层次

`Stage_1/` 下 9 个文档各自独立，没有索引或阅读顺序，新读者不知道从哪开始。

---

## 二、重组方案

### 2.1 scripts/ → 分组 + 重编号

将脚本按功能分为 **4 个子目录 + 1 个工具目录**：

```
scripts/
├── pipeline/                    # 核心 pipeline（论文 §4 Method 对应）
│   ├── s01_prepare_data.py           ← 01_prepare_data.py
│   ├── s02_sample_molecules.py       ← 02_sample_molecules.py
│   ├── s03_extract_embeddings.py     ← 03_extract_embeddings.py
│   ├── s04_train_gp.py              ← 04_train_gp.py
│   ├── s05_evaluate.py              ← 05_evaluate.py
│   ├── s06_ablation.py              ← 06_ablation.py
│   └── s07_generate_figures.py       ← 09_generate_figures.py
│
├── scaling/                     # 大规模实验 & 分片（论文 §5 Experiments 对应）
│   ├── s01_sample_shard.py           ← 08_sample_molecules_shard.py
│   ├── s02_merge_shards.py           ← 07_merge_sampling_shards.py
│   ├── s03_prepare_tier3.py          ← 15_prepare_tier3.py
│   ├── s04_sample_tier3_shard.py     ← 16_sample_tier3_shard.py
│   ├── s05_train_gp_tier3.py        ← 17_train_gp_tier3.py
│   ├── s06_merge_50mol_shards.py     ← 27_merge_50mol_shards.py
│   ├── s07_extract_50mol_embeddings.py ← 29_extract_50mol_embeddings.py
│   └── s08_merge_and_train_eval.py   ← 10_merge_and_train_eval.py
│
├── studies/                     # 辅助研究 & 消融（论文 §5–§6 + SI 对应）
│   ├── embedding_comparison.py       ← 13_embedding_comparison.py
│   ├── embedding_multilayer.py       ← 22_extract_multilayer_embeddings.py
│   ├── embedding_encoder_only.py     ← 19_extract_encoder_embeddings.py
│   ├── embedding_unimol.py           ← 26b_extract_unimol.py
│   ├── embedding_schnet.py           ← 26c_extract_schnet.py
│   ├── embedding_multilayer_full.py  ← 26a_extract_multilayer_full.py
│   ├── embedding_compare_all.py      ← 26d_compare_embeddings.py
│   ├── gp_training_analysis.py       ← 11_gp_training_analysis.py
│   ├── gp_encoder.py                ← 20_train_gp_encoder.py
│   ├── gp_aggregation.py            ← 21_train_gp_aggregation.py
│   ├── gp_multilayer.py             ← 23_train_gp_multilayer.py
│   ├── gp_50mol_study.py            ← 28_50mol_gp_study.py
│   ├── bo_gp_hyperparams.py         ← 14_bo_gp_hyperparams.py
│   ├── robust_evaluation.py         ← 12_robust_evaluation.py
│   ├── regularization_study.py       ← 25_regularization_study.py
│   ├── subsample_ablation.py         ← 28c_subsample_ablation.py
│   ├── tier3_training_curves.py      ← 24_tier3_training_curves_analysis.py
│   ├── train_val_test_analysis.py    ← 18_train_val_test_analysis.py
│   └── reextract_embeddings.py       ← 11_reextract_embeddings.py
│
├── utils/                       # 工具 & 调试
│   ├── check_deps.py                ← _check_deps.py
│   ├── torch_scatter_shim.py        ← torch_scatter_shim.py
│   └── run_full_pipeline.py         ← run_full_pipeline.py
│
└── README.md                   # 脚本索引：每个子目录的用途 + 执行顺序
```

**命名规则**：
- `pipeline/` 内用 `s01_`–`s07_` 编号（s = step），表示顺序执行
- `scaling/` 内用 `s01_`–`s08_` 编号，表示大规模扩展的步骤
- `studies/` 内不编号，用描述性名称（按主题分组，不要求顺序）
- `utils/` 不编号

### 2.2 bayesdiff/ → 添加逻辑分组注释（不拆子包）

考虑到模块只有 8 个，**不创建子包**（避免大量 import 路径变更），而是：

1. 在 `__init__.py` 中按论文章节分组 import：

```python
# === §4.1 Generation Module ===
from .sampler import TargetDiffSampler
from .gen_uncertainty import estimate_gen_uncertainty

# === §4.2 Oracle Module ===
from .gp_oracle import GPOracle

# === §4.3 Fusion Module ===
from .fusion import fuse_uncertainties

# === §4.4 Calibration & OOD ===
from .calibration import IsotonicCalibrator
from .ood import MahalanobisOOD

# === §4.5 Evaluation ===
from .evaluate import evaluate_all

# === Data Utilities ===
from .data import parse_pdbbind_index, protein_family_split
```

2. 每个模块文件头部添加论文章节映射注释：

```python
"""
bayesdiff.fusion — Delta Method Uncertainty Fusion
===================================================
Paper reference: §4.3 "Uncertainty Fusion via the Delta Method"
Equation reference: Eq. (7)–(9) in 03_math_reference.md §4
"""
```

### 2.3 slurm/ → 分组子目录

```
slurm/
├── pipeline/                    # 对应 scripts/pipeline/
│   ├── sample_job.sh
│   ├── sample_array_job.sh
│   ├── train_gp.sh
│   ├── gp_train_eval_viz.sh
│   ├── eval_ablation.sh
│   └── full_pipeline_job.sh
│
├── scaling/                     # 对应 scripts/scaling/
│   ├── sample_maxgpu.sh
│   ├── sample_tier3_array.sh
│   ├── merge_sample_shards_job.sh
│   ├── merge_maxgpu.sh
│   ├── merge_gp_aggregation.sh
│   ├── merge_gp_multilayer.sh
│   ├── merge_and_evaluate_1000step.sh
│   ├── tier3_gp.sh
│   ├── tier3_analysis.sh
│   └── 50mol_gp_study.sh
│
├── studies/                     # 对应 scripts/studies/
│   ├── embedding_1000step_array.sh
│   ├── eval_1000step_final.sh
│   ├── extract_embeddings.sh
│   ├── extract_multilayer.sh
│   ├── emb_compare.sh
│   ├── bo_gp.sh
│   ├── gp_analysis.sh
│   ├── robust_eval.sh
│   ├── regularization_study.sh
│   ├── subsample_ablation.sh
│   ├── subsample_ablation_cpu.sh
│   ├── train_gp_encoder.sh
│   ├── train_gp_aggregation.sh
│   ├── train_gp_multilayer.sh
│   ├── pretrained_phase_a.sh
│   ├── pretrained_phase_bcd.sh
│   ├── pretrained_v2.sh
│   ├── pretrained_v2_cpu.sh
│   ├── pretrained_v3_a100.sh
│   ├── pretrained_v3_l40s.sh
│   ├── rdkit_pipeline.sh
│   ├── schnet_a5.sh
│   ├── schnet_only_a100.sh
│   └── schnet_v2_cpu.sh
│
├── utils/                       # 诊断 & 测试
│   ├── gpu_verify.slurm
│   └── smoke_test.slurm
│
└── logs/                        # 保持不变
```

### 2.4 results/ → 按论文实验重组

```
results/
├── main_experiment/             # 论文 §5 主实验（PDBbind 48/93 pockets）
│   ├── sampling/                     # all_embeddings.npz, sampling_summary.json
│   ├── gp_model/                     # gp_model.pt, train_meta.json, train_data.npz
│   ├── evaluation/                   # eval_metrics.json, per_pocket_results.json
│   ├── ablation/                     # ablation_results.json
│   └── figures/                      # fig1–fig6 publication figures
│
├── crossdocked_validation/      # 论文 §5 CrossDocked 大规模泛化验证（932 pockets）
│   ├── sampling/
│   ├── gp_model/
│   └── evaluation/
│
├── sampling_density_analysis/   # 论文 §6 采样密度分析（93 pockets × 50 mol）
│   ├── embeddings/
│   ├── gp_results/
│   └── training_curves/
│
├── embedding_comparison/        # 论文 §6 表征对比（2D vs 3D）
│   ├── fcfp4/
│   ├── rdkit/
│   ├── encoder/
│   ├── multilayer/
│   ├── unimol/
│   ├── schnet/
│   └── comparison_table.json
│
├── hyperparameter_search/       # SI: BO 超参搜索
│   └── bo_results.json
│
├── supplementary/               # SI: 其他辅助实验
│   ├── robust_evaluation/
│   ├── regularization/
│   └── subsample_ablation/
│
├── pipeline_log.txt             # 保持不变
└── pipeline_results.json        # 保持不变
```

### 2.5 doc/ → 添加索引 + 重命名

```
doc/
├── README.md                    # 文档索引（新增），列出各阶段文档的阅读顺序
│
├── Stage_1/                     # Phase 1: 设计 & 实现
│   ├── 00_reading_guide.md          # 新增：本目录阅读指南
│   ├── 01_01_overall_plan.md           ← 01_overall_plan.md
│   ├── 02_math_tutorial.md          ← 02_math_tutorial.md
│   ├── 03_math_reference.md         ← 03_math_reference.md
│   ├── 04_opendata_plan.md          ← 04_opendata_plan.md
│   ├── 05_code_math_audit.md        ← 05_code_math_audit.md
│   ├── 06_gp_optimization.md        ← 06_gp_optimization.md
│   ├── 07_50mol_study_plan.md       ← 07_50mol_study_plan.md
│   ├── 08_embedding_plan.md         ← 08_embedding_plan.md
│   ├── 09_09_progress_log.md           ← 09_progress_log.md
│   └── restructure_plan.md          # 本文件（保持不变）
│
├── Stage_2/                     # Phase 2: 问题分析 & 未来方向
│   └── problem_and_solution.md  ← problem&solution.md（去掉 & 符号）
│
└── hpc/                         # HPC 运维文档（保持不变）
    ├── HPC_ENV_STATUS.md
    ├── bayesdiff_nyu_torch_hpc_agent_guide.md
    ├── hpc_execution_plan.md
    └── nyu_torch_coding_agent_guide.md
```

### 2.6 write_up/ → 添加论文-代码映射表

在 `write_up/` 中新增 `code_map.md`，建立论文章节 → 代码/结果的映射：

```
write_up/
├── main.tex                     # 保持不变
├── write_up_guide.md            # 保持不变
├── write_up_structure.md        # 保持不变
├── figures/                     # 保持不变
└── code_map.md                  # 新增：论文 ↔ 代码映射
```

`code_map.md` 内容示例：

| 论文章节 | 对应代码 | 对应结果 | 对应数学推导 |
|----------|----------|----------|-------------|
| §4.1 分子生成 | `bayesdiff/sampler.py`, `bayesdiff/gen_uncertainty.py` | `results/main_experiment/sampling/` | `doc/Stage_1/03_math_reference.md` §2–3 |
| §4.2 Oracle 预测 | `bayesdiff/gp_oracle.py` | `results/main_experiment/gp_model/` | §4 |
| §4.3 不确定性融合 | `bayesdiff/fusion.py` | — | §5 |
| §4.4 校准 & OOD | `bayesdiff/calibration.py`, `bayesdiff/ood.py` | — | §6–7 |
| §5 实验结果 | `scripts/pipeline/s05_evaluate.py` | `results/main_experiment/evaluation/` | — |
| §5 消融实验 | `scripts/pipeline/s06_ablation.py` | `results/main_experiment/ablation/` | — |
| §5 表征对比 | `scripts/studies/embedding_comparison.py` | `results/embedding_comparison/` | — |
| §5 大规模验证 | `scripts/scaling/` | `results/crossdocked_validation/` | — |
| Fig 1 | `scripts/pipeline/s07_generate_figures.py` | `results/main_experiment/figures/fig1_dashboard.png` | — |
| Fig 2 | 同上 | `results/main_experiment/figures/fig2_embeddings.png` | — |
| Fig 3 | 同上 | `results/main_experiment/figures/fig3_uncertainty.png` | — |
| Fig 4 | 同上 | `results/main_experiment/figures/fig4_ablation.png` | — |
| Fig 5 | 同上 | `results/main_experiment/figures/fig5_calibration.png` | — |
| Fig 6 | 同上 | `results/main_experiment/figures/fig6_pocket_ranking.png` | — |

### 2.7 tests/ → 添加 README

```
tests/
├── README.md                    # 新增：测试说明
├── test_pipeline.py             ← debug_pipeline.py（重命名更规范）
└── test_phase1_validation.py    ← validate_phase1.py（重命名更规范）
```

### 2.8 根目录整理

```
BayesDiff/
├── README.md                    # 更新：反映新结构 + 添加目录说明
├── requirements.txt             # 保持不变
├── .gitignore                   # 保持不变
├── .gitmodules                  # 保持不变
├── bayesdiff/                   # 核心包（内容不变，注释增强）
├── scripts/                     # 重组后的脚本（4 子目录）
├── slurm/                       # 重组后的 HPC 脚本（4 子目录）
├── data/                        # 保持不变
├── results/                     # 重组后的结果（按实验分组）
├── doc/                         # 重组后的文档（添加索引）
├── write_up/                    # 论文目录（添加 code_map.md）
├── tests/                       # 重命名后的测试
├── external/                    # 保持不变（TargetDiff submodule）
└── logs/                        # 保持不变
```

---

## 三、新旧文件名映射表

### 3.1 scripts/ 完整映射

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `scripts/pipeline/s01_prepare_data.py` | `scripts/pipeline/s01_prepare_data.py` | 核心 pipeline |
| `scripts/pipeline/s02_sample_molecules.py` | `scripts/pipeline/s02_sample_molecules.py` | 核心 pipeline |
| `scripts/pipeline/s03_extract_embeddings.py` | `scripts/pipeline/s03_extract_embeddings.py` | 核心 pipeline |
| `scripts/pipeline/s04_train_gp.py` | `scripts/pipeline/s04_train_gp.py` | 核心 pipeline |
| `scripts/pipeline/s05_evaluate.py` | `scripts/pipeline/s05_evaluate.py` | 核心 pipeline |
| `scripts/pipeline/s06_ablation.py` | `scripts/pipeline/s06_ablation.py` | 核心 pipeline |
| `scripts/pipeline/s07_generate_figures.py` | `scripts/pipeline/s07_generate_figures.py` | 核心 pipeline（重编号） |
| `scripts/scaling/s02_merge_shards.py` | `scripts/scaling/s02_merge_shards.py` | 大规模实验 |
| `scripts/scaling/s01_sample_shard.py` | `scripts/scaling/s01_sample_shard.py` | 大规模实验 |
| `scripts/scaling/s08_merge_and_train_eval.py` | `scripts/scaling/s08_merge_and_train_eval.py` | 大规模实验 |
| `scripts/scaling/s03_prepare_tier3.py` | `scripts/scaling/s03_prepare_tier3.py` | 大规模实验 |
| `scripts/scaling/s04_sample_tier3_shard.py` | `scripts/scaling/s04_sample_tier3_shard.py` | 大规模实验 |
| `scripts/scaling/s05_train_gp_tier3.py` | `scripts/scaling/s05_train_gp_tier3.py` | 大规模实验 |
| `scripts/scaling/s06_merge_50mol_shards.py` | `scripts/scaling/s06_merge_50mol_shards.py` | 大规模实验 |
| `scripts/scaling/s07_extract_50mol_embeddings.py` | `scripts/scaling/s07_extract_50mol_embeddings.py` | 大规模实验 |
| `scripts/studies/gp_training_analysis.py` | `scripts/studies/gp_training_analysis.py` | 辅助研究 |
| `scripts/studies/reextract_embeddings.py` | `scripts/studies/reextract_embeddings.py` | 辅助研究 |
| `scripts/studies/robust_evaluation.py` | `scripts/studies/robust_evaluation.py` | 辅助研究 |
| `scripts/studies/embedding_comparison.py` | `scripts/studies/embedding_comparison.py` | 辅助研究 |
| `scripts/studies/bo_gp_hyperparams.py` | `scripts/studies/bo_gp_hyperparams.py` | 辅助研究 |
| `scripts/studies/train_val_test_analysis.py` | `scripts/studies/train_val_test_analysis.py` | 辅助研究 |
| `scripts/studies/embedding_encoder_only.py` | `scripts/studies/embedding_encoder_only.py` | 辅助研究 |
| `scripts/studies/gp_encoder.py` | `scripts/studies/gp_encoder.py` | 辅助研究 |
| `scripts/studies/gp_aggregation.py` | `scripts/studies/gp_aggregation.py` | 辅助研究 |
| `scripts/studies/embedding_multilayer.py` | `scripts/studies/embedding_multilayer.py` | 辅助研究 |
| `scripts/studies/gp_multilayer.py` | `scripts/studies/gp_multilayer.py` | 辅助研究 |
| `scripts/studies/tier3_training_curves.py` | `scripts/studies/tier3_training_curves.py` | 辅助研究 |
| `scripts/studies/regularization_study.py` | `scripts/studies/regularization_study.py` | 辅助研究 |
| `scripts/studies/embedding_multilayer_full.py` | `scripts/studies/embedding_multilayer_full.py` | 辅助研究 |
| `scripts/studies/embedding_unimol.py` | `scripts/studies/embedding_unimol.py` | 辅助研究 |
| `scripts/studies/embedding_schnet.py` | `scripts/studies/embedding_schnet.py` | 辅助研究 |
| `scripts/studies/embedding_compare_all.py` | `scripts/studies/embedding_compare_all.py` | 辅助研究 |
| `scripts/studies/gp_50mol_study.py` | `scripts/studies/gp_50mol_study.py` | 辅助研究 |
| `scripts/studies/subsample_ablation.py` | `scripts/studies/subsample_ablation.py` | 辅助研究 |
| `scripts/utils/check_deps.py` | `scripts/utils/check_deps.py` | 工具 |
| `scripts/utils/torch_scatter_shim.py` | `scripts/utils/torch_scatter_shim.py` | 工具 |
| `scripts/utils/run_full_pipeline.py` | `scripts/utils/run_full_pipeline.py` | 工具 |

### 3.2 doc/Stage_1/ 映射

| 旧文件名 | 新文件名 | 理由 |
|----------|----------|------|
| `01_overall_plan.md` | `01_01_overall_plan.md` | 添加阅读顺序编号 |
| `02_math_tutorial.md` | `02_math_tutorial.md` | 区分 tutorial vs reference |
| `03_math_reference.md` | `03_math_reference.md` | 区分 tutorial vs reference |
| `04_opendata_plan.md` | `04_opendata_plan.md` | 统一命名风格 |
| `05_code_math_audit.md` | `05_code_math_audit.md` | 更具描述性 |
| `06_gp_optimization.md` | `06_gp_optimization.md` | 简化 |
| `07_50mol_study_plan.md` | `07_50mol_study_plan.md` | 统一编号 |
| `08_embedding_plan.md` | `08_embedding_plan.md` | 简化 |
| `09_progress_log.md` | `09_09_progress_log.md` | 放最后（日志性质） |

### 3.3 tests/ 映射

| 旧文件名 | 新文件名 | 理由 |
|----------|----------|------|
| `debug_pipeline.py` | `test_pipeline.py` | 符合 `test_` 命名规范 |
| `validate_phase1.py` | `test_phase1_validation.py` | 符合 `test_` 命名规范 |

---

## 四、需要新增的文件

| 文件 | 用途 |
|------|------|
| `scripts/README.md` | 脚本索引：每个子目录的作用、执行顺序、依赖关系 |
| `tests/README.md` | 测试说明：如何运行、覆盖范围 |
| `doc/README.md` | 文档总索引 |
| `doc/Stage_1/00_reading_guide.md` | Stage_1 阅读指南 |
| `write_up/code_map.md` | 论文 ↔ 代码/结果映射表 |

---

## 五、需要更新的文件

| 文件 | 更新内容 |
|------|----------|
| `bayesdiff/__init__.py` | 按论文章节分组 import，添加注释 |
| `bayesdiff/*.py`（每个模块） | 文件头添加论文章节映射 docstring |
| `README.md`（根目录） | 更新目录结构说明，反映新的 scripts/ 组织 |
| `scripts/utils/run_full_pipeline.py` | 更新内部路径引用（指向新的 pipeline/ 子目录） |
| 所有 `slurm/*.sh` | 更新脚本路径引用（如 `scripts/pipeline/s01_prepare_data.py` → `scripts/pipeline/s01_prepare_data.py`） |

---

## 六、执行步骤

### Step 1: 创建目录结构
```bash
mkdir -p scripts/{pipeline,scaling,studies,utils}
mkdir -p slurm/{pipeline,scaling,studies,utils}
mkdir -p results/{main_experiment/{sampling,gp_model,evaluation,ablation,figures},crossdocked_validation/{sampling,gp_model,evaluation},sampling_density_analysis/{embeddings,gp_results,training_curves},embedding_comparison/{fcfp4,rdkit,encoder,multilayer,unimol,schnet},hyperparameter_search,supplementary/{robust_evaluation,regularization,subsample_ablation}}
```

### Step 2: 移动脚本文件（git mv 保留历史）
```bash
# pipeline
git mv scripts/pipeline/s01_prepare_data.py scripts/pipeline/s01_prepare_data.py
git mv scripts/pipeline/s02_sample_molecules.py scripts/pipeline/s02_sample_molecules.py
# ... (按映射表执行)
```

### Step 3: 重命名文档
```bash
git mv doc/Stage_1/01_overall_plan.md doc/Stage_1/01_01_overall_plan.md
# ... (按映射表执行)
```

### Step 4: 更新内部引用
- 更新 `run_full_pipeline.py` 中的脚本路径
- 更新所有 slurm 脚本中的 `python scripts/XX_xxx.py` 路径
- 更新 `README.md`

### Step 5: 新增文件
- 创建 `scripts/README.md`、`tests/README.md`、`doc/README.md`
- 创建 `doc/Stage_1/00_reading_guide.md`
- 创建 `write_up/code_map.md`

### Step 6: 增强 bayesdiff/ 注释
- 更新 `__init__.py` 分组 import
- 每个模块添加论文映射 docstring

### Step 7: 验证
```bash
python scripts/utils/check_deps.py        # 依赖检查
python tests/test_pipeline.py             # pipeline 测试
python tests/test_phase1_validation.py    # phase1 验证
```

### Step 8: 提交
```bash
git add -A
git commit -m "refactor: restructure project for paper writing

- scripts/: split into pipeline/, scaling/, studies/, utils/
- slurm/: mirror scripts/ structure
- results/: reorganize by experiment (main, tier3, 50mol, embedding)
- doc/Stage_1/: add reading order numbers + reading guide
- tests/: rename to test_ convention
- write_up/: add code_map.md (paper ↔ code mapping)
- bayesdiff/: add paper section references in docstrings
- No functional code changes"
```

---

## 七、风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| `git mv` 后路径引用失效 | Step 4 统一更新所有引用；Step 7 运行测试验证 |
| slurm 历史 job 无法重现 | 在 commit message 中记录旧→新映射 |
| `results/` 移动大文件耗时 | 先 `git mv` 记录变更，再物理移动（或用 symlink 过渡） |
| 外部合作者的本地路径失效 | 在 README 中标注"结构重组于 YYYY-MM-DD" |

---

## 八、重组前后对比

### 写文章时的查找效率

| 场景 | 重组前 | 重组后 |
|------|--------|--------|
| "找消融实验代码" | 在 35 个脚本中找 `06_ablation.py` | 直接看 `scripts/pipeline/s06_ablation.py` |
| "找采样密度研究的全部结果" | 在 `results/` 下翻 3 个目录 | 全在 `results/sampling_density_analysis/` |
| "找 Fig 3 对应的代码" | 不确定是哪个脚本 | 查 `write_up/code_map.md` → `s07_generate_figures.py` |
| "数学推导在哪" | `02_math_tutorial.md` 还是 `03_math_reference.md`？ | `02_math_tutorial.md`（入门）vs `03_math_reference.md`（详细） |
| "跑 tier3 实验用哪些 slurm" | 在 46 个 .sh 中搜索 | 全在 `slurm/scaling/` |
| "Stage_1 文档从哪读起" | 9 个文件无序 | `00_reading_guide.md` 提供路线图 |
