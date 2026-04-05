# 论文 ↔ 代码映射表

## 方法章节 → 代码

| 论文章节 | 核心代码 | 结果目录 | 数学推导 |
|----------|----------|----------|----------|
| §4.1 分子生成 & 不确定性 | `bayesdiff/sampler.py`, `bayesdiff/gen_uncertainty.py` | `results/generated_molecules/` | `doc/Stage_1/03_math_reference.md` §2–3 |
| §4.2 Oracle 预测 (SVGP) | `bayesdiff/gp_oracle.py` | `results/gp_model/` | §4 |
| §4.3 不确定性融合 (Delta Method) | `bayesdiff/fusion.py` | — | §5 |
| §4.4 校准 & OOD 检测 | `bayesdiff/calibration.py`, `bayesdiff/ood.py` | — | §6–7 |

## 实验章节 → 代码

| 论文章节 | 脚本 | 结果目录 |
|----------|------|----------|
| §5 主实验 (PDBbind) | `scripts/pipeline/s05_evaluate.py` | `results/evaluation/` |
| §5 消融实验 (A1–A7) | `scripts/pipeline/s06_ablation.py` | `results/ablation/` |
| §5 表征对比 (2D vs 3D) | `scripts/studies/embedding_comparison.py` | `results/embedding_comparison/` |
| §5 大规模验证 (CrossDocked) | `scripts/scaling/` | `results/tier3_sampling/` |
| §6 采样密度分析 (50mol) | `scripts/studies/gp_50mol_study.py` | `results/50mol_gp/` |
| §6 鲁棒评估 (Bootstrap CI) | `scripts/studies/robust_evaluation.py` | `results/evaluation/` |

## 图表 → 代码

| 图表 | 生成脚本 | 输出文件 |
|------|----------|----------|
| Fig 1: Dashboard | `scripts/pipeline/s07_generate_figures.py` | `results/figures/fig1_dashboard.png` |
| Fig 2: Embeddings | 同上 | `results/figures/fig2_embeddings.png` |
| Fig 3: Uncertainty | 同上 | `results/figures/fig3_uncertainty.png` |
| Fig 4: Ablation | 同上 | `results/figures/fig4_ablation.png` |
| Fig 5: Calibration | 同上 | `results/figures/fig5_calibration.png` |
| Fig 6: Pocket Ranking | 同上 | `results/figures/fig6_pocket_ranking.png` |

## 数据 → 代码

| 数据集 | 准备脚本 | 数据目录 |
|--------|----------|----------|
| PDBbind v2020 | `scripts/pipeline/s01_prepare_data.py` | `data/splits/` |
| CrossDocked (Tier3) | `scripts/scaling/s03_prepare_tier3.py` | `data/tier3_pockets/` |
