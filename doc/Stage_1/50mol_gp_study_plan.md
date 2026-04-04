# 50mol GP Study Plan

> 日期：2026-03-28 → 03-30
> 数据集：`results/embedding_50mol/` — 31 shards, ~93 pockets × 50 molecules, 128-dim TargetDiff encoder embeddings
> 目标：用更大数据集训练 GP，系统搜索 kernel/embedding/PCA/ARD，生成 training curves、test 结果、ablation study、可视化

### 关键结论

| Embedding | N | LOOCV ρ | R² | 备注 |
|-----------|---|---------|-----|------|
| Combined-128 (50mol) | 947 | 0.399 | 0.147 | 50mol encoder embeddings |
| Encoder-128 (tier3) | 942 | 0.400 | 0.148 | ~5 mols/pocket baseline |
| FCFP4-2048 | 942 | **0.749** | **0.531** | 化学指纹远超 encoder |
| 50mol-only | 18 | 0.379 | -0.142 | 数据太少 |

> **结论**: 50mol encoder embeddings 相比 tier3 几乎无提升 (ρ=0.399 vs 0.400)。FCFP4 指纹大幅领先。

---

## Phase 0: Data Merge & Preparation

**脚本**: `scripts/29_extract_50mol_embeddings.py` (replaces old merge-only script)

> **关键修复**: 50mol sampling pipeline 的 diffusion 步骤不提取 encoder embeddings（全为零）。需要 post-hoc encoder forward pass (`fix_x=True`) 来获取真正的 128-dim embeddings。

1. [x] 加载 TargetDiff pretrained model
2. [x] 遍历 31 shard 目录，找到所有 pocket SDF 文件
3. [x] 对每个 pocket：加载 test_set 蛋白质 PDB + 读取 SDF → encoder forward pass → `final_ligand_h` (128-dim)
4. [x] `scatter_mean` 聚合 per-molecule → per-pocket embeddings
5. [x] 匹配 pKd 标签 (`external/targetdiff/data/affinity_info.pkl`)
6. [x] 与 tier3 合并：替换重叠 pockets 的 embeddings (50 mols/pocket 比 ~5 mols/pocket 更精确)
7. [x] 输出: `results/50mol_gp/{X_50mol_128, X_combined_128, y_pkd_*, families_*}.npy/json`

**已完成结果** (merge only, before encoder fix): 77 pockets total, 37 with pKd, combined=956

---

## Phase 1: Data Analysis & Visualization

**脚本**: `scripts/28_50mol_gp_study.py` (Part 1)

1. **数据分布**: pKd 直方图, 每 pocket 分子数分布, embedding 维度统计
2. **PCA 降维可视化**: 2D PCA scatter (colored by pKd)
3. **生成不确定性**: 每 pocket $U_{gen} = \text{tr}(\hat{\Sigma}_{gen})$ 分布
4. **与 Tier 3 对比**: 分子数提升效果

---

## Phase 2: GP Training with Bayesian Hyperparameter Optimization

**脚本**: `scripts/28_50mol_gp_study.py` (Part 2)

### 搜索空间

| Axis | Options |
|------|---------|
| **Kernel type** | RBF, Matérn-3/2, Matérn-5/2, RQ |
| **PCA dimensionality** | None, 10, 20, 32, 64 |
| **ARD** | True (per-dim lengthscale) vs False (isotropic) |
| **Learning rate** | [0.01, 0.2] |
| **Epochs** | [50, 300] |
| **Noise lower bound** | [1e-5, 0.1] |

Total: ~4 × 5 × 2 = 40 grid configs + 200 Optuna trials

### 评估协议

- **LOOCV** (analytic, $K^{-1}$ diagonal) — 主要指标
- **10× repeated 60/20/20 split** — 稳定性评估
- **Metrics**: RMSE, Spearman ρ, R², NLL, 95% CI coverage

---

## Phase 3: Training Curves & Test Results

**脚本**: `scripts/28_50mol_gp_study.py` (Part 3)

Top-5 configurations:

1. **Training curves**: epoch-by-epoch NLL (train), RMSE/ρ/R² (train/val/test)
2. **Test scatter**: predicted vs true pKd + GP uncertainty (±2σ)
3. **Uncertainty calibration**: observed vs predicted coverage at 50/75/90/95%
4. **Learning rate & noise convergence**: hyperparameter traces

---

## Phase 4: Ablation Study

**脚本**: `scripts/28_50mol_gp_study.py` (Part 4)

| Ablation | What's Changed | Expected Effect |
|----------|---------------|-----------------|
| Full (best) | Best BO config | Baseline |
| A1: Kernel swap | RBF↔Matérn↔RQ | Kernel sensitivity |
| A2: No ARD | Isotropic LS | ARD benefit |
| A3: PCA sweep | PCA-10/20/32/64/None | Optimal dim |
| A4: No PCA | Full 128-dim (= Best config, pca_dims=0) | ✅ Already covered by baseline |
| A5: Fewer molecules | Subsample 20/10/5 mols/pocket | Molecule count effect |

**A5 补充脚本**: `scripts/28c_subsample_ablation.py`
- 对每个 subsample count (20/10/5)，从 `per_pocket_embeddings.npz` 随机抽取 n_mols 分子
- 重算 mean embedding → 合并 tier3 → 用 best config (RQ, PCA=None, ARD=True) 跑 GP
- 3 seeds 平均 LOOCV + 5× splits
- SLURM: `slurm/subsample_ablation_cpu.sh` (CPU job, account: `torch_pr_872_general`)

**A5 结果** (job 5127962, 72s on CPU):

| Subsample | Eligible Pockets | LOOCV ρ | Test ρ | Overfit Gap |
|-----------|-----------------|---------|--------|-------------|
| 50mol (all) | 18 | 0.399 | 0.315±0.058 | 0.607 |
| 20mol | 10 | 0.385 | 0.326±0.086 | 0.565 |
| 10mol | 11 | 0.385 | 0.325±0.086 | 0.566 |
| 5mol | 13 | 0.385 | 0.325±0.086 | 0.565 |

> **结论**: 分子数 (5→50) 对 encoder embeddings 的 GP 性能几乎无影响 (Δρ ≈ 0.014)。
> 说明 encoder 提取的表征信息高度冗余，少量分子即可捕获 pocket 的 embedding 特征。

---

## Phase 5: Visualization & Doc Update

### 生成图表

| # | Figure | Description |
|---|--------|-------------|
| 1 | `01_data_overview.png` | pKd distribution, PCA scatter, mol count histogram |
| 2 | `02_bo_optimization.png` | Optuna trace, parameter importance |
| 3 | `03_kernel_comparison.png` | Bar chart: Test ρ per kernel type |
| 4 | `04_pca_sweep.png` | Performance vs PCA dimensions |
| 5 | `05_ard_effect.png` | ARD vs isotropic comparison |
| 6 | `06_training_curves.png` | Train/Val/Test per epoch (top configs) |
| 7 | `07_test_scatter.png` | Predicted vs true + uncertainty bands |
| 8 | `08_ablation_summary.png` | Ablation table figure |
| 9 | `09_overfit_analysis.png` | Train ρ vs Test ρ all configs |
| 10 | `10_calibration.png` | Uncertainty calibration plots |

### 文档更新

- `doc/progress_log.md` — 添加 50mol 实验结果
- `doc/gp_analysis_and_optimization.md` — 更新最优配置
- `doc/50mol_gp_study_plan.md` — 标记完成状态

---

## 执行环境

| Item | Value |
|------|-------|
| SLURM partition | `l40s_public` |
| Account | `torch_pr_281_general` (fallback from `torch_pr_872_general`) |
| GPU | L40S |
| Conda env | `bayesdiff` |
| Script | `slurm/50mol_gp_study.sh` |

---

## Status

- [x] Phase 0: Data Merge (job 5122686, 39 pockets extracted, 18 with pKd, 947 combined)
- [x] Phase 1: Data Analysis (N=947, D=128, 19 PCs for 90% variance)
- [x] Phase 2: BO + Grid Search (40 grid + 200 BO trials; best: RQ+PCA=None+ARD=True, LOOCV ρ=0.403)
- [x] Phase 3: Training Curves (top-3 configs, severe overfitting: train ρ>0.9, test ρ≈0.3)
- [x] Phase 4: Ablation Study — A1-A4 complete (9 configs); A5 complete (12 configs total, job 5127962)
- [x] Phase 5: Visualization (10 figures generated, 05_ablation.png updated with A5)
- [x] Doc Update: A5 结果已补充
