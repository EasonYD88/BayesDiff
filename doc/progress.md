# BayesDiff Stage 2 — Progress Log

## Sub-Plan 0: PDBbind v2020 数据集准备

**Status**: ✅ 代码完成，等待数据下载  
**Commit**: `9129938` — `feat: PDBbind v2020 dataset preparation pipeline (Sub-Plan 0)`

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

### 里程碑验证（合成数据，7/7 通过）

| 里程碑 | 验证结果 |
|--------|----------|
| M0.1 INDEX 解析 | ✅ 正确解析 pdb_code, pkd, affinity_type；pKd 范围 [4.0, 8.5] |
| M0.2 Pocket 提取 | ✅ `extract_pocket_from_protein()` 生成有效 10Å PDB（30 atoms） |
| M0.3 数据集划分 | ✅ 不重叠的 train/val/cal/test split；mmseqs2 不可用时 fallback 到 time-based |
| M0.4 DataLoader | ✅ 变长 batch 正确（protein_pos, ligand_pos 正确拼接，batch index 正确） |
| 单复合物 featurize | ✅ protein 28 atoms + ligand 3 atoms → .pt 文件 |
| 全流水线 | ✅ 15 个合成复合物：parse → featurize → merge → split → DataLoader |
| 分片并行 | ✅ 3 分片处理 12 个复合物，结果与单分片一致 |

### HPC 并行策略

- 50 个 SLURM array task × 16 CPU/task = **800 CPU 核心**并行
- ~5,316 复合物 / 50 分片 ≈ 106/分片，预计 5–10 min/分片
- 原始估计 7.5–15h → 并行后 **wall time ~10 min**

### 阻塞项

- ⚠️ **PDBbind v2020 raw data 未下载** — 需从 http://www.pdbbind.org.cn/ 注册下载
- 下载后执行：
  ```bash
  bash scripts/utils/download_pdbbind.sh path/to/PDBbind_v2020_refined.tar.gz
  bash slurm/pipeline/s00_launch_all.sh
  ```
