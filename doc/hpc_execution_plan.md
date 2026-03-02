# BayesDiff HPC 执行计划

> **目标**：在 HPC 集群上完成 Phase 2（批量采样 + embedding 提取）和 Phase 3（正式实验 + 消融），产出论文所需的全部定量结果。
>
> **前置条件**：Phase 0 & 1 已在 Mac 本地完成，代码已通过全部 41 项验证检查，所有 HPC 脚本已本地测试通过。

---

## 0. 执行概览

| 阶段 | 任务 | 预计耗时 | 资源 | 验收标准 |
|------|------|----------|------|----------|
| **S0** | 环境搭建 | 30 min | 登录节点 | `python -c "import torch; print(torch.cuda.is_available())"` → `True` |
| **S1** | 代码部署 | 15 min | 登录节点 | `python scripts/_check_deps.py` 全部通过 |
| **S2** | 数据准备 | 10 min | 登录节点 | `data/splits/test_pockets.txt` 存在且包含 93 行 |
| **S3** | 批量采样 | **~12–48h** | 1×GPU | `results/generated_molecules/all_embeddings.npz` 包含 93 个 key |
| **S4** | Embedding 再提取 | 2–4h | 1×GPU | `results/generated_molecules/all_embeddings_reextracted.npz` |
| **S5** | GP 训练 | 15 min | CPU 节点 | `results/gp_model/gp_model.pt` 存在 |
| **S6** | 评估 + 校准 | 10 min | CPU 节点 | `results/evaluation/eval_metrics.json` 含 7 项指标 |
| **S7** | 消融实验 | 15 min | CPU 节点 | `results/ablation/ablation_summary.json` 含 7 个 ablation |
| **S8** | 结果回收 | 10 min | 本地 Mac | 所有 JSON/NPZ/PNG 已下载 |

---

## S0. HPC 环境搭建

### S0.1 Conda 环境创建

```bash
# 登录 HPC
ssh <username>@<hpc-hostname>

# 创建 conda 环境
module load anaconda3          # 或 module load miniconda3，视 HPC 而定
conda create -n bayesdiff python=3.10 -y
conda activate bayesdiff
```

### S0.2 安装 CUDA 版 PyTorch + 依赖

```bash
# PyTorch (CUDA 12.1) — 根据 HPC 的 CUDA 版本调整
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# PyG — 必须匹配 torch + CUDA 版本
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# GPyTorch + 其他依赖
pip install gpytorch rdkit scikit-learn scipy numpy pandas
pip install easydict pyyaml tqdm matplotlib seaborn
```

### S0.3 验证 GPU

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
import torch_geometric
print(f'PyG: {torch_geometric.__version__}')
"
```

**预期输出**：`CUDA available: True`，GPU 名称应为 A100 / V100 / H100 等。

### S0.4 常见问题

| 问题 | 解决 |
|------|------|
| `module load` 找不到 anaconda | `module avail` 搜索；或用 `~/miniconda3/bin/conda` |
| torch-cluster 编译失败 | 确认 `nvcc --version` 与 torch CUDA 版本匹配；尝试 `pip install --no-build-isolation` |
| 磁盘配额不足 | 将 conda env 装在 `/scratch/$USER/` 或 `$WORK/`，不要装在 `$HOME/` |
| CUDA OOM | 减小 `NUM_SAMPLES`（32 → 16）或 batch_size |

---

## S1. 代码部署

### S1.1 克隆仓库

```bash
cd /scratch/$USER   # 或你的 HPC 工作目录
git clone https://github.com/EasonYD88/BayesDiff.git
cd BayesDiff
```

### S1.2 部署 TargetDiff

```bash
# 克隆 TargetDiff
git clone https://github.com/guanjq/targetdiff.git external/targetdiff

# 下载预训练权重
mkdir -p external/targetdiff/pretrained_models
cd external/targetdiff/pretrained_models

# 方法 1: 从 TargetDiff release 下载
wget <targetdiff_pretrained_url> -O pretrained_diffusion.pt

# 方法 2: 从已有的 Mac 上传
# (在 Mac 上) scp external/targetdiff/pretrained_models/pretrained_diffusion.pt <user>@<hpc>:/scratch/$USER/BayesDiff/external/targetdiff/pretrained_models/

cd /scratch/$USER/BayesDiff
```

### S1.3 部署 CrossDocked2020 测试集数据

```bash
# 方法 1: 从 Mac 上传整个 test_set 目录
# (在 Mac 上)
rsync -avz external/targetdiff/data/ <user>@<hpc>:/scratch/$USER/BayesDiff/external/targetdiff/data/

# 方法 2: 在 HPC 上重新下载（如果有 test_set.zip）
cd external/targetdiff/data
# wget <test_set_url> -O test_set.zip && unzip test_set.zip
```

### S1.4 部署 torch_scatter shim

TargetDiff 代码使用旧的 `import torch_scatter` API，需要兼容 shim：

```bash
# 确认 shim 已存在
cat external/targetdiff/torch_scatter.py
# 应包含：将 torch_scatter API 映射到 torch_geometric.utils.scatter
```

如果没有，从 Mac 上传或手动创建：

```python
# external/targetdiff/torch_scatter.py
"""Shim: maps torch_scatter API to torch_geometric.utils.scatter."""
from torch_geometric.utils import scatter
def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
def scatter_max(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
```

### S1.5 检查依赖完整性

```bash
conda activate bayesdiff
cd /scratch/$USER/BayesDiff
python scripts/_check_deps.py
```

预期：所有依赖项 ✅。如果有缺失，按提示 `pip install` 补装。

### S1.6 验证目录结构

```bash
echo "=== Project Structure ==="
ls -la bayesdiff/*.py | wc -l     # 应为 9 (含 __init__.py)
ls -la scripts/*.py | wc -l       # 应为 8
ls external/targetdiff/pretrained_models/pretrained_diffusion.pt  # 应存在
ls external/targetdiff/data/test_set/ | wc -l  # 应为 93
```

---

## S2. 数据准备

### S2.1 生成测试集 pocket 列表

```bash
cd /scratch/$USER/BayesDiff

# 从 test_set 目录自动生成 pocket 列表（93 个 target）
ls external/targetdiff/data/test_set/ > data/splits/test_pockets.txt

# 验证
wc -l data/splits/test_pockets.txt   # 应输出 93
head -5 data/splits/test_pockets.txt
```

### S2.2 （可选）先用小子集做冒烟测试

```bash
# 创建 5-pocket 子集做快速验证
head -5 data/splits/test_pockets.txt > data/splits/smoke_pockets.txt

# 快速冒烟测试（~10 min on GPU）
python scripts/02_sample_molecules.py \
    --pocket_list data/splits/smoke_pockets.txt \
    --pdbbind_dir external/targetdiff/data/test_set \
    --targetdiff_dir external/targetdiff \
    --num_samples 4 \
    --num_steps 100 \
    --device cuda \
    --output_dir results/smoke_test

# 验证输出
ls results/smoke_test/
cat results/smoke_test/sampling_summary.json
```

**这一步非常重要**：在提交长时间 batch job 之前，确认 GPU 采样能正常跑通。验证：
- 每个 pocket 目录下有 `<name>_generated.sdf` 和 `<name>_embeddings.npy`
- `sampling_summary.json` 中无 `"error"` 字段
- embedding shape 为 `(4, 128)`

---

## S3. 批量采样（核心 GPU 任务）

### S3.1 创建日志目录

```bash
mkdir -p slurm/logs
```

### S3.2 配置采样参数

打开 `slurm/sample_job.sh`，确认或修改以下配置：

```bash
# 关键参数
POCKET_LIST="data/splits/test_pockets.txt"   # 93 个 target
NUM_SAMPLES=64                                 # 每个 pocket 生成 64 个分子
NUM_STEPS=100                                  # DDPM 扩散步数（默认即可）
DEVICE="cuda"
PDBBIND_DIR="external/targetdiff/data/test_set"
TARGETDIFF_DIR="external/targetdiff"
OUTPUT_DIR="results/generated_molecules"
```

### S3.3 SLURM 资源配置

根据 HPC 集群类型调整 `slurm/sample_job.sh` 中的资源请求：

```bash
#SBATCH --partition=gpu          # 改为你的 GPU partition 名称
#SBATCH --gres=gpu:1             # 申请 1 块 GPU
#SBATCH --cpus-per-task=8        # 8 个 CPU 核心
#SBATCH --mem=32G                # 32GB 内存
#SBATCH --time=48:00:00          # 48 小时（保守估计）
```

**时间估算**：

| GPU 型号 | 每分子采样时间 (100 steps) | 93 pockets × 64 samples | 建议 `--time` |
|----------|--------------------------|------------------------|--------------|
| A100 80GB | ~20-30s | ~33-50h | `48:00:00` |
| V100 32GB | ~40-60s | ~66-99h | `72:00:00` 或多卡 |
| A6000 48GB | ~25-35s | ~41-58h | `60:00:00` |
| H100 80GB | ~15-20s | ~25-33h | `36:00:00` |

> ⚠️ **如果时间不够**：减少 `NUM_SAMPLES` 到 32，或先只跑前 50 个 pocket。

### S3.4 Conda 激活方式

确认 `slurm/sample_job.sh` 中的环境激活方式匹配你的 HPC：

```bash
# 常见写法，选择适用的一种：

# 写法 1: source activate（Miniconda）
source activate bayesdiff

# 写法 2: conda activate（需要先 conda init）
eval "$(conda shell.bash hook)"
conda activate bayesdiff

# 写法 3: module load + conda
module load anaconda3
conda activate bayesdiff
```

修改 `slurm/sample_job.sh` 第 40 行附近的激活命令。

### S3.5 提交任务

```bash
cd /scratch/$USER/BayesDiff
sbatch slurm/sample_job.sh
```

记录 Job ID：
```
Submitted batch job 1234567
```

### S3.6 监控任务

```bash
# 查看队列状态
squeue -u $USER

# 实时查看日志
tail -f slurm/logs/1234567_sample.log

# 查看 GPU 使用率（登录到计算节点）
srun --jobid=1234567 --pty nvidia-smi

# 检查已完成的 pocket 数
ls results/generated_molecules/ | grep -c "_embeddings.npy"
# 或
ls results/generated_molecules/*/  | grep -c "_generated.sdf"
```

### S3.7 断点续跑方案

如果任务超时或中断，已生成的结果不会丢失（每个 pocket 独立保存）。修改脚本跳过已完成的 pocket：

```bash
# 在 02_sample_molecules.py 中已有这个逻辑：
# 检查 pocket_out / f"{pdb_code}_embeddings.npy" 是否存在
# 如果满足某种条件可以跳过

# 更简单的做法：生成新的 pocket list，排除已完成的
cd /scratch/$USER/BayesDiff
python -c "
from pathlib import Path
all_pockets = Path('data/splits/test_pockets.txt').read_text().splitlines()
done = {d.name for d in Path('results/generated_molecules').iterdir() if d.is_dir() and list(d.glob('*_embeddings.npy'))}
remaining = [p for p in all_pockets if p.strip() and p.strip() not in done]
Path('data/splits/remaining_pockets.txt').write_text('\n'.join(remaining) + '\n')
print(f'Done: {len(done)}, Remaining: {len(remaining)}')
"

# 修改 sample_job.sh 使用新列表
POCKET_LIST="data/splits/remaining_pockets.txt" sbatch slurm/sample_job.sh
```

### S3.8 验收

```bash
# 检查完成数量
echo "Completed pockets:"
ls results/generated_molecules/ -d */ | wc -l  # 应为 93

# 检查 embedding 完整性
python -c "
import numpy as np
data = np.load('results/generated_molecules/all_embeddings.npz', allow_pickle=True)
print(f'Pockets: {len(data.files)}')
for k in list(data.files)[:3]:
    print(f'  {k}: {data[k].shape}')
# 每个 key 的 shape 应为 (64, 128)
"

# 检查采样汇总
cat results/generated_molecules/sampling_summary.json | python -m json.tool | head -20
```

**验收标准**：
- ✅ 93 个 pocket 目录，每个含 `*_generated.sdf` + `*_embeddings.npy`
- ✅ `all_embeddings.npz` 包含 93 个 key，每个 shape `(64, 128)`
- ✅ `sampling_summary.json` 中无大量 `"error"` 条目

---

## S4. Embedding 再提取（可选）

Step 2 在 `sample_job.sh` 中已自动执行。如果需要单独重新提取：

```bash
python scripts/03_extract_embeddings.py \
    --mode generated \
    --input_dir results/generated_molecules \
    --pdbbind_dir external/targetdiff/data/test_set \
    --targetdiff_dir external/targetdiff \
    --device cuda \
    --output results/generated_molecules/all_embeddings_reextracted.npz
```

如果不需要对比不同 encoder，可以跳过此步——S3 已经生成了 embedding。

---

## S5. GP 训练

### S5.1 准备训练数据

GP 的训练需要 (embedding, pKd) 对。数据来源：

- **embeddings**: S3 输出的 `all_embeddings.npz`（每个 pocket 取均值 → 1 个 93-dim 数据点变为 93×128）
- **labels**: `external/targetdiff/data/affinity_info.pkl`（自动加载）

### S5.2 训练 SVGP

```bash
cd /scratch/$USER/BayesDiff

python scripts/04_train_gp.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/gp_model \
    --n_inducing 128 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200
```

> **注意**：93 个 embedding 较少，使用 `--augment_to 200` 做高斯噪声数据增强。
> 如果后续使用 PDBbind全量 train split（~3400 complexes），则不需要 augment。

### S5.3 验收

```bash
# 检查模型文件
ls -la results/gp_model/
# 应有: gp_model.pt, train_meta.json, train_data.npz

# 查看训练元信息
cat results/gp_model/train_meta.json | python -m json.tool
```

**验收标准**：
- ✅ `gp_model.pt` 存在
- ✅ `train_meta.json` 中 `final_loss` 为有限值（通常 < 5.0）
- ✅ `train_data.npz` 包含 `X` (N×128) 和 `y` (N,)

---

## S6. 评估 + 校准

### S6.1 运行完整评估

```bash
python scripts/05_evaluate.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --gp_model results/gp_model/gp_model.pt \
    --gp_train_data results/gp_model/train_data.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/evaluation \
    --y_target 7.0 \
    --confidence_threshold 0.5 \
    --bootstrap_n 1000
```

### S6.2 多阈值评估

05_evaluate.py 内部已自动计算 y≥7.0 和 y≥8.0 两个阈值的结果。

### S6.3 验收

```bash
# 检查输出文件
ls results/evaluation/
# 应有: eval_metrics.json, eval_multi_threshold.json, per_pocket_results.json

# 查看主要指标
python -c "
import json
with open('results/evaluation/eval_metrics.json') as f:
    m = json.load(f)
print('=== BayesDiff Evaluation ===')
for k in ['ece', 'auroc', 'ef_1pct', 'hit_rate', 'spearman_rho', 'rmse', 'nll']:
    print(f'  {k}: {m[k]:.4f}')
"
```

**验收标准**：
- ✅ 7 项指标均为有限值（非 NaN）
- ✅ AUROC > 0.5（好于随机）
- ✅ ECE < 0.3（校准基本合理）
- ✅ `per_pocket_results.json` 包含 93 个条目

---

## S7. 消融实验

### S7.1 运行全部消融

```bash
python scripts/06_ablation.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --gp_model results/gp_model/gp_model.pt \
    --gp_train_data results/gp_model/train_data.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/ablation \
    --y_target 7.0 \
    --bootstrap_n 1000
```

这将运行 7 个变体：`full`, `A1`, `A2`, `A3`, `A4`, `A5`, `A7`。

### S7.2 验收

```bash
# 查看消融对比表
python -c "
import json
with open('results/ablation/ablation_summary.json') as f:
    d = json.load(f)
print(f'{'Method':<30} {'ECE':>8} {'AUROC':>8} {'EF@1%':>8} {'Spearman':>10} {'RMSE':>8}')
print('-' * 74)
for aid, r in d.items():
    desc = r.get('description', aid)
    print(f'{desc:<30} {r[\"ece\"]:>8.4f} {r[\"auroc\"]:>8.4f} {r[\"ef_1pct\"]:>8.2f} {r[\"spearman_rho\"]:>10.4f} {r[\"rmse\"]:>8.4f}')
"
```

**验收标准**：
- ✅ `ablation_summary.json` 包含 7 个 key（full + 6 ablations）
- ✅ Full 版本在大多数指标上优于消融变体
- ✅ `ablation_per_pocket.json` 每个变体含 93 条 per-pocket 结果

---

## S8. 结果回收

### S8.1 打包结果

```bash
cd /scratch/$USER/BayesDiff

# 打包所有结果（不含 SDF 文件，那些太大）
tar czf bayesdiff_results.tar.gz \
    results/evaluation/ \
    results/ablation/ \
    results/gp_model/gp_model.pt \
    results/gp_model/train_meta.json \
    results/generated_molecules/all_embeddings.npz \
    results/generated_molecules/sampling_summary.json
```

### S8.2 下载到 Mac

```bash
# 在 Mac 本地运行
scp <user>@<hpc>:/scratch/$USER/BayesDiff/bayesdiff_results.tar.gz ~/Downloads/

# 解压到项目目录
cd /Users/daiyizhe/Documents/GitHub/projects/BayesDiff
tar xzf ~/Downloads/bayesdiff_results.tar.gz
```

### S8.3 （可选）下载 SDF 文件

如果需要分析生成的分子结构：

```bash
# 只打包 SDF 文件
cd /scratch/$USER/BayesDiff
find results/generated_molecules -name "*.sdf" | tar czf bayesdiff_sdfs.tar.gz -T -

# 下载
scp <user>@<hpc>:/scratch/$USER/BayesDiff/bayesdiff_sdfs.tar.gz ~/Downloads/
```

---

## 附录 A：一键脚本

以下脚本可以把 S2–S7 串联起来，方便提交单个 SLURM job 跑完所有 GPU 后续步骤：

```bash
#!/bin/bash
#SBATCH --job-name=bayesdiff_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%j_full.log
#SBATCH --error=slurm/logs/%j_full.err

set -euo pipefail
cd /scratch/$USER/BayesDiff

# 环境
eval "$(conda shell.bash hook)"
conda activate bayesdiff

echo "=== BayesDiff Full Pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p slurm/logs results/generated_molecules results/gp_model results/evaluation results/ablation

# ── S2: 生成 pocket 列表 ────────────────────────
ls external/targetdiff/data/test_set/ > data/splits/test_pockets.txt
echo "Pockets: $(wc -l < data/splits/test_pockets.txt)"

# ── S3: 批量采样 ────────────────────────────────
echo ""
echo ">>> [S3] Sampling molecules (93 pockets × 64 samples × 100 steps)..."
python scripts/02_sample_molecules.py \
    --pocket_list data/splits/test_pockets.txt \
    --pdbbind_dir external/targetdiff/data/test_set \
    --targetdiff_dir external/targetdiff \
    --num_samples 64 \
    --num_steps 100 \
    --device cuda \
    --output_dir results/generated_molecules

# ── S5: GP 训练 ─────────────────────────────────
echo ""
echo ">>> [S5] Training GP oracle..."
python scripts/04_train_gp.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/gp_model \
    --n_inducing 128 \
    --n_epochs 200 \
    --batch_size 64 \
    --augment_to 200

# ── S6: 评估 ────────────────────────────────────
echo ""
echo ">>> [S6] Running evaluation..."
python scripts/05_evaluate.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --gp_model results/gp_model/gp_model.pt \
    --gp_train_data results/gp_model/train_data.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/evaluation \
    --y_target 7.0 \
    --bootstrap_n 1000

# ── S7: 消融 ────────────────────────────────────
echo ""
echo ">>> [S7] Running ablation study..."
python scripts/06_ablation.py \
    --embeddings results/generated_molecules/all_embeddings.npz \
    --gp_model results/gp_model/gp_model.pt \
    --gp_train_data results/gp_model/train_data.npz \
    --affinity_pkl external/targetdiff/data/affinity_info.pkl \
    --output results/ablation \
    --y_target 7.0 \
    --bootstrap_n 1000

# ── 结果汇总 ────────────────────────────────────
echo ""
echo "=== Pipeline Complete ==="
echo "End time: $(date)"
echo ""
echo "Results:"
ls -la results/evaluation/
ls -la results/ablation/
echo ""
echo "Sampling summary:"
python -c "
import json
with open('results/generated_molecules/sampling_summary.json') as f:
    s = json.load(f)
n_ok = sum(1 for t in s['timing'] if 'error' not in t)
print(f'  Sampled: {n_ok}/{s[\"n_pockets\"]} pockets')
"
echo ""
echo "Evaluation metrics:"
python -c "
import json
with open('results/evaluation/eval_metrics.json') as f:
    m = json.load(f)
for k in ['ece','auroc','ef_1pct','hit_rate','spearman_rho','rmse','nll']:
    print(f'  {k}: {m[k]:.4f}')
"
```

将上述内容保存为 `slurm/full_pipeline_job.sh`，提交：

```bash
sbatch slurm/full_pipeline_job.sh
```

---

## 附录 B：多 GPU 并行方案

如果 HPC 排队时间短但单卡时间不够（V100 可能需要 >72h），可以拆分 pocket list 并行：

```bash
# 将 93 个 pocket 拆为 4 份
split -n l/4 data/splits/test_pockets.txt data/splits/part_

# 为每份提交独立任务
for part in data/splits/part_*; do
    sbatch --export=POCKET_LIST=$part,OUTPUT_DIR=results/gen_$(basename $part) slurm/sample_job.sh
done

# 采样完成后，合并所有 embedding
python -c "
import numpy as np
from pathlib import Path
all_emb = {}
for d in sorted(Path('results').glob('gen_part_*')):
    for npz in d.glob('**/all_embeddings.npz'):
        data = np.load(npz, allow_pickle=True)
        all_emb.update({k: data[k] for k in data.files})
np.savez('results/generated_molecules/all_embeddings.npz', **all_emb)
print(f'Merged: {len(all_emb)} pockets')
"
```

---

## 附录 C：Troubleshooting

### C1. GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**解决**：
```bash
# 减少每批采样数
NUM_SAMPLES=32 sbatch slurm/sample_job.sh

# 或在 02_sample_molecules.py 中设置 batch_size
python scripts/02_sample_molecules.py --batch_size 16 ...
```

### C2. TargetDiff import 错误

```
ModuleNotFoundError: No module named 'torch_scatter'
```

**解决**：确认 `external/targetdiff/torch_scatter.py` shim 文件存在，且 `bayesdiff/sampler.py` 中 `sys.path.insert(0, str(TARGETDIFF_DIR))` 在 import 之前。

### C3. checkpoint 加载警告

```
FutureWarning: You are using `torch.load` with `weights_only=False`
```

**可忽略**：这是预期行为——TargetDiff checkpoint 包含 `easydict.EasyDict` 对象，必须 `weights_only=False`。

### C4. 任务中途被杀

```
slurmstepd: error: *** JOB 1234567 ON node01 CANCELLED AT ... DUE TO TIME LIMIT
```

**解决**：使用 S3.7 断点续跑方案。已完成的 pocket 不会丢失。

### C5. No GPU available

```
RuntimeError: No CUDA GPUs are available
```

**解决**：
```bash
# 检查 SLURM 是否分配了 GPU
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# 如果为空，检查 #SBATCH --gres=gpu:1 是否正确
# 部分集群需要 #SBATCH --gpus=1 或 #SBATCH --gpus-per-node=1
```

### C6. 结果文件损坏或不完整

```bash
# 验证 NPZ 文件完整性
python -c "
import numpy as np
try:
    d = np.load('results/generated_molecules/all_embeddings.npz', allow_pickle=True)
    print(f'OK: {len(d.files)} keys')
    for k in d.files:
        _ = d[k].shape  # 触发实际读取
    print('All arrays readable')
except Exception as e:
    print(f'CORRUPT: {e}')
"
```

---

## 附录 D：资源文件清单

### HPC 需要的文件

| 文件/目录 | 来源 | 大小 | 说明 |
|-----------|------|------|------|
| `bayesdiff/` (9 .py) | git clone | ~100KB | 核心库 |
| `scripts/` (8 .py) | git clone | ~50KB | 脚本 |
| `slurm/sample_job.sh` | git clone | ~2KB | SLURM 脚本 |
| `external/targetdiff/` | git clone | ~50MB | TargetDiff 代码 |
| `external/targetdiff/pretrained_models/pretrained_diffusion.pt` | 下载 | ~33MB | 预训练权重 |
| `external/targetdiff/data/test_set/` (93 targets) | 下载/上传 | ~500MB | 蛋白口袋 + 参考配体 |
| `external/targetdiff/data/affinity_info.pkl` | 下载/上传 | ~20MB | pKd 标签 |

### HPC 产出的文件（需下载回 Mac）

| 文件 | 大小估算 | 用途 |
|------|----------|------|
| `results/generated_molecules/all_embeddings.npz` | ~50MB | 93×64×128 float32 |
| `results/generated_molecules/sampling_summary.json` | ~50KB | 采样日志 |
| `results/generated_molecules/*/*.sdf` | ~2GB | 生成的分子结构（可选） |
| `results/gp_model/gp_model.pt` | ~5MB | 训练好的 GP 模型 |
| `results/gp_model/train_meta.json` | ~2KB | 训练元信息 |
| `results/evaluation/eval_metrics.json` | ~2KB | 最终评估指标 |
| `results/evaluation/per_pocket_results.json` | ~50KB | 逐 pocket 结果 |
| `results/ablation/ablation_summary.json` | ~10KB | 消融对比表 |
| `results/ablation/ablation_per_pocket.json` | ~200KB | 消融逐 pocket 结果 |

---

## 附录 E：检查清单（提交前逐项确认）

- [ ] Conda 环境创建成功，`torch.cuda.is_available()` → True
- [ ] BayesDiff 代码已 clone，`scripts/_check_deps.py` 全部通过
- [ ] TargetDiff 已 clone + 权重已下载，`pretrained_diffusion.pt` 存在
- [ ] 测试集数据（93 targets）已部署到 `external/targetdiff/data/test_set/`
- [ ] `affinity_info.pkl` 已就位
- [ ] `data/splits/test_pockets.txt` 已生成（93 行）
- [ ] 冒烟测试通过（5 pockets × 4 samples on GPU）
- [ ] `slurm/sample_job.sh` 中 partition / conda 激活方式已适配
- [ ] `slurm/logs/` 目录已创建
- [ ] 磁盘空间充足（至少 5GB for results）
