# BayesDiff NYU Torch 定制 HPC 指南（给 Coding Agent）

更新时间：2026-03-03（America/New_York）

## 1. 适用范围

本指南只覆盖 `projects/BayesDiff` 当前真实使用的 HPC 流程（NYU Torch）：

1. GPU 主任务：`scripts/02_sample_molecules.py`
2. GPU 可选任务：`scripts/03_extract_embeddings.py`（重提取）
3. CPU 任务：`scripts/04_train_gp.py`、`scripts/05_evaluate.py`、`scripts/06_ablation.py`

不改代码、不改现有 Slurm 脚本，仅提供可直接执行的 runbook。

## 2. Agent 执行参数契约

统一使用以下变量（文档层接口）：

| 变量 | 是否必填 | 推荐值（本项目） | 说明 |
|---|---|---|---|
| `NETID` | 是 | `<your_netid>` | NYU 登录用户 |
| `SLURM_ACCOUNT` | 是 | `<your_slurm_account>` | Torch 必填，提交作业必须有 |
| `PDBBIND_DIR` | 是 | `external/targetdiff/data/test_set` | 本项目默认数据布局 |
| `TARGETDIFF_DIR` | 是 | `external/targetdiff` | TargetDiff clone 根目录 |
| `POCKET_LIST` | 是 | `data/splits/test_pockets.txt` | 每行一个 pocket 名 |
| `NUM_SAMPLES` | 是 | `64` | 每个 pocket 采样分子数 |
| `NUM_STEPS` | 是 | `100` | DDPM 步数 |
| `OUTPUT_DIR` | 是 | `results/generated_molecules` | 采样输出目录 |
| `AFFINITY_PKL` | 是 | `external/targetdiff/data/affinity_info.pkl` | GP/eval 标签来源 |

推荐先在会话里导出：

```bash
export NETID="<your_netid>"
export SLURM_ACCOUNT="<your_slurm_account>"
export PDBBIND_DIR="external/targetdiff/data/test_set"
export TARGETDIFF_DIR="external/targetdiff"
export POCKET_LIST="data/splits/test_pockets.txt"
export NUM_SAMPLES="64"
export NUM_STEPS="100"
export OUTPUT_DIR="results/generated_molecules"
export AFFINITY_PKL="external/targetdiff/data/affinity_info.pkl"
```

## 3. 本项目 HPC 功能地图

### 3.1 资源分配

1. 必须上 GPU：`02_sample_molecules.py`
2. 可选上 GPU：`03_extract_embeddings.py`（若对已有 SDF 重提取）
3. 建议放 CPU：`04/05/06`（不应占用 GPU 资源）

### 3.2 产物链（验收主线）

```text
results/generated_molecules/all_embeddings.npz
-> results/gp_model/gp_model.pt + results/gp_model/train_data.npz
-> results/evaluation/eval_metrics.json
-> results/ablation/ablation_summary.json
```

## 4. NYU Torch 预检清单（最小必要）

在登录节点执行：

```bash
ssh ${NETID}@login.torch.hpc.nyu.edu
```

### 4.1 作业权限与配额

```bash
myquota
sacctmgr -n show assoc user=$USER format=Account,User,Partition 2>/dev/null | head
```

成功判据：

1. `myquota` 可返回配额信息
2. 你能确认可用 `SLURM_ACCOUNT`

### 4.2 项目目录与依赖

```bash
cd /scratch/${USER}/BayesDiff
python scripts/_check_deps.py
```

成功判据：关键依赖可导入（至少 `torch`, `torch_geometric`, `rdkit`, `easydict`, `yaml`）。

### 4.3 关键文件存在性

```bash
ls -ld "${TARGETDIFF_DIR}" \
      "${PDBBIND_DIR}" \
      "${AFFINITY_PKL}" \
      "data/splits/test_pockets.txt"

python - <<'PY'
from pathlib import Path
import numpy as np
p = Path("data/splits/test_pockets.txt")
print("pocket_lines:", len([x for x in p.read_text().splitlines() if x.strip()]))
PY
```

成功判据：以上路径均存在，`test_pockets.txt` 行数符合预期（当前计划通常为 93）。

## 5. 作业接口说明

项目当前有两条作业入口：

1. `slurm/sample_job.sh`：GPU 主入口（推荐）
2. `slurm/full_pipeline_job.sh`：兼容一键入口（会占着 GPU 跑后续 CPU 步骤，不是默认推荐）

NYU Torch 要求作业带 account。当前脚本头部未固化 `#SBATCH --account`，所以提交时必须显式加：

```bash
sbatch --account="${SLURM_ACCOUNT}" ...
```

## 6. 模板 A：Smoke Test（首次连通性验证）

目标：快速验证 `02_sample_molecules.py` 在 Torch 上可跑通。

### 6.1 准备 5-pocket 列表

```bash
cd /scratch/${USER}/BayesDiff
mkdir -p data/splits slurm/logs
head -5 data/splits/test_pockets.txt > data/splits/smoke_pockets.txt
wc -l data/splits/smoke_pockets.txt
```

### 6.2 提交作业

```bash
sbatch \
  --account="${SLURM_ACCOUNT}" \
  --export=ALL,POCKET_LIST=data/splits/smoke_pockets.txt,NUM_SAMPLES=4,NUM_STEPS=100,DEVICE=cuda,PDBBIND_DIR=${PDBBIND_DIR},TARGETDIFF_DIR=${TARGETDIFF_DIR},OUTPUT_DIR=results/smoke_generated \
  slurm/sample_job.sh
```

### 6.3 成功判据

```bash
ls -lh results/smoke_generated/sampling_summary.json
ls -lh results/smoke_generated/all_embeddings.npz

python - <<'PY'
import json, numpy as np
s = json.load(open("results/smoke_generated/sampling_summary.json"))
d = np.load("results/smoke_generated/all_embeddings.npz", allow_pickle=True)
print("n_pockets:", s["n_pockets"], "n_sampled:", s["n_sampled"], "keys:", len(d.files))
print("sample_shape_example:", d[d.files[0]].shape if d.files else None)
PY
```

验收：

1. `sampling_summary.json` 与 `all_embeddings.npz` 存在
2. `keys > 0`
3. 每个 key 的形状应为 `(NUM_SAMPLES, 128)`，此模板下即 `(4, 128)`

## 7. 模板 B：正式采样（93 pockets × 64）

目标：产出正式 `all_embeddings.npz`。

### 7.1 提交作业

```bash
sbatch \
  --account="${SLURM_ACCOUNT}" \
  --export=ALL,POCKET_LIST=${POCKET_LIST},NUM_SAMPLES=${NUM_SAMPLES},NUM_STEPS=${NUM_STEPS},DEVICE=cuda,PDBBIND_DIR=${PDBBIND_DIR},TARGETDIFF_DIR=${TARGETDIFF_DIR},OUTPUT_DIR=${OUTPUT_DIR} \
  slurm/sample_job.sh
```

### 7.2 成功判据

```bash
python - <<'PY'
import numpy as np, json
from pathlib import Path
plist = [x.strip() for x in Path("data/splits/test_pockets.txt").read_text().splitlines() if x.strip()]
summary = json.load(open("results/generated_molecules/sampling_summary.json"))
d = np.load("results/generated_molecules/all_embeddings.npz", allow_pickle=True)
print("expected_pockets:", len(plist))
print("n_sampled(summary):", summary["n_sampled"])
print("npz_keys:", len(d.files))
bad = [k for k in d.files if d[k].shape != (64, 128)]
print("bad_shape_count:", len(bad))
PY
```

验收：

1. `results/generated_molecules/all_embeddings.npz` 存在
2. `bad_shape_count == 0`
3. `npz_keys` 接近期望 pocket 数；若少于期望，进入第 9 节续跑流程

## 8. 模板 C：后处理评估（CPU 独立提交）

默认策略：GPU 与 CPU 任务分离，避免 GPU 空占。

```bash
sbatch \
  --account="${SLURM_ACCOUNT}" \
  --partition=cpu \
  --cpus-per-task=8 \
  --mem=32G \
  --time=06:00:00 \
  --output=slurm/logs/%j_post.log \
  --error=slurm/logs/%j_post.err \
  --wrap='
set -euo pipefail
cd /scratch/${USER}/BayesDiff
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate bayesdiff 2>/dev/null || source activate bayesdiff 2>/dev/null || true
python scripts/04_train_gp.py \
  --embeddings results/generated_molecules/all_embeddings.npz \
  --affinity_pkl external/targetdiff/data/affinity_info.pkl \
  --output results/gp_model \
  --n_inducing 128 --n_epochs 200 --batch_size 64 --augment_to 200
python scripts/05_evaluate.py \
  --embeddings results/generated_molecules/all_embeddings.npz \
  --gp_model results/gp_model/gp_model.pt \
  --gp_train_data results/gp_model/train_data.npz \
  --affinity_pkl external/targetdiff/data/affinity_info.pkl \
  --output results/evaluation \
  --y_target 7.0 --confidence_threshold 0.5 --bootstrap_n 1000
python scripts/06_ablation.py \
  --embeddings results/generated_molecules/all_embeddings.npz \
  --gp_model results/gp_model/gp_model.pt \
  --gp_train_data results/gp_model/train_data.npz \
  --affinity_pkl external/targetdiff/data/affinity_info.pkl \
  --output results/ablation \
  --y_target 7.0 --bootstrap_n 1000
'
```

成功判据：

```bash
ls -lh results/gp_model/gp_model.pt results/gp_model/train_data.npz
ls -lh results/evaluation/eval_metrics.json results/ablation/ablation_summary.json
```

并检查 JSON 字段：

1. `eval_metrics.json` 至少包含：`ece`, `auroc`, `ef_1pct`, `hit_rate`, `spearman_rho`, `rmse`, `nll`
2. `ablation_summary.json` 包含 `full`, `A1`, `A2`, `A3`, `A4`, `A5`, `A7`（或你指定子集）

## 9. 失败续跑与故障恢复

### 9.1 自动生成 `remaining_pockets.txt`

当全量作业中断或部分失败时：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
all_pockets = [x.strip() for x in Path("data/splits/test_pockets.txt").read_text().splitlines() if x.strip()]
done = set()
npz = Path("results/generated_molecules/all_embeddings.npz")
if npz.exists():
    d = np.load(npz, allow_pickle=True)
    done = set(d.files)
remaining = [p for p in all_pockets if p not in done]
Path("data/splits/remaining_pockets.txt").write_text("\n".join(remaining) + ("\n" if remaining else ""))
print("total:", len(all_pockets), "done:", len(done), "remaining:", len(remaining))
PY
```

提交续跑：

```bash
sbatch \
  --account="${SLURM_ACCOUNT}" \
  --export=ALL,POCKET_LIST=data/splits/remaining_pockets.txt,NUM_SAMPLES=64,NUM_STEPS=100,DEVICE=cuda,PDBBIND_DIR=${PDBBIND_DIR},TARGETDIFF_DIR=${TARGETDIFF_DIR},OUTPUT_DIR=results/generated_molecules_retry \
  slurm/sample_job.sh
```

### 9.2 分片并行提交（加速）

```bash
split -n l/4 data/splits/test_pockets.txt data/splits/part_
for part in data/splits/part_*; do
  out="results/gen_$(basename "$part")"
  sbatch \
    --account="${SLURM_ACCOUNT}" \
    --export=ALL,POCKET_LIST=${part},NUM_SAMPLES=64,NUM_STEPS=100,DEVICE=cuda,PDBBIND_DIR=${PDBBIND_DIR},TARGETDIFF_DIR=${TARGETDIFF_DIR},OUTPUT_DIR=${out} \
    slurm/sample_job.sh
done
```

合并分片 embeddings：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
merged = {}
for d in Path("results").glob("gen_part_*"):
    f = d / "all_embeddings.npz"
    if not f.exists():
        continue
    z = np.load(f, allow_pickle=True)
    for k in z.files:
        merged[k] = z[k]
Path("results/generated_molecules").mkdir(parents=True, exist_ok=True)
np.savez("results/generated_molecules/all_embeddings.npz", **merged)
print("merged_keys:", len(merged))
PY
```

### 9.3 常见故障闭环（按顺序排）

1. Checkpoint 缺失  
先检查以下任一路径是否有权重：
`external/targetdiff/pretrained_model.pt`、`external/targetdiff/pretrained_models/pretrained_diffusion.pt`、`external/targetdiff/checkpoints/pretrained_model.pt`

2. 数据路径布局不匹配  
确认 `PDBBIND_DIR=external/targetdiff/data/test_set`，并且 `data/splits/test_pockets.txt` 每行对应一个子目录名。

3. GPU 不可见  
检查作业请求是否是 GPU（`--gres=gpu:1` 在脚本里），并在作业日志核对 `nvidia-smi` 输出。

4. `Generated 0/X valid molecules` 过多  
优先排查顺序：
`NUM_STEPS` 是否过低（避免调试用 20 步上正式任务） -> pocket/权重路径是否正确 -> 先对单个 pocket 做 smoke 验证再全量提交。

## 10. 资源与效率策略（本项目默认）

1. 默认单卡跑采样；不先做多卡并行改造。
2. 默认 GPU/CPU 分离：先 `sample_job.sh`，后 CPU `sbatch --wrap` 跑 `04/05/06`。
3. `full_pipeline_job.sh` 只作兼容一键入口，不作默认流程（因为它在后处理阶段仍占 GPU 配额）。

### 10.1 `NUM_SAMPLES` / `NUM_STEPS` 降配建议

1. 首次 smoke：`NUM_SAMPLES=4`, `NUM_STEPS=100`
2. 时间吃紧：先降 `NUM_SAMPLES`（64 -> 32 -> 16），再考虑分片并行
3. 不建议正式任务低于 `NUM_STEPS=100`

## 11. 最小监控命令集

```bash
squeue -u $USER
sacct -j <job_id> --format=JobID,State,Elapsed,AllocTRES
seff <job_id>
tail -f slurm/logs/<jobid>_sample.log
tail -f slurm/logs/<jobid>_sample.err
```

## 12. 结果回收与质量验收

### 12.1 推荐回收文件

```text
results/generated_molecules/all_embeddings.npz
results/generated_molecules/sampling_summary.json
results/gp_model/gp_model.pt
results/gp_model/train_data.npz
results/evaluation/eval_metrics.json
results/ablation/ablation_summary.json
slurm/logs/*.log
slurm/logs/*.err
```

### 12.2 质量验收清单

1. `all_embeddings.npz` key 数与预期 pocket 数一致或可解释
2. 所有 embedding shape 一致（正式任务通常 `(64, 128)`）
3. `eval_metrics.json` 和 `ablation_summary.json` 字段完整
4. 续跑/分片后合并结果无 key 覆盖异常（必要时统计重复 key）

## 13. 延伸阅读

1. `doc/hpc_execution_plan.md`：完整阶段执行计划
2. `doc/plan_opendata.md`：公开数据策略与实验背景
3. `doc/nyu_torch_coding_agent_guide.md`：NYU Torch 通用使用规则
