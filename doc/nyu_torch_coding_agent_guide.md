# NYU Torch 使用指南（给 Coding Agent）

基于 NYU Research Technology Services 的 Torch 文档整理，面向“可执行操作”而非概念介绍。  
整理日期：2026-03-03（美国东部时间）。

## 1. 先决条件（必须满足）

1. 有有效 NYU HPC 账号（非 faculty 账号通常按年续期）。
2. 有 **active allocation**（HPC project management portal 里分配成功），否则无法提交作业。
3. 在 NYU 校园网或已连 NYU VPN。
4. 可完成 Torch 登录时的微软设备登录 + MFA。

关键点：  
`HPC account != 可跑作业`，跑作业还需要 allocation 对应的 `SLURM account`（提交时用 `--account`）。

## 2. 账号与配额流程（Agent 执行顺序）

1. 申请/续期 HPC 账号：`NYU Identity Management service`。
2. 进入 HPC project management portal：
   - 创建项目；
   - 申请 allocation；
   - 查看项目对应 Slurm account；
   - 添加成员并配置 allocation 访问。
3. 如果是远程操作，先连 NYU VPN 再访问上述门户。

## 3. 登录 Torch（SSH + 2FA）

推荐本地 `~/.ssh/config`：

```sshconfig
Host dtn.torch.hpc.nyu.edu
  User <NETID>
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ForwardAgent yes
  UserKnownHostsFile /dev/null
  LogLevel ERROR

Host torch login.torch.hpc.nyu.edu
  Hostname login.torch.hpc.nyu.edu
  User <NETID>
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ForwardAgent yes
  UserKnownHostsFile /dev/null
  LogLevel ERROR
```

登录命令：

```bash
ssh <NETID>@login.torch.hpc.nyu.edu
```

注意事项：

1. Torch 文档明确写了 **不支持 SSH keys**（按其安全策略）。
2. 首次/偶发登录会触发设备登录：终端给出 PIN，去 `https://microsoft.com/devicelogin` 完成 MFA，再回终端回车。
3. 若 SSH 超时，增大 `ServerAliveInterval`。

## 4. 存储与目录策略（强约束）

常用空间（文档中的 Torch 对比表）：

1. `/home` (`$HOME`)：50GB / 30K 文件，备份，有长期配置价值。
2. `/scratch` (`$SCRATCH`)：5TB / 5M 文件，不备份，**60 天未访问会清理**。
3. `/archive` (`$ARCHIVE`)：2TB / 20K 文件，适合长期归档。
4. RPS（Research Project Space）：共享空间，按 TB-year / inode-year 计费。

实践建议：

1. 代码与轻量配置放 `/home`。
2. 训练数据、中间结果、环境目录放 `/scratch`（并自行做归档/备份）。
3. 定期用 `myquota` 检查配额与 inode。

数据合规：

1. Torch 环境只适合 Moderate Risk Data。
2. PII/ePHI/CUI 等高风险数据不要放 Torch（应使用 SRDE）。

## 5. 数据传输（大文件优先 Globus）

优先级：

1. 大规模数据：**Globus（推荐）**。
2. 命令行小/中等规模：`rsync/scp` + `dtn.torch.hpc.nyu.edu`。

核心规则：

1. 不要在登录节点做大流量传输。
2. 用 DTN：`dtn.torch.hpc.nyu.edu`（对应 DTN 节点）。

示例：

```bash
# 本地 -> Torch scratch
rsync -avz -e ssh ./data/ <NETID>@dtn.torch.hpc.nyu.edu:/scratch/<NETID>/data/

# Torch 内部先切 DTN 再拷贝
ssh dtn.torch.hpc.nyu.edu
rsync -av /scratch/<NETID>/project1 /rw/<share_name>/
logout
```

Globus 关键点：

1. Torch 服务器 endpoint/collection：`nyu#torch`。
2. 个人电脑需安装 Globus Connect Personal。

## 6. 软件环境策略（给 Agent 的默认选择）

默认选择（官方建议）：

1. **Apptainer/Singularity + overlay**（最稳、可复现、适配 GPU）。

替代方案（必要时）：

1. `venv` / `virtualenv`（轻量 Python）。
2. Conda（建议 prefix env；文档建议 `source activate`）。

### 6.1 Singularity + overlay（推荐）

```bash
# 先拿交互资源，避免在 login 节点重活
srun --pty -c 2 --mem=5GB /bin/bash

# 进入容器并激活 overlay 内环境
singularity exec --nv \
  --overlay /scratch/<NETID>/pytorch-example/my_pytorch.ext3:rw \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash

source /ext3/env.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 6.2 Conda（prefix 环境）

```bash
module purge
module load anaconda3/2024.02
conda create -p ./penv python
source activate ./penv
```

补充：

1. 文档建议优先 prefix env，便于复现。
2. 若 `~/.local` 干扰包版本，可设 `export PYTHONNOUSERSITE=True`。

### 6.3 venv（轻量）

```bash
module avail python
module load python/intel/3.8.6
mkdir -p /scratch/$USER/my_project && cd /scratch/$USER/my_project
python -m venv venv
source venv/bin/activate
pip install -U pip
```

## 7. Slurm 提交规范（Torch 特有）

Torch 官方页面给出的关键约束：

1. 每个作业都要带 `--account=<SLURM_ACCOUNT>`。
2. 一般不要手动指定 partition（除 preemption 用法）。
3. 低 GPU 利用率作业可能被系统主动取消（策略较严格）。
4. 用户 GPU quota（文档当前描述）：walltime < 48h 的作业，单用户总 GPU 上限 24。

### 7.1 单卡模板（可直接改）

```bash
#!/bin/bash
#SBATCH --job-name=bayesdiff-sgpu
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
mkdir -p logs

srun singularity exec --nv \
  --overlay /scratch/<NETID>/pytorch-example/my_pytorch.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python train.py"
```

### 7.2 DDP 多卡/多节点模板（Torch 页面规则）

硬规则：`--ntasks-per-node == --gres=gpu:<N>`（每 task 对应 1 GPU）。

```bash
#!/bin/bash
#SBATCH --job-name=bayesdiff-ddp
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
mkdir -p logs

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun singularity exec --nv \
  --overlay /scratch/<NETID>/pytorch-example/my_pytorch.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python train_ddp.py"
```

官方实践建议：  
先把单 GPU 吃满并优化，再扩到 DDP；不要做“多节点但每节点只用 1 GPU”的低效配置。

### 7.3 抢占队列（Preemptible）

需要时可在 `#SBATCH --comment` 中声明，例如：

```bash
# 常规+可抢占混合
#SBATCH --comment="preemption=yes;requeue=true"

# 仅抢占分区
#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true"
```

官方说明要点：

1. 抢占资格通常在运行 1 小时后生效。
2. 建议作业支持 checkpoint/restart。

## 8. 监控、排队与排错命令

```bash
# 队列和节点
squeue --me
sinfo --Format=Partition,GRES,CPUs,Features:26,NodeList

# 历史账单/状态
sacct --format=JobID,JobName,State,AllocCPUS,Elapsed,Start,End

# 取消
scancel <job_id>

# 资源效率（常用）
seff <job_id>
```

调试建议：

1. 在脚本里加 `printenv | grep -i slurm | sort`，核对环境变量。
2. DDP 场景检查 `WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`。
3. 先用小数据 + 短时限 + 单卡跑通，再扩资源。

## 9. Open OnDemand（可视化备选）

入口：`https://ood.torch.hpc.nyu.edu`（需校园网或 VPN）。  
可做文件管理、作业监控、交互式 app（Jupyter/RStudio/Desktop）。

日志路径（终端）：

```bash
/home/$USER/ondemand/data/sys/dashboard/batch_connect/sys/
```

## 10. 给 Coding Agent 的执行 SOP（推荐）

1. 检查 VPN 与登录可达性（`ssh torch`）。
2. 确认 `SLURM_ACCOUNT` 可用（无则停止并提示用户去 portal 申请 allocation）。
3. 确认工作目录在 `/scratch/<NETID>/<project>`。
4. 先交互式验证环境（`torch.cuda.is_available()`）。
5. 提交单卡基线作业并记录 `seff`、`sacct`。
6. 利用率正常后再上 DDP，且满足 `ntasks-per-node == gpus-per-node`。
7. 对长任务启用 checkpoint，并根据需要启用 preemption + requeue。
8. 结果定期同步至 `/archive` 或 RPS，避免 scratch 清理风险。

## 11. 参考来源

1. 主入口: https://services.rt.nyu.edu/docs/hpc/getting_started/intro/
2. 账号与续期: https://services.rt.nyu.edu/docs/hpc/getting_started/getting_and_renewing_an_account/
3. 项目门户: https://services.rt.nyu.edu/docs/hpc/getting_started/hpc_project_management_portal/
4. 连接 Torch: https://services.rt.nyu.edu/docs/hpc/connecting_to_hpc/connecting_to_hpc/
5. 存储: https://services.rt.nyu.edu/docs/hpc/storage/intro_and_data_management/
6. 数据传输: https://services.rt.nyu.edu/docs/hpc/storage/data_transfers/
7. Globus: https://services.rt.nyu.edu/docs/hpc/storage/globus/
8. 提交作业: https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/
9. Slurm 命令: https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_main_commands/
10. 工具与环境: https://services.rt.nyu.edu/docs/hpc/tools_and_software/intro/
11. Python venv: https://services.rt.nyu.edu/docs/hpc/tools_and_software/python_packages_with_virtual_environments/
12. Conda: https://services.rt.nyu.edu/docs/hpc/tools_and_software/conda_environments/
13. PyTorch 单卡: https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/pytorch_intro/
14. PyTorch DDP: https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/pytorch_dpp/
15. OOD: https://services.rt.nyu.edu/docs/hpc/ood/ood_intro/
16. Torch 规格: https://services.rt.nyu.edu/docs/hpc/spec_sheet/
