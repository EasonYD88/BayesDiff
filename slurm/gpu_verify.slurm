#!/bin/bash
#SBATCH --job-name=gpu-verify
#SBATCH --account=torch_pr_281_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%j_gpu_verify.out
#SBATCH --error=slurm/logs/%j_gpu_verify.err

set -euo pipefail

echo "=== S0.3 GPU Verification ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
echo ""

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

echo "--- PyTorch GPU Test ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory: {props.total_mem / 1024**3:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
    x = torch.randn(2000, 2000, device='cuda')
    y = x @ x.T
    print(f'GPU matmul test: OK ({x.shape})')
    del x, y
    torch.cuda.empty_cache()
else:
    print('ERROR: CUDA not available!')
    exit(1)
import torch_geometric
print(f'PyG: {torch_geometric.__version__}')
import gpytorch
print(f'GPyTorch: {gpytorch.__version__}')
print()
print('ALL GPU CHECKS PASSED')
"

echo ""
echo "--- S2.2 Smoke Test (5 pockets x 4 samples) ---"
python scripts/run_full_pipeline.py --mode debug --n_pockets 5 --num_samples 4 --num_steps 20 2>&1 || echo "Smoke test failed (non-fatal)"

echo ""
echo "=== GPU Verification Complete ==="
