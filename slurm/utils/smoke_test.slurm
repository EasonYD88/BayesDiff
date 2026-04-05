#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%j_smoke_test.out
#SBATCH --error=slurm/logs/%j_smoke_test.err

set -euo pipefail

echo "=== S2.2 Smoke Test ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff

echo "--- Quick GPU check ---"
python -c "import torch; print(f'cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "--- Smoke Test: 5 pockets x 4 samples x 20 steps (GPU) ---"
python scripts/utils/run_full_pipeline.py --mode debug --n_pockets 5 --num_samples 4 --num_steps 20 --device cuda 2>&1

echo ""
echo "=== Smoke Test Complete ==="
