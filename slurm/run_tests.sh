#!/bin/bash
#SBATCH --job-name=pytest_oracle
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_281_chemistry
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/pytest_oracle_%j.out
#SBATCH --error=slurm/logs/pytest_oracle_%j.err

cd /scratch/yd2915/BayesDiff
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

FAIL=0

echo "=== Unit Tests: test_hybrid_oracle.py ==="
python -m pytest tests/stage2/test_hybrid_oracle.py -v --tb=short 2>&1 || FAIL=1

echo ""
echo "=== Integration Tests: test_hybrid_integration.py ==="
python -m pytest tests/stage2/test_hybrid_integration.py -v --tb=short 2>&1 || FAIL=1

exit $FAIL

echo ""
echo "=== Done ==="
