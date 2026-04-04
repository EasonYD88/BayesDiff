#!/bin/bash
#SBATCH --job-name=schnet_a5
#SBATCH --output=slurm/logs/schnet_a5_%j.out
#SBATCH --error=slurm/logs/schnet_a5_%j.out
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_281_chemistry

echo "=== SchNet + A5 Subsample Ablation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

echo ""
echo "=== Phase 1: SchNet QM9-pretrained embedding extraction ==="
python scripts/26c_extract_schnet.py
echo "SchNet exit code: $?"

echo ""
echo "=== Phase 2: Re-run comparison with SchNet data ==="
python scripts/26d_compare_embeddings.py
echo "Comparison exit code: $?"

echo ""
echo "=== Phase 3: A5 Subsample Ablation (50mol study) ==="
python scripts/28c_subsample_ablation.py
echo "A5 exit code: $?"

echo ""
echo "=== ALL DONE ==="
echo "Date: $(date)"
