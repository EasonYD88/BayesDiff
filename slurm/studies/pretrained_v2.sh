#!/bin/bash
#SBATCH --job-name=pretrained_v2
#SBATCH --output=slurm/logs/pretrained_v2_%j.out
#SBATCH --error=slurm/logs/pretrained_v2_%j.out
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_281_chemistry

echo "=== PRE-TRAINED EMBEDDING v2: Uni-Mol + SchNet + Comparison ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

# Verify deps
python -c "from unimol_tools import UniMolRepr; print('UniMol OK')"
python -c "import ase; print('ASE OK')"

echo ""
echo "=== Phase B: Uni-Mol 512-dim embedding extraction ==="
python scripts/studies/embedding_unimol.py
echo "Phase B exit code: $?"

echo ""
echo "=== Phase C: SchNet QM9-pretrained embedding extraction ==="
python scripts/studies/embedding_schnet.py
echo "Phase C exit code: $?"

echo ""
echo "=== Phase D+E: Compare all embeddings + visualization ==="
python scripts/studies/embedding_compare_all.py

echo ""
echo "=== ALL PHASES COMPLETE ==="
echo "Date: $(date)"
echo "Figures: $(ls results/tier3_gp/figures/pretrained_comparison/ 2>/dev/null | wc -l)"
