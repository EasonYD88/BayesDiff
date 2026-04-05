#!/bin/bash
#SBATCH --job-name=pretrained_v3
#SBATCH --output=slurm/logs/pretrained_v3_%j.out
#SBATCH --error=slurm/logs/pretrained_v3_%j.out
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_281_chemistry

echo "=== PRE-TRAINED EMBEDDING v3: Uni-Mol + SchNet + Comparison ==="
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

# Install schnetpack if needed (required for loading QM9 pre-trained weights)
python -c "import schnetpack" 2>/dev/null || pip install schnetpack --quiet
python -c "import schnetpack; print('SchNetPack OK')"

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
