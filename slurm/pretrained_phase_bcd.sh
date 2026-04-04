#!/bin/bash
#SBATCH --job-name=pretrained_emb
#SBATCH --output=slurm/logs/pretrained_emb_%j.out
#SBATCH --error=slurm/logs/pretrained_emb_%j.out
#SBATCH --partition=a100_chemistry
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_281_chemistry

echo "=== PRE-TRAINED EMBEDDING EXTRACTION + COMPARISON ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff

cd /scratch/yd2915/BayesDiff

# Install unimol_tools if not present
pip show unimol-tools > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing unimol_tools..."
    pip install unimol_tools --no-deps 2>&1 | tail -5
    # Install missing deps if needed
    pip install huggingface_hub 2>/dev/null
fi

echo ""
echo "=== Phase B: Uni-Mol 512-dim embedding extraction ==="
python scripts/26b_extract_unimol.py || echo "Uni-Mol extraction failed (non-fatal)"

echo ""
echo "=== Phase C: SchNet QM9-pretrained embedding extraction ==="
python scripts/26c_extract_schnet.py || echo "SchNet extraction failed (non-fatal)"

echo ""
echo "=== Phase D+E: Compare all embeddings + visualization ==="
python scripts/26d_compare_embeddings.py

echo ""
echo "=== ALL PHASES COMPLETE ==="
echo "Date: $(date)"
echo "Figures: $(ls results/tier3_gp/figures/pretrained_comparison/ 2>/dev/null | wc -l)"
