# BayesDiff

Dual uncertainty-aware confidence scoring for 3D molecular generation.

## Quick Start

```bash
# 1. Create environment
conda create -n bayesdiff python=3.10 -y
conda activate bayesdiff
pip install -r requirements.txt

# 2. Prepare data (after downloading PDBbind v2020)
python scripts/01_prepare_data.py --pdbbind_dir data/pdbbind --output_dir data/splits

# 3. Clone TargetDiff & download pretrained weights
git clone https://github.com/guanjq/targetdiff.git external/targetdiff
# Download pretrained_diffusion.pt into external/targetdiff/pretrained_models/

# 4. Debug pipeline (Mac, M=4, 3 pockets)
python scripts/02_sample_molecules.py \
    --pocket_dir data/pdbbind/refined-set \
    --pdb_list data/splits/debug_pockets.txt \
    --num_samples 4 --device cpu \
    --output_dir data/generated/debug

# 5. Extract embeddings
python scripts/03_extract_embeddings.py \
    --generated_dir data/generated/debug \
    --output data/embeddings/debug.npz
```

## Project Structure

```
BayesDiff/
├── bayesdiff/           # Core library
│   ├── data.py          # Data loading, splits, label transforms
│   ├── sampler.py       # TargetDiff sampling wrapper
│   ├── gen_uncertainty.py  # Σ_gen, GMM multimodal detection
│   ├── gp_oracle.py     # SVGP training/inference
│   ├── fusion.py        # Delta method uncertainty fusion
│   ├── calibration.py   # Isotonic regression + ECE
│   ├── ood.py           # Mahalanobis OOD detection
│   └── evaluate.py      # All evaluation metrics
├── scripts/             # Runnable scripts (numbered)
├── data/                # Data directory (not tracked)
├── external/            # TargetDiff clone (not tracked)
├── doc/                 # Documentation & plans
└── notebooks/           # Jupyter notebooks
```
