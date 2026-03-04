# BayesDiff

Dual uncertainty-aware confidence scoring for 3D molecular generation.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Data + Environment | ✅ Complete |
| Phase 1 | Core Modules (8 modules, 41 validation checks) | ✅ Complete |
| Phase 1.5 | HPC Environment Setup (NYU Torch) | ✅ Complete |
| Phase 2 | HPC Batch Sampling (93 pockets × M=64) | 🟡 Running (job 3254044) |
| Phase 3 | Full Experiments + Ablation | ⬜ Not started |
| Phase 4 | Writing | ⬜ Not started |

### HPC Environment (NYU Torch)

| Component | Version / Path |
|-----------|---------------|
| Cluster | NYU Torch HPC (SLURM) |
| Conda env | `/scratch/yd2915/conda_envs/bayesdiff` |
| PyTorch | 2.5.1+cu121 |
| PyG | 2.7.0 |
| GPyTorch | 1.15.2 |

> See [doc/HPC_ENV_STATUS.md](doc/HPC_ENV_STATUS.md) for full verification details.

## Quick Start

```bash
# 1. Create environment
conda create -n bayesdiff python=3.10 -y
conda activate bayesdiff
pip install -r requirements.txt

# 2. Clone TargetDiff & download pretrained weights
git clone https://github.com/guanjq/targetdiff.git external/targetdiff
# Download pretrained_diffusion.pt into external/targetdiff/pretrained_models/

# 3. Debug pipeline (end-to-end, Mac CPU, ~7 min)
python scripts/run_full_pipeline.py --mode debug --n_pockets 3 --num_samples 2 --num_steps 20

# 4. Or run individual steps:
python scripts/01_prepare_data.py --pdbbind_dir data/pdbbind --output_dir data/splits
python scripts/02_sample_molecules.py --pocket_list data/splits/debug_pockets.txt --pdbbind_dir data/pdbbind --num_samples 4 --device cpu
python scripts/03_extract_embeddings.py --input_dir results/generated_molecules --output data/embeddings/debug.npz
python scripts/04_train_gp.py --embeddings data/embeddings/debug.npz --labels data/splits/labels.csv --output results/gp_model
python scripts/05_evaluate.py --embeddings data/embeddings/debug.npz --gp_model results/gp_model/gp_model.pt --output results/evaluation
python scripts/06_ablation.py --embeddings data/embeddings/debug.npz --gp_model results/gp_model/gp_model.pt --output results/ablation
```

## Project Structure

```
BayesDiff/
├── bayesdiff/               # Core library (8 modules)
│   ├── data.py              # Data loading, splits, label transforms
│   ├── sampler.py           # TargetDiff sampling + SE(3) embedding extraction
│   ├── gen_uncertainty.py   # Σ_gen, Ledoit-Wolf, GMM multimodal detection
│   ├── gp_oracle.py         # SVGP training/inference (PCA, k-means inducing)
│   ├── fusion.py            # Delta method + MC fallback uncertainty fusion
│   ├── calibration.py       # Isotonic/Platt/Temperature + ECE/ACE
│   ├── ood.py               # Mahalanobis OOD detection + confidence modifier
│   └── evaluate.py          # Full metrics + bootstrap CI + multi-threshold
├── scripts/                 # Numbered pipeline scripts + monolithic runner
│   ├── 01_prepare_data.py   # Parse PDBbind INDEX → splits + labels
│   ├── 02_sample_molecules.py  # TargetDiff batch sampling
│   ├── 03_extract_embeddings.py  # SE(3) embedding extraction
│   ├── 04_train_gp.py       # Train SVGP oracle
│   ├── 05_evaluate.py       # Full evaluation pipeline
│   ├── 06_ablation.py       # Ablation experiments (A1-A5, A7)
│   └── run_full_pipeline.py # End-to-end debug pipeline
├── external/                # TargetDiff clone (not tracked)
├── data/                    # Data directory (not tracked)
├── results/                 # Pipeline outputs (not tracked)
├── doc/                     # Plans & progress log
├── notebooks/               # Validation & debugging
├── slurm/                   # HPC job scripts
└── requirements.txt
```

## Architecture

```
Pocket PDB → TargetDiff (M samples) → SE(3) Encoder → z ∈ ℝ^128
                                          ↓
                                    Σ̂_gen (Ledoit-Wolf)
                                    GMM mode detection
                                          ↓
                              SVGP Oracle → μ_oracle, σ²_oracle, J_μ
                                          ↓
                              Delta Method Fusion → σ²_total
                                          ↓
                              P_success = Φ((y_target - μ) / σ_total)
                                          ↓
                              Isotonic Calibration + OOD Gate
                                          ↓
                              Calibrated P_success per molecule
```
