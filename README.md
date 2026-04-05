# BayesDiff

Dual uncertainty-aware confidence scoring for 3D molecular generation.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Data + Environment | ✅ Complete |
| Phase 1 | Core Modules (8 modules, 41 validation checks) | ✅ Complete |
| Phase 1.5 | HPC Environment Setup (NYU Torch) | ✅ Complete |
| Phase 2 | HPC Batch Sampling (93 pockets × M=64) | ✅ Complete (job 3284523, 19h02m) |
| Phase 2.5 | Parallel Sampling (4-shard array) | ✅ Complete |
| Phase 3 | GP Training + Evaluation + Ablation | ✅ Complete (jobs 3386803, 3386892) |
| Phase 3.5 | GP Optimization (Tier 3 data, encoder embeddings, aggregation) | ✅ Complete |
| Phase 3.6 | PDBbind Large-Scale Sampling (93 pockets × 50 mol, 31-GPU array) | ✅ Complete |
| Phase 4 | Writing (Science-style manuscript) | ✅ Draft complete |

### HPC Environment (NYU Torch)

| Component | Version / Path |
|-----------|---------------|
| Cluster | NYU Torch HPC (SLURM) |
| Partition | a100_chemistry |
| GPU | NVIDIA A100-SXM4-80GB |
| Conda env | `/scratch/yd2915/conda_envs/bayesdiff` |
| PyTorch | 2.5.1+cu121 |
| PyG | 2.7.0 |
| GPyTorch | 1.15.2 |

### HPC Results Summary

| Stage | Result |
|-------|--------|
| S3 Sampling | 93 pockets × 64 samples, 100 steps, 19h on A100 |
| S5 GP Training | 48 pockets, d=128, J=48, **14.1s on GPU** |
| S6 Evaluation | ECE=0.034, RMSE=1.87, N=48 |
| S7 Ablation | 7 variants; A2 (no oracle var) → NLL explodes |

> See [doc/hpc/HPC_ENV_STATUS.md](doc/hpc/HPC_ENV_STATUS.md) for full verification details.

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
python scripts/utils/run_full_pipeline.py --mode debug --n_pockets 3 --num_samples 2 --num_steps 20

# 4. Or run individual steps:
python scripts/pipeline/s01_prepare_data.py --pdbbind_dir data/pdbbind --output_dir data/splits
python scripts/pipeline/s02_sample_molecules.py --pocket_list data/splits/debug_pockets.txt --pdbbind_dir data/pdbbind --num_samples 4 --device cpu
python scripts/pipeline/s03_extract_embeddings.py --input_dir results/generated_molecules --output data/embeddings/debug.npz
python scripts/pipeline/s04_train_gp.py --embeddings data/embeddings/debug.npz --labels data/splits/labels.csv --output results/gp_model
python scripts/pipeline/s05_evaluate.py --embeddings data/embeddings/debug.npz --gp_model results/gp_model/gp_model.pt --output results/evaluation
python scripts/pipeline/s06_ablation.py --embeddings data/embeddings/debug.npz --gp_model results/gp_model/gp_model.pt --output results/ablation
```

## Project Structure

```
BayesDiff/
├── bayesdiff/                  # Core library (8 modules)
│   ├── data.py                 # §Data: loading, splits, label transforms
│   ├── sampler.py              # §4.1: TargetDiff sampling + SE(3) embedding
│   ├── gen_uncertainty.py      # §4.1: Σ_gen, Ledoit-Wolf, GMM detection
│   ├── gp_oracle.py            # §4.2: SVGP training/inference
│   ├── fusion.py               # §4.3: Delta method uncertainty fusion
│   ├── calibration.py          # §4.4: Isotonic/Platt/Temperature + ECE/ACE
│   ├── ood.py                  # §4.4: Mahalanobis OOD detection
│   └── evaluate.py             # §5: Full metrics + bootstrap CI
├── scripts/
│   ├── pipeline/               # Core pipeline (§4 Method)
│   │   ├── s01_prepare_data.py
│   │   ├── s02_sample_molecules.py
│   │   ├── s03_extract_embeddings.py
│   │   ├── s04_train_gp.py
│   │   ├── s05_evaluate.py
│   │   ├── s06_ablation.py
│   │   └── s07_generate_figures.py
│   ├── scaling/                # Large-scale experiments (§5)
│   ├── studies/                # Ablation & auxiliary studies (§5–6, SI)
│   └── utils/                  # Tools & debugging
├── slurm/                      # HPC job scripts (NYU Torch)
│   ├── pipeline/               # Core pipeline jobs
│   ├── scaling/                # Large-scale jobs
│   ├── studies/                # Study-specific jobs
│   ├── utils/                  # Diagnostics
│   └── logs/                   # SLURM job logs
├── doc/
│   ├── Stage_1/                # Design & implementation docs
│   ├── Stage_2/                # Problem analysis & future directions
│   └── hpc/                    # HPC-specific docs
├── write_up/                   # Manuscript
│   ├── main.tex                # Science-style manuscript (LaTeX)
│   └── code_map.md             # Paper ↔ code mapping table
├── tests/
│   ├── test_pipeline.py        # Phase 0 sanity check
│   └── test_phase1_validation.py # Phase 1 module validation (41 checks)
├── data/                       # Data directory (not tracked)
├── results/                    # Pipeline outputs (JSON/PNG tracked)
├── external/                   # TargetDiff submodule
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
