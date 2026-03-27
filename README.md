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
├── bayesdiff/                  # Core library (8 modules)
│   ├── data.py                 # Data loading, splits, label transforms
│   ├── sampler.py              # TargetDiff sampling + SE(3) embedding extraction
│   ├── gen_uncertainty.py      # Σ_gen, Ledoit-Wolf, GMM multimodal detection
│   ├── gp_oracle.py            # SVGP training/inference (PCA, k-means inducing)
│   ├── fusion.py               # Delta method + MC fallback uncertainty fusion
│   ├── calibration.py          # Isotonic/Platt/Temperature + ECE/ACE
│   ├── ood.py                  # Mahalanobis OOD detection + confidence modifier
│   └── evaluate.py             # Full metrics + bootstrap CI + multi-threshold
├── scripts/                    # Pipeline scripts
│   ├── 01_prepare_data.py      # Parse PDBbind INDEX → splits + labels
│   ├── 02_sample_molecules.py  # TargetDiff batch sampling (PDBbind + CrossDocked)
│   ├── 03_extract_embeddings.py  # SE(3) embedding extraction
│   ├── 04_train_gp.py          # Train SVGP oracle (--device auto for GPU)
│   ├── 05_evaluate.py          # Full evaluation pipeline
│   ├── 06_ablation.py          # Ablation experiments (A1-A5, A7)
│   ├── 07_merge_sampling_shards.py  # Merge parallel shard outputs
│   ├── 08_sample_molecules_shard.py # Shard wrapper for array jobs
│   ├── 09_generate_figures.py  # Generate publication figures
│   ├── 10_merge_and_train_eval.py   # Merge 1000-step + retrain + visualize
│   ├── 11_gp_training_analysis.py   # GP hyperparameter analysis
│   ├── 11_reextract_embeddings.py   # Re-extract embeddings pipeline
│   ├── 12_robust_evaluation.py      # Robust cross-validated evaluation
│   ├── 13_embedding_comparison.py   # FCFP4 vs encoder embedding comparison
│   ├── 14_bo_gp_hyperparams.py      # Bayesian optimization for GP hyperparams
│   ├── 15_prepare_tier3.py          # Tier 3 LMDB pocket extraction
│   ├── 16_sample_tier3_shard.py     # Tier 3 GPU array sampling
│   ├── 17_train_gp_tier3.py         # Tier 3 GP training
│   ├── 18_train_val_test_analysis.py # Train/val/test split analysis
│   ├── 19_extract_encoder_embeddings.py # TargetDiff encoder embeddings
│   ├── 20_train_gp_encoder.py       # GP with encoder embeddings
│   ├── 21_train_gp_aggregation.py   # Aggregation strategy comparison
│   ├── run_full_pipeline.py    # End-to-end pipeline (debug/pdbbind/full modes)
│   ├── torch_scatter_shim.py   # Compatibility shim for older torch_scatter API
│   └── _check_deps.py          # Verify dependency imports
├── slurm/                      # HPC job scripts (NYU Torch)
│   ├── sample_job.sh           # Single-node sampling
│   ├── sample_array_job.sh     # 4-shard parallel sampling
│   ├── sample_maxgpu.sh        # 31-GPU array (93 pockets × 50 mol)
│   ├── sample_tier3_array.sh   # Tier 3 sampling array
│   ├── embedding_1000step_array.sh  # 1000-step embedding extraction
│   ├── train_gp.sh             # GP training on GPU
│   ├── train_gp_encoder.sh     # GP with encoder embeddings
│   ├── train_gp_aggregation.sh # Aggregation strategy comparison
│   ├── eval_ablation.sh        # Evaluation + ablation
│   ├── gp_train_eval_viz.sh    # Combined GP + eval + viz
│   ├── full_pipeline_job.sh    # Full pipeline on single GPU
│   ├── merge_*.sh              # Merge shard outputs
│   └── logs/                   # SLURM job logs
├── doc/                        # Documentation
│   ├── overall_plan.md         # High-level project plan
│   ├── plan_opendata.md        # Open-data execution plan
│   ├── progress_log.md         # Chronological progress log
│   ├── math.md                 # Mathematical formulation (tutorial)
│   ├── math_explain.md         # Mathematical formulation (reference)
│   ├── gp_analysis_and_optimization.md # GP optimization results (§16)
│   ├── check_01.md             # Code-math alignment audit
│   └── hpc/                    # HPC-specific docs
│       ├── bayesdiff_nyu_torch_hpc_agent_guide.md
│       ├── nyu_torch_coding_agent_guide.md
│       ├── hpc_execution_plan.md
│       └── HPC_ENV_STATUS.md
├── write_up/                   # Manuscript
│   └── main.tex                # Science-style manuscript (LaTeX)
├── tests/                      # Validation scripts
│   ├── debug_pipeline.py       # Phase 0 sanity check
│   └── validate_phase1.py      # Phase 1 module validation (41 checks)
├── data/                       # Data directory (not tracked)
│   ├── pdbbind/                # PDBbind raw data (download manually)
│   └── splits/                 # Pocket lists for pipeline
├── results/                    # Pipeline outputs (JSON/PNG tracked, binaries ignored)
│   ├── figures/                # 6 publication figures
│   ├── evaluation/             # Evaluation metrics (JSON)
│   ├── ablation/               # Ablation study results (JSON)
│   ├── gp_model/               # GP model metadata
│   └── generated_molecules/    # Sampling summary
├── external/                   # TargetDiff submodule (not tracked)
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
