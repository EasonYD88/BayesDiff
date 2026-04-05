# BayesDiff HPC Environment Status

> Auto-generated HPC environment verification report.
> Cluster: NYU Torch HPC | Updated: 2026-03-04

---

## Cluster Info

| Item | Value |
|------|-------|
| Cluster | NYU Torch HPC |
| Login Node | torch-login-5 |
| GPU Node | ga018.hpc.nyu.edu (A100-SXM4-80GB) |
| User | yd2915 |
| Scheduler | SLURM |
| Accounts | torch_pr_281_general, torch_pr_281_chemistry |
| GPU Partition | a100_chemistry (tested), l40s_public (QOS shared) |
| Driver | NVIDIA 580.82.07, CUDA 13.0 |

## Environment

| Item | Value |
|------|-------|
| Conda | /scratch/yd2915/miniconda3 (25.11.1) |
| Env Path | /scratch/yd2915/conda_envs/bayesdiff |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| PyG | 2.7.0 |
| GPyTorch | 1.15.2 |
| RDKit | 2025.9.6 |
| scikit-learn | 1.7.2 |
| gdown | 5.2.1 |

## Activation

```bash
eval "$(/scratch/yd2915/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/yd2915/conda_envs/bayesdiff
cd /scratch/yd2915/BayesDiff
```

## Execution Plan Progress

| Step | Description | Status |
|------|-------------|--------|
| S0 | Environment setup | ✅ Complete |
| S1.1 | Clone BayesDiff + TargetDiff | ✅ Complete |
| S1.2 | Pretrained weights (32M + 30M + 1K) | ✅ Complete |
| S1.3 | Test data (93 targets + affinity_info) | ✅ Complete |
| S1.4 | torch_scatter shim | ✅ Complete |
| S1.5 | _check_deps.py (7/7 passed) | ✅ Complete |
| S1.6 | Structure verification | ✅ Complete |
| S2.1 | test_pockets.txt (93 lines) | ✅ Complete |
| S0.3 | GPU verification (A100-80GB) | ✅ Complete (job 3253941) |
| S2.2 | Smoke test (5 pockets × 4 samples) | ✅ Complete (job 3254006, 105s) |
| S3 | Batch sampling — 1st attempt | ❌ Failed (job 3254044, bugs fixed) |
| S3 | Batch sampling — 2nd attempt (serial) | ✅ Complete (job 3284523, 19h02m, 93/93) |
| S3 | Batch sampling — parallel (4-shard) | ✅ Complete (93 pockets merged) |
| S4 | Embedding re-extraction | ⏭ Skipped (S3 includes embeddings) |
| S5 | GP training (GPU) | ✅ Complete (job 3386803, 14.1s on A100) |
| S6 | Evaluation + calibration | ✅ Complete (job 3386892) |
| S7 | Ablation experiments | ✅ Complete (job 3386892, 7 variants) |
| S8 | Results collection + push | ✅ Complete (commit 146bf70) |

## Structure Verification (S1.6)

```
bayesdiff/    → 9 .py files  ✅
scripts/      → 9 .py files  ✅
external/targetdiff/models/   → present  ✅
pretrained_diffusion.pt       → 32M      ✅
egnn_pdbbind_v2016.pt         → 30M      ✅
pk_reg_para.pkl               → 968B     ✅
test_set/                     → 93 dirs  ✅
affinity_info.pkl             → 19M      ✅
torch_scatter shim            → OK       ✅
data/splits/test_pockets.txt  → 93 lines ✅
data/splits/debug_pockets.txt → present  ✅
results/                      → prior Mac debug run present ✅
```

## Dependency Check (_check_deps.py)

```
torch: 2.5.1+cu121      ✅
PyG: OK                  ✅
rdkit: OK                ✅
matplotlib: OK           ✅
easydict: OK             ✅
lmdb: OK                 ✅
yaml: OK                 ✅
```

## GPU Verification (S0.3) — Job 3253941

```
Node:       ga018.hpc.nyu.edu
GPU:        NVIDIA A100-SXM4-80GB (80 GB)
Driver:     580.82.07 / CUDA 13.0
PyTorch:    2.5.1+cu121, cuda=True
Matmul:     OK (2000×2000)
Wall time:  1m04s
```

## Smoke Test Results (S2.2) — Job 3254006

```
Node:       ga011.hpc.nyu.edu (A100-SXM4-80GB)
Pipeline:   5 pockets × 4 samples × 20 steps, device=cuda
Duration:   105s (1.7 min), wall 2m15s
All 6 steps: Data Prep ✅ | Sampling ✅ | Embeddings ✅ | GP Train ✅ | Eval ✅ | Viz ✅
Metrics:    ECE=0.243, AUROC=1.000, Spearman=1.000, RMSE=1.801
Figures:    6 PNG saved to results/figures/
```

## Resolved Issues

- `openbabel` installed via `conda install -c conda-forge openbabel` ✅
- `torch-cluster` installed via `pip install torch-cluster -f pyg whl` ✅
- `sample_diffusion_ligand` return value mismatch (7 vs 8) — fixed in `run_full_pipeline.py` ✅
- **S3 job 3254044 failure** — fixed in `bayesdiff/sampler.py` (see below) ✅

## S3 Batch Sampling – 1st Attempt Post-Mortem (Job 3254044)

**Symptoms:** Job COMPLETED (exit 0), ran 10h28m on ga038 (A100-80GB), used ~10.5 GB RAM.
93 pocket directories were created but contained **0 embedding files, 0 SDF files**.

**Root Causes:**

1. **Return value unpacking (93/93 pockets affected)**
   `bayesdiff/sampler.py:sample_for_pocket()` expected 8 return values from
   TargetDiff's `sample_diffusion_ligand()`, which only returns 7 (no `mol_embeddings`).
   Error: `not enough values to unpack (expected 8, got 7)`.
   The same bug was previously fixed in `run_full_pipeline.py` but not in `sampler.py`.

2. **CUDA OOM (~20/93 pockets affected)**
   Default `batch_size=num_samples=64` caused OOM on larger pockets (e.g., BAPA_SPHXN
   tried to allocate 27.5 GiB with only 11.5 GiB free).

**Fixes Applied (commit `ecd3d64`):**

| File | Fix |
|------|-----|
| `bayesdiff/sampler.py` | `sample_for_pocket()` now handles both 7- and 8-value returns |
| `bayesdiff/sampler.py` | `sample_and_embed()` starts at `batch_size=16`, auto-halves on OOM |
| `slurm/sample_job.sh` | Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## Known Issues

- `QOSGrpGRES` on `l40s_public` — shared GPU quota with other users
  - Workaround: use `--partition=a100_chemistry --account=torch_pr_281_chemistry`
- Fallback embeddings used (not real SE(3) from TargetDiff encoder) — needs `sample_diffusion` patch for production

## Notes

- `cuda=False` on login nodes is expected; GPU nodes show `cuda=True` ✅
- Prior results in `results/` are from Mac CPU debug run (3 pockets, 2 samples)
- HPC production runs (S3+) will overwrite/extend these results
- For sbatch, always specify `--account=torch_pr_281_chemistry --partition=a100_chemistry`

---

## 2026-03-05 Workflow Update (Parallel Sampling Added)

### New parallel artifacts

- `slurm/sample_array_job.sh` (multi-GPU Slurm array sampling)
- `scripts/scaling/s02_merge_shards.py` (merge shard outputs)
- `slurm/merge_sample_shards_job.sh` (CPU merge job)

### Data safety

Parallel runs now default to isolated run folders:

- `results/generated_molecules_parallel/<run_tag>/shards/`
- `results/generated_molecules_parallel/<run_tag>/all_embeddings.npz`

This avoids overwriting existing outputs under `results/generated_molecules/`.

- `scripts/scaling/s01_sample_shard.py` (shard wrapper; calls original `scripts/pipeline/s02_sample_molecules.py` unchanged)

---

## 2026-03-05 Structure & Progress Update

### Structure delta

- Added: `slurm/sample_array_job.sh`
- Added: `scripts/scaling/s01_sample_shard.py`
- Added: `scripts/scaling/s02_merge_shards.py`
- Added: `slurm/merge_sample_shards_job.sh`
- Unchanged: `scripts/pipeline/s02_sample_molecules.py`, `slurm/sample_job.sh`

### Progress delta

| Item | Status |
|------|--------|
| Parallel workflow implementation | ✅ Complete |
| Parallel workflow docs | ✅ Complete |
| Parallel S3 execution | ✅ Complete (4-shard, 93 pockets merged) |
| Merged parallel embeddings validation | ✅ Complete |
| S5 GP training (GPU, 14.1s) | ✅ Complete (job 3386803) |
| S6 Evaluation | ✅ Complete (job 3386892) |
| S7 Ablation | ✅ Complete (job 3386892, 7 variants) |
| S8 Results push | ✅ Complete (commit 146bf70) |

Note: parallel outputs are isolated under `results/generated_molecules_parallel/<run_tag>/` to avoid overwriting existing results.

### SLURM Scripts

| Script | Purpose | Resources |
|--------|---------|-----------|
| `slurm/sample_job.sh` | Serial batch sampling | 1×A100, 48h |
| `slurm/sample_array_job.sh` | Parallel array sampling | N×A100, 24h |
| `slurm/train_gp.sh` | GP training on GPU | 1×A100, 1h |
| `slurm/eval_ablation.sh` | S6 eval + S7 ablation | 1×A100, 1h |
| `slurm/merge_sample_shards_job.sh` | Merge parallel shards | CPU, 30min |
