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
| S2.2 | Smoke test (5 pockets × 4 samples) | ⚠️ Blocked by missing openbabel |
| S3 | Batch sampling (93 pockets × 64 samples) | ⬜ Not started |
| S4 | Embedding re-extraction | ⬜ Not started |
| S5 | GP training | ⬜ Not started |
| S6 | Evaluation + calibration | ⬜ Not started |
| S7 | Ablation experiments | ⬜ Not started |
| S8 | Results collection | ⬜ Not started |

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

## Known Issues

- `openbabel` not installed → TargetDiff `reconstruct.py` import fails → S2.2 smoke test blocked
  - Fix: `conda install -c conda-forge openbabel` or `pip install openbabel-wheel`
- `QOSGrpGRES` on `l40s_public` — shared GPU quota with other users (yx2892)
  - Workaround: use `--partition=a100_chemistry --account=torch_pr_281_chemistry`

## Notes

- `cuda=False` on login nodes is expected; GPU nodes show `cuda=True` ✅
- Prior results in `results/` are from Mac CPU debug run (3 pockets, 2 samples)
- HPC production runs (S3+) will overwrite/extend these results
- For sbatch, always specify `--account=torch_pr_281_chemistry --partition=a100_chemistry`
