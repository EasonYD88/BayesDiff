# Scripts

## Directory Structure

### `pipeline/` — Core Pipeline (论文 §4 Method)

Execute sequentially: `s01` → `s07`.

| Script | Purpose |
|--------|---------|
| `s01_prepare_data.py` | Parse PDBbind INDEX → protein-family splits + labels |
| `s02_sample_molecules.py` | TargetDiff batch sampling + SE(3) embedding extraction |
| `s03_extract_embeddings.py` | Re-extract SE(3) embeddings from existing SDF files |
| `s04_train_gp.py` | Train SVGP oracle (ARD Matérn-5/2, k-means inducing) |
| `s05_evaluate.py` | Full evaluation: fusion → calibration → OOD → metrics |
| `s06_ablation.py` | Ablation experiments (A1–A7) |
| `s07_generate_figures.py` | Generate 6 publication figures |

### `scaling/` — Large-Scale Experiments (论文 §5 Experiments)

For parallel HPC runs and large dataset experiments.

| Script | Purpose |
|--------|---------|
| `s01_sample_shard.py` | Shard wrapper for parallel GPU sampling |
| `s02_merge_shards.py` | Merge parallel shard outputs |
| `s03_prepare_tier3.py` | Extract CrossDocked LMDB pockets for tier3 |
| `s04_sample_tier3_shard.py` | Tier3 GPU-array sampling |
| `s05_train_gp_tier3.py` | Train SVGP on full tier3 dataset (N≈932) |
| `s06_merge_50mol_shards.py` | Merge 50mol sampling shards |
| `s07_extract_50mol_embeddings.py` | Extract & merge 50mol embeddings |
| `s08_merge_and_train_eval.py` | Merge multi-step embeddings + retrain + evaluate |

### `studies/` — Auxiliary Studies (论文 §5–6, SI)

Independent analyses; no required execution order.

**Embedding Studies:**
- `embedding_comparison.py` — FCFP4 vs encoder (2D vs 3D)
- `embedding_encoder_only.py` — Pure encoder embeddings
- `embedding_multilayer.py` — Multi-layer encoder extraction
- `embedding_multilayer_full.py` — Full multi-layer extraction
- `embedding_unimol.py` — UniMol embeddings
- `embedding_schnet.py` — SchNet embeddings
- `embedding_compare_all.py` — Cross-architecture comparison

**GP Studies:**
- `gp_training_analysis.py` — Hyperparameter sensitivity
- `gp_encoder.py` — GP on encoder-only embeddings
- `gp_aggregation.py` — Aggregation strategy comparison
- `gp_multilayer.py` — GP on multi-layer embeddings
- `gp_50mol_study.py` — 50mol sampling density analysis
- `bo_gp_hyperparams.py` — Bayesian optimization for GP

**Evaluation Studies:**
- `robust_evaluation.py` — Cross-validated robust evaluation
- `regularization_study.py` — L2/dropout regularization
- `subsample_ablation.py` — Subsampling ablation
- `tier3_training_curves.py` — Tier3 training dynamics
- `train_val_test_analysis.py` — Split balance analysis
- `reextract_embeddings.py` — Re-extraction with different params

### `utils/` — Tools & Debugging

| Script | Purpose |
|--------|---------|
| `run_full_pipeline.py` | End-to-end pipeline (debug/pdbbind/full modes) |
| `check_deps.py` | Verify all dependency imports |
| `torch_scatter_shim.py` | Compatibility shim for older torch_scatter API |
