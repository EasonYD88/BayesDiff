# Sub-Plan 6: Physics-Aware Features

> **Priority**: P2 — Medium  
> **Dependency**: Sub-Plan 0 (PDBbind v2020 数据集)  
> **Training Data**: PDBbind v2020 refined set (~5,316 complexes), 见 [00a_supervised_pretraining.md](00a_supervised_pretraining.md)  
> **Estimated Effort**: 1–2 weeks implementation + 1 week testing  
> **Paper Section**: §3.W Physics-Informed Feature Augmentation

---

## 1. Motivation

Pure data-driven embeddings from the SE(3)-equivariant encoder encode geometric and chemical patterns implicitly, but may not directly capture well-understood physical mechanisms that govern binding:

| Physical Mechanism | Contribution to ΔG | Captured by Encoder? |
|-------------------|---------------------|---------------------|
| Hydrogen bonds | −0.5 to −3 kcal/mol each | Implicitly (if at all) |
| Hydrophobic contact | −0.5 to −1.5 kcal/mol per contact | Partially |
| Electrostatic complementarity | Variable | Weakly |
| Desolvation penalty | +1 to +5 kcal/mol | Unlikely |
| Ligand strain | +0.5 to +3 kcal/mol | Unlikely |
| Steric clash | Large penalty | Partially |

**Goal**: Supplement the learned embedding $z$ with a curated physics-aware feature vector $p \in \mathbb{R}^{d_p}$ containing interpretable descriptors that encode binding-relevant physical properties.

**Key principle**: These features serve as "intermediate mechanism variables" — they don't replace the learned representation, but provide domain knowledge shortcuts that the encoder may not have discovered.

---

## 2. Feature Design

### 2.1 Feature Categories

#### Category A: Geometric Contact Features (from generated 3D structures)

| Feature | Dimension | Description | Computation |
|---------|-----------|-------------|-------------|
| `n_contacts_4.5A` | 1 | Number of protein-ligand atom pairs ≤ 4.5 Å | Distance matrix threshold |
| `n_contacts_6.0A` | 1 | Number of contacts ≤ 6.0 Å (extended shell) | Distance matrix threshold |
| `contact_surface_area` | 1 | Approximate buried surface area | SASA difference |
| `pocket_coverage` | 1 | Fraction of pocket surface covered by ligand | Contact area / pocket area |
| `shape_complementarity` | 1 | Lawrence-Colman shape complementarity score | SC algorithm |

#### Category B: Interaction-Specific Features

| Feature | Dimension | Description | Computation |
|---------|-----------|-------------|-------------|
| `n_hbonds` | 1 | Number of hydrogen bonds | Distance + angle criteria |
| `n_hydrophobic` | 1 | Number of hydrophobic contacts | Atom type + distance |
| `n_pi_stacking` | 1 | Number of π-π stacking interactions | Ring geometry |
| `n_salt_bridges` | 1 | Number of salt bridges | Charge + distance |
| `n_halogen_bonds` | 1 | Number of halogen bonds | Atom type + geometry |

#### Category C: Energetic Features (from scoring functions)

| Feature | Dimension | Description | Computation |
|---------|-----------|-------------|-------------|
| `vina_score` | 1 | AutoDock Vina docking score | Vina re-scoring |
| `vina_affinity` | 1 | Vina predicted ΔG | Vina |
| `rf_score` | 1 | RF-Score (ML scoring function) | RF-Score v3 |
| `plec_score` | 1 | Protein-Ligand Extended Connectivity score | PLEC fingerprint |

#### Category D: Ligand Property Features

| Feature | Dimension | Description | Computation |
|---------|-----------|-------------|-------------|
| `mw` | 1 | Molecular weight | RDKit |
| `logp` | 1 | Calculated logP (lipophilicity) | RDKit Crippen |
| `tpsa` | 1 | Topological polar surface area | RDKit |
| `n_rotatable` | 1 | Number of rotatable bonds | RDKit |
| `n_hbd` | 1 | Number of H-bond donors | RDKit |
| `n_hba` | 1 | Number of H-bond acceptors | RDKit |
| `qed` | 1 | Quantitative Estimate of Drug-likeness | RDKit |

#### Category E: Pocket Property Features

| Feature | Dimension | Description | Computation |
|---------|-----------|-------------|-------------|
| `pocket_volume` | 1 | Binding pocket volume | fpocket / SiteMap |
| `pocket_druggability` | 1 | Druggability score | fpocket |
| `pocket_hydrophobicity` | 1 | Average hydrophobicity of pocket residues | Kyte-Doolittle |
| `pocket_charge` | 1 | Net pocket charge at pH 7.4 | Residue pKa |

### 2.2 Recommended Feature Set

**Minimal set** (10 features, easy to compute):

$$
p_{\text{minimal}} = [\text{n\_contacts}, \text{n\_hbonds}, \text{n\_hydrophobic}, \text{vina\_score}, \text{MW}, \text{logP}, \text{TPSA}, \text{n\_rot}, \text{n\_HBD}, \text{n\_HBA}]
$$

**Extended set** (20 features):

$$
p_{\text{extended}} = [p_{\text{minimal}}; \text{shape\_comp}, \text{pocket\_vol}, \text{pocket\_drug}, \text{n\_pi}, \text{n\_salt}, \text{contact\_SA}, \text{pocket\_cov}, \text{QED}, \text{pocket\_hydro}, \text{pocket\_charge}]
$$

---

## 3. Fusion Strategies

### 3.1 Early Fusion (Concatenation)

$$
u = [z; p] \in \mathbb{R}^{d + d_p}
$$

Feed directly to the oracle. Simplest; lets the model learn feature interactions.

### 3.2 Late Fusion (Separate Processing)

$$
\hat{y} = f_{\text{data}}(z) + f_{\text{phys}}(p)
$$

where $f_{\text{data}}$ is the GP oracle and $f_{\text{phys}}$ is a small MLP. Preserves existing model structure.

### 3.3 Gated Fusion (Recommended)

$$
g = \sigma(W_g [z; p] + b_g) \in \mathbb{R}^{d_{\text{out}}}
$$

$$
u = g \odot W_z z + (1 - g) \odot W_p p
$$

The gate learns when to rely on data-driven vs. physics-based features on a per-dimension basis.

### 3.4 Feature-as-Prior

Use physics features to inform the GP mean function:

$$
m(z) = \beta^\top p + \beta_0
$$

$$
f(z) \sim \mathcal{GP}(\beta^\top p + \beta_0, k(z, z'))
$$

GP deviation from the physics-based prior captures what physics alone cannot explain.

---

## 4. Implementation Plan

### 4.1 New Module: `bayesdiff/physics_features.py`

```python
class PhysicsFeatureExtractor:
    """Extract physics-aware features from protein-ligand complexes."""
    
    def __init__(self, feature_set: str = 'minimal', 
                 vina_path: Optional[str] = None):
        """
        Args:
            feature_set: 'minimal' (10 features) or 'extended' (20 features)
            vina_path: path to Vina executable (optional, for scoring features)
        """
        ...
    
    def extract(self, ligand_pos, ligand_types, pocket_pos, pocket_types,
                ligand_mol=None):
        """
        Args:
            ligand_pos: (N_L, 3) atom positions
            ligand_types: (N_L,) atom type indices
            pocket_pos: (N_P, 3) residue positions
            pocket_types: (N_P,) residue type indices
            ligand_mol: RDKit Mol object (optional, for property features)
        
        Returns:
            features: (d_p,) numpy array of physics features
            feature_names: list of str
        """
        ...
    
    def extract_batch(self, complexes):
        """Extract features for a batch of complexes."""
        ...
    
    # Individual feature computation methods:
    def _count_contacts(self, lig_pos, pkt_pos, cutoff): ...
    def _count_hbonds(self, lig_pos, lig_types, pkt_pos, pkt_types): ...
    def _count_hydrophobic(self, lig_pos, lig_types, pkt_pos, pkt_types): ...
    def _compute_vina_score(self, ligand_mol, pocket_pdb): ...
    def _compute_ligand_properties(self, ligand_mol): ...
    def _compute_pocket_properties(self, pocket_pos, pocket_types): ...


class PhysicsFeatureNormalizer:
    """Normalize physics features to zero mean, unit variance."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, features):
        """Compute mean and std from training features."""
        ...
    
    def transform(self, features):
        """Apply normalization."""
        ...
    
    def fit_transform(self, features):
        ...
```

### 4.2 New Module: `bayesdiff/feature_fusion.py`

```python
class EarlyFusion(nn.Module):
    """Concatenate z and p, then project."""
    def __init__(self, embed_dim, phys_dim, output_dim):
        ...
    def forward(self, z, p):
        return self.proj(torch.cat([z, p], dim=-1))


class LateFusion(nn.Module):
    """Separate processing of z and p, combine predictions."""
    def __init__(self, embed_dim, phys_dim):
        ...
    def forward(self, z, p):
        return self.data_head(z) + self.phys_head(p)


class GatedFusion(nn.Module):
    """Gated combination of data-driven and physics features."""
    def __init__(self, embed_dim, phys_dim, output_dim):
        ...
    def forward(self, z, p):
        gate = torch.sigmoid(self.gate_net(torch.cat([z, p], dim=-1)))
        return gate * self.z_proj(z) + (1 - gate) * self.p_proj(p)


class PhysicsPriorGP:
    """GP with physics-based mean function."""
    def __init__(self, embed_dim, phys_dim, n_inducing=512):
        ...
    # Mean function: m(z) = β·p + β₀
    # Kernel: k(z, z') as before
```

### 4.3 New Pipeline Script: `scripts/pipeline/s09b_extract_physics_features.py`

```python
"""
Extract physics-aware features for all generated molecules.

Usage:
    python scripts/pipeline/s09b_extract_physics_features.py \
        --atom_dir data/atom_embeddings/ \
        --pdbbind_dir data/pdbbind_v2020/ \
        --output data/physics_features.npz \
        --feature_set minimal \
        --vina_path /path/to/vina  # optional

Output:
    data/physics_features.npz
        - 'features': (N_total, d_p) physics feature matrix
        - 'feature_names': list of feature names
        - 'pocket_ids': (N_total,) pocket index
        - 'mol_ids': (N_total,) molecule index
"""
```

### 4.4 Modifications to Downstream Modules

**`bayesdiff/gen_uncertainty.py`**: Physics features are per-molecule but deterministic given the generated 3D structure. They don't contribute to generation uncertainty directly, but the fused representation $[z; p]$ does.

**`bayesdiff/fusion.py`**: Delta method Jacobian extends to include physics features:

$$
J_\mu = \frac{\partial \mu}{\partial [z; p]}
$$

For physics features that are pre-computed and fixed, $\partial p / \partial z = 0$, so generation uncertainty only propagates through $z$.

---

## 5. Test Plan

### 5.1 Unit Tests: `tests/stage2/test_physics_features.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T1.1 | `test_contact_counting` | Correct contact count for known geometry |
| T1.2 | `test_hbond_detection` | H-bonds detected with correct distance/angle criteria |
| T1.3 | `test_hydrophobic_contacts` | Hydrophobic contacts match expected count |
| T1.4 | `test_ligand_properties` | MW, logP, TPSA match RDKit reference values |
| T1.5 | `test_feature_dimensions` | Minimal set = 10 dims, extended = 20 dims |
| T1.6 | `test_feature_determinism` | Same input → same features |
| T1.7 | `test_normalizer_fit_transform` | Zero mean, unit variance after normalization |
| T1.8 | `test_normalizer_inverse` | Inverse transform recovers original values |
| T1.9 | `test_missing_ligand_mol` | Graceful handling when RDKit Mol is unavailable |
| T1.10 | `test_no_contacts_case` | All contact features = 0 when ligand far from pocket |

```python
def test_contact_counting():
    """Known geometry: 3 ligand atoms near pocket, 2 far away."""
    lig_pos = torch.tensor([
        [0.0, 0.0, 0.0],   # Near pocket atom at (1,0,0) → contact
        [1.0, 1.0, 0.0],   # Near pocket atom at (1,0,0) → contact (d≈1.4)
        [10.0, 0.0, 0.0],  # Far from pocket → no contact
        [0.5, 0.5, 0.5],   # Near pocket atom at (1,0,0) → contact (d≈0.87)
        [20.0, 20.0, 20.0] # Far from pocket → no contact
    ])
    pkt_pos = torch.tensor([[1.0, 0.0, 0.0]])
    
    extractor = PhysicsFeatureExtractor()
    n = extractor._count_contacts(lig_pos, pkt_pos, cutoff=4.5)
    assert n == 3  # First 3 atoms are within 4.5 Å
```

### 5.2 Fusion Tests: `tests/stage2/test_feature_fusion.py`

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T2.1 | `test_early_fusion_shape` | Output dim correct |
| T2.2 | `test_late_fusion_additivity` | Output ≈ data_pred + phys_pred |
| T2.3 | `test_gated_fusion_gate_range` | Gate values ∈ [0, 1] |
| T2.4 | `test_gated_fusion_gradient_flow` | Gradients through both branches |
| T2.5 | `test_physics_prior_gp` | GP mean function uses physics features |

### 5.3 Integration Tests

| Test ID | Test Name | What It Verifies |
|---------|-----------|-----------------|
| T3.1 | `test_physics_with_gp_oracle` | Physics-augmented features → GP works |
| T3.2 | `test_physics_with_gen_uncertainty` | Generation uncertainty propagates correctly with physics features |
| T3.3 | `test_physics_with_delta_method` | Delta method handles mixed z+p input |
| T3.4 | `test_full_pipeline_physics` | End-to-end with physics features |
| T3.5 | `test_physics_feature_importance` | Feature importance via GP ARD lengthscales |

### 5.4 Ablation Experiments

| Ablation ID | Configuration | Purpose |
|-------------|--------------|---------|
| A6.1 | No physics features (baseline) | Reference |
| A6.2 | Minimal set (10 features), early fusion | Basic addition |
| A6.3 | Extended set (20 features), early fusion | More features |
| A6.4 | Minimal set, gated fusion | Adaptive weighting |
| A6.5 | Minimal set, late fusion | Additive model |
| A6.6 | Minimal set, physics-prior GP | GP with informed mean |
| A6.7 | Individual feature groups (contact only) | Feature group importance |
| A6.8 | Individual feature groups (interaction only) | Feature group importance |
| A6.9 | Individual feature groups (ligand props only) | Feature group importance |
| A6.10 | Physics features + enhanced repr (Sub-Plans 1–3) | Combined upgrades |

---

## 6. Evaluation & Success Criteria

### 6.1 Quantitative Metrics

| Metric | Baseline | Success | Stretch |
|--------|----------|---------|---------|
| $R^2$ | 0.120 | ≥ 0.16 | ≥ 0.22 |
| Spearman $\rho$ | 0.369 | ≥ 0.43 | ≥ 0.50 |
| NLL | baseline | ≥ 3% reduction | ≥ 10% reduction |

### 6.2 Diagnostic Metrics

- **Feature importance ranking**: Via GP ARD lengthscales or gradient-based saliency
- **Gated fusion gate analysis**: Which dimensions prefer physics vs. learned features?
- **Feature correlation with pKd**: Pearson/Spearman per individual physics feature
- **Feature redundancy**: Correlation matrix between physics features and learned embedding dimensions

### 6.3 Expected Outcomes

1. Contact-based features should improve prediction for "easy" cases (clear binding mode)
2. Vina score provides strong baseline signal that GP can refine
3. Gated fusion should outperform naive concatenation (learns when to trust physics)
4. Physics features have diminishing returns when combined with strong representations (Sub-Plans 1–3)

---

## 7. Paper Integration

### 7.1 Methods Section (Draft)

> **§3.W Physics-Informed Feature Augmentation**
> 
> To supplement the data-driven molecular representation with interpretable binding physics, we compute a curated set of $d_p$ physics-aware features for each generated protein-ligand complex. These features encode geometric contacts (distance-based contact counts), specific interactions (hydrogen bonds, hydrophobic contacts), molecular properties (MW, logP, TPSA), and optionally scoring function outputs (Vina score).
> 
> We integrate physics features $p \in \mathbb{R}^{d_p}$ with the learned embedding $z \in \mathbb{R}^d$ via a gated fusion mechanism:
> 
> $$g = \sigma(W_g [z; p] + b_g)$$
> $$u = g \odot W_z z + (1 - g) \odot W_p p$$
> 
> The gate $g$ learns to adaptively balance data-driven and physics-based features on a per-dimension basis. In the Delta method uncertainty propagation, physics features contribute as fixed observations (zero generation variance), while the learned embedding $z$ propagates generation uncertainty as before.

### 7.2 Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig. P.1 | Feature importance bar chart (ARD lengthscales) | Which physics features matter |
| Fig. P.2 | Gate value distribution (gated fusion) | Physics vs. learned reliance |
| Fig. P.3 | Individual feature vs. pKd scatter plots | Feature quality check |
| Fig. P.4 | Ablation: feature set × fusion method | Design justification |

### 7.3 Tables

| Table | Content |
|-------|---------|
| Tab. P.1 | Physics feature list with descriptions and computation methods |
| Tab. P.2 | Feature set ablation (A6.1–A6.6) |
| Tab. P.3 | Feature group importance (A6.7–A6.9) |

---

## 8. Technical Considerations

### 8.1 Feature Computation Dependencies

| Feature Category | Required Software | Installation |
|-----------------|-------------------|-------------|
| Contact features | NumPy/SciPy only | ✅ Available |
| H-bond/hydrophobic | BioPython + distance criteria | `pip install biopython` |
| Ligand properties | RDKit | ✅ Available in env |
| Vina score | AutoDock Vina binary | ⚠️ Optional, separate install |
| Shape complementarity | PyMOL or custom code | ⚠️ Complex, defer to extended set |
| Pocket properties | fpocket or BioPython | `pip install fpocket` |

### 8.2 Handling Missing Features

Some features may be unavailable for certain complexes (e.g., Vina score for invalid geometries):
- Use feature-specific defaults (mean imputation from training set)
- Mark missing features in a mask; exclude from gate computation
- Log missing feature statistics for quality control

### 8.3 Generation Uncertainty with Physics Features

For $M$ generated molecules from the same pocket:
- Physics features vary across molecules (different 3D structures → different contacts)
- Compute $p_1, \dots, p_M$ for each molecule
- The variance in physics features across molecules is an additional generation uncertainty signal
- Can optionally include physics feature variance in total $\sigma^2_{\text{gen}}$

---

## 9. Implementation Checklist

- [ ] Implement `PhysicsFeatureExtractor` with minimal feature set
- [ ] Implement contact counting (distance-based)
- [ ] Implement H-bond and hydrophobic contact detection
- [ ] Implement ligand property computation (RDKit)
- [ ] Implement `PhysicsFeatureNormalizer`
- [ ] Implement `EarlyFusion`, `LateFusion`, `GatedFusion`
- [ ] Implement `PhysicsPriorGP` (GP with physics mean function)
- [ ] Write `s09b_extract_physics_features.py` pipeline script
- [ ] Write unit tests (T1.1–T1.10)
- [ ] Write fusion tests (T2.1–T2.5)
- [ ] Write integration tests (T3.1–T3.5)
- [ ] Run ablation experiments (A6.1–A6.10)
- [ ] Analyze feature importance
- [ ] Generate paper figures and tables
- [ ] Draft methods section text
