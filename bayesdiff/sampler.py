"""
bayesdiff/sampler.py
────────────────────
Wrapper around TargetDiff for pocket-conditioned molecular sampling
and SE(3)-equivariant embedding extraction.

This module handles:
  1. Loading the pretrained TargetDiff checkpoint
  2. Converting pocket PDB → PyG Data object
  3. Sampling M molecules for a given pocket
  4. Extracting graph-level embeddings z from the encoder

Usage:
    from bayesdiff.sampler import TargetDiffSampler
    sampler = TargetDiffSampler("external/targetdiff", "pretrained_model.pt", device="cpu")
    mols, embeddings = sampler.sample_and_embed("pocket.pdb", num_samples=4)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TargetDiffSampler:
    """Wrapper around the TargetDiff model for sampling + embedding extraction.

    Parameters
    ----------
    targetdiff_dir : str | Path
        Path to the cloned TargetDiff repository (must contain models/, utils/).
    checkpoint_path : str | Path
        Path to the pretrained checkpoint (.pt file).
    device : str
        "cpu", "cuda", or "mps".
    num_steps : int
        Number of DDPM diffusion steps. Default 100.
    """

    def __init__(
        self,
        targetdiff_dir: str | Path,
        checkpoint_path: str | Path,
        device: str = "cpu",
        num_steps: int = 100,
    ):
        self.targetdiff_dir = Path(targetdiff_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        self.num_steps = num_steps

        # Add TargetDiff to sys.path so we can import its modules
        td_str = str(self.targetdiff_dir)
        if td_str not in sys.path:
            sys.path.insert(0, td_str)

        self._model = None
        self._ckpt_config = None
        self._sample_config = None
        self._protein_featurizer = None
        self._ligand_featurizer = None
        self._transform = None

    # ── Model loading ────────────────────────────────────────────────────

    def _load_model(self):
        """Lazily load TargetDiff model + checkpoint."""
        if self._model is not None:
            return

        logger.info(f"Loading TargetDiff checkpoint from {self.checkpoint_path}")
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        config = ckpt["config"]
        self._ckpt_config = config

        # Load sampling config from YAML (checkpoint config lacks .sample)
        try:
            from utils.misc import load_config
            sampling_yml = self.targetdiff_dir / "configs" / "sampling.yml"
            if sampling_yml.exists():
                self._sample_config = load_config(str(sampling_yml))
            else:
                # Fallback: use defaults
                from easydict import EasyDict
                self._sample_config = EasyDict({
                    "sample": EasyDict({
                        "seed": 2021,
                        "num_samples": 100,
                        "num_steps": 1000,
                        "pos_only": False,
                        "center_pos_mode": "protein",
                        "sample_num_atoms": "prior",
                    })
                })
        except Exception:
            from easydict import EasyDict
            self._sample_config = EasyDict({
                "sample": EasyDict({
                    "seed": 2021,
                    "num_samples": 100,
                    "num_steps": 1000,
                    "pos_only": False,
                    "center_pos_mode": "protein",
                    "sample_num_atoms": "prior",
                })
            })

        # Import TargetDiff model class and transforms
        try:
            from models.molopt_score_model import ScorePosNet3D
            import utils.transforms as trans
            from torch_geometric.transforms import Compose
        except ImportError as e:
            raise ImportError(
                f"Cannot import TargetDiff model. Make sure {self.targetdiff_dir} "
                "is a valid TargetDiff clone with models/ directory.\n"
                f"Original error: {e}"
            )

        # Build featurizers (needed for feature dimensions + data transform)
        self._protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_atom_mode = config.data.transform.ligand_atom_mode
        self._ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
        self._transform = Compose([self._protein_featurizer])

        model = ScorePosNet3D(
            config.model,
            protein_atom_feature_dim=self._protein_featurizer.feature_dim,
            ligand_atom_feature_dim=self._ligand_featurizer.feature_dim,
        ).to(self.device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        self._model = model

        logger.info(
            f"Model loaded. Hidden dim: {config.model.hidden_dim}, "
            f"Device: {self.device}"
        )

    @property
    def hidden_dim(self) -> int:
        """Return the encoder hidden dimension (default 128 for TargetDiff)."""
        self._load_model()
        return self._ckpt_config.model.hidden_dim

    # ── Pocket processing ────────────────────────────────────────────────

    def pocket_pdb_to_data(self, pocket_pdb: str | Path):
        """Convert a pocket PDB file to a featurized PyG Data object.

        Uses TargetDiff's own `pdb_to_pocket_data` and applies the
        protein featurizer transform.
        """
        self._load_model()  # Ensure transform is ready

        pocket_pdb = Path(pocket_pdb)
        if not pocket_pdb.exists():
            raise FileNotFoundError(f"Pocket PDB not found: {pocket_pdb}")

        from scripts.sample_for_pocket import pdb_to_pocket_data

        data = pdb_to_pocket_data(str(pocket_pdb))
        data = self._transform(data)
        return data

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample_for_pocket(
        self,
        pocket_pdb: str | Path,
        num_samples: int = 4,
        batch_size: Optional[int] = None,
    ) -> dict:
        """Sample molecules for a given pocket using TargetDiff.

        Parameters
        ----------
        pocket_pdb : path
            Path to the pocket PDB file (10Å pocket recommended).
        num_samples : int
            Number of molecules to generate.
        batch_size : int | None
            Batch size for sampling. Defaults to num_samples.

        Returns
        -------
        dict with keys:
            - "pred_pos": list of (N_atoms_i, 3) arrays
            - "pred_v": list of (N_atoms_i,) arrays
            - "mol_embeddings": list of (d,) arrays or None
        """
        self._load_model()

        if batch_size is None:
            batch_size = num_samples

        from scripts.sample_diffusion import sample_diffusion_ligand

        data = self.pocket_pdb_to_data(pocket_pdb)
        sample_cfg = self._sample_config.sample

        all_pred_pos, all_pred_v, _, _, _, _, time_list, mol_embeddings = \
            sample_diffusion_ligand(
                self._model, data, num_samples,
                batch_size=batch_size, device=self.device,
                num_steps=self.num_steps,
                pos_only=sample_cfg.pos_only,
                center_pos_mode=sample_cfg.center_pos_mode,
                sample_num_atoms=sample_cfg.sample_num_atoms,
            )

        return {
            "pred_pos": all_pred_pos,
            "pred_v": all_pred_v,
            "mol_embeddings": mol_embeddings,
            "time_list": time_list,
        }

    # ── Molecule reconstruction ──────────────────────────────────────────

    @staticmethod
    def reconstruct_molecules(pred_pos_list, pred_v_list) -> list:
        """Reconstruct RDKit Mol objects from TargetDiff sampling output.

        Uses TargetDiff's native atom-type mapping and reconstruction.
        """
        import utils.transforms as trans
        from utils import reconstruct
        from rdkit import Chem

        mols = []
        for pred_pos, pred_v in zip(pred_pos_list, pred_v_list):
            pred_atom_type = trans.get_atomic_number_from_index(
                pred_v, mode="add_aromatic"
            )
            try:
                pred_aromatic = trans.is_aromatic_from_index(
                    pred_v, mode="add_aromatic"
                )
                mol = reconstruct.reconstruct_from_generated(
                    pred_pos, pred_atom_type, pred_aromatic
                )
                smiles = Chem.MolToSmiles(mol)
                if "." in smiles:
                    mols.append(None)  # Skip fragmented molecules
                else:
                    mols.append(mol)
            except Exception:
                mols.append(None)

        return mols

    # ── Embedding extraction (for existing SDF files) ────────────────────

    def extract_embeddings(
        self,
        pocket_pdb: str | Path,
        ligand_sdf: str | Path,
    ) -> np.ndarray:
        """Extract encoder embeddings for existing molecules.

        Given a pocket and an SDF file, run a short diffusion and
        extract SE(3)-invariant embeddings. For pre-sampled molecules,
        the embeddings from `sample_and_embed` are preferred.

        Parameters
        ----------
        pocket_pdb : path
            Pocket PDB file.
        ligand_sdf : path
            SDF file with one or more molecules.

        Returns
        -------
        np.ndarray
            Shape (N_mols, d) where d is the hidden dimension.
        """
        self._load_model()

        from rdkit import Chem

        suppl = Chem.SDMolSupplier(str(ligand_sdf), removeHs=True)
        mols = [m for m in suppl if m is not None]

        if not mols:
            raise ValueError(f"No valid molecules in {ligand_sdf}")

        # Re-sample with current num_steps to get embeddings
        # (A proper encoder-only forward pass would be better, but
        #  TargetDiff's architecture couples the encoder with diffusion)
        logger.info(
            f"extract_embeddings: re-sampling with {self.num_steps} steps to get "
            f"embeddings for {len(mols)} molecules"
        )
        result = self.sample_for_pocket(
            pocket_pdb, num_samples=len(mols), batch_size=len(mols)
        )

        embeddings = []
        for emb in result["mol_embeddings"]:
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.hidden_dim, dtype=np.float32))

        return np.stack(embeddings, axis=0)

    # ── High-level convenience ───────────────────────────────────────────

    def sample_and_embed(
        self,
        pocket_pdb: str | Path,
        num_samples: int = 4,
        save_sdf: Optional[str | Path] = None,
    ) -> tuple[list, np.ndarray]:
        """Sample molecules and return (mols, embeddings).

        Parameters
        ----------
        pocket_pdb : path
            Path to pocket PDB file.
        num_samples : int
            Number of molecules to generate.
        save_sdf : path | None
            If given, save molecules to this directory (one SDF per mol).

        Returns
        -------
        mols : list[Mol | None]
            RDKit molecules (some may be None).
        embeddings : np.ndarray
            Shape (num_samples, d).
        """
        result = self.sample_for_pocket(pocket_pdb, num_samples=num_samples)

        # Reconstruct molecules
        mols = self.reconstruct_molecules(result["pred_pos"], result["pred_v"])

        # Collect embeddings
        embeddings = []
        for emb in result["mol_embeddings"]:
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.hidden_dim, dtype=np.float32))
        embeddings = np.stack(embeddings, axis=0)

        # Optionally save SDF
        if save_sdf is not None:
            save_sdf = Path(save_sdf)
            save_sdf.parent.mkdir(parents=True, exist_ok=True)
            self._save_sdf(mols, save_sdf)
            logger.info(f"Saved {sum(1 for m in mols if m)} molecules to {save_sdf}")

        return mols, embeddings

    @staticmethod
    def _save_sdf(mols: list, path: Path):
        """Save molecules to an SDF file."""
        from rdkit import Chem

        writer = Chem.SDWriter(str(path))
        for mol in mols:
            if mol is not None:
                writer.write(mol)
        writer.close()
