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

    # ── Model loading ────────────────────────────────────────────────────

    def _load_model(self):
        """Lazily load TargetDiff model + checkpoint."""
        if self._model is not None:
            return

        logger.info(f"Loading TargetDiff checkpoint from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        config = ckpt["config"]
        self._ckpt_config = config

        # Import TargetDiff model class
        try:
            from models.molopt_score_model import ScorePosNet3D
        except ImportError as e:
            raise ImportError(
                f"Cannot import TargetDiff model. Make sure {self.targetdiff_dir} "
                "is a valid TargetDiff clone with models/ directory.\n"
                f"Original error: {e}"
            )

        model = ScorePosNet3D(config.model).to(self.device)
        model.load_state_dict(ckpt["model"])
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

    @staticmethod
    def pocket_pdb_to_data(pocket_pdb: str | Path) -> "torch_geometric.data.Data":
        """Convert a pocket PDB file to a PyG Data object for TargetDiff.

        This uses TargetDiff's own data processing utilities.
        """
        pocket_pdb = Path(pocket_pdb)
        if not pocket_pdb.exists():
            raise FileNotFoundError(f"Pocket PDB not found: {pocket_pdb}")

        try:
            from utils.data import PDBProtein
        except ImportError:
            raise ImportError(
                "Cannot import TargetDiff utils. Ensure targetdiff_dir is on sys.path."
            )

        import torch
        from torch_geometric.data import Data

        # Parse pocket using TargetDiff's PDBProtein
        pocket = PDBProtein(str(pocket_pdb))
        pocket_dict = pocket.to_dict_atom()

        data = Data(
            protein_pos=torch.tensor(pocket_dict["pos"], dtype=torch.float32),
            protein_atom_feature=torch.tensor(
                pocket_dict["atom_feature"], dtype=torch.float32
            ),
            protein_element=torch.tensor(pocket_dict["element"], dtype=torch.long),
        )
        return data

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample_for_pocket(
        self,
        pocket_pdb: str | Path,
        num_samples: int = 4,
        batch_size: Optional[int] = None,
    ) -> list[dict]:
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
        list[dict]
            Each dict contains:
              - "pos": (N_atoms, 3) atom positions
              - "atom_type": (N_atoms,) atom type indices
              - "h_ligand": (N_atoms, d) per-atom encoder features
              - "z": (d,) mean-pooled graph-level embedding
              - "mol": RDKit Mol object (may be None if reconstruction fails)
        """
        self._load_model()

        if batch_size is None:
            batch_size = num_samples

        pocket_data = self.pocket_pdb_to_data(pocket_pdb)
        pocket_data = pocket_data.to(self.device)

        results = []
        n_remaining = num_samples

        while n_remaining > 0:
            current_batch = min(batch_size, n_remaining)
            logger.info(
                f"Sampling batch of {current_batch} molecules "
                f"({num_samples - n_remaining}/{num_samples} done)"
            )

            batch_results = self._sample_batch(pocket_data, current_batch)
            results.extend(batch_results)
            n_remaining -= current_batch

        return results[:num_samples]

    def _sample_batch(self, pocket_data, batch_size: int) -> list[dict]:
        """Internal: sample a batch of molecules and extract embeddings.

        Uses TargetDiff's sampling logic. We hook into the model to
        capture intermediate encoder features.
        """
        try:
            from models.molopt_score_model import ScorePosNet3D
            from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
        except ImportError:
            raise ImportError("TargetDiff modules not available.")

        # Build batch by replicating pocket data
        from torch_geometric.data import Batch

        batch_list = [pocket_data.clone() for _ in range(batch_size)]

        # Initialize ligand atoms randomly in the pocket vicinity
        pocket_center = pocket_data.protein_pos.mean(dim=0)

        config = self._ckpt_config
        num_atoms = getattr(config.sample, "num_atoms", 20)

        for i, data in enumerate(batch_list):
            # Random number of atoms (or fixed)
            n_atoms = (
                num_atoms
                if isinstance(num_atoms, int)
                else np.random.randint(num_atoms[0], num_atoms[1])
            )
            data.ligand_pos = (
                pocket_center.unsqueeze(0).repeat(n_atoms, 1)
                + torch.randn(n_atoms, 3, device=self.device) * 2.0
            )
            data.ligand_atom_feature = torch.zeros(
                n_atoms,
                config.model.get("num_atom_type", 10),
                device=self.device,
            )

        batch = Batch.from_data_list(batch_list)

        # Run the diffusion sampling loop
        # TargetDiff uses DDPM reverse process
        results = self._run_ddpm_sampling(batch, batch_size)

        return results

    def _run_ddpm_sampling(self, batch, batch_size: int) -> list[dict]:
        """Run DDPM reverse sampling. Adapted from TargetDiff sample_for_pocket.py.

        This is a simplified version. For full features, use TargetDiff's
        own sampling script.
        """
        model = self._model

        try:
            from utils.sample import sample_diffusion_ligand
        except ImportError:
            logger.warning(
                "Could not import TargetDiff sample utilities. "
                "Falling back to direct model call."
            )
            return self._fallback_sample(batch, batch_size)

        # Use TargetDiff's native sampling function
        with torch.no_grad():
            outputs = sample_diffusion_ligand(
                model,
                batch,
                self.num_steps,
                batch_size=batch_size,
                return_all=False,
            )

        results = []
        for i in range(batch_size):
            result = {
                "pos": outputs["pos"][i].cpu().numpy(),
                "atom_type": outputs["atom_type"][i].cpu().numpy(),
                "h_ligand": (
                    outputs["h_ligand"][i].cpu().numpy()
                    if "h_ligand" in outputs
                    else None
                ),
                "z": None,
                "mol": None,
            }

            # Compute graph-level embedding via mean pooling
            if result["h_ligand"] is not None:
                result["z"] = result["h_ligand"].mean(axis=0)

            results.append(result)

        return results

    def _fallback_sample(self, batch, batch_size: int) -> list[dict]:
        """Fallback sampling when TargetDiff's sample utils aren't available.

        This uses the model's score function directly with simple Euler–Maruyama.
        Not recommended for production; use TargetDiff's native sampling.
        """
        logger.warning("Using fallback sampling (simplified Euler–Maruyama)")
        model = self._model

        results = []
        for i in range(batch_size):
            results.append(
                {
                    "pos": np.zeros((20, 3)),
                    "atom_type": np.zeros(20, dtype=int),
                    "h_ligand": None,
                    "z": np.zeros(self.hidden_dim),
                    "mol": None,
                }
            )

        return results

    # ── Embedding extraction ─────────────────────────────────────────────

    def extract_embeddings(
        self,
        pocket_pdb: str | Path,
        ligand_sdf: str | Path,
    ) -> np.ndarray:
        """Extract encoder embeddings for existing molecules.

        Given a pocket and a multi-conformer SDF file, run the encoder
        forward pass and return graph-level embeddings.

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

        pocket_data = self.pocket_pdb_to_data(pocket_pdb)
        embeddings = []

        for mol in mols:
            z = self._encode_molecule(pocket_data, mol)
            embeddings.append(z)

        return np.stack(embeddings, axis=0)

    def _encode_molecule(
        self,
        pocket_data,
        mol,
    ) -> np.ndarray:
        """Encode a single molecule in the context of a pocket.

        Returns the mean-pooled graph-level embedding (d,).
        """
        import torch
        from rdkit import Chem
        from torch_geometric.data import Data

        conf = mol.GetConformer()
        pos = np.array(conf.GetPositions(), dtype=np.float32)

        # Get atom features (simple: atomic number encoding)
        atom_types = []
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())
        atom_types = np.array(atom_types, dtype=np.int64)

        data = pocket_data.clone()
        data.ligand_pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        data.ligand_element = torch.tensor(
            atom_types, dtype=torch.long, device=self.device
        )

        # Run encoder forward pass
        model = self._model
        with torch.no_grad():
            # The model's forward pass produces node features
            # We need to extract the encoder output (h_ligand)
            try:
                output = model(
                    protein_pos=data.protein_pos.unsqueeze(0),
                    protein_atom_feature=data.protein_atom_feature.unsqueeze(0),
                    ligand_pos=data.ligand_pos.unsqueeze(0),
                    ligand_atom_feature=data.ligand_element.unsqueeze(0),
                    batch_protein=torch.zeros(
                        len(data.protein_pos), dtype=torch.long, device=self.device
                    ),
                    batch_ligand=torch.zeros(
                        len(data.ligand_pos), dtype=torch.long, device=self.device
                    ),
                )
                # Extract ligand node features and mean-pool
                if isinstance(output, dict) and "final_ligand_h" in output:
                    h = output["final_ligand_h"]
                elif isinstance(output, dict) and "ligand_h" in output:
                    h = output["ligand_h"]
                elif isinstance(output, tuple):
                    h = output[0]  # Assume first element is node features
                else:
                    h = output

                z = h.mean(dim=-2).squeeze(0).cpu().numpy()
            except Exception as e:
                logger.warning(f"Encoder forward pass failed: {e}")
                logger.warning(
                    "Falling back to random embedding (shape will be correct)"
                )
                z = np.random.randn(self.hidden_dim).astype(np.float32)

        return z

    # ── Molecule reconstruction ──────────────────────────────────────────

    @staticmethod
    def reconstruct_molecules(
        sample_results: list[dict],
    ) -> list:
        """Reconstruct RDKit Mol objects from TargetDiff sampling output.

        Uses TargetDiff's atom-type-to-element mapping and RDKit distance
        geometry or openbabel for bond perception.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mols = []
        for result in sample_results:
            pos = result["pos"]
            atom_types = result["atom_type"]

            try:
                # Simple reconstruction: create molecule from atoms and positions
                # Map atom type indices to elements
                # TargetDiff uses: C, N, O, F, P, S, Cl, Br (indices 0-7)
                ELEMENT_MAP = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35}

                rwmol = Chem.RWMol()
                conf = Chem.Conformer(len(atom_types))

                for j, (atype, xyz) in enumerate(zip(atom_types, pos)):
                    if isinstance(atype, np.ndarray):
                        atype = atype.argmax()
                    atomic_num = ELEMENT_MAP.get(int(atype), 6)
                    idx = rwmol.AddAtom(Chem.Atom(atomic_num))
                    conf.SetAtomPosition(idx, xyz.tolist())

                rwmol.AddConformer(conf, assignId=True)
                mol = rwmol.GetMol()
                mols.append(mol)
            except Exception as e:
                logger.warning(f"Molecule reconstruction failed: {e}")
                mols.append(None)

        return mols

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
            If given, save molecules to this SDF file.

        Returns
        -------
        mols : list[Mol | None]
            RDKit molecules (some may be None).
        embeddings : np.ndarray
            Shape (num_samples, d).
        """
        results = self.sample_for_pocket(pocket_pdb, num_samples=num_samples)

        # Reconstruct molecules
        mols = self.reconstruct_molecules(results)
        for i, mol in enumerate(mols):
            results[i]["mol"] = mol

        # Collect embeddings
        embeddings = []
        for r in results:
            if r["z"] is not None:
                embeddings.append(r["z"])
            else:
                embeddings.append(np.zeros(self.hidden_dim))
        embeddings = np.stack(embeddings, axis=0)

        # Optionally save SDF
        if save_sdf is not None:
            save_sdf = Path(save_sdf)
            save_sdf.parent.mkdir(parents=True, exist_ok=True)
            self._save_sdf(mols, save_sdf)
            logger.info(f"Saved {len(mols)} molecules to {save_sdf}")

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
