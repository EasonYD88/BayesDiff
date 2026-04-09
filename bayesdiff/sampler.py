"""
bayesdiff/sampler.py — §4.1 Generation Module
────────────────────────────────────────────────
Wrapper around TargetDiff for pocket-conditioned molecular sampling
and SE(3)-equivariant embedding extraction.
Paper reference: §4.1 "Molecular Generation and Embedding Extraction"

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

    # ── Data loading from .pt files ──────────────────────────────────────

    def load_pocket_data(self, pt_path: str | Path):
        """Load a pre-processed pocket Data object from a .pt file.

        These files are produced by scripts/scaling/s03_prepare_tier3.py from LMDB.
        The data dict contains protein_element, protein_pos, etc.
        We reconstruct a PyG Data object and apply FeaturizeProteinAtom.
        """
        self._load_model()  # Ensure transform is ready

        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"Pocket data file not found: {pt_path}")

        pocket_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

        # Import PyG Data class
        try:
            from datasets.pl_data import ProteinLigandData
        except ImportError:
            from torch_geometric.data import Data as ProteinLigandData

        # Build Data object from stored protein fields
        protein_dict = {
            "element": pocket_dict["protein_element"],
            "pos": pocket_dict["protein_pos"],
            "is_backbone": pocket_dict["protein_is_backbone"],
            "atom_name": pocket_dict["protein_atom_name"],
            "atom_to_aa_type": pocket_dict["protein_atom_to_aa_type"],
            "molecule_name": pocket_dict.get("protein_molecule_name", "pocket"),
        }
        ligand_dict = {
            "element": torch.empty([0], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0], dtype=torch.long),
        }

        if hasattr(ProteinLigandData, "from_protein_ligand_dicts"):
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=protein_dict, ligand_dict=ligand_dict
            )
        else:
            # Fallback: build Data manually with protein_ prefix
            from torch_geometric.data import Data
            data = Data()
            for k, v in protein_dict.items():
                setattr(data, f"protein_{k}", v)
            for k, v in ligand_dict.items():
                setattr(data, f"ligand_{k}", v)

        # Apply protein featurizer transform
        data = self._transform(data)
        return data

    def sample_for_data(
        self,
        data,
        num_samples: int = 4,
        batch_size: Optional[int] = None,
    ) -> dict:
        """Sample molecules given a pre-built PyG Data object.

        Same as sample_for_pocket but skips PDB parsing.
        The data object must already have protein_atom_feature
        (apply FeaturizeProteinAtom first, or use load_pocket_data).

        Parameters
        ----------
        data : PyG Data
            Featurized pocket data with protein_atom_feature field.
        num_samples : int
            Number of molecules to generate.
        batch_size : int | None
            Batch size for sampling. Defaults to num_samples.

        Returns
        -------
        dict with keys: pred_pos, pred_v, mol_embeddings, time_list
        """
        self._load_model()

        if batch_size is None:
            batch_size = num_samples

        from scripts.sample_diffusion import sample_diffusion_ligand

        sample_cfg = self._sample_config.sample

        result = sample_diffusion_ligand(
            self._model, data, num_samples,
            batch_size=batch_size, device=self.device,
            num_steps=self.num_steps,
            pos_only=sample_cfg.pos_only,
            center_pos_mode=sample_cfg.center_pos_mode,
            sample_num_atoms=sample_cfg.sample_num_atoms,
        )

        if len(result) == 8:
            all_pred_pos, all_pred_v, _, _, _, _, time_list, mol_embeddings = result
        elif len(result) == 7:
            all_pred_pos, all_pred_v, _, _, _, _, time_list = result
            mol_embeddings = [None] * len(all_pred_pos)
        else:
            raise ValueError(
                f"Unexpected number of return values from "
                f"sample_diffusion_ligand: {len(result)}"
            )

        return {
            "pred_pos": all_pred_pos,
            "pred_v": all_pred_v,
            "mol_embeddings": mol_embeddings,
            "time_list": time_list,
        }

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

        result = sample_diffusion_ligand(
            self._model, data, num_samples,
            batch_size=batch_size, device=self.device,
            num_steps=self.num_steps,
            pos_only=sample_cfg.pos_only,
            center_pos_mode=sample_cfg.center_pos_mode,
            sample_num_atoms=sample_cfg.sample_num_atoms,
        )

        # TargetDiff returns 7 values; our earlier draft expected 8 (with
        # mol_embeddings).  Handle both gracefully.
        if len(result) == 8:
            all_pred_pos, all_pred_v, _, _, _, _, time_list, mol_embeddings = result
        elif len(result) == 7:
            all_pred_pos, all_pred_v, _, _, _, _, time_list = result
            mol_embeddings = [None] * len(all_pred_pos)
        else:
            raise ValueError(
                f"Unexpected number of return values from "
                f"sample_diffusion_ligand: {len(result)}"
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

    def sample_and_embed_data(
        self,
        data,
        num_samples: int = 4,
        save_sdf: Optional[str | Path] = None,
    ) -> tuple[list, np.ndarray]:
        """Sample molecules from pre-built Data object and return (mols, embeddings).

        Same as sample_and_embed but takes a PyG Data object instead of PDB path.
        Use load_pocket_data() to create the Data object from a .pt file.
        """
        batch_size = min(16, num_samples)
        while True:
            try:
                result = self.sample_for_data(
                    data, num_samples=num_samples, batch_size=batch_size,
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        f"CUDA OOM – retrying with batch_size={batch_size}"
                    )
                else:
                    raise

        mols = self.reconstruct_molecules(result["pred_pos"], result["pred_v"])

        embeddings = []
        for emb in result["mol_embeddings"]:
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.hidden_dim, dtype=np.float32))
        embeddings = np.stack(embeddings, axis=0)

        if save_sdf is not None:
            save_sdf = Path(save_sdf)
            save_sdf.parent.mkdir(parents=True, exist_ok=True)
            self._save_sdf(mols, save_sdf)
            logger.info(f"Saved {sum(1 for m in mols if m)} molecules to {save_sdf}")

        return mols, embeddings

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
        # Use batch_size=16 to avoid CUDA OOM on large pockets, with
        # automatic retry at half the batch size on OOM.
        batch_size = min(16, num_samples)
        while True:
            try:
                result = self.sample_for_pocket(
                    pocket_pdb, num_samples=num_samples, batch_size=batch_size,
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        f"CUDA OOM – retrying with batch_size={batch_size}"
                    )
                else:
                    raise

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

    # ── Multi-layer embedding extraction ─────────────────────────────────

    def load_complex_data(self, pt_path: str | Path):
        """Load a pre-processed complex (protein + ligand) from a .pt file.

        Unlike load_pocket_data() which discards ligand info, this method
        preserves both protein and ligand data for embedding extraction.

        Parameters
        ----------
        pt_path : path
            Path to a .pt file from data/pdbbind_v2020/processed/.

        Returns
        -------
        data : PyG Data
            Featurized complex with protein_atom_feature and
            ligand_atom_feature_full fields.
        """
        self._load_model()

        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"Complex data file not found: {pt_path}")

        pocket_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

        try:
            from datasets.pl_data import ProteinLigandData
        except ImportError:
            from torch_geometric.data import Data as ProteinLigandData

        protein_dict = {
            "element": pocket_dict["protein_element"],
            "pos": pocket_dict["protein_pos"],
            "is_backbone": pocket_dict["protein_is_backbone"],
            "atom_name": pocket_dict["protein_atom_name"],
            # Clamp non-standard amino acids (value >= 20) to 19
            # to avoid F.one_hot overflow in FeaturizeProteinAtom
            "atom_to_aa_type": pocket_dict["protein_atom_to_aa_type"].clamp(max=19),
            "molecule_name": pocket_dict.get("protein_molecule_name", "pocket"),
        }
        ligand_dict = {
            "element": pocket_dict["ligand_element"],
            "pos": pocket_dict["ligand_pos"],
            "atom_feature": pocket_dict["ligand_atom_feature"],
            "bond_index": pocket_dict["ligand_bond_index"],
            "bond_type": pocket_dict["ligand_bond_type"],
        }

        if hasattr(ProteinLigandData, "from_protein_ligand_dicts"):
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=protein_dict, ligand_dict=ligand_dict
            )
        else:
            from torch_geometric.data import Data
            data = Data()
            for k, v in protein_dict.items():
                setattr(data, f"protein_{k}", v)
            for k, v in ligand_dict.items():
                setattr(data, f"ligand_{k}", v)

        # Apply protein featurizer
        data = self._protein_featurizer(data)

        # Compute ligand_atom_feature_full manually because
        # FeaturizeLigandAtom requires ligand_hybridization which
        # is NOT stored in the PDBbind .pt files.
        # In 'add_aromatic' mode, only element + aromatic are needed.
        try:
            from utils.transforms import MAP_ATOM_TYPE_AROMATIC_TO_INDEX
            from utils.data import ATOM_FAMILIES_ID
            AROMATIC_IDX = ATOM_FAMILIES_ID['Aromatic']
        except ImportError:
            MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
                (1, False): 0, (6, False): 1, (6, True): 2,
                (7, False): 3, (7, True): 4, (8, False): 5,
                (8, True): 6, (9, False): 7, (15, False): 8,
                (15, True): 9, (16, False): 10, (16, True): 11,
                (17, False): 12,
            }
            AROMATIC_IDX = 2

        element_list = data.ligand_element
        aromatic_list = data.ligand_atom_feature[:, AROMATIC_IDX]
        feat_full = []
        for e, a in zip(element_list, aromatic_list):
            key = (int(e), bool(a))
            if key in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
                feat_full.append(MAP_ATOM_TYPE_AROMATIC_TO_INDEX[key])
            else:
                # Unknown element → fallback to H (index 0)
                feat_full.append(0)
        data.ligand_atom_feature_full = torch.tensor(feat_full, dtype=torch.long)

        return data

    @torch.no_grad()
    def extract_multilayer_embeddings(
        self,
        pt_path: str | Path,
    ) -> dict[str, np.ndarray]:
        """Extract per-layer mean-pooled embeddings from a crystal complex.

        Runs a single forward pass through the encoder with
        return_layer_h=True, using the crystal ligand pose (fix_x=True).

        Parameters
        ----------
        pt_path : path
            Path to a processed complex .pt file.

        Returns
        -------
        dict with keys:
            'layer_0' ... 'layer_9': (d,) mean-pooled ligand embedding per layer
            'z_global': (d,) last-layer mean-pooled (backward compatible)
            'n_layers': int, total number of layers
        """
        self._load_model()

        data = self.load_complex_data(pt_path)

        from models.common import compose_context
        from torch_geometric.data import Batch
        try:
            from datasets.pl_data import FOLLOW_BATCH
        except ImportError:
            FOLLOW_BATCH = ["protein_element", "ligand_element"]

        # Create a single-item batch
        batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH).to(self.device)

        batch_protein = batch.protein_element_batch
        batch_ligand = batch.ligand_element_batch

        # Featurize for model input
        h_protein = self._model.protein_atom_emb(
            batch.protein_atom_feature.float()
        )
        h_ligand = self._model.ligand_atom_emb(
            torch.nn.functional.one_hot(
                batch.ligand_atom_feature_full, self._model.num_classes
            ).float()
        )

        if self._model.config.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1, device=self.device)], -1
            )
            h_ligand = torch.cat(
                [h_ligand, torch.ones(len(h_ligand), 1, device=self.device)], -1
            )

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=batch.protein_pos,
            pos_ligand=batch.ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self._model.refine_net(
            h_all, pos_all, mask_ligand, batch_all,
            return_all=False, fix_x=True, return_layer_h=True,
        )

        layer_h_list = outputs.get("layer_h_list", [])
        result = {"n_layers": len(layer_h_list)}

        for i, h in enumerate(layer_h_list):
            ligand_h = h[mask_ligand]  # (N_lig, d)
            z = ligand_h.mean(dim=0).cpu().numpy()  # (d,)
            result[f"layer_{i}"] = z

        # Last layer = z_global for backward compatibility
        if layer_h_list:
            last_h = layer_h_list[-1][mask_ligand]
            result["z_global"] = last_h.mean(dim=0).cpu().numpy()

        return result

    @torch.no_grad()
    def extract_multilayer_atom_embeddings(
        self,
        pt_path: str | Path,
    ) -> dict[str, np.ndarray | int]:
        """Extract per-layer ATOM-LEVEL embeddings from a crystal complex.

        Same encoder forward pass as extract_multilayer_embeddings, but
        returns full (N_lig, d) tensors instead of mean-pooled (d,).

        Parameters
        ----------
        pt_path : path
            Path to a processed complex .pt file.

        Returns
        -------
        dict with keys:
            'layer_0' ... 'layer_9': (N_lig, d) atom-level ligand embedding
            'pocket_mean': (d,) mean-pooled pocket embedding from last layer
            'n_ligand_atoms': int
            'n_layers': int
        """
        self._load_model()

        data = self.load_complex_data(pt_path)

        from models.common import compose_context
        from torch_geometric.data import Batch
        try:
            from datasets.pl_data import FOLLOW_BATCH
        except ImportError:
            FOLLOW_BATCH = ["protein_element", "ligand_element"]

        batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH).to(self.device)

        batch_protein = batch.protein_element_batch
        batch_ligand = batch.ligand_element_batch

        h_protein = self._model.protein_atom_emb(
            batch.protein_atom_feature.float()
        )
        h_ligand = self._model.ligand_atom_emb(
            torch.nn.functional.one_hot(
                batch.ligand_atom_feature_full, self._model.num_classes
            ).float()
        )

        if self._model.config.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1, device=self.device)], -1
            )
            h_ligand = torch.cat(
                [h_ligand, torch.ones(len(h_ligand), 1, device=self.device)], -1
            )

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=batch.protein_pos,
            pos_ligand=batch.ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self._model.refine_net(
            h_all, pos_all, mask_ligand, batch_all,
            return_all=False, fix_x=True, return_layer_h=True,
        )

        layer_h_list = outputs.get("layer_h_list", [])
        mask_pocket = ~mask_ligand
        result = {
            "n_layers": len(layer_h_list),
        }

        for i, h in enumerate(layer_h_list):
            ligand_h = h[mask_ligand]  # (N_lig, d)
            result[f"layer_{i}"] = ligand_h.cpu().numpy()

        # Pocket mean from last layer (for optional cross-attention)
        if layer_h_list:
            pocket_h = layer_h_list[-1][mask_pocket]
            result["pocket_mean"] = pocket_h.mean(dim=0).cpu().numpy()

        result["n_ligand_atoms"] = int(mask_ligand.sum().item())
        return result

    @property
    def num_encoder_layers(self) -> int:
        """Return total number of hookable encoder layers (init + base_block)."""
        self._load_model()
        n_base = len(self._model.refine_net.base_block)
        return 1 + n_base  # init_h_emb_layer + base_block layers
