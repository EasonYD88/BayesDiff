"""
bayesdiff/pretrain_dataset.py — PDBbind Pair-Level Dataset
──────────────────────────────────────────────────────────
PyTorch Dataset for supervised pretraining on PDBbind v2020 refined set.

Each sample returns:
    - protein_pos:    (N_pocket, 3)    pocket atom coordinates
    - protein_feat:   (N_pocket, d_p)  pocket atom features
    - ligand_pos:     (N_ligand, 3)    ligand atom coordinates
    - ligand_feat:    (N_ligand, d_l)  ligand atom features
    - pkd:            scalar           experimental pKd label

Uses the same featurizers as TargetDiff to ensure feature dimension
consistency between pretraining and generation-time evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class PDBbindPairDataset(Dataset):
    """Protein-ligand pair dataset from PDBbind v2020 processed .pt files.

    Expected directory structure (produced by s00_prepare_pdbbind.py):
        data/pdbbind_v2020/
        ├── processed/        # .pt files per complex
        │   ├── XXXX.pt
        │   └── ...
        ├── labels.csv        # pdb_code, pkd, ...
        └── splits.json       # {train: [...], val: [...], ...}
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        fold_id: Optional[int] = None,
        transform=None,
    ):
        """
        Args:
            data_dir:  Root directory containing processed/, labels.csv, splits.json.
            split:     One of 'train', 'val', 'cal', 'test', or 'all'.
            fold_id:   If specified, load train/val from splits_5fold.json
                       using this fold index (0–4). Test split always uses
                       the fixed CASF-2016 codes. If None, uses splits.json.
            transform: Optional callable applied to each data dict.
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.transform = transform

        # Load splits — either from splits_5fold.json or splits.json
        if fold_id is not None:
            fivefold_path = self.data_dir / "splits_5fold.json"
            if not fivefold_path.exists():
                raise FileNotFoundError(
                    f"splits_5fold.json not found at {fivefold_path}. "
                    f"Run s00_prepare_pdbbind.py --stage split first."
                )
            with open(fivefold_path) as f:
                fivefold = json.load(f)
            fold_key = str(fold_id)
            if fold_key not in fivefold["folds"]:
                available = list(fivefold["folds"].keys())
                raise ValueError(f"fold_id={fold_id} not found. Available: {available}")
            if split == "test":
                self.pdb_codes = fivefold["test"]
            elif split in ("train", "val"):
                self.pdb_codes = fivefold["folds"][fold_key][split]
            elif split == "all":
                self.pdb_codes = (
                    fivefold["folds"][fold_key]["train"]
                    + fivefold["folds"][fold_key]["val"]
                    + fivefold["test"]
                )
            else:
                raise ValueError(
                    f"Unknown split '{split}' for fold mode. Use 'train', 'val', 'test', or 'all'."
                )
            logger.info(f"Using fold {fold_id} (seed={fivefold['folds'][fold_key]['seed']})")
        else:
            splits_path = self.data_dir / "splits.json"
            if not splits_path.exists():
                raise FileNotFoundError(f"splits.json not found at {splits_path}")
            with open(splits_path) as f:
                splits = json.load(f)

            if split == "all":
                self.pdb_codes = []
                for codes in splits.values():
                    self.pdb_codes.extend(codes)
            elif split not in splits:
                raise ValueError(f"Unknown split '{split}'. Available: {list(splits.keys())}")
            else:
                self.pdb_codes = splits[split]

        # Filter to only codes that have .pt files
        available = set(p.stem for p in self.processed_dir.glob("*.pt"))
        before = len(self.pdb_codes)
        self.pdb_codes = [c for c in self.pdb_codes if c in available]
        if len(self.pdb_codes) < before:
            logger.warning(
                f"Split '{split}': {before} → {len(self.pdb_codes)} codes "
                f"({before - len(self.pdb_codes)} missing .pt files)"
            )

        # Load labels for quick access
        labels_path = self.data_dir / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            self._pkd_map = dict(zip(df["pdb_code"], df["pkd"]))
        else:
            self._pkd_map = {}

        logger.info(f"PDBbindPairDataset: split={split}, n={len(self.pdb_codes)}")

    def __len__(self) -> int:
        return len(self.pdb_codes)

    def __getitem__(self, idx: int) -> dict:
        pdb_code = self.pdb_codes[idx]
        pt_path = self.processed_dir / f"{pdb_code}.pt"
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        sample = {
            "pdb_code": pdb_code,
            "protein_pos": data["protein_pos"],
            "protein_element": data["protein_element"],
            "protein_is_backbone": data["protein_is_backbone"],
            "protein_atom_to_aa_type": data["protein_atom_to_aa_type"],
            "ligand_pos": data["ligand_pos"],
            "ligand_element": data["ligand_element"],
            "ligand_atom_feature": data["ligand_atom_feature"],
            "ligand_bond_index": data["ligand_bond_index"],
            "ligand_bond_type": data["ligand_bond_type"],
            "pkd": data["pkd"],
            "n_protein_atoms": data["n_protein_atoms"],
            "n_ligand_atoms": data["n_ligand_atoms"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_pair_data(batch: list[dict]) -> dict:
    """Collate variable-size protein-ligand pairs into a batch.

    Uses concatenation with batch indices (similar to PyG batching)
    rather than padding, to be memory-efficient.
    """
    pdb_codes = [b["pdb_code"] for b in batch]
    pkd = torch.stack([b["pkd"] for b in batch])

    # Protein: concatenate with batch indices
    protein_pos_list = []
    protein_elem_list = []
    protein_backbone_list = []
    protein_aa_list = []
    protein_batch_idx = []

    # Ligand: concatenate with batch indices + offset bond indices
    ligand_pos_list = []
    ligand_elem_list = []
    ligand_feat_list = []
    ligand_bond_idx_list = []
    ligand_bond_type_list = []
    ligand_batch_idx = []

    protein_offset = 0
    ligand_offset = 0

    for i, b in enumerate(batch):
        n_p = b["n_protein_atoms"]
        n_l = b["n_ligand_atoms"]

        protein_pos_list.append(b["protein_pos"])
        protein_elem_list.append(b["protein_element"])
        protein_backbone_list.append(b["protein_is_backbone"])
        protein_aa_list.append(b["protein_atom_to_aa_type"])
        protein_batch_idx.append(torch.full((n_p,), i, dtype=torch.long))

        ligand_pos_list.append(b["ligand_pos"])
        ligand_elem_list.append(b["ligand_element"])
        ligand_feat_list.append(b["ligand_atom_feature"])
        ligand_batch_idx.append(torch.full((n_l,), i, dtype=torch.long))

        # Offset bond indices
        bond_idx = b["ligand_bond_index"]
        if bond_idx.numel() > 0:
            ligand_bond_idx_list.append(bond_idx + ligand_offset)
            ligand_bond_type_list.append(b["ligand_bond_type"])

        protein_offset += n_p
        ligand_offset += n_l

    result = {
        "pdb_code": pdb_codes,
        "pkd": pkd,
        "protein_pos": torch.cat(protein_pos_list, dim=0),
        "protein_element": torch.cat(protein_elem_list, dim=0),
        "protein_is_backbone": torch.cat(protein_backbone_list, dim=0),
        "protein_atom_to_aa_type": torch.cat(protein_aa_list, dim=0),
        "protein_batch": torch.cat(protein_batch_idx, dim=0),
        "ligand_pos": torch.cat(ligand_pos_list, dim=0),
        "ligand_element": torch.cat(ligand_elem_list, dim=0),
        "ligand_atom_feature": torch.cat(ligand_feat_list, dim=0),
        "ligand_batch": torch.cat(ligand_batch_idx, dim=0),
        "batch_size": len(batch),
    }

    if ligand_bond_idx_list:
        result["ligand_bond_index"] = torch.cat(ligand_bond_idx_list, dim=1)
        result["ligand_bond_type"] = torch.cat(ligand_bond_type_list, dim=0)
    else:
        result["ligand_bond_index"] = torch.empty([2, 0], dtype=torch.long)
        result["ligand_bond_type"] = torch.empty([0], dtype=torch.long)

    return result


def get_pdbbind_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    pin_memory: bool = True,
    fold_id: Optional[int] = None,
) -> DataLoader:
    """Convenience function to create a PDBbind DataLoader."""
    dataset = PDBbindPairDataset(data_dir, split=split, fold_id=fold_id, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_data,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
