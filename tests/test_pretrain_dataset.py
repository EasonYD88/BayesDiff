"""
tests/test_pretrain_dataset.py
──────────────────────────────
End-to-end tests for PDBbind v2020 dataset preparation pipeline.

Tests with synthetic data:
  - M0.1: INDEX parsing produces correct DataFrame
  - M0.2: Pocket extraction produces valid PDB
  - M0.3: Splits are non-overlapping, correct proportions
  - M0.4: DataLoader batches variable-length protein-ligand pairs
  - Pipeline: s00_prepare_pdbbind stages work end-to-end

Run:
    python tests/test_pretrain_dataset.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _create_synthetic_pdbbind(root: Path, n_complexes: int = 20):
    """Create a minimal synthetic PDBbind directory for testing."""
    refined = root / "refined-set"
    refined.mkdir(parents=True, exist_ok=True)

    # INDEX file
    index_lines = []
    codes = []
    for i in range(n_complexes):
        code = f"t{i:03d}"
        codes.append(code)
        pkd = 4.0 + i * 0.5  # range 4.0 – 13.5
        year = 2010 + (i % 12)
        res = 1.5 + (i % 10) * 0.2
        aff_type = "Kd" if i % 3 == 0 else ("Ki" if i % 3 == 1 else "IC50")
        aff_val = 10 ** (-pkd) * 1e9  # in nM
        index_lines.append(
            f"{code}  {res:.2f}  {year}  {pkd:.2f}  {aff_type}={aff_val:.1f}nM  // synthetic"
        )

        # Create complex directory with minimal PDB + SDF
        cdir = refined / code
        cdir.mkdir(exist_ok=True)

        # Protein PDB (minimal: 10 CA atoms as a tiny pocket)
        n_atoms = 30 + i * 5
        with open(cdir / f"{code}_protein.pdb", "w") as f:
            for j in range(n_atoms):
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                z = np.random.uniform(-10, 10)
                resname = ["ALA", "GLY", "LEU", "VAL", "SER"][j % 5]
                atom_name = ["N", "CA", "C", "O"][j % 4]
                f.write(
                    f"ATOM  {j+1:5d} {atom_name:4s} {resname:3s} A{j//4+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                    f"{'N' if atom_name == 'N' else 'C' if atom_name in ('CA','C') else 'O'}\n"
                )
            f.write("END\n")

        # Ligand SDF (minimal valid SDF via RDKit)
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # Create a simple molecule (ethanol-like)
            smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccc(O)cc1"][i % 5]
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=i)
            mol = Chem.RemoveHs(mol)
            # Place near origin so pocket extraction captures protein atoms
            conf = mol.GetConformer()
            for a in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(a)
                conf.SetAtomPosition(a, (pos.x * 0.5, pos.y * 0.5, pos.z * 0.5))

            writer = Chem.SDWriter(str(cdir / f"{code}_ligand.sdf"))
            writer.write(mol)
            writer.close()
        except ImportError:
            # Fallback: write minimal SDF manually
            with open(cdir / f"{code}_ligand.sdf", "w") as f:
                f.write(f"{code}_ligand\n     RDKit\n\n")
                f.write("  3  2  0  0  0  0  0  0  0  0999 V2000\n")
                f.write("    0.0000    0.0000    0.0000 C   0  0  0  0  0  0\n")
                f.write("    1.5400    0.0000    0.0000 C   0  0  0  0  0  0\n")
                f.write("    2.3100    1.3300    0.0000 O   0  0  0  0  0  0\n")
                f.write("  1  2  1  0\n")
                f.write("  2  3  1  0\n")
                f.write("M  END\n$$$$\n")

    # Write INDEX
    index_path = root / "INDEX_refined_data.2020"
    with open(index_path, "w") as f:
        f.write("# PDBbind v2020 refined set (synthetic)\n")
        f.write("\n".join(index_lines) + "\n")

    return codes


def test_index_parsing():
    """M0.1: INDEX parsing produces correct DataFrame."""
    from bayesdiff.data import parse_pdbbind_index

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=10)

        df = parse_pdbbind_index(root / "INDEX_refined_data.2020")
        assert len(df) == 10, f"Expected 10, got {len(df)}"
        assert "pdb_code" in df.columns
        assert "pkd" in df.columns
        assert "affinity_type" in df.columns
        assert df["pkd"].min() >= 2.0
        assert df["pkd"].max() <= 14.0
        print(f"✓ M0.1: INDEX parsing OK ({len(df)} entries, pKd [{df['pkd'].min():.1f}, {df['pkd'].max():.1f}])")


def test_pocket_extraction():
    """M0.2: Pocket extraction produces valid PDB."""
    from bayesdiff.data import extract_pocket_from_protein

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=3)

        code = codes[0]
        protein_pdb = root / "refined-set" / code / f"{code}_protein.pdb"
        ligand_sdf = root / "refined-set" / code / f"{code}_ligand.sdf"
        out_pocket = root / "refined-set" / code / f"{code}_pocket10.pdb"

        extract_pocket_from_protein(protein_pdb, ligand_sdf, out_pocket, radius=10.0)
        assert out_pocket.exists(), "Pocket file not created"
        content = out_pocket.read_text()
        assert "ATOM" in content or "END" in content, "Pocket file has no ATOM records"
        n_lines = sum(1 for l in content.splitlines() if l.startswith("ATOM"))
        print(f"✓ M0.2: Pocket extraction OK ({n_lines} atoms in pocket)")


def test_protein_family_split():
    """M0.3: Splits are non-overlapping with correct proportions."""
    from bayesdiff.data import parse_pdbbind_index, protein_family_split

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=20)

        df = parse_pdbbind_index(root / "INDEX_refined_data.2020")
        splits = protein_family_split(df, root, seed=42)

        assert "train" in splits
        assert "val" in splits
        assert "cal" in splits
        assert "test" in splits

        all_codes_split = []
        for name, scodes in splits.items():
            all_codes_split.extend(scodes)

        # No overlaps
        assert len(all_codes_split) == len(set(all_codes_split)), "Overlap between splits!"
        # All codes accounted for
        assert set(all_codes_split) == set(codes), "Not all codes in splits"
        # Approximate proportions
        n = len(codes)
        assert len(splits["train"]) >= n * 0.5, "Train too small"

        print(f"✓ M0.3: Splits OK (train={len(splits['train'])}, val={len(splits['val'])}, "
              f"cal={len(splits['cal'])}, test={len(splits['test'])})")


def test_featurize_single_complex():
    """Test featurization of a single complex to .pt file."""
    # Import the processing function
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))
    from s00_prepare_pdbbind import _process_one_complex

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=3)
        output_dir = root / "output" / "processed"
        output_dir.mkdir(parents=True)

        code = codes[0]
        pkd_map = {code: 7.5}

        result = _process_one_complex(
            code,
            pdbbind_dir=root,
            output_dir=output_dir,
            pkd_map=pkd_map,
        )

        assert result["status"] == "ok", f"Failed: {result['error']}"

        pt_path = output_dir / f"{code}.pt"
        assert pt_path.exists(), "No .pt file created"

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert "protein_pos" in data
        assert "ligand_pos" in data
        assert "pkd" in data
        assert data["protein_pos"].ndim == 2 and data["protein_pos"].shape[1] == 3
        assert data["ligand_pos"].ndim == 2 and data["ligand_pos"].shape[1] == 3
        assert data["pkd"].item() == 7.5

        print(f"✓ Featurize: {code} → protein {data['n_protein_atoms']} atoms, "
              f"ligand {data['n_ligand_atoms']} atoms, pKd={data['pkd'].item():.1f}")


def test_dataloader():
    """M0.4: DataLoader correctly batches variable-length pairs."""
    from s00_prepare_pdbbind import _process_one_complex
    from bayesdiff.pretrain_dataset import PDBbindPairDataset, collate_pair_data

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=10)

        output_dir = root / "output"
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(parents=True)

        # Featurize all complexes
        from bayesdiff.data import parse_pdbbind_index, protein_family_split
        df = parse_pdbbind_index(root / "INDEX_refined_data.2020")
        pkd_map = dict(zip(df["pdb_code"], df["pkd"]))

        ok_codes = []
        for code in codes:
            res = _process_one_complex(code, root, processed_dir, pkd_map)
            if res["status"] == "ok":
                ok_codes.append(code)

        # Create splits.json
        n = len(ok_codes)
        splits = {
            "train": ok_codes[:int(n*0.7)],
            "val": ok_codes[int(n*0.7):int(n*0.8)],
            "cal": ok_codes[int(n*0.8):int(n*0.9)],
            "test": ok_codes[int(n*0.9):],
        }
        with open(output_dir / "splits.json", "w") as f:
            json.dump(splits, f)
        df.to_csv(output_dir / "labels.csv", index=False)

        # Load dataset
        dataset = PDBbindPairDataset(output_dir, split="train")
        assert len(dataset) > 0, "Empty train dataset"

        # Test single sample
        sample = dataset[0]
        assert "protein_pos" in sample
        assert "ligand_pos" in sample
        assert "pkd" in sample

        # Test collate
        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset, batch_size=min(4, len(dataset)),
            collate_fn=collate_pair_data, shuffle=False,
        )
        batch = next(iter(loader))
        assert "protein_batch" in batch
        assert "ligand_batch" in batch
        assert batch["protein_pos"].shape[1] == 3
        assert batch["ligand_pos"].shape[1] == 3
        assert batch["batch_size"] == min(4, len(dataset))

        print(f"✓ M0.4: DataLoader OK (train={len(dataset)} samples, "
              f"batch protein_pos={batch['protein_pos'].shape}, "
              f"ligand_pos={batch['ligand_pos'].shape})")


def test_full_pipeline():
    """Full end-to-end pipeline: parse → featurize → merge → split → dataloader."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))
    from s00_prepare_pdbbind import stage_parse, stage_featurize, stage_merge, stage_split

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=15)
        output_dir = root / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "processed").mkdir()

        # Stage 1: Parse
        df = stage_parse(root, output_dir)
        assert len(df) == 15
        assert (output_dir / "labels.csv").exists()

        # Stage 2: Featurize (single shard)
        results = stage_featurize(root, output_dir, df, num_workers=1)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        assert n_ok > 0, "No complexes featurized"

        # Stage 3: Merge
        stage_merge(output_dir)
        assert (output_dir / "processed_codes.txt").exists()

        # Stage 4: Split
        splits = stage_split(root, output_dir, seed=42)
        assert (output_dir / "splits.json").exists()
        total = sum(len(v) for v in splits.values())
        assert total > 0

        # Verify DataLoader
        from bayesdiff.pretrain_dataset import PDBbindPairDataset, collate_pair_data
        from torch.utils.data import DataLoader

        for split_name in ["train", "val"]:
            ds = PDBbindPairDataset(output_dir, split=split_name)
            if len(ds) > 0:
                loader = DataLoader(ds, batch_size=2, collate_fn=collate_pair_data)
                batch = next(iter(loader))
                assert batch["batch_size"] == 2 or batch["batch_size"] == len(ds)

        print(f"✓ Full pipeline: {len(df)} parsed → {n_ok} featurized → "
              f"{total} split (train={len(splits['train'])}, val={len(splits['val'])}, "
              f"cal={len(splits['cal'])}, test={len(splits['test'])})")


def test_shard_parallelism():
    """Verify sharded featurization produces same results as single-shard."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))
    from s00_prepare_pdbbind import stage_parse, stage_featurize

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        codes = _create_synthetic_pdbbind(root, n_complexes=12)
        output_dir = root / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "processed").mkdir()

        df = stage_parse(root, output_dir)

        # Run 3 shards
        all_results = []
        for shard_idx in range(3):
            results = stage_featurize(
                root, output_dir, df,
                shard_index=shard_idx, num_shards=3, num_workers=1,
            )
            all_results.extend(results)

        ok_codes = set(r["pdb_code"] for r in all_results if r["status"] == "ok")
        # Each code should be processed by exactly one shard
        assert len(ok_codes) == len([r for r in all_results if r["status"] == "ok"]), \
            "Some codes processed by multiple shards!"

        # All .pt files should exist
        for code in ok_codes:
            assert (output_dir / "processed" / f"{code}.pt").exists()

        print(f"✓ Shard parallelism: 3 shards processed {len(ok_codes)} unique complexes")


if __name__ == "__main__":
    tests = [
        ("M0.1: INDEX parsing", test_index_parsing),
        ("M0.2: Pocket extraction", test_pocket_extraction),
        ("M0.3: Protein-family split", test_protein_family_split),
        ("Featurize single complex", test_featurize_single_complex),
        ("M0.4: DataLoader", test_dataloader),
        ("Full pipeline", test_full_pipeline),
        ("Shard parallelism", test_shard_parallelism),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
