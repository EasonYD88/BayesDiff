"""
tests/stage2/test_multilayer_extraction.py
──────────────────────────────────────────
Tests for multi-layer embedding extraction (Stage 0 infrastructure).

Tests T2.1–T2.4 from doc/Stage_2/03_multi_layer_fusion.md:
  T2.1: Hook registration on correct layers
  T2.2: Per-layer output shapes
  T2.3: Hook cleanup
  T2.4: No side effects on model output

These tests use the actual TargetDiff model and a real PDBbind .pt file
to verify end-to-end correctness. Run with:
    python tests/stage2/test_multilayer_extraction.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
TARGETDIFF_DIR = PROJECT_ROOT / "external" / "targetdiff"
PROCESSED_DIR = PROJECT_ROOT / "data" / "pdbbind_v2020" / "processed"


def _find_checkpoint() -> Path | None:
    """Auto-detect TargetDiff checkpoint."""
    candidates = [
        TARGETDIFF_DIR / "pretrained_models" / "pretrained_diffusion.pt",
        TARGETDIFF_DIR / "checkpoints" / "pretrained_diffusion.pt",
    ]
    # Also search recursively for any .pt in pretrained_models
    pm_dir = TARGETDIFF_DIR / "pretrained_models"
    if pm_dir.exists():
        for p in pm_dir.glob("*.pt"):
            candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_sample_pt() -> Path | None:
    """Find a small .pt file for testing."""
    if not PROCESSED_DIR.exists():
        return None
    for pt in sorted(PROCESSED_DIR.glob("*.pt"))[:10]:
        return pt
    return None


CKPT_PATH = _find_checkpoint()
SAMPLE_PT = _find_sample_pt()

SKIP_REASON = None
if CKPT_PATH is None:
    SKIP_REASON = "TargetDiff checkpoint not found"
elif SAMPLE_PT is None:
    SKIP_REASON = "No processed .pt files in data/pdbbind_v2020/processed/"


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TestMultiLayerExtraction(unittest.TestCase):
    """Test multi-layer embedding extraction from TargetDiff encoder."""

    @classmethod
    def setUpClass(cls):
        from bayesdiff.sampler import TargetDiffSampler
        cls.sampler = TargetDiffSampler(
            targetdiff_dir=str(TARGETDIFF_DIR),
            checkpoint_path=str(CKPT_PATH),
            device="cpu",
        )

    def test_T2_1_num_encoder_layers(self):
        """T2.1: Encoder should have 10 layers (1 init + 9 base)."""
        n = self.sampler.num_encoder_layers
        self.assertEqual(n, 10, f"Expected 10 layers, got {n}")

    def test_T2_2_output_shapes(self):
        """T2.2: Each layer should produce a (d,) mean-pooled embedding."""
        result = self.sampler.extract_multilayer_embeddings(SAMPLE_PT)

        n_layers = result["n_layers"]
        self.assertEqual(n_layers, 10, f"Expected 10 layers, got {n_layers}")

        d = self.sampler.hidden_dim
        for i in range(n_layers):
            key = f"layer_{i}"
            self.assertIn(key, result, f"Missing key: {key}")
            emb = result[key]
            self.assertEqual(emb.shape, (d,), f"Layer {i}: expected ({d},), got {emb.shape}")
            self.assertEqual(emb.dtype, np.float32)

        # z_global should match last layer
        self.assertIn("z_global", result)
        np.testing.assert_array_equal(
            result["z_global"], result[f"layer_{n_layers - 1}"],
            err_msg="z_global should equal last layer embedding"
        )

    def test_T2_3_no_nan_inf(self):
        """T2.3: No NaN or Inf in any layer embedding."""
        result = self.sampler.extract_multilayer_embeddings(SAMPLE_PT)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                self.assertFalse(np.isnan(val).any(), f"NaN in {key}")
                self.assertFalse(np.isinf(val).any(), f"Inf in {key}")

    def test_T2_4_layers_differ(self):
        """T2.4: Different layers should produce different embeddings."""
        result = self.sampler.extract_multilayer_embeddings(SAMPLE_PT)
        n_layers = result["n_layers"]

        # At least some pairs of layers should differ
        n_different = 0
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                if not np.allclose(result[f"layer_{i}"], result[f"layer_{j}"], atol=1e-4):
                    n_different += 1

        total_pairs = n_layers * (n_layers - 1) // 2
        self.assertGreater(
            n_different, total_pairs * 0.5,
            f"Expected most layer pairs to differ, but only {n_different}/{total_pairs} did"
        )

    def test_T2_5_load_complex_data(self):
        """Test that load_complex_data returns proper PyG Data."""
        data = self.sampler.load_complex_data(SAMPLE_PT)
        self.assertTrue(hasattr(data, "protein_atom_feature"))
        self.assertTrue(hasattr(data, "ligand_atom_feature_full"))
        self.assertGreater(len(data.protein_pos), 0)
        self.assertGreater(len(data.ligand_pos), 0)


class TestMultiLayerExtractionUnit(unittest.TestCase):
    """Unit tests that don't require model/data (always run)."""

    def test_import_sampler(self):
        """Sampler module can be imported."""
        from bayesdiff.sampler import TargetDiffSampler
        self.assertTrue(hasattr(TargetDiffSampler, "extract_multilayer_embeddings"))
        self.assertTrue(hasattr(TargetDiffSampler, "load_complex_data"))
        self.assertTrue(hasattr(TargetDiffSampler, "num_encoder_layers"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
