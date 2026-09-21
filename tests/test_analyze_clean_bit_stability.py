from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "analyze_clean_bit_stability.py"
    spec = importlib.util.spec_from_file_location("analyze_clean_bit_stability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_clean_bit_stability"] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeCleanBitStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_script()

    def test_writes_clean_bit_stability_tables_without_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            output_json = Path(directory) / "stability.json"
            output_csv = Path(directory) / "stability.csv"
            (root / "sample").mkdir(parents=True)
            Image.new("L", (64, 64), 96).save(root / "sample" / "cal.png")
            Image.new("L", (64, 64), 128).save(root / "sample" / "eval.png")
            args = Namespace(
                root=root,
                manifest_csv=None,
                manifest_derived_root=None,
                dtd_root=None,
                coco_root=None,
                output_json=output_json,
                output_csv=output_csv,
                calibration_images_per_subcorpus=1,
                evaluation_images_per_subcorpus=1,
                hash_files=False,
                master_key_seed=1234,
                delta_embed_candidates=[2.0, 4.0],
                min_psnr_db=30.0,
                max_clean_bit_error_rate=0.01,
                auth_feature_step=8.0,
                auth_feature_mode="sensitive_bands",
                auth_feature_band_count=2,
                delta_mode="constant",
                calibration_basis_mode="fisher",
                basis_modes=["fisher", "random"],
            )
            manifest = self.tool.run(args)
            written = json.loads(output_json.read_text(encoding="utf-8"))
            csv_text = output_csv.read_text(encoding="utf-8")

        self.assertEqual(manifest["schema"], "clean-bit-stability-diagnostic/v1")
        self.assertFalse(written["promotionReady"])
        self.assertFalse(written["dataCopied"])
        self.assertFalse(written["masterKeyStored"])
        self.assertFalse(written["masterKeySeedStored"])
        self.assertEqual(written["basisModes"], ["fisher", "random"])
        self.assertEqual(len(written["rows"]), 32)
        self.assertIn("split,basisMode,subcorpus,bitIndex", csv_text)
        serialized = json.dumps(written)
        self.assertNotIn(str(root), serialized)
        self.assertNotIn("masterKeyValue", serialized)
        self.assertNotIn("secretKey", serialized)


if __name__ == "__main__":
    unittest.main()
