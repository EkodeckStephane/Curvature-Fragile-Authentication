from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "run_lsb_hmac_baseline.py"
    spec = importlib.util.spec_from_file_location("run_lsb_hmac_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_lsb_hmac_baseline"] = module
    spec.loader.exec_module(module)
    return module


class LsbHmacBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_script()

    def test_clean_embedding_verifies_without_flags(self) -> None:
        key = b"unit-test-key"
        image = np.full((64, 64), 128, dtype=np.float64)
        embedded = self.tool.embed_lsb_hmac_image(image, key)
        verified = self.tool.verify_lsb_hmac_image(embedded["watermarked"], key, tau=0.0)

        self.assertEqual(int(np.sum(verified["tamperBlocks"])), 0)
        self.assertEqual(verified["cleanBitErrorRate"], 0.0)
        self.assertGreater(embedded["psnrDb"], 50.0)

    def test_run_writes_sanitized_scratch_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            output = Path(directory) / "lsb.json"
            (root / "sample").mkdir(parents=True)
            Image.new("L", (64, 64), 96).save(root / "sample" / "a.png")
            Image.new("L", (64, 64), 128).save(root / "sample" / "b.png")
            args = Namespace(
                root=root,
                output=output,
                calibration_images_per_subcorpus=1,
                evaluation_images_per_subcorpus=1,
                master_key_seed=1234,
                target_false_positive_rate=0.01,
                attacks=["center_mean"],
            )
            summary = self.tool.run(args)
            written = json.loads(output.read_text(encoding="utf-8"))

        serialized = json.dumps(written)
        self.assertEqual(summary["schema"], "external-lsb-hmac-baseline-scratch/v1")
        self.assertEqual(written["baselineId"], "block-lsb-hmac-v1")
        self.assertFalse(written["pixelsStored"])
        self.assertFalse(written["localPathsStored"])
        self.assertFalse(written["perImageFileNamesStored"])
        self.assertFalse(written["masterKeyStored"])
        self.assertEqual(written["evaluationImageCount"], 1)
        self.assertIn("attacksDetailed", written)
        self.assertNotIn(str(root), serialized)
        self.assertNotIn("masterKeyValue", serialized)


if __name__ == "__main__":
    unittest.main()
