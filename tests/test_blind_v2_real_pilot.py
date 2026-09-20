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
    path = ROOT / "scripts" / "run_blind_v2_real_pilot.py"
    spec = importlib.util.spec_from_file_location("run_blind_v2_real_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_blind_v2_real_pilot"] = module
    spec.loader.exec_module(module)
    return module


class BlindV2RealPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_script()

    def test_collect_image_paths_is_deterministic_and_limited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b").mkdir()
            (root / "a").mkdir()
            Image.new("L", (64, 64), 1).save(root / "b" / "2.png")
            Image.new("L", (64, 64), 2).save(root / "b" / "1.png")
            Image.new("L", (64, 64), 3).save(root / "a" / "1.png")
            selected = self.runner.collect_image_paths(root, max_images_per_subcorpus=1)

        self.assertEqual(list(selected), ["a", "b"])
        self.assertEqual([path.name for path in selected["b"]], ["1.png"])

    def test_collect_image_splits_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample").mkdir()
            for index in range(3):
                Image.new("L", (64, 64), index).save(root / "sample" / f"{index}.png")
            calibration, evaluation = self.runner.collect_image_splits(
                root,
                calibration_images_per_subcorpus=1,
                evaluation_images_per_subcorpus=2,
            )

        self.assertEqual([path.name for path in calibration["sample"]], ["0.png"])
        self.assertEqual([path.name for path in evaluation["sample"]], ["1.png", "2.png"])

    def test_run_writes_private_scratch_manifest_without_key_or_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            output = Path(directory) / "pilot.json"
            (root / "sample").mkdir(parents=True)
            Image.new("L", (64, 64), 96).save(root / "sample" / "x.png")
            Image.new("L", (64, 64), 128).save(root / "sample" / "y.png")
            args = Namespace(
                root=root,
                output=output,
                max_images_per_subcorpus=2,
                calibration_images_per_subcorpus=1,
                evaluation_images_per_subcorpus=1,
                hash_files=True,
                master_key_seed=1234,
                delta_embed_candidates=[2.0, 4.0],
                min_psnr_db=30.0,
                max_clean_bit_error_rate=0.01,
                target_false_positive_rate=0.0,
                calibration_basis_mode="fisher",
                basis_modes=["fisher"],
                attacks=["center_mean"],
            )
            summary = self.runner.run(args)
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["schema"], "blind-v2-real-image-scratch-pilot/v1")
        self.assertEqual(
            summary["pilotMode"],
            "engineering_scratch_separated_calibration_evaluation",
        )
        self.assertFalse(summary["promotionReady"])
        self.assertFalse(summary["masterKeyStored"])
        self.assertFalse(summary["dataCopied"])
        self.assertEqual(written["calibrationImageCount"], 1)
        self.assertEqual(written["evaluationImageCount"], 1)
        self.assertEqual(written["imageCount"], 1)
        self.assertIn("sha256", written["baselines"][0]["clean"][0])
        self.assertEqual(
            len(written["baselines"][0]["calibrationCleanThresholdCurve"]), 9
        )
        self.assertEqual(
            len(written["baselines"][0]["evaluationCleanThresholdCurve"]), 9
        )
        self.assertEqual(len(written["baselines"][0]["attacks"][0]["thresholdCurve"]), 9)
        self.assertIn("timingSeconds", written)
        self.assertGreaterEqual(written["timingSeconds"]["load_corpus"], 0.0)
        self.assertGreaterEqual(written["timingSeconds"]["delta_calibration"], 0.0)
        self.assertGreaterEqual(
            written["timingSeconds"]["fisher.verify_clean_evaluation"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
