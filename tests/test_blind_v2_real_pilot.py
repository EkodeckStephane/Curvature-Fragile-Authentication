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

    def test_manifest_splits_resolve_derived_images_and_verify_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived_root = root / "derived"
            subcorpus = derived_root / "dtd_textures"
            subcorpus.mkdir(parents=True)
            first = subcorpus / "a.png"
            second = subcorpus / "b.png"
            Image.new("L", (64, 64), 64).save(first)
            Image.new("L", (64, 64), 128).save(second)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "record_id,dataset,subcorpus,source_split,source_dataset_url,source_version,source_citation,source_reuse_statement,source_image_id,source_file_name,width,height,derived_artifact_id,derived_sha256,license_id,license_name,license_url,coco_url,provenance_status,redistribution_policy",
                        f"dtd_textures:a.png,DTD textures,dtd_textures,test_final,https://example.test,dtd-r1.0.1,citation,research,grid/grid_0001.jpg,,64,64,a.png,{self.runner.sha256_file(first)},,,,,identity-citation-and-reuse-statement-pinned,policy",
                        f"dtd_textures:b.png,DTD textures,dtd_textures,test_final,https://example.test,dtd-r1.0.1,citation,research,grid/grid_0002.jpg,,64,64,b.png,{self.runner.sha256_file(second)},,,,,identity-citation-and-reuse-statement-pinned,policy",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            calibration, evaluation, pilot_mode = self.runner.load_manifest_image_splits(
                manifest,
                manifest_derived_root=derived_root,
                calibration_images_per_subcorpus=1,
                evaluation_images_per_subcorpus=1,
                hash_files=True,
            )

        self.assertEqual(pilot_mode, "manifest_restricted_separated_calibration_evaluation")
        self.assertEqual(calibration[0].record_id, "dtd_textures:a.png")
        self.assertEqual(evaluation[0].record_id, "dtd_textures:b.png")
        self.assertEqual(evaluation[0].relative_path, "dtd_textures:b.png")
        self.assertEqual(evaluation[0].source_image_id, "grid/grid_0002.jpg")
        self.assertIsNotNone(evaluation[0].sha256)

    def test_run_writes_private_scratch_manifest_without_key_or_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            output = Path(directory) / "pilot.json"
            (root / "sample").mkdir(parents=True)
            Image.new("L", (64, 64), 96).save(root / "sample" / "x.png")
            Image.new("L", (64, 64), 128).save(root / "sample" / "y.png")
            args = Namespace(
                root=root,
                manifest_csv=None,
                manifest_derived_root=None,
                dtd_root=None,
                coco_root=None,
                output=output,
                max_images_per_subcorpus=2,
                calibration_images_per_subcorpus=1,
                evaluation_images_per_subcorpus=1,
                hash_files=True,
                master_key_seed=1234,
                delta_embed_candidates=[2.0, 4.0],
                min_psnr_db=30.0,
                max_clean_bit_error_rate=0.01,
                auth_feature_step=8.0,
                auth_feature_mode="sensitive_bands",
                auth_feature_band_count=2,
                target_false_positive_rate=0.0,
                score_mode="fisher_reliability",
                delta_mode="constant",
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
        self.assertEqual(written["authFeatureStep"], 8.0)
        self.assertEqual(written["authFeatureMode"], "sensitive_bands")
        self.assertEqual(written["authFeatureBandCount"], 2)
        self.assertIn("sha256", written["baselines"][0]["clean"][0])
        self.assertEqual(
            len(written["baselines"][0]["calibrationCleanThresholdCurve"]), 9
        )
        self.assertEqual(
            len(written["baselines"][0]["calibrationCleanBitMismatchCounts"]), 8
        )
        self.assertEqual(
            len(written["baselines"][0]["calibrationCleanBitErrorRates"]), 8
        )
        self.assertGreater(written["baselines"][0]["calibrationCleanBitTotalCount"], 0)
        self.assertEqual(len(written["baselines"][0]["cleanBitMismatchCounts"]), 8)
        self.assertEqual(len(written["baselines"][0]["cleanBitErrorRates"]), 8)
        self.assertGreater(written["baselines"][0]["cleanBitTotalCount"], 0)
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
