from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "paired_compare_real_pilot.py"
    spec = importlib.util.spec_from_file_location("paired_compare_real_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["paired_compare_real_pilot"] = module
    spec.loader.exec_module(module)
    return module


class PairedCompareRealPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparer = load_script()

    def test_comparison_rows_pair_by_record_and_attack(self) -> None:
        rows = self.comparer.comparison_rows(fake_manifest())
        f1 = [
            row
            for row in rows
            if row["comparison"] == "attack:center_mean"
            and row["basisMode"] == "random"
            and row["metric"] == "f1"
        ][0]

        self.assertEqual(f1["n"], 2)
        self.assertAlmostEqual(f1["fisherMean"], 0.7)
        self.assertAlmostEqual(f1["baselineMean"], 0.6)
        self.assertAlmostEqual(f1["meanDeltaFisherMinusBaseline"], 0.1)
        self.assertEqual(f1["favorablePairs"], 2)
        self.assertEqual(f1["unfavorablePairs"], 0)

    def test_summarize_writes_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_dir = root / "paired"
            manifest_path.write_text(json.dumps(fake_manifest()), encoding="utf-8")

            paths = self.comparer.summarize(manifest_path, output_dir)
            with paths["paired"].open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            summary = paths["summary"].read_text(encoding="utf-8")

        self.assertTrue(rows)
        self.assertIn("Paired real-pilot comparison", summary)
        self.assertIn("center_mean", summary)
        self.assertNotIn("secretKey", summary)


def fake_manifest():
    return {
        "schema": "blind-v2-real-image-scratch-pilot/v1",
        "pilotMode": "manifest_restricted_separated_calibration_evaluation",
        "manifestName": "fake_manifest.csv",
        "evaluationImageCount": 2,
        "targetFalsePositiveRate": 0.01,
        "baselines": [
            fake_baseline("fisher", [0.8, 0.6], [72.0, 73.0]),
            fake_baseline("random", [0.7, 0.5], [71.0, 74.0]),
        ],
    }


def fake_baseline(mode: str, f1_values: list[float], psnr_values: list[float]):
    clean = []
    images = []
    for index, (f1, psnr) in enumerate(zip(f1_values, psnr_values)):
        record_id = f"record-{index}"
        clean.append(
            {
                "recordId": record_id,
                "relativePath": record_id,
                "psnrDb": psnr,
                "cleanBitErrorRate": 0.0,
                "flaggedBlockCount": index,
            }
        )
        images.append(
            {
                "recordId": record_id,
                "relativePath": record_id,
                "f1": f1,
                "iou": f1 / 2.0,
                "recall": 1.0,
                "precision": f1,
                "fpr": 0.01,
                "fnr": 0.0,
            }
        )
    return {
        "basisMode": mode,
        "clean": clean,
        "attacks": [
            {
                "attack": "center_mean",
                "images": images,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
