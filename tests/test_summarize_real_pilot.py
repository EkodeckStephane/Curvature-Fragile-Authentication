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
    path = ROOT / "scripts" / "summarize_real_pilot.py"
    spec = importlib.util.spec_from_file_location("summarize_real_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarize_real_pilot"] = module
    spec.loader.exec_module(module)
    return module


class SummarizeRealPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summarizer = load_script()

    def test_summarize_writes_csv_and_markdown_without_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_dir = root / "summary"
            manifest_path.write_text(json.dumps(fake_manifest()), encoding="utf-8")

            paths = self.summarizer.summarize(manifest_path, output_dir)
            with paths["modes"].open(encoding="utf-8") as handle:
                modes = list(csv.DictReader(handle))
            with paths["attacks"].open(encoding="utf-8") as handle:
                attacks = list(csv.DictReader(handle))
            with paths["timings"].open(encoding="utf-8") as handle:
                timings = list(csv.DictReader(handle))
            with paths["clean_bits"].open(encoding="utf-8") as handle:
                clean_bits = list(csv.DictReader(handle))
            summary = paths["summary"].read_text(encoding="utf-8")

        self.assertEqual(modes[0]["basisMode"], "fisher")
        self.assertEqual(attacks[0]["attack"], "center_mean")
        self.assertEqual(timings[0]["phase"], "delta_calibration")
        self.assertEqual(clean_bits[0]["bitIndex"], "0")
        self.assertEqual(clean_bits[0]["mismatchCount"], "1")
        self.assertIn("Promotion ready: `False`", summary)
        self.assertIn("Clean bit reliability", summary)
        self.assertNotIn("masterKeyValue", summary)
        self.assertNotIn("secretKey", summary)


def fake_manifest():
    return {
        "schema": "blind-v2-real-image-scratch-pilot/v1",
        "pilotMode": "engineering_scratch_separated_calibration_evaluation",
        "promotionReady": False,
        "datasetRootName": "fake",
        "dataCopied": False,
        "masterKeyStored": False,
        "calibrationImageCount": 1,
        "evaluationImageCount": 1,
        "selectedDeltaEmbed": 2.0,
        "targetFalsePositiveRate": 0.01,
        "timingSeconds": {
            "delta_calibration": 1.2,
            "fisher.verify_clean_evaluation": 0.4,
        },
        "baselines": [
            {
                "basisMode": "fisher",
                "tau": 0.0,
                "validationFalsePositiveRate": 0.005,
                "cleanFlaggedBlockCount": 2,
                "cleanBitMismatchCounts": [1, 0, 0, 0, 0, 0, 0, 0],
                "cleanBitTotalCount": 10,
                "cleanBitErrorRates": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "meanCleanPsnrDb": 72.0,
                "minCleanPsnrDb": 71.0,
                "maxCleanBitErrorRate": 0.001,
                "attacks": [
                    {
                        "attack": "center_mean",
                        "meanBlockF1": 0.7,
                        "meanBlockIoU": 0.55,
                        "meanBlockFpr": 0.01,
                        "meanBlockFnr": 0.0,
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
