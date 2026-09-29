from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "results" / "dataset_manifests"


class MulticorpusProvenanceTests(unittest.TestCase):
    def test_multicorpus_provenance_json_is_complete_and_sanitized(self) -> None:
        path = MANIFEST_DIR / "multicorpus_provenance_v1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(data)

        self.assertEqual(data["schema"], "cfa-multicorpus-provenance/v1")
        self.assertEqual(
            data["linkedAggregate"],
            "results/blind_v2_multicorpus_1000eval_v1/summary.json",
        )
        self.assertFalse(data["safety"]["pixelsStored"])
        self.assertFalse(data["safety"]["localPathsStored"])
        self.assertFalse(data["safety"]["perImageFileNamesStored"])
        self.assertFalse(data["safety"]["secretKeyMaterialStored"])
        self.assertEqual(data["splitPolicy"]["calibrationImagesPerSubcorpus"], 200)
        self.assertEqual(data["splitPolicy"]["evaluationImagesPerSubcorpus"], 200)

        subcorpora = {item["subcorpus"]: item for item in data["subcorpora"]}
        self.assertEqual(
            set(subcorpora),
            {
                "bossbase_1_01",
                "bows_2",
                "dtd_textures",
                "inria_holidays",
                "ms_coco_val2017",
            },
        )
        for item in subcorpora.values():
            self.assertTrue(item["sourceUrl"])
            self.assertTrue(item["sourceVersion"])
            self.assertTrue(item["citationKey"])
            self.assertTrue(item["citation"])
            self.assertTrue(item["reuseOrLicenseStatus"])
            self.assertEqual(item["selectedImages"]["calibration"], 200)
            self.assertEqual(item["selectedImages"]["evaluation"], 200)
            self.assertIn("not redistributed", item["redistributionPolicy"])

        forbidden = [
            "C:\\Users",
            "Documents\\Articles",
            "masterKeyValue",
            "secretKeyValue",
            "source_path",
            "original_path",
        ]
        for marker in forbidden:
            self.assertNotIn(marker, serialized)

    def test_multicorpus_provenance_csv_matches_json_subcorpora(self) -> None:
        json_path = MANIFEST_DIR / "multicorpus_provenance_v1.json"
        csv_path = MANIFEST_DIR / "multicorpus_provenance_v1.csv"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), len(data["subcorpora"]))
        self.assertEqual(
            {row["subcorpus"] for row in rows},
            {item["subcorpus"] for item in data["subcorpora"]},
        )
        for row in rows:
            self.assertTrue(row["source_url"])
            self.assertTrue(row["source_version"])
            self.assertTrue(row["citation_key"])
            self.assertTrue(row["reuse_or_license_status"])
            self.assertEqual(row["calibration_images"], "200")
            self.assertEqual(row["evaluation_images"], "200")


if __name__ == "__main__":
    unittest.main()
