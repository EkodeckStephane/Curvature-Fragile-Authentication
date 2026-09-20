from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "build_public_pilot_manifest.py"
    spec = importlib.util.spec_from_file_location("build_public_pilot_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_public_pilot_manifest"] = module
    spec.loader.exec_module(module)
    return module


def base_row(dataset: str, status: str) -> dict[str, str]:
    return {
        "subcorpus": "ms_coco_val2017" if "COCO" in dataset else "dtd_textures",
        "local_path": "ms_coco_val2017/abc.png" if "COCO" in dataset else "dtd_textures/def.png",
        "local_sha256": "a" * 64,
        "width": "512",
        "height": "384",
        "dataset": dataset,
        "source_id": r"Z:\private\processed.jpg",
        "source_path": r"Z:\private\processed.jpg",
        "original_path": r"Z:\private\original.jpg",
        "source_split": "test_final",
        "exact_id": "private-derived-id",
        "source_dataset_url": "https://cocodataset.org/" if "COCO" in dataset else "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        "source_version": "val2017" if "COCO" in dataset else "dtd-r1.0.1",
        "source_citation": "citation",
        "source_reuse_statement": "" if "COCO" in dataset else "research purposes",
        "source_image_id": "123" if "COCO" in dataset else "grid/grid_0001.jpg",
        "source_file_name": "000000000123.jpg" if "COCO" in dataset else "",
        "license_id": "4" if "COCO" in dataset else "",
        "license_name": "Attribution License" if "COCO" in dataset else "",
        "license_url": "http://creativecommons.org/licenses/by/2.0/" if "COCO" in dataset else "",
        "coco_url": "http://images.cocodataset.org/val2017/000000000123.jpg" if "COCO" in dataset else "",
        "flickr_url": r"Z:\private\should_not_escape",
        "provenance_status": status,
    }


class BuildPublicPilotManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_script()

    def test_filters_public_datasets_and_removes_private_fields(self) -> None:
        rows = [
            base_row("MS-COCO val2017", "identity-and-license-pinned"),
            base_row("DTD textures", "identity-citation-and-reuse-statement-pinned"),
            base_row("BOSSbase 1.01", "identity-pinned; license-pending"),
        ]

        public_rows = self.builder.build_public_rows(rows)

        self.assertEqual(len(public_rows), 2)
        for row in public_rows:
            self.assertNotIn("source_path", row)
            self.assertNotIn("original_path", row)
            self.assertNotIn("flickr_url", row)
            self.assertFalse(any(":\\private" in value for value in row.values()))

    def test_summary_declares_no_pixels_or_private_paths(self) -> None:
        public_rows = self.builder.build_public_rows(
            [base_row("MS-COCO val2017", "identity-and-license-pinned")]
        )

        payload = self.builder.summary(public_rows, source_count=3)

        self.assertEqual(payload["rowCount"], 1)
        self.assertFalse(payload["pixelDataIncluded"])
        self.assertFalse(payload["privatePathsIncluded"])
        self.assertIn("MS-COCO val2017", payload["includedDatasets"])

    def test_rejects_private_path_in_public_record(self) -> None:
        with self.assertRaises(ValueError):
            self.builder.validate_public_record(
                {
                    "record_id": "x",
                    "dataset": "MS-COCO val2017",
                    "bad": r"Z:\private\user\private.jpg",
                }
            )


if __name__ == "__main__":
    unittest.main()
