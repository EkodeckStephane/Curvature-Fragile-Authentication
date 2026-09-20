from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "enrich_multicorpus_metadata.py"
    spec = importlib.util.spec_from_file_location("enrich_multicorpus_metadata", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["enrich_multicorpus_metadata"] = module
    spec.loader.exec_module(module)
    return module


class EnrichMulticorpusMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.enricher = load_script()

    def test_enriches_coco_license_from_official_annotation_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "instances_val2017.json"
            annotation.write_text(
                json.dumps(
                    {
                        "licenses": [
                            {
                                "id": 4,
                                "name": "Attribution License",
                                "url": "http://creativecommons.org/licenses/by/2.0/",
                            }
                        ],
                        "images": [
                            {
                                "id": 123,
                                "file_name": "000000000123.jpg",
                                "license": 4,
                                "coco_url": "http://images.cocodataset.org/val2017/000000000123.jpg",
                                "flickr_url": "http://example.test/image",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = [
                {
                    "dataset": "MS-COCO val2017",
                    "matched": "true",
                    "original_path": r"C:\coco\val2017\000000000123.jpg",
                    "source_id": "processed/coco/x.jpg",
                }
            ]
            enriched = self.enricher.enrich(rows, coco_annotation_json=annotation)

        self.assertEqual(enriched[0]["provenance_status"], "identity-and-license-pinned")
        self.assertEqual(enriched[0]["source_image_id"], "123")
        self.assertEqual(enriched[0]["license_id"], "4")
        self.assertEqual(enriched[0]["license_name"], "Attribution License")
        self.assertEqual(
            enriched[0]["license_url"],
            "http://creativecommons.org/licenses/by/2.0/",
        )

    def test_enriches_dtd_with_official_source_and_citation(self) -> None:
        rows = [
            {
                "dataset": "DTD textures",
                "matched": "true",
                "original_path": r"C:\data\raw\dtd\dtd\images\grid\grid_0091.jpg",
                "source_id": "processed/dtd/abc.jpg",
            }
        ]

        enriched = self.enricher.enrich(rows)

        self.assertEqual(
            enriched[0]["provenance_status"],
            "identity-citation-and-reuse-statement-pinned",
        )
        self.assertEqual(enriched[0]["source_version"], "dtd-r1.0.1")
        self.assertEqual(enriched[0]["source_image_id"], "grid/grid_0091.jpg")
        self.assertIn("Describing Textures in the Wild", enriched[0]["source_citation"])


if __name__ == "__main__":
    unittest.main()
