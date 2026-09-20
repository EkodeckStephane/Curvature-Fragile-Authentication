from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "map_multicorpus_identities.py"
    spec = importlib.util.spec_from_file_location("map_multicorpus_identities", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["map_multicorpus_identities"] = module
    spec.loader.exec_module(module)
    return module


class MapMulticorpusIdentitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapper = load_script()

    def test_maps_raw_bossbase_digest_to_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boss_root = root / "boss"
            derived_root = root / "derived"
            boss_root.mkdir()
            (derived_root / "bossbase_1_01").mkdir(parents=True)

            source = boss_root / "7.pgm"
            Image.new("L", (8, 8), 7).save(source)
            stem = self.mapper.derived_stem("BOSSbase 1.01", "7.pgm")
            Image.new("L", (8, 8), 7).save(
                derived_root / "bossbase_1_01" / f"{stem}.png"
            )

            rows = self.mapper.map_identities(
                derived_root=derived_root, bossbase_root=boss_root
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["matched"], "true")
        self.assertEqual(rows[0]["dataset"], "BOSSbase 1.01")
        self.assertEqual(rows[0]["source_id"], "7.pgm")

    def test_maps_dynacis_split_record_to_original_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dynacis = root / "dynacis"
            processed = dynacis / "processed"
            processed_coco = processed / "coco"
            derived_root = root / "derived"
            processed_coco.mkdir(parents=True)
            (derived_root / "ms_coco_val2017").mkdir(parents=True)

            processed_image = processed_coco / "abc.jpg"
            original_image = dynacis / "raw" / "coco" / "val2017" / "0000001.jpg"
            original_image.parent.mkdir(parents=True)
            Image.new("RGB", (5, 4), (1, 2, 3)).save(processed_image)
            Image.new("RGB", (5, 4), (1, 2, 3)).save(original_image)

            split_csv = processed / "coco_splits.csv"
            with split_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["original_path", "processed_path", "split", "exact_id"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "original_path": str(original_image),
                        "processed_path": str(processed_image),
                        "split": "test_final",
                        "exact_id": "abc-full",
                    }
                )

            identifier = self.mapper.source_id(processed_image, dynacis)
            stem = self.mapper.derived_stem("MS-COCO val2017", identifier)
            Image.new("L", (5, 4), 8).save(
                derived_root / "ms_coco_val2017" / f"{stem}.png"
            )

            rows = self.mapper.map_identities(
                derived_root=derived_root, dynacis_root=dynacis
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["matched"], "true")
        self.assertEqual(rows[0]["dataset"], "MS-COCO val2017")
        self.assertEqual(rows[0]["source_split"], "test_final")
        self.assertEqual(rows[0]["exact_id"], "abc-full")
        self.assertTrue(rows[0]["original_path"].endswith("0000001.jpg"))


if __name__ == "__main__":
    unittest.main()
