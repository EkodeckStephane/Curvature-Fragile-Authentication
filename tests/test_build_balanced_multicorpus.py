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
    path = ROOT / "scripts" / "build_balanced_multicorpus.py"
    spec = importlib.util.spec_from_file_location("build_balanced_multicorpus", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_balanced_multicorpus"] = module
    spec.loader.exec_module(module)
    return module


class BuildBalancedMulticorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_script()

    def test_builds_split_manifest_and_resizes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for index in range(4):
                Image.new("RGB", (80 + index, 40), (index, 2, 3)).save(
                    source / f"{index}.png"
                )
            output = root / "derived"
            manifest = root / "manifest.csv"
            corpus = self.tool.SourceCorpus(
                "sample",
                source,
                calibration_count=2,
                evaluation_count=2,
            )

            self.tool.build(
                [corpus],
                output_root=output,
                manifest_csv=manifest,
                seed="unit-test",
                max_long_edge=32,
                overwrite=False,
            )

            rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
            derived = sorted((output / "sample").glob("*.png"))

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(derived), 4)
        self.assertEqual(
            [row["split"] for row in rows].count("calibration"),
            2,
        )
        self.assertEqual([row["split"] for row in rows].count("evaluation"), 2)
        self.assertTrue(all(max(int(row["width"]), int(row["height"])) <= 32 for row in rows))
        self.assertTrue(all(row["max_long_edge"] == "32" for row in rows))


if __name__ == "__main__":
    unittest.main()
