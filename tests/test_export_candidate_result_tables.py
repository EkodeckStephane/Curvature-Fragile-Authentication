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
    path = ROOT / "scripts" / "export_candidate_result_tables.py"
    spec = importlib.util.spec_from_file_location("export_candidate_result_tables", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_candidate_result_tables"] = module
    spec.loader.exec_module(module)
    return module


class ExportCandidateResultTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_script()

    def test_exports_article_ready_csv_tables(self) -> None:
        source = (
            ROOT
            / "results"
            / "blind_v2_sensitive_bands_candidate_v2"
            / "summary.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = self.tool.export_tables(source, output_dir)

            with Path(paths["attack_f1"]).open(encoding="utf-8") as handle:
                attack_rows = list(csv.DictReader(handle))
            with Path(paths["clean_metrics"]).open(encoding="utf-8") as handle:
                clean_rows = list(csv.DictReader(handle))
            with Path(paths["paired_attack_f1_deltas"]).open(encoding="utf-8") as handle:
                paired_rows = list(csv.DictReader(handle))

        self.assertEqual(len(attack_rows), 5)
        self.assertEqual(len(clean_rows), 5)
        self.assertEqual(len(paired_rows), 16)
        fisher = next(row for row in attack_rows if row["basisMode"] == "fisher")
        self.assertEqual(fisher["center_mean"], "0.973333")
        self.assertEqual(fisher["inter_block_substitution"], "0.953333")

    def test_rejects_unexpected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"schema": "other"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                self.tool.load_summary(path)


if __name__ == "__main__":
    unittest.main()
