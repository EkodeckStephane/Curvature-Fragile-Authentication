from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "generate_candidate_result_figures.py"
    spec = importlib.util.spec_from_file_location("generate_candidate_result_figures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_candidate_result_figures"] = module
    spec.loader.exec_module(module)
    return module


class GenerateCandidateResultFiguresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_script()

    def test_generates_sanitized_svg_figures(self) -> None:
        input_dir = (
            ROOT / "results" / "blind_v2_sensitive_bands_candidate_v2" / "tables"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = self.tool.generate_figures(input_dir, output_dir)
            attack_svg = Path(paths["attack_f1"]).read_text(encoding="utf-8")
            clean_svg = Path(paths["clean_metrics"]).read_text(encoding="utf-8")

        self.assertIn("<svg", attack_svg)
        self.assertIn("Candidate v2: block-level F1 by attack", attack_svg)
        self.assertIn("0.973", attack_svg)
        self.assertIn("Candidate v2: clean-image behavior", clean_svg)
        serialized = attack_svg + clean_svg
        self.assertNotIn("C:\\Users", serialized)
        self.assertNotIn("Documents\\Articles", serialized)
        self.assertNotIn("masterKeyValue", serialized)
        self.assertNotIn("secretKey", serialized)


if __name__ == "__main__":
    unittest.main()
