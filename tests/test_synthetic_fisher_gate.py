from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_gate_script():
    path = ROOT / "scripts" / "run_synthetic_fisher_gate.py"
    spec = importlib.util.spec_from_file_location("run_synthetic_fisher_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_synthetic_fisher_gate"] = module
    spec.loader.exec_module(module)
    return module


class SyntheticFisherGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = load_gate_script()

    def test_run_writes_accepted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            summary = self.gate.run(
                ROOT / "configs" / "synthetic_fisher_gate.json",
                output,
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(summary["allAccepted"])
        self.assertEqual(summary["caseCount"], 22)
        self.assertEqual(written["acceptedCaseCount"], written["caseCount"])
        self.assertEqual(written["schema"], "synthetic-fisher-gate/v1")


if __name__ == "__main__":
    unittest.main()
