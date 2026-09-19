from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "audit_image_corpus.py"
    spec = importlib.util.spec_from_file_location("audit_image_corpus", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_image_corpus"] = module
    spec.loader.exec_module(module)
    return module


class AuditImageCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit_script = load_script()

    def test_audit_counts_dimensions_modes_and_optional_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            Image.new("L", (8, 9), 12).save(root / "first" / "a.png")
            Image.new("RGB", (5, 6), (1, 2, 3)).save(root / "second" / "b.jpg")
            summary = self.audit_script.audit(root, hash_files=True)

        self.assertEqual(summary["schema"], "image-corpus-audit/v1")
        self.assertEqual(summary["totalImageCount"], 2)
        self.assertEqual(summary["subcorpora"]["first"]["dimensions"], {"8x9": 1})
        self.assertEqual(summary["subcorpora"]["first"]["modes"], {"L": 1})
        self.assertIn("sha256", summary["subcorpora"]["second"]["samples"][0])
        self.assertFalse(summary["dataCopied"])

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            root.mkdir()
            (root / "sample").mkdir()
            Image.new("L", (4, 4), 0).save(root / "sample" / "x.png")
            output = Path(directory) / "audit.json"
            summary = self.audit_script.audit(root)
            output.write_text(json.dumps(summary), encoding="utf-8")
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(written["totalImageCount"], 1)


if __name__ == "__main__":
    unittest.main()
