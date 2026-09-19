from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LiteratureSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.search = load_script("s2_literature_search")
        cls.shortlist = load_script("s2_build_shortlist")

    def test_safe_id_removes_unsafe_characters(self) -> None:
        self.assertEqual(self.search.safe_id("TIFS: Fisher / Rao"), "TIFS_Fisher_Rao")

    def test_rate_limiter_enforces_interval(self) -> None:
        limiter = self.search.RateLimiter(interval=1.10)
        with mock.patch.object(
            self.search.time, "monotonic", side_effect=[10.0, 10.0, 10.4, 11.1]
        ), mock.patch.object(self.search.time, "sleep") as sleep:
            limiter.wait()
            limiter.wait()
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.7, places=7)

    def test_load_queries_rejects_incomplete_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            path.write_text(json.dumps([{"id": "missing-query"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.search.load_queries(path)

    def test_merge_deduplicates_and_preserves_query_provenance(self) -> None:
        paper = {
            "paperId": "p1",
            "title": "Fragile watermarking for image authentication",
            "year": 2024,
            "venue": "Example",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "20260919T120000Z_fragile.json",
                "20260919T120001Z_tifs.json",
            ):
                (root / name).write_text(
                    json.dumps({"data": [paper]}), encoding="utf-8"
                )
            records, sources = self.shortlist.merge(root)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(sources), 2)
        self.assertEqual(records["p1"]["discoveredBy"], ["fragile", "tifs"])
        self.assertGreater(self.shortlist.relevance(records["p1"]), 0)

    def test_tifs_venue_receives_declared_boost(self) -> None:
        base = {"paperId": "p1", "title": "Unrelated title", "year": 2010}
        generic = {**base, "venue": "Example Journal"}
        tifs = {
            **base,
            "venue": "IEEE Transactions on Information Forensics and Security",
        }
        self.assertEqual(
            self.shortlist.relevance(tifs) - self.shortlist.relevance(generic), 20
        )


class LiteratureInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.indexer = load_script("index_literature")

    def test_inventory_is_recursive_deterministic_and_hashes_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            first = root / "A.pdf"
            second = root / "nested" / "b.PDF"
            first.write_bytes(b"paper-a")
            second.write_bytes(b"paper-b")
            rows = self.indexer.inventory(root)
            expected_hash = self.indexer.sha256_file(first)
        self.assertEqual([row["file"] for row in rows], ["A.pdf", "nested/b.PDF"])
        self.assertEqual(rows[0]["sha256"], expected_hash)
        self.assertEqual(rows[0]["identityStatus"], "unverified")


if __name__ == "__main__":
    unittest.main()
