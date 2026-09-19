from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.blind_v2 import master_key_from_seed
from src.blind_v2_image import (
    calibrate_delta_embed,
    calibrate_tau_from_clean_images,
    embed_image,
    image_blocks,
    psnr,
    verify_image,
)


ROOT = Path(__file__).resolve().parents[1]


def load_gate_script():
    path = ROOT / "scripts" / "run_blind_v2_synthetic_image_gate.py"
    spec = importlib.util.spec_from_file_location("run_blind_v2_synthetic_image_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_blind_v2_synthetic_image_gate"] = module
    spec.loader.exec_module(module)
    return module


class BlindV2ImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate_script()
        self.images = self.gate.synthetic_images(123, 2, 32, 32)
        self.master = master_key_from_seed(456)

    def test_image_blocks_cover_grid(self) -> None:
        blocks = list(image_blocks(self.images[0]))
        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[0][2].shape, (16, 16))

    def test_embed_and_verify_clean_image(self) -> None:
        embedded = embed_image(self.images[0], self.master, delta_embed=4.0)
        verified = verify_image(embedded.watermarked, self.master, delta_embed=4.0, tau=0.0)
        self.assertGreater(embedded.psnr_db, 40.0)
        self.assertEqual(verified.clean_bit_error_rate, 0.0)
        self.assertEqual(int(np.sum(verified.tamper_blocks)), 0)
        self.assertEqual(verified.tamper_pixels.shape, embedded.watermarked.shape)

    def test_delta_and_tau_calibration_accept_clean_images(self) -> None:
        delta, records = calibrate_delta_embed(
            self.images,
            self.master,
            candidates=[2.0, 4.0],
            min_psnr_db=40.0,
            max_clean_bit_error_rate=0.01,
        )
        self.assertEqual(delta, 2.0)
        self.assertEqual(len(records), 2)
        watermarked = [embed_image(image, self.master, delta).watermarked for image in self.images]
        tau, fpr, scores = calibrate_tau_from_clean_images(watermarked, self.master, delta)
        self.assertEqual(tau, 0.0)
        self.assertEqual(fpr, 0.0)
        self.assertTrue(np.all(scores == 0.0))

    def test_psnr_handles_identity(self) -> None:
        self.assertEqual(psnr(self.images[0], self.images[0]), float("inf"))

    def test_gate_script_writes_manifest_without_secret_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            summary = self.gate.run(
                ROOT / "configs" / "blind_v2_synthetic_image_gate.json",
                output,
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(summary["allAccepted"])
        self.assertTrue(written["allAccepted"])
        self.assertFalse(written["masterKeyStored"])
        self.assertIn("masterKeyId", written)
        self.assertNotIn("masterKeyValue", written)
        self.assertNotIn("masterKeyHex", written)
        self.assertNotIn("secretKey", written)
        self.assertEqual(len(written["masterKeyId"]), 16)


if __name__ == "__main__":
    unittest.main()
