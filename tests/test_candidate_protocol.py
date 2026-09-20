from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CandidateProtocolTests(unittest.TestCase):
    def test_sensitive_bands_candidate_config_is_predeclared(self) -> None:
        path = ROOT / "configs" / "blind_v2_sensitive_bands_candidate_v1.json"
        config = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(config["schema"], "cfa-candidate-protocol/v1")
        self.assertEqual(config["status"], "predeclared-candidate")
        self.assertEqual(
            config["protocolId"], "blind-v2-sensitive-bands-step24-alpha001-v1"
        )
        self.assertEqual(config["candidate"]["authFeatureStep"], 24.0)
        self.assertEqual(config["candidate"]["authFeatureMode"], "sensitive_bands")
        self.assertEqual(config["candidate"]["authFeatureBandCount"], 3)
        self.assertEqual(config["candidate"]["scoreMode"], "hamming")
        self.assertEqual(config["candidate"]["deltaMode"], "constant")
        self.assertEqual(config["candidate"]["targetFalsePositiveRate"], 0.01)
        self.assertEqual(config["split"]["calibrationImagesPerSubcorpus"], 10)
        self.assertEqual(config["split"]["evaluationImagesPerSubcorpus"], 10)
        self.assertIn("fisher", config["basisModes"])
        self.assertIn("fixed", config["basisModes"])
        self.assertIn("random", config["basisModes"])
        self.assertIn("smallest", config["basisModes"])

    def test_sensitive_bands_analysis_plan_has_no_machine_local_paths(self) -> None:
        path = ROOT / "results" / "analysis_plans" / "blind_v2_sensitive_bands_candidate_v1.md"
        text = path.read_text(encoding="utf-8")

        self.assertIn("Blind V2 sensitive-bands candidate protocol v1", text)
        self.assertIn("auth_feature_step = 24.0", text)
        self.assertNotIn("C:\\Users", text)
        self.assertNotIn("Documents\\Articles", text)
        self.assertNotIn("masterKeyValue", text)
        self.assertNotIn("secretKey", text)


if __name__ == "__main__":
    unittest.main()
