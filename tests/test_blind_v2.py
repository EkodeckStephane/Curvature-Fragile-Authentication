from __future__ import annotations

import unittest

import numpy as np

from src.blind_v2 import (
    BLOCK_SIZE,
    PAYLOAD_BPP,
    RESERVED_COORDS,
    authentication_bits,
    block_tamper_map,
    build_block_model,
    bit_mismatch_score,
    calibrate_threshold,
    canonicalize_coefficients,
    delta_allocation_scales,
    derive_keys,
    dct2,
    dither,
    embed_block_coefficients,
    embed_block_coefficients_with_keys,
    expand_block_map,
    extract_block_score,
    extract_block_score_with_keys,
    fisher_cost_matrices,
    idct2,
    key_id,
    master_key_from_seed,
    qim_embed_scalar,
    qim_extract_bit,
    random_p_orthonormal_basis,
    serialize_features,
    to_luminance,
    trim_to_block_grid,
)


class BlindV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1234)
        self.block = rng.normal(128.0, 30.0, size=(BLOCK_SIZE, BLOCK_SIZE))
        self.coefficients = dct2(self.block)
        self.master = master_key_from_seed(20260919)
        self.keys = derive_keys(self.master)

    def test_luminance_and_trim_are_deterministic(self) -> None:
        rgb = np.zeros((17, 18, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 20
        rgb[..., 2] = 30
        luminance = to_luminance(rgb)
        self.assertAlmostEqual(float(luminance[0, 0]), 18.15)
        trimmed = trim_to_block_grid(rgb)
        self.assertEqual(trimmed.shape, (16, 16))

    def test_orthonormal_dct_round_trip(self) -> None:
        reconstructed = idct2(self.coefficients)
        np.testing.assert_allclose(reconstructed, self.block, atol=1e-10)

    def test_canonicalization_masks_only_reserved_coefficients(self) -> None:
        canonical = canonicalize_coefficients(self.coefficients)
        for row, col in RESERVED_COORDS:
            self.assertEqual(canonical[row, col], 0.0)
        self.assertEqual(canonical[0, 0], self.coefficients[0, 0])
        self.assertEqual(canonical[1, 1], self.coefficients[1, 1])

    def test_fisher_and_cost_are_positive_diagonal_matrices(self) -> None:
        canonical = canonicalize_coefficients(self.coefficients)
        fisher, cost, energies = fisher_cost_matrices(canonical)
        self.assertEqual(fisher.shape, (8, 8))
        self.assertEqual(cost.shape, (8, 8))
        self.assertTrue(np.all(np.diag(fisher) > 0.0))
        self.assertTrue(np.all(np.diag(cost) > 0.0))
        self.assertTrue(np.all(energies >= 1.0))
        np.testing.assert_allclose(fisher, np.diag(np.diag(fisher)))
        np.testing.assert_allclose(cost, np.diag(np.diag(cost)))

    def test_feature_serialization_and_hmac_bits_are_stable(self) -> None:
        model = build_block_model(self.coefficients)
        first = serialize_features(model.canonical, model.energies, (16, 16), (0, 0))
        second = serialize_features(model.canonical, model.energies, (16, 16), (0, 0))
        other = serialize_features(model.canonical, model.energies, (16, 16), (0, 1))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

        bits = authentication_bits(model.canonical, model.energies, (16, 16), (0, 0), self.keys.auth)
        self.assertEqual(bits.shape, (8,))
        self.assertTrue(set(bits.tolist()).issubset({0, 1}))

    def test_qim_round_trips_both_bits(self) -> None:
        delta = 6.0
        dither_value = dither(self.keys.embed, (2, 3), 4, delta)
        for bit in (0, 1):
            embedded = qim_embed_scalar(17.3, bit, delta, dither_value)
            self.assertEqual(qim_extract_bit(embedded, delta, dither_value), bit)

    def test_key_and_dither_caches_are_stable(self) -> None:
        self.assertIs(derive_keys(self.master), derive_keys(self.master))
        first = dither(self.keys.embed, (5, 7), 2, 4.0)
        second = dither(self.keys.embed, (5, 7), 2, 4.0)
        other_bit = dither(self.keys.embed, (5, 7), 3, 4.0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_bit)

    def test_block_embedding_extracts_clean_authentication_bits(self) -> None:
        embedded, embedded_bits = embed_block_coefficients(
            self.coefficients,
            image_shape=(16, 16),
            block_index=(0, 0),
            master_key=self.master,
            delta_embed=8.0,
        )
        score, extracted, expected = extract_score_for_test(
            embedded, self.master, delta_embed=8.0
        )
        self.assertEqual(score, 0.0)
        np.testing.assert_array_equal(extracted, expected)
        np.testing.assert_array_equal(embedded_bits, expected)

    def test_fisher_sqrt_delta_allocation_is_normalized_and_ordered(self) -> None:
        values = np.array([9.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        constant = delta_allocation_scales(values, "constant")
        allocated = delta_allocation_scales(values, "fisher_sqrt")

        np.testing.assert_allclose(constant, np.ones(8))
        self.assertAlmostEqual(float(np.mean(allocated)), 1.0)
        self.assertGreater(allocated[0], allocated[1])
        self.assertGreater(allocated[1], allocated[2])

    def test_fisher_sqrt_delta_allocation_round_trips_clean_block(self) -> None:
        embedded, embedded_bits = embed_block_coefficients(
            self.coefficients,
            image_shape=(16, 16),
            block_index=(0, 0),
            master_key=self.master,
            delta_embed=8.0,
            delta_mode="fisher_sqrt",
        )
        score, extracted, expected = extract_block_score(
            embedded,
            image_shape=(16, 16),
            block_index=(0, 0),
            master_key=self.master,
            delta_embed=8.0,
            delta_mode="fisher_sqrt",
        )

        self.assertEqual(score, 0.0)
        np.testing.assert_array_equal(extracted, expected)
        np.testing.assert_array_equal(embedded_bits, expected)

    def test_weighted_mismatch_score_uses_model_values(self) -> None:
        extracted = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        expected = np.zeros(8, dtype=np.uint8)
        weights = np.array([8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        hamming = bit_mismatch_score(extracted, expected, weights, "hamming")
        weighted = bit_mismatch_score(
            extracted, expected, weights, "fisher_weighted"
        )

        self.assertEqual(hamming, 0.125)
        self.assertAlmostEqual(weighted, 8.0 / 15.0)

    def test_top4_mismatch_score_uses_only_most_sensitive_bits(self) -> None:
        expected = np.zeros(8, dtype=np.uint8)
        weights = np.array([8.0, 7.0, 6.0, 5.0, 1.0, 1.0, 1.0, 1.0])
        low_mismatch = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)
        high_mismatch = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)

        self.assertEqual(
            bit_mismatch_score(low_mismatch, expected, weights, "fisher_top4"),
            0.0,
        )
        self.assertEqual(
            bit_mismatch_score(high_mismatch, expected, weights, "fisher_top4"),
            0.25,
        )

    def test_prederived_key_block_api_matches_public_api(self) -> None:
        public_coeffs, public_bits = embed_block_coefficients(
            self.coefficients,
            image_shape=(16, 16),
            block_index=(0, 0),
            master_key=self.master,
            delta_embed=8.0,
            basis_mode="random",
        )
        keyed_coeffs, keyed_bits = embed_block_coefficients_with_keys(
            self.coefficients,
            image_shape=(16, 16),
            block_index=(0, 0),
            keys=self.keys,
            delta_embed=8.0,
            basis_mode="random",
        )
        np.testing.assert_allclose(public_coeffs, keyed_coeffs)
        np.testing.assert_array_equal(public_bits, keyed_bits)

        public_score = extract_block_score(
            public_coeffs,
            image_shape=(16, 16),
            block_index=(0, 0),
            master_key=self.master,
            delta_embed=8.0,
            basis_mode="random",
        )
        keyed_score = extract_block_score_with_keys(
            keyed_coeffs,
            image_shape=(16, 16),
            block_index=(0, 0),
            keys=self.keys,
            delta_embed=8.0,
            basis_mode="random",
        )
        self.assertEqual(public_score[0], keyed_score[0])
        np.testing.assert_array_equal(public_score[1], keyed_score[1])
        np.testing.assert_array_equal(public_score[2], keyed_score[2])

    def test_basis_modes_are_p_orthonormal_or_declared(self) -> None:
        for mode in ("fisher", "smallest", "random", "fixed", "identity_cost"):
            model = build_block_model(
                self.coefficients,
                basis_mode=mode,
                keys=self.keys,
                block_index=(1, 2),
            )
            metric = np.eye(8) if mode == "identity_cost" else model.cost
            np.testing.assert_allclose(
                model.basis.T @ metric @ model.basis,
                np.eye(8),
                atol=1e-10,
            )

        first = build_block_model(
            self.coefficients, "random", self.keys, (1, 2)
        ).basis
        second = build_block_model(
            self.coefficients, "random", self.keys, (1, 2)
        ).basis
        np.testing.assert_allclose(first, second)

    def test_random_basis_cache_returns_independent_arrays(self) -> None:
        canonical = canonicalize_coefficients(self.coefficients)
        _, cost, _ = fisher_cost_matrices(canonical)
        first = random_p_orthonormal_basis(cost, self.keys.perm, (3, 4))
        first[0, 0] = 12345.0
        second = random_p_orthonormal_basis(cost, self.keys.perm, (3, 4))
        self.assertNotEqual(second[0, 0], 12345.0)
        np.testing.assert_allclose(second.T @ cost @ second, np.eye(8), atol=1e-10)

    def test_diagonal_fisher_basis_solves_generalized_problem(self) -> None:
        for mode in ("fisher", "smallest", "identity_cost"):
            model = build_block_model(self.coefficients, basis_mode=mode)
            residual = model.fisher @ model.basis - model.cost @ model.basis @ np.diag(
                model.values
            )
            np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-12)
            if mode == "smallest":
                self.assertTrue(np.all(np.diff(model.values) >= 0.0))
            else:
                self.assertTrue(np.all(np.diff(model.values) <= 0.0))

    def test_threshold_and_maps(self) -> None:
        tau, fpr = calibrate_threshold(np.array([0.0, 0.0, 0.125, 0.25]), alpha=0.25)
        self.assertEqual(tau, 0.125)
        self.assertEqual(fpr, 0.25)
        tampered = block_tamper_map(np.array([[0.0, 0.5]]), tau)
        np.testing.assert_array_equal(tampered, np.array([[False, True]]))
        self.assertEqual(expand_block_map(tampered).shape, (16, 32))

    def test_public_constants_match_v2_payload(self) -> None:
        self.assertEqual(len(RESERVED_COORDS), 8)
        self.assertAlmostEqual(PAYLOAD_BPP, 0.03125)
        self.assertEqual(len(key_id(self.master)), 16)


def extract_score_for_test(coefficients: np.ndarray, master: bytes, delta_embed: float):
    return extract_block_score(
        coefficients,
        image_shape=(16, 16),
        block_index=(0, 0),
        master_key=master,
        delta_embed=delta_embed,
    )


if __name__ == "__main__":
    unittest.main()
