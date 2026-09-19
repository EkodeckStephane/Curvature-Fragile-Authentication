from __future__ import annotations

import unittest

import numpy as np

from src.fisher_geometry import (
    fisher_from_gaussian_mean_covariance,
    generalized_eigen_decomposition,
    principal_generalized_direction,
    rayleigh_quotient,
    spectral_gap,
)


class FisherGeometryTests(unittest.TestCase):
    def test_gaussian_mean_fisher_is_inverse_covariance(self) -> None:
        covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
        fisher = fisher_from_gaussian_mean_covariance(covariance)
        np.testing.assert_allclose(fisher, np.linalg.inv(covariance))

    def test_rayleigh_quotient_is_scale_invariant(self) -> None:
        fisher = np.diag([6.0, 2.0])
        cost = np.diag([3.0, 1.0])
        direction = np.array([2.0, -1.0])
        self.assertAlmostEqual(
            rayleigh_quotient(fisher, cost, direction),
            rayleigh_quotient(fisher, cost, 7.0 * direction),
        )

    def test_principal_direction_matches_known_diagonal_problem(self) -> None:
        fisher = np.diag([8.0, 9.0])
        cost = np.diag([2.0, 9.0])
        value, vector = principal_generalized_direction(fisher, cost)
        self.assertAlmostEqual(value, 4.0)
        np.testing.assert_allclose(vector, np.array([1.0 / np.sqrt(2.0), 0.0]))

    def test_generalized_vectors_are_p_orthonormal_and_satisfy_residual(self) -> None:
        fisher = np.array([[5.0, 1.0], [1.0, 2.0]])
        cost = np.array([[2.0, 0.3], [0.3, 1.5]])
        result = generalized_eigen_decomposition(fisher, cost)

        np.testing.assert_allclose(
            result.vectors.T @ cost @ result.vectors,
            np.eye(2),
            atol=1e-10,
        )
        for index, value in enumerate(result.values):
            vector = result.vectors[:, index]
            np.testing.assert_allclose(
                fisher @ vector,
                value * (cost @ vector),
                atol=1e-10,
            )

    def test_spectral_gap_uses_largest_two_values(self) -> None:
        self.assertAlmostEqual(spectral_gap(np.array([9.0, 3.0, 1.0])), 3.0)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            generalized_eigen_decomposition(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.eye(2),
            )
        with self.assertRaises(ValueError):
            generalized_eigen_decomposition(
                np.eye(2),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
            )
        with self.assertRaises(ValueError):
            rayleigh_quotient(np.eye(2), np.eye(2), np.array([np.nan, 1.0]))


if __name__ == "__main__":
    unittest.main()
