from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneralizedEigenResult:
    """Ordered solution of F v = lambda P v with P-normalized vectors."""

    values: np.ndarray
    vectors: np.ndarray

    @property
    def principal_value(self) -> float:
        return float(self.values[0])

    @property
    def principal_vector(self) -> np.ndarray:
        return self.vectors[:, 0].copy()


def fisher_from_gaussian_mean_covariance(covariance: np.ndarray) -> np.ndarray:
    """Return the Fisher matrix for Gaussian mean perturbations."""

    cov = _require_spd_matrix("covariance", covariance)
    return np.linalg.inv(cov)


def rayleigh_quotient(
    fisher: np.ndarray, cost: np.ndarray, direction: np.ndarray
) -> float:
    """Compute (u^T F u) / (u^T P u) for a nonzero direction."""

    f = _require_symmetric_matrix("fisher", fisher)
    p = _require_spd_matrix("cost", cost)
    u = np.asarray(direction, dtype=float)
    if u.ndim != 1:
        raise ValueError("direction must be a vector")
    if u.shape[0] != f.shape[0] or p.shape != f.shape:
        raise ValueError("fisher, cost, and direction dimensions must match")
    if not np.all(np.isfinite(u)):
        raise ValueError("direction must contain only finite values")
    denominator = float(u.T @ p @ u)
    if denominator <= 0.0:
        raise ValueError("direction must have positive cost norm")
    return float((u.T @ f @ u) / denominator)


def generalized_eigen_decomposition(
    fisher: np.ndarray, cost: np.ndarray
) -> GeneralizedEigenResult:
    """Solve F v = lambda P v for symmetric F and SPD P.

    Eigenvalues are sorted from largest to smallest. Columns of `vectors` are
    deterministic up to sign and normalized with `v.T @ P @ v == 1`.
    """

    f = _require_symmetric_matrix("fisher", fisher)
    p = _require_spd_matrix("cost", cost)
    if f.shape != p.shape:
        raise ValueError("fisher and cost must have the same shape")

    chol = np.linalg.cholesky(p)
    identity = np.eye(p.shape[0])
    inv_chol_t = np.linalg.solve(chol.T, identity)
    standard = np.linalg.solve(chol, f @ inv_chol_t)
    standard = 0.5 * (standard + standard.T)

    values, y_vectors = np.linalg.eigh(standard)
    order = np.argsort(values)[::-1]
    values = values[order]
    y_vectors = y_vectors[:, order]
    vectors = np.linalg.solve(chol.T, y_vectors)

    for index in range(vectors.shape[1]):
        norm = float(np.sqrt(vectors[:, index].T @ p @ vectors[:, index]))
        vectors[:, index] /= norm
        pivot = int(np.argmax(np.abs(vectors[:, index])))
        if vectors[pivot, index] < 0:
            vectors[:, index] *= -1.0

    return GeneralizedEigenResult(values=values, vectors=vectors)


def principal_generalized_direction(
    fisher: np.ndarray, cost: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the largest generalized eigenvalue and its P-normalized vector."""

    result = generalized_eigen_decomposition(fisher, cost)
    return result.principal_value, result.principal_vector


def spectral_gap(values: np.ndarray) -> float:
    """Return lambda_1 / lambda_2 for an ordered generalized spectrum."""

    spectrum = np.asarray(values, dtype=float)
    if spectrum.ndim != 1 or spectrum.size < 2:
        raise ValueError("at least two eigenvalues are required")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("eigenvalues must contain only finite values")
    if spectrum[1] <= 0.0:
        raise ValueError("the second eigenvalue must be positive")
    return float(spectrum[0] / spectrum[1])


def _require_symmetric_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return value


def _require_spd_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    value = _require_symmetric_matrix(name, matrix)
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return value
