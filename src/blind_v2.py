from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

BasisMode = Literal["fisher", "smallest", "random", "fixed", "identity_cost"]

PROTOCOL_ID = b"CFA-blind-v2"
BLOCK_SIZE = 16
RESERVED_COORDS: tuple[tuple[int, int], ...] = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
)
PAYLOAD_BITS_PER_BLOCK = len(RESERVED_COORDS)
PAYLOAD_BPP = PAYLOAD_BITS_PER_BLOCK / float(BLOCK_SIZE * BLOCK_SIZE)
MODEL_RADIAL_MIN = 2
MODEL_RADIAL_MAX = 8
EPSILON = 1.0
FREQUENCY_COST_SLOPE = 0.10
DELTA_FEATURE = 4.0


@dataclass(frozen=True)
class V2Keys:
    auth: bytes
    embed: bytes
    perm: bytes


@dataclass(frozen=True)
class BlockModel:
    canonical: np.ndarray
    energies: np.ndarray
    fisher: np.ndarray
    cost: np.ndarray
    values: np.ndarray
    basis: np.ndarray


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Return a float64 luminance image from grayscale or RGB input."""

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.float64, copy=False)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise ValueError("image must be grayscale or RGB-like")


def trim_to_block_grid(image: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Trim bottom/right borders so dimensions are multiples of block_size."""

    luminance = to_luminance(image)
    height = luminance.shape[0] - luminance.shape[0] % block_size
    width = luminance.shape[1] - luminance.shape[1] % block_size
    if height == 0 or width == 0:
        raise ValueError("image is smaller than one block")
    return luminance[:height, :width].copy()


@lru_cache(maxsize=None)
def dct_matrix(size: int = BLOCK_SIZE) -> np.ndarray:
    """Return the orthonormal DCT-II transform matrix."""

    if size <= 0:
        raise ValueError("size must be positive")
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    matrix = np.sqrt(2.0 / size) * np.cos(np.pi * (cols + 0.5) * rows / size)
    matrix[0, :] = np.sqrt(1.0 / size)
    return matrix


def dct2(block: np.ndarray) -> np.ndarray:
    """Apply an orthonormal 2D DCT-II to a square block."""

    value = np.asarray(block, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("block must be square")
    transform = dct_matrix(value.shape[0])
    return transform @ value @ transform.T


def idct2(coefficients: np.ndarray) -> np.ndarray:
    """Invert dct2 for an orthonormal DCT-II matrix."""

    value = np.asarray(coefficients, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("coefficients must be square")
    transform = dct_matrix(value.shape[0])
    return transform.T @ value @ transform


@lru_cache(maxsize=None)
def model_coordinates(size: int = BLOCK_SIZE) -> tuple[tuple[int, int], ...]:
    reserved = set(RESERVED_COORDS)
    coords: list[tuple[int, int]] = []
    for row in range(size):
        for col in range(size):
            radial = row + col
            coord = (row, col)
            if coord == (0, 0) or coord in reserved:
                continue
            if MODEL_RADIAL_MIN <= radial <= MODEL_RADIAL_MAX:
                coords.append(coord)
    return tuple(coords)


def canonicalize_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Return the DCT block with reserved coefficients neutralized."""

    value = np.asarray(coefficients, dtype=np.float64)
    if value.shape != (BLOCK_SIZE, BLOCK_SIZE):
        raise ValueError("coefficients must be a 16x16 DCT block")
    canonical = value.copy()
    for row, col in RESERVED_COORDS:
        canonical[row, col] = 0.0
    return canonical


def reserved_vector(coefficients: np.ndarray) -> np.ndarray:
    value = np.asarray(coefficients, dtype=np.float64)
    return np.array([value[row, col] for row, col in RESERVED_COORDS], dtype=np.float64)


def set_reserved_vector(coefficients: np.ndarray, vector: np.ndarray) -> np.ndarray:
    value = np.asarray(coefficients, dtype=np.float64).copy()
    coeffs = np.asarray(vector, dtype=np.float64)
    if coeffs.shape != (PAYLOAD_BITS_PER_BLOCK,):
        raise ValueError("reserved vector must contain eight coefficients")
    for index, (row, col) in enumerate(RESERVED_COORDS):
        value[row, col] = coeffs[index]
    return value


def band_energies(canonical: np.ndarray) -> np.ndarray:
    """Compute one local energy per reserved coefficient radial band."""

    value = np.asarray(canonical, dtype=np.float64)
    coords = model_coordinates(value.shape[0])
    model_values = np.array([value[row, col] ** 2 for row, col in coords])
    fallback = float(np.median(model_values)) if model_values.size else 0.0

    energies = []
    for row, col in RESERVED_COORDS:
        radial = row + col
        band = [value[i, j] ** 2 for i, j in coords if i + j == radial]
        energy = float(np.mean(band)) if band else fallback
        energies.append(EPSILON + energy)
    return np.asarray(energies, dtype=np.float64)


def fisher_cost_matrices(canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return diagonal F, diagonal P, and the energies used for F."""

    energies = band_energies(canonical)
    fisher = np.diag(1.0 / energies)
    return fisher, np.diag(reserved_frequency_cost_diagonal()), energies


@lru_cache(maxsize=1)
def reserved_frequency_cost_diagonal() -> tuple[float, ...]:
    return tuple(
        1.0 + FREQUENCY_COST_SLOPE * float(row * row + col * col)
        for row, col in RESERVED_COORDS
    )


def derive_keys(master_key: bytes) -> V2Keys:
    if len(master_key) < 32:
        raise ValueError("master_key must contain at least 32 bytes")
    return V2Keys(
        auth=hmac.new(master_key, b"auth-v2", hashlib.sha256).digest(),
        embed=hmac.new(master_key, b"embed-v2", hashlib.sha256).digest(),
        perm=hmac.new(master_key, b"perm-v2", hashlib.sha256).digest(),
    )


def master_key_from_seed(seed: int) -> bytes:
    """Derive a reproducible 256-bit research key from an integer seed."""

    return hashlib.sha256(f"CFA-v2-seed:{seed}".encode("ascii")).digest()


def key_id(master_key: bytes) -> str:
    return hashlib.sha256(master_key).hexdigest()[:16]


def build_block_model(
    coefficients: np.ndarray,
    basis_mode: BasisMode = "fisher",
    keys: V2Keys | None = None,
    block_index: tuple[int, int] = (0, 0),
) -> BlockModel:
    canonical = canonicalize_coefficients(coefficients)
    fisher, cost, energies = fisher_cost_matrices(canonical)
    active_cost = np.eye(PAYLOAD_BITS_PER_BLOCK) if basis_mode == "identity_cost" else cost

    if basis_mode in ("fisher", "identity_cost"):
        values, basis = diagonal_generalized_basis(fisher, active_cost, descending=True)
    elif basis_mode == "smallest":
        values, basis = diagonal_generalized_basis(fisher, active_cost, descending=False)
    elif basis_mode == "fixed":
        basis = fixed_basis(cost)
        values = np.diag(fisher)
    elif basis_mode == "random":
        if keys is None:
            raise ValueError("random basis requires derived keys")
        basis = random_p_orthonormal_basis(cost, keys.perm, block_index)
        values = np.array([float(v.T @ fisher @ v) for v in basis.T])
    else:
        raise ValueError(f"unknown basis mode: {basis_mode}")

    return BlockModel(
        canonical=canonical,
        energies=energies,
        fisher=fisher,
        cost=active_cost,
        values=values,
        basis=basis,
    )


def diagonal_generalized_basis(
    fisher: np.ndarray,
    cost: np.ndarray,
    descending: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve diagonal `F v = lambda P v` with `P`-orthonormal vectors.

    V2 constructs diagonal Fisher and cost matrices by design. This analytic
    path preserves the generalized-eigenvector objective while avoiding a dense
    eigensolver for every image block.
    """

    f_diag = np.diag(np.asarray(fisher, dtype=np.float64))
    p_diag = np.diag(np.asarray(cost, dtype=np.float64))
    if f_diag.shape != (PAYLOAD_BITS_PER_BLOCK,) or p_diag.shape != (
        PAYLOAD_BITS_PER_BLOCK,
    ):
        raise ValueError("fisher and cost must be 8x8 diagonal matrices")
    if np.any(f_diag <= 0.0) or np.any(p_diag <= 0.0):
        raise ValueError("fisher and cost diagonals must be positive")

    values = f_diag / p_diag
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    sorted_values = values[order]
    basis = np.zeros((PAYLOAD_BITS_PER_BLOCK, PAYLOAD_BITS_PER_BLOCK), dtype=np.float64)
    for column, index in enumerate(order):
        basis[index, column] = 1.0 / np.sqrt(p_diag[index])
    return sorted_values, basis


def fixed_basis(cost: np.ndarray) -> np.ndarray:
    diagonal = np.diag(cost)
    return np.diag(1.0 / np.sqrt(diagonal))


def random_p_orthonormal_basis(
    cost: np.ndarray, key: bytes, block_index: tuple[int, int]
) -> np.ndarray:
    cost_diagonal = tuple(float(value) for value in np.diag(np.asarray(cost, dtype=np.float64)))
    return _cached_random_p_orthonormal_basis(
        cost_diagonal,
        key,
        int(block_index[0]),
        int(block_index[1]),
    ).copy()


@lru_cache(maxsize=65536)
def _cached_random_p_orthonormal_basis(
    cost_diagonal: tuple[float, ...],
    key: bytes,
    block_row: int,
    block_col: int,
) -> np.ndarray:
    cost = np.diag(np.asarray(cost_diagonal, dtype=np.float64))
    seed = _seed_from_hmac(key, b"random-basis", (block_row, block_col), -1)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=cost.shape)
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    chol = np.linalg.cholesky(cost)
    return np.linalg.solve(chol.T, q)


def serialize_features(
    canonical: np.ndarray,
    energies: np.ndarray,
    image_shape: tuple[int, int],
    block_index: tuple[int, int],
    feature_step: float = DELTA_FEATURE,
) -> bytes:
    """Serialize quantized canonical block features deterministically."""

    if feature_step <= 0:
        raise ValueError("feature_step must be positive")
    height, width = image_shape
    block_row, block_col = block_index
    coords = model_coordinates(canonical.shape[0])
    payload = bytearray(PROTOCOL_ID + b"\0")
    payload.extend(struct.pack(">IIII", height, width, block_row, block_col))
    payload.extend(struct.pack(">H", len(coords)))
    for row, col in coords:
        payload.extend(struct.pack(">i", _quantize(canonical[row, col], feature_step)))
    payload.extend(struct.pack(">H", len(energies)))
    for energy in energies:
        payload.extend(struct.pack(">i", _quantize(energy, feature_step)))
    return bytes(payload)


def authentication_bits(
    canonical: np.ndarray,
    energies: np.ndarray,
    image_shape: tuple[int, int],
    block_index: tuple[int, int],
    auth_key: bytes,
) -> np.ndarray:
    message = serialize_features(canonical, energies, image_shape, block_index)
    digest = hmac.new(auth_key, message, hashlib.sha256).digest()
    return np.array([(digest[0] >> shift) & 1 for shift in range(7, -1, -1)], dtype=np.uint8)


def dither(embed_key: bytes, block_index: tuple[int, int], bit_index: int, delta: float) -> float:
    if delta <= 0:
        raise ValueError("delta must be positive")
    seed = _seed_from_hmac(embed_key, b"dither", block_index, bit_index)
    unit = seed / float(2**64)
    return (unit - 0.5) * delta / 2.0


def qim_embed_scalar(alpha: float, bit: int, delta: float, dither_value: float) -> float:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    if delta <= 0:
        raise ValueError("delta must be positive")
    shifted = (alpha - dither_value) / delta - bit / 2.0
    return float(dither_value + delta * (np.rint(shifted) + bit / 2.0))


def qim_extract_bit(alpha: float, delta: float, dither_value: float) -> int:
    if delta <= 0:
        raise ValueError("delta must be positive")
    shifted = (alpha - dither_value) / delta
    q0 = abs(shifted - np.rint(shifted))
    q1 = abs(shifted - (np.rint(shifted - 0.5) + 0.5))
    return int(0 if q0 <= q1 else 1)


def embed_block_coefficients(
    coefficients: np.ndarray,
    image_shape: tuple[int, int],
    block_index: tuple[int, int],
    master_key: bytes,
    delta_embed: float,
    basis_mode: BasisMode = "fisher",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed one block's authentication bits into reserved DCT coefficients."""

    keys = derive_keys(master_key)
    model = build_block_model(coefficients, basis_mode, keys, block_index)
    bits = authentication_bits(
        model.canonical, model.energies, image_shape, block_index, keys.auth
    )
    vector = reserved_vector(coefficients)
    for index, bit in enumerate(bits):
        direction = model.basis[:, index]
        alpha = float(direction.T @ model.cost @ vector)
        target = qim_embed_scalar(
            alpha, int(bit), delta_embed, dither(keys.embed, block_index, index, delta_embed)
        )
        vector = vector + (target - alpha) * direction
    return set_reserved_vector(coefficients, vector), bits


def extract_block_score(
    coefficients: np.ndarray,
    image_shape: tuple[int, int],
    block_index: tuple[int, int],
    master_key: bytes,
    delta_embed: float,
    basis_mode: BasisMode = "fisher",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return Hamming score, extracted bits, and recomputed authentication bits."""

    keys = derive_keys(master_key)
    model = build_block_model(coefficients, basis_mode, keys, block_index)
    expected = authentication_bits(
        model.canonical, model.energies, image_shape, block_index, keys.auth
    )
    vector = reserved_vector(coefficients)
    extracted = []
    for index in range(PAYLOAD_BITS_PER_BLOCK):
        direction = model.basis[:, index]
        alpha = float(direction.T @ model.cost @ vector)
        extracted.append(qim_extract_bit(alpha, delta_embed, dither(keys.embed, block_index, index, delta_embed)))
    extracted_array = np.asarray(extracted, dtype=np.uint8)
    score = float(np.mean(extracted_array != expected))
    return score, extracted_array, expected


def calibrate_threshold(scores: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """Select the smallest k/8 threshold whose empirical FPR is <= alpha."""

    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    for k in range(PAYLOAD_BITS_PER_BLOCK + 1):
        tau = k / PAYLOAD_BITS_PER_BLOCK
        fpr = float(np.mean(values > tau))
        if fpr <= alpha:
            return tau, fpr
    raise RuntimeError("threshold grid is exhausted")


def block_tamper_map(scores: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) > tau


def expand_block_map(block_map: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    labels = np.asarray(block_map, dtype=bool)
    return np.repeat(np.repeat(labels, block_size, axis=0), block_size, axis=1)


def _quantize(value: float, step: float) -> int:
    return int(np.rint(float(value) / step))


def _seed_from_hmac(
    key: bytes, label: bytes, block_index: tuple[int, int], bit_index: int
) -> int:
    message = label + struct.pack(">IIi", block_index[0], block_index[1], bit_index)
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
