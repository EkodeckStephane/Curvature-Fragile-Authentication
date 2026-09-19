from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.blind_v2 import (
    BLOCK_SIZE,
    PAYLOAD_BPP,
    BasisMode,
    block_tamper_map,
    calibrate_threshold,
    dct2,
    embed_block_coefficients,
    expand_block_map,
    extract_block_score,
    idct2,
    trim_to_block_grid,
)


@dataclass(frozen=True)
class ImageEmbedResult:
    original: np.ndarray
    watermarked: np.ndarray
    embedded_bits: np.ndarray
    psnr_db: float
    payload_bpp: float


@dataclass(frozen=True)
class ImageVerifyResult:
    scores: np.ndarray
    extracted_bits: np.ndarray
    expected_bits: np.ndarray
    tamper_blocks: np.ndarray | None = None
    tamper_pixels: np.ndarray | None = None
    clean_bit_error_rate: float | None = None


def image_blocks(image: np.ndarray, block_size: int = BLOCK_SIZE) -> Iterable[tuple[int, int, np.ndarray]]:
    """Yield non-overlapping blocks as (block_row, block_col, block)."""

    if image.ndim != 2:
        raise ValueError("image must be a 2D luminance array")
    if image.shape[0] % block_size or image.shape[1] % block_size:
        raise ValueError("image dimensions must be multiples of block_size")
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            yield row // block_size, col // block_size, image[row : row + block_size, col : col + block_size]


def embed_image(
    image: np.ndarray,
    master_key: bytes,
    delta_embed: float,
    basis_mode: BasisMode = "fisher",
) -> ImageEmbedResult:
    """Embed V2 authentication bits into every block of an image."""

    original = trim_to_block_grid(image)
    watermarked = np.empty_like(original, dtype=np.float64)
    block_rows = original.shape[0] // BLOCK_SIZE
    block_cols = original.shape[1] // BLOCK_SIZE
    bits = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)

    for block_row, block_col, block in image_blocks(original):
        coefficients = dct2(block)
        embedded_coeffs, block_bits = embed_block_coefficients(
            coefficients,
            image_shape=original.shape,
            block_index=(block_row, block_col),
            master_key=master_key,
            delta_embed=delta_embed,
            basis_mode=basis_mode,
        )
        row = block_row * BLOCK_SIZE
        col = block_col * BLOCK_SIZE
        watermarked[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE] = idct2(embedded_coeffs)
        bits[block_row, block_col, :] = block_bits

    return ImageEmbedResult(
        original=original,
        watermarked=watermarked,
        embedded_bits=bits,
        psnr_db=psnr(original, watermarked),
        payload_bpp=PAYLOAD_BPP,
    )


def verify_image(
    image: np.ndarray,
    master_key: bytes,
    delta_embed: float,
    tau: float | None = None,
    basis_mode: BasisMode = "fisher",
) -> ImageVerifyResult:
    """Verify an image and optionally threshold the block scores."""

    received = trim_to_block_grid(image)
    block_rows = received.shape[0] // BLOCK_SIZE
    block_cols = received.shape[1] // BLOCK_SIZE
    scores = np.zeros((block_rows, block_cols), dtype=np.float64)
    extracted = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)
    expected = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)

    for block_row, block_col, block in image_blocks(received):
        score, block_extracted, block_expected = extract_block_score(
            dct2(block),
            image_shape=received.shape,
            block_index=(block_row, block_col),
            master_key=master_key,
            delta_embed=delta_embed,
            basis_mode=basis_mode,
        )
        scores[block_row, block_col] = score
        extracted[block_row, block_col, :] = block_extracted
        expected[block_row, block_col, :] = block_expected

    tamper_blocks = None
    tamper_pixels = None
    if tau is not None:
        tamper_blocks = block_tamper_map(scores, tau)
        tamper_pixels = expand_block_map(tamper_blocks)

    return ImageVerifyResult(
        scores=scores,
        extracted_bits=extracted,
        expected_bits=expected,
        tamper_blocks=tamper_blocks,
        tamper_pixels=tamper_pixels,
        clean_bit_error_rate=float(np.mean(extracted != expected)),
    )


def calibrate_delta_embed(
    images: list[np.ndarray],
    master_key: bytes,
    candidates: list[float],
    min_psnr_db: float = 40.0,
    max_clean_bit_error_rate: float = 0.01,
    basis_mode: BasisMode = "fisher",
) -> tuple[float, list[dict[str, float]]]:
    """Choose the smallest candidate satisfying quality and clean decoding."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    records: list[dict[str, float]] = []
    selected: float | None = None
    for candidate in candidates:
        psnrs = []
        errors = []
        for image in images:
            embedded = embed_image(image, master_key, candidate, basis_mode)
            verified = verify_image(embedded.watermarked, master_key, candidate, basis_mode=basis_mode)
            psnrs.append(embedded.psnr_db)
            errors.append(float(verified.clean_bit_error_rate))
        record = {
            "deltaEmbed": float(candidate),
            "minPsnrDb": float(np.min(psnrs)),
            "meanPsnrDb": float(np.mean(psnrs)),
            "maxCleanBitErrorRate": float(np.max(errors)),
            "meanCleanBitErrorRate": float(np.mean(errors)),
        }
        record["accepted"] = bool(
            record["minPsnrDb"] >= min_psnr_db
            and record["maxCleanBitErrorRate"] <= max_clean_bit_error_rate
        )
        records.append(record)
        if selected is None and record["accepted"]:
            selected = float(candidate)
    if selected is None:
        raise RuntimeError("no delta candidate satisfies the calibration targets")
    return selected, records


def calibrate_tau_from_clean_images(
    watermarked_images: list[np.ndarray],
    master_key: bytes,
    delta_embed: float,
    alpha: float = 0.01,
    basis_mode: BasisMode = "fisher",
) -> tuple[float, float, np.ndarray]:
    """Calibrate tau from authentic watermarked images."""

    scores = []
    for image in watermarked_images:
        verified = verify_image(image, master_key, delta_embed, basis_mode=basis_mode)
        scores.append(verified.scores.ravel())
    clean_scores = np.concatenate(scores)
    tau, fpr = calibrate_threshold(clean_scores, alpha=alpha)
    return tau, fpr, clean_scores


def psnr(reference: np.ndarray, candidate: np.ndarray, peak: float = 255.0) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")
    mse = float(np.mean((ref - cand) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / mse))
