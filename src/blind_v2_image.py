from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from src.blind_v2 import (
    BLOCK_SIZE,
    PAYLOAD_BPP,
    BasisMode,
    ScoreMode,
    block_tamper_map,
    calibrate_threshold,
    derive_keys,
    dct2,
    embed_block_coefficients_with_keys,
    expand_block_map,
    extract_block_score_with_keys,
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


SyntheticAttack = Literal[
    "center_mean",
    "copy_move",
    "constant_average_block",
    "inter_block_substitution",
]


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
    keys = derive_keys(master_key)

    for block_row, block_col, block in image_blocks(original):
        coefficients = dct2(block)
        embedded_coeffs, block_bits = embed_block_coefficients_with_keys(
            coefficients,
            image_shape=original.shape,
            block_index=(block_row, block_col),
            keys=keys,
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
    score_mode: ScoreMode = "hamming",
) -> ImageVerifyResult:
    """Verify an image and optionally threshold the block scores."""

    received = trim_to_block_grid(image)
    block_rows = received.shape[0] // BLOCK_SIZE
    block_cols = received.shape[1] // BLOCK_SIZE
    scores = np.zeros((block_rows, block_cols), dtype=np.float64)
    extracted = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)
    expected = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)
    keys = derive_keys(master_key)

    for block_row, block_col, block in image_blocks(received):
        score, block_extracted, block_expected = extract_block_score_with_keys(
            dct2(block),
            image_shape=received.shape,
            block_index=(block_row, block_col),
            keys=keys,
            delta_embed=delta_embed,
            basis_mode=basis_mode,
            score_mode=score_mode,
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
    score_mode: ScoreMode = "hamming",
) -> tuple[float, float, np.ndarray]:
    """Calibrate tau from authentic watermarked images."""

    scores = []
    for image in watermarked_images:
        verified = verify_image(
            image,
            master_key,
            delta_embed,
            basis_mode=basis_mode,
            score_mode=score_mode,
        )
        scores.append(verified.scores.ravel())
    clean_scores = np.concatenate(scores)
    tau, fpr = calibrate_threshold(clean_scores, alpha=alpha)
    return tau, fpr, clean_scores


def apply_synthetic_attack(
    image: np.ndarray,
    attack: SyntheticAttack,
    source: np.ndarray | None = None,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a deterministic block-aligned synthetic tamper attack.

    Returns the attacked image and the block-level ground-truth tamper mask.
    """

    value = trim_to_block_grid(image, block_size)
    attacked = value.copy()
    block_rows = value.shape[0] // block_size
    block_cols = value.shape[1] // block_size
    if block_rows < 4 or block_cols < 4:
        raise ValueError("synthetic attacks require at least a 4x4 block grid")
    mask = np.zeros((block_rows, block_cols), dtype=bool)

    if attack == "center_mean":
        rows = range(block_rows // 2 - 1, block_rows // 2 + 1)
        cols = range(block_cols // 2 - 1, block_cols // 2 + 1)
        fill = float(np.mean(value))
        for block_row in rows:
            for block_col in cols:
                _fill_block(attacked, block_row, block_col, fill, block_size)
                mask[block_row, block_col] = True
    elif attack == "copy_move":
        for offset_row in range(2):
            for offset_col in range(2):
                src = value[
                    offset_row * block_size : (offset_row + 1) * block_size,
                    offset_col * block_size : (offset_col + 1) * block_size,
                ]
                dst_row = block_rows - 2 + offset_row
                dst_col = block_cols - 2 + offset_col
                _set_block(attacked, dst_row, dst_col, src, block_size)
                mask[dst_row, dst_col] = True
    elif attack == "constant_average_block":
        targets = ((1, 1), (1, block_cols - 2), (block_rows - 2, 1))
        for block_row, block_col in targets:
            block = attacked[
                block_row * block_size : (block_row + 1) * block_size,
                block_col * block_size : (block_col + 1) * block_size,
            ]
            _fill_block(attacked, block_row, block_col, float(np.mean(block)), block_size)
            mask[block_row, block_col] = True
    elif attack == "inter_block_substitution":
        donor = trim_to_block_grid(source, block_size) if source is not None else value
        donor_block = donor[0:block_size, 0:block_size]
        targets = ((0, block_cols - 1), (block_rows - 1, 0))
        for block_row, block_col in targets:
            _set_block(attacked, block_row, block_col, donor_block, block_size)
            mask[block_row, block_col] = True
    else:
        raise ValueError(f"unknown synthetic attack: {attack}")

    return attacked, mask


def block_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    """Compute block-level binary localization metrics."""

    pred = np.asarray(predicted, dtype=bool)
    gt = np.asarray(truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError("predicted and truth masks must have the same shape")
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def psnr(reference: np.ndarray, candidate: np.ndarray, peak: float = 255.0) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")
    mse = float(np.mean((ref - cand) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / mse))


def _fill_block(
    image: np.ndarray, block_row: int, block_col: int, value: float, block_size: int
) -> None:
    row = block_row * block_size
    col = block_col * block_size
    image[row : row + block_size, col : col + block_size] = value


def _set_block(
    image: np.ndarray, block_row: int, block_col: int, block: np.ndarray, block_size: int
) -> None:
    row = block_row * block_size
    col = block_col * block_size
    image[row : row + block_size, col : col + block_size] = block
