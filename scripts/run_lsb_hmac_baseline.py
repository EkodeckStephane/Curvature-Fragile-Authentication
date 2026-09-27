from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import platform
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_blind_v2_real_pilot import (  # noqa: E402
    DEFAULT_ATTACKS,
    CorpusImage,
    load_corpus_image_splits,
    summarize_subcorpora,
)
from src.blind_v2 import BLOCK_SIZE, calibrate_threshold, key_id, master_key_from_seed, trim_to_block_grid  # noqa: E402
from src.blind_v2_image import SyntheticAttack, apply_synthetic_attack, block_metrics, psnr  # noqa: E402


EMBED_POSITIONS: tuple[tuple[int, int], ...] = tuple((0, col) for col in range(8))


@dataclass
class PhaseTimer:
    records: dict[str, float]

    @contextmanager
    def measure(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.records[name] = self.records.get(name, 0.0) + perf_counter() - start

    def rounded(self) -> dict[str, float]:
        return {name: round(value, 6) for name, value in sorted(self.records.items())}


def _uint8_grid(image: np.ndarray) -> np.ndarray:
    trimmed = trim_to_block_grid(image)
    return np.clip(np.rint(trimmed), 0, 255).astype(np.uint8)


def _clear_lsb(block: np.ndarray) -> np.ndarray:
    return np.bitwise_and(block.astype(np.uint8), np.uint8(0xFE))


def _auth_bits(
    block: np.ndarray,
    *,
    image_shape: tuple[int, int],
    block_index: tuple[int, int],
    key: bytes,
) -> np.ndarray:
    normalized = _clear_lsb(block)
    message = bytearray()
    message.extend(int(image_shape[0]).to_bytes(4, "big"))
    message.extend(int(image_shape[1]).to_bytes(4, "big"))
    message.extend(int(block_index[0]).to_bytes(4, "big"))
    message.extend(int(block_index[1]).to_bytes(4, "big"))
    message.extend(normalized.tobytes())
    digest = hmac.new(key, bytes(message), hashlib.sha256).digest()
    return np.unpackbits(np.frombuffer(digest[:1], dtype=np.uint8)).astype(np.uint8)


def embed_lsb_hmac_image(image: np.ndarray, key: bytes) -> dict[str, Any]:
    original = _uint8_grid(image)
    watermarked = original.copy()
    block_rows = original.shape[0] // BLOCK_SIZE
    block_cols = original.shape[1] // BLOCK_SIZE
    bits = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)

    for block_row in range(block_rows):
        for block_col in range(block_cols):
            row = block_row * BLOCK_SIZE
            col = block_col * BLOCK_SIZE
            block = watermarked[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
            block_bits = _auth_bits(
                block,
                image_shape=original.shape,
                block_index=(block_row, block_col),
                key=key,
            )
            for bit_index, (pos_row, pos_col) in enumerate(EMBED_POSITIONS):
                block[pos_row, pos_col] = (block[pos_row, pos_col] & 0xFE) | int(
                    block_bits[bit_index]
                )
            bits[block_row, block_col, :] = block_bits

    return {
        "original": original.astype(np.float64),
        "watermarked": watermarked.astype(np.float64),
        "embeddedBits": bits,
        "psnrDb": psnr(original.astype(np.float64), watermarked.astype(np.float64)),
    }


def verify_lsb_hmac_image(
    image: np.ndarray,
    key: bytes,
    tau: float | None = None,
) -> dict[str, Any]:
    received = _uint8_grid(image)
    block_rows = received.shape[0] // BLOCK_SIZE
    block_cols = received.shape[1] // BLOCK_SIZE
    extracted = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)
    expected = np.zeros((block_rows, block_cols, 8), dtype=np.uint8)
    scores = np.zeros((block_rows, block_cols), dtype=np.float64)

    for block_row in range(block_rows):
        for block_col in range(block_cols):
            row = block_row * BLOCK_SIZE
            col = block_col * BLOCK_SIZE
            block = received[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
            expected_bits = _auth_bits(
                block,
                image_shape=received.shape,
                block_index=(block_row, block_col),
                key=key,
            )
            extracted_bits = np.array(
                [block[pos_row, pos_col] & 1 for pos_row, pos_col in EMBED_POSITIONS],
                dtype=np.uint8,
            )
            expected[block_row, block_col, :] = expected_bits
            extracted[block_row, block_col, :] = extracted_bits
            scores[block_row, block_col] = float(np.mean(extracted_bits != expected_bits))

    tamper_blocks = scores > tau if tau is not None else None
    return {
        "scores": scores,
        "extractedBits": extracted,
        "expectedBits": expected,
        "tamperBlocks": tamper_blocks,
        "cleanBitErrorRate": float(np.mean(extracted != expected)),
    }


def _calibrate_tau(
    watermarked_images: list[np.ndarray],
    key: bytes,
    alpha: float,
) -> tuple[float, float, np.ndarray]:
    scores = [
        verify_lsb_hmac_image(image, key)["scores"].ravel()
        for image in watermarked_images
    ]
    clean_scores = np.concatenate(scores)
    tau, fpr = calibrate_threshold(clean_scores, alpha=alpha)
    return float(tau), float(fpr), clean_scores


def _clean_threshold_curve(scores: np.ndarray) -> list[dict[str, float | int]]:
    values = np.asarray(scores, dtype=np.float64)
    return [
        {
            "tau": float(k / 8.0),
            "flaggedBlockCount": int(np.sum(values > k / 8.0)),
            "blockCount": int(values.size),
            "falsePositiveRate": float(np.mean(values > k / 8.0)),
        }
        for k in range(9)
    ]


def _attack_threshold_curve(
    score_truth_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for k in range(9):
        tau = k / 8.0
        metrics = [
            block_metrics(np.asarray(scores, dtype=np.float64) > tau, truth)
            for scores, truth in score_truth_pairs
        ]
        records.append(
            {
                "tau": float(tau),
                "meanBlockF1": float(np.mean([item["f1"] for item in metrics])),
                "meanBlockIoU": float(np.mean([item["iou"] for item in metrics])),
                "meanBlockFpr": float(np.mean([item["fpr"] for item in metrics])),
                "meanBlockFnr": float(np.mean([item["fnr"] for item in metrics])),
                "meanPrecision": float(np.mean([item["precision"] for item in metrics])),
                "meanRecall": float(np.mean([item["recall"] for item in metrics])),
            }
        )
    return records


def _image_key(item: CorpusImage, index: int) -> str:
    return f"{item.subcorpus}:{index:04d}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    timer = PhaseTimer({})
    with timer.measure("load_corpus"):
        calibration_images, evaluation_images = load_corpus_image_splits(
            args.root.resolve(),
            calibration_images_per_subcorpus=args.calibration_images_per_subcorpus,
            evaluation_images_per_subcorpus=args.evaluation_images_per_subcorpus,
            hash_files=False,
        )

    key = master_key_from_seed(args.master_key_seed)
    attacks = list(args.attacks)

    with timer.measure("embed_calibration"):
        calibration_embedded = [
            embed_lsb_hmac_image(item.array, key) for item in calibration_images
        ]
    calibration_watermarked = [item["watermarked"] for item in calibration_embedded]

    with timer.measure("tau_calibration"):
        tau, validation_fpr, clean_scores = _calibrate_tau(
            calibration_watermarked,
            key,
            alpha=args.target_false_positive_rate,
        )

    with timer.measure("embed_evaluation"):
        evaluation_embedded = [
            embed_lsb_hmac_image(item.array, key) for item in evaluation_images
        ]

    clean_records = []
    evaluation_clean_scores = []
    with timer.measure("verify_clean_evaluation"):
        for index, embedded_item in enumerate(evaluation_embedded):
            source_item = evaluation_images[index]
            verified = verify_lsb_hmac_image(embedded_item["watermarked"], key, tau=tau)
            scores = verified["scores"]
            evaluation_clean_scores.append(scores.ravel())
            clean_records.append(
                {
                    "imageKey": _image_key(source_item, index),
                    "imageIndex": index,
                    "subcorpus": source_item.subcorpus,
                    "psnrDb": float(embedded_item["psnrDb"]),
                    "cleanBitErrorRate": float(verified["cleanBitErrorRate"]),
                    "flaggedBlockCount": int(np.sum(verified["tamperBlocks"])),
                    "blockCount": int(scores.size),
                    "maxBlockScore": float(np.max(scores)),
                    "tau": float(tau),
                }
            )

    attack_records = []
    for attack in attacks:
        metric_items = []
        score_truth_pairs = []
        with timer.measure(f"attack.{attack}"):
            for index, embedded_item in enumerate(evaluation_embedded):
                source_item = evaluation_images[index]
                source = evaluation_embedded[(index + 1) % len(evaluation_embedded)][
                    "watermarked"
                ]
                attacked, truth = apply_synthetic_attack(
                    embedded_item["watermarked"],
                    attack,
                    source=source,
                )
                verified = verify_lsb_hmac_image(attacked, key, tau=tau)
                scores = verified["scores"]
                score_truth_pairs.append((scores, truth))
                metrics = block_metrics(verified["tamperBlocks"], truth)
                metrics["imageKey"] = _image_key(source_item, index)
                metrics["imageIndex"] = index
                metrics["subcorpus"] = source_item.subcorpus
                metrics["tau"] = float(tau)
                metrics["truthBlockCount"] = int(np.sum(truth))
                metrics["predictedBlockCount"] = int(np.sum(verified["tamperBlocks"]))
                metric_items.append(metrics)
        attack_records.append(
            {
                "attack": attack,
                "meanBlockF1": float(np.mean([item["f1"] for item in metric_items])),
                "meanBlockIoU": float(np.mean([item["iou"] for item in metric_items])),
                "meanBlockFpr": float(np.mean([item["fpr"] for item in metric_items])),
                "meanBlockFnr": float(np.mean([item["fnr"] for item in metric_items])),
                "thresholdCurve": _attack_threshold_curve(score_truth_pairs),
                "images": metric_items,
            }
        )

    summary = {
        "schema": "external-lsb-hmac-baseline-scratch/v1",
        "baselineId": "block-lsb-hmac-v1",
        "baselineFamily": "classical block-wise LSB fragile watermark with keyed hash authentication",
        "literatureAnchor": {
            "citationKey": "wong2001secret",
            "note": "External family baseline inspired by block-wise fragile authentication watermarks; this implementation uses HMAC-SHA256 rather than reproducing a specific paper exactly.",
            "doi": "10.1109/83.951543",
        },
        "promotionReady": False,
        "inputMode": "directory",
        "datasetRootName": args.root.resolve().name,
        "datasetRootStored": False,
        "dataCopied": False,
        "pixelsStored": False,
        "localPathsStored": False,
        "perImageFileNamesStored": False,
        "masterKeyId": key_id(key),
        "masterKeyStored": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "calibrationImageCount": len(calibration_images),
        "evaluationImageCount": len(evaluation_images),
        "imageCount": len(evaluation_images),
        "calibrationSubcorpora": summarize_subcorpora(calibration_images),
        "evaluationSubcorpora": summarize_subcorpora(evaluation_images),
        "targetFalsePositiveRate": float(args.target_false_positive_rate),
        "tau": float(tau),
        "validationFalsePositiveRate": float(validation_fpr),
        "attacks": attacks,
        "timingSeconds": timer.rounded(),
        "calibrationCleanThresholdCurve": _clean_threshold_curve(clean_scores),
        "evaluationCleanThresholdCurve": _clean_threshold_curve(
            np.concatenate(evaluation_clean_scores)
        ),
        "clean": clean_records,
        "attacksDetailed": attack_records,
        "meanCleanPsnrDb": float(np.mean([item["psnrDb"] for item in clean_records])),
        "minCleanPsnrDb": float(np.min([item["psnrDb"] for item in clean_records])),
        "maxCleanBitErrorRate": float(
            np.max([item["cleanBitErrorRate"] for item in clean_records])
        ),
        "cleanFlaggedBlockCount": int(
            np.sum([item["flaggedBlockCount"] for item in clean_records])
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a classical block-wise LSB-HMAC fragile watermark baseline."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-images-per-subcorpus", type=int, required=True)
    parser.add_argument("--evaluation-images-per-subcorpus", type=int, required=True)
    parser.add_argument("--master-key-seed", type=int, default=20260920)
    parser.add_argument("--target-false-positive-rate", type=float, default=0.01)
    parser.add_argument(
        "--attacks",
        choices=DEFAULT_ATTACKS,
        nargs="+",
        default=list(DEFAULT_ATTACKS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(
        "wrote private LSB-HMAC baseline: "
        f"images={summary['imageCount']} tau={summary['tau']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
