from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.blind_v2 import BasisMode, key_id, master_key_from_seed, trim_to_block_grid
from src.blind_v2_image import (
    SyntheticAttack,
    apply_synthetic_attack,
    block_metrics,
    calibrate_delta_embed,
    calibrate_tau_from_clean_images,
    embed_image,
    verify_image,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff"}
DEFAULT_ATTACKS: tuple[SyntheticAttack, ...] = (
    "center_mean",
    "copy_move",
    "constant_average_block",
    "inter_block_substitution",
)
DEFAULT_BASIS_MODES: tuple[BasisMode, ...] = (
    "fisher",
    "smallest",
    "random",
    "fixed",
    "identity_cost",
)
THRESHOLD_GRID = tuple(k / 8.0 for k in range(9))


@dataclass(frozen=True)
class CorpusImage:
    subcorpus: str
    relative_path: str
    path: Path
    array: np.ndarray
    original_shape: tuple[int, int]
    trimmed_shape: tuple[int, int]
    sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_paths(root: Path, max_images_per_subcorpus: int) -> dict[str, list[Path]]:
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    if max_images_per_subcorpus <= 0:
        raise ValueError("max_images_per_subcorpus must be positive")

    selected: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            selected[directory.name] = files[:max_images_per_subcorpus]
    if not selected:
        raise ValueError(f"no images found under {root}")
    return selected


def load_corpus_images(
    root: Path,
    max_images_per_subcorpus: int,
    hash_files: bool = False,
) -> list[CorpusImage]:
    items: list[CorpusImage] = []
    for subcorpus, paths in collect_image_paths(root, max_images_per_subcorpus).items():
        for path in paths:
            with Image.open(path) as image:
                luminance = np.asarray(image.convert("L"), dtype=np.float64)
            trimmed = trim_to_block_grid(luminance)
            items.append(
                CorpusImage(
                    subcorpus=subcorpus,
                    relative_path=path.relative_to(root).as_posix(),
                    path=path,
                    array=trimmed,
                    original_shape=tuple(int(value) for value in luminance.shape),
                    trimmed_shape=tuple(int(value) for value in trimmed.shape),
                    sha256=sha256_file(path) if hash_files else None,
                )
            )
    return items


def clean_threshold_curve(scores: np.ndarray) -> list[dict[str, float | int]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    records: list[dict[str, float | int]] = []
    for tau in THRESHOLD_GRID:
        flagged = values > tau
        records.append(
            {
                "tau": float(tau),
                "flaggedBlockCount": int(np.sum(flagged)),
                "blockCount": int(values.size),
                "falsePositiveRate": float(np.mean(flagged)),
            }
        )
    return records


def attack_threshold_curve(
    score_truth_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float | int]]:
    if not score_truth_pairs:
        raise ValueError("score_truth_pairs must not be empty")
    records: list[dict[str, float | int]] = []
    for tau in THRESHOLD_GRID:
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    images = load_corpus_images(
        root,
        max_images_per_subcorpus=args.max_images_per_subcorpus,
        hash_files=args.hash_files,
    )
    arrays = [item.array for item in images]
    master_key = master_key_from_seed(args.master_key_seed)
    candidates = [float(value) for value in args.delta_embed_candidates]
    basis_modes = list(args.basis_modes)
    attacks = list(args.attacks)

    delta_embed, delta_records = calibrate_delta_embed(
        arrays,
        master_key,
        candidates,
        min_psnr_db=args.min_psnr_db,
        max_clean_bit_error_rate=args.max_clean_bit_error_rate,
        basis_mode=args.calibration_basis_mode,
    )

    baseline_records = []
    for mode in basis_modes:
        embedded = [
            embed_image(item.array, master_key, delta_embed, basis_mode=mode)
            for item in images
        ]
        watermarked_images = [item.watermarked for item in embedded]
        tau, validation_fpr, clean_scores = calibrate_tau_from_clean_images(
            watermarked_images,
            master_key,
            delta_embed,
            alpha=args.target_false_positive_rate,
            basis_mode=mode,
        )

        clean_records = []
        for index, embedded_item in enumerate(embedded):
            source_item = images[index]
            verified = verify_image(
                embedded_item.watermarked,
                master_key,
                delta_embed,
                tau=tau,
                basis_mode=mode,
            )
            clean_records.append(
                {
                    "imageIndex": index,
                    "subcorpus": source_item.subcorpus,
                    "relativePath": source_item.relative_path,
                    "sha256": source_item.sha256,
                    "originalShape": list(source_item.original_shape),
                    "trimmedShape": list(source_item.trimmed_shape),
                    "psnrDb": float(embedded_item.psnr_db),
                    "cleanBitErrorRate": float(verified.clean_bit_error_rate),
                    "flaggedBlockCount": int(np.sum(verified.tamper_blocks)),
                    "blockCount": int(verified.scores.size),
                    "maxBlockScore": float(np.max(verified.scores)),
                }
            )

        attack_records = []
        for attack in attacks:
            metric_items = []
            score_truth_pairs = []
            for index, embedded_item in enumerate(embedded):
                source = embedded[(index + 1) % len(embedded)].watermarked
                attacked, truth = apply_synthetic_attack(
                    embedded_item.watermarked,
                    attack,
                    source=source,
                )
                verified = verify_image(
                    attacked,
                    master_key,
                    delta_embed,
                    tau=tau,
                    basis_mode=mode,
                )
                score_truth_pairs.append((verified.scores, truth))
                metrics = block_metrics(verified.tamper_blocks, truth)
                metrics["imageIndex"] = index
                metrics["subcorpus"] = images[index].subcorpus
                metrics["relativePath"] = images[index].relative_path
                metrics["truthBlockCount"] = int(np.sum(truth))
                metrics["predictedBlockCount"] = int(np.sum(verified.tamper_blocks))
                metric_items.append(metrics)
            attack_records.append(
                {
                    "attack": attack,
                    "meanBlockF1": float(np.mean([item["f1"] for item in metric_items])),
                    "meanBlockIoU": float(np.mean([item["iou"] for item in metric_items])),
                    "meanBlockFpr": float(np.mean([item["fpr"] for item in metric_items])),
                    "meanBlockFnr": float(np.mean([item["fnr"] for item in metric_items])),
                    "thresholdCurve": attack_threshold_curve(score_truth_pairs),
                    "images": metric_items,
                }
            )

        baseline_records.append(
            {
                "basisMode": mode,
                "tau": float(tau),
                "validationFalsePositiveRate": float(validation_fpr),
                "cleanScoreHistogram": {
                    f"{k}/8": int(np.sum(clean_scores == k / 8.0)) for k in range(9)
                },
                "cleanThresholdCurve": clean_threshold_curve(clean_scores),
                "clean": clean_records,
                "attacks": attack_records,
                "meanCleanPsnrDb": float(np.mean([item["psnrDb"] for item in clean_records])),
                "minCleanPsnrDb": float(np.min([item["psnrDb"] for item in clean_records])),
                "maxCleanBitErrorRate": float(
                    np.max([item["cleanBitErrorRate"] for item in clean_records])
                ),
                "cleanFlaggedBlockCount": int(
                    np.sum([item["flaggedBlockCount"] for item in clean_records])
                ),
            }
        )

    subcorpora = {}
    for item in images:
        record = subcorpora.setdefault(
            item.subcorpus,
            {
                "selectedImageCount": 0,
                "trimmedShapes": {},
            },
        )
        record["selectedImageCount"] += 1
        shape_key = f"{item.trimmed_shape[0]}x{item.trimmed_shape[1]}"
        record["trimmedShapes"][shape_key] = record["trimmedShapes"].get(shape_key, 0) + 1

    summary = {
        "schema": "blind-v2-real-image-scratch-pilot/v1",
        "pilotMode": "engineering_scratch_same_images_for_calibration_and_attack",
        "promotionReady": False,
        "promotionBlocker": (
            "dataset source, version, license, citation, split, and preprocessing "
            "manifests must be frozen before public reporting"
        ),
        "datasetRootName": root.name,
        "datasetRootStored": False,
        "dataCopied": False,
        "hashFiles": bool(args.hash_files),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "imageCount": len(images),
        "subcorpora": subcorpora,
        "masterKeyId": key_id(master_key),
        "masterKeyStored": False,
        "calibrationBasisMode": args.calibration_basis_mode,
        "basisModes": basis_modes,
        "attacks": attacks,
        "deltaEmbedCandidates": candidates,
        "selectedDeltaEmbed": float(delta_embed),
        "deltaCalibration": delta_records,
        "targetFalsePositiveRate": float(args.target_false_positive_rate),
        "minPsnrDb": float(args.min_psnr_db),
        "maxCleanBitErrorRate": float(args.max_clean_bit_error_rate),
        "baselines": baseline_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a private scratch pilot of the V2 blind detector on a local image corpus."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "_scratch" / "blind_v2_real_pilot.json",
    )
    parser.add_argument("--max-images-per-subcorpus", type=int, default=4)
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--master-key-seed", type=int, default=20260920)
    parser.add_argument(
        "--delta-embed-candidates",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--min-psnr-db", type=float, default=40.0)
    parser.add_argument("--max-clean-bit-error-rate", type=float, default=0.01)
    parser.add_argument("--target-false-positive-rate", type=float, default=0.0)
    parser.add_argument(
        "--calibration-basis-mode",
        choices=DEFAULT_BASIS_MODES,
        default="fisher",
    )
    parser.add_argument(
        "--basis-modes",
        choices=DEFAULT_BASIS_MODES,
        nargs="+",
        default=list(DEFAULT_BASIS_MODES),
    )
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
        f"wrote private scratch real-image pilot: images={summary['imageCount']} "
        f"delta={summary['selectedDeltaEmbed']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
