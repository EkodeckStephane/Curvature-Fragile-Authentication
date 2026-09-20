from __future__ import annotations

import argparse
import csv
import hashlib
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

from src.blind_v2 import (
    DELTA_FEATURE,
    BasisMode,
    DeltaMode,
    key_id,
    master_key_from_seed,
    trim_to_block_grid,
)
from src.blind_v2 import ScoreMode
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
    record_id: str | None = None
    dataset: str | None = None
    source_image_id: str | None = None
    source_file_name: str | None = None


@dataclass
class PhaseTimer:
    records: dict[str, float]

    @contextmanager
    def measure(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.records[name] = self.records.get(name, 0.0) + elapsed

    def rounded(self) -> dict[str, float]:
        return {name: round(value, 6) for name, value in sorted(self.records.items())}


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


def collect_image_splits(
    root: Path,
    calibration_images_per_subcorpus: int,
    evaluation_images_per_subcorpus: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    if calibration_images_per_subcorpus <= 0:
        raise ValueError("calibration_images_per_subcorpus must be positive")
    if evaluation_images_per_subcorpus <= 0:
        raise ValueError("evaluation_images_per_subcorpus must be positive")

    needed = calibration_images_per_subcorpus + evaluation_images_per_subcorpus
    available = collect_image_paths(root, max_images_per_subcorpus=needed)
    calibration: dict[str, list[Path]] = {}
    evaluation: dict[str, list[Path]] = {}
    for subcorpus, paths in available.items():
        if len(paths) < needed:
            raise ValueError(
                f"{subcorpus} contains {len(paths)} images but {needed} are required "
                "for separated calibration/evaluation splits"
            )
        calibration[subcorpus] = paths[:calibration_images_per_subcorpus]
        evaluation[subcorpus] = paths[
            calibration_images_per_subcorpus:
            calibration_images_per_subcorpus + evaluation_images_per_subcorpus
        ]
    return calibration, evaluation


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


def load_corpus_images_from_selection(
    root: Path,
    selection: dict[str, list[Path]],
    hash_files: bool = False,
) -> list[CorpusImage]:
    items: list[CorpusImage] = []
    for subcorpus, paths in selection.items():
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


def load_corpus_image_splits(
    root: Path,
    calibration_images_per_subcorpus: int,
    evaluation_images_per_subcorpus: int,
    hash_files: bool = False,
) -> tuple[list[CorpusImage], list[CorpusImage]]:
    calibration, evaluation = collect_image_splits(
        root,
        calibration_images_per_subcorpus=calibration_images_per_subcorpus,
        evaluation_images_per_subcorpus=evaluation_images_per_subcorpus,
    )
    return (
        load_corpus_images_from_selection(root, calibration, hash_files=hash_files),
        load_corpus_images_from_selection(root, evaluation, hash_files=hash_files),
    )


def read_public_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def resolve_manifest_path(
    row: dict[str, str],
    *,
    manifest_derived_root: Path | None = None,
    dtd_root: Path | None = None,
    coco_root: Path | None = None,
) -> Path:
    if manifest_derived_root is not None:
        candidate = manifest_derived_root / row["subcorpus"] / row["derived_artifact_id"]
        if candidate.exists():
            return candidate

    dataset = row["dataset"]
    if dataset == "DTD textures":
        if dtd_root is None:
            raise ValueError("dtd_root is required to resolve DTD manifest rows")
        candidate = dtd_root / Path(row["source_image_id"])
    elif dataset == "MS-COCO val2017":
        if coco_root is None:
            raise ValueError("coco_root is required to resolve COCO manifest rows")
        candidate = coco_root / row["source_file_name"]
    else:
        raise ValueError(f"unsupported manifest dataset: {dataset}")

    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def corpus_image_from_manifest_row(
    row: dict[str, str],
    *,
    path: Path,
    hash_files: bool = False,
    verify_derived_hash: bool = False,
) -> CorpusImage:
    file_hash = sha256_file(path) if hash_files or verify_derived_hash else None
    if verify_derived_hash and file_hash != row["derived_sha256"]:
        raise ValueError(
            f"hash mismatch for {row['record_id']}: expected "
            f"{row['derived_sha256']} but got {file_hash}"
        )
    with Image.open(path) as image:
        luminance = np.asarray(image.convert("L"), dtype=np.float64)
    trimmed = trim_to_block_grid(luminance)
    return CorpusImage(
        subcorpus=row["subcorpus"],
        relative_path=row["record_id"],
        path=path,
        array=trimmed,
        original_shape=tuple(int(value) for value in luminance.shape),
        trimmed_shape=tuple(int(value) for value in trimmed.shape),
        sha256=file_hash if hash_files else None,
        record_id=row["record_id"],
        dataset=row["dataset"],
        source_image_id=row["source_image_id"],
        source_file_name=row["source_file_name"],
    )


def load_manifest_images_from_rows(
    rows: list[dict[str, str]],
    *,
    manifest_derived_root: Path | None = None,
    dtd_root: Path | None = None,
    coco_root: Path | None = None,
    hash_files: bool = False,
) -> list[CorpusImage]:
    items: list[CorpusImage] = []
    for row in rows:
        path = resolve_manifest_path(
            row,
            manifest_derived_root=manifest_derived_root,
            dtd_root=dtd_root,
            coco_root=coco_root,
        )
        resolved_from_derived = (
            manifest_derived_root is not None
            and path == manifest_derived_root / row["subcorpus"] / row["derived_artifact_id"]
        )
        items.append(
            corpus_image_from_manifest_row(
                row,
                path=path,
                hash_files=hash_files,
                verify_derived_hash=resolved_from_derived,
            )
        )
    return items


def split_manifest_rows(
    rows: list[dict[str, str]],
    calibration_images_per_subcorpus: int,
    evaluation_images_per_subcorpus: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if calibration_images_per_subcorpus <= 0:
        raise ValueError("calibration_images_per_subcorpus must be positive")
    if evaluation_images_per_subcorpus <= 0:
        raise ValueError("evaluation_images_per_subcorpus must be positive")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["subcorpus"], []).append(row)
    calibration: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    needed = calibration_images_per_subcorpus + evaluation_images_per_subcorpus
    for subcorpus in sorted(grouped):
        items = grouped[subcorpus]
        if len(items) < needed:
            raise ValueError(
                f"{subcorpus} contains {len(items)} manifest rows but {needed} are "
                "required for separated calibration/evaluation splits"
            )
        calibration.extend(items[:calibration_images_per_subcorpus])
        evaluation.extend(
            items[
                calibration_images_per_subcorpus:
                calibration_images_per_subcorpus + evaluation_images_per_subcorpus
            ]
        )
    return calibration, evaluation


def load_manifest_image_splits(
    manifest_csv: Path,
    *,
    manifest_derived_root: Path | None = None,
    dtd_root: Path | None = None,
    coco_root: Path | None = None,
    calibration_images_per_subcorpus: int | None = None,
    evaluation_images_per_subcorpus: int | None = None,
    hash_files: bool = False,
) -> tuple[list[CorpusImage], list[CorpusImage], str]:
    rows = read_public_manifest(manifest_csv)
    split_requested = (
        calibration_images_per_subcorpus is not None
        or evaluation_images_per_subcorpus is not None
    )
    if split_requested:
        if (
            calibration_images_per_subcorpus is None
            or evaluation_images_per_subcorpus is None
        ):
            raise ValueError(
                "calibration_images_per_subcorpus and evaluation_images_per_subcorpus "
                "must be provided together"
            )
        calibration_rows, evaluation_rows = split_manifest_rows(
            rows,
            calibration_images_per_subcorpus=calibration_images_per_subcorpus,
            evaluation_images_per_subcorpus=evaluation_images_per_subcorpus,
        )
        pilot_mode = "manifest_restricted_separated_calibration_evaluation"
    else:
        calibration_rows = rows
        evaluation_rows = rows
        pilot_mode = "manifest_restricted_same_images_for_calibration_and_attack"

    resolver_args = {
        "manifest_derived_root": manifest_derived_root,
        "dtd_root": dtd_root,
        "coco_root": coco_root,
        "hash_files": hash_files,
    }
    return (
        load_manifest_images_from_rows(calibration_rows, **resolver_args),
        load_manifest_images_from_rows(evaluation_rows, **resolver_args),
        pilot_mode,
    )


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


def estimate_clean_bit_reliability(
    watermarked_images: list[np.ndarray],
    master_key: bytes,
    delta_embed: float,
    basis_mode: BasisMode,
    delta_mode: DeltaMode,
    auth_feature_step: float,
) -> dict[str, Any]:
    mismatch_counts = np.zeros(8, dtype=np.int64)
    total_count = 0
    for image in watermarked_images:
        verified = verify_image(
            image,
            master_key,
            delta_embed,
            basis_mode=basis_mode,
            score_mode="hamming",
            delta_mode=delta_mode,
            auth_feature_step=auth_feature_step,
        )
        mismatches = verified.extracted_bits != verified.expected_bits
        mismatch_counts += np.sum(mismatches, axis=(0, 1))
        total_count += int(mismatches.shape[0] * mismatches.shape[1])
    return {
        "mismatchCounts": [int(value) for value in mismatch_counts.tolist()],
        "totalCount": int(total_count),
        "errorRates": [
            float(value / total_count) if total_count else 0.0
            for value in mismatch_counts.tolist()
        ],
    }


def summarize_subcorpora(images: list[CorpusImage]) -> dict[str, dict[str, Any]]:
    subcorpora: dict[str, dict[str, Any]] = {}
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
    return subcorpora


def run(args: argparse.Namespace) -> dict[str, Any]:
    timer = PhaseTimer({})
    root = args.root.resolve() if getattr(args, "root", None) is not None else None
    with timer.measure("load_corpus"):
        manifest_csv = getattr(args, "manifest_csv", None)
        if manifest_csv is not None:
            calibration_images, evaluation_images, pilot_mode = load_manifest_image_splits(
                manifest_csv,
                manifest_derived_root=getattr(args, "manifest_derived_root", None),
                dtd_root=getattr(args, "dtd_root", None),
                coco_root=getattr(args, "coco_root", None),
                calibration_images_per_subcorpus=getattr(
                    args, "calibration_images_per_subcorpus", None
                ),
                evaluation_images_per_subcorpus=getattr(
                    args, "evaluation_images_per_subcorpus", None
                ),
                hash_files=args.hash_files,
            )
        else:
            if root is None:
                raise ValueError("either root or manifest_csv must be provided")
            split_requested = (
                getattr(args, "calibration_images_per_subcorpus", None) is not None
                or getattr(args, "evaluation_images_per_subcorpus", None) is not None
            )
            if split_requested:
                if (
                    getattr(args, "calibration_images_per_subcorpus", None) is None
                    or getattr(args, "evaluation_images_per_subcorpus", None) is None
                ):
                    raise ValueError(
                        "calibration_images_per_subcorpus and evaluation_images_per_subcorpus "
                        "must be provided together"
                    )
                calibration_images, evaluation_images = load_corpus_image_splits(
                    root,
                    calibration_images_per_subcorpus=args.calibration_images_per_subcorpus,
                    evaluation_images_per_subcorpus=args.evaluation_images_per_subcorpus,
                    hash_files=args.hash_files,
                )
                pilot_mode = "engineering_scratch_separated_calibration_evaluation"
            else:
                calibration_images = load_corpus_images(
                    root,
                    max_images_per_subcorpus=args.max_images_per_subcorpus,
                    hash_files=args.hash_files,
                )
                evaluation_images = calibration_images
                pilot_mode = "engineering_scratch_same_images_for_calibration_and_attack"

    calibration_arrays = [item.array for item in calibration_images]
    master_key = master_key_from_seed(args.master_key_seed)
    candidates = [float(value) for value in args.delta_embed_candidates]
    basis_modes = list(args.basis_modes)
    attacks = list(args.attacks)

    with timer.measure("delta_calibration"):
        delta_embed, delta_records = calibrate_delta_embed(
            calibration_arrays,
            master_key,
            candidates,
            min_psnr_db=args.min_psnr_db,
            max_clean_bit_error_rate=args.max_clean_bit_error_rate,
            basis_mode=args.calibration_basis_mode,
            delta_mode=args.delta_mode,
            auth_feature_step=args.auth_feature_step,
        )

    baseline_records = []
    for mode in basis_modes:
        with timer.measure(f"{mode}.embed_calibration"):
            calibration_embedded = [
                embed_image(
                    item.array,
                    master_key,
                    delta_embed,
                    basis_mode=mode,
                    delta_mode=args.delta_mode,
                    auth_feature_step=args.auth_feature_step,
                )
                for item in calibration_images
            ]
        calibration_watermarked_images = [item.watermarked for item in calibration_embedded]
        with timer.measure(f"{mode}.estimate_clean_reliability"):
            calibration_clean_reliability = estimate_clean_bit_reliability(
                calibration_watermarked_images,
                master_key,
                delta_embed,
                basis_mode=mode,
                delta_mode=args.delta_mode,
                auth_feature_step=args.auth_feature_step,
            )
        score_clean_error_rates = (
            np.asarray(calibration_clean_reliability["errorRates"], dtype=np.float64)
            if args.score_mode == "fisher_reliability"
            else None
        )
        with timer.measure(f"{mode}.tau_calibration"):
            tau, validation_fpr, clean_scores = calibrate_tau_from_clean_images(
                calibration_watermarked_images,
                master_key,
                delta_embed,
                alpha=args.target_false_positive_rate,
                basis_mode=mode,
                score_mode=args.score_mode,
                delta_mode=args.delta_mode,
                clean_error_rates=score_clean_error_rates,
                auth_feature_step=args.auth_feature_step,
            )
        with timer.measure(f"{mode}.embed_evaluation"):
            evaluation_embedded = [
                embed_image(
                    item.array,
                    master_key,
                    delta_embed,
                    basis_mode=mode,
                    delta_mode=args.delta_mode,
                    auth_feature_step=args.auth_feature_step,
                )
                for item in evaluation_images
            ]

        clean_records = []
        evaluation_clean_score_items = []
        clean_bit_mismatch_counts = np.zeros(8, dtype=np.int64)
        clean_bit_total_count = 0
        with timer.measure(f"{mode}.verify_clean_evaluation"):
            for index, embedded_item in enumerate(evaluation_embedded):
                source_item = evaluation_images[index]
                verified = verify_image(
                    embedded_item.watermarked,
                    master_key,
                    delta_embed,
                    tau=tau,
                    basis_mode=mode,
                    score_mode=args.score_mode,
                    delta_mode=args.delta_mode,
                    clean_error_rates=score_clean_error_rates,
                    auth_feature_step=args.auth_feature_step,
                )
                evaluation_clean_score_items.append(verified.scores.ravel())
                bit_mismatches = verified.extracted_bits != verified.expected_bits
                clean_bit_mismatch_counts += np.sum(bit_mismatches, axis=(0, 1))
                clean_bit_total_count += int(
                    bit_mismatches.shape[0] * bit_mismatches.shape[1]
                )
                clean_records.append(
                    {
                        "imageIndex": index,
                        "subcorpus": source_item.subcorpus,
                        "relativePath": source_item.relative_path,
                        "recordId": source_item.record_id,
                        "dataset": source_item.dataset,
                        "sourceImageId": source_item.source_image_id,
                        "sourceFileName": source_item.source_file_name,
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
            with timer.measure(f"{mode}.attack.{attack}"):
                for index, embedded_item in enumerate(evaluation_embedded):
                    source = evaluation_embedded[(index + 1) % len(evaluation_embedded)].watermarked
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
                        score_mode=args.score_mode,
                        delta_mode=args.delta_mode,
                        clean_error_rates=score_clean_error_rates,
                        auth_feature_step=args.auth_feature_step,
                    )
                    score_truth_pairs.append((verified.scores, truth))
                    metrics = block_metrics(verified.tamper_blocks, truth)
                    metrics["imageIndex"] = index
                    metrics["subcorpus"] = evaluation_images[index].subcorpus
                    metrics["relativePath"] = evaluation_images[index].relative_path
                    metrics["recordId"] = evaluation_images[index].record_id
                    metrics["dataset"] = evaluation_images[index].dataset
                    metrics["sourceImageId"] = evaluation_images[index].source_image_id
                    metrics["sourceFileName"] = evaluation_images[index].source_file_name
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
                "calibrationCleanBitMismatchCounts": calibration_clean_reliability[
                    "mismatchCounts"
                ],
                "calibrationCleanBitTotalCount": calibration_clean_reliability[
                    "totalCount"
                ],
                "calibrationCleanBitErrorRates": calibration_clean_reliability[
                    "errorRates"
                ],
                "cleanBitMismatchCounts": [
                    int(value) for value in clean_bit_mismatch_counts.tolist()
                ],
                "cleanBitTotalCount": int(clean_bit_total_count),
                "cleanBitErrorRates": [
                    float(value / clean_bit_total_count)
                    if clean_bit_total_count
                    else 0.0
                    for value in clean_bit_mismatch_counts.tolist()
                ],
                "calibrationCleanThresholdCurve": clean_threshold_curve(clean_scores),
                "evaluationCleanThresholdCurve": clean_threshold_curve(
                    np.concatenate(evaluation_clean_score_items)
                ),
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

    summary = {
        "schema": "blind-v2-real-image-scratch-pilot/v1",
        "pilotMode": pilot_mode,
        "promotionReady": False,
        "promotionBlocker": (
            "dataset source, version, license, citation, split, and preprocessing "
            "manifests must be frozen before public reporting"
        ),
        "inputMode": "manifest" if manifest_csv is not None else "directory",
        "manifestName": manifest_csv.name if manifest_csv is not None else None,
        "manifestStored": False,
        "datasetRootName": root.name if root is not None else None,
        "datasetRootStored": False,
        "dataCopied": False,
        "hashFiles": bool(args.hash_files),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "imageCount": len(evaluation_images),
        "calibrationImageCount": len(calibration_images),
        "evaluationImageCount": len(evaluation_images),
        "calibrationSubcorpora": summarize_subcorpora(calibration_images),
        "evaluationSubcorpora": summarize_subcorpora(evaluation_images),
        "masterKeyId": key_id(master_key),
        "masterKeyStored": False,
        "calibrationBasisMode": args.calibration_basis_mode,
        "scoreMode": args.score_mode,
        "deltaMode": args.delta_mode,
        "authFeatureStep": float(args.auth_feature_step),
        "basisModes": basis_modes,
        "attacks": attacks,
        "deltaEmbedCandidates": candidates,
        "selectedDeltaEmbed": float(delta_embed),
        "deltaCalibration": delta_records,
        "targetFalsePositiveRate": float(args.target_false_positive_rate),
        "minPsnrDb": float(args.min_psnr_db),
        "maxCleanBitErrorRate": float(args.max_clean_bit_error_rate),
        "timingSeconds": timer.rounded(),
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
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        help="Restricted public pilot manifest CSV to resolve instead of scanning --root.",
    )
    parser.add_argument(
        "--manifest-derived-root",
        type=Path,
        help=(
            "Local derived corpus root containing subcorpus/derived_artifact_id files. "
            "When provided, derived SHA-256 hashes from the manifest are verified."
        ),
    )
    parser.add_argument(
        "--dtd-root",
        type=Path,
        help="Official DTD images root used to resolve DTD source_image_id values.",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        help="Official COCO val2017 root used to resolve COCO source_file_name values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "_scratch" / "blind_v2_real_pilot.json",
    )
    parser.add_argument("--max-images-per-subcorpus", type=int, default=4)
    parser.add_argument("--calibration-images-per-subcorpus", type=int)
    parser.add_argument("--evaluation-images-per-subcorpus", type=int)
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
    parser.add_argument(
        "--auth-feature-step",
        type=float,
        default=DELTA_FEATURE,
        help=(
            "Quantization step for the canonical features authenticated by HMAC. "
            "Larger values increase clean robustness and may reduce attack sensitivity."
        ),
    )
    parser.add_argument("--target-false-positive-rate", type=float, default=0.0)
    parser.add_argument(
        "--score-mode",
        choices=("hamming", "fisher_weighted", "fisher_top4", "fisher_reliability"),
        default="hamming",
    )
    parser.add_argument(
        "--delta-mode",
        choices=("constant", "fisher_sqrt"),
        default="constant",
    )
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
