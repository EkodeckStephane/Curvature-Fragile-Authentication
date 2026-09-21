from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_blind_v2_real_pilot import (  # noqa: E402
    DEFAULT_BASIS_MODES,
    CorpusImage,
    load_corpus_image_splits,
    load_manifest_image_splits,
)
from src.blind_v2 import (  # noqa: E402
    DELTA_FEATURE,
    RESERVED_COORDS,
    AuthFeatureMode,
    BasisMode,
    DeltaMode,
    master_key_from_seed,
)
from src.blind_v2_image import calibrate_delta_embed, embed_image, verify_image  # noqa: E402


def accumulate_bit_rows(
    *,
    split_name: str,
    basis_mode: BasisMode,
    items: list[CorpusImage],
    watermarked_images: list[np.ndarray],
    master_key: bytes,
    delta_embed: float,
    delta_mode: DeltaMode,
    auth_feature_step: float,
    auth_feature_mode: AuthFeatureMode,
    auth_feature_band_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, np.ndarray | int]] = {}
    for item, watermarked in zip(items, watermarked_images):
        verified = verify_image(
            watermarked,
            master_key,
            delta_embed,
            basis_mode=basis_mode,
            delta_mode=delta_mode,
            auth_feature_step=auth_feature_step,
            auth_feature_mode=auth_feature_mode,
            auth_feature_band_count=auth_feature_band_count,
        )
        mismatches = verified.extracted_bits != verified.expected_bits
        entry = grouped.setdefault(
            item.subcorpus,
            {
                "mismatch_counts": np.zeros(len(RESERVED_COORDS), dtype=np.int64),
                "block_count": 0,
            },
        )
        entry["mismatch_counts"] = entry["mismatch_counts"] + np.sum(
            mismatches, axis=(0, 1)
        )
        entry["block_count"] = int(entry["block_count"]) + int(
            mismatches.shape[0] * mismatches.shape[1]
        )

    rows: list[dict[str, Any]] = []
    for subcorpus in sorted(grouped):
        mismatch_counts = np.asarray(grouped[subcorpus]["mismatch_counts"], dtype=np.int64)
        block_count = int(grouped[subcorpus]["block_count"])
        for bit_index, ((row, col), mismatch_count) in enumerate(
            zip(RESERVED_COORDS, mismatch_counts.tolist())
        ):
            rows.append(
                {
                    "split": split_name,
                    "basisMode": basis_mode,
                    "subcorpus": subcorpus,
                    "bitIndex": bit_index,
                    "reservedRow": row,
                    "reservedCol": col,
                    "radial": row + col,
                    "mismatchCount": int(mismatch_count),
                    "blockCount": block_count,
                    "errorRate": float(mismatch_count / block_count)
                    if block_count
                    else 0.0,
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest_csv:
        calibration_images, evaluation_images, input_mode = load_manifest_image_splits(
            args.manifest_csv,
            manifest_derived_root=args.manifest_derived_root,
            dtd_root=args.dtd_root,
            coco_root=args.coco_root,
            calibration_images_per_subcorpus=args.calibration_images_per_subcorpus,
            evaluation_images_per_subcorpus=args.evaluation_images_per_subcorpus,
            hash_files=args.hash_files,
        )
    elif args.root:
        calibration_images, evaluation_images = load_corpus_image_splits(
            args.root,
            calibration_images_per_subcorpus=args.calibration_images_per_subcorpus,
            evaluation_images_per_subcorpus=args.evaluation_images_per_subcorpus,
            hash_files=args.hash_files,
        )
        input_mode = "root_separated_calibration_evaluation"
    else:
        raise ValueError("either --root or --manifest-csv is required")

    master_key = master_key_from_seed(args.master_key_seed)
    candidates = [float(value) for value in args.delta_embed_candidates]
    calibration_arrays = [item.array for item in calibration_images]
    delta_embed, delta_records = calibrate_delta_embed(
        calibration_arrays,
        master_key,
        candidates,
        min_psnr_db=args.min_psnr_db,
        max_clean_bit_error_rate=args.max_clean_bit_error_rate,
        basis_mode=args.calibration_basis_mode,
        delta_mode=args.delta_mode,
        auth_feature_step=args.auth_feature_step,
        auth_feature_mode=args.auth_feature_mode,
        auth_feature_band_count=args.auth_feature_band_count,
    )

    rows: list[dict[str, Any]] = []
    for basis_mode in args.basis_modes:
        calibration_watermarked = [
            embed_image(
                item.array,
                master_key,
                delta_embed,
                basis_mode=basis_mode,
                delta_mode=args.delta_mode,
                auth_feature_step=args.auth_feature_step,
                auth_feature_mode=args.auth_feature_mode,
                auth_feature_band_count=args.auth_feature_band_count,
            ).watermarked
            for item in calibration_images
        ]
        evaluation_watermarked = [
            embed_image(
                item.array,
                master_key,
                delta_embed,
                basis_mode=basis_mode,
                delta_mode=args.delta_mode,
                auth_feature_step=args.auth_feature_step,
                auth_feature_mode=args.auth_feature_mode,
                auth_feature_band_count=args.auth_feature_band_count,
            ).watermarked
            for item in evaluation_images
        ]
        rows.extend(
            accumulate_bit_rows(
                split_name="calibration",
                basis_mode=basis_mode,
                items=calibration_images,
                watermarked_images=calibration_watermarked,
                master_key=master_key,
                delta_embed=delta_embed,
                delta_mode=args.delta_mode,
                auth_feature_step=args.auth_feature_step,
                auth_feature_mode=args.auth_feature_mode,
                auth_feature_band_count=args.auth_feature_band_count,
            )
        )
        rows.extend(
            accumulate_bit_rows(
                split_name="evaluation",
                basis_mode=basis_mode,
                items=evaluation_images,
                watermarked_images=evaluation_watermarked,
                master_key=master_key,
                delta_embed=delta_embed,
                delta_mode=args.delta_mode,
                auth_feature_step=args.auth_feature_step,
                auth_feature_mode=args.auth_feature_mode,
                auth_feature_band_count=args.auth_feature_band_count,
            )
        )

    manifest = {
        "schema": "clean-bit-stability-diagnostic/v1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputMode": input_mode,
        "dataCopied": False,
        "promotionReady": False,
        "masterKeyStored": False,
        "masterKeySeedStored": False,
        "calibrationImageCount": len(calibration_images),
        "evaluationImageCount": len(evaluation_images),
        "calibrationImagesPerSubcorpus": args.calibration_images_per_subcorpus,
        "evaluationImagesPerSubcorpus": args.evaluation_images_per_subcorpus,
        "basisModes": list(args.basis_modes),
        "deltaMode": args.delta_mode,
        "authFeatureStep": float(args.auth_feature_step),
        "authFeatureMode": args.auth_feature_mode,
        "authFeatureBandCount": int(args.auth_feature_band_count),
        "selectedDeltaEmbed": float(delta_embed),
        "deltaCalibration": delta_records,
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "basisMode",
        "subcorpus",
        "bitIndex",
        "reservedRow",
        "reservedCol",
        "radial",
        "mismatchCount",
        "blockCount",
        "errorRate",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure clean bit stability by basis, subcorpus and reserved DCT bit."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--manifest-derived-root", type=Path)
    parser.add_argument("--dtd-root", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--calibration-images-per-subcorpus", type=int, required=True)
    parser.add_argument("--evaluation-images-per-subcorpus", type=int, required=True)
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
    )
    parser.add_argument(
        "--auth-feature-mode",
        choices=("full", "sensitive_bands"),
        default="full",
    )
    parser.add_argument(
        "--auth-feature-band-count",
        type=int,
        default=2,
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
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
