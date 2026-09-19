from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.blind_v2 import key_id, master_key_from_seed
from src.blind_v2_image import (
    apply_synthetic_attack,
    block_metrics,
    calibrate_delta_embed,
    calibrate_tau_from_clean_images,
    embed_image,
    verify_image,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def synthetic_images(
    seed: int, count: int, height: int, width: int
) -> list[np.ndarray]:
    """Generate deterministic, mildly textured luminance images."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    images = []
    for index in range(count):
        phase = 0.37 * index
        gradient = 70.0 + 80.0 * xx / max(width - 1, 1) + 45.0 * yy / max(height - 1, 1)
        texture = 18.0 * np.sin(2.0 * np.pi * (xx / 17.0 + phase))
        texture += 12.0 * np.cos(2.0 * np.pi * (yy / 23.0 - phase))
        checker = 8.0 * (((xx // 8 + yy // 8 + index) % 2) * 2 - 1)
        noise = rng.normal(0.0, 3.0, size=(height, width))
        images.append(np.clip(gradient + texture + checker + noise, 0.0, 255.0))
    return images


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    master_key = master_key_from_seed(int(config["masterKeySeed"]))
    images = synthetic_images(
        seed=int(config["seed"]),
        count=int(config["imageCount"]),
        height=int(config["height"]),
        width=int(config["width"]),
    )
    candidates = [float(value) for value in config["deltaEmbedCandidates"]]
    basis_mode = config["basisMode"]
    baseline_modes = list(config.get("baselineModes", [basis_mode]))
    attacks = list(config.get("attacks", []))

    delta_embed, delta_records = calibrate_delta_embed(
        images,
        master_key,
        candidates,
        min_psnr_db=float(config["minPsnrDb"]),
        max_clean_bit_error_rate=float(config["maxCleanBitErrorRate"]),
        basis_mode=basis_mode,
    )

    embedded_results = [
        embed_image(image, master_key, delta_embed, basis_mode=basis_mode)
        for image in images
    ]
    watermarked_images = [result.watermarked for result in embedded_results]
    tau, validation_fpr, clean_scores = calibrate_tau_from_clean_images(
        watermarked_images,
        master_key,
        delta_embed,
        alpha=float(config["targetFalsePositiveRate"]),
        basis_mode=basis_mode,
    )

    verification_records = []
    for index, embedded in enumerate(embedded_results):
        verified = verify_image(
            embedded.watermarked,
            master_key,
            delta_embed,
            tau=tau,
            basis_mode=basis_mode,
        )
        verification_records.append(
            {
                "imageIndex": index,
                "shape": list(embedded.watermarked.shape),
                "psnrDb": embedded.psnr_db,
                "payloadBpp": embedded.payload_bpp,
                "cleanBitErrorRate": verified.clean_bit_error_rate,
                "maxBlockScore": float(np.max(verified.scores)),
                "flaggedBlockCount": int(np.sum(verified.tamper_blocks)),
                "blockCount": int(verified.scores.size),
            }
        )

    baseline_records = []
    for mode in baseline_modes:
        mode_embedded = [
            embed_image(image, master_key, delta_embed, basis_mode=mode) for image in images
        ]
        mode_watermarked = [result.watermarked for result in mode_embedded]
        mode_tau, mode_fpr, mode_clean_scores = calibrate_tau_from_clean_images(
            mode_watermarked,
            master_key,
            delta_embed,
            alpha=float(config["targetFalsePositiveRate"]),
            basis_mode=mode,
        )
        clean_records = []
        for index, embedded in enumerate(mode_embedded):
            verified = verify_image(
                embedded.watermarked,
                master_key,
                delta_embed,
                tau=mode_tau,
                basis_mode=mode,
            )
            clean_records.append(
                {
                    "imageIndex": index,
                    "psnrDb": embedded.psnr_db,
                    "cleanBitErrorRate": verified.clean_bit_error_rate,
                    "flaggedBlockCount": int(np.sum(verified.tamper_blocks)),
                    "blockCount": int(verified.scores.size),
                }
            )

        attack_records = []
        for attack in attacks:
            metric_items = []
            for index, embedded in enumerate(mode_embedded):
                source = mode_embedded[(index + 1) % len(mode_embedded)].watermarked
                attacked, truth = apply_synthetic_attack(
                    embedded.watermarked,
                    attack,
                    source=source,
                )
                verified = verify_image(
                    attacked,
                    master_key,
                    delta_embed,
                    tau=mode_tau,
                    basis_mode=mode,
                )
                metrics = block_metrics(verified.tamper_blocks, truth)
                metrics["imageIndex"] = index
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
                    "images": metric_items,
                }
            )

        baseline_records.append(
            {
                "basisMode": mode,
                "tau": mode_tau,
                "validationFalsePositiveRate": mode_fpr,
                "cleanScoreHistogram": {
                    f"{k}/8": int(np.sum(mode_clean_scores == k / 8.0)) for k in range(9)
                },
                "clean": clean_records,
                "attacks": attack_records,
                "meanCleanPsnrDb": float(np.mean([item["psnrDb"] for item in clean_records])),
                "maxCleanBitErrorRate": float(
                    np.max([item["cleanBitErrorRate"] for item in clean_records])
                ),
                "cleanFlaggedBlockCount": int(
                    np.sum([item["flaggedBlockCount"] for item in clean_records])
                ),
            }
        )

    summary = {
        "schema": "blind-v2-synthetic-image-gate/v2",
        "config": str(config_path.as_posix()),
        "configSha256": sha256_file(config_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": int(config["seed"]),
        "basisMode": basis_mode,
        "imageCount": len(images),
        "masterKeyId": key_id(master_key),
        "masterKeyStored": False,
        "selectedDeltaEmbed": delta_embed,
        "selectedTau": tau,
        "validationFalsePositiveRate": validation_fpr,
        "cleanScoreHistogram": {
            f"{k}/8": int(np.sum(clean_scores == k / 8.0)) for k in range(9)
        },
        "deltaCalibration": delta_records,
        "images": verification_records,
        "baselines": baseline_records,
        "allAccepted": bool(
            validation_fpr <= float(config["targetFalsePositiveRate"])
            and all(item["cleanBitErrorRate"] <= float(config["maxCleanBitErrorRate"]) for item in verification_records)
            and all(item["psnrDb"] >= float(config["minPsnrDb"]) for item in verification_records)
            and all(item["flaggedBlockCount"] == 0 for item in verification_records)
            and all(
                item["validationFalsePositiveRate"] <= float(config["targetFalsePositiveRate"])
                and item["maxCleanBitErrorRate"] <= float(config["maxCleanBitErrorRate"])
                and item["meanCleanPsnrDb"] >= float(config["minPsnrDb"])
                and item["cleanFlaggedBlockCount"] == 0
                for item in baseline_records
            )
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic image gate for the V2 blind detector."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "blind_v2_synthetic_image_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "blind_v2_synthetic_image_gate" / "summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.config, args.output)
    if not summary["allAccepted"]:
        raise SystemExit(1)
    print(
        f"accepted blind V2 synthetic image gate: "
        f"delta={summary['selectedDeltaEmbed']} tau={summary['selectedTau']} "
        f"images={summary['imageCount']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
