from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ATTACK_ORDER = [
    "center_mean",
    "copy_move",
    "constant_average_block",
    "inter_block_substitution",
    "non_aligned_patch",
    "channel_jpeg_q90",
    "channel_blur_sigma0_6",
    "channel_resize_roundtrip",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def load_scratch(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != "blind-v2-real-image-scratch-pilot/v1":
        raise ValueError("input is not a Blind V2 scratch pilot")
    return data


def mode_clean_metrics(scratch: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in scratch["baselines"]:
        rows.append(
            {
                "basisMode": item["basisMode"],
                "tau": round_float(item["tau"]),
                "validationFalsePositiveRate": round_float(
                    item["validationFalsePositiveRate"]
                ),
                "cleanFlaggedBlockCount": item["cleanFlaggedBlockCount"],
                "meanCleanPsnrDb": round_float(item["meanCleanPsnrDb"]),
                "minCleanPsnrDb": round_float(item["minCleanPsnrDb"]),
                "maxCleanBitErrorRate": round_float(item["maxCleanBitErrorRate"]),
            }
        )
    return rows


def attack_mean_block_f1(scratch: dict[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for baseline in scratch["baselines"]:
        basis = baseline["basisMode"]
        output[basis] = {}
        for attack in baseline["attacks"]:
            f1_values = [image["f1"] for image in attack["images"]]
            output[basis][attack["attack"]] = round(sum(f1_values) / len(f1_values), 6)
    return output


def attack_mean_block_metrics(scratch: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in scratch["baselines"]:
        basis = baseline["basisMode"]
        for attack in baseline["attacks"]:
            images = attack["images"]
            rows.append(
                {
                    "basisMode": basis,
                    "attack": attack["attack"],
                    "meanBlockF1": round_float(
                        sum(image["f1"] for image in images) / len(images)
                    ),
                    "meanBlockIoU": round_float(
                        sum(image["iou"] for image in images) / len(images)
                    ),
                    "meanBlockFpr": round_float(
                        sum(image["fpr"] for image in images) / len(images)
                    ),
                    "meanBlockFnr": round_float(
                        sum(image["fnr"] for image in images) / len(images)
                    ),
                    "meanBlockRecall": round_float(
                        sum(image["recall"] for image in images) / len(images)
                    ),
                }
            )
    return rows


def failure_analysis(scratch: dict[str, Any], reference_mode: str = "fisher") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference = next(
        item for item in scratch["baselines"] if item["basisMode"] == reference_mode
    )
    for attack in reference["attacks"]:
        images = attack["images"]
        imperfect = [item for item in images if float(item["f1"]) < 1.0]
        worst = min(images, key=lambda item: float(item["f1"]))
        subcorpus_counts: dict[str, int] = {}
        for item in imperfect:
            subcorpus = item["subcorpus"]
            subcorpus_counts[subcorpus] = subcorpus_counts.get(subcorpus, 0) + 1
        records.append(
            {
                "basisMode": reference_mode,
                "attack": attack["attack"],
                "evaluatedImages": len(images),
                "imperfectImageCount": len(imperfect),
                "perfectImageCount": len(images) - len(imperfect),
                "minBlockF1": round(float(worst["f1"]), 6),
                "maxFalsePositiveBlocks": max(int(item["fp"]) for item in images),
                "maxFalseNegativeBlocks": max(int(item["fn"]) for item in images),
                "imperfectSubcorpora": subcorpus_counts,
            }
        )
    return records


def paired_deltas(paired_csv: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(paired_csv):
        if "baseline" not in row:
            comparison = row["comparison"]
            if not comparison.startswith("attack:") or row["metric"] != "f1":
                continue
            rows.append(
                {
                    "baseline": row["basisMode"],
                    "attack": comparison.split(":", 1)[1],
                    "deltaFisherMinusBaseline": round(
                        float(row["meanDeltaFisherMinusBaseline"]), 6
                    ),
                    "ci95": [
                        round(float(row["deltaCi95Low"]), 6),
                        round(float(row["deltaCi95High"]), 6),
                    ],
                    "pairedSignFlipP": round(float(row["pairedSignFlipP"]), 6),
                    "favorablePairs": int(row["favorablePairs"]),
                    "unfavorablePairs": int(row["unfavorablePairs"]),
                    "ties": int(row["ties"]),
                }
            )
            continue
        rows.append(
            {
                "baseline": row["baseline"],
                "attack": row["attack"],
                "deltaFisherMinusBaseline": round(float(row["deltaFisherMinusBaseline"]), 6),
                "ci95": [
                    round(float(row["ci95Low"]), 6),
                    round(float(row["ci95High"]), 6),
                ],
                "pairedSignFlipP": round(float(row["pairedSignFlipP"]), 6),
                "favorablePairs": int(row["favorablePairs"]),
                "unfavorablePairs": int(row["unfavorablePairs"]),
                "ties": int(row["ties"]),
            }
        )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def promote(
    scratch_path: Path,
    paired_csv: Path,
    output_dir: Path,
    protocol_id: str,
) -> None:
    scratch = load_scratch(scratch_path)
    clean = mode_clean_metrics(scratch)
    attack_f1 = attack_mean_block_f1(scratch)
    attack_metrics = attack_mean_block_metrics(scratch)
    paired = paired_deltas(paired_csv)
    failure = failure_analysis(scratch)
    attack_order = [
        attack for attack in DEFAULT_ATTACK_ORDER if attack in scratch["attacks"]
    ] + [
        attack for attack in scratch["attacks"] if attack not in DEFAULT_ATTACK_ORDER
    ]

    summary = {
        "schema": "blind-v2-promoted-aggregate-result/v1",
        "protocolId": protocol_id,
        "status": "promoted-aggregate-result",
        "source": {
            "scratchSchema": scratch["schema"],
            "datasetRootName": scratch["datasetRootName"],
            "inputMode": scratch["inputMode"],
            "promotionNote": "Sanitized aggregate extracted from a local multicorpus run; no pixels, local paths, per-image filenames, or key material are stored.",
        },
        "safety": {
            "dataCopied": scratch["dataCopied"],
            "pixelsStored": False,
            "localPathsStored": False,
            "perImageFileNamesStored": False,
            "masterKeyStored": scratch["masterKeyStored"],
            "machineLocalRootsStored": scratch["datasetRootStored"],
        },
        "dataset": {
            "scope": "five-subcorpus local multicorpus validation",
            "subcorpora": list(scratch["evaluationSubcorpora"].keys()),
            "calibrationImages": scratch["calibrationImageCount"],
            "evaluationImages": scratch["evaluationImageCount"],
            "calibrationImagesBySubcorpus": scratch["calibrationSubcorpora"],
            "evaluationImagesBySubcorpus": scratch["evaluationSubcorpora"],
            "rightsPolicy": "Dataset pixels remain under their original terms; the public result stores aggregate metrics only.",
        },
        "candidate": {
            "basisModes": scratch["basisModes"],
            "attacks": scratch["attacks"],
            "scoreMode": scratch["scoreMode"],
            "deltaMode": scratch["deltaMode"],
            "selectedDeltaEmbed": scratch["selectedDeltaEmbed"],
            "deltaEmbedCandidates": scratch["deltaEmbedCandidates"],
            "targetFalsePositiveRate": scratch["targetFalsePositiveRate"],
            "thresholdScope": scratch["thresholdScope"],
            "authFeatureStep": scratch["authFeatureStep"],
            "authFeatureMode": scratch["authFeatureMode"],
            "authFeatureBandCount": scratch["authFeatureBandCount"],
        },
        "modeCleanMetrics": clean,
        "attackMeanBlockF1": attack_f1,
        "attackMeanBlockMetrics": attack_metrics,
        "pairedAttackF1Deltas": paired,
        "failureAnalysis": failure,
        "interpretation": {
            "primaryPattern": "Fisher has the highest mean block-level F1 among the five declared basis modes on the four content-tamper attacks and remains near the top on the high-integrity channel events.",
            "baselineBreadth": "The comparison uses four matched baselines: reverse Fisher/cost, random, fixed, and identity-cost.",
            "datasetBreadth": "The evaluation covers five subcorpora with 40 calibration and 40 evaluation images.",
            "quality": "Fisher clean PSNR is comparable to fixed, random and reverse Fisher/cost, and higher than identity-cost under the selected operating point.",
            "highIntegrityReading": "Benign transformations are treated as operational channel events that should trigger reauthentication or be evaluated in a separate semi-fragile profile, not as silently tolerated modifications in the fragile profile.",
        },
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "tables" / "clean_metrics.csv",
        clean,
        [
            "basisMode",
            "tau",
            "validationFalsePositiveRate",
            "cleanFlaggedBlockCount",
            "meanCleanPsnrDb",
            "minCleanPsnrDb",
            "maxCleanBitErrorRate",
        ],
    )
    attack_rows = [
        {"basisMode": basis, **{attack: values[attack] for attack in attack_order}}
        for basis, values in attack_f1.items()
    ]
    write_csv(output_dir / "tables" / "attack_f1.csv", attack_rows, ["basisMode", *attack_order])
    write_csv(
        output_dir / "tables" / "attack_metrics.csv",
        attack_metrics,
        [
            "basisMode",
            "attack",
            "meanBlockF1",
            "meanBlockIoU",
            "meanBlockFpr",
            "meanBlockFnr",
            "meanBlockRecall",
        ],
    )
    paired_rows = [
        {
            "baseline": item["baseline"],
            "attack": item["attack"],
            "deltaFisherMinusBaseline": item["deltaFisherMinusBaseline"],
            "ci95Low": item["ci95"][0],
            "ci95High": item["ci95"][1],
            "pairedSignFlipP": item["pairedSignFlipP"],
            "favorablePairs": item["favorablePairs"],
            "unfavorablePairs": item["unfavorablePairs"],
            "ties": item["ties"],
        }
        for item in paired
    ]
    write_csv(
        output_dir / "tables" / "paired_attack_f1_deltas.csv",
        paired_rows,
        [
            "baseline",
            "attack",
            "deltaFisherMinusBaseline",
            "ci95Low",
            "ci95High",
            "pairedSignFlipP",
            "favorablePairs",
            "unfavorablePairs",
            "ties",
        ],
    )
    failure_rows = [
        {
            **{key: value for key, value in item.items() if key != "imperfectSubcorpora"},
            "imperfectSubcorpora": json.dumps(
                item["imperfectSubcorpora"], sort_keys=True, separators=(",", ":")
            ),
        }
        for item in failure
    ]
    write_csv(
        output_dir / "tables" / "failure_analysis.csv",
        failure_rows,
        [
            "basisMode",
            "attack",
            "evaluatedImages",
            "imperfectImageCount",
            "perfectImageCount",
            "minBlockF1",
            "maxFalsePositiveBlocks",
            "maxFalseNegativeBlocks",
            "imperfectSubcorpora",
        ],
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Blind V2 multicorpus candidate v2",
                "",
                "Sanitized aggregate result for the FRSB detector on the local",
                "five-subcorpus multicorpus validation. The directory stores no",
                "pixels, no watermarked images, no attacked images, no local paths,",
                "and no key material.",
                "",
                "The evaluated subcorpora are listed in `summary.json`; each dataset",
                "retains its original licence and terms of use. Public reproduction",
                "requires access to the corresponding source datasets or to a local",
                "derived corpus with the same manifest discipline.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a local multicorpus scratch run to a sanitized aggregate result."
    )
    parser.add_argument("--scratch", dest="scratch_path", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    return parser.parse_args()


def main() -> None:
    promote(**vars(parse_args()))


if __name__ == "__main__":
    main()
