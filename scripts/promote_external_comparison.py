from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fisher_baseline(frsb: dict[str, Any]) -> dict[str, Any]:
    if frsb.get("schema") == "blind-v2-real-image-scratch-pilot/v1":
        return next(item for item in frsb["baselines"] if item["basisMode"] == "fisher")
    if frsb.get("schema") == "blind-v2-promoted-aggregate-result/v1":
        return {
            "basisMode": "fisher",
            "meanCleanPsnrDb": next(
                item for item in frsb["modeCleanMetrics"] if item["basisMode"] == "fisher"
            )["meanCleanPsnrDb"],
            "cleanFlaggedBlockCount": next(
                item for item in frsb["modeCleanMetrics"] if item["basisMode"] == "fisher"
            )["cleanFlaggedBlockCount"],
            "attacks": [
                {
                    "attack": row["attack"],
                    "meanBlockF1": row["meanBlockF1"],
                    "meanBlockIoU": row["meanBlockIoU"],
                    "meanBlockFpr": row["meanBlockFpr"],
                    "meanBlockFnr": row["meanBlockFnr"],
                }
                for row in frsb["attackMeanBlockMetrics"]
                if row["basisMode"] == "fisher"
            ],
        }
    raise ValueError(f"unsupported FRSB schema: {frsb.get('schema')}")


def external_attack_rows(external: dict[str, Any]) -> list[dict[str, Any]]:
    if external.get("schema") != "external-lsb-hmac-baseline-scratch/v1":
        raise ValueError(f"unsupported external schema: {external.get('schema')}")
    return external["attacksDetailed"]


def comparison_rows(frsb: dict[str, Any], external: dict[str, Any]) -> list[dict[str, Any]]:
    fisher = fisher_baseline(frsb)
    external_by_attack = {item["attack"]: item for item in external_attack_rows(external)}
    rows: list[dict[str, Any]] = []
    for attack in fisher["attacks"]:
        name = attack["attack"]
        if name not in external_by_attack:
            continue
        ext = external_by_attack[name]
        rows.append(
            {
                "attack": name,
                "fisherMeanBlockF1": round(float(attack["meanBlockF1"]), 6),
                "externalMeanBlockF1": round(float(ext["meanBlockF1"]), 6),
                "deltaFisherMinusExternalF1": round(
                    float(attack["meanBlockF1"]) - float(ext["meanBlockF1"]), 6
                ),
                "fisherMeanBlockIoU": round(float(attack["meanBlockIoU"]), 6),
                "externalMeanBlockIoU": round(float(ext["meanBlockIoU"]), 6),
                "fisherMeanBlockFpr": round(float(attack["meanBlockFpr"]), 6),
                "externalMeanBlockFpr": round(float(ext["meanBlockFpr"]), 6),
                "fisherMeanBlockFnr": round(float(attack["meanBlockFnr"]), 6),
                "externalMeanBlockFnr": round(float(ext["meanBlockFnr"]), 6),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def promote(frsb_path: Path, external_path: Path, output_dir: Path) -> None:
    frsb = load_json(frsb_path)
    external = load_json(external_path)
    fisher = fisher_baseline(frsb)
    rows = comparison_rows(frsb, external)
    clean = {
        "fisherMeanCleanPsnrDb": round(float(fisher["meanCleanPsnrDb"]), 6),
        "externalMeanCleanPsnrDb": round(float(external["meanCleanPsnrDb"]), 6),
        "deltaFisherMinusExternalPsnrDb": round(
            float(fisher["meanCleanPsnrDb"]) - float(external["meanCleanPsnrDb"]),
            6,
        ),
        "fisherCleanFlaggedBlockCount": int(fisher["cleanFlaggedBlockCount"]),
        "externalCleanFlaggedBlockCount": int(external["cleanFlaggedBlockCount"]),
    }
    summary = {
        "schema": "frsb-external-comparison/v1",
        "status": "promoted-aggregate-result",
        "comparison": "FRSB Fisher basis versus external block-wise LSB-HMAC fragile watermark baseline",
        "safety": {
            "pixelsStored": False,
            "localPathsStored": False,
            "perImageFileNamesStored": False,
            "masterKeyStored": False,
        },
        "dataset": {
            "scope": "five-subcorpus local multicorpus expanded validation",
            "calibrationImages": frsb["calibrationImageCount"],
            "evaluationImages": frsb["evaluationImageCount"],
            "subcorpora": list(frsb["evaluationSubcorpora"].keys()),
        },
        "externalBaseline": external["literatureAnchor"],
        "cleanComparison": clean,
        "attackComparison": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_csv(output_dir / "tables" / "attack_comparison.csv", rows)
    write_csv(output_dir / "tables" / "clean_comparison.csv", [clean])
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# FRSB external LSB-HMAC comparison v1",
                "",
                "Sanitized aggregate comparison between the FRSB Fisher basis and",
                "a classical block-wise LSB-HMAC fragile watermark baseline.",
                "The directory stores no pixels, no watermarked images, no attacked",
                "images, no local paths, no per-image filenames and no key material.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote an aggregate FRSB-vs-external-baseline comparison."
    )
    parser.add_argument("--frsb", dest="frsb_path", type=Path, required=True)
    parser.add_argument("--external", dest="external_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    promote(**vars(parse_args()))


if __name__ == "__main__":
    main()
