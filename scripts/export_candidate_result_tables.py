from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ATTACK_ORDER = [
    "center_mean",
    "copy_move",
    "constant_average_block",
    "inter_block_substitution",
]


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != "blind-v2-promoted-aggregate-result/v1":
        raise ValueError("input is not a promoted Blind V2 aggregate result")
    return data


def write_attack_f1(summary: dict[str, Any], output: Path) -> None:
    rows = []
    for basis_mode, attacks in summary["attackMeanBlockF1"].items():
        row = {"basisMode": basis_mode}
        for attack in ATTACK_ORDER:
            row[attack] = attacks[attack]
        rows.append(row)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["basisMode", *ATTACK_ORDER])
        writer.writeheader()
        writer.writerows(rows)


def write_clean_metrics(summary: dict[str, Any], output: Path) -> None:
    fieldnames = [
        "basisMode",
        "tau",
        "validationFalsePositiveRate",
        "cleanFlaggedBlockCount",
        "meanCleanPsnrDb",
        "maxCleanBitErrorRate",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["modeCleanMetrics"])


def write_paired_deltas(summary: dict[str, Any], output: Path) -> None:
    fieldnames = [
        "baseline",
        "attack",
        "deltaFisherMinusBaseline",
        "ci95Low",
        "ci95High",
        "pairedSignFlipP",
        "favorablePairs",
        "unfavorablePairs",
        "ties",
    ]
    rows = []
    for item in summary["pairedAttackF1Deltas"]:
        rows.append(
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
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(input_path: Path, output_dir: Path) -> dict[str, str]:
    summary = load_summary(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "attack_f1": output_dir / "attack_f1.csv",
        "clean_metrics": output_dir / "clean_metrics.csv",
        "paired_attack_f1_deltas": output_dir / "paired_attack_f1_deltas.csv",
    }
    write_attack_f1(summary, paths["attack_f1"])
    write_clean_metrics(summary, paths["clean_metrics"])
    write_paired_deltas(summary, paths["paired_attack_f1_deltas"])
    return {name: path.as_posix() for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export article-ready CSV tables from a promoted aggregate result."
    )
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = export_tables(**vars(parse_args()))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
