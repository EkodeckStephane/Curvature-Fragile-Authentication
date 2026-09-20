from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "blind-v2-real-image-scratch-pilot/v1":
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    return manifest


def mode_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline in manifest["baselines"]:
        rows.append(
            {
                "basisMode": baseline["basisMode"],
                "tau": baseline["tau"],
                "validationFalsePositiveRate": baseline["validationFalsePositiveRate"],
                "cleanFlaggedBlockCount": baseline["cleanFlaggedBlockCount"],
                "meanCleanPsnrDb": baseline["meanCleanPsnrDb"],
                "minCleanPsnrDb": baseline["minCleanPsnrDb"],
                "maxCleanBitErrorRate": baseline["maxCleanBitErrorRate"],
            }
        )
    return rows


def attack_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline in manifest["baselines"]:
        for attack in baseline.get("attacks", []):
            rows.append(
                {
                    "basisMode": baseline["basisMode"],
                    "attack": attack["attack"],
                    "meanBlockF1": attack["meanBlockF1"],
                    "meanBlockIoU": attack["meanBlockIoU"],
                    "meanBlockFpr": attack["meanBlockFpr"],
                    "meanBlockFnr": attack["meanBlockFnr"],
                    "meanBlockRecall": 1.0 - attack["meanBlockFnr"],
                }
            )
    return rows


def timing_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"phase": phase, "seconds": seconds}
        for phase, seconds in sorted(
            manifest.get("timingSeconds", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def clean_bit_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline in manifest["baselines"]:
        counts = baseline.get("cleanBitMismatchCounts")
        rates = baseline.get("cleanBitErrorRates")
        total = baseline.get("cleanBitTotalCount")
        if counts is None or rates is None or total is None:
            continue
        for bit_index, (count, rate) in enumerate(zip(counts, rates)):
            rows.append(
                {
                    "basisMode": baseline["basisMode"],
                    "bitIndex": bit_index,
                    "mismatchCount": count,
                    "totalCount": total,
                    "errorRate": rate,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(
    manifest: dict[str, Any],
    modes: list[dict[str, Any]],
    attacks: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    clean_bits: list[dict[str, Any]],
) -> str:
    lines = [
        "# Real-image scratch pilot summary",
        "",
        f"- Schema: `{manifest['schema']}`",
        f"- Pilot mode: `{manifest['pilotMode']}`",
        f"- Promotion ready: `{manifest['promotionReady']}`",
        f"- Dataset root name: `{manifest['datasetRootName']}`",
        f"- Calibration images: {manifest.get('calibrationImageCount')}",
        f"- Evaluation images: {manifest.get('evaluationImageCount')}",
        f"- Selected Delta_embed: {manifest.get('selectedDeltaEmbed')}",
        f"- Target FPR alpha: {manifest.get('targetFalsePositiveRate')}",
        f"- Data copied: `{manifest.get('dataCopied')}`",
        f"- Master key stored: `{manifest.get('masterKeyStored')}`",
        "",
        "## Mode clean metrics",
        "",
        "| Basis | tau | validation FPR | clean flags | mean PSNR | max clean BER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in modes:
        lines.append(
            "| {basisMode} | {tau:.3f} | {validationFalsePositiveRate:.6f} | "
            "{cleanFlaggedBlockCount} | {meanCleanPsnrDb:.3f} | "
            "{maxCleanBitErrorRate:.6f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Attack metrics",
            "",
            "| Basis | Attack | F1 | IoU | recall | FPR | FNR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in attacks:
        lines.append(
            "| {basisMode} | {attack} | {meanBlockF1:.3f} | {meanBlockIoU:.3f} | "
            "{meanBlockRecall:.3f} | {meanBlockFpr:.6f} | {meanBlockFnr:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Top timing phases",
            "",
            "| Phase | seconds |",
            "|---|---:|",
        ]
    )
    for row in timings[:12]:
        lines.append("| {phase} | {seconds:.6f} |".format(**row))

    if clean_bits:
        lines.extend(
            [
                "",
                "## Clean bit reliability",
                "",
                "| Basis | bit | mismatches | total | error rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in clean_bits:
            lines.append(
                "| {basisMode} | {bitIndex} | {mismatchCount} | {totalCount} | "
                "{errorRate:.6f} |".format(**row)
            )

    lines.extend(
        [
            "",
            "## Interpretation status",
            "",
            "This is a private scratch summary. It supports engineering decisions. "
            "Public reporting requires completed dataset provenance, license, "
            "citation and split manifests.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(input_path: Path, output_dir: Path) -> dict[str, Path]:
    manifest = load_manifest(input_path)
    modes = mode_rows(manifest)
    attacks = attack_rows(manifest)
    timings = timing_rows(manifest)
    clean_bits = clean_bit_rows(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "modes": output_dir / "modes.csv",
        "attacks": output_dir / "attacks.csv",
        "timings": output_dir / "timings.csv",
        "clean_bits": output_dir / "clean_bits.csv",
        "summary": output_dir / "summary.md",
    }
    write_csv(paths["modes"], modes)
    write_csv(paths["attacks"], attacks)
    write_csv(paths["timings"], timings)
    write_csv(paths["clean_bits"], clean_bits)
    paths["summary"].write_text(
        markdown_summary(manifest, modes, attacks, timings, clean_bits),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a private real-image scratch pilot manifest."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = summarize(args.input, args.output_dir)
    print(
        "wrote real-pilot scratch summary: "
        + ", ".join(f"{name}={path}" for name, path in paths.items())
    )


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
