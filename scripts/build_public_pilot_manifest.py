from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import PureWindowsPath, Path
from typing import Any


PUBLIC_STATUSES = {
    "identity-and-license-pinned",
    "identity-citation-and-reuse-statement-pinned",
}

PUBLIC_DATASETS = {"DTD textures", "MS-COCO val2017"}

PUBLIC_FIELDS = [
    "record_id",
    "dataset",
    "subcorpus",
    "source_split",
    "source_dataset_url",
    "source_version",
    "source_citation",
    "source_reuse_statement",
    "source_image_id",
    "source_file_name",
    "width",
    "height",
    "derived_artifact_id",
    "derived_sha256",
    "license_id",
    "license_name",
    "license_url",
    "coco_url",
    "provenance_status",
    "redistribution_policy",
]


PRIVATE_FIELD_NAMES = {
    "local_path",
    "source_path",
    "original_path",
    "source_id",
    "exact_id",
    "flickr_url",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def looks_like_private_path(value: str) -> bool:
    if not value:
        return False
    if re.search(r"(^|[,\s])([A-Za-z]:\\|[A-Za-z]:/)", value):
        return True
    if value.startswith("\\\\"):
        return True
    return False


def derived_artifact_id(local_path: str) -> str:
    normalized = local_path.replace("\\", "/")
    return PureWindowsPath(normalized).name


def redistribution_policy(dataset: str) -> str:
    if dataset == "MS-COCO val2017":
        return (
            "publish source identifiers, hashes and aggregate results; "
            "obtain image pixels from official COCO sources"
        )
    if dataset == "DTD textures":
        return (
            "publish source identifiers, hashes and aggregate results; "
            "obtain image pixels from official DTD sources"
        )
    return "private-only"


def public_record(row: dict[str, str]) -> dict[str, str]:
    dataset = row["dataset"]
    record = {
        "record_id": f"{row['subcorpus']}:{derived_artifact_id(row['local_path'])}",
        "dataset": dataset,
        "subcorpus": row["subcorpus"],
        "source_split": row["source_split"],
        "source_dataset_url": row["source_dataset_url"],
        "source_version": row["source_version"],
        "source_citation": row["source_citation"],
        "source_reuse_statement": row["source_reuse_statement"],
        "source_image_id": row["source_image_id"],
        "source_file_name": row["source_file_name"],
        "width": row["width"],
        "height": row["height"],
        "derived_artifact_id": derived_artifact_id(row["local_path"]),
        "derived_sha256": row["local_sha256"],
        "license_id": row["license_id"],
        "license_name": row["license_name"],
        "license_url": row["license_url"],
        "coco_url": row["coco_url"],
        "provenance_status": row["provenance_status"],
        "redistribution_policy": redistribution_policy(dataset),
    }
    validate_public_record(record)
    return record


def validate_public_record(record: dict[str, str]) -> None:
    forbidden = PRIVATE_FIELD_NAMES.intersection(record)
    if forbidden:
        raise ValueError(f"public record contains private fields: {sorted(forbidden)}")
    for key, value in record.items():
        if looks_like_private_path(value):
            raise ValueError(f"public record field {key!r} contains a private path")


def build_public_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    public_rows = [
        public_record(row)
        for row in rows
        if row.get("dataset") in PUBLIC_DATASETS
        and row.get("provenance_status") in PUBLIC_STATUSES
    ]
    return sorted(
        public_rows,
        key=lambda row: (
            row["dataset"],
            row["source_split"],
            row["source_image_id"],
            row["source_file_name"],
            row["derived_artifact_id"],
        ),
    )


def summary(rows: list[dict[str, str]], source_count: int) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for row in rows:
        info = datasets.setdefault(
            row["dataset"],
            {
                "count": 0,
                "subcorpus": row["subcorpus"],
                "sourceVersion": row["source_version"],
                "sourceDatasetUrl": row["source_dataset_url"],
                "splits": {},
                "licenses": {},
            },
        )
        info["count"] += 1
        split = row["source_split"]
        info["splits"][split] = info["splits"].get(split, 0) + 1
        license_key = row["license_name"] or row["source_reuse_statement"]
        if license_key:
            info["licenses"][license_key] = info["licenses"].get(license_key, 0) + 1
    return {
        "schema": "curvature-fragile-authentication/public-pilot-manifest/v1",
        "rowCount": len(rows),
        "sourceRowCount": source_count,
        "includedDatasets": datasets,
        "pixelDataIncluded": False,
        "privatePathsIncluded": False,
        "policy": (
            "The manifest publishes source identifiers, hashes, citations, "
            "license or reuse metadata, dimensions and split labels. It does "
            "not publish dataset image pixels, attacked images, watermarked "
            "images, or machine-local paths."
        ),
    }


def write_outputs(
    rows: list[dict[str, str]], output_csv: Path, output_json: Path, source_count: int
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = summary(rows, source_count)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public DTD+COCO pilot manifest from an enriched private map."
    )
    parser.add_argument("--enriched-map", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_csv(args.enriched_map)
    public_rows = build_public_rows(source_rows)
    write_outputs(public_rows, args.output_csv, args.output_json, len(source_rows))
    print(f"wrote {len(public_rows)} public pilot records -> {args.output_csv}")


if __name__ == "__main__":
    main()
