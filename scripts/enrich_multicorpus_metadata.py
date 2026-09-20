from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DTD_SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"
DTD_VERSION = "dtd-r1.0.1"
DTD_CITATION = (
    "M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi, "
    '"Describing Textures in the Wild," CVPR, 2014.'
)
DTD_REUSE_STATEMENT = "made available to the computer vision community for research purposes"

COCO_SOURCE_URL = "https://cocodataset.org/"
COCO_VERSION = "val2017"
COCO_CITATION = (
    "T.-Y. Lin et al., "
    '"Microsoft COCO: Common Objects in Context," ECCV, 2014.'
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_coco_metadata(annotation_json: Path) -> dict[str, Any]:
    with annotation_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    licenses = {str(item["id"]): item for item in payload.get("licenses", [])}
    images = {item["file_name"]: item for item in payload.get("images", [])}
    return {"licenses": licenses, "images": images}


def coco_file_name(row: dict[str, str]) -> str:
    original_path = row.get("original_path", "")
    if original_path:
        return Path(original_path).name
    source_id = row.get("source_id", "")
    return Path(source_id).name


def dtd_source_id(row: dict[str, str]) -> str:
    original_path = row.get("original_path", "").replace("\\", "/")
    marker = "/dtd/images/"
    if marker in original_path:
        return original_path.split(marker, 1)[1]
    source_id = row.get("source_id", "").replace("\\", "/")
    if "processed/dtd/" in source_id:
        return source_id.split("processed/dtd/", 1)[1]
    return source_id


def blank_metadata() -> dict[str, str]:
    return {
        "source_dataset_url": "",
        "source_version": "",
        "source_citation": "",
        "source_reuse_statement": "",
        "source_image_id": "",
        "source_file_name": "",
        "license_id": "",
        "license_name": "",
        "license_url": "",
        "coco_url": "",
        "flickr_url": "",
        "provenance_status": "identity-pending",
    }


def enrich_row(
    row: dict[str, str], coco_metadata: dict[str, Any] | None
) -> dict[str, str]:
    enriched = dict(row)
    metadata = blank_metadata()
    dataset = row.get("dataset", "")

    if row.get("matched") == "true":
        metadata["provenance_status"] = "identity-pinned; license-pending"

    if dataset == "MS-COCO val2017" and coco_metadata is not None:
        file_name = coco_file_name(row)
        image = coco_metadata["images"].get(file_name)
        metadata.update(
            {
                "source_dataset_url": COCO_SOURCE_URL,
                "source_version": COCO_VERSION,
                "source_citation": COCO_CITATION,
                "source_file_name": file_name,
            }
        )
        if image is not None:
            license_id = str(image.get("license", ""))
            license_info = coco_metadata["licenses"].get(license_id, {})
            metadata.update(
                {
                    "source_image_id": str(image.get("id", "")),
                    "license_id": license_id,
                    "license_name": str(license_info.get("name", "")),
                    "license_url": str(license_info.get("url", "")),
                    "coco_url": str(image.get("coco_url", "")),
                    "flickr_url": str(image.get("flickr_url", "")),
                    "provenance_status": "identity-and-license-pinned",
                }
            )

    if dataset == "DTD textures":
        metadata.update(
            {
                "source_dataset_url": DTD_SOURCE_URL,
                "source_version": DTD_VERSION,
                "source_citation": DTD_CITATION,
                "source_reuse_statement": DTD_REUSE_STATEMENT,
                "source_image_id": dtd_source_id(row),
                "provenance_status": "identity-citation-and-reuse-statement-pinned",
            }
        )

    enriched.update(metadata)
    return enriched


def enrich(
    rows: list[dict[str, str]], coco_annotation_json: Path | None = None
) -> list[dict[str, str]]:
    coco_metadata = (
        load_coco_metadata(coco_annotation_json)
        if coco_annotation_json is not None
        else None
    )
    return [enrich_row(row, coco_metadata) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich a multicorpus identity map with dataset-level and "
            "image-level provenance metadata."
        )
    )
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.identity_map)
    enriched = enrich(rows, coco_annotation_json=args.coco_annotations)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(enriched[0].keys()) if enriched else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    status_counts: dict[str, int] = {}
    for row in enriched:
        status = row["provenance_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    )
    print(f"enriched {len(enriched)} rows -> {args.output_csv} ({status_text})")


if __name__ == "__main__":
    main()
