from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    subcorpus: str
    source_id: str
    source_path: str
    original_path: str
    split: str
    exact_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def source_id(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def derived_stem(dataset: str, source_identifier: str) -> str:
    return hashlib.sha256(f"{dataset}:{source_identifier}".encode("utf-8")).hexdigest()[
        :16
    ]


def add_raw_records(
    records: dict[tuple[str, str], SourceRecord],
    *,
    dataset: str,
    subcorpus: str,
    root: Path,
) -> None:
    for path in image_files(root):
        identifier = source_id(path, root)
        records[(subcorpus, derived_stem(dataset, identifier))] = SourceRecord(
            dataset=dataset,
            subcorpus=subcorpus,
            source_id=identifier,
            source_path=str(path),
            original_path=str(path),
            split="deterministic-sample-source-pool",
            exact_id="",
        )


def add_split_records(
    records: dict[tuple[str, str], SourceRecord],
    *,
    dataset: str,
    subcorpus: str,
    split_csv: Path,
    dynacis_root: Path,
) -> None:
    if not split_csv.exists():
        return
    with split_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            processed_path = Path(row["processed_path"])
            identifier = source_id(processed_path, dynacis_root)
            records[(subcorpus, derived_stem(dataset, identifier))] = SourceRecord(
                dataset=dataset,
                subcorpus=subcorpus,
                source_id=identifier,
                source_path=row["processed_path"],
                original_path=row.get("original_path", ""),
                split=row.get("split", ""),
                exact_id=row.get("exact_id", ""),
            )


def build_source_index(
    *,
    bossbase_root: Path | None = None,
    bows2_root: Path | None = None,
    dynacis_root: Path | None = None,
) -> dict[tuple[str, str], SourceRecord]:
    records: dict[tuple[str, str], SourceRecord] = {}
    if bossbase_root is not None:
        add_raw_records(
            records,
            dataset="BOSSbase 1.01",
            subcorpus="bossbase_1_01",
            root=bossbase_root,
        )
    if bows2_root is not None:
        add_raw_records(
            records,
            dataset="BOWS-2",
            subcorpus="bows_2",
            root=bows2_root,
        )
    if dynacis_root is not None:
        processed = dynacis_root / "processed"
        add_split_records(
            records,
            dataset="MS-COCO val2017",
            subcorpus="ms_coco_val2017",
            split_csv=processed / "coco_splits.csv",
            dynacis_root=dynacis_root,
        )
        add_split_records(
            records,
            dataset="DTD textures",
            subcorpus="dtd_textures",
            split_csv=processed / "dtd_splits.csv",
            dynacis_root=dynacis_root,
        )
        add_split_records(
            records,
            dataset="INRIA Holidays",
            subcorpus="inria_holidays",
            split_csv=processed / "holidays_splits.csv",
            dynacis_root=dynacis_root,
        )
    return records


def map_identities(
    *,
    derived_root: Path,
    bossbase_root: Path | None = None,
    bows2_root: Path | None = None,
    dynacis_root: Path | None = None,
) -> list[dict[str, str]]:
    source_index = build_source_index(
        bossbase_root=bossbase_root,
        bows2_root=bows2_root,
        dynacis_root=dynacis_root,
    )
    rows: list[dict[str, str]] = []
    for subdir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        for path in image_files(subdir):
            with Image.open(path) as image:
                width, height = image.size
            stem = path.stem
            record = source_index.get((subdir.name, stem))
            rows.append(
                {
                    "subcorpus": subdir.name,
                    "local_path": path.relative_to(derived_root).as_posix(),
                    "local_sha256": sha256_file(path),
                    "width": str(width),
                    "height": str(height),
                    "matched": "true" if record is not None else "false",
                    "dataset": record.dataset if record is not None else "",
                    "source_id": record.source_id if record is not None else "",
                    "source_path": record.source_path if record is not None else "",
                    "original_path": record.original_path if record is not None else "",
                    "source_split": record.split if record is not None else "",
                    "exact_id": record.exact_id if record is not None else "",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map deterministic multicorpus PNG names back to upstream source "
            "identities without copying image pixels."
        )
    )
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bossbase-root", type=Path)
    parser.add_argument("--bows2-root", type=Path)
    parser.add_argument("--dynacis-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = map_identities(
        derived_root=args.derived_root,
        bossbase_root=args.bossbase_root,
        bows2_root=args.bows2_root,
        dynacis_root=args.dynacis_root,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subcorpus",
        "local_path",
        "local_sha256",
        "width",
        "height",
        "matched",
        "dataset",
        "source_id",
        "source_path",
        "original_path",
        "source_split",
        "exact_id",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    matched = sum(1 for row in rows if row["matched"] == "true")
    print(f"mapped {matched}/{len(rows)} images -> {args.output_csv}")


if __name__ == "__main__":
    main()
