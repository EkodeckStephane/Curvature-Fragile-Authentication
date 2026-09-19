"""Create a non-destructive SHA-256 inventory of locally supplied papers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


FIELDNAMES = [
    "file",
    "bytes",
    "sha256",
    "identityStatus",
    "readingStatus",
    "notesFile",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_existing(
    path: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    """Load review metadata keyed by path and content hash.

    Binding metadata to both values prevents a replaced PDF from inheriting the
    verification status of the previous file at the same location.
    """
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            (row["file"], row["sha256"]): row
            for row in rows
            if row.get("file") and row.get("sha256")
        }


def inventory(
    input_root: Path,
    existing: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
) -> list[dict[str, str | int]]:
    existing = existing or {}
    rows: list[dict[str, str | int]] = []
    for path in sorted(input_root.rglob("*.pdf"), key=lambda item: str(item).lower()):
        relative_path = path.relative_to(input_root).as_posix()
        digest = sha256_file(path)
        previous = existing.get((relative_path, digest), {})
        rows.append({
            "file": relative_path,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "identityStatus": previous.get("identityStatus", "unverified"),
            "readingStatus": previous.get("readingStatus", "not-read"),
            "notesFile": previous.get("notesFile", ""),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="literature/inbox")
    parser.add_argument(
        "--output", default="literature/manifests/pdf_inventory.csv"
    )
    parser.add_argument(
        "--run-manifest", default="literature/manifests/pdf_inventory_run.json"
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    run_manifest = Path(args.run_manifest).resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input directory not found: {input_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output)
    rows = inventory(input_root, existing)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "inputDirectory": str(input_root),
        "pdfCount": len(rows),
        "inventoryFile": str(output),
        "inventorySha256": sha256_file(output),
        "operation": (
            "source PDFs were not moved or renamed; review metadata was "
            "preserved only for unchanged path-and-SHA-256 pairs"
        ),
    }
    run_manifest.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pdfs": len(rows), "inventory": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
