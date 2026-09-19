from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path, hash_files: bool = False) -> dict[str, Any]:
    """Inventory an image corpus without copying image data."""

    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    records: dict[str, Any] = {}
    total_count = 0
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        dimensions: dict[str, int] = {}
        modes: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        byte_count = 0
        for path in files:
            with Image.open(path) as image:
                dimensions[f"{image.size[0]}x{image.size[1]}"] = (
                    dimensions.get(f"{image.size[0]}x{image.size[1]}", 0) + 1
                )
                modes[image.mode] = modes.get(image.mode, 0) + 1
            byte_count += path.stat().st_size
            sample: dict[str, Any] = {
                "file": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
            }
            if hash_files:
                sample["sha256"] = sha256_file(path)
            samples.append(sample)
        records[directory.name] = {
            "count": len(files),
            "bytes": byte_count,
            "dimensions": dict(sorted(dimensions.items())),
            "modes": dict(sorted(modes.items())),
            "samples": samples,
        }
        total_count += len(files)

    return {
        "schema": "image-corpus-audit/v1",
        "rootName": root.name,
        "totalImageCount": total_count,
        "subcorpora": records,
        "hashFiles": hash_files,
        "dataCopied": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an image corpus.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(args.root, hash_files=args.hash_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"audited {summary['totalImageCount']} images from {args.root} -> {args.output}"
    )


if __name__ == "__main__":
    main()
