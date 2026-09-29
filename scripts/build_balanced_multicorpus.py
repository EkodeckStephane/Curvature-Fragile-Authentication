from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceCorpus:
    subcorpus: str
    root: Path
    calibration_count: int
    evaluation_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(subcorpus: str, root: Path, path: Path, seed: str) -> str:
    rel = path.relative_to(root).as_posix()
    return hashlib.sha256(f"{seed}|{subcorpus}|{rel}".encode("utf-8")).hexdigest()


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no images found under {root}")
    return files


def selected_images(corpus: SourceCorpus, seed: str) -> tuple[list[Path], list[Path]]:
    files = collect_images(corpus.root)
    needed = corpus.calibration_count + corpus.evaluation_count
    if len(files) < needed:
        raise ValueError(
            f"{corpus.subcorpus} has {len(files)} images but {needed} are required"
        )
    ordered = sorted(files, key=lambda path: stable_key(corpus.subcorpus, corpus.root, path, seed))
    return (
        ordered[: corpus.calibration_count],
        ordered[corpus.calibration_count : needed],
    )


def _resize_to_max_long_edge(image: Image.Image, max_long_edge: int | None) -> Image.Image:
    if max_long_edge is None:
        return image
    if max_long_edge <= 0:
        raise ValueError("max_long_edge must be positive")
    width, height = image.size
    current = max(width, height)
    if current <= max_long_edge:
        return image
    scale = max_long_edge / current
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    return resized


def write_png_luma(
    source: Path,
    destination: Path,
    max_long_edge: int | None,
) -> tuple[int, int, str]:
    with Image.open(source) as image:
        luminance = _resize_to_max_long_edge(image.convert("L"), max_long_edge)
        width, height = luminance.size
        destination.parent.mkdir(parents=True, exist_ok=True)
        luminance.save(destination, format="PNG", optimize=True)
    return width, height, sha256_file(destination)


def build(
    corpora: list[SourceCorpus],
    *,
    output_root: Path,
    manifest_csv: Path,
    seed: str,
    max_long_edge: int | None,
    overwrite: bool,
) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for corpus in corpora:
        calibration, evaluation = selected_images(corpus, seed)
        for split, paths in (("calibration", calibration), ("evaluation", evaluation)):
            for split_index, source in enumerate(paths):
                source_hash = sha256_file(source)
                artifact_id = f"{hashlib.sha256((corpus.subcorpus + '|' + source_hash).encode()).hexdigest()[:16]}.png"
                destination = output_root / corpus.subcorpus / artifact_id
                width, height, artifact_hash = write_png_luma(
                    source,
                    destination,
                    max_long_edge=max_long_edge,
                )
                rows.append(
                    {
                        "subcorpus": corpus.subcorpus,
                        "split": split,
                        "split_index": split_index,
                        "source_root_name": corpus.root.name,
                        "source_relative_path": source.relative_to(corpus.root).as_posix(),
                        "source_sha256": source_hash,
                        "derived_relative_path": destination.relative_to(output_root).as_posix(),
                        "derived_sha256": artifact_hash,
                        "width": width,
                        "height": height,
                        "max_long_edge": max_long_edge or "",
                        "selection_seed": seed,
                    }
                )

    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic balanced private multicorpus from local sources."
    )
    parser.add_argument("--bossbase-root", type=Path, required=True)
    parser.add_argument("--bows2-root", type=Path, required=True)
    parser.add_argument("--dtd-root", type=Path, required=True)
    parser.add_argument("--holidays-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--calibration-per-subcorpus", type=int, required=True)
    parser.add_argument("--evaluation-per-subcorpus", type=int, required=True)
    parser.add_argument(
        "--max-long-edge",
        type=int,
        help="Resize luminance images so their longest side is at most this value.",
    )
    parser.add_argument("--seed", default="cfa-balanced-multicorpus-v2-20260927")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_per_subcorpus <= 0 or args.evaluation_per_subcorpus <= 0:
        raise ValueError("split counts must be positive")
    corpora = [
        SourceCorpus("bossbase_1_01", args.bossbase_root, args.calibration_per_subcorpus, args.evaluation_per_subcorpus),
        SourceCorpus("bows_2", args.bows2_root, args.calibration_per_subcorpus, args.evaluation_per_subcorpus),
        SourceCorpus("dtd_textures", args.dtd_root, args.calibration_per_subcorpus, args.evaluation_per_subcorpus),
        SourceCorpus("inria_holidays", args.holidays_root, args.calibration_per_subcorpus, args.evaluation_per_subcorpus),
        SourceCorpus("ms_coco_val2017", args.coco_root, args.calibration_per_subcorpus, args.evaluation_per_subcorpus),
    ]
    build(
        corpora,
        output_root=args.output_root,
        manifest_csv=args.manifest_csv,
        seed=args.seed,
        max_long_edge=args.max_long_edge,
        overwrite=args.overwrite,
    )
    total = len(corpora) * (args.calibration_per_subcorpus + args.evaluation_per_subcorpus)
    print(f"built balanced multicorpus: images={total} -> {args.output_root}")


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required")
    main()
