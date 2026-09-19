"""Merge raw Semantic Scholar searches into an auditable candidate shortlist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TERM_WEIGHTS = {
    "fragile watermark": 12,
    "semi-fragile watermark": 10,
    "image authentication": 8,
    "tamper localization": 8,
    "tampering localization": 8,
    "image forgery": 6,
    "image manipulation": 6,
    "fisher information": 8,
    "fisher-rao": 10,
    "information geometry": 8,
    "watermark": 3,
    "authentication": 2,
    "tamper": 3,
    "forgery": 2,
    "localization": 2,
}


def query_id(path: Path) -> str:
    match = re.match(r"\d{8}T\d{6}Z_(.+)\.json$", path.name)
    return match.group(1) if match else path.stem


def merge(raw_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    records: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in sorted(raw_root.glob("*.json")):
        body = path.read_bytes()
        payload = json.loads(body)
        source_query = query_id(path)
        sources.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "queryId": source_query,
            }
        )
        for paper in payload.get("data", []):
            paper_id = paper.get("paperId")
            if not paper_id:
                continue
            existing = records.get(paper_id, {})
            discovered = set(existing.get("discoveredBy", []))
            discovered.add(source_query)
            merged = {**existing, **paper, "discoveredBy": sorted(discovered)}
            records[paper_id] = merged
    return records, sources


def venue_name(paper: dict[str, Any]) -> str:
    publication_venue = paper.get("publicationVenue") or {}
    return str(publication_venue.get("name") or paper.get("venue") or "")


def relevance(paper: dict[str, Any]) -> int:
    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    venue = venue_name(paper).lower()
    score = 0
    for term, weight in TERM_WEIGHTS.items():
        if term in title:
            score += 3 * weight
        elif term in abstract:
            score += weight
    if "transactions on information forensics and security" in venue:
        score += 20
    year = paper.get("year")
    if isinstance(year, int) and year >= 2020:
        score += min(year - 2019, 7)
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="literature/api/raw")
    parser.add_argument("--output", default="literature/manifests/s2_candidates.csv")
    parser.add_argument(
        "--manifest", default="literature/manifests/s2_candidate_build.json"
    )
    args = parser.parse_args()

    raw_root = Path(args.raw).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records, sources = merge(raw_root)

    fieldnames = [
        "relevanceScore",
        "verificationStatus",
        "paperId",
        "year",
        "title",
        "venue",
        "doi",
        "arxiv",
        "citationCount",
        "openAccessPdf",
        "semanticScholarUrl",
        "discoveredBy",
    ]
    rows = []
    for paper in records.values():
        external = paper.get("externalIds") or {}
        open_pdf = paper.get("openAccessPdf") or {}
        rows.append(
            {
                "relevanceScore": relevance(paper),
                "verificationStatus": "candidate-unverified",
                "paperId": paper.get("paperId") or "",
                "year": paper.get("year") or "",
                "title": paper.get("title") or "",
                "venue": venue_name(paper),
                "doi": external.get("DOI") or "",
                "arxiv": external.get("ArXiv") or "",
                "citationCount": paper.get("citationCount") or 0,
                "openAccessPdf": open_pdf.get("url") or "",
                "semanticScholarUrl": paper.get("url") or "",
                "discoveredBy": ";".join(paper.get("discoveredBy", [])),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["relevanceScore"]),
            -int(row["year"] or 0),
            str(row["title"]),
        )
    )

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "rawDirectory": str(raw_root),
        "rawSources": sources,
        "uniqueCandidates": len(rows),
        "candidateFile": str(output),
        "candidateSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rankingNature": "discovery aid only; every candidate remains unverified",
        "termWeights": TERM_WEIGHTS,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidates": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
