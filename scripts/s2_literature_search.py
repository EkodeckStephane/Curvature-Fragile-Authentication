"""Auditable Semantic Scholar discovery with conservative rate limiting.

The API key is read only from ``SEMANTIC_SCHOLAR_API_KEY``. Raw responses are
stored in an untracked local literature directory so discovery can be audited
without publishing abstracts or credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = (
    "paperId,title,abstract,authors,year,venue,publicationVenue,externalIds,"
    "url,openAccessPdf,citationCount,referenceCount,publicationTypes,journal"
)


@dataclass
class RateLimiter:
    """Keep starts of consecutive requests at least ``interval`` apart."""

    interval: float = 1.10
    _last_request: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_request is not None:
            remaining = self.interval - (now - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned[:80] or "query"


def request_json(
    endpoint: str,
    params: dict[str, Any],
    api_key: str,
    limiter: RateLimiter,
    retries: int = 4,
) -> tuple[dict[str, Any], bytes]:
    encoded = urllib.parse.urlencode(params)
    url = f"{API_ROOT}/{endpoint}?{encoded}"
    request = urllib.request.Request(
        url,
        headers={
            "x-api-key": api_key,
            "User-Agent": "curvature-fragile-authentication-literature-audit/0.1",
        },
    )

    for attempt in range(retries + 1):
        limiter.wait()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return json.loads(body), body
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2.0 ** (attempt + 1)
            time.sleep(max(delay, limiter.interval))
        except urllib.error.URLError:
            if attempt == retries:
                raise
            time.sleep(2.0 ** (attempt + 1))
    raise RuntimeError("unreachable")


def load_queries(path: Path) -> list[dict[str, str]]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list) or not content:
        raise ValueError("query file must contain a non-empty JSON list")
    queries: list[dict[str, str]] = []
    for item in content:
        if not isinstance(item, dict) or not item.get("id") or not item.get("query"):
            raise ValueError("each query needs non-empty 'id' and 'query' fields")
        queries.append({"id": str(item["id"]), "query": str(item["query"])})
    return queries


def search(args: argparse.Namespace) -> int:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print(
            "SEMANTIC_SCHOLAR_API_KEY is required in the process environment.",
            file=sys.stderr,
        )
        return 2

    query_path = Path(args.queries).resolve()
    output_root = Path(args.output).resolve()
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    limiter = RateLimiter(interval=args.interval)
    records: dict[str, dict[str, Any]] = {}
    run_queries: list[dict[str, Any]] = []

    queries = load_queries(query_path)
    if args.query_id:
        queries = [query for query in queries if query["id"] in set(args.query_id)]
        missing = sorted(set(args.query_id) - {query["id"] for query in queries})
        if missing:
            raise ValueError(f"unknown query id(s): {', '.join(missing)}")

    for query in queries:
        payload, body = request_json(
            "paper/search",
            {
                "query": query["query"],
                "limit": args.limit,
                "offset": 0,
                "fields": DEFAULT_FIELDS,
            },
            api_key,
            limiter,
        )
        raw_path = raw_root / f"{stamp}_{safe_id(query['id'])}.json"
        raw_path.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        data = payload.get("data", [])
        for paper in data:
            paper_id = paper.get("paperId")
            if paper_id:
                record = dict(paper)
                record.setdefault("discoveredBy", [])
                record["discoveredBy"] = sorted(
                    set(record["discoveredBy"] + [query["id"]])
                )
                if paper_id in records:
                    record["discoveredBy"] = sorted(
                        set(records[paper_id].get("discoveredBy", []))
                        | set(record["discoveredBy"])
                    )
                records[paper_id] = record
        run_queries.append(
            {
                "id": query["id"],
                "query": query["query"],
                "returned": len(data),
                "reportedTotal": payload.get("total"),
                "rawFile": raw_path.name,
                "sha256": digest,
            }
        )

    merged_path = output_root / f"{stamp}_candidates.jsonl"
    with merged_path.open("w", encoding="utf-8", newline="\n") as stream:
        for paper_id in sorted(records):
            stream.write(json.dumps(records[paper_id], ensure_ascii=False) + "\n")

    manifest = {
        "startedAtUtc": stamp,
        "endpoint": f"{API_ROOT}/paper/search",
        "minimumRequestIntervalSeconds": args.interval,
        "limitPerQuery": args.limit,
        "fields": DEFAULT_FIELDS.split(","),
        "queryFile": str(query_path),
        "uniqueCandidates": len(records),
        "mergedFile": merged_path.name,
        "mergedSha256": hashlib.sha256(merged_path.read_bytes()).hexdigest(),
        "queries": run_queries,
    }
    manifest_path = output_root / f"{stamp}_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "papers": len(records)}))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--queries",
        default="configs/literature_queries.json",
        help="JSON query list (default: configs/literature_queries.json)",
    )
    result.add_argument(
        "--output",
        default="literature/api",
        help="untracked output directory (default: literature/api)",
    )
    result.add_argument("--limit", type=int, default=100)
    result.add_argument(
        "--query-id",
        action="append",
        help="run only this query id; repeat the option to select several",
    )
    result.add_argument(
        "--interval",
        type=float,
        default=1.10,
        help="minimum seconds between request starts; values below 1.05 are rejected",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    if args.interval < 1.05:
        raise SystemExit("--interval must be at least 1.05 seconds")
    return search(args)


if __name__ == "__main__":
    raise SystemExit(main())
