from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from itertools import product
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

ATTACK_METRICS = ("f1", "iou", "recall", "precision", "fpr", "fnr")
CLEAN_METRICS = ("psnrDb", "cleanBitErrorRate", "flaggedBlockCount")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "blind-v2-real-image-scratch-pilot/v1":
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    return manifest


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("values must not be empty")
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def bootstrap_ci(values: list[float], seed: str, iterations: int = 5000) -> tuple[float, float]:
    if len(values) <= 1:
        value = values[0] if values else 0.0
        return value, value
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(values)
    for _ in range(iterations):
        estimates.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    estimates.sort()
    lo = estimates[int(0.025 * (iterations - 1))]
    hi = estimates[int(0.975 * (iterations - 1))]
    return lo, hi


def sign_flip_p_value(deltas: list[float]) -> float:
    nonzero = [value for value in deltas if value != 0.0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    n = len(nonzero)
    if n <= 20:
        count = 0
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=n):
            total = abs(sum(sign * value for sign, value in zip(signs, nonzero)))
            count += 1
            if total >= observed - 1e-15:
                extreme += 1
        return extreme / count
    seed = hashlib.sha256(",".join(f"{value:.17g}" for value in nonzero).encode()).hexdigest()
    rng = random.Random(seed)
    iterations = 20000
    extreme = 0
    for _ in range(iterations):
        total = abs(sum((1.0 if rng.random() < 0.5 else -1.0) * value for value in nonzero))
        if total >= observed - 1e-15:
            extreme += 1
    return extreme / iterations


def baseline_by_mode(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {baseline["basisMode"]: baseline for baseline in manifest["baselines"]}


def keyed_clean_rows(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item.get("recordId") or item["relativePath"]: item
        for item in baseline.get("clean", [])
    }


def keyed_attack_rows(baseline: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    keyed: dict[str, dict[str, dict[str, Any]]] = {}
    for attack in baseline.get("attacks", []):
        keyed[attack["attack"]] = {
            item.get("recordId") or item["relativePath"]: item
            for item in attack.get("images", [])
        }
    return keyed


def paired_summary(
    *,
    comparison: str,
    metric: str,
    basis_mode: str,
    baseline_values: list[float],
    fisher_values: list[float],
    higher_is_better: bool,
    seed_prefix: str,
) -> dict[str, Any]:
    if len(baseline_values) != len(fisher_values):
        raise ValueError("paired arrays must have the same length")
    deltas = [fisher - baseline for fisher, baseline in zip(fisher_values, baseline_values)]
    ci_low, ci_high = bootstrap_ci(deltas, f"{seed_prefix}:{comparison}:{basis_mode}:{metric}")
    wins = sum(1 for value in deltas if value > 0)
    losses = sum(1 for value in deltas if value < 0)
    ties = sum(1 for value in deltas if value == 0)
    direction = "higher" if higher_is_better else "lower"
    favorable = wins if higher_is_better else losses
    unfavorable = losses if higher_is_better else wins
    return {
        "comparison": comparison,
        "basisMode": basis_mode,
        "metric": metric,
        "preferredDirection": direction,
        "n": len(deltas),
        "fisherMean": mean(fisher_values),
        "baselineMean": mean(baseline_values),
        "meanDeltaFisherMinusBaseline": mean(deltas),
        "medianDeltaFisherMinusBaseline": median(deltas),
        "deltaCi95Low": ci_low,
        "deltaCi95High": ci_high,
        "pairedSignFlipP": sign_flip_p_value(deltas),
        "favorablePairs": favorable,
        "unfavorablePairs": unfavorable,
        "ties": ties,
    }


def compare_clean(manifest: dict[str, Any], reference_mode: str) -> list[dict[str, Any]]:
    baselines = baseline_by_mode(manifest)
    reference = baselines[reference_mode]
    reference_rows = keyed_clean_rows(reference)
    rows: list[dict[str, Any]] = []
    for mode, baseline in sorted(baselines.items()):
        if mode == reference_mode:
            continue
        candidate_rows = keyed_clean_rows(baseline)
        shared = sorted(set(reference_rows).intersection(candidate_rows))
        for metric in CLEAN_METRICS:
            rows.append(
                paired_summary(
                    comparison="clean",
                    metric=metric,
                    basis_mode=mode,
                    fisher_values=[float(reference_rows[key][metric]) for key in shared],
                    baseline_values=[float(candidate_rows[key][metric]) for key in shared],
                    higher_is_better=metric == "psnrDb",
                    seed_prefix=str(manifest.get("manifestName", "manifest")),
                )
            )
    return rows


def compare_attacks(manifest: dict[str, Any], reference_mode: str) -> list[dict[str, Any]]:
    baselines = baseline_by_mode(manifest)
    reference = keyed_attack_rows(baselines[reference_mode])
    rows: list[dict[str, Any]] = []
    for mode, baseline in sorted(baselines.items()):
        if mode == reference_mode:
            continue
        candidate = keyed_attack_rows(baseline)
        for attack in sorted(set(reference).intersection(candidate)):
            shared = sorted(set(reference[attack]).intersection(candidate[attack]))
            for metric in ATTACK_METRICS:
                rows.append(
                    paired_summary(
                        comparison=f"attack:{attack}",
                        metric=metric,
                        basis_mode=mode,
                        fisher_values=[
                            float(reference[attack][key][metric]) for key in shared
                        ],
                        baseline_values=[
                            float(candidate[attack][key][metric]) for key in shared
                        ],
                        higher_is_better=metric not in {"fpr", "fnr"},
                        seed_prefix=str(manifest.get("manifestName", "manifest")),
                    )
                )
    return rows


def comparison_rows(manifest: dict[str, Any], reference_mode: str = "fisher") -> list[dict[str, Any]]:
    baselines = baseline_by_mode(manifest)
    if reference_mode not in baselines:
        raise ValueError(f"reference mode not found: {reference_mode}")
    return compare_clean(manifest, reference_mode) + compare_attacks(manifest, reference_mode)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(manifest: dict[str, Any], rows: list[dict[str, Any]], reference_mode: str) -> str:
    lines = [
        "# Paired real-pilot comparison",
        "",
        f"- Schema: `{manifest['schema']}`",
        f"- Pilot mode: `{manifest.get('pilotMode')}`",
        f"- Manifest: `{manifest.get('manifestName')}`",
        f"- Reference mode: `{reference_mode}`",
        f"- Evaluation images: {manifest.get('evaluationImageCount')}",
        f"- Target FPR alpha: {manifest.get('targetFalsePositiveRate')}",
        "",
        "## Attack F1 deltas",
        "",
        "| Attack | Baseline | n | Fisher mean | Baseline mean | Delta | 95% CI | p | favorable | unfavorable | ties |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    attack_f1 = [
        row for row in rows if row["comparison"].startswith("attack:") and row["metric"] == "f1"
    ]
    for row in attack_f1:
        attack = row["comparison"].split(":", 1)[1]
        lines.append(
            "| {attack} | {basisMode} | {n} | {fisherMean:.3f} | {baselineMean:.3f} | "
            "{meanDeltaFisherMinusBaseline:.3f} | [{deltaCi95Low:.3f}, {deltaCi95High:.3f}] | "
            "{pairedSignFlipP:.3f} | {favorablePairs} | {unfavorablePairs} | {ties} |".format(
                attack=attack,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Clean PSNR deltas",
            "",
            "| Baseline | n | Fisher mean | Baseline mean | Delta | 95% CI | p | favorable | unfavorable | ties |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    clean_psnr = [row for row in rows if row["comparison"] == "clean" and row["metric"] == "psnrDb"]
    for row in clean_psnr:
        lines.append(
            "| {basisMode} | {n} | {fisherMean:.3f} | {baselineMean:.3f} | "
            "{meanDeltaFisherMinusBaseline:.3f} | [{deltaCi95Low:.3f}, {deltaCi95High:.3f}] | "
            "{pairedSignFlipP:.3f} | {favorablePairs} | {unfavorablePairs} | {ties} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation status",
            "",
            "This is a private scratch paired comparison. It supports engineering "
            "decisions and falsification checks. Public reporting requires a "
            "predeclared analysis plan and promoted result manifests.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(input_path: Path, output_dir: Path, reference_mode: str = "fisher") -> dict[str, Path]:
    manifest = load_manifest(input_path)
    rows = comparison_rows(manifest, reference_mode=reference_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "paired": output_dir / "paired_comparisons.csv",
        "summary": output_dir / "paired_summary.md",
    }
    write_csv(paths["paired"], rows)
    paths["summary"].write_text(
        markdown_summary(manifest, rows, reference_mode),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired Fisher-vs-baseline comparisons for a real-pilot scratch manifest."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-mode", default="fisher")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = summarize(args.input, args.output_dir, reference_mode=args.reference_mode)
    print(
        "wrote paired real-pilot comparison: "
        + ", ".join(f"{name}={path}" for name, path in paths.items())
    )


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
