from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fisher_geometry import generalized_eigen_decomposition, rayleigh_quotient


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_spd(rng: np.random.Generator, dimension: int, ridge: float) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension))
    return raw.T @ raw + ridge * np.eye(dimension)


def evaluate_case(
    name: str,
    fisher: np.ndarray,
    cost: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    tolerance: float,
    expected_value: float | None = None,
    expected_direction_abs: list[float] | None = None,
) -> dict[str, Any]:
    result = generalized_eigen_decomposition(fisher, cost)
    residuals = []
    for index, value in enumerate(result.values):
        vector = result.vectors[:, index]
        residuals.append(float(np.linalg.norm(fisher @ vector - value * (cost @ vector))))

    p_orthonormal_error = float(
        np.linalg.norm(result.vectors.T @ cost @ result.vectors - np.eye(cost.shape[0]))
    )
    sampled_excess = 0.0
    for _ in range(samples):
        direction = rng.normal(size=fisher.shape[0])
        quotient = rayleigh_quotient(fisher, cost, direction)
        sampled_excess = max(sampled_excess, float(quotient - result.principal_value))

    checks: dict[str, bool] = {
        "residualsWithinTolerance": max(residuals) <= tolerance,
        "pOrthonormalWithinTolerance": p_orthonormal_error <= tolerance,
        "sampledDirectionsBelowPrincipal": sampled_excess <= tolerance,
    }

    if expected_value is not None:
        checks["expectedPrincipalValue"] = (
            abs(result.principal_value - expected_value) <= tolerance
        )
    if expected_direction_abs is not None:
        expected = np.asarray(expected_direction_abs, dtype=float)
        actual = np.abs(result.principal_vector)
        checks["expectedPrincipalDirectionAbs"] = bool(
            np.allclose(actual, expected, rtol=0.0, atol=tolerance)
        )

    return {
        "name": name,
        "dimension": int(fisher.shape[0]),
        "principalValue": result.principal_value,
        "spectralValues": [float(value) for value in result.values],
        "maxResidual": max(residuals),
        "pOrthonormalError": p_orthonormal_error,
        "maxSampledRayleighExcess": sampled_excess,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["seed"])
    tolerance = float(config["tolerance"])
    samples = int(config["randomDirectionSamples"])
    rng = np.random.default_rng(seed)

    cases = []
    for case in config["knownCases"]:
        cases.append(
            evaluate_case(
                name=case["name"],
                fisher=np.asarray(case["fisher"], dtype=float),
                cost=np.asarray(case["cost"], dtype=float),
                rng=rng,
                samples=samples,
                tolerance=tolerance,
                expected_value=float(case["expectedPrincipalValue"]),
                expected_direction_abs=case["expectedPrincipalDirectionAbs"],
            )
        )

    random_config = config["randomCases"]
    for index in range(int(random_config["count"])):
        fisher = make_spd(rng, int(random_config["dimension"]), float(random_config["ridge"]))
        cost = make_spd(rng, int(random_config["dimension"]), float(random_config["ridge"]))
        cases.append(
            evaluate_case(
                name=f"random_spd_{index:02d}",
                fisher=fisher,
                cost=cost,
                rng=rng,
                samples=samples,
                tolerance=tolerance,
            )
        )

    summary = {
        "schema": "synthetic-fisher-gate/v1",
        "config": str(config_path.as_posix()),
        "configSha256": sha256_file(config_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": seed,
        "caseCount": len(cases),
        "acceptedCaseCount": sum(1 for case in cases if case["accepted"]),
        "allAccepted": all(case["accepted"] for case in cases),
        "cases": cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic Fisher-sensitivity validation gate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "synthetic_fisher_gate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "synthetic_fisher_gate" / "summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.config, args.output)
    if not summary["allAccepted"]:
        raise SystemExit(1)
    print(
        f"accepted {summary['acceptedCaseCount']}/{summary['caseCount']} "
        f"synthetic Fisher cases -> {args.output}"
    )


if __name__ == "__main__":
    main()
