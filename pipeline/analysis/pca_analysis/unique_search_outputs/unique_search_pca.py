"""Search source-1 PCA component subsets for source-1 unique PID information."""

from __future__ import annotations

import csv
from itertools import combinations
from math import comb
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.pid_pipeline import PIDPipeline, PIDPipelineFunctions


CSV_FIELDS = ("subset", "subset_size", "cmi_score", "unique", "red", "unq2", "syn", "elapsed_seconds", "status")


def run_pid_pc_subset_search(
    target: Any,
    source_1: Any,
    source_2: Any,
    pid_callable: Callable[..., Any] | None = None,
    *,
    cmi_threshold: float = 1e-6,
    unique_threshold: float = 1e-6,
    beam_width: int = 5,
    max_subset_size: int = 3,
    initial_subset_size: int = 1,
    initial_subset_count: int | None = None,
    floating_tolerance: float = 1e-9,
    max_runtime_seconds: float = 600,
    rng_seed: int = 56,
    pid_kwargs: dict[str, Any] | None = None,
    all_csv_path: str | Path | None = None,
    best_csv_path: str | Path | None = None,
    use_floating_backward: bool = True,
) -> dict[str, Any]:
    """Run beam search over source_1 PCA columns for source-1 unique PID.

    Inputs: target/source_1/source_2 are 2D array-like objects with equal rows;
        pid_callable is a PID function or None for pid_calc_adapter; thresholds,
        beam_width, subset sizes, runtime, seed, pid_kwargs, CSV paths, and
        use_floating_backward configure the search. initial_subset_count can
        randomly sample the initial subsets from the CMI-passing PCs.
    Outputs: dict with status, singleton CMI scores, evaluated rows, top rows,
        best subset, best unique information, and best PID result.
    """

    target = _as_2d_array(target, "target")
    source_1 = _as_2d_array(source_1, "source_1")
    source_2 = _as_2d_array(source_2, "source_2")
    if len({target.shape[0], source_1.shape[0], source_2.shape[0]}) != 1:
        raise ValueError(f"target, source_1, and source_2 must have the same rows, got {[target.shape, source_1.shape, source_2.shape]}")
    if initial_subset_size < 1:
        raise ValueError("initial_subset_size must be at least 1")
    if max_subset_size < initial_subset_size:
        raise ValueError("max_subset_size must be greater than or equal to initial_subset_size")
    if initial_subset_count is not None and initial_subset_count < 1:
        raise ValueError("initial_subset_count must be None or at least 1")

    start = time.monotonic()
    pid_kwargs = dict(pid_kwargs or {})
    pid_kwargs.setdefault("rng_seed", rng_seed)
    cmi_scores = [_gaussian_cmi_bits(target, source_1[:, [j]], source_2) for j in range(source_1.shape[1])]
    candidates = [i for i, score in enumerate(cmi_scores) if score > cmi_threshold]

    for i, score in enumerate(cmi_scores):
        if i not in candidates:
            _append_csv_row(all_csv_path, (i,), start, "cmi_fail", cmi_score=score)

    if not candidates:
        return {
            "status": "no_singleton_passed_cmi",
            "message": "No singleton source_1 PC passed the conditional mutual information threshold.",
            "cmi_scores": cmi_scores,
            "selected_candidates": [],
            "all_evaluated_subsets": [],
            "top_subsets": [],
            "best_subset": None,
            "best_unique": None,
            "best_pid_components": None,
            "best_red": None,
            "best_unq1": None,
            "best_unq2": None,
            "best_syn": None,
            "best_pid_result": None,
        }

    if pid_callable is None:
        from pipeline.pipeline_utils import pid_calc_adapter

        pid_callable = pid_calc_adapter
        pid_kwargs.setdefault("method", "tilde")

    pipeline = PIDPipeline(
        PIDPipelineFunctions(
            target_extraction=lambda target: {"target": target},
            sources_extraction=lambda source_1, source_2: {"X1": source_1, "X2": source_2},
            choose_layer=lambda sources: {"X1": None, "X2": None},
            feature_extraction=lambda source_context, layer_name, target_context: source_context,
            pid_calculation=pid_callable,
        )
    )
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    beam: list[dict[str, Any]] = []
    status = "completed"
    initial_subsets = _initial_subsets(candidates, initial_subset_size, initial_subset_count, rng_seed)
    if not initial_subsets:
        return {
            "status": "no_initial_subsets",
            "message": "Not enough CMI-passing source_1 PCs to build the requested initial subset size.",
            "cmi_scores": cmi_scores,
            "selected_candidates": candidates,
            "all_evaluated_subsets": [],
            "top_subsets": [],
            "best_subset": None,
            "best_unique": None,
            "best_pid_components": None,
            "best_red": None,
            "best_unq1": None,
            "best_unq2": None,
            "best_syn": None,
            "best_pid_result": None,
        }

    for subset in initial_subsets:
        if time.monotonic() - start >= max_runtime_seconds:
            status = "timeout"
            break
        cmi_score = cmi_scores[subset[0]] if len(subset) == 1 else None
        beam.append(_evaluate_subset(subset, target, source_1, source_2, pipeline, pid_kwargs, cache, all_csv_path, unique_threshold, start, cmi_score))

    if status != "timeout":
        beam = _top_rows(beam, beam_width)
        if beam:
            _append_csv_row(best_csv_path, beam[0]["subset"], start, f"best_size_{initial_subset_size}", row=beam[0])

        for size in range(initial_subset_size + 1, min(max_subset_size, len(candidates)) + 1):
            expanded: list[dict[str, Any]] = []
            for row in beam:
                for pc in candidates:
                    if pc in row["subset"]:
                        continue
                    if time.monotonic() - start >= max_runtime_seconds:
                        status = "timeout"
                        break
                    next_row = _evaluate_subset(tuple(sorted((*row["subset"], pc))), target, source_1, source_2, pipeline, pid_kwargs, cache, all_csv_path, unique_threshold, start, None)
                    if use_floating_backward:
                        next_row = _floating_backward(next_row, target, source_1, source_2, pipeline, pid_kwargs, cache, all_csv_path, unique_threshold, floating_tolerance, start, max_runtime_seconds)
                        if next_row.get("status") == "timeout":
                            status = "timeout"
                            break
                    expanded.append(next_row)
                if status == "timeout":
                    break
            if status == "timeout" or not expanded:
                break
            beam = _top_rows(expanded, beam_width)
            _append_csv_row(best_csv_path, beam[0]["subset"], start, f"best_size_{size}", row=beam[0])

    rows = sorted(cache.values(), key=lambda row: (len(row["subset"]), row["subset"]))
    best = max(rows, key=lambda row: row["unique"]) if rows else None
    best_components = None if best is None else {key: _to_float(best["pid"][key]) for key in ("red", "unq1", "unq2", "syn") if key in best["pid"]}
    return {
        "status": status,
        "cmi_scores": cmi_scores,
        "selected_candidates": candidates,
        "all_evaluated_subsets": rows,
        "top_subsets": beam,
        "best_subset": None if best is None else best["subset"],
        "best_unique": None if best is None else best["unique"],
        "best_pid_components": best_components,
        "best_red": None if best_components is None else best_components.get("red"), #The redundancy from the subset that yielded that best unique component.
        "best_unq1": None if best_components is None else best_components.get("unq1"),
        "best_unq2": None if best_components is None else best_components.get("unq2"), #Same as redundancy
        "best_syn": None if best_components is None else best_components.get("syn"), #Same as redundancy
        "best_pid_result": None if best is None else best["pid_result"],
        "config": {"pid_kwargs": pid_kwargs, "initial_subset_size": initial_subset_size, "initial_subset_count": initial_subset_count},
    }


def _initial_subsets(candidates: list[int], subset_size: int, subset_count: int | None, rng_seed: int) -> list[tuple[int, ...]]:
    """Create initial source_1 PC subsets for the beam search.

    Inputs: candidates is CMI-passing PC indices; subset_size is the starting
        subset size; subset_count is optional random subset count; rng_seed is
        the seed used when sampling.
    Outputs: list of sorted PC-index tuples to evaluate first.
    """

    if subset_size > len(candidates):
        return []
    total_subsets = comb(len(candidates), subset_size)
    if subset_count is None or subset_count >= total_subsets:
        return list(combinations(candidates, subset_size))
    rng = np.random.default_rng(rng_seed)
    subsets: set[tuple[int, ...]] = set()
    while len(subsets) < subset_count:
        subsets.add(tuple(sorted(rng.choice(candidates, size=subset_size, replace=False).tolist())))
    return sorted(subsets)


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    """Convert one input to a finite non-empty 2D float array.

    Inputs: value is array-like data; name is a str used in error messages.
    Outputs: np.ndarray with float dtype, two dimensions, and finite values.
    """

    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a non-empty 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _gaussian_cmi_bits(x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: float = 1e-10) -> float:
    """Calculate Gaussian conditional MI I(x; y | z) in bits.

    Inputs: x, y, z are 2D sample matrices; eps is covariance regularization.
    Outputs: float conditional mutual information in bits.
    """

    xy = np.hstack([x, y])
    xyz = np.hstack([xy, z])
    cov = np.cov(xyz, rowvar=False)
    dxy = xy.shape[1]
    cov_xy = _conditional_cov(cov[:dxy, :dxy], cov[:dxy, dxy:], cov[dxy:, dxy:], eps)
    cov_x = cov_xy[: x.shape[1], : x.shape[1]]
    cov_y = cov_xy[x.shape[1] :, x.shape[1] :]
    return max(0.0, 0.5 * (_logdet(cov_x, eps) + _logdet(cov_y, eps) - _logdet(cov_xy, eps)) / np.log(2))


def _conditional_cov(cov_a: np.ndarray, cross_ab: np.ndarray, cov_b: np.ndarray, eps: float) -> np.ndarray:
    """Compute covariance of variable a conditioned on variable b.

    Inputs: cov_a, cross_ab, and cov_b are covariance blocks; eps is regularization.
    Outputs: np.ndarray conditional covariance cov(a | b).
    """

    cov_b = cov_b + eps * np.eye(cov_b.shape[0])
    return cov_a - cross_ab @ np.linalg.solve(cov_b, cross_ab.T)


def _logdet(matrix: np.ndarray, eps: float) -> float:
    """Return a stable log determinant for a covariance-like matrix.

    Inputs: matrix is square np.ndarray; eps is diagonal regularization.
    Outputs: float natural-log determinant.
    """

    sign, value = np.linalg.slogdet(matrix + eps * np.eye(matrix.shape[0]))
    if sign <= 0:
        raise ValueError("Conditional covariance is not positive definite after regularization")
    return float(value)


def _evaluate_subset(
    subset: tuple[int, ...],
    target: np.ndarray,
    source_1: np.ndarray,
    source_2: np.ndarray,
    pipeline: PIDPipeline,
    pid_kwargs: dict[str, Any],
    cache: dict[tuple[int, ...], dict[str, Any]],
    all_csv_path: str | Path | None,
    unique_threshold: float,
    start: float,
    cmi_score: float | None,
) -> dict[str, Any]:
    """Evaluate one source_1 PC subset with PIDPipeline and cache the result.

    Inputs: subset is PC indices; arrays are 2D np.ndarray objects; pipeline is
        PIDPipeline; pid_kwargs/cache/CSV path/threshold/start/cmi_score control
        execution and persistence.
    Outputs: dict row with subset, CMI, unique, PID result, PID components, status.
    """

    subset = tuple(sorted(subset))
    if subset in cache:
        return cache[subset]
    context = pipeline.run(
        target_kwargs={"target": target},
        sources_kwargs={"source_1": source_1[:, subset], "source_2": source_2},
        pid_kwargs=pid_kwargs,
    )
    pid_result = context["pid_results"]
    pid = _pid_components(pid_result)
    row = {
        "subset": subset,
        "cmi_score": cmi_score,
        "unique": _to_float(pid["unq1"]),
        "pid_result": pid_result,
        "pid": pid,
    }
    row["status"] = "unique_pass" if row["unique"] >= unique_threshold else "unique_below_threshold"
    cache[subset] = row
    _append_csv_row(all_csv_path, subset, start, row["status"], row=row)
    return row


def _floating_backward(
    row: dict[str, Any],
    target: np.ndarray,
    source_1: np.ndarray,
    source_2: np.ndarray,
    pipeline: PIDPipeline,
    pid_kwargs: dict[str, Any],
    cache: dict[tuple[int, ...], dict[str, Any]],
    all_csv_path: str | Path | None,
    unique_threshold: float,
    tolerance: float,
    start: float,
    max_runtime_seconds: float,
) -> dict[str, Any]:
    """Prune PCs whose removal does not meaningfully reduce unique information.

    Inputs: row/cache are evaluated subset dictionaries; arrays and pipeline run
        PID; pid_kwargs/CSV path/threshold/tolerance/start/runtime configure pruning.
    Outputs: dict pruned evaluation row or timeout marker.
    """

    changed = True
    while changed and len(row["subset"]) > 1:
        changed = False
        for pc in row["subset"]:
            if time.monotonic() - start >= max_runtime_seconds:
                return {"status": "timeout", "subset": row["subset"], "unique": row["unique"]}
            trial = tuple(i for i in row["subset"] if i != pc)
            trial_row = _evaluate_subset(trial, target, source_1, source_2, pipeline, pid_kwargs, cache, all_csv_path, unique_threshold, start, None)
            if trial_row["unique"] >= row["unique"] - tolerance:
                row = trial_row
                changed = True
                break
    return row


def _pid_components(pid_result: Any) -> dict[str, Any]:
    """Extract a PID component dictionary from common project PID result shapes.

    Inputs: pid_result is any object returned by the PID callable.
    Outputs: dict containing at least the "unq1" PID component.
    """

    if isinstance(pid_result, dict) and "pid" in pid_result:
        return pid_result["pid"]
    if isinstance(pid_result, (tuple, list)) and pid_result:
        return pid_result[0]
    if isinstance(pid_result, dict):
        return pid_result
    raise TypeError(f"Unsupported PID result type: {type(pid_result)!r}")


def _to_float(value: Any) -> float:
    """Convert numeric scalar-like values to float.

    Inputs: value is a Python, NumPy, or tensor scalar-like object.
    Outputs: float scalar value.
    """

    return float(value.item() if hasattr(value, "item") else value)


def _top_rows(rows: list[dict[str, Any]], beam_width: int) -> list[dict[str, Any]]:
    """Keep highest-unique rows, de-duplicated by subset.

    Inputs: rows is evaluated subset dicts; beam_width is max retained row count.
    Outputs: list of top rows sorted by unique information descending.
    """

    deduped = {row["subset"]: row for row in rows}
    return sorted(deduped.values(), key=lambda row: row["unique"], reverse=True)[:beam_width]


def _append_csv_row(
    path: str | Path | None,
    subset: tuple[int, ...],
    start: float,
    status: str,
    *,
    cmi_score: float | None = None,
    row: dict[str, Any] | None = None,
) -> None:
    """Append one compact row to a CSV file, creating the header when needed.

    Inputs: path is str, Path, or None; subset is PC indices; start is monotonic
        start; status is label; cmi_score and row are optional CSV values.
    Outputs: None, with a disk write only when path is not None.
    """

    if path is None:
        return
    pid = {} if row is None else row["pid"]
    cmi = cmi_score if row is None else row.get("cmi_score")
    unique = None if row is None else row["unique"]
    csv_row = {
        "subset": ";".join(str(i) for i in subset),
        "subset_size": len(subset),
        "cmi_score": "" if cmi is None else cmi,
        "unique": "" if unique is None else unique,
        "red": "" if "red" not in pid else _to_float(pid["red"]),
        "unq2": "" if "unq2" not in pid else _to_float(pid["unq2"]),
        "syn": "" if "syn" not in pid else _to_float(pid["syn"]),
        "elapsed_seconds": time.monotonic() - start,
        "status": status,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: csv_row.get(field, "") for field in CSV_FIELDS})


def _toy_pid(target: np.ndarray, source_1: np.ndarray, source_2: np.ndarray, **pid_kwargs: Any) -> dict[str, dict[str, float]]:
    """Return a tiny Gaussian-CMI-based PID-like result for local smoke runs.

    Inputs: target/source_1/source_2 are 2D np.ndarray objects; pid_kwargs ignored.
    Outputs: dict containing pid["unq1"] and sibling PID fields.
    """

    del pid_kwargs
    unique = _gaussian_cmi_bits(target, source_1, source_2)
    return {"pid": {"red": 0.0, "unq1": unique, "unq2": 0.0, "syn": 0.0}, "mi": {}}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    unique_signal = rng.standard_normal((n, 1))
    toy_target = unique_signal + 0.25 * rng.standard_normal((n, 1))
    toy_source_1 = np.hstack([unique_signal + 0.25 * rng.standard_normal((n, 1)), rng.standard_normal((n, 3))])
    toy_source_2 = rng.standard_normal((n, 2))
    result = run_pid_pc_subset_search(toy_target, toy_source_1, toy_source_2, pid_callable=_toy_pid, max_subset_size=2, beam_width=2)
    print({"status": result["status"], "best_subset": result["best_subset"], "best_unique": result["best_unique"]})
