"""Checkpoint persistence helpers for eigenvector PCA cross-validation."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np


CHECKPOINT_VERSION = "1"
CHECKPOINT_META_FIELDS = [
    "checkpoint_version",
    "data_hash",
    "n_samples",
    "n_features",
    "max_components",
    "center",
    "scale",
    "include_zero_components",
    "method_pca",
    "eps",
    "pca_fit_fn",
]

PCAFitFunction = Callable[[np.ndarray, int], np.ndarray]


def _checkpoint_fields(max_components: int) -> tuple[list[str], list[str]]:
    """Build the checkpoint CSV columns.

    Input:
        max_components: int, largest component count evaluated by CV.

    Output:
        tuple[list[str], list[str]] containing all field names and the
        PRESS-only field names.
    """
    press_fields = [f"press_{component}" for component in range(max_components + 1)]
    return [*CHECKPOINT_META_FIELDS, "sample_index", *press_fields], press_fields


def build_checkpoint_metadata(
    X: np.ndarray,
    max_components: int,
    pca_fit_fn: PCAFitFunction,
    center: bool,
    scale: bool,
    include_zero_components: bool,
    method_pca: str | None,
    eps: float,
) -> dict[str, str]:
    """Build strict metadata used to validate a resumable checkpoint.

    Input:
        X: np.ndarray with shape (n_samples, n_features).
        max_components: int, largest component count evaluated by CV.
        pca_fit_fn: callable that returns PCA loadings.
        center: bool controlling training-column centering.
        scale: bool controlling training-column standardization.
        include_zero_components: bool controlling the zero-component candidate.
        method_pca: str or None identifying the PCA fitting mode.
        eps: float numerical tolerance.

    Output:
        dict[str, str] of CSV-safe metadata values that must match to resume.
    """
    n_samples, n_features = X.shape
    X_hash = hashlib.sha256()
    contiguous = np.ascontiguousarray(X)
    X_hash.update(str(contiguous.dtype).encode("utf-8"))
    X_hash.update(json.dumps(contiguous.shape).encode("utf-8"))
    X_hash.update(contiguous.tobytes())
    fit_name = getattr(
        pca_fit_fn,
        "__qualname__",
        getattr(pca_fit_fn, "__name__", type(pca_fit_fn).__qualname__),
    )
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "data_hash": X_hash.hexdigest(),
        "n_samples": str(n_samples),
        "n_features": str(n_features),
        "max_components": str(max_components),
        "center": json.dumps(center),
        "scale": json.dumps(scale),
        "include_zero_components": json.dumps(include_zero_components),
        "method_pca": json.dumps(method_pca),
        "eps": repr(float(eps)),
        "pca_fit_fn": (
            f"{getattr(pca_fit_fn, '__module__', type(pca_fit_fn).__module__)}."
            f"{fit_name}"
        ),
    }


def load_eigenvector_pca_checkpoint(
    checkpoint_csv_path: str | Path,
    metadata: dict[str, str],
    max_components: int,
) -> dict[int, np.ndarray]:
    """Load a compatible eigenvector PCA checkpoint.

    Input:
        checkpoint_csv_path: str or Path pointing to the checkpoint CSV.
        metadata: dict[str, str] of expected run metadata.
        max_components: int, largest component count evaluated by CV.

    Output:
        dict[int, np.ndarray] mapping each completed sample index to its PRESS
        contribution with shape (max_components + 1,).
    """
    path = Path(checkpoint_csv_path)
    if not path.exists():
        return {}

    expected_fieldnames, press_fields = _checkpoint_fields(max_components)
    completed_press: dict[int, np.ndarray] = {}
    n_samples = int(metadata["n_samples"])

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != expected_fieldnames:
            raise ValueError(
                f"Checkpoint CSV has an incompatible schema: {path}. "
                f"Expected columns: {expected_fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            mismatch = next(
                (key for key, value in metadata.items() if row.get(key) != value),
                None,
            )
            if mismatch is not None:
                raise ValueError(
                    f"Checkpoint CSV is incompatible at row {row_number}: "
                    f"{mismatch}={row.get(mismatch)!r}, expected {metadata[mismatch]!r}."
                )

            sample_index = int(row["sample_index"])
            if sample_index < 0 or sample_index >= n_samples:
                raise ValueError(
                    f"Checkpoint CSV contains invalid sample_index={sample_index} "
                    f"at row {row_number}."
                )
            if sample_index in completed_press:
                raise ValueError(
                    f"Checkpoint CSV contains duplicate sample_index={sample_index}."
                )

            # CSV fields -> (max_components + 1,)
            completed_press[sample_index] = np.array(
                [float(row[field]) for field in press_fields],
                dtype=float,
            )
    return completed_press


def write_eigenvector_pca_checkpoint(
    checkpoint_csv_path: str | Path,
    metadata: dict[str, str],
    completed_press: dict[int, np.ndarray],
    max_components: int,
) -> None:
    """Atomically write completed eigenvector PCA checkpoint rows.

    Input:
        checkpoint_csv_path: str or Path pointing to the destination CSV.
        metadata: dict[str, str] of run metadata written on every row.
        completed_press: dict[int, np.ndarray] of completed sample results.
        max_components: int, largest component count evaluated by CV.

    Output:
        None. The destination CSV is atomically replaced.
    """
    path = Path(checkpoint_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames, press_fields = _checkpoint_fields(max_components)

    try:
        with temp_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for sample_index, press in sorted(completed_press.items()):
                writer.writerow({
                    **metadata,
                    "sample_index": str(sample_index),
                    **{
                        field: repr(float(value))
                        for field, value in zip(press_fields, press)
                    },
                })
            csv_file.flush()
            os.fsync(csv_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
