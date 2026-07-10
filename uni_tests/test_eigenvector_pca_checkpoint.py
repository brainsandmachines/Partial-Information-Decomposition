import csv
from pathlib import Path

import numpy as np
import pytest

from Simulations.PCA_rank.eigenvector_pca import (
    EigenvectorPCACVResult,
    eigenvector_pca_cv,
    fit_pca_loadings_svd,
)


def _small_matrix() -> np.ndarray:
    """
    Build a deterministic small matrix for checkpoint tests.

    Input:
        None.

    Output:
        X: np.ndarray of shape (6, 4), deterministic float test data.
    """

    rng = np.random.default_rng(123)
    return rng.normal(size=(6, 4))


def _run_cv(
    X: np.ndarray,
    checkpoint_csv_path: Path | None = None,
) -> EigenvectorPCACVResult:
    """
    Run eigenvector PCA CV with fixed test parameters.

    Input:
        X: np.ndarray of shape (n_samples, n_features), test data.
        checkpoint_csv_path: Path or None, optional checkpoint CSV path.

    Output:
        result: EigenvectorPCACVResult from eigenvector_pca_cv.
    """

    return eigenvector_pca_cv(
        X,
        max_components=2,
        pca_fit_fn=fit_pca_loadings_svd,
        center=True,
        scale=True,
        method_pca="SVD",
        checkpoint_csv_path=checkpoint_csv_path,
    )


def _read_checkpoint(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """
    Read a checkpoint CSV into field names and rows.

    Input:
        path: Path, checkpoint CSV file.

    Output:
        checkpoint: tuple containing fieldnames and row dictionaries.
    """

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def _write_checkpoint(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """
    Write selected checkpoint rows back to a CSV file.

    Input:
        path: Path, checkpoint CSV file to overwrite.
        fieldnames: list[str], CSV columns to write.
        rows: list[dict[str, str]], checkpoint rows to keep.

    Output:
        None. The file at path is replaced by the provided rows.
    """

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_checkpointed_run_matches_baseline(tmp_path: Path) -> None:
    """
    Check that checkpointing does not change eigenvector PCA CV output.

    Input:
        tmp_path: Path, pytest temporary directory.

    Output:
        None. Assertions verify selected components, PRESS, MSEP, and row count.
    """

    X = _small_matrix()
    baseline = _run_cv(X)
    checkpoint_path = tmp_path / "eigenvector_pca_cv.csv"

    checkpointed = _run_cv(X, checkpoint_path)

    assert checkpointed.selected_n_components == baseline.selected_n_components
    np.testing.assert_allclose(checkpointed.press, baseline.press)
    np.testing.assert_allclose(checkpointed.msep, baseline.msep)
    _, rows = _read_checkpoint(checkpoint_path)
    assert len(rows) == X.shape[0]


def test_partial_checkpoint_resumes_without_duplicate_rows(tmp_path: Path) -> None:
    """
    Check that a partial checkpoint resumes missing samples only.

    Input:
        tmp_path: Path, pytest temporary directory.

    Output:
        None. Assertions verify final output and unique completed sample rows.
    """

    X = _small_matrix()
    baseline = _run_cv(X)
    checkpoint_path = tmp_path / "partial_eigenvector_pca_cv.csv"
    _run_cv(X, checkpoint_path)
    fieldnames, rows = _read_checkpoint(checkpoint_path)
    _write_checkpoint(checkpoint_path, fieldnames, rows[:2])

    resumed = _run_cv(X, checkpoint_path)

    np.testing.assert_allclose(resumed.press, baseline.press)
    np.testing.assert_allclose(resumed.msep, baseline.msep)
    _, resumed_rows = _read_checkpoint(checkpoint_path)
    sample_indices = [row["sample_index"] for row in resumed_rows]
    assert len(resumed_rows) == X.shape[0]
    assert len(sample_indices) == len(set(sample_indices))


def test_checkpoint_rejects_changed_data_and_parameters(tmp_path: Path) -> None:
    """
    Check that strict checkpoint metadata prevents unsafe resume.

    Input:
        tmp_path: Path, pytest temporary directory.

    Output:
        None. Assertions verify ValueError on changed data and parameters.
    """

    X = _small_matrix()
    checkpoint_path = tmp_path / "strict_eigenvector_pca_cv.csv"
    _run_cv(X, checkpoint_path)

    changed_X = X.copy()
    changed_X[0, 0] += 0.01
    with pytest.raises(ValueError, match="data_hash"):
        _run_cv(changed_X, checkpoint_path)

    with pytest.raises(ValueError, match="scale"):
        eigenvector_pca_cv(
            X,
            max_components=2,
            pca_fit_fn=fit_pca_loadings_svd,
            center=True,
            scale=False,
            method_pca="SVD",
            checkpoint_csv_path=checkpoint_path,
        )
