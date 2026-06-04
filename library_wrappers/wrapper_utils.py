"""Shared helpers for Python PID wrapper scripts."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_sizes(value: str) -> tuple[int, int, int]:
    """Parse comma-separated source1, source2, target block sizes."""
    try:
        sizes = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sizes must look like 1,1,1") from exc
    if len(sizes) != 3 or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes must contain three positive integers")
    return sizes


def csv_shape(path: Path) -> tuple[int, int]:
    """Return the shape of a numeric no-header CSV, rejecting ragged files."""
    rows = 0
    columns = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if columns is None:
                columns = len(row)
            elif len(row) != columns:
                raise ValueError(f"{path} is ragged at row {row_number}")
            for cell in row:
                float(cell)
            rows += 1
    if rows == 0 or columns is None:
        raise ValueError(f"{path} is empty")
    return rows, columns


def find_rscript(value: str | None = None) -> str:
    """Resolve Rscript from an explicit path/name or from PATH."""
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return str(path.resolve())
        found = shutil.which(value)
        if found:
            return found
        raise FileNotFoundError(f"Rscript was not found: {value}")

    found = shutil.which("Rscript")
    if found:
        return found
    raise FileNotFoundError("Rscript was not found on PATH. Pass --rscript.")
