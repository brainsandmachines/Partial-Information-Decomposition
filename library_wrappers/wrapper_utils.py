"""Shared helpers for Python PID wrapper scripts."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
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


def project_root() -> Path:
    """Return the thesis repository root for wrapper path discovery."""
    return Path(__file__).resolve().parents[1]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        normalized = path.expanduser()
        key = str(normalized)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def pid_repo_candidates(pid_repo: Path | str | None = None) -> list[Path]:
    """Return candidate JWKay/PID repo locations."""
    candidates: list[Path] = []
    if pid_repo:
        candidates.append(Path(pid_repo))
    for env_name in ("PID_REPO", "JWKAY_PID_REPO"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]))

    root = project_root()
    cwd = Path.cwd().resolve()
    bases = [root, cwd, cwd.parent, Path(__file__).resolve().parent]
    for base in bases:
        candidates.extend(
            [
                base / "external" / "PID",
                base / "PID",
            ]
        )

    return _dedupe_paths(candidates)


def pid_file_candidates(file_name: str, pid_repo: Path | str | None = None) -> list[Path]:
    """Return candidate paths for a file inside JWKay/PID."""
    return [candidate / file_name for candidate in pid_repo_candidates(pid_repo)]


def find_pid_repo(
    pid_repo: Path | str | None = None,
    required_files: tuple[str, ...] = ("IGFuns.R", "IdepGauss.R"),
) -> Path:
    """Find a JWKay/PID repo containing the required R files."""
    for candidate in pid_repo_candidates(pid_repo):
        if all((candidate / file_name).exists() for file_name in required_files):
            return candidate.resolve()

    searched = "\n  ".join(str(path) for path in pid_repo_candidates(pid_repo))
    required = ", ".join(required_files)
    raise FileNotFoundError(f"Could not find JWKay/PID with {required}. Searched:\n  {searched}")


def find_pid_file(file_name: str, pid_repo: Path | str | None = None) -> Path:
    """Find a file inside JWKay/PID."""
    for candidate in pid_file_candidates(file_name, pid_repo):
        if candidate.exists():
            return candidate.resolve()

    searched = "\n  ".join(str(path) for path in pid_file_candidates(file_name, pid_repo))
    raise FileNotFoundError(f"Could not find {file_name}. Searched:\n  {searched}")


def gpid_src_candidates(gpid_repo: Path | str | None = None) -> list[Path]:
    """Return candidate gpid src directories."""
    candidates: list[Path] = []
    if gpid_repo:
        candidates.append(Path(gpid_repo) / "src")
        candidates.append(Path(gpid_repo))
    for env_name in ("GPID_SRC", "GPID_REPO"):
        if os.environ.get(env_name):
            env_path = Path(os.environ[env_name])
            candidates.extend([env_path, env_path / "src"])

    root = project_root()
    cwd = Path.cwd().resolve()
    bases = [root, cwd, cwd.parent, Path(__file__).resolve().parent]
    for base in bases:
        candidates.extend(
            [
                base / "external" / "gpid" / "src",
                base / "gpid" / "src",
            ]
        )

    return _dedupe_paths(candidates)


def add_gpid_src_to_path(gpid_repo: Path | str | None = None) -> Path:
    """Find gpid/src and add it to sys.path for direct `import gpid`."""
    for candidate in gpid_src_candidates(gpid_repo):
        if (candidate / "gpid").is_dir():
            resolved = candidate.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved

    searched = "\n  ".join(str(path) for path in gpid_src_candidates(gpid_repo))
    raise FileNotFoundError(f"Could not find gpid src directory. Searched:\n  {searched}")


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
