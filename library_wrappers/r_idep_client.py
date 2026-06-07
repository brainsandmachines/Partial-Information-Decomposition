"""Programmatic Python client for JWKay/PID/IdepGauss.R."""

from __future__ import annotations

import csv
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .wrapper_utils import find_pid_file, find_rscript
except ImportError:  # pragma: no cover - script-style import fallback
    from wrapper_utils import find_pid_file, find_rscript


DEFAULT_IDEP_URL = "https://raw.githubusercontent.com/JWKay/PID/main/IdepGauss.R"
ATOMS = ("unique_X1", "unique_X2", "redundancy", "synergy")
BITS_TO_NATS = math.log(2.0)


R_DRIVER = r'''
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 5) {
  stop("Expected: covariance_csv, sizes_csv, output_csv, idep_url, local_idep",
       call. = FALSE)
}

covariance_csv <- args[[1]]
sizes_csv <- args[[2]]
output_csv <- args[[3]]
idep_url <- args[[4]]
local_idep <- args[[5]]

load_idep_gauss <- function(url, local_file) {
  loaded <- FALSE

  if (nzchar(local_file) && file.exists(local_file)) {
    message("Sourcing local ", local_file)
    source(local_file, local = globalenv())
    loaded <- TRUE
  }

  if (!loaded) {
    tmp <- tempfile("IdepGauss_", fileext = ".R")
    loaded <- tryCatch({
      message("Downloading and sourcing ", url)
      download.file(url, destfile = tmp, mode = "wb", quiet = TRUE)
      source(tmp, local = globalenv())
      TRUE
    }, error = function(e) {
      warning("Could not load IdepGauss.R from URL: ", conditionMessage(e),
              call. = FALSE)
      FALSE
    })
  }

  if (!loaded || !exists("idepGM", mode = "function", envir = globalenv())) {
    stop("Could not load idepGM() from IdepGauss.R.", call. = FALSE)
  }
}

Sigma <- as.matrix(read.csv(covariance_csv, header = FALSE, check.names = FALSE))
storage.mode(Sigma) <- "double"
sizes <- scan(sizes_csv, what = integer(), sep = ",", quiet = TRUE)

if (length(sizes) != 3) {
  stop("sizes must contain exactly three integers: n0,n1,n2.", call. = FALSE)
}
if (nrow(Sigma) != sum(sizes) || ncol(Sigma) != sum(sizes)) {
  stop("Sigma dimensions do not match sum(sizes).", call. = FALSE)
}

load_idep_gauss(idep_url, local_idep)

pid <- idepGM(sizes, Sigma)
atoms <- c("unique_X1", "unique_X2", "redundancy", "synergy")

out <- data.frame(
  atom = atoms,
  idep_value = as.numeric(pid$idep),
  mmi_value = as.numeric(pid$mmi),
  stringsAsFactors = FALSE
)
write.csv(out, output_csv, row.names = FALSE)
'''


@dataclass(frozen=True)
class RIdePResult:
    idep: dict[str, float]
    mmi: dict[str, float]
    stdout: str
    stderr: str


def _to_2d_float_rows(matrix: Any) -> list[list[float]]:
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().tolist()
    elif hasattr(matrix, "cpu") and hasattr(matrix, "numpy"):
        matrix = matrix.cpu().numpy().tolist()
    elif hasattr(matrix, "tolist"):
        matrix = matrix.tolist()

    rows = [list(row) for row in matrix]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Covariance matrix must be a non-empty rectangular matrix.")
    if len(rows) != len(rows[0]):
        raise ValueError("Covariance matrix must be square.")

    return [[float(value) for value in row] for row in rows]


def _write_matrix_csv(path: Path, rows: Sequence[Sequence[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _read_result_csv(
    path: Path,
    stdout: str,
    stderr: str,
    *,
    bits_to_nats: bool,
) -> RIdePResult:
    idep: dict[str, float] = {}
    mmi: dict[str, float] = {}
    scale = BITS_TO_NATS if bits_to_nats else 1.0

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            atom = row["atom"]
            idep[atom] = scale * float(row["idep_value"])
            mmi[atom] = scale * float(row["mmi_value"])

    missing = [atom for atom in ATOMS if atom not in idep or atom not in mmi]
    if missing:
        raise RuntimeError(f"R output is missing atoms: {missing}")

    return RIdePResult(idep=idep, mmi=mmi, stdout=stdout, stderr=stderr)


def run_idep_from_covariance(
    sigma: Any,
    sizes: Sequence[int],
    *,
    rscript: str | Path | None = None,
    idep_url: str = DEFAULT_IDEP_URL,
    local_idep: str | Path | None = "IdepGauss.R",
    bits_to_nats: bool = True,
    keep_temp: bool = False,
) -> RIdePResult:
    """Run R idepGM(sizes, sigma) and return named Idep/MMI atoms."""
    rows = _to_2d_float_rows(sigma)
    sizes = tuple(int(value) for value in sizes)
    if len(sizes) != 3:
        raise ValueError("sizes must contain exactly three integers.")
    if sum(sizes) != len(rows):
        raise ValueError("sum(sizes) must match the covariance dimension.")

    rscript_path = find_rscript(None if rscript is None else str(rscript))
    local_idep_path = _resolve_local_idep(local_idep)

    if keep_temp:
        temp_dir_obj = None
        temp_dir = Path(tempfile.mkdtemp(prefix="r_idep_"))
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="r_idep_")
        temp_dir = Path(temp_dir_obj.name)

    try:
        covariance_csv = temp_dir / "covariance.csv"
        sizes_csv = temp_dir / "sizes.csv"
        output_csv = temp_dir / "idep_output.csv"
        driver_r = temp_dir / "run_idep.R"

        _write_matrix_csv(covariance_csv, rows)
        sizes_csv.write_text(",".join(str(value) for value in sizes), encoding="utf-8")
        driver_r.write_text(R_DRIVER, encoding="utf-8")

        completed = subprocess.run(
            [
                rscript_path,
                str(driver_r),
                str(covariance_csv),
                str(sizes_csv),
                str(output_csv),
                idep_url,
                local_idep_path,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "R idepGM call failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        return _read_result_csv(
            output_csv,
            completed.stdout,
            completed.stderr,
            bits_to_nats=bits_to_nats,
        )
    finally:
        if keep_temp:
            print(f"Kept R Idep temporary directory: {temp_dir}")
        elif temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def run_idep_for_cases(
    case_covariances: Mapping[str, Any],
    sizes: Sequence[int],
    **kwargs: Any,
) -> dict[str, RIdePResult]:
    """Run R Idep/MMI for several named covariance matrices."""
    return {
        case_name: run_idep_from_covariance(sigma, sizes, **kwargs)
        for case_name, sigma in case_covariances.items()
    }


def atoms_as_ordered_values(values: Mapping[str, float]) -> list[float]:
    return [float(values[atom]) for atom in ATOMS]


def _resolve_local_idep(local_idep: str | Path | None) -> str:
    if local_idep is None:
        try:
            return str(find_pid_file("IdepGauss.R"))
        except FileNotFoundError:
            return ""

    path = Path(local_idep).expanduser()
    if path.exists():
        return str(path.resolve())
    if str(local_idep) == "IdepGauss.R":
        try:
            return str(find_pid_file("IdepGauss.R"))
        except FileNotFoundError:
            return str(path.resolve())
    return str(path.resolve())
