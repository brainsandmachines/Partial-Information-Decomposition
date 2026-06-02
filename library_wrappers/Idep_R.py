#!/usr/bin/env python3
"""Small Python wrapper for JWKay/PID/IdepGauss.R."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from wrapper_utils import BASE_PID_COLUMNS, SIMPLE_GAUSSIAN_SIZES, print_pid_result, write_simple_gaussian_covariance


DEFAULT_IDEP_URL = "https://raw.githubusercontent.com/JWKay/PID/main/IdepGauss.R"
DEFAULT_OUTPUT = "evil_twin_idep_results.csv"


R_CODE = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
  stop("Expected: p output_csv idep_url local_idep_file verbose matrix_csv sizes_csv",
       call. = FALSE)
}

p <- as.integer(args[[1]])
output_csv <- args[[2]]
idep_url <- args[[3]]
local_idep_file <- args[[4]]
verbose <- args[[5]] == "1"
matrix_csv <- args[[6]]
sizes_csv <- args[[7]]

if (is.na(p) || p < 1L) {
  stop("p must be a positive integer.", call. = FALSE)
}

load_idep_gauss <- function(url, local_file) {
  if (nzchar(local_file) && file.exists(local_file)) {
    source(local_file, local = globalenv())
  } else {
    tmp <- tempfile("IdepGauss_", fileext = ".R")
    download.file(url, destfile = tmp, mode = "wb", quiet = TRUE)
    source(tmp, local = globalenv())
  }

  if (!exists("idepGM", mode = "function", envir = globalenv())) {
    stop("idepGM() was not loaded.", call. = FALSE)
  }
}

names_for_p <- function(p) {
  if (p == 1L) return(c("X1", "X2", "T"))
  c(paste0("X1_", seq_len(p)), paste0("X2_", seq_len(p)), paste0("T_", seq_len(p)))
}

expand_scalar_cov <- function(sigma_scalar, p) {
  sigma <- kronecker(sigma_scalar, diag(1, p))
  dimnames(sigma) <- list(names_for_p(p), names_for_p(p))
  sigma
}

make_sonic_sigma <- function(p) {
  v_r <- 0.5; v_n <- 2.5; v_u1 <- 2.5; v_u2 <- 0.5; v_nt <- 1.0
  expand_scalar_cov(matrix(c(
    v_r + v_n + v_u1, v_r + v_n,        v_r + v_u1,
    v_r + v_n,        v_r + v_n + v_u2, v_r + v_u2,
    v_r + v_u1,       v_r + v_u2,       v_r + v_u1 + v_u2 + v_nt
  ), nrow = 3, byrow = TRUE), p)
}

make_shadow_sigma <- function(p) {
  v_r <- 1.0; v_n <- 2.0; v_u1 <- 2.0; v_nt <- 1.0
  v_e1 <- 0.5; v_e2 <- 0.5; v_et <- 0.5
  expand_scalar_cov(matrix(c(
    v_r + v_n + v_u1 + v_e1, v_r + v_n,        v_r + v_u1,
    v_r + v_n,              v_r + v_n + v_e2, v_r,
    v_r + v_u1,             v_r,              v_r + v_u1 + v_nt + v_et
  ), nrow = 3, byrow = TRUE), p)
}

check_covariance <- function(name, sigma) {
  eigenvalues <- eigen(sigma, symmetric = TRUE, only.values = TRUE)$values
  if (!isTRUE(all.equal(sigma, t(sigma), check.attributes = FALSE))) {
    stop(name, " covariance is not symmetric.", call. = FALSE)
  }
  if (!all(eigenvalues > 1e-10)) {
    stop(name, " covariance is not positive definite.", call. = FALSE)
  }
}

row_from_pid <- function(case_name, definition, values) {
  u1 <- unname(values["unique_X1"])
  u2 <- unname(values["unique_X2"])
  red <- unname(values["redundancy"])
  syn <- unname(values["synergy"])
  data.frame(
    case = case_name,
    pid_definition = definition,
    unique_source1 = u1,
    unique_source2 = u2,
    redundancy = red,
    synergy = syn,
    I_source1_target = u1 + red,
    I_source2_target = u2 + red,
    joint_mutual_information = u1 + u2 + red + syn,
    interaction_information = syn - red,
    stringsAsFactors = FALSE
  )
}

run_case <- function(case_name, sigma, sizes) {
  check_covariance(case_name, sigma)
  pid <- idepGM(sizes, sigma)
  names(pid$idep) <- c("unique_X1", "unique_X2", "redundancy", "synergy")
  names(pid$mmi) <- c("unique_X1", "unique_X2", "redundancy", "synergy")

  if (verbose) {
    cat("\n", case_name, "\n", sep = "")
    print(pid)
  }

  rbind(
    row_from_pid(case_name, "Idep", pid$idep),
    row_from_pid(case_name, "MMI", pid$mmi)
  )
}

load_idep_gauss(idep_url, local_idep_file)

if (nzchar(matrix_csv)) {
  sizes <- as.integer(strsplit(sizes_csv, ",", fixed = TRUE)[[1]])
  sigma <- as.matrix(read.csv(matrix_csv, header = FALSE, check.names = FALSE))
  rows <- run_case("InputCov", sigma, sizes)
} else {
  sizes <- c(p, p, p)
  rows <- rbind(
    run_case("Sonic", make_sonic_sigma(p), sizes),
    run_case("Shadow", make_shadow_sigma(p), sizes)
  )
}

write.csv(rows, output_csv, row.names = FALSE)
cat("Wrote ", output_csv, "\n", sep = "")
'''


def parse_sizes(value: str) -> tuple[int, int, int]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sizes must look like 1,1,1") from exc
    if len(sizes) != 3 or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes must contain three positive integers")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Idep/MMI on a covariance CSV or on the built-in examples.",
        epilog="Example: python library_wrappers/Idep_R.py --example simple-gaussian --output /tmp/idep_simple.csv",
    )
    parser.add_argument("--example", choices=("simple-gaussian",), help="Run a small built-in Gaussian example.")
    parser.add_argument("--p", type=int, default=1, help="Dimension of each block.")
    parser.add_argument("--matrix-csv", type=Path, help="No-header covariance/correlation CSV ordered as source1,source2,target.")
    parser.add_argument("--sizes", type=parse_sizes, help="Comma-separated source1,source2,target dimensions.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--idep-url", default=DEFAULT_IDEP_URL, help="IdepGauss.R URL.")
    parser.add_argument("--local-idep", default="IdepGauss.R", help="Local IdepGauss.R fallback.")
    parser.add_argument("--rscript", help="Path to Rscript. Defaults to PATH lookup.")
    parser.add_argument("--verbose", action="store_true", help="Print R PID objects.")
    parser.add_argument("--keep-r-driver", action="store_true", help="Keep the temporary R script.")
    return parser.parse_args()


def simple_example_args() -> argparse.Namespace:
    """Small debug example: run Idep/MMI on the shared 1D Gaussian covariance."""
    return argparse.Namespace(
        example="simple-gaussian",
        p=1,
        matrix_csv=None,
        sizes=None,
        output=None,
        idep_url=DEFAULT_IDEP_URL,
        local_idep="IdepGauss.R",
        rscript=None,
        verbose=False,
        keep_r_driver=False,
    )


def csv_shape(path: Path) -> tuple[int, int]:
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


def validate_matrix_args(args: argparse.Namespace) -> None:
    if args.matrix_csv is None and args.sizes is None:
        return
    if args.matrix_csv is None or args.sizes is None:
        raise ValueError("--matrix-csv and --sizes must be supplied together")
    if not args.matrix_csv.exists():
        raise FileNotFoundError(f"matrix CSV does not exist: {args.matrix_csv}")
    total = sum(args.sizes)
    shape = csv_shape(args.matrix_csv)
    if shape != (total, total):
        raise ValueError(f"matrix shape must be {(total, total)}, got {shape}: {args.matrix_csv}")


def find_rscript(value: str | None) -> str:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return str(path.resolve())
        found = shutil.which(value)
        if found:
            return found
        raise FileNotFoundError(f"Rscript was not found: {value}")

    found = shutil.which("Rscript")
    if not found:
        raise FileNotFoundError("Rscript was not found on PATH. Pass --rscript.")
    return found


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def print_idep_rows(path: Path) -> None:
    """Print Idep/MMI CSV rows in the same debug style as the Python wrappers."""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            print_pid_result(row, BASE_PID_COLUMNS)


def apply_simple_gaussian_example(args: argparse.Namespace, matrix_csv: Path) -> None:
    if args.matrix_csv is not None or args.sizes is not None:
        raise ValueError("--example simple-gaussian cannot be combined with --matrix-csv or --sizes")
    write_simple_gaussian_covariance(matrix_csv)
    args.matrix_csv = matrix_csv
    args.sizes = SIMPLE_GAUSSIAN_SIZES
    if args.output == DEFAULT_OUTPUT:
        args.output = "simple_gaussian_idep_results.csv"


def main() -> int:
    args = simple_example_args() if len(sys.argv) == 1 else parse_args()
    if args.p < 1:
        print("error: --p must be a positive integer", file=sys.stderr)
        return 2

    try:
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if args.example == "simple-gaussian":
            temp_dir = tempfile.TemporaryDirectory(prefix="idep_simple_gaussian_")
            apply_simple_gaussian_example(args, Path(temp_dir.name) / "simple_gaussian_covariance_1_1_1.csv")
        validate_matrix_args(args)
        rscript = find_rscript(args.rscript)
    except (FileNotFoundError, ValueError) as exc:
        if "temp_dir" in locals() and temp_dir is not None:
            temp_dir.cleanup()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = absolute_path(args.output) if args.output is not None else Path(temp_dir.name) / "simple_gaussian_idep_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    local_idep = absolute_path(args.local_idep)

    driver_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_idep.R", delete=False, encoding="utf-8") as handle:
            handle.write(R_CODE)
            driver_path = Path(handle.name)

        completed = subprocess.run(
            [
                rscript,
                str(driver_path),
                str(args.p),
                str(output),
                args.idep_url,
                str(local_idep),
                "1" if args.verbose else "0",
                "" if args.matrix_csv is None else str(args.matrix_csv.expanduser().resolve()),
                "" if args.sizes is None else ",".join(map(str, args.sizes)),
            ],
            check=False,
        )
        if completed.returncode == 0 and args.output is None:
            print_idep_rows(output)
        return completed.returncode
    finally:
        if "temp_dir" in locals() and temp_dir is not None:
            temp_dir.cleanup()
        if driver_path and args.keep_r_driver:
            print(f"Kept temporary R driver: {driver_path}")
        elif driver_path:
            driver_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
