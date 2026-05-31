#!/usr/bin/env python3
"""Small Python wrapper for JWKay/PID/IdepGauss.R."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_IDEP_URL = "https://raw.githubusercontent.com/JWKay/PID/main/IdepGauss.R"
DEFAULT_OUTPUT = "evil_twin_idep_results.csv"


R_CODE = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Expected: p output_csv idep_url local_idep_file verbose", call. = FALSE)
}

p <- as.integer(args[[1]])
output_csv <- args[[2]]
idep_url <- args[[3]]
local_idep_file <- args[[4]]
verbose <- args[[5]] == "1"

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

sizes <- c(p, p, p)
rows <- rbind(
  run_case("Sonic", make_sonic_sigma(p), sizes),
  run_case("Shadow", make_shadow_sigma(p), sizes)
)

write.csv(rows, output_csv, row.names = FALSE)
cat("Wrote ", output_csv, "\n", sep = "")
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Idep evil-twin example.")
    parser.add_argument("--p", type=int, default=1, help="Dimension of each block.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--idep-url", default=DEFAULT_IDEP_URL, help="IdepGauss.R URL.")
    parser.add_argument("--local-idep", default="IdepGauss.R", help="Local IdepGauss.R fallback.")
    parser.add_argument("--rscript", help="Path to Rscript. Defaults to PATH lookup.")
    parser.add_argument("--verbose", action="store_true", help="Print R PID objects.")
    parser.add_argument("--keep-r-driver", action="store_true", help="Keep the temporary R script.")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    if args.p < 1:
        print("error: --p must be a positive integer", file=sys.stderr)
        return 2

    try:
        rscript = find_rscript(args.rscript)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = absolute_path(args.output)
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
            ],
            check=False,
        )
        return completed.returncode
    finally:
        if driver_path and args.keep_r_driver:
            print(f"Kept temporary R driver: {driver_path}")
        elif driver_path:
            driver_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
