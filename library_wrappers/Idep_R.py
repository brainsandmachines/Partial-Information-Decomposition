#!/usr/bin/env python3
"""Run the evil twin Gaussian PID examples through JWKay/PID/IdepGauss.R.

This is a Python wrapper around R code. It creates a temporary R driver,
downloads/sources IdepGauss.R, calls idepGM(sizes, Sigma) with the full
covariance matrix ordered as c(X1, X2, T), prints debugging output, and writes
evil_twin_idep_results.csv.

No third-party Python dependencies are required. You do need R with Rscript.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_IDEP_URL = "https://raw.githubusercontent.com/JWKay/PID/main/IdepGauss.R"
DEFAULT_OUTPUT = "evil_twin_idep_results.csv"


R_DRIVER = r'''
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Expected arguments: p, output_csv, idep_url, local_idep_file",
       call. = FALSE)
}

p <- as.integer(args[[1]])
output_csv <- args[[2]]
idep_url <- args[[3]]
local_idep_file <- args[[4]]

if (is.na(p) || p < 1L) {
  stop("p must be a positive integer.", call. = FALSE)
}

load_idep_gauss <- function(url, local_file) {
  tmp <- tempfile("IdepGauss_", fileext = ".R")
  loaded <- tryCatch({
    message("Downloading and sourcing ", url)
    download.file(url, destfile = tmp, mode = "wb", quiet = FALSE)
    source(tmp, local = globalenv())
    TRUE
  }, error = function(e) {
    warning("Could not load IdepGauss.R from URL: ", conditionMessage(e),
            call. = FALSE)
    FALSE
  })

  if (!loaded && nzchar(local_file) && file.exists(local_file)) {
    message("Falling back to local ", local_file)
    source(local_file, local = globalenv())
    loaded <- TRUE
  }

  if (!loaded) {
    stop("Could not load IdepGauss.R from the URL or local fallback.",
         call. = FALSE)
  }

  if (!exists("idepGM", mode = "function", envir = globalenv())) {
    stop("idepGM() was not loaded from IdepGauss.R", call. = FALSE)
  }
}

atoms <- c("unique_X1", "unique_X2", "redundancy", "synergy")
sizes <- c(p, p, p)

block_names <- function(p) {
  if (p == 1L) {
    c("X1", "X2", "T")
  } else {
    c(paste0("X1_", seq_len(p)),
      paste0("X2_", seq_len(p)),
      paste0("T_", seq_len(p)))
  }
}

expand_scalar_cov <- function(sigma_scalar, p) {
  sigma <- kronecker(sigma_scalar, diag(1, p))
  dimnames(sigma) <- list(block_names(p), block_names(p))
  sigma
}

make_sonic_sigma <- function(p) {
  # Sonic latent model:
  # X1 = R + N + U1
  # X2 = R + N + U2
  # T  = R + U1 + U2 + Nt
  # U2 is present in both X2 and T, but absent from X1, so X2 has a genuine
  # latent-mechanistic private signal about T.
  v_r <- 0.5
  v_n <- 2.5
  v_u1 <- 2.5
  v_u2 <- 0.5
  v_nt <- 1.0

  sigma_scalar <- matrix(c(
    v_r + v_n + v_u1, v_r + v_n,        v_r + v_u1,
    v_r + v_n,        v_r + v_n + v_u2, v_r + v_u2,
    v_r + v_u1,       v_r + v_u2,       v_r + v_u1 + v_u2 + v_nt
  ), nrow = 3, byrow = TRUE)

  expand_scalar_cov(sigma_scalar, p)
}

make_shadow_sigma <- function(p) {
  # Shadow latent model:
  # X1 = R + N + U1 + E1
  # X2 = R + N + E2
  # T  = R + U1 + Nt + Et
  # X2 and T only share R. X2 has no latent component that is both present in
  # T and absent from X1, even though the observable covariance matches Sonic.
  v_r <- 1.0
  v_n <- 2.0
  v_u1 <- 2.0
  v_nt <- 1.0
  v_e1 <- 0.5
  v_e2 <- 0.5
  v_et <- 0.5

  sigma_scalar <- matrix(c(
    v_r + v_n + v_u1 + v_e1, v_r + v_n,        v_r + v_u1,
    v_r + v_n,              v_r + v_n + v_e2, v_r,
    v_r + v_u1,             v_r,              v_r + v_u1 + v_nt + v_et
  ), nrow = 3, byrow = TRUE)

  expand_scalar_cov(sigma_scalar, p)
}

assert_valid_covariance <- function(case_name, sigma, tol = 1e-10) {
  symmetric <- isTRUE(all.equal(sigma, t(sigma), tolerance = tol,
                                check.attributes = FALSE))
  eigenvalues <- eigen(sigma, symmetric = TRUE, only.values = TRUE)$values
  positive_definite <- all(eigenvalues > tol)

  cat("\n", case_name, "\n", strrep("=", nchar(case_name)), "\n", sep = "")
  cat("Symmetric:", symmetric, "\n")
  cat("Eigenvalues:", paste(format(eigenvalues, digits = 10), collapse = ", "), "\n")
  cat("Positive definite:", positive_definite, "\n\n")

  if (!symmetric) {
    stop(case_name, " covariance matrix is not symmetric.", call. = FALSE)
  }
  if (!positive_definite) {
    stop(case_name, " covariance matrix is not positive definite.", call. = FALSE)
  }

  invisible(eigenvalues)
}

cov_to_cor <- function(mat) {
  d <- diag(1 / sqrt(diag(mat)), nrow = nrow(mat), ncol = ncol(mat))
  d %*% mat %*% d
}

manual_whitening <- function(sizes, mat) {
  # This follows idepGM exactly: first convert mat to correlation form, then
  # extract [X0, X1, Y] blocks and apply Cholesky whitening.
  mat <- cov_to_cor(mat)

  n0 <- sizes[1]
  n1 <- sizes[2]
  n2 <- sizes[3]

  ind0 <- seq_len(n0)
  ind1 <- seq.int(n0 + 1L, n0 + n1)
  ind2 <- seq.int(n0 + n1 + 1L, n0 + n1 + n2)

  S00 <- mat[ind0, ind0, drop = FALSE]
  S01 <- mat[ind0, ind1, drop = FALSE]
  S02 <- mat[ind0, ind2, drop = FALSE]
  S11 <- mat[ind1, ind1, drop = FALSE]
  S12 <- mat[ind1, ind2, drop = FALSE]
  S22 <- mat[ind2, ind2, drop = FALSE]

  InvSq00 <- backsolve(chol(S00), diag(1, n0))
  InvSq11 <- backsolve(chol(S11), diag(1, n1))
  InvSq22 <- backsolve(chol(S22), diag(1, n2))

  P <- t(InvSq00) %*% S01 %*% InvSq11
  Q <- t(InvSq00) %*% S02 %*% InvSq22
  R <- t(InvSq11) %*% S12 %*% InvSq22

  list(P = P, Q = Q, R = R)
}

print_matrix <- function(label, mat) {
  cat(label, "\n", sep = "")
  print(round(mat, 10))
  cat("\n")
}

name_pid_output <- function(pid_output) {
  if (!is.list(pid_output) || !all(c("idep", "mmi") %in% names(pid_output))) {
    stop("idepGM() returned an unexpected object.", call. = FALSE)
  }

  idep <- as.numeric(pid_output$idep)
  mmi <- as.numeric(pid_output$mmi)

  if (length(idep) != length(atoms) || length(mmi) != length(atoms)) {
    stop("idepGM() returned PID vectors with an unexpected length.", call. = FALSE)
  }

  names(idep) <- atoms
  names(mmi) <- atoms
  list(idep = idep, mmi = mmi)
}

run_case <- function(case_name, sigma, sizes) {
  assert_valid_covariance(case_name, sigma)
  print_matrix("Covariance matrix Sigma ordered as c(X1, X2, T):", sigma)

  whitened <- manual_whitening(sizes, sigma)
  print_matrix("Manual whitening P = whitened cov(X1, X2):", whitened$P)
  print_matrix("Manual whitening Q = whitened cov(X1, T):", whitened$Q)
  print_matrix("Manual whitening R = whitened cov(X2, T):", whitened$R)

  pid <- name_pid_output(idepGM(sizes, sigma))

  cat("Idep PID:\n")
  print(pid$idep)
  cat("\nMMI PID:\n")
  print(pid$mmi)
  cat("\n")

  rows <- data.frame(
    case = case_name,
    atom = atoms,
    idep_value = unname(pid$idep),
    mmi_value = unname(pid$mmi),
    stringsAsFactors = FALSE
  )

  list(sigma = sigma, whitened = whitened, pid = pid, rows = rows)
}

all_equal <- function(x, y, tol = 1e-10) {
  isTRUE(all.equal(x, y, tolerance = tol, check.attributes = FALSE))
}

load_idep_gauss(idep_url, local_idep_file)

cases <- list(
  Sonic = make_sonic_sigma(p),
  Shadow = make_shadow_sigma(p)
)

results <- lapply(names(cases), function(case_name) {
  run_case(case_name, cases[[case_name]], sizes)
})
names(results) <- names(cases)

cat("Cross-case checks\n")
cat("=================\n")
same_sigma <- all_equal(results$Sonic$sigma, results$Shadow$sigma)
same_p <- all_equal(results$Sonic$whitened$P, results$Shadow$whitened$P)
same_q <- all_equal(results$Sonic$whitened$Q, results$Shadow$whitened$Q)
same_r <- all_equal(results$Sonic$whitened$R, results$Shadow$whitened$R)
same_idep <- all_equal(results$Sonic$pid$idep, results$Shadow$pid$idep)
same_mmi <- all_equal(results$Sonic$pid$mmi, results$Shadow$pid$mmi)

cat("Identical Sigma:", same_sigma, "\n")
cat("Identical P:", same_p, "\n")
cat("Identical Q:", same_q, "\n")
cat("Identical R:", same_r, "\n")
cat("Identical Idep PID:", same_idep, "\n")
cat("Identical MMI PID:", same_mmi, "\n\n")

# Expected failure mode:
# Sonic and Shadow have different latent-mechanistic interpretations, but they
# induce the same observable Gaussian covariance matrix. Gaussian Idep and MMI
# only see that covariance matrix, so they cannot distinguish the two cases.
if (same_sigma) {
  cat("Expected Gaussian failure mode: Sonic and Shadow have the same covariance,\n")
  cat("so Gaussian Idep and MMI must assign the same PID values to both cases.\n\n")
}

csv_rows <- do.call(rbind, lapply(results, `[[`, "rows"))
write.csv(csv_rows, output_csv, row.names = FALSE)
cat("Wrote ", output_csv, "\n", sep = "")
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evil twin Gaussian PID examples by calling R idepGM()."
    )
    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="Dimension of each variable block. Defaults to scalar p=1.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--idep-url",
        default=DEFAULT_IDEP_URL,
        help="URL for JWKay/PID/IdepGauss.R.",
    )
    parser.add_argument(
        "--local-idep",
        default="IdepGauss.R",
        help="Local fallback path for IdepGauss.R if download fails.",
    )
    parser.add_argument(
        "--rscript",
        default=None,
        help="Path to Rscript. If omitted, the script searches PATH.",
    )
    parser.add_argument(
        "--keep-r-driver",
        action="store_true",
        help="Keep the temporary R driver for debugging.",
    )
    return parser.parse_args()


def resolve_rscript(user_value: str | None) -> str:
    if user_value:
        candidate = Path(user_value)
        if not candidate.exists():
            raise FileNotFoundError(f"Rscript path does not exist: {candidate}")
        return str(candidate)

    found = shutil.which("Rscript")
    if not found:
        raise FileNotFoundError(
            "Could not find Rscript on PATH. Install R or pass --rscript "
            "with the full path to Rscript.exe."
        )
    return found


def main() -> int:
    args = parse_args()

    if args.p < 1:
        print("--p must be a positive integer.", file=sys.stderr)
        return 2

    try:
        rscript = resolve_rscript(args.rscript)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_idep = Path(args.local_idep)
    if not local_idep.is_absolute():
        local_idep = Path.cwd() / local_idep

    driver_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_evil_twin_idep.R", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(R_DRIVER)
            driver_path = Path(handle.name)

        cmd = [
            rscript,
            str(driver_path),
            str(args.p),
            str(output_path),
            args.idep_url,
            str(local_idep),
        ]
        completed = subprocess.run(cmd, cwd=Path.cwd())
        return completed.returncode
    finally:
        if driver_path is not None:
            if args.keep_r_driver:
                print(f"Kept temporary R driver: {driver_path}")
            else:
                try:
                    driver_path.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
