#!/usr/bin/env python3
"""Small Python wrapper for JWKay/PID/IGFuns.R."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from wrapper_utils import SIMPLE_GAUSSIAN_CASE, SIMPLE_GAUSSIAN_SIZES, write_simple_gaussian_covariance

try:
    from .wrapper_utils import csv_shape, find_pid_file, find_rscript, parse_sizes, pid_file_candidates
except ImportError:  # pragma: no cover - script-style import fallback
    from wrapper_utils import csv_shape, find_pid_file, find_rscript, parse_sizes, pid_file_candidates


EVIL_TWIN = {
    "mode": "pqr",
    "sizes": (1, 1, 1),
    "p": 0.6837634587578276,
    "q": 0.6030226891555273,
    "r": 0.2519763153394848,
}

PID_COLUMNS = [
    "unique_source1",
    "unique_source2",
    "redundancy",
    "synergy",
    "I_source1_target",
    "I_source2_target",
    "joint_mutual_information",
    "interaction_information",
]


R_CODE = r'''
args <- commandArgs(trailingOnly = TRUE)

arg <- function(flag, default = NA_character_) {
  pos <- match(flag, args)
  if (is.na(pos)) return(default)
  if (pos == length(args)) stop(paste("missing value for", flag), call. = FALSE)
  args[[pos + 1]]
}

json_string <- function(value) {
  value <- gsub("\\\\", "\\\\\\\\", value)
  value <- gsub("\"", "\\\\\"", value)
  paste0("\"", value, "\"")
}

json_number <- function(value) {
  if (length(value) == 0 || is.na(value) || is.nan(value)) return("null")
  if (is.infinite(value)) return(ifelse(value > 0, "1e999", "-1e999"))
  sprintf("%.17g", as.numeric(value))
}

json_named_numbers <- function(values, labels) {
  parts <- vapply(seq_along(labels), function(i) {
    paste0(json_string(labels[[i]]), ":", json_number(values[[i]]))
  }, character(1))
  paste0("{", paste(parts, collapse = ","), "}")
}

json_number_array <- function(values) {
  paste0("[", paste(vapply(values, json_number, character(1)), collapse = ","), "]")
}

read_matrix <- function(path) {
  as.matrix(read.csv(path, header = FALSE, check.names = FALSE))
}

with_optional_plot <- function(expr, plot_path) {
  if (is.na(plot_path) || !nzchar(plot_path)) {
    temp_plot <- tempfile(fileext = ".png")
    png(temp_plot, width = 900, height = 600)
    on.exit({ dev.off(); unlink(temp_plot) }, add = TRUE)
  } else {
    png(plot_path, width = 900, height = 600)
    on.exit(dev.off(), add = TRUE)
  }
  eval(expr, envir = parent.frame())
}

mode <- arg("--mode")
source(arg("--source"))

inf_labels <- c(
  "I_source1_target",
  "I_source2_target",
  "I_source1_target_given_source2",
  "I_source2_target_given_source1",
  "joint_mutual_information",
  "interaction_information"
)
pid_labels <- c("unique_source1", "unique_source2", "redundant", "synergistic")

if (mode == "univariate") {
  result <- IG_GaussU_pqr(as.numeric(arg("--p")), as.numeric(arg("--q")), as.numeric(arg("--r")))
  function_name <- "IG_GaussU_pqr"
} else if (mode == "covariance") {
  sizes <- as.integer(strsplit(arg("--sizes"), ",", fixed = TRUE)[[1]])
  result <- with_optional_plot(
    quote(IG_GaussM_Dat(sizes, read_matrix(arg("--matrix-csv")))),
    arg("--plot", "")
  )
  function_name <- "IG_GaussM_Dat"
} else if (mode == "pqr") {
  sizes <- as.integer(strsplit(arg("--sizes"), ",", fixed = TRUE)[[1]])
  result <- with_optional_plot(
    quote(IG_GaussM_PQR(sizes, read_matrix(arg("--p-csv")), read_matrix(arg("--q-csv")), read_matrix(arg("--r-csv")))),
    arg("--plot", "")
  )
  function_name <- "IG_GaussM_PQR"
} else {
  stop(paste("unknown mode:", mode), call. = FALSE)
}

fields <- c(
  paste0(json_string("mode"), ":", json_string(mode)),
  paste0(json_string("function"), ":", json_string(function_name)),
  paste0(json_string("inf_bits"), ":", json_named_numbers(result$inf, inf_labels)),
  paste0(json_string("pid_bits"), ":", json_named_numbers(result$pid, pid_labels))
)

if (!is.null(result$PD)) {
  fields <- c(fields, paste0(json_string("pd"), ":", json_string(result$PD)))
}
if (!is.null(result$feas)) {
  fields <- c(fields, paste0(json_string("feasible_t"), ":", json_number_array(result$feas)))
}
if (!is.null(result$t_star)) {
  fields <- c(fields, paste0(json_string("t_star"), ":", json_number(result$t_star)))
}
if (!is.na(arg("--plot", "")) && nzchar(arg("--plot", ""))) {
  fields <- c(fields, paste0(json_string("plot"), ":", json_string(normalizePath(arg("--plot"), mustWork = FALSE))))
}

writeLines(paste0("{", paste(fields, collapse = ","), "}"), arg("--output"))
'''


def parse_correlation(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("correlations must be numeric") from exc
    if not -1.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("correlations must be in [-1, 1]")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JWKay/PID IGFuns.R. With no arguments, runs the simple Gaussian example.",
        epilog="Example: python library_wrappers/IG_R.py --example simple-gaussian --output /tmp/ig_simple.json",
    )
    parser.add_argument("--mode", choices=("univariate", "covariance", "pqr"))
    parser.add_argument("--example", choices=("evil-twin", "simple-gaussian"))
    parser.add_argument("--r-source", type=Path, help="Path to IGFuns.R.")
    parser.add_argument("--pid-repo", type=Path, help="Path to a local JWKay/PID clone.")
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", "Rscript"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path, help="Optional PNG path for covariance or pqr mode.")
    parser.add_argument("--p", type=parse_correlation)
    parser.add_argument("--q", type=parse_correlation)
    parser.add_argument("--r", type=parse_correlation)
    parser.add_argument("--sizes", type=parse_sizes)
    parser.add_argument("--matrix-csv", type=Path)
    parser.add_argument("--p-csv", type=Path)
    parser.add_argument("--q-csv", type=Path)
    parser.add_argument("--r-csv", type=Path)
    return parser.parse_args()


def simple_example_args() -> argparse.Namespace:
    """Small debug example: run IG on the shared 1D Gaussian covariance."""
    return argparse.Namespace(
        mode=None,
        example="simple-gaussian",
        r_source=None,
        pid_repo=None,
        rscript=os.environ.get("RSCRIPT", "Rscript"),
        output=None,
        plot=None,
        p=None,
        q=None,
        r=None,
        sizes=None,
        matrix_csv=None,
        p_csv=None,
        q_csv=None,
        r_csv=None,
    )


def apply_example_defaults(args: argparse.Namespace) -> None:
    manual_values = [
        args.mode,
        args.example,
        args.p,
        args.q,
        args.r,
        args.sizes,
        args.matrix_csv,
        args.p_csv,
        args.q_csv,
        args.r_csv,
    ]
    if any(value is not None for value in manual_values):
        if args.example == "evil-twin":
            set_evil_twin_inputs(args)
        return

    args.example = "simple-gaussian"


def set_evil_twin_inputs(args: argparse.Namespace) -> None:
    if any(
        value is not None
        for value in (args.mode, args.p, args.q, args.r, args.sizes, args.matrix_csv, args.p_csv, args.q_csv, args.r_csv)
    ):
        raise ValueError("--example evil-twin cannot be combined with manual IG inputs")
    args.mode = EVIL_TWIN["mode"]
    args.sizes = EVIL_TWIN["sizes"]
    args.p = EVIL_TWIN["p"]
    args.q = EVIL_TWIN["q"]
    args.r = EVIL_TWIN["r"]


def set_simple_gaussian_inputs(args: argparse.Namespace, matrix_csv: Path) -> None:
    if any(
        value is not None
        for value in (args.mode, args.p, args.q, args.r, args.sizes, args.matrix_csv, args.p_csv, args.q_csv, args.r_csv)
    ):
        raise ValueError("--example simple-gaussian cannot be combined with manual IG inputs")
    write_simple_gaussian_covariance(matrix_csv)
    args.mode = "covariance"
    args.sizes = SIMPLE_GAUSSIAN_SIZES
    args.matrix_csv = matrix_csv
    args.output = args.output or Path("simple_gaussian_ig_result.json")


def ig_source_candidates(pid_repo: Path | None) -> list[Path]:
    return pid_file_candidates("IGFuns.R", pid_repo)


def find_ig_source(args: argparse.Namespace) -> Path:
    if args.r_source:
        source = args.r_source.expanduser()
        if source.exists():
            return source.resolve()
        raise FileNotFoundError(f"IGFuns.R does not exist: {source}")

    try:
        return find_pid_file("IGFuns.R", args.pid_repo)
    except FileNotFoundError:
        pass

    searched = "\n  ".join(str(path) for path in ig_source_candidates(args.pid_repo))
    raise FileNotFoundError(f"Could not find IGFuns.R. Searched:\n  {searched}")


def require_shape(path: Path, expected: tuple[int, int], label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    actual = csv_shape(path)
    if actual != expected:
        raise ValueError(f"{label} must have shape {expected}, got {actual}: {path}")


def validate_inputs(args: argparse.Namespace) -> None:
    if args.mode is None:
        raise ValueError("choose --mode or --example")

    if args.mode == "univariate":
        missing = [name for name in ("p", "q", "r") if getattr(args, name) is None]
        if missing:
            raise ValueError("univariate mode requires " + ", ".join(f"--{name}" for name in missing))
        if args.plot:
            raise ValueError("--plot is only for covariance and pqr mode")
        return

    if args.sizes is None:
        raise ValueError(f"{args.mode} mode requires --sizes")
    n1, n2, nt = args.sizes

    if args.mode == "covariance":
        if args.matrix_csv is None:
            raise ValueError("covariance mode requires --matrix-csv")
        total = n1 + n2 + nt
        require_shape(args.matrix_csv, (total, total), "covariance matrix")
        return

    pqr_csvs = (args.p_csv, args.q_csv, args.r_csv)
    if all(path is not None for path in pqr_csvs):
        require_shape(args.p_csv, (n1, n2), "P block")
        require_shape(args.q_csv, (n1, nt), "Q block")
        require_shape(args.r_csv, (n2, nt), "R block")
        return
    if any(path is not None for path in pqr_csvs):
        raise ValueError("pqr mode needs all three CSVs: --p-csv, --q-csv, --r-csv")
    if args.sizes == (1, 1, 1) and all(value is not None for value in (args.p, args.q, args.r)):
        return
    raise ValueError("pqr mode needs P/Q/R CSVs, or scalar --p --q --r with --sizes 1,1,1")


def run_r(args: argparse.Namespace) -> dict:
    rscript = find_rscript(args.rscript)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ig_r_") as temp_dir:
        temp = Path(temp_dir)
        driver = temp / "ig_bridge.R"
        output = temp / "result.json"
        driver.write_text(R_CODE, encoding="utf-8")

        command = [
            rscript,
            str(driver),
            "--mode",
            args.mode,
            "--source",
            str(args.r_source),
            "--output",
            str(output),
        ]

        if args.mode == "univariate":
            command += ["--p", str(args.p), "--q", str(args.q), "--r", str(args.r)]
        elif args.mode == "covariance":
            command += [
                "--sizes",
                ",".join(map(str, args.sizes)),
                "--matrix-csv",
                str(args.matrix_csv),
                "--plot",
                "" if args.plot is None else str(args.plot),
            ]
        else:
            p_csv, q_csv, r_csv = args.p_csv, args.q_csv, args.r_csv
            if p_csv is None:
                p_csv = write_scalar_csv(temp / "P.csv", args.p)
                q_csv = write_scalar_csv(temp / "Q.csv", args.q)
                r_csv = write_scalar_csv(temp / "R.csv", args.r)
            command += [
                "--sizes",
                ",".join(map(str, args.sizes)),
                "--p-csv",
                str(p_csv),
                "--q-csv",
                str(q_csv),
                "--r-csv",
                str(r_csv),
                "--plot",
                "" if args.plot is None else str(args.plot),
            ]

        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Rscript failed while running IGFuns.R.\n"
                + " ".join(command)
                + f"\nstdout:\n{result.stdout.strip()}\nstderr:\n{result.stderr.strip()}"
            )
        return json.loads(output.read_text(encoding="utf-8"))


def write_scalar_csv(path: Path, value: float) -> Path:
    path.write_text(f"{value}\n", encoding="utf-8")
    return path


def add_standard_table(result: dict) -> dict:
    pid = result["pid_bits"]
    info = result["inf_bits"]
    result["standard_pid_table"] = [
        {
            "case": "EvilTwin",
            "pid_definition": "IG",
            "unique_source1": pid["unique_source1"],
            "unique_source2": pid["unique_source2"],
            "redundancy": pid["redundant"],
            "synergy": pid["synergistic"],
            "I_source1_target": info["I_source1_target"],
            "I_source2_target": info["I_source2_target"],
            "joint_mutual_information": info["joint_mutual_information"],
            "interaction_information": info["interaction_information"],
        }
    ]
    return result


def write_result(result: dict, output: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote JSON output to {output}", file=sys.stderr)


def print_result(result: dict) -> None:
    row = result["standard_pid_table"][0]
    print("\nIG PID result, values in bits\n")
    for column in PID_COLUMNS:
        print(f"{column:28} {float(row[column]):.8f}")


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="ig_simple_gaussian_") as temp_dir:
            args = simple_example_args() if len(sys.argv) == 1 else parse_args()
            apply_example_defaults(args)
            if args.example == "simple-gaussian":
                set_simple_gaussian_inputs(args, Path(temp_dir) / "simple_gaussian_covariance_1_1_1.csv")
            args.r_source = find_ig_source(args)
            validate_inputs(args)
            print(f"Using R source: {args.r_source}", file=sys.stderr)
            result = add_standard_table(run_r(args))
            if args.example == "simple-gaussian":
                result["standard_pid_table"][0]["case"] = SIMPLE_GAUSSIAN_CASE
            write_result(result, args.output)
            print_result(result)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
