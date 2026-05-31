#!/usr/bin/env python3
"""Python CLI bridge for JWKay/PID's IGFuns.R.

The wrapper keeps the original R implementation intact and calls it through
Rscript. Inputs are passed as command-line scalars or CSV matrix files, and
the R result is returned as JSON with values in bits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


INF_NAMES = [
    "I_source1_target",
    "I_source2_target",
    "I_source1_target_given_source2",
    "I_source2_target_given_source1",
    "joint_mutual_information",
    "interaction_information",
]

PID_NAMES = [
    "unique_source1",
    "unique_source2",
    "redundant",
    "synergistic",
]

EVIL_TWIN_EXAMPLE = {
    "mode": "pqr",
    "sizes": (1, 1, 1),
    "p": 0.6837634587578276,
    "q": 0.6030226891555273,
    "r": 0.2519763153394848,
}

DEFAULT_EXAMPLE_OUTPUT = Path("evil_twin_result.json")


def parse_sizes(value: str) -> tuple[int, int, int]:
    """Parse and validate sizes as n_source1,n_source2,n_target.

    Args:
        value: Comma-separated positive integers, such as "1,1,1" or "2,2,1".

    Returns:
        A three-integer tuple in the variable order expected by IGFuns.R.

    Raises:
        argparse.ArgumentTypeError: If the string is malformed or non-positive.
    """
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--sizes must be comma-separated positive integers, for example 1,1,1"
        ) from exc

    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "--sizes must have exactly three positive integers: source1,source2,target"
        )
    return parts


def parse_correlation(value: str) -> float:
    """Parse one scalar correlation for the univariate Gaussian mode.

    Args:
        value: Numeric correlation coefficient.

    Returns:
        The coefficient as a float.

    Raises:
        argparse.ArgumentTypeError: If the value is not numeric or outside [-1, 1].
    """
    try:
        corr = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("correlations must be numeric") from exc

    if corr < -1.0 or corr > 1.0:
        raise argparse.ArgumentTypeError("correlations must be in [-1, 1]")
    return corr


def numeric_csv_shape(path: Path) -> tuple[int, int]:
    """Validate a no-header numeric CSV and return its matrix shape.

    Args:
        path: CSV file path. Rows are observations of matrix rows; columns are
            comma-separated numeric entries. Header rows are not supported.

    Returns:
        A (row_count, column_count) tuple.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ValueError: If the CSV is empty, ragged, or contains non-numeric cells.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    row_count = 0
    col_count: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            values = [cell.strip() for cell in row]
            if col_count is None:
                col_count = len(values)
            elif len(values) != col_count:
                raise ValueError(
                    f"{path} is ragged: row {row_number} has {len(values)} columns, "
                    f"expected {col_count}"
                )
            for col_number, cell in enumerate(values, start=1):
                try:
                    float(cell)
                except ValueError as exc:
                    raise ValueError(
                        f"{path} has a non-numeric value at row {row_number}, "
                        f"column {col_number}: {cell!r}"
                    ) from exc
            row_count += 1

    if row_count == 0 or col_count is None:
        raise ValueError(f"{path} is empty or contains only blank rows")
    return row_count, col_count


def require_shape(path: Path, expected: tuple[int, int], label: str) -> None:
    """Ensure a CSV matrix has the required dimensions before R is called.

    Args:
        path: CSV matrix path.
        expected: Required (rows, columns) shape.
        label: Human-readable matrix name used in error messages.

    Raises:
        ValueError: If the CSV shape does not match the expected shape.
    """
    observed = numeric_csv_shape(path)
    if observed != expected:
        raise ValueError(f"{label} must have shape {expected}, got {observed}: {path}")


def source_search_paths(pid_repo: Path | None) -> list[Path]:
    """Return IGFuns.R locations checked when --r-source is not supplied.

    Args:
        pid_repo: Optional path to a cloned JWKay/PID repository.

    Returns:
        Candidate IGFuns.R paths in priority order.
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates: list[Path] = []

    if pid_repo is not None:
        candidates.append(pid_repo.expanduser() / "IGFuns.R")

    env_repo = os.environ.get("PID_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser() / "IGFuns.R")

    candidates.extend(
        [
            script_dir / "IGFuns.R",
            script_dir / "PID" / "IGFuns.R",
            script_dir.parent / "PID" / "IGFuns.R",
            cwd / "IGFuns.R",
            cwd / "PID" / "IGFuns.R",
            cwd.parent / "PID" / "IGFuns.R",
        ]
    )
    return candidates


def resolve_r_source(r_source: Path | None, pid_repo: Path | None) -> Path:
    """Resolve IGFuns.R from an explicit path or a cloned JWKay/PID repo.

    Args:
        r_source: Optional direct path to IGFuns.R.
        pid_repo: Optional cloned repository root. The wrapper reads
            pid_repo/IGFuns.R when provided.

    Returns:
        Existing IGFuns.R path.

    Raises:
        FileNotFoundError: If no source file can be found.
    """
    if r_source is not None:
        direct = r_source.expanduser()
        if direct.exists():
            return direct.resolve()
        raise FileNotFoundError(f"IGFuns.R source file does not exist: {direct}")

    searched = source_search_paths(pid_repo)
    for candidate in searched:
        if candidate.exists():
            return candidate.resolve()

    formatted = "\n  ".join(str(path) for path in searched)
    raise FileNotFoundError(
        "IGFuns.R source file does not exist. Pass --r-source, pass --pid-repo, "
        "set PID_REPO, or place the cloned PID repo next to the wrapper.\n"
        f"Searched:\n  {formatted}"
    )


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate mode-specific CLI inputs without changing the mathematical data.

    Args:
        args: Parsed command-line arguments.

    Raises:
        FileNotFoundError: If the R source or any input CSV is missing.
        ValueError: If required arguments are absent or matrix shapes are invalid.
    """
    if not args.r_source.exists():
        raise FileNotFoundError(f"IGFuns.R source file does not exist: {args.r_source}")

    if args.mode is None:
        raise ValueError("either --mode or --example is required")

    if args.mode == "univariate":
        missing = [name for name in ("p", "q", "r") if getattr(args, name) is None]
        if missing:
            raise ValueError(f"univariate mode requires: {', '.join('--' + m for m in missing)}")
        if args.plot is not None:
            raise ValueError("--plot is only used by covariance and pqr modes")
        return

    if args.sizes is None:
        raise ValueError(f"{args.mode} mode requires --sizes")
    n0, n1, n2 = args.sizes

    if args.mode == "covariance":
        if args.matrix_csv is None:
            raise ValueError("covariance mode requires --matrix-csv")
        total = n0 + n1 + n2
        require_shape(args.matrix_csv, (total, total), "covariance matrix")
        return

    if args.mode == "pqr":
        csv_values = (args.p_csv, args.q_csv, args.r_csv)
        scalar_values = (args.p, args.q, args.r)
        if all(path is not None for path in csv_values):
            require_shape(args.p_csv, (n0, n1), "P block")
            require_shape(args.q_csv, (n0, n2), "Q block")
            require_shape(args.r_csv, (n1, n2), "R block")
            return
        if any(path is not None for path in csv_values):
            raise ValueError("pqr mode needs all of --p-csv, --q-csv, and --r-csv")
        if args.sizes == (1, 1, 1) and all(value is not None for value in scalar_values):
            return
        raise ValueError(
            "pqr mode requires either all P/Q/R CSV files, or scalar --p --q --r "
            "with --sizes 1,1,1"
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


def apply_example(args: argparse.Namespace) -> None:
    """Apply a named built-in example to parsed CLI arguments.

    Args:
        args: Parsed command-line arguments. Output, R source, and Rscript flags
            are left unchanged; mathematical inputs are filled from the example.

    Returns:
        None. The namespace is updated in place.

    Raises:
        ValueError: If the user mixes an example with explicit mode/input values.
    """
    if args.example is None:
        return

    supplied_inputs = [
        flag
        for flag, value in (
            ("--mode", args.mode),
            ("--p", args.p),
            ("--q", args.q),
            ("--r", args.r),
            ("--sizes", args.sizes),
            ("--matrix-csv", args.matrix_csv),
            ("--p-csv", args.p_csv),
            ("--q-csv", args.q_csv),
            ("--r-csv", args.r_csv),
        )
        if value is not None
    ]
    if supplied_inputs:
        raise ValueError(
            f"--example {args.example} supplies its own inputs; remove "
            f"{', '.join(supplied_inputs)} or run without --example"
        )

    if args.example == "evil-twin":
        args.mode = EVIL_TWIN_EXAMPLE["mode"]
        args.sizes = EVIL_TWIN_EXAMPLE["sizes"]
        args.p = EVIL_TWIN_EXAMPLE["p"]
        args.q = EVIL_TWIN_EXAMPLE["q"]
        args.r = EVIL_TWIN_EXAMPLE["r"]
        return

    raise ValueError(f"Unknown example: {args.example}")


def apply_click_run_defaults(args: argparse.Namespace) -> None:
    """Make a bare IDE/terminal run execute the evil-twin example.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None. When no mode, example, or manual inputs are supplied, the namespace
        is updated to run the built-in evil-twin example and write a default JSON
        output file.
    """
    manual_inputs = (
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
    )
    if any(value is not None for value in manual_inputs):
        return

    args.example = "evil-twin"
    if args.output is None:
        args.output = DEFAULT_EXAMPLE_OUTPUT


def r_bridge_script() -> str:
    """Return the temporary base-R script that calls IGFuns.R.

    Args:
        None.

    Returns:
        R source code as a string. The script reads CLI arguments, sources
        IGFuns.R, calls the selected R function, and writes JSON without
        requiring jsonlite or any other R package.
    """
    return r'''
args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag, default = NA_character_) {
  pos <- match(flag, args)
  if (is.na(pos)) return(default)
  if (pos == length(args)) stop(paste("missing value for", flag))
  args[[pos + 1]]
}

split_sizes <- function(value) {
  as.integer(strsplit(value, ",", fixed = TRUE)[[1]])
}

read_matrix_csv <- function(path) {
  as.matrix(read.csv(path, header = FALSE, check.names = FALSE))
}

json_string <- function(value) {
  escaped <- gsub("\\\\", "\\\\\\\\", value)
  escaped <- gsub("\"", "\\\\\"", escaped)
  paste0("\"", escaped, "\"")
}

json_number <- function(value) {
  if (length(value) == 0 || is.na(value) || is.nan(value)) return("null")
  if (is.infinite(value)) return(ifelse(value > 0, "1e999", "-1e999"))
  sprintf("%.17g", as.numeric(value))
}

json_number_array <- function(values) {
  paste0("[", paste(vapply(values, json_number, character(1)), collapse = ","), "]")
}

json_named_numbers <- function(values, labels) {
  n <- min(length(values), length(labels))
  if (n == 0) return("{}")
  parts <- vapply(seq_len(n), function(i) {
    paste0(json_string(labels[[i]]), ":", json_number(values[[i]]))
  }, character(1))
  paste0("{", paste(parts, collapse = ","), "}")
}

call_with_plot_capture <- function(expr, plot_path) {
  if (!is.na(plot_path) && nzchar(plot_path)) {
    png(filename = plot_path, width = 900, height = 600)
    on.exit(dev.off(), add = TRUE)
    return(eval(expr, envir = parent.frame()))
  }

  temp_plot <- tempfile(fileext = ".png")
  png(filename = temp_plot, width = 900, height = 600)
  on.exit({
    dev.off()
    unlink(temp_plot)
  }, add = TRUE)
  eval(expr, envir = parent.frame())
}

mode <- arg_value("--mode")
source_path <- arg_value("--source")
output_path <- arg_value("--output")
plot_path <- arg_value("--plot", "")
source(source_path)

inf_labels <- c(
  "I_source1_target",
  "I_source2_target",
  "I_source1_target_given_source2",
  "I_source2_target_given_source1",
  "joint_mutual_information",
  "interaction_information"
)

pid_labels <- c(
  "unique_source1",
  "unique_source2",
  "redundant",
  "synergistic"
)

if (mode == "univariate") {
  p <- as.numeric(arg_value("--p"))
  q <- as.numeric(arg_value("--q"))
  r <- as.numeric(arg_value("--r"))
  result <- IG_GaussU_pqr(p, q, r)
  function_name <- "IG_GaussU_pqr"
} else if (mode == "covariance") {
  sizes <- split_sizes(arg_value("--sizes"))
  mat <- read_matrix_csv(arg_value("--matrix-csv"))
  result <- call_with_plot_capture(quote(IG_GaussM_Dat(sizes, mat)), plot_path)
  function_name <- "IG_GaussM_Dat"
} else if (mode == "pqr") {
  sizes <- split_sizes(arg_value("--sizes"))
  P <- read_matrix_csv(arg_value("--p-csv"))
  Q <- read_matrix_csv(arg_value("--q-csv"))
  R <- read_matrix_csv(arg_value("--r-csv"))
  result <- call_with_plot_capture(quote(IG_GaussM_PQR(sizes, P, Q, R)), plot_path)
  function_name <- "IG_GaussM_PQR"
} else {
  stop(paste("unknown mode:", mode))
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
if (!is.na(plot_path) && nzchar(plot_path)) {
  fields <- c(fields, paste0(json_string("plot"), ":", json_string(normalizePath(plot_path, mustWork = FALSE))))
}

writeLines(paste0("{", paste(fields, collapse = ","), "}"), output_path)
'''


def resolve_rscript(executable: str) -> str:
    """Resolve the Rscript executable path used for the Python-to-R bridge.

    Args:
        executable: Either a command on PATH, such as "Rscript", or an absolute
            path to an Rscript executable.

    Returns:
        A path or command string that can be passed to subprocess.

    Raises:
        FileNotFoundError: If the executable cannot be found.
    """
    candidate = Path(executable)
    if candidate.exists():
        return str(candidate)

    resolved = shutil.which(executable)
    if resolved:
        return resolved

    raise FileNotFoundError(
        "Rscript was not found. Install R, add Rscript to PATH, or pass "
        "--rscript C:\\path\\to\\Rscript.exe."
    )


def selected_function_name(mode: str) -> str:
    """Map a wrapper mode to the IGFuns.R function it calls.

    Args:
        mode: One of "univariate", "covariance", or "pqr".

    Returns:
        The R function name for status messages.
    """
    return {
        "univariate": "IG_GaussU_pqr",
        "covariance": "IG_GaussM_Dat",
        "pqr": "IG_GaussM_PQR",
    }[mode]


def run_r(args: argparse.Namespace) -> dict:
    """Call IGFuns.R through Rscript and return the parsed JSON result.

    Args:
        args: Validated command-line arguments. Matrix inputs are passed as CSV
            paths; no whitening, normalization, or reordering is performed by
            Python. IGFuns.R performs its own covariance-to-correlation handling
            in covariance mode.

    Returns:
        A dictionary containing information quantities and PID components in bits.

    Raises:
        RuntimeError: If Rscript exits with an error or writes invalid JSON.
    """
    rscript = resolve_rscript(args.rscript)

    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pid_r_bridge_") as temp_dir:
        temp_path = Path(temp_dir)
        bridge_path = temp_path / "run_igfuns.R"
        result_path = temp_path / "result.json"
        bridge_path.write_text(r_bridge_script(), encoding="utf-8")

        command = [
            rscript,
            str(bridge_path),
            "--mode",
            args.mode,
            "--source",
            str(args.r_source),
            "--output",
            str(result_path),
            "--plot",
            "" if args.plot is None else str(args.plot),
        ]

        if args.mode == "univariate":
            command.extend(["--p", str(args.p), "--q", str(args.q), "--r", str(args.r)])
        elif args.mode == "covariance":
            command.extend(
                [
                    "--sizes",
                    ",".join(str(part) for part in args.sizes),
                    "--matrix-csv",
                    str(args.matrix_csv),
                ]
            )
        elif args.mode == "pqr":
            p_csv = args.p_csv
            q_csv = args.q_csv
            r_csv = args.r_csv
            if p_csv is None and q_csv is None and r_csv is None:
                p_csv = temp_path / "P.csv"
                q_csv = temp_path / "Q.csv"
                r_csv = temp_path / "R.csv"
                p_csv.write_text(f"{args.p}\n", encoding="utf-8")
                q_csv.write_text(f"{args.q}\n", encoding="utf-8")
                r_csv.write_text(f"{args.r}\n", encoding="utf-8")
            command.extend(
                [
                    "--sizes",
                    ",".join(str(part) for part in args.sizes),
                    "--p-csv",
                    str(p_csv),
                    "--q-csv",
                    str(q_csv),
                    "--r-csv",
                    str(r_csv),
                ]
            )

        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            details = "\n".join(
                part
                for part in (
                    "Rscript failed while running IGFuns.R.",
                    f"Command: {' '.join(command)}",
                    f"stdout:\n{completed.stdout.strip()}",
                    f"stderr:\n{completed.stderr.strip()}",
                )
                if part.strip()
            )
            raise RuntimeError(details)

        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Rscript did not produce valid JSON at {result_path}") from exc


def write_result(result: dict, output: Path | None) -> None:
    """Write the PID result either to JSON file or stdout.

    Args:
        result: Parsed result from R.
        output: Optional JSON output path. If absent, JSON is printed to stdout.

    Returns:
        None. The result is saved or printed.
    """
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote JSON output to {output}", file=sys.stderr)


def format_number(value: object) -> str:
    """Format one numeric result for an aligned terminal table.

    Args:
        value: Numeric value from the JSON result.

    Returns:
        A fixed-width decimal string when possible, otherwise a string version
        of the value.
    """
    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return str(value)


def print_named_row(title: str, values: dict[str, object], names: list[str]) -> None:
    """Print a compact two-line table for named scalar results.

    Args:
        title: Section title printed above the table.
        values: Mapping from result names to numeric values.
        names: Names to print, in display order.

    Returns:
        None.
    """
    widths = [
        max(len(name), len(format_number(values.get(name, ""))))
        for name in names
    ]
    header = " ".join(name.rjust(width) for name, width in zip(names, widths))
    row = " ".join(
        format_number(values.get(name, "")).rjust(width)
        for name, width in zip(names, widths)
    )
    print(f"\n{title}:")
    print(header)
    print(row)


def print_pretty_result(result: dict) -> None:
    """Print the IG PID result in a human-readable table.

    Args:
        result: Parsed JSON result returned by the R bridge. IGFuns.R returns
            IG PID values, not Idep/MMI values, so this prints the IG PID table
            and a second table with the mutual-information quantities.

    Returns:
        None.
    """
    pid_values = result.get("pid_bits", {})
    inf_values = result.get("inf_bits", {})

    print_named_row(
        "IG PID",
        pid_values,
        ["unique_source1", "unique_source2", "redundant", "synergistic"],
    )
    print_named_row(
        "Information terms",
        inf_values,
        [
            "I_source1_target",
            "I_source2_target",
            "joint_mutual_information",
            "interaction_information",
        ],
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Build and parse the command-line interface for the wrapper.

    Args:
        argv: Optional iterable of command-line tokens. Defaults to sys.argv.

    Returns:
        argparse.Namespace with normalized Path objects and mode-specific values.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run JWKay/PID IGFuns.R from Python and return PID values as JSON. "
            "With no arguments, runs the built-in evil-twin example."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("univariate", "covariance", "pqr"),
        help="Which IGFuns.R entry point to call.",
    )
    parser.add_argument(
        "--example",
        choices=("evil-twin",),
        help=(
            "Run a named built-in example. evil-twin uses the whitened "
            "Sonic/Shadow P/Q/R correlations from the covariance PDF."
        ),
    )
    parser.add_argument(
        "--r-source",
        type=Path,
        help=(
            "Path to IGFuns.R. If omitted, the wrapper searches common cloned "
            "JWKay/PID locations."
        ),
    )
    parser.add_argument(
        "--pid-repo",
        type=Path,
        help="Path to a cloned JWKay/PID repo; the wrapper uses PID_REPO/IGFuns.R.",
    )
    parser.add_argument(
        "--rscript",
        default=os.environ.get("RSCRIPT", "Rscript"),
        help="Rscript executable or full path. Defaults to RSCRIPT env var or Rscript.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional PNG path for the g(t) plot made by covariance and pqr modes.",
    )

    parser.add_argument("--p", type=parse_correlation, help="corr(source1, source2).")
    parser.add_argument("--q", type=parse_correlation, help="corr(source1, target).")
    parser.add_argument("--r", type=parse_correlation, help="corr(source2, target).")

    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        help="Comma-separated group sizes in source1,source2,target order.",
    )
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        help="No-header covariance/correlation CSV for covariance mode.",
    )
    parser.add_argument("--p-csv", type=Path, help="P block CSV for pqr mode.")
    parser.add_argument("--q-csv", type=Path, help="Q block CSV for pqr mode.")
    parser.add_argument("--r-csv", type=Path, help="R block CSV for pqr mode.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Validate inputs, call R, and write the PID result.

    Args:
        argv: Optional command-line token iterable, useful for smoke tests.

    Returns:
        Process exit code: 0 on success, 2 for validation/runtime errors.
    """
    try:
        args = parse_args(argv)
        apply_click_run_defaults(args)
        apply_example(args)
        args.r_source = resolve_r_source(args.r_source, args.pid_repo)
        validate_inputs(args)
        if args.example is not None:
            print(f"Using built-in example: {args.example}", file=sys.stderr)
        print(f"Using R source: {args.r_source}", file=sys.stderr)
        print(f"Calling R function: {selected_function_name(args.mode)}", file=sys.stderr)
        result = run_r(args)
        write_result(result, args.output)
        print_pretty_result(result)
        print("Produced PID output in bits.", file=sys.stderr)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
