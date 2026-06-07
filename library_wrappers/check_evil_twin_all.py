#!/usr/bin/env python3
"""Run the IG and Idep evil-twin checks and combine their PID tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path

try:
    from .wrapper_utils import find_pid_repo, find_rscript
except ImportError:  # pragma: no cover - script-style import fallback
    from wrapper_utils import find_pid_repo, find_rscript


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MATRIX_CSV = SCRIPT_DIR / "evil_twin_whitened_correlation_1_1_1.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evil_twin_check_outputs"

PID_COLUMNS = [
    "unique_source1",
    "unique_source2",
    "redundancy",
    "synergy",
    "I_source1_target",
    "I_source2_target",
    "joint_mutual_information",
    "union_information",
    "optimization_objective",
    "interaction_information",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all library-wrapper evil-twin checks.")
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--pid-repo", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run(command: list[str], label: str, verbose: bool) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if verbose or result.returncode != 0:
        print(f"\n[{label}]")
        print(" ".join(command))
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")


def load_ig_rows(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return normalize_rows(data["standard_pid_table"])


def load_idep_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if {row["case"] for row in rows} == {"InputCov"}:
        return normalize_rows(rows)

    # Sonic and Shadow have the same observable covariance. Keep one row per
    # PID definition after checking the two cases agree numerically.
    by_definition: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_definition.setdefault(row["pid_definition"], {})[row["case"]] = row

    combined = []
    for definition in ("Idep", "MMI"):
        sonic = by_definition[definition]["Sonic"]
        shadow = by_definition[definition]["Shadow"]
        for column in PID_COLUMNS:
            if abs(float(sonic[column]) - float(shadow[column])) > 1e-9:
                raise RuntimeError(f"{definition} differs between Sonic and Shadow in {column}")
        row = dict(sonic)
        row["case"] = "EvilTwin"
        combined.append(row)

    return normalize_rows(combined)


def load_single_row_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row in {path}, got {len(rows)}")
    return normalize_rows(rows)


def normalize_rows(rows: list[dict]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "pid_definition": str(row["pid_definition"]),
                **{
                    column: f"{float(row[column]):.8f}" if row.get(column) not in (None, "") else ""
                    for column in PID_COLUMNS
                },
            }
        )
    return normalized


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid_definition", *PID_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows: list[dict[str, str]], path: Path) -> None:
    labels = {
        "pid_definition": "Definition",
        "unique_source1": "U1",
        "unique_source2": "U2",
        "redundancy": "Red",
        "synergy": "Syn",
        "I_source1_target": "I(S1;T)",
        "I_source2_target": "I(S2;T)",
        "joint_mutual_information": "I(S1,S2;T)",
        "union_information": "Union",
        "optimization_objective": "Obj",
        "interaction_information": "II",
    }
    columns = ["pid_definition", *PID_COLUMNS]
    widths = [120, 95, 95, 90, 90, 110, 110, 135, 90, 90, 105]
    row_height = 38
    margin = 24
    title_height = 58
    header_height = 36
    width = sum(widths) + margin * 2
    height = margin * 2 + title_height + header_height + row_height * len(rows)

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{margin}" y="{margin + 22}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">Evil-twin PID comparison</text>',
        f'<text x="{margin}" y="{margin + 44}" font-family="Arial, sans-serif" font-size="13" fill="#475569">Values are in bits</text>',
    ]

    y = margin + title_height
    svg.append(f'<rect x="{margin}" y="{y}" width="{sum(widths)}" height="{header_height}" fill="#e5e7eb"/>')

    x = margin
    for column, column_width in zip(columns, widths):
        anchor = "start" if column == "pid_definition" else "end"
        text_x = x + 10 if anchor == "start" else x + column_width - 10
        svg.append(
            f'<text x="{text_x}" y="{y + 24}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#334155">{esc(labels[column])}</text>'
        )
        x += column_width

    y += header_height
    for row_index, row in enumerate(rows):
        fill = "#ffffff" if row_index % 2 == 0 else "#f1f5f9"
        svg.append(f'<rect x="{margin}" y="{y}" width="{sum(widths)}" height="{row_height}" fill="{fill}"/>')
        x = margin
        for column, column_width in zip(columns, widths):
            anchor = "start" if column == "pid_definition" else "end"
            text_x = x + 10 if anchor == "start" else x + column_width - 10
            weight = "700" if column == "pid_definition" else "400"
            svg.append(
                f'<text x="{text_x}" y="{y + 24}" text-anchor="{anchor}" '
                f'font-family="Arial, sans-serif" font-size="12" font-weight="{weight}" fill="#111827">{esc(row[column])}</text>'
            )
            x += column_width
        y += row_height

    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def print_table(rows: list[dict[str, str]]) -> None:
    labels = {"pid_definition": "Definition", **{name: name for name in PID_COLUMNS}}
    columns = ["pid_definition", *PID_COLUMNS]
    widths = {
        column: max(len(labels[column]), *(len(row[column]) for row in rows))
        for column in columns
    }

    print("\nEvil-twin PID comparison, values in bits\n")
    print("  ".join(labels[column].rjust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].rjust(widths[column]) for column in columns))


def main() -> int:
    args = parse_args()
    matrix_csv = args.matrix_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        pid_repo = find_pid_repo(args.pid_repo)
        rscript = find_rscript(args.rscript)
        require_file(matrix_csv, "matrix CSV")
        require_file(pid_repo / "IGFuns.R", "IGFuns.R")
        require_file(pid_repo / "IdepGauss.R", "IdepGauss.R")

        output_dir.mkdir(parents=True, exist_ok=True)
        ig_json = output_dir / "ig_evil_twin_covariance.json"
        ig_plot = output_dir / "ig_evil_twin_covariance.png"
        idep_csv = output_dir / "idep_evil_twin.csv"
        tilde_csv = output_dir / "tilde_evil_twin.csv"
        thin_csv = output_dir / "thin_evil_twin.csv"
        delta_csv = output_dir / "delta_evil_twin.csv"
        combined_csv = output_dir / "combined_pid_table.csv"
        combined_svg = output_dir / "combined_pid_table.svg"

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "IG_R.py"),
                "--mode",
                "covariance",
                "--sizes",
                "1,1,1",
                "--matrix-csv",
                str(matrix_csv),
                "--r-source",
                str(pid_repo / "IGFuns.R"),
                "--rscript",
                rscript,
                "--output",
                str(ig_json),
                "--plot",
                str(ig_plot),
            ],
            "IG_R.py",
            args.verbose,
        )

        rows = load_ig_rows(ig_json)

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "Idep_R.py"),
                "--matrix-csv",
                str(matrix_csv),
                "--sizes",
                "1,1,1",
                "--output",
                str(idep_csv),
                "--idep-url",
                (pid_repo / "IdepGauss.R").resolve().as_uri(),
                "--local-idep",
                str(pid_repo / "IdepGauss.R"),
                "--rscript",
                rscript,
            ],
            "Idep_R.py",
            args.verbose,
        )

        rows.extend(load_idep_rows(idep_csv))

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "Tilde_PID.py"),
                "--matrix-csv",
                str(matrix_csv),
                "--sizes",
                "1,1,1",
                "--output",
                str(tilde_csv),
                "--case",
                "InputCov",
            ],
            "Tilde_PID.py",
            args.verbose,
        )
        rows.extend(load_single_row_csv(tilde_csv))

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "Thin_PID.py"),
                "--matrix-csv",
                str(matrix_csv),
                "--sizes",
                "1,1,1",
                "--output",
                str(thin_csv),
                "--case",
                "InputCov",
            ],
            "Thin_PID.py",
            args.verbose,
        )
        rows.extend(load_single_row_csv(thin_csv))

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "Delta_PID.py"),
                "--matrix-csv",
                str(matrix_csv),
                "--sizes",
                "1,1,1",
                "--output",
                str(delta_csv),
                "--case",
                "InputCov",
            ],
            "Delta_PID.py",
            args.verbose,
        )
        rows.extend(load_single_row_csv(delta_csv))

        write_csv(rows, combined_csv)
        write_svg(rows, combined_svg)
        print_table(rows)
        print(f"\nWrote combined table: {combined_csv}")
        print(f"Wrote table image: {combined_svg}")
        print(f"Raw outputs: {output_dir}")
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
