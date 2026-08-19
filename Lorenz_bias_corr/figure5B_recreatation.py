#!/usr/bin/env python3
"""Recreate Lorenz Figure 5B and add a Lorenz-corrected Eigen-PID curve."""

import csv
import operator
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LORENZ_ROOT = PROJECT_ROOT / "external" / "Sampling_bias_corrections_Syn_Red"
GPID_ROOT = PROJECT_ROOT / "external" / "gpid"
GPID_SOURCE_ROOT = GPID_ROOT / "src"
EIGEN_PID_ROOT = PROJECT_ROOT / "external" / "Gaussian_eig_PID"
EIGEN_PID_SOURCE_ROOT = EIGEN_PID_ROOT / "src"
LORENZ_FUNCTIONS_ROOT = LORENZ_ROOT / "Functions"

OUTPUT_ROOT = Path(os.environ.get("FIGURE5B_OUTPUT_ROOT", SCRIPT_DIR)).resolve()
OUTPUT_BASENAME = os.environ.get(
    "FIGURE5B_OUTPUT_BASENAME",
    "figure5b - eigenvalue_pid",
)
RESULT_DIR = OUTPUT_ROOT / "Results" / "Gauss"
RESULT_PATH = RESULT_DIR / f"{OUTPUT_BASENAME}.mat"
FIGURE_PATH = OUTPUT_ROOT / f"{OUTPUT_BASENAME}.svg"
TIMING_CSV_PATH = OUTPUT_ROOT / f"{OUTPUT_BASENAME}.csv"

EXPECTED_LORENZ_COMMIT = "728e55024227dfa2b1915bc7df56f54b34117f41"
EXPECTED_GPID_COMMIT = "179fd78ef426c34837e23d95c31db0293e74585d"
EXPECTED_EIGEN_PID_COMMIT = "72294d7b26f8b77a9663866a017f3167b4906a53"

NTRIALS = int(os.environ.get("FIGURE5B_NTRIALS", "256"))
N_REPETITIONS = int(os.environ.get("FIGURE5B_N_REPETITIONS", "100"))
BIAS_ITERATIONS = int(os.environ.get("FIGURE5B_BIAS_ITERATIONS", "20"))
CASE_TYPE = "bit_of_all"
ALPHAS = [1.0, 1.1, 1.2, 1.5, 2.0, 10000.0]
INFO_INDEX = 2
DEFAULT_DIMENSIONS = [4, 12, 20, 28, 36, 44, 50]
DIMENSIONS = [
    int(value)
    for value in os.environ.get(
        "FIGURE5B_DIMENSIONS",
        ",".join(str(value) for value in DEFAULT_DIMENSIONS),
    ).split(",")
]
ROUTINE_OUTPUT_ORDER = (0, 3, 4, 8, 9, 2, 1, 10, 11, 12, 7, 13, 14, 15)
METHOD_NAMES = (
    "resampling",
    "shuffle",
    "merged",
    "Venkatesh",
    "Eigen-PID + Lorenz",
)
METHOD_COLORS = (
    "#edb120",
    "#d95319",
    "#7e2f8e",
    "#77ac30",
    "#0072bd",
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lorenz-figure5b-matplotlib"),
)
for source_root in (
    PROJECT_ROOT,
    GPID_SOURCE_ROOT,
    EIGEN_PID_SOURCE_ROOT,
    LORENZ_FUNCTIONS_ROOT,
    LORENZ_ROOT,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.io as sio
    from gaussian_eigen_pid import exact_gauss_eigen_pid
    from toolz import compose

    import Simulations_Gaussian as simulations
except ImportError as exc:
    raise SystemExit(
        "Missing or incompatible Python dependency. Run this script with the "
        f"PID/Lorenz environment: {exc}"
    ) from exc


if __name__ == "__main__":
    if NTRIALS < 2:
        raise SystemExit("FIGURE5B_NTRIALS must be at least 2.")
    if N_REPETITIONS < 1:
        raise SystemExit("FIGURE5B_N_REPETITIONS must be at least 1.")
    if BIAS_ITERATIONS < 1:
        raise SystemExit("FIGURE5B_BIAS_ITERATIONS must be at least 1.")
    if not DIMENSIONS or any(dimension <= 0 for dimension in DIMENSIONS):
        raise SystemExit("FIGURE5B_DIMENSIONS must contain positive integers.")

    required_files = (
        LORENZ_ROOT / "Simulations_Gaussian.py",
        LORENZ_FUNCTIONS_ROOT / "tools.py",
        GPID_SOURCE_ROOT / "gpid" / "tilde_pid.py",
        EIGEN_PID_SOURCE_ROOT / "gaussian_eigen_pid" / "eigen_pid.py",
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise SystemExit("Missing required file(s): " + ", ".join(missing_files))

    repositories = (
        ("Lorenz", LORENZ_ROOT, EXPECTED_LORENZ_COMMIT),
        ("gPID", GPID_ROOT, EXPECTED_GPID_COMMIT),
        ("Gaussian Eigen-PID", EIGEN_PID_ROOT, EXPECTED_EIGEN_PID_COMMIT),
    )
    resolved_commits = {}
    for repository_name, repository_root, expected_commit in repositories:
        try:
            resolved_commit = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                f"Could not verify the pinned {repository_name} commit: {exc}"
            ) from exc
        if resolved_commit != expected_commit:
            raise SystemExit(
                f"Unexpected {repository_name} commit {resolved_commit}; "
                f"expected {expected_commit}."
            )
        resolved_commits[repository_name] = resolved_commit

    optimizer_simulation_estimator = simulations.exact_gauss_tilde_pid
    optimizer_tools_estimator = simulations.tools.exact_gauss_tilde_pid
    optimizer_routines = (
        ("resampling", simulations.tools.informative_bias_correction_routine),
        ("shuffle", simulations.tools.shuffle_subtr_bias_correction_routine),
        ("Venkatesh", simulations.tools.uniform_bias_correction_routine),
    )
    eigen_lorenz_routines = (
        simulations.tools.informative_bias_correction_routine,
        simulations.tools.shuffle_subtr_bias_correction_routine,
    )

    # No result -> components x trials x dimensions x plotted methods x alphas x repetitions.
    sampled_results = np.zeros(
        (
            14,
            1,
            len(DIMENSIONS),
            len(METHOD_NAMES),
            len(ALPHAS),
            N_REPETITIONS,
        )
    )
    # No result -> components x trials x dimensions x plotted methods x alphas.
    ground_truth_results = np.zeros(
        (14, 1, len(DIMENSIONS), len(METHOD_NAMES), len(ALPHAS))
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TIMING_CSV_PATH.open("w", newline="", encoding="utf-8") as timing_file:
        timing_writer = csv.DictWriter(
            timing_file,
            fieldnames=(
                "dimension",
                "ntrials",
                "n_repetitions",
                "bias_iterations",
                "resampling_seconds",
                "shuffle_seconds",
                "merged_seconds",
                "venkatesh_seconds",
                "eigenvalue_pid_seconds",
            ),
        )
        timing_writer.writeheader()
        timing_file.flush()

        for dimension_index, dimension in enumerate(DIMENSIONS):
            print(
                f"Running Figure 5B dimension {dimension} "
                f"({dimension_index + 1}/{len(DIMENSIONS)})",
                flush=True,
            )
            optimizer_outputs = {}
            optimizer_timings = {}

            simulations.exact_gauss_tilde_pid = optimizer_simulation_estimator
            simulations.tools.exact_gauss_tilde_pid = optimizer_tools_estimator
            for routine_name, routine in optimizer_routines:
                print(f"  {routine_name}", flush=True)
                adapted_routine = compose(
                    operator.itemgetter(*ROUTINE_OUTPUT_ORDER),
                    routine,
                )
                routine_started = time.perf_counter()
                sampled, ground_truth = simulations.run_simulation(
                    NTRIALS,
                    dimension,
                    BIAS_ITERATIONS,
                    N_REPETITIONS,
                    ALPHAS[INFO_INDEX],
                    routine_name,
                    adapted_routine,
                    CASE_TYPE,
                )
                optimizer_timings[routine_name] = (
                    time.perf_counter() - routine_started
                )
                optimizer_outputs[routine_name] = (sampled, ground_truth)

            merge_started = time.perf_counter()
            # Two (14, repetitions) results -> one merged (14, repetitions) result.
            optimizer_merged_sampled = (
                optimizer_outputs["resampling"][0]
                + optimizer_outputs["shuffle"][0]
            ) / 2.0
            # Two (14,) results -> one merged (14,) result.
            optimizer_merged_ground_truth = (
                optimizer_outputs["resampling"][1]
                + optimizer_outputs["shuffle"][1]
            ) / 2.0
            merge_overhead_seconds = time.perf_counter() - merge_started
            merged_seconds = (
                optimizer_timings["resampling"]
                + optimizer_timings["shuffle"]
                + merge_overhead_seconds
            )

            simulations.exact_gauss_tilde_pid = exact_gauss_eigen_pid
            simulations.tools.exact_gauss_tilde_pid = exact_gauss_eigen_pid
            eigen_started = time.perf_counter()
            eigen_outputs = []
            for routine in eigen_lorenz_routines:
                adapted_routine = compose(
                    operator.itemgetter(*ROUTINE_OUTPUT_ORDER),
                    routine,
                )
                eigen_outputs.append(
                    simulations.run_simulation(
                        NTRIALS,
                        dimension,
                        BIAS_ITERATIONS,
                        N_REPETITIONS,
                        ALPHAS[INFO_INDEX],
                        "Eigen-PID + Lorenz",
                        adapted_routine,
                        CASE_TYPE,
                    )
                )
            # Two (14, repetitions) results -> one Eigen-Lorenz (14, repetitions) result.
            eigen_merged_sampled = (
                eigen_outputs[0][0] + eigen_outputs[1][0]
            ) / 2.0
            # Two (14,) results -> one Eigen-Lorenz (14,) result.
            eigen_merged_ground_truth = (
                eigen_outputs[0][1] + eigen_outputs[1][1]
            ) / 2.0
            eigenvalue_pid_seconds = time.perf_counter() - eigen_started

            simulations.exact_gauss_tilde_pid = optimizer_simulation_estimator
            simulations.tools.exact_gauss_tilde_pid = optimizer_tools_estimator

            method_outputs = (
                optimizer_outputs["resampling"],
                optimizer_outputs["shuffle"],
                (optimizer_merged_sampled, optimizer_merged_ground_truth),
                optimizer_outputs["Venkatesh"],
                (eigen_merged_sampled, eigen_merged_ground_truth),
            )
            for method_index, (sampled, ground_truth) in enumerate(method_outputs):
                if sampled.shape != (14, N_REPETITIONS):
                    raise RuntimeError(
                        f"Unexpected sampled result shape {sampled.shape} for "
                        f"d={dimension}, method={METHOD_NAMES[method_index]}; "
                        f"expected (14, {N_REPETITIONS})."
                    )
                if ground_truth.shape != (14,):
                    raise RuntimeError(
                        f"Unexpected ground-truth shape {ground_truth.shape} for "
                        f"d={dimension}, method={METHOD_NAMES[method_index]}; "
                        "expected (14,)."
                    )

                # (14,) -> (6,) PID atoms and (7,) remaining finite fields.
                zero_information_ground_truth = (
                    np.isfinite(ground_truth[0])
                    and ground_truth[0] == 0.0
                    and np.isnan(ground_truth[1:7]).all()
                    and np.isfinite(ground_truth[7:]).all()
                )
                if zero_information_ground_truth:
                    # (14,) -> (14,) with undefined zero-information atoms set to zero.
                    ground_truth = ground_truth.copy()
                    ground_truth[1:7] = 0.0

                if not np.isfinite(sampled).all():
                    raise RuntimeError(
                        f"Non-finite sampled result for d={dimension}, "
                        f"method={METHOD_NAMES[method_index]}."
                    )
                if not np.isfinite(ground_truth).all():
                    raise RuntimeError(
                        f"Non-finite ground truth for d={dimension}, "
                        f"method={METHOD_NAMES[method_index]}."
                    )

                # Components x repetitions -> plotted components x repetitions.
                sampled_for_plot = sampled.copy()
                sampled_for_plot[1, :] = sampled[1, :] + sampled[2, :]
                sampled_for_plot[4, :] = sampled[5, :]
                sampled_for_plot[5, :] = sampled[6, :]

                # Components -> plotted components.
                ground_truth_for_plot = ground_truth.copy()
                ground_truth_for_plot[1] = ground_truth[1] + ground_truth[2]
                ground_truth_for_plot[4] = ground_truth[5]
                ground_truth_for_plot[5] = ground_truth[6]

                sampled_results[
                    :,
                    0,
                    dimension_index,
                    method_index,
                    INFO_INDEX,
                    :,
                ] = sampled_for_plot
                ground_truth_results[
                    :,
                    0,
                    dimension_index,
                    method_index,
                    INFO_INDEX,
                ] = ground_truth_for_plot

            timing_writer.writerow(
                {
                    "dimension": dimension,
                    "ntrials": NTRIALS,
                    "n_repetitions": N_REPETITIONS,
                    "bias_iterations": BIAS_ITERATIONS,
                    "resampling_seconds": f'{optimizer_timings["resampling"]:.9f}',
                    "shuffle_seconds": f'{optimizer_timings["shuffle"]:.9f}',
                    "merged_seconds": f"{merged_seconds:.9f}",
                    "venkatesh_seconds": f'{optimizer_timings["Venkatesh"]:.9f}',
                    "eigenvalue_pid_seconds": f"{eigenvalue_pid_seconds:.9f}",
                }
            )
            timing_file.flush()

    sio.savemat(
        RESULT_PATH,
        {
            "sampled_results": sampled_results,
            "GT_results": ground_truth_results,
            # Scalar dimensions -> (number of dimensions,).
            "M_vals": np.asarray(DIMENSIONS),
            # One scalar sample count -> (1,).
            "ntrials_vals": np.asarray([NTRIALS]),
            # Six scalar information parameters -> (6,).
            "alphas": np.asarray(ALPHAS),
            # Five method names -> (5,).
            "bias_titles": np.asarray(METHOD_NAMES, dtype=object),
        },
    )

    # No axes -> one row of three axes for joint, synergy, and redundancy bias.
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
    component_specs = (("Joint", 0), ("Syn", 5), ("Red", 4))
    for axis, (component_name, component_index) in zip(axes, component_specs):
        for method_index, (method_name, color) in enumerate(
            zip(METHOD_NAMES, METHOD_COLORS)
        ):
            # Results tensor -> dimensions x repetitions.
            method_samples = sampled_results[
                component_index,
                0,
                :,
                method_index,
                INFO_INDEX,
                :,
            ]
            # Ground-truth tensor -> dimensions.
            method_ground_truth = ground_truth_results[
                component_index,
                0,
                :,
                method_index,
                INFO_INDEX,
            ]
            # (dimensions, repetitions) and (dimensions,) -> dimensions x repetitions.
            bias_samples = method_samples - method_ground_truth[:, np.newaxis]
            # Dimensions x repetitions -> dimensions.
            mean_bias = np.mean(bias_samples, axis=1)
            if N_REPETITIONS > 1:
                # Dimensions x repetitions -> dimensions.
                three_sem = (
                    3.0
                    * np.std(bias_samples, axis=1, ddof=1)
                    / np.sqrt(N_REPETITIONS)
                )
            else:
                # One repetition per dimension -> zero-width dimensions vector.
                three_sem = np.zeros(len(DIMENSIONS))
            axis.fill_between(
                DIMENSIONS,
                mean_bias - three_sem,
                mean_bias + three_sem,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
            axis.plot(
                DIMENSIONS,
                mean_bias,
                color=color,
                linewidth=1.4,
                label=method_name,
            )

        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(
            np.floor(NTRIALS / (4 * 3)),
            color="black",
            linestyle=":",
            linewidth=0.8,
        )
        axis.set_title(component_name, fontweight="bold")
        axis.set_xlabel("Dimensions")
        axis.set_xlim(min(DIMENSIONS), max(DIMENSIONS))
        axis.set_ylim(-2.0, 8.0)
        axis.grid(False)

    axes[0].set_ylabel("Bias [bits]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        f"Gaussian Simulation (N = {NTRIALS})",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.92))
    figure.savefig(FIGURE_PATH, format="svg", bbox_inches="tight")
    plt.close(figure)

    print(f"Lorenz commit: {resolved_commits['Lorenz']}")
    print(f"gPID commit: {resolved_commits['gPID']}")
    print(f"Gaussian Eigen-PID commit: {resolved_commits['Gaussian Eigen-PID']}")
    print(
        "Figure 5B parameters: "
        f"scenario={CASE_TYPE}, N={NTRIALS}, repetitions={N_REPETITIONS}, "
        f"T={BIAS_ITERATIONS}, alpha={ALPHAS[INFO_INDEX]}, "
        f"dimensions={DIMENSIONS}"
    )
    print(
        "Merged timing is the resampling runtime plus shuffle runtime plus "
        "the averaging overhead."
    )
    print(f"MAT result: {RESULT_PATH}")
    print(f"Timing CSV: {TIMING_CSV_PATH}")
    print(f"Figure: {FIGURE_PATH}")
