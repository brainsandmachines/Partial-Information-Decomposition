#!/usr/bin/env python3
"""Recreate the Gaussian row of Lorenz et al. Figure 5 using the authors' code."""

import operator
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LORENZ_ROOT = PROJECT_ROOT / "external" / "Sampling_bias_corrections_Syn_Red"
GPID_ROOT = PROJECT_ROOT / "external" / "gpid"
GPID_SOURCE_ROOT = GPID_ROOT / "src"
LORENZ_FUNCTIONS_ROOT = LORENZ_ROOT / "Functions"
RESULT_DIR = SCRIPT_DIR / "Results" / "Gauss"
RESULT_PATH = RESULT_DIR / "Finalresults_across_M_and_ntrials_d80.mat"
FIGURE_PATH = SCRIPT_DIR / "Figure_5B_recreated.svg"
MATLAB_SESSION_NAME = "Figure5MATLAB"

EXPECTED_LORENZ_COMMIT = "728e55024227dfa2b1915bc7df56f54b34117f41"
EXPECTED_GPID_COMMIT = "179fd78ef426c34837e23d95c31db0293e74585d"

NTRIALS = 256
N_REPETITIONS = 100
BIAS_ITERATIONS = 20
CASE_TYPE = "bit_of_all"
ALPHAS = [1.0, 1.1, 1.2, 1.5, 2.0, 10000.0]
INFO_INDEX = 2
DIMENSIONS = list(range(4, 85, 8))
ROUTINE_OUTPUT_ORDER = (0, 3, 4, 8, 9, 2, 1, 10, 11, 12, 7, 13, 14, 15)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lorenz-figure5b-matplotlib"),
)
for source_root in (
    PROJECT_ROOT,
    GPID_SOURCE_ROOT,
    LORENZ_FUNCTIONS_ROOT,
    LORENZ_ROOT,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

try:
    import numpy as np
    import scipy.io as sio
    from toolz import compose

    import Simulations_Gaussian as simulations
    import matlab.engine
except ImportError as exc:
    raise SystemExit(
        "Missing or incompatible Python dependency. Use the Lorenz environment.yml "
        f"environment and make gPID importable: {exc}"
    ) from exc


if __name__ == "__main__":
    required_files = (
        LORENZ_ROOT / "Simulations_Gaussian.py",
        LORENZ_FUNCTIONS_ROOT / "tools.py",
        LORENZ_FUNCTIONS_ROOT / "PlotFigure5.m",
        GPID_SOURCE_ROOT / "gpid" / "tilde_pid.py",
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise SystemExit("Missing required file(s): " + ", ".join(missing_files))

    try:
        shared_matlab_sessions = matlab.engine.find_matlab()
    except matlab.engine.EngineError as exc:
        raise SystemExit(f"Could not search for shared MATLAB sessions: {exc}") from exc
    if MATLAB_SESSION_NAME not in shared_matlab_sessions:
        raise SystemExit(
            f"Shared MATLAB session {MATLAB_SESSION_NAME!r} was not found. Start "
            "an authenticated MATLAB session, run "
            f"matlab.engine.shareEngine('{MATLAB_SESSION_NAME}'), and keep that "
            "session open while this script runs."
        )
    try:
        matlab_session = matlab.engine.connect_matlab(MATLAB_SESSION_NAME)
    except matlab.engine.EngineError as exc:
        raise SystemExit(
            f"Could not connect to shared MATLAB session {MATLAB_SESSION_NAME!r}: {exc}"
        ) from exc

    try:
        lorenz_commit = subprocess.run(
            ["git", "-C", str(LORENZ_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        gpid_commit = subprocess.run(
            ["git", "-C", str(GPID_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not verify the pinned submodule commits: {exc}") from exc

    if lorenz_commit != EXPECTED_LORENZ_COMMIT:
        raise SystemExit(
            f"Unexpected Lorenz commit {lorenz_commit}; expected {EXPECTED_LORENZ_COMMIT}."
        )
    if gpid_commit != EXPECTED_GPID_COMMIT:
        raise SystemExit(
            f"Unexpected gPID commit {gpid_commit}; expected {EXPECTED_GPID_COMMIT}."
        )

    bias_routines = (
        ("No_bias_correction", simulations.tools.No_bias_correction_routine),
        (
            "informative_bias_correction",
            simulations.tools.informative_bias_correction_routine,
        ),
        ("shuffle_subtr", simulations.tools.shuffle_subtr_bias_correction_routine),
        ("Venka_bias_correction", simulations.tools.uniform_bias_correction_routine),
    )

    # No result -> components x trials x dimensions x methods x alphas x repetitions.
    sampled_results = np.zeros(
        (
            14,
            1,
            len(DIMENSIONS),
            len(bias_routines),
            len(ALPHAS),
            N_REPETITIONS,
        )
    )
    # No result -> components x trials x dimensions x methods x alphas.
    ground_truth_results = np.zeros(
        (14, 1, len(DIMENSIONS), len(bias_routines), len(ALPHAS))
    )

    for dimension_index, dimension in enumerate(DIMENSIONS):
        for routine_index, (routine_name, routine) in enumerate(bias_routines):
            print(
                f"Running Figure 5B: d={dimension}, method={routine_name}, "
                f"N={NTRIALS}, alpha={ALPHAS[INFO_INDEX]}"
            )

            # The adapter accepts the authors' four routine inputs and returns the
            # 14 scalar outputs expected by Simulations_Gaussian.run_simulation.
            adapted_routine = compose(
                operator.itemgetter(*ROUTINE_OUTPUT_ORDER),
                routine,
            )
            # No result -> sampled (14, 100) and ground truth (14,).
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

            if sampled.shape != (14, N_REPETITIONS):
                raise RuntimeError(
                    f"Unexpected sampled result shape {sampled.shape}; "
                    f"expected (14, {N_REPETITIONS})."
                )
            if ground_truth.shape != (14,):
                raise RuntimeError(
                    f"Unexpected ground-truth shape {ground_truth.shape}; expected (14,)."
                )

            # (14,) -> (6,) for PID atoms; (14,) -> (7,) for remaining fields.
            zero_information_ground_truth = (
                np.isfinite(ground_truth[0])
                and ground_truth[0] == 0.0
                and np.isnan(ground_truth[1:7]).all()
                and np.isfinite(ground_truth[7:]).all()
            )
            if zero_information_ground_truth:
                # !!! gPID returns NaN atoms after dividing 0 by 0 when joint MI is zero.
                # (14,) -> (14,).
                ground_truth = ground_truth.copy()
                ground_truth[1:7] = 0.0

            if not np.isfinite(sampled).all():
                raise RuntimeError(
                    f"Non-finite sampled result for d={dimension}, method={routine_name}."
                )
            if not np.isfinite(ground_truth).all():
                raise RuntimeError(
                    f"Non-finite ground truth for d={dimension}, method={routine_name}. "
                    "The clean upstream gPID must return finite Figure 5B atoms."
                )

            # components x repetitions -> PlotFigure5 components x repetitions.
            sampled_for_plot = sampled.copy()
            sampled_for_plot[1, :] = sampled[1, :] + sampled[2, :]
            sampled_for_plot[4, :] = sampled[5, :]
            sampled_for_plot[5, :] = sampled[6, :]

            # components -> PlotFigure5 components.
            ground_truth_for_plot = ground_truth.copy()
            ground_truth_for_plot[1] = ground_truth[1] + ground_truth[2]
            ground_truth_for_plot[4] = ground_truth[5]
            ground_truth_for_plot[5] = ground_truth[6]

            sampled_results[
                :,
                0,
                dimension_index,
                routine_index,
                INFO_INDEX,
                :,
            ] = sampled_for_plot
            ground_truth_results[
                :,
                0,
                dimension_index,
                routine_index,
                INFO_INDEX,
            ] = ground_truth_for_plot

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        RESULT_PATH,
        {
            "sampled_results": sampled_results,
            "GT_results": ground_truth_results,
            # 11 scalar dimensions -> (11,).
            "M_vals": np.asarray(DIMENSIONS),
            # One scalar sample count -> (1,).
            "ntrials_vals": np.asarray([NTRIALS]),
            # Six scalar information parameters -> (6,).
            "alphas": np.asarray(ALPHAS),
            # Four method names -> (4,).
            "bias_titles": np.asarray(
                [name for name, _ in bias_routines],
                dtype=object,
            ),
        },
    )

    matlab_functions = str(LORENZ_FUNCTIONS_ROOT).replace("'", "''")
    figure_path = str(FIGURE_PATH).replace("'", "''")
    matlab_command = (
        f"addpath(genpath('{matlab_functions}'));"
        "f=figure('Visible','off','Units','centimeters','Position',[1,1,18,6]);"
        "t=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');"
        "PlotFigure5(256,4:8:84,'',"
        "{'resample','shuffle','shuff-resamp','Venkatesh'},"
        "3,'',1,t,1,3,'Gauss',-4,10,true,false,false);"
        "lgd=findobj(f,'Type','Legend');"
        "if ~isempty(lgd),"
        "set(lgd,'String',{'resampling','shuffle','merged','Venkatesh'});"
        "end;"
        "title(t,'Gaussian Simulation (N = 256)','FontWeight','bold');"
        "set(f,'Renderer','painters');"
        f"exportgraphics(f,'{figure_path}','ContentType','vector');"
        "close(f);"
    )
    original_matlab_directory = matlab_session.pwd()
    try:
        matlab_session.cd(str(SCRIPT_DIR), nargout=0)
        matlab_session.eval(matlab_command, nargout=0)
    except (matlab.engine.EngineError, matlab.engine.MatlabExecutionError) as exc:
        raise SystemExit(
            f"MATLAB failed while calling the authors' PlotFigure5 function: {exc}"
        ) from exc
    finally:
        matlab_session.cd(original_matlab_directory, nargout=0)

    print(f"Lorenz commit: {lorenz_commit}")
    print(f"gPID commit: {gpid_commit}")
    print(
        "Figure 5B parameters: "
        f"scenario={CASE_TYPE}, N={NTRIALS}, repetitions={N_REPETITIONS}, "
        f"T={BIAS_ITERATIONS}, alpha index={INFO_INDEX + 1}, "
        f"alpha={ALPHAS[INFO_INDEX]}, dimensions={DIMENSIONS}"
    )
    print(f"MATLAB result: {RESULT_PATH}")
    print(f"Figure: {FIGURE_PATH}")
