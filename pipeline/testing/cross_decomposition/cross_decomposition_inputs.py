"""Load and adapt synthetic or OTC data for the authors' implementation."""

import inspect
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Bonner reads these settings during import. Use harmless cache locations and
# never reuse a fit belonging to another fold, seed, subject, or simulation.
for variable, folder in (
    ("BONNER_CACHING_HOME", "bonner-caching"),
    ("BONNER_DATASETS_HOME", "bonner-datasets"),
    ("BONNER_BRAINIO_HOME", "bonner-brainio"),
    ("BONNER_MODELS_HOME", "bonner-models"),
):
    os.environ.setdefault(variable, str(Path.home() / ".cache" / folder))
os.environ["BONNER_CACHING_MODE"] = "ignore"


def skip_font_installation() -> None:
    """Disable Bonner's unrelated import-time font installation.

    Inputs:
        None.

    Output:
        None. No fonts or font-configuration files are written.
    """


# Importing the authors' package otherwise downloads fonts and rewrites fonts.conf.
fonts_stub = types.ModuleType("bonner.plotting._fonts")
fonts_stub.install_newcomputermodern = skip_font_installation
sys.modules.setdefault("bonner.plotting._fonts", fonts_stub)

from bonner.caching import cache
from lib.datasets import split_by_repetition

from pipeline.pca.choosing_minimal_best_pca.choosing_minimal_best_pca import (
    load_repeat_responses,
)
from pipeline.testing.synthetic_repeated_responses import (
    generate_low_rank_repeated_responses,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("cross_decomposition_criteria.yaml")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the cross-decomposition experiment YAML.

    Inputs:
        config_path: str or Path to a YAML file; relative paths are resolved
            from the repository root.

    Output:
        config: dict[str, Any] containing the parsed experiment settings.
    """

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as handle:
        return yaml.safe_load(handle)


def assert_caching_disabled() -> None:
    """Require the authors' cache wrapper to recompute every result.

    Inputs:
        None.

    Output:
        None. Raises RuntimeError if Bonner was imported before caching was
            configured as ``ignore``.
    """

    mode = inspect.signature(cache).parameters["mode"].default
    if mode != "ignore":
        raise RuntimeError(
            f"bonner caching mode is {mode!r}; import this module before bonner."
        )


def build_repetition_dataarrays(
    responses: np.ndarray,
    image_ids: np.ndarray,
    dataset_name: str,
    n_repetitions: int = 2,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Adapt repeated responses to the authors' two-repetition interface.

    Inputs:
        responses: finite np.ndarray shaped (images, voxels, repeats).
        image_ids: unique np.ndarray shaped (images,) identifying stimuli.
        dataset_name: str stable name used by the authors' cache labels.
        n_repetitions: int number of leading repetitions exposed to the
            authors' ``split_by_repetition``; it must be exactly two.

    Output:
        x, y: the authors' repetition-0 and repetition-1 ``xr.DataArray``
            objects, each shaped (presentation, neuroid). No averaging,
            scaling, centering, fitting, or decomposition is performed here.
    """

    response_array = np.asarray(responses)
    stimulus_ids = np.asarray(image_ids)
    if response_array.ndim != 3 or not np.isfinite(response_array).all():
        raise ValueError("responses must be finite with shape (images, voxels, repeats).")
    if stimulus_ids.shape != (response_array.shape[0],):
        raise ValueError("image_ids must have one entry per response image.")
    if np.unique(stimulus_ids).size != stimulus_ids.size:
        raise ValueError("image_ids must be unique so images cannot cross folds.")
    if n_repetitions != 2:
        raise ValueError("The authors' within-subject calculation requires two repetitions.")
    if response_array.shape[2] < n_repetitions:
        raise ValueError(
            f"responses contain {response_array.shape[2]} repeats, "
            f"but {n_repetitions} are required."
        )

    n_images, n_voxels, _ = response_array.shape
    # (images, voxels, repeats) -> (images, repetitions, voxels)
    selected = response_array[:, :, :n_repetitions].transpose(0, 2, 1)
    # (images, repetitions, voxels) -> (presentations, neuroids)
    values = selected.reshape(n_images * n_repetitions, n_voxels)
    repeated = xr.DataArray(
        values,
        dims=("presentation", "neuroid"),
        coords={
            "presentation": np.arange(n_images * n_repetitions),
            "neuroid": np.arange(n_voxels),
            # (images,) -> (presentations,)
            "stimulus": ("presentation", np.repeat(stimulus_ids, n_repetitions)),
            # (repetitions,) -> (presentations,)
            "repetition": (
                "presentation",
                np.tile(np.arange(n_repetitions), n_images),
            ),
        },
        name=dataset_name,
    ).set_xindex(["stimulus", "repetition"])
    repetitions = split_by_repetition(repeated, n_repetitions=n_repetitions)
    return repetitions[0], repetitions[1]


def load_cross_decomposition_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Load one configured data source and adapt it for the authors' code.

    Inputs:
        config: dict[str, Any] containing the YAML's ``data`` section.

    Output:
        inputs: dict[str, Any] containing authors-compatible ``x`` and ``y``
            arrays shaped (presentation, neuroid), source metadata, image IDs
            shaped (images,), optional untouched synthetic generator output,
            and its known rank or None. No rank calculation is performed.
    """

    source = config.get("source", "synthetic")
    synthetic_data = None
    true_rank = None
    if source == "synthetic":
        synthetic_config = config.get("synthetic", config)
        synthetic_data = generate_low_rank_repeated_responses(**synthetic_config)
        responses = synthetic_data["responses"]
        true_rank = int(synthetic_config["true_rank"])
        # Scalar number of images -> (images,)
        image_ids = np.arange(responses.shape[0])
        dataset_name = (
            "synthetic"
            f".seed={synthetic_config['seed']}"
            f".rank={synthetic_config['true_rank']}"
        )
        source_metadata = {
            "source": "synthetic",
            "subject": None,
            "image_set": "generated",
            "generator_config": synthetic_config,
        }
    elif source == "otc":
        otc_config = config["otc"]
        neural_data_dir = Path(otc_config["neural_data_dir"])
        pkl_info_path = Path(otc_config["pkl_info_path"])
        if not neural_data_dir.is_absolute():
            neural_data_dir = REPO_ROOT / neural_data_dir
        if not pkl_info_path.is_absolute():
            pkl_info_path = REPO_ROOT / pkl_info_path
        loaded = load_repeat_responses(
            otc_config["subject"],
            image_set=otc_config["image_set"],
            neural_data_dir=neural_data_dir,
            pkl_info_path=pkl_info_path,
        )
        responses = loaded["responses"]
        image_ids = loaded["image_ids"]
        dataset_name = (
            f"nsd.otc.subject={otc_config['subject']}"
            f".image_set={otc_config['image_set']}"
        )
        source_metadata = {
            "source": "otc",
            "subject": otc_config["subject"],
            "image_set": otc_config["image_set"],
            "neural_data_dir": str(neural_data_dir),
            "pkl_info_path": str(pkl_info_path),
            "n_outside_set_removed": loaded["n_outside_set_removed"],
        }
    else:
        raise ValueError("data.source must be 'synthetic' or 'otc'.")

    # (images, voxels, repeats) -> two (presentations, neuroids) DataArrays
    x, y = build_repetition_dataarrays(
        responses,
        image_ids,
        dataset_name,
        n_repetitions=int(config.get("n_repetitions", 2)),
    )
    return {
        "x": x,
        "y": y,
        "image_ids": image_ids,
        "synthetic_data": synthetic_data,
        "true_rank": true_rank,
        "data_metadata": source_metadata | {
            "n_images": int(responses.shape[0]),
            "n_voxels": int(responses.shape[1]),
            "n_available_repeats": int(responses.shape[2]),
            "used_repetitions": [0, 1],
            "preprocessing": "as_stored_no_additional_scaling",
        },
    }
