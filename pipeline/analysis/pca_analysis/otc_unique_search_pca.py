"""Run PCA unique-information subset search on full OTC data and two sources."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.analysis.pca_analysis.unique_search_pca import run_pid_pc_subset_search
from pipeline.full_OTC.otc_experiment import PIPELINE_STEP_FUNCTIONS
from pipeline.pipeline_utils import (
    pid_calc_adapter,
    run_configured_pid_pipeline,
    validate_pipeline_config_sections,
)


DEFAULT_SEARCH_CONFIG = Path(__file__).with_name("otc_unique_search_config.yaml")


def run_otc_unique_search_pca(
    config: dict[str, Any] | str | Path,
    max_source_components: int,
    *,
    search_source: str = "X1",
    search_kwargs: dict[str, Any] | None = None,
    all_csv_path: str | Path | None = None,
    best_csv_path: str | Path | None = None,
    pid_callable: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load OTC/source arrays, fix target/other-source PCs, and search one source.

    Inputs: config is an OTC pipeline dict or YAML path; max_source_components
        is the maximum PCA PCs for the searched source; search_source is "X1" or
        "X2"; search_kwargs go to run_pid_pc_subset_search; CSV paths are
        optional; pid_callable overrides the default pid_calc_adapter.
    Outputs: dict search result with added source, layer, and PCA metadata.
    """

    config = _load_config(config)
    if search_source not in {"X1", "X2"}:
        raise ValueError("search_source must be 'X1' or 'X2'.")

    context = _load_otc_arrays(config)
    target, searched, fixed, pca_info = _pca_search_inputs(context, config, search_source, max_source_components)
    pid_kwargs = dict(config.get("pid_kwargs", {}))
    kwargs = dict(search_kwargs or {})
    kwargs.setdefault("rng_seed", pid_kwargs.get("rng_seed", 56))

    result = run_pid_pc_subset_search(
        target=target,
        source_1=searched,
        source_2=fixed,
        pid_callable=pid_calc_adapter if pid_callable is None else pid_callable,
        pid_kwargs=pid_kwargs,
        all_csv_path=all_csv_path,
        best_csv_path=best_csv_path,
        **kwargs,
    )
    result.update(
        {
            "search_source": search_source,
            "fixed_source": "X2" if search_source == "X1" else "X1",
            "selected_layers": context.get("selected_layers"),
            "pca_components": pca_info,
        }
    )
    return result


def run_otc_unique_search_from_yaml(config_path: str | Path = DEFAULT_SEARCH_CONFIG) -> dict[str, Any]:
    """Run OTC unique search from one YAML analysis config file.

    Inputs: config_path is a str or Path to a YAML file with pipeline_config,
        max_source_components, search_source, optional CSV paths, and search_kwargs.
    Outputs: dict search result from run_otc_unique_search_pca.
    """

    config_path = Path(config_path)
    analysis_config = _load_config(config_path)
    base_dir = config_path.resolve().parent
    return run_otc_unique_search_pca(
        config=_resolve_path(analysis_config["pipeline_config"], base_dir),
        max_source_components=int(analysis_config["max_source_components"]),
        search_source=analysis_config.get("search_source", "X1"),
        all_csv_path=_resolve_path(analysis_config.get("all_csv_path"), base_dir),
        best_csv_path=_resolve_path(analysis_config.get("best_csv_path"), base_dir),
        search_kwargs=dict(analysis_config.get("search_kwargs", {})),
    )


def _load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load an OTC config dictionary or YAML file.

    Inputs: config is either a dict already in memory or a str/Path YAML path.
    Outputs: dict OTC pipeline config.
    """

    if isinstance(config, dict):
        return dict(config)
    import yaml

    with Path(config).open("r") as config_file:
        return yaml.safe_load(config_file)


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    """Resolve optional YAML paths relative to the YAML file directory.

    Inputs: value is a str, Path, or None; base_dir is the YAML parent Path.
    Outputs: absolute Path or None.
    """

    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _load_otc_arrays(config: dict[str, Any]) -> dict[str, Any]:
    """Run PIDPipeline only through target/source/layer/feature extraction.

    Inputs: config is an OTC pipeline config with functions and kwargs sections.
    Outputs: dict PIDPipeline context containing raw target, source_1, source_2.
    """

    validate_pipeline_config_sections(
        config,
        ("functions", "target_kwargs", "sources_kwargs", "choose_layer_kwargs", "feature_extraction_kwargs", "pid_kwargs"),
    )
    load_config = dict(config)
    load_config["functions"] = dict(config["functions"])
    load_config["functions"]["feature_manipulation"] = None
    load_config["functions"]["pid_calculation"] = "_skip_pid"
    load_config["functions"]["pid_report"] = None
    return run_configured_pid_pipeline(
        load_config,
        {**PIPELINE_STEP_FUNCTIONS, "_skip_pid": _skip_pid},
    )


def _skip_pid(target: Any, source_1: Any, source_2: Any, **pid_kwargs: Any) -> None:
    """Satisfy PIDPipeline while loading arrays without running PID.

    Inputs: target, source_1, source_2, and pid_kwargs match a PID callable.
    Outputs: None.
    """

    del target, source_1, source_2, pid_kwargs
    return None


def _pca_search_inputs(
    context: dict[str, Any],
    config: dict[str, Any],
    search_source: str,
    max_source_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Project target, searched source, and fixed source for subset search.

    Inputs: context is PIDPipeline output; config carries original PCA counts;
        search_source is "X1" or "X2"; max_source_components caps searched PCs.
    Outputs: tuple(target_pcs, searched_source_pcs, fixed_source_pcs, metadata).
    """

    pca_kwargs = dict(config.get("feature_manipulation_kwargs", {}))
    fixed_source = "X2" if search_source == "X1" else "X1"
    searched_key = "source_1" if search_source == "X1" else "source_2"
    fixed_key = "source_2" if search_source == "X1" else "source_1"
    fixed_components = pca_kwargs.get(f"n_components_source_{2 if fixed_source == 'X2' else 1}")

    target = _project_or_keep(context["target"], pca_kwargs.get("n_components_target"), "target")
    searched = _project_or_keep(context[searched_key], max_source_components, search_source)
    fixed = _project_or_keep(context[fixed_key], fixed_components, fixed_source)
    return target, searched, fixed, {
        "target": target.shape[1],
        search_source: searched.shape[1],
        fixed_source: fixed.shape[1],
        "requested_search_max": int(max_source_components),
        "requested_fixed": fixed_components,
        "requested_target": pca_kwargs.get("n_components_target"),
    }


def _project_or_keep(features: Any, n_components: int | None, name: str) -> np.ndarray:
    """Apply PCA when requested, capping components to the valid matrix size.

    Inputs: features is array-like samples; n_components is int or None; name is
        used in errors.
    Outputs: 2D np.ndarray, either original features or PCA projection.
    """

    array = np.asarray(features.detach().cpu().numpy() if hasattr(features, "detach") else features)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array after loading, got shape {array.shape}")
    if n_components is None:
        return array
    n_components = min(int(n_components), min(array.shape))
    if n_components < 1:
        raise ValueError(f"{name} needs at least one PCA component")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=56)
    return pca.fit_transform(array)


def main(config_path: str | Path | None = None) -> None:
    """Run OTC PCA unique search from a YAML config file.

    Inputs: config_path is a str, Path, or None; when None, sys.argv[1] or the
        default YAML path is used.
    Outputs: None, prints a compact result summary.
    """

    if config_path is None:
        config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEARCH_CONFIG
    result = run_otc_unique_search_from_yaml(config_path)
    keys = ("status", "search_source", "fixed_source", "best_subset", "best_pid_components", "pca_components")
    print({key: result.get(key) for key in keys})


if __name__ == "__main__":
    main()
