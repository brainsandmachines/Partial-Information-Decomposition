"""Names, entry points, and data requirements of every rank-selection method.

Each method module exposes ``run(data: RankData, **options) -> RankResult``.
Modules are imported one by one so that a missing dependency only disables
the methods that need it (``cross_decomposition`` needs the authors' ``lib``
and ``bonner``, available in the ``cross_decomposition`` conda environment).
"""

from dataclasses import dataclass
import inspect
from typing import Any, Callable

from pipeline.rank_selection.data import RankData


@dataclass(frozen=True)
class MethodSpec:
    """What a method needs from the data and from its environment.

    Fields:
        run: callable ``run(data, **options) -> RankResult`` or None when the
            module could not be imported.
        import_error: Exception or None.
        min_repeats: int, repeats per image required.
        needs_pair: bool, requires a second subject (``target_responses``).
        needs_n_at_least_p: bool, requires n_images >= n_voxels.
        environment: str, conda environment that provides the dependencies.
    """

    run: Callable[..., Any] | None
    import_error: Exception | None
    min_repeats: int = 1
    needs_pair: bool = False
    needs_n_at_least_p: bool = False
    environment: str = "pid_env"


_IMPORT_ERRORS: dict[str, Exception] = {}

try:
    from pipeline.rank_selection.methods import variance
except Exception as error:
    variance = None
    _IMPORT_ERRORS["variance_threshold"] = error
try:
    from pipeline.rank_selection.methods import eigenvector
except Exception as error:
    eigenvector = None
    _IMPORT_ERRORS["eigenvector"] = error
try:
    from pipeline.rank_selection.methods import repeat_reconstruction
except Exception as error:
    repeat_reconstruction = None
    _IMPORT_ERRORS["repeat_reconstruction"] = error
try:
    from pipeline.rank_selection.methods import mle
except Exception as error:
    mle = None
    _IMPORT_ERRORS["mle"] = error
try:
    from pipeline.rank_selection.methods import cross_decomposition
except Exception as error:
    cross_decomposition = None
    _IMPORT_ERRORS["cross_decomposition"] = error
try:
    from pipeline.rank_selection.methods import pls
except Exception as error:
    pls = None
    _IMPORT_ERRORS["pls"] = error
try:
    from pipeline.rank_selection.methods import missmda
except Exception as error:
    missmda = None
    _IMPORT_ERRORS["missmda"] = error


def _spec(name: str, module: Any, **requirements: Any) -> MethodSpec:
    """Build a MethodSpec from an imported (or missing) module.

    Inputs: name, str method name; module, imported module or None;
        requirements, MethodSpec keyword fields.
    Output: MethodSpec.
    """
    return MethodSpec(
        run=None if module is None else module.run,
        import_error=_IMPORT_ERRORS.get(name),
        **requirements,
    )


METHODS: dict[str, MethodSpec] = {
    "variance_threshold": _spec("variance_threshold", variance),
    "eigenvector": _spec("eigenvector", eigenvector),
    "repeat_reconstruction": _spec("repeat_reconstruction", repeat_reconstruction, min_repeats=2),
    "mle": _spec("mle", mle, needs_n_at_least_p=True),
    "cross_decomposition": _spec(
        "cross_decomposition", cross_decomposition, min_repeats=2,
        environment="cross_decomposition",
    ),
    "pls": _spec("pls", pls, needs_pair=True),
    "missmda": _spec("missmda", missmda),
}


def get_run_function(name: str) -> Callable[..., Any]:
    """Return a method's ``run`` function or explain why it is unavailable.

    Input: name, str key of METHODS.
    Output: callable ``run(data, **options) -> RankResult``. Raises KeyError for
        unknown names and ImportError naming the needed conda environment.
    """
    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}; choose from {sorted(METHODS)}.")
    spec = METHODS[name]
    if spec.run is None:
        raise ImportError(
            f"Method {name!r} could not be imported ({spec.import_error!r}). "
            f"Run it in the {spec.environment!r} conda environment."
        ) from spec.import_error
    return spec.run


def option_names(name: str) -> list[str]:
    """List the options a method accepts, without ``data`` and ``seed``.

    Input: name, str key of METHODS.
    Output: list[str] of keyword names of the method's ``run``.
    """
    parameters = inspect.signature(get_run_function(name)).parameters
    return [key for key in parameters if key not in ("data", "seed")]


def uses_seed(name: str, options: dict[str, Any]) -> bool:
    """Whether a method's result depends on the seed.

    Inputs: name, str; options, dict of the method's options.
    Output: bool. False for ``mle``, leave-one-image-out eigenvector CV, and
        missMDA GCV.
    """
    if name == "mle":
        return False
    if name == "missmda" and options.get("method_cv", "gcv") == "gcv":
        return False
    if name == "eigenvector" and options.get("folds") == "loo":
        return False
    return True


def unmet_requirement(name: str, options: dict[str, Any], data: RankData) -> str | None:
    """Explain why a method cannot run on a dataset, if it cannot.

    Inputs: name, str; options, dict; data, RankData.
    Output: str reason, or None when the method can run.
    """
    spec = METHODS[name]
    n_images, n_voxels, n_repeats = data.responses.shape
    min_repeats = spec.min_repeats
    if name == "eigenvector" and options.get("basis") == "gsn":
        min_repeats = 2
    if n_repeats < min_repeats:
        return f"needs at least {min_repeats} repeats, data have {n_repeats}"
    if spec.needs_pair and data.target_responses is None:
        return "needs a second subject"
    if spec.needs_n_at_least_p and n_images < n_voxels:
        return f"needs n_images >= n_voxels, data have {n_images} < {n_voxels}"
    return None
