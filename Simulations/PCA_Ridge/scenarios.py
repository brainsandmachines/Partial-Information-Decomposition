"""Scenario definitions shared by the consolidated PCA–Ridge runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from Simulations.PCA_Ridge.pid_feature_middleware import expand_independent_covariance
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.All_above_zero import (
    all_above_zero_weighted,
    con_all_above_zero_weighted,
)
from Simulations.Theoretical_Examples.RVs_Story.regular_examples.equal_unique import (
    equal_unique,
)
from Simulations.Theoretical_Examples.RVs_Story.suppresion_examples.full_suppresion import (
    full_suppresion,
)
from Simulations.Theoretical_Examples.RVs_Story.suppresion_examples.unq2_zero import (
    unq2_zero,
)


SampleGenerator = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
CovarianceBuilder = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class PcaRidgeScenario:
    """Describe one PCA–Ridge simulation without duplicating runner code.

    Inputs:
        generator: Callable producing aligned ``(X1, X2, T)`` NumPy arrays.
        covariance_builder: Callable producing the population covariance tensor.
        dimension_multiplier: int mapping the block dimension to each RV width.
        n_components: int default number of retained PCA components.
        pid_method: str PID method passed to the shared middleware.
        experiment_name: str label printed in experiment output.
        plot_slug: str filename-safe identifier for the saved comparison plot.
        plot_title: str title displayed above the comparison plot.
        generator_kwargs: Mapping of scenario-specific generator parameters.
        covariance_kwargs: Mapping of scenario-specific covariance parameters.

    Outputs:
        Immutable scenario metadata consumed by ``run_pca_ridge.run_scenario``.
    """

    generator: SampleGenerator
    covariance_builder: CovarianceBuilder
    dimension_multiplier: int
    n_components: int
    pid_method: str
    experiment_name: str
    plot_slug: str
    plot_title: str
    generator_kwargs: Mapping[str, float]
    covariance_kwargs: Mapping[str, float]


def concatenated_equal_unique(
    rng: np.random.Generator,
    n: int,
    p: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate equal unique information in separate concatenated target blocks.

    Inputs:
        rng: np.random.Generator controlling every generated latent and noise.
        n: int number of aligned samples.
        p: int dimension of each U1 and U2 block.
        noise_std: float standard deviation of each independent noise term.

    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray] containing X1, X2, and T;
        every array has shape ``(n, 2*p)`` and block order ``[U1, U2]``.
    """

    unique_1 = rng.standard_normal((n, p))  # scalar dimensions -> (n, p)
    unique_2 = rng.standard_normal((n, p))  # scalar dimensions -> (n, p)
    zeros = np.zeros_like(unique_1)  # (n, p) -> (n, p)
    target_signal = np.hstack((unique_1, unique_2))  # two (n, p) -> (n, 2*p)
    source_1_signal = np.hstack((unique_1, zeros))  # two (n, p) -> (n, 2*p)
    source_2_signal = np.hstack((zeros, unique_2))  # two (n, p) -> (n, 2*p)
    target = target_signal + noise_std * rng.standard_normal(target_signal.shape)  # (n, 2*p) -> (n, 2*p)
    source_1 = source_1_signal + noise_std * rng.standard_normal(source_1_signal.shape)  # (n, 2*p) -> (n, 2*p)
    source_2 = source_2_signal + noise_std * rng.standard_normal(source_2_signal.shape)  # (n, 2*p) -> (n, 2*p)
    return source_1, source_2, target


def build_all_above_zero_covariance(
    p: int,
    noise_std: float,
    unique1_weight: float,
    unique2_weight: float,
    redundant_weight: float,
    shared_noise_weight: float,
) -> torch.Tensor:
    """Build the all-above-zero covariance in grouped ``[X1, X2, T]`` order.

    Inputs:
        p: int number of independent coordinates per random variable.
        noise_std: float standard deviation of every independent noise term.
        unique1_weight: float weight of the X1-specific latent signal.
        unique2_weight: float weight of the X2-specific latent signal.
        redundant_weight: float weight of the latent signal shared with T.
        shared_noise_weight: float weight of source-only shared noise.

    Outputs:
        torch.Tensor float64 covariance with shape ``(3*p, 3*p)``.
    """

    source_noise = noise_std**2 * (1 + shared_noise_weight**2)
    coordinate_covariance = torch.tensor(
        [
            [
                unique1_weight**2 + redundant_weight**2 + source_noise,
                redundant_weight**2 + (shared_noise_weight * noise_std) ** 2,
                unique1_weight**2 + redundant_weight**2,
            ],
            [
                redundant_weight**2 + (shared_noise_weight * noise_std) ** 2,
                unique2_weight**2 + redundant_weight**2 + source_noise,
                unique2_weight**2 + redundant_weight**2,
            ],
            [
                unique1_weight**2 + redundant_weight**2,
                unique2_weight**2 + redundant_weight**2,
                unique1_weight**2
                + unique2_weight**2
                + redundant_weight**2
                + noise_std**2,
            ],
        ],
        dtype=torch.float64,
    )  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


def build_concatenated_all_above_zero_covariance(
    p: int,
    noise_std: float,
    redundant_weight: float,
    shared_noise_weight: float,
) -> torch.Tensor:
    """Build the concatenated all-above-zero covariance.

    Inputs:
        p: int coordinate count in each U1, U2, and redundant block.
        noise_std: float standard deviation of independent noise.
        redundant_weight: float weight of the redundant source block.
        shared_noise_weight: float weight of source-only shared noise.

    Outputs:
        torch.Tensor float64 covariance with shape ``(9*p, 9*p)``.
    """

    source_shared = redundant_weight**2 * (1 + (shared_noise_weight * noise_std) ** 2)
    source_redundant = source_shared + noise_std**2
    coordinate_covariance = torch.tensor(
        [
            [1 + noise_std**2, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, noise_std**2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, source_redundant, 0, 0, source_shared, 0, 0, redundant_weight],
            [0, 0, 0, noise_std**2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1 + noise_std**2, 0, 0, 1, 0],
            [0, 0, source_shared, 0, 0, source_redundant, 0, 0, redundant_weight],
            [1, 0, 0, 0, 0, 0, 1 + noise_std**2, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1 + noise_std**2, 0],
            [0, 0, redundant_weight, 0, 0, redundant_weight, 0, 0, 1 + noise_std**2],
        ],
        dtype=torch.float64,
    )  # construction scalars -> (9, 9), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (9, 9) -> (9*p, 9*p)


def build_concatenated_equal_unique_covariance(
    p: int,
    noise_std: float,
) -> torch.Tensor:
    """Build the concatenated equal-unique covariance.

    Inputs:
        p: int coordinate count in each U1 and U2 block.
        noise_std: float standard deviation of independent noise.

    Outputs:
        torch.Tensor float64 covariance with shape ``(6*p, 6*p)``.
    """

    total_variance = 1 + noise_std**2
    noise_variance = noise_std**2
    coordinate_covariance = torch.tensor(
        [
            [total_variance, 0, 0, 0, 1, 0],
            [0, noise_variance, 0, 0, 0, 0],
            [0, 0, noise_variance, 0, 0, 0],
            [0, 0, 0, total_variance, 0, 1],
            [1, 0, 0, 0, total_variance, 0],
            [0, 0, 0, 1, 0, total_variance],
        ],
        dtype=torch.float64,
    )  # construction scalars -> (6, 6), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (6, 6) -> (6*p, 6*p)


def build_equal_unique_covariance(p: int, noise_std: float) -> torch.Tensor:
    """Build the equal-unique covariance.

    Inputs:
        p: int coordinates per random variable.
        noise_std: float standard deviation of independent noise.

    Outputs:
        torch.Tensor float64 covariance with shape ``(3*p, 3*p)``.
    """

    coordinate_covariance = torch.tensor(
        [
            [1 + noise_std**2, 0.0, 1.0],
            [0.0, 1 + noise_std**2, 1.0],
            [1.0, 1.0, 2 + noise_std**2],
        ],
        dtype=torch.float64,
    )  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


def build_full_suppresion_covariance(p: int, noise_std: float) -> torch.Tensor:
    """Build the full-suppression covariance.

    Inputs:
        p: int coordinates per random variable.
        noise_std: float standard deviation of independent noise.

    Outputs:
        torch.Tensor float64 covariance with shape ``(3*p, 3*p)``.
    """

    noise_variance = noise_std**2
    coordinate_covariance = torch.tensor(
        [
            [1 + 3 * noise_variance, noise_variance, 1 + noise_variance],
            [noise_variance, noise_variance, 0],
            [1 + noise_variance, 0, 1 + noise_variance],
        ],
        dtype=torch.float64,
    )  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


def build_unq2_zero_covariance(p: int, noise_std: float) -> torch.Tensor:
    """Build the zero-source-2-unique covariance.

    Inputs:
        p: int coordinates per random variable.
        noise_std: float standard deviation of shared and target noise.

    Outputs:
        torch.Tensor float64 covariance with shape ``(3*p, 3*p)``.
    """

    noise_variance = noise_std**2
    coordinate_covariance = torch.tensor(
        [
            [2 + noise_variance, 1 + noise_variance, 2],
            [1 + noise_variance, 1 + noise_variance, 1],
            [2, 1, 2 + noise_variance],
        ],
        dtype=torch.float64,
    )  # construction scalars -> (3, 3), ordered [X1, X2, T]
    return expand_independent_covariance(coordinate_covariance, p)  # (3, 3) -> (3*p, 3*p)


SCENARIOS: dict[str, PcaRidgeScenario] = {
    "all_above_zero": PcaRidgeScenario(
        generator=all_above_zero_weighted,
        covariance_builder=build_all_above_zero_covariance,
        dimension_multiplier=1,
        n_components=5,
        pid_method="tilde",
        experiment_name="ALL ABOVE ZERO",
        plot_slug="all_above_zero",
        plot_title="All Above Zero PID: RAW vs PCA vs Ridge CV",
        generator_kwargs={
            "unique1_weight": 6.0,
            "unique2_weight": 7.0,
            "redundant_weight": 1.0,
            "shared_noise_weight": 1.0,
        },
        covariance_kwargs={
            "unique1_weight": 6.0,
            "unique2_weight": 7.0,
            "redundant_weight": 1.0,
            "shared_noise_weight": 1.0,
        },
    ),
    "con_all_above_zero": PcaRidgeScenario(
        generator=con_all_above_zero_weighted,
        covariance_builder=build_concatenated_all_above_zero_covariance,
        dimension_multiplier=3,
        n_components=30,
        pid_method="tilde",
        experiment_name="CONCATENATED ALL ABOVE ZERO",
        plot_slug="con_all_above_zero",
        plot_title="Concatenated All Above Zero PID: RAW vs PCA vs Ridge CV",
        generator_kwargs={"redundant_weight": 1.0, "shared_noise_weight": 1.0},
        covariance_kwargs={"redundant_weight": 1.0, "shared_noise_weight": 1.0},
    ),
    "con_equal_unique": PcaRidgeScenario(
        generator=concatenated_equal_unique,
        covariance_builder=build_concatenated_equal_unique_covariance,
        dimension_multiplier=2,
        n_components=30,
        pid_method="thin",
        experiment_name="CONCATENATED EQUAL UNIQUE",
        plot_slug="con_equal_unique",
        plot_title="Concatenated Equal Unique PID: RAW vs PCA vs Ridge CV",
        generator_kwargs={},
        covariance_kwargs={},
    ),
    "equal_unique": PcaRidgeScenario(
        generator=equal_unique,
        covariance_builder=build_equal_unique_covariance,
        dimension_multiplier=1,
        n_components=30,
        pid_method="thin",
        experiment_name="EQUAL UNIQUE",
        plot_slug="equal_unique",
        plot_title="Equal Unique PID: RAW vs PCA vs Ridge CV",
        generator_kwargs={},
        covariance_kwargs={},
    ),
    "full_suppresion": PcaRidgeScenario(
        generator=full_suppresion,
        covariance_builder=build_full_suppresion_covariance,
        dimension_multiplier=1,
        n_components=30,
        pid_method="tilde",
        experiment_name="FULL SUPPRESSION",
        plot_slug="full_suppresion",
        plot_title="Full Suppression PID: RAW vs PCA vs Ridge CV",
        generator_kwargs={},
        covariance_kwargs={},
    ),
    "unq2_zero": PcaRidgeScenario(
        generator=unq2_zero,
        covariance_builder=build_unq2_zero_covariance,
        dimension_multiplier=1,
        n_components=30,
        pid_method="tilde",
        experiment_name="SOURCE-2 UNIQUE ZERO",
        plot_slug="unq2_zero",
        plot_title="Source-2 Unique Zero PID: RAW vs PCA vs Ridge CV",
        generator_kwargs={},
        covariance_kwargs={},
    ),
}


def get_scenario(name: str) -> PcaRidgeScenario:
    """Return one registered PCA–Ridge scenario.

    Inputs:
        name: str scenario key from ``SCENARIOS``.

    Outputs:
        PcaRidgeScenario associated with the requested key.
    """

    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown PCA–Ridge scenario {name!r}; choose from {sorted(SCENARIOS)}."
        ) from error
