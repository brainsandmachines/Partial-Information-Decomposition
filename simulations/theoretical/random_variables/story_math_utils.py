"""Small adapters around shared MI and bias helpers for RVs_Story truth rows."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pid.utils import create_cov_matrix
from pid.mutual_information import gaussian_mutual_information


def calculate_story_mi_values(sources: list[torch.Tensor], target: list[torch.Tensor]) -> tuple[dict, dict]:
    """Calculate raw Gaussian MI values and Wishart biases in bits.

    Inputs:
        sources: list[torch.Tensor], two source tensors ordered as [X1, X2].
        target: list[torch.Tensor], one target tensor ordered as [T].

    Outputs:
        tuple[dict, dict], raw MI values and bias values in bits, keyed for
        the existing story PID callers.
    """


    x1, x2 = sources
    t = target[0]
    dims = [x1.shape[1], x2.shape[1], t.shape[1]]
    covariance = create_cov_matrix(
        rvs=[x1, x2, t],
        dims=dims,
        device=x1.device,
    )  # [(N, dx1), (N, dx2), (N, dt)] -> covariance blocks for (D, D)
    imx = gaussian_mutual_information(
        covariance["joint_x1_t"],
        dims[0],
        n_samples=x1.shape[0],
        bias_correction="wishart",
    )
    imy = gaussian_mutual_information(
        covariance["joint_x2_t"],
        dims[1],
        n_samples=x1.shape[0],
        bias_correction="wishart",
    )
    imxy = gaussian_mutual_information(
        covariance["full_cov"],
        dims[0] + dims[1],
        n_samples=x1.shape[0],
        bias_correction="wishart",
    )
    raw = {
        "tri_mi": imxy["raw_bits"],
        "bi_mi_1_t": imx["raw_bits"],
        "bi_mi_2_t": imy["raw_bits"],
    }
    bias = {
        "bias_tri_mi": imxy["bias_bits"],
        "bias_mi_1_t": imx["bias_bits"],
        "bias_mi_2_t": imy["bias_bits"],
    }
    return raw, bias
