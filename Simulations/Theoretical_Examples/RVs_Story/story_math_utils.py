"""Small adapters around shared MI and bias helpers for RVs_Story truth rows."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Partial_Information_Decomposition.PID_util import create_cov_matrix
from Partial_Information_Decomposition.bias_functions import mi_wishart_bias
from Partial_Information_Decomposition.mi_functions import calculate_mi_raw


def calculate_story_mi_values(sources: list[torch.Tensor], target: list[torch.Tensor]) -> tuple[dict, dict]:
    """Calculate raw Gaussian MI values and legacy Wishart bias values.

    Inputs:
        sources: list[torch.Tensor], two source tensors ordered as [X1, X2].
        target: list[torch.Tensor], one target tensor ordered as [T].

    Outputs:
        tuple[dict, dict], raw MI values and bias values keyed like the old helpers.
    """


    x1, x2 = sources
    t = target[0]
    dims = [x1.shape[1], x2.shape[1], t.shape[1]]
    cov = create_cov_matrix(rvs=[x1, x2, t], dims=dims, device=x1.device)["full_cov"]
    return calculate_mi_raw(device=x1.device, sigma=cov, dims=dims), mi_wishart_bias(dims, x1.shape[0])
