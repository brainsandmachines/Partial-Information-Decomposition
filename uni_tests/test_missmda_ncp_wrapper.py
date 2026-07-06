"""Smoke test for the minimal missMDA wrapper."""

from pathlib import Path

import numpy as np

from library_wrappers.missmda_ncp import estimate_ncp_pca


RSCRIPT = Path("/home/ohadshee/anaconda3/envs/PID_env/bin/Rscript")


def test_estimate_ncp_pca() -> None:
    """Run one missing-data GCV call and verify its dictionary output.

    Args:
        None.

    Returns:
        ``None``. The test checks the selected component and MSEP keys.
    """
    data = np.random.default_rng(56).normal(size=(30, 5))
    data[3, 1] = np.nan
    result = estimate_ncp_pca(data, ncp_max=2, rscript=RSCRIPT)
    assert result["ncp"] in {0, 1, 2}
    assert set(result["criterion"]) == {0, 1, 2}
