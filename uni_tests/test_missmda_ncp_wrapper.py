"""Smoke test for the minimal missMDA wrapper."""

import numpy as np

from library_wrappers.missmda_ncp import estimate_ncp_pca


def test_estimate_ncp_pca() -> None:
    """Run one missing-data GCV call and verify its dictionary output.

    Args:
        None.

    Returns:
        ``None``. The test checks the selected component and MSEP keys.
    """
    data = np.random.default_rng(56).normal(size=(30, 5))
    data[3, 1] = np.nan
    result = estimate_ncp_pca(data, ncp_max=2)
    assert result["ncp"] in {0, 1, 2}
    assert set(result["criterion"]) == {0, 1, 2}
