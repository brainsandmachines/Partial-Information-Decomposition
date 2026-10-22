from pathlib import Path

import pandas as pd

from pipeline.pca.plotting import plot_cumulative_unique_information


def test_plot_output_suffix_is_replaced_with_pdf(tmp_path: Path) -> None:
    """Verify a requested PNG path produces and returns a PDF instead.

    Inputs:
        tmp_path: Path fixture naming an isolated temporary directory.

    Outputs:
        None. Assertions verify the returned suffix and PDF file signature.
    """
    csv_path = tmp_path / "pairwise.csv"
    pd.DataFrame(
        {
            "model_1": ["model_a"],
            "model_2": ["model_b"],
            "unq1": [0.2],
            "unq2": [0.1],
        }
    ).to_csv(csv_path, index=False)

    requested_path = tmp_path / "cumulative.png"
    saved_path = plot_cumulative_unique_information(csv_path, requested_path)

    assert saved_path == requested_path.with_suffix(".pdf")
    assert saved_path.read_bytes().startswith(b"%PDF")
    assert not requested_path.exists()
