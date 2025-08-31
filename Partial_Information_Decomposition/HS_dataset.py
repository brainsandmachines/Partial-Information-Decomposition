#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HS.data -> run user's CommonalityAnalysis on (paragrap ~ general + sentence + wordc + wordm)
and print the components with variable names, ready for comparison to Nimon et al. (2008).

Requires:
  - rpy2
  - R with package MBESS installed (this script auto-installs if missing)
  - Your CommonalityAnalysis class available on PYTHONPATH or in the same folder.

Usage:
  python test_commonality_on_HS.py
"""

import sys
import numpy as np
import pandas as pd

# ---- import rpy2 (modern conversion) ----
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

# ---- import your CommonalityAnalysis ----
# If your class is in another file, e.g., commonality.py, use:
# from commonality import CommonalityAnalysis
# For your snippet, it's presumably in the same runtime. If needed, adjust the import line above.
from Commonality_Analysis import CommonalityAnalysis  # adjust if your filename/class name differ


def ensure_r_pkg(pkg: str):
    """Import an R package; install from CRAN if not present."""
    try:
        return importr(pkg)
    except Exception:
        ro.r(f'install.packages("{pkg}", repos="https://cloud.r-project.org")')
        return importr(pkg)


def load_HS_data() -> pd.DataFrame:
    """Load HS.data from MBESS into a pandas DataFrame (no deprecated pandas2ri.activate())."""
    ensure_r_pkg("MBESS")
    ro.r('data("HS.data", package="MBESS")')
    r_df = ro.r["HS.data"]
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs = ro.conversion.rpy2py(r_df)
    # Some R datasets come as tibble-like objects; ensure pandas DataFrame
    hs = pd.DataFrame(hs)
    return hs


def run_commonality_on_HS():
    # 1) Load HS.data
    hs = load_HS_data()

    # 2) Select variables used in the paper’s example:
    #    paragrap ~ general + sentence + wordc + wordm
    y_name = "paragrap"
    x_names = ["general", "sentence", "wordc", "wordm"]

    # Keep only these columns and drop NA rows if present
    hs_sub = hs[[y_name] + x_names].dropna(axis=0).reset_index(drop=True)

    # 3) Build numpy arrays for your class
    y = hs_sub[y_name].to_numpy().reshape(-1, 1)              # (n,1)
    X = hs_sub[x_names].to_numpy()                            # (n,m)

    # 4) Run your CommonalityAnalysis
    ca = CommonalityAnalysis(predictions=X, target=y)
    ca.r_squared()                 # fills ca.r2_scores and ca.all_r2
    ca._find_intersetion_r2()      # fills ca.CA_scores
    namewise = translate_index_keys_to_names(ca.get_unique(), x_names)

    # 5) Print: full-model R^2 and components
    print("\n=== HS.data — Commonality Analysis (your implementation) ===")
    print(f"Full-model R^2 (your code): {ca.all_r2:.4f}")
    print("Expected R^2 from paper ≈ 0.6114")  # Nimon et al., 2008 (Behavior Research Methods)
    print("\nComponents (name-tuples -> value):")

    # Pretty print by subset size, then lexicographically
    rows = [(k, v) for k, v in namewise.items()]
    rows.sort(key=lambda kv: (len(kv[0]), kv[0]))
    for k, v in rows:
        tag = "Unique" if len(k) == 1 else "Common"
        print(f"{tag:7s} {k}: {v:.4f}")

    # Quick check: sum of all components should equal full R^2
    s = float(np.sum([v for _, v in rows]))
    print(f"\nSum of components: {s:.4f}")
    print(f"Matches full R^2? {'YES' if np.isclose(s, ca.all_r2, atol=1e-3) else 'NO'}")


def translate_index_keys_to_names(index_keyed_dict, x_names):
    """
    Your CA() returns keys as tuples of indices, e.g., (0,1) for general+sentence.
    Convert those to tuples of variable NAMES in a stable order.
    """
    out = {}
    for idx_tuple, val in index_keyed_dict.items():
        if isinstance(idx_tuple, int):
            idx_tuple = (idx_tuple,)
        names = tuple(x_names[i] for i in idx_tuple)
        # Ensure deterministic alphabetical ordering within tuple for display
        names = tuple(sorted(names))
        out[names] = float(val)
    return out


if __name__ == "__main__":
    run_commonality_on_HS()
