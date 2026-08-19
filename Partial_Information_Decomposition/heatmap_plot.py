from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from Partial_Information_Decomposition.output_utils import safe_filename
except ImportError:
    from output_utils import safe_filename


PLOT_COMPONENTS = {
    # PID components
    "red": {
        "full_name": "Redundancy",
        "prefixes": ["red", "redundancy"],
    },
    "unq1": {
        "full_name": "Unique 1",
        "prefixes": ["unq1", "unique1", "unique_1", "unique_x1"],
    },
    "unq2": {
        "full_name": "Unique 2",
        "prefixes": ["unq2", "unique2", "unique_2", "unique_x2"],
    },
    "syn": {
        "full_name": "Synergy",
        "prefixes": ["syn", "synergy"],
    },

    # Mutual information components
    "mi_x1_t": {
        "full_name": "Mutual Information X1 And T",
        "prefixes": [
            "mi_x1_t",
            "mi_x1",
            "ix1t",
            "i_x1_t",
            "i_x1t",
            "i_x1_y",
            "mi_x1_y",
        ],
    },
    "mi_x2_t": {
        "full_name": "Mutual Information X2 And T",
        "prefixes": [
            "mi_x2_t",
            "mi_x2",
            "ix2t",
            "i_x2_t",
            "i_x2t",
            "i_x2_y",
            "mi_x2_y",
        ],
    },
    "mi_x1x2_t": {
        "full_name": "Mutual Information X1 X2 And T",
        "prefixes": [
            "mi_x1x2_t",
            "mi_x1_x2_t",
            "mi_joint",
            "mi_joint_t",
            "i_x1x2_t",
            "i_x1_x2_t",
            "i_joint_t",
            "i_joint",
        ],
    },
    "mi_m7": {
        "full_name": "Mutual Information M7",
        "prefixes": [
            "mi_m7",
            "m7_mi",
            "i_m7",
            "im7",
        ],
    },
    "mi_m8": {
        "full_name": "Mutual Information M8",
        "prefixes": [
            "mi_m8",
            "m8_mi",
            "i_m8",
            "im8",
        ],
    },
}


STAT_SUFFIXES = {
    "mean": ["mean", "avg", "average"],
    "std": ["std", "std_dev", "standard_deviation"],
    "ground_truth": ["ground_truth", "gt", "true", "truth"],
    "emp_bias": ["emp_bias", "empirical_bias", "bias"],
    "var": ["var", "variance"],
    "mse": ["mse", "mean_squared_error"],
}


def title_case_words(value):
    """
    Convert names like:
        'idep_gaussian' -> 'Idep Gaussian'
        'm7_pid'        -> 'M7 PID'
    """

    value = str(value).replace("_", " ").replace("-", " ")
    value = " ".join(value.split())
    value = value.title()

    replacements = {
        "Pid": "PID",
        "Idep": "Idep",
        "Mmi": "MMI",
        "Broja": "BROJA",
        "Gpid": "GPID",
        "Mse": "MSE",
        "Mi": "MI",
        "M7": "M7",
        "M8": "M8",
        "X1": "X1",
        "X2": "X2",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def make_full_title(base_title, pid_ver_name, component_name):
    """
    Build the full plot title.

    Example:
        base_title = "Bias Correction Simulation"

    Output:
        "Bias Correction Simulation — Idep — Redundancy"
    """

    if base_title is None or str(base_title).strip() == "":
        return f"{pid_ver_name} — {component_name}"

    return f"{base_title} — {pid_ver_name} — {component_name}"


def find_column(df, component_key, stat_name):
    """
    Find the correct column for a component and statistic.

    Examples accepted:
        red_mean
        mean_red
        redundancy_mean

        unq1_mean
        unique1_mean

        mi_x1_t_mean
        mean_mi_x1_t

        mi_m7_mean
        m7_mi_mean
    """

    component_info = PLOT_COMPONENTS[component_key]
    component_prefixes = component_info["prefixes"]
    stat_suffixes = STAT_SUFFIXES[stat_name]

    lower_to_original = {col.lower(): col for col in df.columns}

    candidates = []

    for prefix in component_prefixes:
        for suffix in stat_suffixes:
            candidates.extend(
                [
                    f"{prefix}_{suffix}",
                    f"{suffix}_{prefix}",
                ]
            )

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lower_to_original:
            return lower_to_original[candidate_lower]

    # Fallback:
    # If mean is requested and the CSV has a raw component column like "red"
    # or "mi_m7", use it as the mean.
    if stat_name == "mean":
        for prefix in component_prefixes:
            prefix_lower = prefix.lower()
            if prefix_lower in lower_to_original:
                return lower_to_original[prefix_lower]

    return None


def make_p_column(df, p_col="p"):
    """
    Create a p-like column from dx1, dx2, dt if p does not already exist.

    Your CSV usually has:
        dx1, dx2, dt

    This function creates:
        p = (dx1, dx2, dt)
    """

    df = df.copy()

    if p_col in df.columns:
        return df

    required = ["dx1", "dx2", "dt"]

    if all(col in df.columns for col in required):
        df[p_col] = list(zip(df["dx1"], df["dx2"], df["dt"]))
        return df

    raise ValueError(
        f"Could not create '{p_col}'. Expected either a '{p_col}' column "
        f"or the columns dx1, dx2, dt."
    )


def display_p_label(v):
    """
    Pretty display for p values on the y-axis.
    """

    if isinstance(v, tuple):
        return f"[{', '.join(map(str, v))}]"

    if isinstance(v, list):
        return f"[{', '.join(map(str, v))}]"

    return str(v)


def sort_p_index(values):
    """
    Sort p values numerically when they are tuples like:
        (dx1, dx2, dt)
    """

    def key(v):
        if isinstance(v, tuple):
            return tuple(float(x) for x in v)
        return v

    return sorted(values, key=key)


def optional_pivot_table(
    df,
    *,
    index,
    columns,
    values,
    aggfunc,
    reference_index,
    reference_columns,
):
    """
    Build a pivot table for an optional statistic and align it with the mean matrix.
    """

    if values is None:
        return None

    mat = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
    )

    mat = mat.reindex(index=reference_index, columns=reference_columns)

    return mat


def plot_single_component_heatmap(
    df,
    *,
    pid_ver,
    component_key,
    base_title=None,
    x_col="N",
    y_col="p",
    aggfunc="last",
    cmap="viridis",
    figsize=(9, 7),
    save_dir=None,
    show=True,
    mean_fmt=".3f",
    std_fmt=".3f",
):
    """
    Plot one heatmap for one PID version and one component.

    component_key can be:
        "red"
        "unq1"
        "unq2"
        "syn"
        "mi_x1_t"
        "mi_x2_t"
        "mi_x1x2_t"
        "mi_m7"
        "mi_m8"

    The title has the structure:

        Base Title — PID Version — Component

    Example:
        Bias Correction Simulation — Idep — Mutual Information M7
    """

    if component_key not in PLOT_COMPONENTS:
        raise ValueError(
            f"Unknown component_key={component_key}. "
            f"Allowed values are {list(PLOT_COMPONENTS.keys())}."
        )

    component_name = PLOT_COMPONENTS[component_key]["full_name"]
    pid_ver_name = title_case_words(pid_ver)

    mean_col = find_column(df, component_key, "mean")
    std_col = find_column(df, component_key, "std")
    gt_col = find_column(df, component_key, "ground_truth")
    eb_col = find_column(df, component_key, "emp_bias")
    var_col = find_column(df, component_key, "var")
    mse_col = find_column(df, component_key, "mse")

    if mean_col is None:
        print(f"Skipping {component_name}: no mean column found.")
        return None

    sub = df[df["pid_ver"] == pid_ver].copy()

    if sub.empty:
        print(f"Skipping {component_name}: no rows for pid_ver={pid_ver}.")
        return None

    mean_mat = sub.pivot_table(
        index=y_col,
        columns=x_col,
        values=mean_col,
        aggfunc=aggfunc,
    )

    if mean_mat.empty:
        print(f"Skipping {component_name}: empty pivot table.")
        return None

    # Sort rows and columns
    mean_mat = mean_mat.loc[sort_p_index(mean_mat.index)]
    mean_mat = mean_mat.reindex(sorted(mean_mat.columns), axis=1)

    std_mat = optional_pivot_table(
        sub,
        index=y_col,
        columns=x_col,
        values=std_col,
        aggfunc=aggfunc,
        reference_index=mean_mat.index,
        reference_columns=mean_mat.columns,
    )

    gt_mat = optional_pivot_table(
        sub,
        index=y_col,
        columns=x_col,
        values=gt_col,
        aggfunc=aggfunc,
        reference_index=mean_mat.index,
        reference_columns=mean_mat.columns,
    )

    eb_mat = optional_pivot_table(
        sub,
        index=y_col,
        columns=x_col,
        values=eb_col,
        aggfunc=aggfunc,
        reference_index=mean_mat.index,
        reference_columns=mean_mat.columns,
    )

    var_mat = optional_pivot_table(
        sub,
        index=y_col,
        columns=x_col,
        values=var_col,
        aggfunc=aggfunc,
        reference_index=mean_mat.index,
        reference_columns=mean_mat.columns,
    )

    mse_mat = optional_pivot_table(
        sub,
        index=y_col,
        columns=x_col,
        values=mse_col,
        aggfunc=aggfunc,
        reference_index=mean_mat.index,
        reference_columns=mean_mat.columns,
    )

    data = mean_mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        cmap=cmap,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(np.arange(len(mean_mat.columns)))
    ax.set_xticklabels(mean_mat.columns)

    ax.set_yticks(np.arange(len(mean_mat.index)))
    ax.set_yticklabels([display_p_label(v) for v in mean_mat.index])

    ax.set_xlabel("N")
    ax.set_ylabel("[Dx1, Dx2, Dt]")

    title = make_full_title(
        base_title=base_title,
        pid_ver_name=pid_ver_name,
        component_name=component_name,
    )

    ax.set_title(title)

    threshold = np.nanmean(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            m = mean_mat.iloc[i, j]

            if pd.isna(m):
                text = "NA"
                color = "black"
            else:
                lines = [f"Mean = {m:{mean_fmt}}"]

                if std_mat is not None and pd.notna(std_mat.iloc[i, j]):
                    lines.append(f"Std = {std_mat.iloc[i, j]:{std_fmt}}")

                if gt_mat is not None and pd.notna(gt_mat.iloc[i, j]):
                    lines.append(f"Ground Truth = {gt_mat.iloc[i, j]:{mean_fmt}}")

                if eb_mat is not None and pd.notna(eb_mat.iloc[i, j]):
                    lines.append(f"Empirical Bias = {eb_mat.iloc[i, j]:{mean_fmt}}")

                if var_mat is not None and pd.notna(var_mat.iloc[i, j]):
                    lines.append(f"Variance = {var_mat.iloc[i, j]:{mean_fmt}}")

                if mse_mat is not None and pd.notna(mse_mat.iloc[i, j]):
                    lines.append(f"MSE = {mse_mat.iloc[i, j]:{mean_fmt}}")

                text = "\n".join(lines)
                color = "white" if m < threshold else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=7,
                linespacing=1.15,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{component_name} Mean")

    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = safe_filename(title) + ".png"
        save_path = save_dir / file_name

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_pid_and_mi_heatmaps_from_csv(
    csv_path,
    *,
    base_title=None,
    save_dir=None,
    pid_versions=None,
    components=(
        "red",
        "unq1",
        "unq2",
        "syn",
        "mi_x1_t",
        "mi_x2_t",
        "mi_x1x2_t",
        "mi_m7",
        "mi_m8",
    ),
    x_col="N",
    y_col="p",
    seed=None,
    aggfunc="last",
    cmap="viridis",
    figsize=(9, 7),
    show=True,
):
    """
    Read a checkpoint CSV and create heatmaps for PID components
    and mutual information values.

    One figure is created for each pair:

        pid_ver × component

    The title has the structure:

        Base Title — PID Version — Component

    Example:
        Bias Correction Simulation — Idep — Redundancy
        Bias Correction Simulation — Idep — Mutual Information M7
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    df = make_p_column(df, p_col=y_col)

    if "pid_ver" not in df.columns:
        raise ValueError("CSV must contain a 'pid_ver' column.")

    if seed is not None:
        if "seed" not in df.columns:
            raise ValueError("You passed seed=..., but CSV has no 'seed' column.")

        df = df[df["seed"] == seed].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check csv_path, seed, or pid_versions.")

    if pid_versions is None:
        pid_versions = sorted(df["pid_ver"].dropna().unique())

    figures = {}

    for pid_ver in pid_versions:
        for component_key in components:
            result = plot_single_component_heatmap(
                df,
                pid_ver=pid_ver,
                component_key=component_key,
                base_title=base_title,
                x_col=x_col,
                y_col=y_col,
                aggfunc=aggfunc,
                cmap=cmap,
                figsize=figsize,
                save_dir=save_dir,
                show=show,
            )

            if result is not None:
                figures[(pid_ver, component_key)] = result

    return figures
