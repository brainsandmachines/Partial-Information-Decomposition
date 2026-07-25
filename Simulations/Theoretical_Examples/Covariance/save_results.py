import torch
import numpy as np
import yaml
from pathlib import Path
import sys









def save_sample_simulation_results_table(
    results: dict,
    config: dict,
    save_path: str | Path,
    decimals: int = 4,
    title: str = "PID Method Comparison",
    dpi: int = 200,
) -> Path:
    """Save sample-simulation PID/MI summaries as a styled table image.

    Inputs:
        results: dict, output from simulation. Expected shape is
            {method: {"theoretical": ..., "mean_sampled": ..., "bias": ...,
            "variance": ..., "mse": ...}}. The theoretical and mean_sampled
            entries are rendered as separate method-by-component tables.
            Optional `cmi_*_test` strings add the two CMI validation columns.
        config: dict, simulation configuration with n/n_samples, dimensions,
            p_scale, q_scale, r_scale, seed, bias_correction, and n_trials.
        save_path: str | Path, target image file path.
        decimals: int, number of decimal places to show in numeric cells.
        title: str, title shown above the image table.
        dpi: int, output image resolution.

    Outputs:
        Path, the saved image path.
    """
    import matplotlib.pyplot as plt

    cfg = config.get("parameters", config)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["method", "Red", "Unq1", "Unq2", "Syn", "I(X1;T)", "I(X2;T)", "I(X1,X2;T)", "Given X1\nCMI(T;X2|X1)", "Given X2\nCMI(T;X1|X2)"]
    component_keys = {
        "Red": ("red",),
        "Unq1": ("unq1", "unq0"),
        "Unq2": ("unq2", "unq1"),
        "Syn": ("syn",),
        "I(X1;T)": ("bi_mi_1", "I(M1;T)", "I(M0;T)"),
        "I(X2;T)": ("bi_mi_2", "I(M2;T)", "I(M1;T)"),
        "I(X1,X2;T)": ("tri_mi", "I(M1,M2;T)", "I(M0,M1;T)"),
        "Given X1\nCMI(T;X2|X1)": ("cmi_x2_given_x1_test",),
        "Given X2\nCMI(T;X1|X2)": ("cmi_x1_given_x2_test",),
    }
    method_names = {"flow": "Flow", "tilde": "Tilde", "delta": "Delta", "idep": "Idep"}
    metric_names = [
        ("Theoretical Covariance", "theoretical", "theoretical_values"),
        ("Mean Across Trials", "mean_sampled", "mean_sampled_values"),
        ("Bias", "bias", "bias"),
        ("Variance", "variance", "variance"),
        ("MSE", "mse", "mse"),
    ]

    method_results_shape = all(isinstance(value, dict) for value in results.values())
    methods = list(results.keys()) if method_results_shape else []
    if not methods:
        for _, _, top_level_key in metric_names:
            values_by_method = results.get(top_level_key, {})
            if isinstance(values_by_method, dict):
                for method in values_by_method:
                    if method not in methods:
                        methods.append(method)

    section_tables = []
    has_cmi_validation = False
    for section_name, method_key, top_level_key in metric_names:
        rows = []
        for method in methods:
            if method_results_shape:
                method_payload = results.get(method, {})
                if method_key not in method_payload:
                    continue
                value = method_payload[method_key]
            else:
                values_by_method = results.get(top_level_key, {})
                if not isinstance(values_by_method, dict) or method not in values_by_method:
                    continue
                value = values_by_method[method]
            rows.append((method_names.get(str(method).lower(), str(method)), value))

        table_rows = []
        for method_label, value in rows:
            value_dict = {}
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                if isinstance(value[0], dict):
                    value_dict.update(value[0])
                if isinstance(value[1], dict):
                    value_dict.update(value[1])
            elif isinstance(value, dict):
                if isinstance(value.get("pid"), dict):
                    value_dict.update(value["pid"])
                if isinstance(value.get("mi"), dict):
                    value_dict.update(value["mi"])
                value_dict.update({key: val for key, val in value.items() if key not in ("pid", "mi")})

            table_row = {"method": method_label}
            for column in columns[1:]:
                cell_value = next((value_dict[key] for key in component_keys[column] if key in value_dict), None)
                if isinstance(cell_value, torch.Tensor):
                    cell_value = cell_value.detach().cpu().item() if cell_value.numel() == 1 else None
                if isinstance(cell_value, np.ndarray):
                    cell_value = cell_value.item() if cell_value.size == 1 else None
                table_row[column] = cell_value
                if column.startswith("Given ") and cell_value is not None:
                    has_cmi_validation = True
            table_rows.append(table_row)

        if table_rows:
            section_tables.append((section_name, table_rows))

    if not section_tables:
        raise ValueError("No table data found in results.")
    if not has_cmi_validation:
        columns = columns[:-2]

    legend_items = {
        "n": cfg.get("n", cfg.get("n_samples")),
        "dx1": cfg.get("dx1", cfg.get("n0", cfg.get("p"))),
        "dx2": cfg.get("dx2", cfg.get("n1", cfg.get("p"))),
        "dt": cfg.get("dt", cfg.get("n2", cfg.get("p"))),
        "seed": cfg.get("seed"),
        "bias_correction": cfg.get("bias_correction"),
        "p_scale": cfg.get("p_scale"),
        "q_scale": cfg.get("q_scale"),
        "r_scale": cfg.get("r_scale"),
        "n_trials": cfg.get("n_trials"),
        "cmi_tolerance": cfg.get("cmi_tolerance"),
    }
    legend = " | ".join(f"{key}={value}" for key, value in legend_items.items() if value is not None)

    fig_height = max(2.8, 1.35 + sum(0.42 * (len(rows) + 1) + 0.45 for _, rows in section_tables))
    fig, axes = plt.subplots(len(section_tables), 1, figsize=(18 if has_cmi_validation else 14, fig_height))
    if len(section_tables) == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    if legend:
        fig.text(0.5, 0.955, legend, ha="center", va="center", fontsize=9, color="#4b5563")

    for ax, (section_name, rows) in zip(axes, section_tables):
        ax.axis("off")
        ax.set_title(section_name, loc="left", fontsize=11, fontweight="bold", pad=6)
        cell_text = []
        for row in rows:
            cell_text.append([
                f"{row[col]:.{decimals}f}" if isinstance(row[col], (int, float, np.number)) else ("" if row[col] is None else str(row[col]))
                for col in columns
            ])

        table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        for (table_row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            if table_row == 0:
                cell.set_facecolor("#111827")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f9fafb" if table_row % 2 else "white")

    plt.tight_layout(rect=[0, 0, 1, 0.93 if legend else 0.95])
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return save_path
