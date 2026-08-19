import numpy as np
from pathlib import Path
from typing import Any
import pandas as pd



def calculate_mean_mse(csv_path: Path,max_pcs: int, column_name: str = "press_", n_features: int = 8088) -> dict[int, float]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    max_press = max_pcs

    pcs_mean_mse = {}
    for pc in range(1, max_press + 1):
        pc_df = df[column_name + str(pc)]
        pcs_mean_mse[pc] = np.sum(pc_df)/(len(pc_df)*n_features)
    return pcs_mean_mse

def report_top_minimal_mean(mean_mse_dict: dict[int, float], top_n: int = 5) -> None:
    sorted_mse = sorted(mean_mse_dict.items(), key=lambda x: x[1])
    print(f"Top {top_n} minimal mean MSE values:")
    for i in range(min(top_n, len(sorted_mse))):
        pc, mse = sorted_mse[i]
        print(f"PC {pc}: Mean MSE = {mse}")

if __name__ == "__main__":
    csv_path = Path("/home/ohadshee/Desktop/Thesis_Ohad_Sheelo/pipeline/subj_PCs/saved_pcs_nostandardization/eigenvector_max=200/checkpoint.csv")
    max_pcs = 200  # Set the maximum number of principal components
    mean_mse_dict = calculate_mean_mse(csv_path, max_pcs)
    report_top_minimal_mean(mean_mse_dict, top_n=10)