import csv

import numpy as np
import unicodedata

""""A file with functions help to choose the layer to each source"""


def random_layer_selection(n_layers):
    """Choose a random layer index from the available layers.
    
    Input:
        - n_layers: total number of layers available (integer)
        
    Output:
        - layer_idx: randomly selected layer index (integer in range [0, n_layers-1])
    """

    layer_idx = np.random.randint(0, n_layers)
    return layer_idx


def specific_index_layer_selection(layer_names, index):
    """Choose a specific layer index from the available layers.
    
    Input:
        - layer_names: list of available layer names
        - index: specific layer index to select (integer in range [0, len(layer_names)-1])
        
        
    Output:
        - layer_name: name of the selected layer (string)
        """
    
    
    

    return layer_names[index]





def voxel_best_layer(voxel_index: int = None, index_layer: int = None, path_to_results: str = None) -> dict:
    """Choose the best model layer for one voxel, or a representative voxel for one layer.

    Input:
        - voxel_index: voxel index to look up (integer or None).
        - index_layer: best layer index to look up (integer or None).
        - path_to_results: path to a CSV file with columns 'voxel_index' and
          'best_layer_index' (string or None when both indexes are provided).

    Output:
        - dict with keys 'v' and 'l', where 'v' is the selected voxel index
          and 'l' is the selected best layer index. Missing or failed lookups
          return {'v': None, 'l': None}.
    """

    if voxel_index is not None and index_layer is not None:
        return {'v': int(voxel_index), 'l': int(index_layer)}

    if path_to_results is None:
        print("No path_to_results provided for layer lookup.")
        return {'v': None, 'l': None}

    try:
        rows, columns = _read_csv_rows(path_to_results)
        required_columns = {'voxel_index', 'best_layer_index'}
        missing_columns = required_columns.difference(columns)
        if missing_columns:
            print(f"Missing required columns in results CSV: {sorted(missing_columns)}")
            return {'v': None, 'l': None}

        if voxel_index is not None:
            voxel_rows = [row for row in rows if int(row['voxel_index']) == int(voxel_index)]
            if not voxel_rows:
                print(f"No results found for voxel index {voxel_index}")
                return {'v': None, 'l': None}
            index_layer = voxel_rows[0]['best_layer_index']

        elif index_layer is not None:
            layer_rows = [row for row in rows if int(row['best_layer_index']) == int(index_layer)]
            if not layer_rows:
                print(f"No results found for layer index {index_layer}")
                return {'v': None, 'l': None}
            voxel_index = layer_rows[0]['voxel_index']

        else:
            print("No voxel index or layer index provided. Choosing random layer.")
            unique_layers = sorted({row['best_layer_index'] for row in rows if row.get('best_layer_index') not in (None, '')})
            if len(unique_layers) == 0:
                print("No layer indexes found in results CSV.")
                return {'v': None, 'l': None}
            index_layer = np.random.choice(unique_layers)
            layer_rows = [row for row in rows if row['best_layer_index'] == str(index_layer)]
            voxel_index = layer_rows[0]['voxel_index']

    except Exception as e:
        print(f"Error loading best layer selection results: {e}")
        return {'v': None, 'l': None}

    return {'v': int(voxel_index), 'l': int(index_layer)}



    
def overall_best_layer(model_name: str, path_to_results: str) -> dict:
    """Choose the overall best layer index for one model from an OTC CSV.

    Inputs:
        model_name: str, source model name to look up in the CSV.
        path_to_results: str, path to a CSV file with columns 'model_name' and
            'best_layer_index'.

    Output:
        best_layer: dict, contains 'model_name' and 'l' with the selected layer
            index. Missing or failed lookups return {'model_name': model_name,
            'l': None}.
    """

    try:
        rows, columns = _read_csv_rows(path_to_results)
        required_columns = {'model_name', 'best_layer_index'}
        missing_columns = required_columns.difference(columns)
        if missing_columns:
            print(f"Missing required columns in results CSV: {sorted(missing_columns)}")
            return {'model_name': model_name, 'l': None}

        normalized_model_name = _normalize_csv_value(model_name)
        model_rows = [
            row for row in rows
            if _normalize_csv_value(row['model_name']) == normalized_model_name
        ]
        if not model_rows:
            print(f"No overall best layer found for model {model_name}")
            return {'model_name': model_name, 'l': None}
        index_layer = model_rows[0]['best_layer_index']

    except Exception as e:
        print(f"Error loading overall best layer results: {e}")
        return {'model_name': model_name, 'l': None}

    return {'model_name': model_name, 'l': int(index_layer)}

    
def _read_csv_rows(path_to_results: str) -> tuple[list[dict[str, str]], set[str]]:
    """Read CSV rows and column names for layer-selection helpers.

    Inputs:
        path_to_results: str, path to a CSV file.

    Output:
        rows_and_columns: tuple, list of row dictionaries and set of column names.
    """

    with open(path_to_results, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    return rows, columns

    
def _normalize_csv_value(value) -> str:
    """Normalize CSV values before exact lookup comparisons.

    Inputs:
        value: any, CSV or config value to compare.

    Output:
        normalized_value: str, stripped string value with invisible format
            characters removed.
    """

    text = unicodedata.normalize("NFKC", str(value))
    return "".join(
        character
        for character in text.strip()
        if unicodedata.category(character) != "Cf"
    )

    

    
