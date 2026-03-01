import os
import numpy as np
from datetime import datetime
from typing import List, Union

def create_timestamped_subfolder(parent_directory: str) -> str:
    """
    This function creates a subfolder with a timestamp in the parent directory.

    :param parent_directory: The path to the parent directory where the subfolder will be created.
    :returns: The full path of the created subfolder.
    """
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Create the full path for the new subfolder
    subfolder_path = os.path.join(parent_directory, timestamp)
    # Create the new subfolder
    os.makedirs(subfolder_path, exist_ok=True)
    return subfolder_path


def log_metadata(log_file_path: str, metadata: Union[str, list, dict]) -> None:
    """
    This function appends metadata to a log file.

    :param log_file_path: The path to the log file where metadata will be appended.
    :param metadata: The metadata to be logged. This can be a string, a list, or a dictionary.
    :returns: None
    """
    with open(log_file_path, 'a') as log_file:
        if isinstance(metadata, str):
            log_file.write(metadata + '\n')
        elif isinstance(metadata, list):
            log_file.write('\n'.join(str(item) for item in metadata) + '\n')
        elif isinstance(metadata, dict):
            for key, value in metadata.items():
                log_file.write(f"{key}: {value}\n")
        else:
            raise TypeError("Metadata must be a string, list, or dictionary.")


def log_initial_data(log_file_path: str, start_year: int, end_year: int, change_years: List[int], neigh_weights: np.ndarray,
                     conv_res: np.ndarray, allow: np.ndarray, lus_conv: np.ndarray,
                     demand: np.ndarray, dem_weights: np.ndarray, max_iter: int,
                     max_diff_allow: float, totdiff_allow: float, metadata:List[str]) -> None:
    """
    Logs initial input data to a log file.


    :param log_file_path: The path to the log file where data will be logged.
    :param start_year: The starting year (int).
    :param end_year: The ending year (int).
    :param change_years: List of change years for dynamic suitability
    :param neigh_weights: List of neighbourhood weights (float).
    :param conv_res: Conversion resistance vector (float).
    :param allow: Allowance matrix (2d np.array).
    :param lus_conv: Conversion factors (2d np.array)
    :param demand: Demand matrix (2d np.array).
    :param dem_weights: List of demand weights (float).
    :param max_iter: Maximum number of iterations (int).
    :param max_diff_allow: Maximum single difference allowed (float).
    :param totdiff_allow: Total difference allowed (float).
    :param metadata: A list of path names (str) for data used in the main function
    :return: None
    """
    with open(log_file_path, 'a') as log_file:
        log_file.write("Initial Input Data:\n")
        log_file.write(f"Start Year: {start_year}\n")
        log_file.write(f"End Year: {end_year}\n")
        log_file.write(f"Change Years: {change_years}\n")
        log_file.write(f"Maximum iterations: {max_iter}\n")
        log_file.write(f"Total difference allowed: {totdiff_allow}\n")
        log_file.write(f"Maximum single difference allowed: {max_diff_allow}\n")
        log_file.write(f"Number of Classes: {len(neigh_weights)}\n")
        log_file.write(f"Neighbourhood weights: {neigh_weights}\n")
        log_file.write(f"Number of Demands: {len(dem_weights)}\n")
        log_file.write(f"Demand weights: {dem_weights}\n")
        log_file.write(f"Conversion Resistance: \n{conv_res}\n")
        log_file.write(f"Allowance Matrix: \n{allow}\n")
        log_file.write(f"Conversion Factor Matrix: \n{lus_conv}\n")
        log_file.write(f"Demand Matrix: \n{demand}\n")
        # Logging metadata paths, each on a new line
        log_file.write("Raster Paths:\n")
        log_file.write("\n".join(metadata) + "\n")
        log_file.write("##########################################\n")
