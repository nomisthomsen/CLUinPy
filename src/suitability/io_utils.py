import os
import numpy as np
import rasterio
from typing import List, Optional, Union

def find_files(folder_path: str, patterns: Union[str, List[str]], ending: Optional[str] = None) -> List[str]:
    """
    Returns a list of files in a specified directory that match given start or end patterns.

    Parameters:
    folder_path (str): The root directory path to search in.
    patterns (Union[str, List[str]]): A string or a list of strings representing filename start or end patterns.
    ending (Optional[str]): The file extension to filter by (e.g., '.txt'). Defaults to None.

    Returns:
    List[str]: A list of matching file paths.
    """
    found_files = []

    # Ensure patterns is a list for uniform processing
    if isinstance(patterns, str):
        patterns = [patterns]

    # Iterate over files in the specified directory (non-recursively)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):  # Check if it's a file (not a directory)
            # Skip files that do not match the specified extension, if provided
            if ending and not file.endswith(ending):
                continue

            # Check if filename matches any specified start or end patterns
            if any(file.startswith(pattern) or file.endswith(pattern) for pattern in patterns):
                found_files.append(file_path)

    return found_files




def write_logfile(out_path: str, **kwargs):
    """Writes parameters to a log file in the specified output directory."""
    log_filename = os.path.join(out_path, "suitability_log.txt")
    with open(log_filename, "w") as log_file:
        log_file.write("suitability Analysis Log\n")
        log_file.write("========================\n\n")
        for key, value in kwargs.items():
            if isinstance(value, list):
                log_file.write(f"{key}:\n")
                for item in value:
                    log_file.write(f"  - {item}\n")
            else:
                log_file.write(f"{key}: {value}\n")



def create_folder_and_file(base_folder_name: str, auc_file: str, stats_file: str) -> None:
    """
    Creates a folder and specified files if they do not already exist.

    Parameters:
    - base_folder_name (str): Name of the folder to be created.
    - auc_file (str): Name of the AUC file to be created inside the folder.
    - stats_file (str): Name of the stats file to be created inside the folder.
    """
    # Create folder if it doesn't exist
    if not os.path.exists(base_folder_name):
        os.makedirs(base_folder_name)

    # Create the AUC file if it doesn't exist
    auc_path = os.path.join(base_folder_name, auc_file)
    if not os.path.exists(auc_path):
        open(auc_path, 'a').close()

    # Create the stats file if it doesn't exist
    stats_path = os.path.join(base_folder_name, stats_file)
    if not os.path.exists(stats_path):
        open(stats_path, 'a').close()



def write_array_to_geotiff(array: np.ndarray, out_path: str, reference_raster: str,
                           no_data_out: int, crs: Optional[str] = None, dtype: str = "float") -> None:
    """
    Writes a 2D or 3D array to a GeoTIFF (.tif) file. Automatically handles single-band (2D) or multi-band (3D) arrays.

    Parameters:
    - array: The 2D or 3D array to be written to the GeoTIFF file.
    - out_path: The output path for the GeoTIFF file.
    - reference_raster: Path to an existing raster file to extract metadata (nrows, ncols, transform, CRS).
    - no_data_out: Value for no-data pixels in the output file.
    - crs: (Optional) A string indicating the EPSG in the format "EPSG:4326". If None, it uses the reference raster's CRS.
    - dtype: The data type of the output raster file (default is "float").
    """
    # Read metadata from reference raster
    with rasterio.open(reference_raster) as ref:
        nrows, ncols = ref.height, ref.width
        transform = ref.transform
        reference_crs = ref.crs

    # Use reference raster's CRS if none is provided
    if crs is None:
        crs = reference_crs

    # Ensure array has the correct shape (convert 2D to 3D)
    if len(array.shape) == 2:
        array = array[np.newaxis, :, :]  # Add band dimension
    elif len(array.shape) != 3:
        raise ValueError("Input array must be either 2D or 3D.")

    num_bands, height, width = array.shape

    if (height, width) != (nrows, ncols):
        raise ValueError(f"Array dimensions {height}x{width} do not match reference raster dimensions {nrows}x{ncols}.")

    # Handle data type conversion
    dtype_mapping = {
        "int16": (np.int16, 'int16'),
        "int8": (np.int8, 'int8'),
        "float": (np.float32, 'float32')
    }

    if dtype not in dtype_mapping:
        raise ValueError("Invalid dtype. Choose either 'int8', 'int16', or 'float'.")

    np_dtype, rasterio_dtype = dtype_mapping[dtype]
    array = array.astype(np_dtype)
    nodata_value = int(no_data_out) if "int" in dtype else float(no_data_out)

    # Write the array to a GeoTIFF file
    with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=num_bands,
            dtype=rasterio_dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value
    ) as dst:
        for band in range(num_bands):
            dst.write(array[band, :, :], band + 1)