import numpy as np
from typing import List
import rasterio
from rasterio.transform import from_origin


def writeArray2GeoTIFF(array: np.ndarray, outpath: str, rasInfo: List[float], no_data_out: int, crs: str,
                       dtype: str = "float") -> None:
    """
    Writes a 2D array to a GeoTIFF (.tif) file.

    :param array: The 2D array to be written to the GeoTIFF file.
    :param outpath: The output path for the GeoTIFF file.
    :param rasInfo: A list containing raster information: [NCOLS, NROWS, XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE]
    :param no_data_out: Value to insert for no_data_values in the output file. Depending on datatype, either -127 (int8) or -9999 (int16, float32) is recommended
    :param dtype: One of "int8", "int16", "float32"
    :param crs: A string indicating the EPSG in the following format "EPSG:4326"
    :return: None
    """
    # Extract raster info
    ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value = rasInfo
    nodata_value = no_data_out

    # Define the affine transformation
    transform = from_origin(xllcorner, yllcorner + nrows * cellsize, cellsize, cellsize)

    if dtype == "int16":
        array = array.astype(np.int16)
        nodata_value = int(nodata_value)
        rasterio_dtype = 'int16'
    elif dtype == "int8":
        array = array.astype(np.int8)
        nodata_value = int(nodata_value)
        rasterio_dtype = 'int8'
    elif dtype == "float":
        array = array.astype(np.float32)
        rasterio_dtype = 'float32'
    else:
        raise ValueError("Invalid dtype. Choose either 'int' or 'float'.")

    # Write the array to a GeoTIFF file
    with rasterio.open(
            outpath,
            'w',
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,
            dtype=rasterio_dtype,
            crs=crs,  # Define CRS (update if needed)
            transform=transform,
            nodata=nodata_value
    ) as dst:
        dst.write(array, 1)  # Write the array to the first band



def check_no_data_value(arr_2d: np.ndarray, arr_3d: np.ndarray, no_data_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Checks for and replaces no_data_value in both a 2D and a 3D numpy array.

    :param arr_2d: The 2D numpy array to check and replace no_data_value.
    :param arr_3d: The 3D numpy array to check and replace no_data_value across all slices.
    :param no_data_value: The value indicating no data.
    :return: Tuple containing the updated 2D and 3D numpy arrays with replaced no_data_value.
    """
    mask2d = (arr_2d == no_data_value)
    combined_mask = mask2d.copy()

    for i in range(arr_3d.shape[0]):
        combined_mask |= (arr_3d[i] == no_data_value)

    arr_2d[combined_mask] = no_data_value

    for i in range(arr_3d.shape[0]):
        arr_3d[i][combined_mask] = no_data_value

    return arr_2d, arr_3d