import numpy as np
import rasterio
from scipy.ndimage import convolve
from typing import List, Optional, Union

def convert_list_to_np_stack(file_list: List[str]):
    """
    Converts a list of strings with paths to raster files (single band) into a 3d np array
    :param file_list: List of strings with paths to raster files.
    :return: 3d np array
    """
    raster_arrays = []
    for ras_file in file_list:
        with rasterio.open(ras_file) as asc:
            raster_arrays.append(asc.read(1))
    return np.stack(raster_arrays, axis=0)

def fill_nan_with_adjacent_mean(env_array: np.ndarray, region_array: np.ndarray, region_mask: int = 0, no_data_value:int = -9999) -> np.ndarray:
    """
    Fills no data entries in predictor variables (3d np array) with mean values from adjacent cells
    :param env_array: 3d np array with predictor variables.
    :param region_array: 2d np array indicating study region cells.
    :param region_mask: Integer value indicating cells inside the study region in region_array. Default is set to 0.
    :param no_data_value: No data value.
    :return: Returns a 3d np array (env_array) with filled values. Default set to -9999.
    """
    filled_array = env_array.copy()
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    for i in range(env_array.shape[0]):
        layer = filled_array[i]
        missing_mask = (region_array == region_mask) & ((layer == no_data_value) | np.isnan(layer))
        if np.any(missing_mask):
            valid_mask = (layer != no_data_value) & (~np.isnan(layer))
            sum_adj = convolve(layer * valid_mask, kernel, mode='constant', cval=0)
            count_adj = convolve(valid_mask.astype(int), kernel, mode='constant', cval=0)
            mean_adj = np.divide(sum_adj, count_adj, where=count_adj > 0)
            layer[missing_mask] = mean_adj[missing_mask]
    return filled_array

def standardize_array(env_array: np.ndarray, region_array: np.ndarray, region_mask: int = 0) -> np.ndarray:
    """
    Standardizes values in a 3d numpy array with predictor variables
    :param env_array: 3d np array with predictor variables.
    :param region_array: 2d np array indicating study region cells.
    :param region_mask: Integer value indicating cells inside the study region in region_array. Default is set to 0.
    :return:
    """
    standardized_array = env_array.copy()
    for i in range(env_array.shape[0]):
        layer = env_array[i]
        valid_mask = region_array == region_mask
        if np.any(valid_mask):
            valid_values = layer[valid_mask].astype(float)
            mean, std = np.mean(valid_values), np.std(valid_values)
            if std > 0:
                standardized_array[i][valid_mask] = (valid_values - mean) / std
    return standardized_array