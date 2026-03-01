import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from typing import List, Optional, Union


def class_freq_raster(classification: str, no_data_value: Union[int, float]) -> pd.DataFrame:
    """
    Computes the frequency of unique classes in a raster file, ignoring a specified no-data value.

    Parameters:
    classification (str): Path to the raster file.
    no_data_value (Union[int, float]): The value in the raster to be treated as no-data.

    Returns:
    pd.DataFrame: A DataFrame containing unique class values and their respective counts.
    """
    arr = rasterio.open(classification).read(1)
    arr_no_data = np.where(arr == no_data_value, np.nan, arr)
    arr_unique, arr_counts = np.unique(arr_no_data[~np.isnan(arr_no_data)], return_counts=True)
    table = np.column_stack((arr_unique, arr_counts))
    return pd.DataFrame(table, columns=["class", "frequency"])



def extract_corr_samples(variable_array: np.ndarray, region_array: np.ndarray, region_mask_value: int, n_sample: int,
                         random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Extracts samples for correlation and collinearity analysis from a 3D numpy array.

    Parameters:
    variable_array (np.ndarray): 3D numpy array with the variables from which to extract data.
    region_array (np.ndarray): 2D numpy array indicating the area from which to draw samples.
    region_mask_value (int): Integer value indicating the mask value in the region_array.
    n_sample (int): Number of samples to draw.
    random_state (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
    pd.DataFrame: A DataFrame with drawn samples for all variables.
    """
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.where(region_array == region_mask_value)
    sample_indices = np.random.choice(range(len(indices[0])), size=n_sample, replace=False)
    sample_list = [
        [sub_array[indices[0][sample_index], indices[1][sample_index]] for sample_index in sample_indices]
        for sub_array in variable_array
    ]
    return pd.DataFrame(sample_list).T



def sample_per_class(arr: np.ndarray, no_data_value: Union[int, float], method: str = 'fraction',
                     sample_fraction: Optional[float] = None, min_value: Optional[int] = None,
                     max_value: Optional[int] = None, sample_count: Optional[int] = None,
                     custom_samples: Optional[List[int]] = None) -> List[int]:
    """
    Samples data from an array based on class frequencies using different methods.

    Parameters:
    - arr (np.ndarray): Input array from which to sample.
    - no_data_value (Union[int, float]): Value representing no data in the array.
    - method (str): Sampling method ('fraction', 'count', or 'custom').
    - sample_fraction (Optional[float]): Fraction of each class to sample (used if method='fraction').
    - min_value (Optional[int]): Minimum number of samples per class (used if method='fraction').
    - max_value (Optional[int]): Maximum number of samples per class (used if method='fraction').
    - sample_count (Optional[int]): Fixed number of samples per class (used if method='count').
    - custom_samples (Optional[List[int]]): Custom list of sample counts per class (used if method='custom').

    Returns:
    - List[int]: List of sample counts per class.
    """
    # Flatten the array and remove no data values
    arr = arr.flatten()[arr.flatten() != no_data_value]

    # Get unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Initialize list to store samples per class
    samples_per_class: List[int] = []

    if method == 'fraction':
        if sample_fraction is None:
            raise ValueError("sample_fraction must be provided for 'fraction' method.")
        # Sampling based on a fraction of the total count per class
        for count in counts:
            fraction_sampled = int(count * sample_fraction)
            if min_value is not None:
                fraction_sampled = max(fraction_sampled, min_value)
            if max_value is not None:
                fraction_sampled = min(fraction_sampled, max_value)
            samples_per_class.append(fraction_sampled)

    elif method == 'count':
        if sample_count is None:
            raise ValueError("sample_count must be provided for 'count' method.")
        # Sampling based on a fixed sample count per class
        samples_per_class = [sample_count] * len(counts.tolist())

    elif method == 'custom':
        if custom_samples is None:
            raise ValueError("custom_samples must be provided for 'custom' method.")
        # Custom sampling based on provided list of sample counts
        samples_per_class = custom_samples
    else:
        # Raise an error if an invalid method is provided
        raise ValueError("Invalid method. Choose 'fraction', 'count', or 'custom'.")

    return samples_per_class



def draw_stratified_sample_by_class(data_array: np.ndarray, data_array_land_use_id: int,
                                    land_use_classes: List[int], sample_size_list: List[int],
                                    distance_width: int, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Creates a random stratified sample based on land use information.

    Parameters:
    - data_array (np.ndarray): 3D array containing explanatory variables and a land use layer.
    - data_array_land_use_id (int): Index of the land use layer in the 3D array.
    - land_use_classes (List[int]): List of land use classes for stratification.
    - sample_size_list (List[int]): Number of pixels to sample per class.
    - distance_width (int): Width of exclusion window around selected pixels.
    - random_state (Optional[int]): Seed for random number generation.

    Returns:
    - pd.DataFrame: DataFrame containing the stratified samples.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Extract the land use layer from the data array
    land_use_array = data_array[data_array_land_use_id]
    all_classes_indices: List[List[np.ndarray]] = []

    for i, land_use_class in enumerate(land_use_classes):
        print(land_use_class)
        # Find indices of the current land use class
        indices = np.transpose(np.where(land_use_array == land_use_class))
        num_pixels = sample_size_list[i]
        np.random.shuffle(indices)
        selected_indices = [indices[0]]  # Start with first randomly selected index

        for index in indices[1:]:
            adjacent_indices = []
            # Check a window around the current index to exclude adjacent pixels
            for di in range(-distance_width, distance_width + 1):
                for dj in range(-distance_width, distance_width + 1):
                    if di == 0 and dj == 0:
                        continue
                    neighbor_i, neighbor_j = index[0] + di, index[1] + dj
                    if 0 <= neighbor_i < land_use_array.shape[0] and 0 <= neighbor_j < land_use_array.shape[1]:
                        adjacent_indices.append((neighbor_i, neighbor_j))
            # Check for adjacency with selected indices
            adjacent_indices = list(set(adjacent_indices) & set(map(tuple, selected_indices)))
            if not adjacent_indices:  # If no adjacent indices found in selected_indices
                selected_indices.append(index)
            if len(selected_indices) == num_pixels:
                break
        else:
            print("Warning: Could not reach the desired percentage of non-adjacent pixels. Stopping the selection.")

        all_classes_indices.append(selected_indices)

    # Flatten the list of all class indices
    all_classes_indices_flat = [item for sublist in all_classes_indices for item in sublist]

    # Create a list of samples from the original data array based on selected indices
    sample_list_reg = [
        [data_array[layer][index[0], index[1]] for index in all_classes_indices_flat]
        for layer in range(data_array.shape[0])
    ]

    return pd.DataFrame(sample_list_reg).T


def extract_sample_from_shapefile(
    shapefile_path: str,
    raster_path : str,
    data_array: np.ndarray,
    variable_names: List[str]) -> pd.DataFrame:
    """
    Extracts sample values from a raster stack based on point geometries in a shapefile.

    Parameters:
    - shapefile_path (str): Path to the shapefile containing point geometries.
    - data_array (np.ndarray): 3D array of explanatory variables + land cover raster(layers, rows, cols).
    - variable_names (List[str]): Names of the explanatory variable layers.

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted values and land cover class per point.
    """
    gdf = gpd.read_file(shapefile_path)

    with rasterio.open(raster_path) as src:
        transform = src.transform
        raster_crs = src.crs

    # Reproject shapefile if CRS differs
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    if not gdf.geometry.is_valid.all():
        raise ValueError("Some geometries in the shapefile are invalid.")

    if not all(gdf.geometry.type == 'Point'):
        raise ValueError("Shapefile must contain only Point geometries.")

    land_cover_array = data_array[data_array.shape[0]-1]

    sample_list = []
    for _, row in gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        col, row_idx = rasterio.transform.rowcol(transform, x, y)

        if not (0 <= row_idx < land_cover_array.shape[0] and 0 <= col < land_cover_array.shape[1]):
            continue
        values = [layer[row_idx, col] for layer in data_array]
        sample_list.append(values)

    return pd.DataFrame(sample_list, columns=variable_names + ['land_cover'])