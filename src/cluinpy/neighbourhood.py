import numpy as np
from numba import njit, prange

@njit(parallel=True)
def calc_neigh(land_cover_array: np.ndarray, width: int, weights: np.ndarray,
               no_data_value: float = -9999) -> np.ndarray:
    """
    Calculates neighborhood statistics for each unique class in a land cover array.

    :param land_cover_array: 2D numpy array representing the land cover data.
    :param width: Width parameter for defining the neighborhood window size.
    :param weights: Array of weights corresponding to each unique class.
    :param no_data_value: Value indicating no data (default is -9999).
    :return: 3D numpy array containing neighborhood class statistics.
    """
    # Calculate the window size based on the given width
    w = 1 + (width * 2)  # window size (width on each side of the center pixel)

    # Get the number of rows and columns in the input array
    n_rows = land_cover_array.shape[0]
    n_cols = land_cover_array.shape[1]

    # Initialize an array to handle no_data_value by converting them to np.nan
    land_cover_nan = np.empty_like(land_cover_array, dtype=np.float32)

    # Convert no_data_value to np.nan
    for i in range(n_rows):
        for j in range(n_cols):
            if land_cover_array[i, j] == no_data_value:
                land_cover_nan[i, j] = np.nan
            else:
                land_cover_nan[i, j] = land_cover_array[i, j]

    # Number of unique classes
    n_class = weights.shape[0]

    # Initialize an array to store neighborhood class data
    neigh_class_arr = np.zeros((n_class, n_rows, n_cols), dtype=np.float32)

    # Loop over each unique class to calculate neighborhood statistics
    for i in prange(n_class):
        class_value = i

        # Create a binary mask for the current class
        mask = np.zeros((n_rows, n_cols), dtype=np.int32)
        for j in range(n_rows):
            for k in range(n_cols):
                if land_cover_nan[j, k] == class_value:
                    mask[j, k] = 1

        # Pad the mask array with zeros to handle edge cases during windowing
        mask_pad = np.zeros((n_rows + 2 * width, n_cols + 2 * width), dtype=np.int32)
        for j in range(n_rows):
            for k in range(n_cols):
                mask_pad[j + width, k + width] = mask[j, k]

        # Calculate the sum of the mask within the window for each cell
        for j in range(n_rows):
            for k in range(n_cols):
                # Extract the window for the current cell
                window = mask_pad[j:j + w, k:k + w]

                # Sum the values in the window
                sum_window = np.sum(window)

                # Normalize by the window area and apply the weight for the current class
                neigh_class_arr[i, j, k] = (sum_window / (w * w)) * weights[i]

    # Return the array containing neighborhood class statistics
    return neigh_class_arr