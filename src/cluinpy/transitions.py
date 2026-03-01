import numpy as np
from numba import njit, prange


@njit(parallel=True)
def calc_change(land_cover_array: np.ndarray,
                suit_array_stack: np.ndarray,
                region_array: np.ndarray,
                neigh_array_stack: np.ndarray,
                dem_weights: np.ndarray,
                dem_elas: np.ndarray,
                conv_res: np.ndarray,
                allow: np.ndarray,
                lus_conv: np.ndarray,
                zonal_array: np.ndarray=None,
                preference_array: np.ndarray=None,
                preference_weights: np.ndarray=None,
                age_array: np.ndarray=None,
                no_data_value: int = -9999) -> np.ndarray:
    """
    Calculate land cover change based on various factors.

    Parameters:
        land_cover_array (numpy.ndarray): Array representing the current land cover.
        suit_array_stack (numpy.ndarray): Array representing suitability values for land cover classes.
        region_array (numpy.ndarray): Array representing regions where land cover change may be restricted.
        neigh_array_stack (numpy.ndarray): Array representing neighbourhood influence for land cover classes.
        dem_weights (numpy.ndarray): Array of weights for land use services demands.
        dem_elas (numpy.ndarray): Array of elasticities for land use services demands.
        conv_res (numpy.ndarray): Array representing conversion resistance for land use classes.
        allow (numpy.ndarray): Array representing allowed land use changes.
        lus_conv (numpy.ndarray): Array representing conversion factors for land use services demands.
        zonal_array (numpy.ndarray, optional): Array representing defined zones for specific land cover expansion (e.g. concession areas)
        preference_array (numpy.ndarray, optional): Array representing zones which should be prefered for specific land cover expansion.
                                                    Similar to zonal_array, but less restrictive.
        preference_weights (np.ndarray, optional): 1d array representing the weights of preferences in preference_array for each land cover class.
        age_array (numpy.ndarray, optional): Array representing the age of land cover. Defaults to None.
        no_data_value (int, optional): Value representing no data in land_cover_array. Defaults to -9999.

    Returns:
        numpy.ndarray: Array representing the new land cover after change.
    """
    # Set variables for iteration
    n_row, n_col = land_cover_array.shape  # number of rows and columns
    n_demand = dem_weights.shape[0]  # number of land use services demands
    n_lcov = conv_res.shape[0]  # number of land use classes

    # Initialize new land cover array
    new_lc = np.zeros(land_cover_array.shape, dtype=np.int32)

    # Calculate demand elasticities
    dem_elas = (dem_weights * dem_elas)

    # Loop through rows and columns of the land use array
    for j in prange(n_row):
        for k in range(n_col):
            # If value is lower -9990 or region restriction is applied, no change in land cover
            old_cov = int(land_cover_array[j, k])
            suit_values = suit_array_stack[:, j, k]
            if zonal_array is not None:
                zonal_values = np.zeros(n_lcov, dtype=np.uint8)
                zonal_values = zonal_array[:, j, k].astype(np.uint8)
            else:
                zonal_values = np.zeros(n_lcov, dtype=np.uint8)
            if preference_array is not None and preference_weights is not None:
                pref_values = np.zeros(n_lcov, dtype=np.float32)
                pref_values = (preference_array[:, j, k] * preference_weights).astype(np.float32)
            else:
                pref_values = np.zeros(n_lcov, dtype=np.float32)
            if no_data_value in suit_values:
                region_array[j, k] = 1
            if old_cov == no_data_value or region_array[j, k] == 1:
                new_lc[j, k] = old_cov
            else:
                suit_values = suit_array_stack[:, j, k]
                neigh_values = neigh_array_stack[:, j, k]
                max_pot = -30
                allowed_row = allow[old_cov, :]

                # Calculate potential change in land cover
                for i in range(n_lcov):
                    # If conversion is completely forbidden
                    if allowed_row[i] == 0:
                        temp_max = -30

                    # If conversion is only allowed after a certain age
                    elif allowed_row[i] > 100 and age_array is not None:
                        allow_age = allowed_row[i] - 100
                        if age_array[j, k] >= allow_age:
                            # Age condition met: compute elasticity-based score
                            sum_elas = 0
                            for d in range(n_demand):
                                if lus_conv[old_cov, d] < lus_conv[i, d]:
                                    sum_elas += dem_elas[d]
                                elif lus_conv[old_cov, d] > lus_conv[i, d]:
                                    sum_elas -= dem_elas[d]
                            temp_max = suit_values[i] + pref_values[i] + neigh_values[i] + sum_elas
                        else:
                            # Not old enough for conversion
                            temp_max = -30

                    # If conversion is not allowed after a certain age
                    elif allowed_row[i] < -100 and age_array is not None:
                        allow_age = abs(allowed_row[i]) - 100
                        if age_array[j, k] >= allow_age:
                            # Too old to convert: block change
                            temp_max = -30
                        else:
                            # Still eligible for change based on age
                            sum_elas = 0
                            for d in range(n_demand):
                                if lus_conv[old_cov, d] < lus_conv[i, d]:
                                    sum_elas += dem_elas[d]
                                elif lus_conv[old_cov, d] > lus_conv[i, d]:
                                    sum_elas -= dem_elas[d]
                            if old_cov == i:
                                temp_max = suit_values[i] + pref_values[i] + neigh_values[i] + sum_elas + conv_res[i]
                            else:
                                temp_max = suit_values[i] + pref_values[i] + neigh_values[i] + sum_elas

                    # Check whether transition is only allowed in  specific zone (indicated by 2 instead of 2)
                    elif allowed_row[i] == 2:
                        if zonal_values[i] > 0:
                            sum_elas = 0
                            for d in range(n_demand):
                                if lus_conv[old_cov, d] < lus_conv[i, d]:
                                    sum_elas += dem_elas[d]
                                elif lus_conv[old_cov, d] > lus_conv[i, d]:
                                    sum_elas -= dem_elas[d]
                            temp_max = suit_values[i] + pref_values[i] + neigh_values[i] + sum_elas
                        else:
                            temp_max = -30

                    else:
                        # Normal case: conversion is allowed without age constraints
                        sum_elas = 0
                        for d in range(n_demand):
                            if lus_conv[old_cov, d] < lus_conv[i, d]:
                                sum_elas += dem_elas[d]
                            elif lus_conv[old_cov, d] > lus_conv[i, d]:
                                sum_elas -= dem_elas[d]
                        if old_cov == i:
                            temp_max = suit_values[i] + pref_values[i] + neigh_values[i] + sum_elas + conv_res[i]
                        else:
                            temp_max = suit_values[i] + pref_values[i] + neigh_values[i] + sum_elas

                    # Track the land cover type with the highest score
                    if temp_max > max_pot:
                        max_pot = temp_max
                        max_cov = i

                # Check for error condition and update new land cover
                if max_pot == -30:
                    print('Error')
                    break
                new_lc[j, k] = max_cov

    return new_lc