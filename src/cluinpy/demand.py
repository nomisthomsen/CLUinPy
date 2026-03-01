import numpy as np
import random

def comp_demand(demand_i: np.ndarray,
                cur_land_use_array: np.ndarray,
                lus_matrix: np.ndarray,
                dem_elas: np.ndarray,
                speed: float,
                demand_max : float = 3,
                demand_setback : float = 0.5,
                no_data_value: int = -9999) -> tuple:
    """
    Calculate demand elasticity and differences based on current land use and demand.

    Parameters:
        demand_i (numpy.ndarray): Array representing the demand for each land use service.
        cur_land_use_array (numpy.ndarray): Array representing the current land use.
        lus_matrix (numpy.ndarray): Array representing land use service matrix.
        dem_elas (numpy.ndarray): Array representing elasticity of demand for each land use service.
        speed (float): Speed variable.
        demand_max (float): Maximum demand elasticity values (absolute), before it is set back.
        demand_setback (float): Absolute set back value after reaching maximum demand elasticity (demand_max).
        no_data_value (int, optional): Value representing no data in cur_land_use_array. Defaults to -9999.

    Returns:
        tuple: A tuple containing the updated demand elasticities, maximum difference, and total difference.
    """
    # Update speed variable
    speed += 0.0001

    # Initialize an empty NumPy array for ran_list
    ran_list = np.empty(len(demand_i))

    # Generate random values and store them in ran_list
    for i in range(len(demand_i)):
        random.seed(101 - (1 / speed))
        ran_list[i] = random.randint(1, 100) + (1 / speed)

    # Get land cover frequency in the array
    arr_no_data = np.where(cur_land_use_array == no_data_value, np.nan, cur_land_use_array)
    arr1, arr2 = np.unique(arr_no_data, return_counts=True)
    freq_list = arr2.tolist()[:-1]

    # Calculate current provided demand by current land cover map
    demand_cur = np.zeros(len(demand_i))
    for i in range(len(freq_list)):
        demand_cur += lus_matrix[i] * freq_list[i]

    # Calculate differences and elasticity adjustments
    diff_arr = np.zeros(len(demand_i))
    max_diff_arr = np.zeros(len(demand_i))
    tot_diff = abs(np.sum(demand_cur) - np.sum(demand_i)) / np.sum(demand_i) * 100

    for i in range(len(demand_i)):
        dem_elas[i] -= ((demand_cur[i] - demand_i[i]) / demand_i[i]) / (speed * ran_list[i])
        if abs(dem_elas[i]) > demand_max:
            if dem_elas[i] > 0:
                dem_elas[i] = demand_setback
            else:
                dem_elas[i] = demand_setback * -1
        max_diff_arr[i] = (abs((demand_cur[i] - demand_i[i])) / demand_i[i]) * 100
        diff_arr[i] = ((demand_cur[i] - demand_i[i]) / demand_i[i]) * 100

    max_diff = np.max(max_diff_arr)

    return dem_elas, max_diff, tot_diff, diff_arr