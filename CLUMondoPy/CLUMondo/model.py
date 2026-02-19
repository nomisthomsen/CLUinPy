import os
from osgeo import gdal
import numpy as np
import pandas as pd
import random
import rasterio
from typing import List, Optional, Union

from .io_utils import writeArray2GeoTIFF, check_no_data_value
from .logging_utils import create_timestamped_subfolder, log_metadata, log_initial_data
from .neighbourhood import calc_neigh
from .transitions import calc_change
from .demand import comp_demand
from .age import calc_age, autonomous_change

def clumondo_dynamic(land_array: np.ndarray,
                     suit_array: np.ndarray,
                     region_array: np.ndarray,
                     neigh_weights: np.ndarray,
                     start_year: int,
                     end_year: int,
                     demand: np.ndarray,
                     dem_weights: np.ndarray,
                     lus_conv: np.ndarray,
                     lus_matrix_path: str,
                     allow: np.ndarray,
                     conv_res: np.ndarray,
                     max_diff_allow: float,
                     totdiff_allow: float,
                     max_iter: int,
                     out_dir: str,
                     crs: str,
                     dtype: str,
                     ref_raster_path: str,
                     change_years: List[int],
                     change_paths: List[str],
                     metadata: List[str],
                     age_array: Optional[np.ndarray] = None,
                     zonal_array: Optional[np.ndarray] = None,
                     preference_array: Optional[np.ndarray] = None,
                     preference_weights: Optional[np.ndarray] = None,
                     width_neigh: int = 1,
                     demand_max: float = 3,
                     demand_setback: float = 0.5,
                     demand_reset: int = 0,
                     no_data_out: int = 9999,
                     out_year: Optional[Union[int, List[int]]] = None,
                     no_data_value: int = -9999) -> None:
    """
    Perform Clumondo model simulation.

    Parameters:
        land_array (numpy.ndarray): Array representing the initial land cover.
        suit_array (numpy.ndarray): Array representing regional suitability values.
        region_array (numpy.ndarray): Array representing regions where land cover change may be restricted.
        neigh_weights (numpy.ndarray): Array of weights for neighborhood influence on land cover change.
        start_year (int): Start year of the simulation.
        end_year (int): End year of the simulation.
        demand (list of numpy.ndarray): List of arrays representing land use service demands for each year.
        dem_weights (numpy.ndarray): Array of weights for land use services demands.
        lus_conv (numpy.ndarray): Array representing conversion factors for land use services demands.
        lus_matrix_path (str): Array representing land use service matrix OR path to dynamic lus_matrix files.
        allow (numpy.ndarray): Array representing allowed land use changes.
        conv_res (numpy.ndarray): 1d array which represents the conversion resistances per land use class.
        max_diff_allow (float): Maximum allowed difference in land cover change.
        totdiff_allow (float): Maximum allowed total difference in land cover change.
        max_iter (int): Maximum number of iterations for each year.
        out_dir (str): Output directory path where results are stored.
        crs (str): CRS for the output files (e.g. 'EPSG:32750').
        dtype(str): Datatype for output files. One of 'int8', 'int16' or 'float32'.
        ref_raster_path (str): Path to the reference raster file.
        change_years (list of int): Years in which changes in suitability (`suit_array`) occur.
        change_paths (list of str): Paths to the suitability change raster files.
        metadata (list of str): List of paths to input raster data (land cover, suitability etc.)
        age_array (numpy.ndarray, optional): Array representing the age of land cover. Defaults to None.
        zonal_array (numpy.ndarray, optional): Array representing defined zones for specific land cover expansion (e.g. concession areas)
        preference_array (numpy.ndarray, optional): Array representing zones which should be prefered for specific land cover expansion.
                                                    Similar to zonal_array, but less restrictive.
        preference_weights (np.ndarray, optional): 1d array representing the weights of preferences in preference_array for each land cover class.
        width_neigh (int, optional): Width that should be applied to calculate the influence of neighbouring pixels. A width of 1 equals a 3x3 window, 2 a 5x5 window etc.
        demand_max (float, optional): Maximum demand elasticity values (absolute), before it is set back.
        demand_setback (float, optional): Absolute set back value after reaching maximum demand elasticity (demand_max).
        demand_reset (int, optional): Either 0 or 1. Whether demand elasticities should be reset to 0 for a new time step (1). If 0 (default), matching demand elasticities from previous time step are taken.
        no_data_out (int, optional): Value to insert for no_data_values in the output file. Depending on datatype, either -127 (int8) or -9999 (int16, float32) is recommended.
        out_year (int or list of int, optional): Either an integer or a list of integers indicating years for which output rasters should be written.
                                                If nothing is provided, output will be written for all years.
        no_data_value (int, optional): Value representing no data. Defaults to -9999.

    """
    # Information from reference raster to write rasters in the function
    raster_ds = gdal.Open(ref_raster_path)
    # Extract information from reference raster
    cols = raster_ds.RasterXSize
    rows = raster_ds.RasterYSize
    cell_res = int(raster_ds.GetGeoTransform()[1])
    x_origin = int(raster_ds.GetGeoTransform()[0])
    y_origin = int(raster_ds.GetGeoTransform()[3] - (cell_res * rows))
    ras_info = [cols, rows, x_origin, y_origin, cell_res]
    ras_info.append(no_data_value)

    # Update region array with no data values from suitability layer
    region_array[suit_array[0]==no_data_value] = 1

    if out_year is None:
        out_year_set = None  # means "all years"
    elif isinstance(out_year, int):
        out_year_set = {out_year}
    elif isinstance(out_year, (list, tuple)):
        out_year_set = set(out_year) if len(out_year) > 0 else None  # empty -> all
    else:
        raise ValueError("Parameter 'out_year' must be an int, list/tuple of ints, or None")

    # Autonomous change activate?
    autonomous_change_mode = False
    if np.max(allow) > 1000:
        # If there is a value > 1000 in the allow matrix, autonomous change is applied
        autonomous_change_mode = True

    years = range(start_year, end_year+1)
    for year in years:
        print(year)
        i = year - min(years)

        # Check if lus_matrix is xlsx file or not (in this case update lus_matrix by year)
        if lus_matrix_path.endswith(".xlsx"):
            lus_matrix = pd.read_excel(lus_matrix_path).iloc[:, 1:].to_numpy()
        else:
            new_filename = f"{lus_matrix_path}yield_data_{year}.xlsx"
            # Check if the dynamically constructed filename ends with ".xlsx"
            if new_filename.endswith(".xlsx"):
                lus_matrix = pd.read_excel(new_filename).iloc[:, 1:].to_numpy()
            else:
                print(f"No valid file found for year {year}")

        if year in change_years:
            index = change_years.index(year)
            suit_array = rasterio.open(change_paths[index]).read()
            land_array, suit_array = check_no_data_value(land_array, suit_array, no_data_value)

        if i == 0:
            # Initialize demand elasticities array
            dem_elas = np.zeros(len(dem_weights), dtype="float32")
            # Create a timestamped subfolder for each year
            subdir = create_timestamped_subfolder(out_dir)
            log_file_path = os.path.join(subdir, 'logfile.txt')

            log_initial_data(log_file_path=log_file_path,
                             start_year=start_year,
                             end_year=end_year,
                             change_years=change_years,
                             neigh_weights=neigh_weights,
                             conv_res=conv_res,
                             allow=allow,
                             demand=demand,
                             dem_weights=dem_weights,
                             max_iter=max_iter,
                             max_diff_allow=max_diff_allow,
                             totdiff_allow=totdiff_allow,
                             metadata=metadata,
                             lus_conv=lus_conv)

            # Set initial land cover and age arrays
            old_cov = land_array
            if age_array is not None:
                old_age = age_array

        if demand_reset == 1:
            # Set demand elasticities back to zero before entering the while loop
            dem_elas = np.zeros(len(dem_weights), dtype="float32")

        loop = 0
        maxdiff = 1000
        totdiff = 1000

        # Initialize demand elasticities array

        # Generate a random seed for speed calculation
        seed = random.random()
        if seed > 0.9 or seed < 0.001:
            seed = 0.05
        speed = seed

        # Calculate neighbourhood
        neigh_array = calc_neigh(land_cover_array=old_cov,
                                 width=width_neigh,
                                 weights=neigh_weights,
                                 no_data_value=no_data_value)

        # Iterate until convergence or max iterations reached
        while loop < max_iter and (maxdiff > max_diff_allow or totdiff > totdiff_allow):
            # Calculate land cover change
            land_array = calc_change(land_cover_array=old_cov,
                                     suit_array_stack=suit_array,
                                     region_array=region_array,
                                     neigh_array_stack=neigh_array,
                                     dem_weights=dem_weights,
                                     dem_elas=dem_elas,
                                     conv_res=conv_res,
                                     allow=allow,
                                     lus_conv=lus_conv,
                                     zonal_array=zonal_array,
                                     preference_array=preference_array,
                                     preference_weights=preference_weights,
                                     age_array=age_array)
            # Calculate demand elasticities
            dem_elas, maxdiff, totdiff, diffarr = comp_demand(demand_i=demand[i],
                                                              cur_land_use_array=land_array,
                                                              lus_matrix=lus_matrix,
                                                              dem_elas=dem_elas,
                                                              speed=speed,
                                                              demand_max=demand_max,
                                                              demand_setback=demand_setback)
            loop += 1
            print(f"year: {year}, loop: {loop}, totdiff: {totdiff}, maxdiff: {maxdiff}, differences: {diffarr} ,elasticities: {dem_elas}")
            # Log metadata for each iteration
            log_metadata(log_file_path,
                         f"Year: {year}, loop: {loop}, demand elasticities: {dem_elas}, differences: {diffarr},total difference: {totdiff}, maximum difference: {maxdiff}")
        if loop == max_iter:
            # If maximum number of loops is reached and no solution is found, break the process and write last iterated raster to drive
            print('Error')
            outname = os.path.join(subdir, 'cov' + str(year) + '_error.tif')
            new_cov_out = land_array.copy()
            new_cov_out[new_cov_out == no_data_value] = no_data_out
            writeArray2GeoTIFF(new_cov_out, outname, ras_info, no_data_out, crs, dtype)
            break
            
        # Log separator between iterations
        log_metadata(log_file_path, '##########################################')
        new_cov = land_array
        if age_array is not None:
            # Apply bottom-up autonomous changes based on age, if enabled
            if autonomous_change_mode:
                new_cov = autonomous_change(new_cov, old_cov, old_age, allow, no_data_value)
            # Calculate new age array if provided
            new_age = calc_age(old_cov, new_cov, old_age)
            if out_year_set is None or year in out_year_set:
                new_age_out = new_age.copy()
                new_age_out[new_age_out == no_data_value] = no_data_out
                outname = os.path.join(subdir, 'age' + str(year) + '.tif')
                writeArray2GeoTIFF(new_age_out, outname, ras_info, no_data_out, crs, dtype)
            old_age = new_age

        # Write new land cover raster
        if out_year_set is None or year in out_year_set:
            outname = os.path.join(subdir, 'cov' + str(year) + '.tif')
            new_cov_out = new_cov.copy()
            new_cov_out[new_cov_out == no_data_value] = no_data_out
            writeArray2GeoTIFF(new_cov_out, outname, ras_info, no_data_out, crs, dtype)
        old_cov = new_cov