import argparse
import rasterio
import pandas as pd
import numpy as np

from CLUMondo.model import clumondo_dynamic



def parse_args():
    parser = argparse.ArgumentParser(description='Run the CLUMondo model.')
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--land_array', type=str, required=False, help='Path to land array.')
    parser.add_argument('--suit_array', type=str, required=False, help='Path to suitability array.')
    parser.add_argument('--region_array', type=str, required=False, help='Path to region array.')
    parser.add_argument('--neigh_weights', type=str, required=False, help='Neighbourhood weights, comma-separated.')
    parser.add_argument('--start_year', type=int, required=False, help='Start year.')
    parser.add_argument('--end_year', type=int, required=False, help='End year.')
    parser.add_argument('--demand', type=str, required=False, help='Path to demand file.')
    parser.add_argument('--dem_weights', type=str, required=False, help='Demand weights, comma-separated.')
    parser.add_argument('--lus_conv', type=str, required=False, help='Path to LUS conversion file.')
    parser.add_argument('--lus_matrix_path', type=str, required=False, help='Path to LUS matrix file.')
    parser.add_argument('--conv_res', type=str, required=False, help='Conversion resistance, comma-separated.')
    parser.add_argument('--allow', type=str, required=False, help='Path to allow file.')
    parser.add_argument('--out_dir', type=str, required=False, help='Output directory.')
    parser.add_argument('--crs', type=str, required=False, help='Output CRS.')
    parser.add_argument('--max_diff_allow', type=float, default=3.0, help='Maximum allowed difference.')
    parser.add_argument('--totdiff_allow', type=float, default=1.0, help='Total allowed difference.')
    parser.add_argument('--max_iter', type=int, default=3000, help='Maximum iterations.')
    parser.add_argument('--dtype', type=str, default='int16', help='Output data type.')
    parser.add_argument('--no_data_out', type=int, default=-9999, help='No data value for output.')
    parser.add_argument('--change_years', type=str, required=False, default='', help='Years to change reg_suit_array as a comma-separated string')
    parser.add_argument('--change_paths', type=str, required=False, default='',
                        help='Paths to reg_suit_array files for change years as a comma-separated string')
    parser.add_argument('--zonal_array', type=str, required=False, help='Path to zonal array.')
    parser.add_argument('--preference_array', type=str, required=False, help='Path to preference array')
    parser.add_argument('--preference_weights', type=str, required=False, help='Preference weight values, comma separated')
    parser.add_argument('--age_array', type=str, required=False, help='Path to age array.')
    parser.add_argument('--width_neigh', type=int, default=1, help='Window width for neighborhood analysis.')
    parser.add_argument('--demand_max', type=float, default=3.0, help='Maximum elasticity value for demand.')
    parser.add_argument('--demand_setback', type=float, default=0.5, help='Elasticity setback value in while loop if max value is reached')
    parser.add_argument('--demand_reset', type=int, default=0, help='Whether demand elasticities should be set back to zero for each time step (0/1)')
    parser.add_argument('--out_year', type=str, required=False, default='',
                        help='Comma-separated list of years or a single year to output results (e.g., "2030,2040" or "2050").')
    parser.add_argument('--no_data_value', type=int, default=-9999, help='No data value.')

    args = parser.parse_args()

    if args.config:
        config_args = {}
        with open(args.config, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                key = key.lstrip('--')  # remove leading dashes
                if ',' in value:
                    value = value.split(',')
                config_args[key] = value

        parser.set_defaults(**config_args)
        args = parser.parse_args()  # Re-parse with the new defaults

    return args


def main():
    args = parse_args()

    # Load data
    ref_raster_path = args.land_array
    land_array = rasterio.open(args.land_array).read(1)
    region_array = rasterio.open(args.region_array).read(1)
    suit_array = rasterio.open(args.suit_array).read()
    demand = pd.read_excel(args.demand).to_numpy()
    lus_conv = pd.read_excel(args.lus_conv).iloc[:, 1:].to_numpy()
    allow = pd.read_excel(args.allow).iloc[:, 1:].to_numpy()
    conv_res = np.array([float(x) for x in args.conv_res])
    neigh_weights = np.array([float(x) for x in args.neigh_weights])
    dem_weights = np.array([float(x) for x in args.dem_weights])
    zonal_array = rasterio.open(args.zonal_array).read() if args.zonal_array else None
    preference_array = rasterio.open(args.preference_array).read() if args.preference_array else None
    preference_weights = np.array([float(x) for x in args.preference_weights]) if args.preference_weights else None
    age_array = rasterio.open(args.age_array).read(1) if args.age_array else None
    no_data_value = args.no_data_value

    # Handle change_years
    if isinstance(args.change_years, str):
        change_years = [int(year.strip()) for year in args.change_years.split(',') if year.strip().isdigit()]
    elif isinstance(args.change_years, list):
        change_years = [int(year) for year in args.change_years if isinstance(year, (int, str)) and str(year).isdigit()] # Already a list, use as-is
    else:
        change_years = [] # Default to an empty list if None

    # Hande change paths
    if isinstance(args.change_paths, str):
        change_paths = args.change_paths.split(',')
    elif isinstance(args.change_paths, list):
        change_paths = args.change_paths  # Already a list, use as-is
    else:
        change_paths = []  # Default to an empty list if None

    # Handle output years to write rasters
    if isinstance(args.out_year, str):
        out_year = [int(year.strip()) for year in args.out_year.split(',') if year.strip().isdigit()]
    elif isinstance(args.out_year, list):
        out_year = [int(year) for year in args.out_year if isinstance(year, (int, str)) and str(year).isdigit()] # Already a list, use as-is
    else:
        out_year = None

    # Create a list of input variable as metadata to store in logfile
    metadata = [args.land_array, args.region_array, args.suit_array, args.lus_matrix_path]

    clumondo_dynamic(
        land_array=land_array,
        suit_array=suit_array,
        region_array=region_array,
        neigh_weights=neigh_weights,
        start_year=args.start_year,
        end_year=args.end_year,
        demand=demand,
        dem_weights=dem_weights,
        lus_conv=lus_conv,
        lus_matrix_path=args.lus_matrix_path,
        allow=allow,
        max_diff_allow=args.max_diff_allow,
        totdiff_allow=args.totdiff_allow,
        max_iter=args.max_iter,
        out_dir=args.out_dir,
        crs=args.crs,
        dtype=args.dtype,
        no_data_out=args.no_data_out,
        ref_raster_path=ref_raster_path,
        conv_res=conv_res,
        change_years=change_years,
        change_paths=change_paths,
        metadata=metadata,
        zonal_array=zonal_array,
        preference_array=preference_array,
        preference_weights=preference_weights,
        age_array=age_array,
        width_neigh=args.width_neigh,
        demand_max=args.demand_max,
        demand_setback=args.demand_setback,
        demand_reset=args.demand_reset,
        out_year=out_year,
        no_data_value=no_data_value
    )


if __name__ == "__main__":
    main()