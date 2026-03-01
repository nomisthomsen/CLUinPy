import os
import rasterio
import numpy as np
from datetime import datetime
from typing import List, Optional, Union

from preprocessing import convert_list_to_np_stack
from sampling import extract_corr_samples, extract_sample_from_shapefile, draw_stratified_sample_by_class
from feature_selection import threshold_vif
from models import calc_log_regression_for_class, calc_random_forest_for_class, calc_xgboost_for_class, calc_mlp_for_class, calc_svm_for_class
from prediction import predict_models
from io_utils import write_logfile, write_array_to_geotiff


def suitability(classification: str, env_vars: List[str], mode: Union[str, List[str]],
                out_path: str, n_samples_corr: int,
                vif_threshold: int = 5, min_distance: int = 1, test_fraction: float = 0.3,
                random_state: Optional[int] = None, sample_points_shapefile: Optional[str] = None, sample_size_list: Optional[List[int]] = None,
                dynamic: bool = False, dyn_years: Optional[List[int]] = None,
                dyn_vars: Optional[List[np.ndarray]] = None, ensemble: bool = False,
                no_data_value: int = -9999, predict_outputs: bool = True) -> None:
    """
    Performs land suitability classification using selected models.
    Supports model-only mode (no predictions) and ensemble modeling.

    Parameters:
    - classification: Path to land cover raster file.
    - env_vars: List of paths to environmental variable rasters.
    - mode: Model(s) to use ('logistic', 'random_forest', 'XGBoost', 'SVM', 'MLP').
    - out_path: Directory for saving outputs.
    - n_samples_corr: Number of samples for correlation analysis.
    - sample_size_list: Number of samples to draw per class.
    - vif_threshold: VIF threshold for multicollinearity filtering.
    - min_distance: Minimum spatial distance between samples.
    - test_fraction: Fraction of data used for testing.
    - random_state: Seed for reproducibility.
    - dynamic: If True, dynamic variables are included.
    - dyn_years: List of dynamic years (if dynamic = True).
    - dyn_vars: List of dynamic variable arrays (if dynamic = True).
    - ensemble: If True, compute ensemble from multiple models.
    - no_data_value: Value indicating missing data in input rasters.
    - predict_outputs: If False, skip prediction and output writing.
    """

    # Ensure mode is a list even if a single string is passed
    if isinstance(mode, str):
        mode = [mode]

    # Define which models require feature scaling
    scale_required = {
        "logistic": True,
        "random_forest": False,
        "XGBoost": False,
        "MLP": True,
        "SVM": True
    }

    # Mapping of model names to their training functions
    model_functions = {
        "logistic": calc_log_regression_for_class,
        "random_forest": calc_random_forest_for_class,
        "XGBoost": calc_xgboost_for_class,
        "MLP": calc_mlp_for_class,
        "SVM": calc_svm_for_class
    }

    # Validate that all requested models are available
    invalid_modes = [m for m in mode if m not in model_functions]
    if invalid_modes:
        raise ValueError(f"Invalid mode(s): {invalid_modes}. Choose from {list(model_functions.keys())}")

    # --- Step 1: Load raster data and create valid region mask ---
    land_cover_array = rasterio.open(classification).read(1)
    suit_factor_array = convert_list_to_np_stack(env_vars)
    suit_factor_names = [os.path.basename(var)[:-4] for var in env_vars]

    # Create a binary mask: 1 for valid pixels, 0 for no-data
    mask = np.any(suit_factor_array == no_data_value, axis=0)
    binary_mask = (~mask).astype(int)
    region_array = binary_mask
    region_mask = 1


    # --- Step 2: Sample data for correlation analysis & VIF filtering ---
    sample_df = extract_corr_samples(suit_factor_array, region_array, region_mask, n_samples_corr, random_state)
    vif_results = threshold_vif(sample_df, suit_factor_names, vif_threshold)

    # Select variables that passed VIF filtering
    variable_names = vif_results['name'].tolist()
    variable_indices = vif_results['variable'].tolist()
    suit_factor_vif = suit_factor_array[variable_indices, :, :]

    # Extract dynamic variable names if applicable
    dyn_variables_name = [os.path.basename(var)[:-4] for var in dyn_vars] if dynamic else None

    # --- Step 3: Draw or load stratified samples for each class ---
    land_use_reshaped = land_cover_array[np.newaxis, :, :]
    data_array = np.concatenate((suit_factor_vif, land_use_reshaped), axis=0)
    land_use_unique = np.unique(land_cover_array)[np.unique(land_cover_array) >= 0].tolist()

    if sample_points_shapefile:
        print("Loading stratified sample from shapefile...")

        random_strat_sample = extract_sample_from_shapefile(sample_points_shapefile, classification, data_array, variable_names)

    else:
        if sample_size_list is None:
            raise ValueError("sample_size_list must be provided if sample_points_shapefile is not used.")
        print("Drawing stratified sample...")
        random_strat_sample = draw_stratified_sample_by_class(
            data_array, data_array.shape[0] - 1, land_use_unique,
            sample_size_list, min_distance, random_state
        )
        random_strat_sample.columns = variable_names + ['land_cover']


    # Timestamp for output folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize containers for ensemble predictions and AUC scores
    mode_predictions = {}
    model_auc_scores = {}

    # --- Step 4: Train models per class for each selected mode ---
    for current_mode in mode:
        outdir_name = os.path.join(out_path, f"{current_mode}_{timestamp}/")
        os.makedirs(outdir_name, exist_ok=True)

        # Train a model for each class
        model_list = [
            model_functions[current_mode](
                random_strat_sample, variable_names, class_id,
                test_fraction, out_path,
                timestamp, scale_required[current_mode], random_state
            )
            for class_id in land_use_unique
        ]

        # Log metadata and configuration
        write_logfile(
            out_path=outdir_name,
            classification=classification,
            env_variables= env_vars,
            n_samples_corr=n_samples_corr,
            sample_size_list=sample_size_list,
            sample_points_shapefile = sample_points_shapefile,
            vif_threshold=vif_threshold,
            vif_filtered_env_vars = variable_names,
            min_distance=min_distance,
            test_fraction=test_fraction,
            random_state=random_state,
            dynamic=dynamic,
            dyn_years=dyn_years,
            no_data_value=no_data_value
        )

        # --- Step 5: Optionally predict suitability maps and write outputs ---
        if predict_outputs:
            # Predict for each model and class
            mode_results = [
                predict_models(
                    class_id=class_id, model=model, filtered_vars=filtered_vars,
                    vif_vars=suit_factor_vif, variable_names=variable_names,
                    model_dir=os.path.join(out_path, f"{current_mode}_{timestamp}"),
                    year_list=dyn_years,
                    dyn_vars=dyn_vars, dyn_vars_name=dyn_variables_name
                )
                for class_id, model, filtered_vars in model_list
            ]

            # Save predictions for ensemble computation
            mode_predictions[current_mode] = mode_results

            # Read AUC scores from file
            model_auc_scores[current_mode] = {}
            auc_file_path = os.path.join(out_path, f"{current_mode}_{timestamp}", "auc.txt")
            with open(auc_file_path, "r") as auc_file:
                for line in auc_file:
                    if ":" in line:
                        class_id_str, auc_str = line.strip().split(":")
                        class_id = int(float(class_id_str.strip()))
                        auc = float(auc_str.strip())
                        model_auc_scores[current_mode][class_id] = auc

            # Write predicted maps to GeoTIFF
            if dyn_years:
                disagg_out_arr = list(map(list, zip(*mode_results)))
                out_arr = np.stack(disagg_out_arr[0])
                out_arr = np.where(binary_mask[None, :, :] == 1, out_arr, no_data_value)
                write_array_to_geotiff(out_arr, f"{outdir_name}suitability_stack.tif", classification, no_data_value)
                for i in range(len(dyn_years)):
                    outname = f"{outdir_name}suitability_stack_{dyn_years[i]}.tif"
                    out_arr = np.stack(disagg_out_arr[i+1])
                    out_arr = np.where(binary_mask[None, :, :] == 1, out_arr, no_data_value)
                    write_array_to_geotiff(out_arr, outname, classification, no_data_value)
            else:
                out_arr = np.stack(mode_results)
                out_arr = np.where(binary_mask[None, :, :] == 1, out_arr, no_data_value)
                write_array_to_geotiff(out_arr, f"{outdir_name}suitability_stack.tif", classification, no_data_value)

    # --- Step 6: Optional ensemble prediction ---
    if ensemble and len(mode) >= 3 and predict_outputs:
        print("Ensemble prediction in progress...")

        # Group predictions by class across modes
        classwise_predictions = list(zip(*[mode_predictions[m] for m in mode]))
        ensemble_results = []

        for class_idx, class_preds in enumerate(classwise_predictions):
            class_id = land_use_unique[class_idx]

            # Get AUC-based weights for models
            auc_values = np.array([
                model_auc_scores[m].get(class_id, 1.0) for m in mode
            ])
            auc_weights = auc_values / auc_values.sum()

            # Compute weighted average across modes
            if dyn_years:
                per_year_preds = list(zip(*class_preds))
                per_year_avg = []
                for year_preds in per_year_preds:
                    stacked_preds = np.stack(year_preds)
                    weighted = np.tensordot(auc_weights, stacked_preds, axes=(0, 0))
                    per_year_avg.append(weighted)
                ensemble_results.append(per_year_avg)
            else:
                stacked_preds = np.stack(class_preds)
                weighted = np.tensordot(auc_weights, stacked_preds, axes=(0, 0))
                ensemble_results.append(weighted)

        # Write ensemble predictions
        ensemble_outdir = os.path.join(out_path, f"ensemble_{timestamp}/")
        os.makedirs(ensemble_outdir, exist_ok=True)

        if dyn_years:
            disagg_ensemble = list(map(list, zip(*ensemble_results)))
            for i, arr in enumerate(disagg_ensemble):
                outname = f"{ensemble_outdir}suitability_ensemble_stack_{dyn_years[i + 1]}.tif" if i else f"{ensemble_outdir}suitability_ensemble_stack.tif"
                out_arr = np.stack(arr)
                out_arr = np.where(binary_mask[None, :, :] == 1, out_arr, no_data_value)
                write_array_to_geotiff(out_arr, outname, classification, no_data_value)
        else:
            out_arr = np.stack(ensemble_results)
            out_arr = np.where(binary_mask[None, :, :] == 1, out_arr, no_data_value)
            write_array_to_geotiff(out_arr, f"{ensemble_outdir}suitability_ensemble_stack.tif", classification, no_data_value)

        print(f"Ensemble suitability maps saved to: {ensemble_outdir}")

