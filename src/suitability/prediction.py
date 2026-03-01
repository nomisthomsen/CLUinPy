import os
import joblib
import numpy as np
import rasterio
from sklearn.base import ClassifierMixin
from typing import List, Optional, Union


def batch_predict_proba(model, data, chunk_size=100000):
    result = np.empty((data.shape[0],), dtype=np.float32)
    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        result[start:end] = model.predict_proba(data[start:end])[:, 1]
    return result

def predict_models(class_id: int, model: ClassifierMixin, filtered_vars: List[str], vif_vars: np.ndarray,
                   variable_names: List[str], model_dir: Optional[str] = None,year_list: Optional[List[int]] = None,
                   dyn_vars: Optional[List[np.ndarray]] = None, dyn_vars_name: Optional[List[str]] = None,
                   no_data_value: int = -9999) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Predicts model probabilities for input variables and optionally applies dynamic variables for specific years.

    Parameters:
    - model (ClassifierMixin): A trained classifier model (e.g., XGBClassifier).
    - filtered_vars (List[str]): The list of filtered variable names to use in the prediction.
    - vif_vars (np.ndarray): The input variable data for prediction.
    - variable_names (List[str]): List of variable names corresponding to the input data.
    - year_list (Optional[List[int]]): A list of years for which dynamic variables are applied (default is None).
    - dyn_vars (Optional[List[np.ndarray]]): List of dynamic variable arrays (default is None).
    - dyn_vars_name (Optional[List[str]]): List of dynamic variable names (default is None).
    - no_data_value (int): The value representing no data in the input (default is -9999).

    Returns:
    - Union[np.ndarray, List[np.ndarray]]: The predicted probabilities for the variables. If `year_list` is provided, a list of arrays for each year is returned; otherwise, a single prediction array is returned.
    """


    # Masking the input data to exclude values equal to the no_data_value
    data_array = np.ma.masked_equal(vif_vars, no_data_value)

    # Identify the indices of the variables that are in the filtered list
    indices = [i for i, item in enumerate(variable_names) if item in filtered_vars]

    # Extract the relevant data using the identified indices
    data_array_flt = data_array[indices]

    # Flatten each layer of the selected data for prediction
    flattened_layers = [layer.flatten() for layer in data_array_flt]

    # Stack the flattened layers into a 2D array
    data_array_2d = np.vstack(flattened_layers).transpose()
    data_array_2d = data_array_2d.astype(np.float32)

    # Apply scaling
    if model_dir:
        scaler_path = model_dir +  "/scaler_" + str(class_id) + ".pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            data_array_2d = np.where(data_array_2d == no_data_value, np.nan, data_array_2d)
            valid_mask = ~np.isnan(data_array_2d).any(axis=1)
            valid_data = data_array_2d[valid_mask]
            valid_data = scaler.transform(valid_data)
            pred_array = model.predict_proba(valid_data)[:, 1]
            full_pred_year = np.full(data_array_2d.shape[0], no_data_value, dtype=np.float32)
            full_pred_year[valid_mask] = pred_array
            pred_array_2d = np.reshape(full_pred_year, (vif_vars.shape[1], vif_vars.shape[2]))
        else:
            # Apply the model to predict probabilities for the input data
            pred_array = batch_predict_proba(model, data_array_2d)  # Select the probability of class 1 (binary classification)
            # Reshape the predicted probabilities back to the original 2D shape
            pred_array_2d = np.reshape(pred_array, (vif_vars.shape[1], vif_vars.shape[2]))


    # Set the predicted values to no_data_value where the input was no data
    pred_array_2d[vif_vars[0] == no_data_value] = no_data_value

    # If a year_list is provided, process predictions for each year
    if year_list is not None:
        return_list = [pred_array_2d]  # Start with the initial predictions

        for year in year_list:
            # Extract indices for dynamic variables corresponding to the current year
            initial_indices = []
            dynamic_indices = []

            # Find dynamic variables that match the year and filtered patterns
            for pattern_index, pattern in enumerate(filtered_vars):
                for filename_index, filename in enumerate(dyn_vars_name):
                    if pattern in filename and str(year) in filename:
                        initial_indices.append(pattern_index)
                        dynamic_indices.append(filename_index)

            # Make a copy of the filtered data array to update with dynamic variables
            data_array_year = np.copy(data_array_flt)

            # Replace the corresponding initial variables with the dynamic variables for the current year
            for i in range(len(dynamic_indices)):
                dyn_sub_array = rasterio.open(dyn_vars[dynamic_indices[i]]).read(1)
                data_array_year[initial_indices[i]] = dyn_sub_array

            # Mask the data where no data values are present
            data_array_year = np.ma.masked_equal(data_array_flt, no_data_value)

            # Flatten the layers of the updated data for prediction
            flattened_layers_year = [layer.flatten() for layer in data_array_year]

            # Stack the flattened layers for the prediction
            data_array_2d_year = np.vstack(flattened_layers_year).transpose()

            # Predict probabilities for the current year using the model
            pred_array_year = batch_predict_proba(model, data_array_2d)

            # Reshape the predicted array back to the original 2D shape
            pred_array_2d_year = np.reshape(pred_array_year, (vif_vars.shape[1], vif_vars.shape[2]))
            pred_array_2d_year[vif_vars[0] == no_data_value] = no_data_value

            # Append the predicted array for the year to the return list
            return_list.append(pred_array_2d_year)

        # Return a list of prediction arrays (initial and year-specific)
        return return_list
    else:
        # If no year_list is provided, return the initial prediction array
        return pred_array_2d