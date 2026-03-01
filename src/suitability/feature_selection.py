import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List

def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for each variable in a DataFrame.

    Parameters:
    data (pd.DataFrame): A DataFrame containing the variables.

    Returns:
    pd.DataFrame: A DataFrame with variables and their corresponding VIF values.
    """
    vif_data = pd.DataFrame()
    vif_data["variable"] = data.columns
    vif_data["vif"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data


def threshold_vif(data: pd.DataFrame, names: List[str], threshold: float) -> pd.DataFrame:
    """
    Computes VIF and applies a threshold, iteratively removing variables with the highest VIF.

    Parameters:
    data (pd.DataFrame): A DataFrame with variables.
    names (List[str]): A list of variable names.
    threshold (float): VIF threshold for variable removal.

    Returns:
    pd.DataFrame: A DataFrame with selected variables and their corresponding VIF values.
    """
    vif_data = pd.DataFrame()
    vif_data["variable"] = data.columns
    vif_data["name"] = names
    vif_data["vif"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    while vif_data["vif"].max() > threshold:
        remove_index = vif_data["vif"].idxmax()
        data = data.drop(data.columns[remove_index], axis=1)
        names.pop(remove_index)
        vif_data = pd.DataFrame()
        vif_data["variable"] = data.columns
        vif_data["vif"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vif_data["name"] = names

    return vif_data