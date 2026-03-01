import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import xgboost as xgb
from typing import List, Optional
import joblib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)


from io_utils import create_folder_and_file

def calc_log_regression_for_class(df: pd.DataFrame, variable_names: List[str], class_id: int, test_size: float,
                                  root_dir: str, timestamp: str, apply_scaling: bool=False, random_state: Optional[int] = None) -> tuple:
    """
    Calculates logistic regression for a specific land use class.

    Parameters:
    - df: DataFrame with explanatory variables and land use information.
    - variable_names: List of variable names.
    - class_id: Target ID of the land use class for logistic regression.
    - test_size: Fraction of the sample for testing.
    - root_dir: Root directory for saving results.
    - timestamp: Timestamp for folder naming.
    - random_state: Optional seed for reproducibility.

    Returns:
    - class_id: ID of the land use class.
    - model: Trained logistic regression model.
    - filtered_variables: Selected explanatory variables.
    """
    folder_name = os.path.join(root_dir, f'logistic_{timestamp}')
    os.makedirs(folder_name, exist_ok=True)

    # Create balanced binary dataset: target class vs all others
    df_sub = df[df["land_cover"] == class_id]
    df_sub2 = df[df["land_cover"] != class_id].sample(df_sub.shape[0], random_state=random_state)
    df_comb = pd.concat([df_sub, df_sub2])
    df_comb["presence"] = [1] * df_sub.shape[0] + [0] * df_sub.shape[0]

    x = df_comb[variable_names]
    y = df_comb["presence"]

    # Step 1: Split before feature selection
    x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Step 2: Temporarily scale all features for RFECV (but don’t save this scaler)
    if apply_scaling:
        temp_scaler = StandardScaler()
        x_train_scaled_temp = temp_scaler.fit_transform(x_train_full)
    else:
        x_train_scaled_temp = x_train_full

    # Step 3: Feature selection with RFECV on (optionally scaled) training data
    base_model = LogisticRegression(max_iter=3000, solver="liblinear")
    rfecv = RFECV(estimator=base_model, step=1, cv=StratifiedKFold(5), n_jobs=2,
                  min_features_to_select=3, scoring='roc_auc')
    rfecv.fit(x_train_scaled_temp, y_train)

    # Step 4: Extract selected variables
    filtered_variables = [var for var, keep in zip(variable_names, rfecv.get_support()) if keep]

    # Step 5: Reduce to selected features
    x_train_selected = x_train_full[filtered_variables]
    x_test_selected = x_test_full[filtered_variables]

    # Step 6: Apply final scaling and save it
    if apply_scaling:
        final_scaler = StandardScaler()
        x_train_selected = final_scaler.fit_transform(x_train_selected)
        x_test_selected = final_scaler.transform(x_test_selected)
        joblib.dump(final_scaler, os.path.join(folder_name, 'scaler_' + str(class_id) + '.pkl'))

    # Step 7: Fit final model on selected + scaled features
    final_model = LogisticRegression(max_iter=3000, solver="liblinear")
    final_model.fit(x_train_selected, y_train)

    # Step 8: Evaluate performance
    auc_roc = roc_auc_score(y_test, final_model.predict_proba(x_test_selected)[:, 1])
    intercept = final_model.intercept_[0]
    coefficients = final_model.coef_.tolist()[0]

    # Compute permutation importances
    perm_result = permutation_importance(
        final_model,
        x_test_selected,
        y_test,
        n_repeats=30,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1
    )
    perm_means = perm_result.importances_mean
    perm_stds = perm_result.importances_std
    sorted_idx = perm_means.argsort()[::-1]

    # Save results
    create_folder_and_file(folder_name, 'auc.txt', 'stats.txt')

    with open(os.path.join(folder_name, 'auc.txt'), 'a') as f:
        f.write(f"{class_id}: {auc_roc:.4f}\n\n")

    with open(os.path.join(folder_name, 'stats.txt'), 'a') as f:
        f.write(f"{class_id}\n")
        f.write(f"Intercept: {intercept:.4f}\n")
        f.write("Coefficients:\n")
        for var, coef in zip(filtered_variables, coefficients):
            f.write(f"{var}: {coef:.4f}\n")
        f.write("\nPermutation Importances (ROC AUC drop):\n")
        for i in sorted_idx:
            f.write(f"{filtered_variables[i]}: Mean={perm_means[i]:.4f}, Std={perm_stds[i]:.4f}\n")
        f.write("\n\n")

    return class_id, final_model, filtered_variables


def calc_random_forest_for_class(df: pd.DataFrame, variable_names: List[str], class_id: int, test_size: float,
                                 root_dir: str, timestamp: str, apply_scaling: bool=False, random_state: Optional[int] = None) -> tuple:
    """
    Calculates a Random Forest model for a specific land use class.

    Parameters:
    - df: DataFrame with explanatory variables and land use information.
    - variable_names: List of variable names.
    - class_id: Target ID of the land use class for Random Forest.
    - test_size: Fraction of the sample for testing.
    - root_dir: Root directory for saving results.
    - timestamp: Timestamp for folder naming.
    - random_state: Optional seed for reproducibility.

    Returns:
    - class_id: ID of the land use class.
    - best_rf: Trained Random Forest model.
    - variable_names: List of variable names.
    """
    folder_name = os.path.join(root_dir, f'random_forest_{timestamp}')
    os.makedirs(folder_name, exist_ok=True)

    df_sub = df[df["land_cover"] == class_id]
    df_sub2 = df[df["land_cover"] != class_id].sample(df_sub.shape[0], random_state=random_state)
    df_sub_comb = pd.concat([df_sub, df_sub2])
    df_sub_comb["presence"] = [1] * int(df_sub_comb.shape[0] / 2) + [0] * int(df_sub_comb.shape[0] / 2)

    df_x = df_sub_comb[[str(name) for name in variable_names]]
    df_y = df_sub_comb["presence"]

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)

    if apply_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Save scaler for prediction later
        joblib.dump(scaler, os.path.join(folder_name, 'scaler_' + str(class_id) + '.pkl'))

    rf = RandomForestClassifier(random_state=random_state)

    rf_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist,
                                          n_iter=50, cv=3, scoring='roc_auc', verbose=1,
                                          random_state=random_state, n_jobs=-1)
    rf_random_search.fit(x_train, y_train)
    best_rf = rf_random_search.best_estimator_

    y_pred_rf = best_rf.predict_proba(x_test)[:, 1]
    y_pred_rf = np.clip(y_pred_rf, 0, 1)

    rf_auc = roc_auc_score(y_test, y_pred_rf)

    feature_importances = pd.DataFrame({
        'feature': variable_names,
        'importance': best_rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    create_folder_and_file(folder_name, 'auc.txt', 'stats.txt')

    with open(os.path.join(folder_name, 'auc.txt'), 'a') as file:
        file.write(str(int(class_id)) + ': ' + str(rf_auc) + '\n\n')

    with open(os.path.join(folder_name, 'stats.txt'), 'a') as file:
        file.write(f"{class_id}\n\n")
        file.write("Feature Importances:\n")
        feature_importances.to_csv(file, index=False, mode='a', header=False)
        file.write("\nBest Parameters:\n")
        for param, value in best_rf.get_params().items():
            file.write(f"{param}: {value}\n")
        file.write('\n\n')

    return class_id, best_rf, variable_names


def calc_xgboost_for_class(df: pd.DataFrame, variable_names: List[str], class_id: int, test_size: float,
                           root_dir: str, timestamp: str, apply_scaling: bool=False, random_state: Optional[int] = None) -> tuple:
    """
    Calculates an XGBoost model for a specific land use class.

    Parameters:
    - df: DataFrame with explanatory variables and land use information.
    - variable_names: List of variable names.
    - class_id: Target ID of the land use class for XGBoost.
    - test_size: Fraction of the sample for testing.
    - root_dir: Root directory for saving results.
    - timestamp: Timestamp for folder naming.
    - random_state: Optional seed for reproducibility.

    Returns:
    - class_id: ID of the land use class.
    - best_xgb: Trained XGBoost model.
    - variable_names: List of variable names.
    """
    folder_name = os.path.join(root_dir, f'XGBoost_{timestamp}')
    os.makedirs(folder_name, exist_ok=True)

    df_sub = df[df["land_cover"] == class_id]
    df_sub2 = df[df["land_cover"] != class_id].sample(df_sub.shape[0], random_state=random_state)
    df_sub_comb = pd.concat([df_sub, df_sub2])
    df_sub_comb["presence"] = [1] * int(df_sub_comb.shape[0] / 2) + [0] * int(df_sub_comb.shape[0] / 2)

    df_x = df_sub_comb[[str(name) for name in variable_names]]
    df_y = df_sub_comb["presence"]

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)

    if apply_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Save scaler for prediction later
        joblib.dump(scaler, os.path.join(folder_name, 'scaler_' + str(class_id) + '.pkl'))

    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=random_state)

    xgb_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb_random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=xgb_param_dist,
                                           n_iter=50, cv=3, scoring='roc_auc', verbose=1,
                                           random_state=random_state, n_jobs=-1)
    xgb_random_search.fit(x_train, y_train)
    best_xgb = xgb_random_search.best_estimator_

    y_pred_xgb = best_xgb.predict_proba(x_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, y_pred_xgb)

    feature_importances = pd.DataFrame({
        'feature': variable_names,
        'importance': best_xgb.feature_importances_
    }).sort_values(by='importance', ascending=False)

    create_folder_and_file(folder_name, 'auc.txt', 'stats.txt')

    with open(os.path.join(folder_name, 'auc.txt'), 'a') as file:
        file.write(str(int(class_id)) + ': ' + str(xgb_auc) + '\n\n')

    with open(os.path.join(folder_name, 'stats.txt'), 'a') as file:
        file.write(f"{int(class_id)}\n\n")
        file.write("Feature Importances:\n")
        feature_importances.to_csv(file, index=False, mode='a', header=False)
        file.write("\nBest Parameters:\n")
        for param, value in best_xgb.get_params().items():
            file.write(f"{param}: {value}\n")
        file.write('\n\n')

    return class_id, best_xgb, variable_names


def calc_mlp_for_class(
    df: pd.DataFrame,
    variable_names: List[str],
    class_id: int,
    test_size: float,
    root_dir: str,
    timestamp: str,
    apply_scaling: bool=True,
    random_state: Optional[int] = None
) -> tuple[int, MLPClassifier, List[str]]:
    """
    Trains a Multi-Layer Perceptron (MLP) classifier to model land suitability for a given class,
    using hyperparameter tuning with RandomizedSearchCV.

    Parameters:
    - df (pd.DataFrame): DataFrame containing explanatory variables and land cover information.
    - variable_names (List[str]): List of feature names used for training.
    - class_id (int): Target land cover class ID to model.
    - test_size (float): Fraction of the data to be used as the test set.
    - root_dir (str): Root directory for saving results.
    - timestamp (str): Timestamp string for naming output folders.
    - random_state (Optional[int]): Random seed for reproducibility.

    Returns:
    - Tuple[int, MLPClassifier, List[str]]:
        - class_id (int): The target class ID.
        - best_mlp (MLPClassifier): The best trained MLP model after hyperparameter tuning.
        - variable_names (List[str]): The list of feature names used.
    """
    folder_name = os.path.join(root_dir, f'MLP_{timestamp}')
    os.makedirs(folder_name, exist_ok=True)

    # Subset target class samples
    df_sub = df[df["land_cover"] == class_id]
    # Sample an equal number of non-target samples for balanced training
    df_sub2 = df[df["land_cover"] != class_id].sample(df_sub.shape[0], random_state=random_state)
    # Combine target and non-target samples
    df_sub_comb = pd.concat([df_sub, df_sub2])
    df_sub_comb["presence"] = [1] * df_sub.shape[0] + [0] * df_sub.shape[0]

    # Define feature matrix (X) and target vector (y)
    df_x = df_sub_comb[variable_names]
    df_y = df_sub_comb["presence"]

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)

    if apply_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Save scaler for prediction later
        joblib.dump(scaler, os.path.join(folder_name, 'scaler_' + str(class_id) + '.pkl'))

    # Define base MLP model
    mlp = MLPClassifier(random_state=random_state, max_iter=1500, early_stopping=True)

    # Define hyperparameter search space
    param_dist = {
        'hidden_layer_sizes': [
            (50,), (100,), (50, 50), (100, 50), (100, 100)
        ],
        'solver': ['adam'],  # Keeping it stable; can add 'lbfgs' or 'sgd' later if needed
        'alpha': [1e-4, 1e-3, 1e-2],  # Regularization strength
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000],
        'batch_size': [32, 64]
    }
    # Perform hyperparameter tuning using RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=10,
        cv=2,
        scoring='roc_auc',
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(x_train, y_train)
    best_mlp = random_search.best_estimator_

    # Predict probabilities on the test set and calculate AUC
    y_pred_mlp = best_mlp.predict_proba(x_test)[:, 1]
    mlp_auc = roc_auc_score(y_test, y_pred_mlp)

    # Save AUC and model parameters to files
    create_folder_and_file(folder_name, "auc.txt", "stats.txt")

    perm_result = permutation_importance(
        best_mlp, x_test, y_test,
        scoring='roc_auc',
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1
    )

    importances = sorted(
        zip(variable_names, perm_result.importances_mean),
        key=lambda x: x[1], reverse=True
    )

    with open(os.path.join(folder_name, "auc.txt"), "a") as file:
        file.write(f"{int(class_id)}: {mlp_auc}\n\n")

    with open(os.path.join(folder_name, "stats.txt"), "a") as file:
        file.write(f"{class_id}\n")
        # MLPClassifier does not have a single intercept, so we log its parameters instead.
        file.write("MLP Parameters:\n")
        for param, value in best_mlp.get_params().items():
            file.write(f"{param}: {value}\n")
        # Optionally, you can also log the learned weights:
        file.write("Learned Weights (coefs_):\n")
        for i, coef in enumerate(best_mlp.coefs_):
            file.write(f"Layer {i} weights shape {coef.shape}\n")
        file.write("\nPermutation Feature Importances:\n")
        for feature, importance in importances:
            file.write(f"{feature}: {importance:.4f}\n")
        file.write("\n\n")

    return class_id, best_mlp, variable_names


def calc_svm_for_class(
    df: pd.DataFrame,
    variable_names: List[str],
    class_id: int,
    test_size: float,
    root_dir: str,
    timestamp: str,
    apply_scaling: bool=False,
    random_state: Optional[int] = None
) -> tuple[int, SVC, list[str]]:
    """
    Trains a Support Vector Machine (SVM) classifier to model land suitability for a given class.
    Uses hyperparameter tuning with RandomizedSearchCV to optimize model performance.

    Parameters:
    - df (pd.DataFrame): DataFrame containing explanatory variables and land cover class labels.
    - variable_names (List[str]): List of feature (predictor) variable names used for training.
    - class_id (int): Target land cover class ID to model.
    - test_size (float): Fraction of the data to be used as the test set.
    - random_state (Optional[int]): Random seed for reproducibility (default: None).

    Returns:
    - Tuple[int, SVC, float]:
        - class_id (int): The target class ID.
        - best_svm (SVC): The best trained SVM model after hyperparameter tuning.
        - svm_auc (float): The AUC score of the trained model.
    """
    folder_name = os.path.join(root_dir, f'SVM_{timestamp}')
    os.makedirs(folder_name, exist_ok=True)

    # Select target class samples
    df_sub = df[df.land_cover == class_id]

    # Select an equal number of non-target samples for balanced training
    df_sub2 = df[df.land_cover != class_id].sample(df_sub.shape[0], random_state=random_state)

    # Combine target and non-target samples
    df_sub_comb = pd.concat([df_sub, df_sub2])

    # Create a binary target variable: 1 for target class, 0 for non-target class
    df_sub_comb['presence'] = [1] * (df_sub_comb.shape[0] // 2) + [0] * (df_sub_comb.shape[0] // 2)

    # Define feature matrix (X) and target variable (y)
    df_x = df_sub_comb[variable_names]
    df_y = df_sub_comb['presence']

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)

    if apply_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Save scaler for prediction later
        joblib.dump(scaler, os.path.join(folder_name, 'scaler_' + str(class_id) + '.pkl'))

    # Define base SVM model
    svm = SVC(probability=True, random_state=random_state)

    # Define hyperparameter search space
    param_dist = {
        'C': np.logspace(-3, 3, 10),  # Regularization strength
        'kernel': ['linear'],  # Kernel type ['linear', 'rbf', 'poly', 'sigmoid']
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 5)),  # Kernel coefficient
        'degree': [2, 3, 4, 5]  # Only relevant for polynomial kernel
    }

    # Perform Randomized Search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_iter=20,  # Number of random configurations to test
        cv=2,
        scoring='roc_auc',  # 3-fold cross-validation
        verbose=1,  # Print search progress
        random_state=random_state,
        n_jobs=-1  # Use all available CPU cores
    )

    # Train SVM with the best found hyperparameters
    random_search.fit(x_train, y_train)
    best_svm = random_search.best_estimator_

    # Predict probabilities on the test set
    y_pred_svm = best_svm.predict_proba(x_test)[:, 1]

    # Calculate the AUC (Area Under the Curve) score
    svm_auc = roc_auc_score(y_test, y_pred_svm)

    perm_result = permutation_importance(
        best_svm, x_test, y_test,
        scoring='roc_auc',
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1
    )

    importances = sorted(
        zip(variable_names, perm_result.importances_mean),
        key=lambda x: x[1], reverse=True
    )

    # Write AUC and model stats to files
    create_folder_and_file(folder_name, "auc.txt", "stats.txt")

    with open(os.path.join(folder_name, "auc.txt"), "a") as file:
        file.write(f"{int(class_id)}: {svm_auc}\n\n")

    with open(os.path.join(folder_name, "stats.txt"), "a") as file:
        file.write(f"{class_id}\n")
        file.write("SVM Parameters:\n")
        for param, value in best_svm.get_params().items():
            file.write(f"{param}: {value}\n")
        if hasattr(best_svm, "coef_"):
            file.write("Coefficients:\n")
            file.write(f"{best_svm.coef_}\n")
            file.write("\nPermutation Feature Importances:\n")
            for feature, importance in importances:
                file.write(f"{feature}: {importance:.4f}\n")
        file.write("\n\n")

    return class_id, best_svm, variable_names