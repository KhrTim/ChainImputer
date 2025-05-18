import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import poisson
from update import ChainImputer
from sklearn.impute import KNNImputer

import os
import pandas as pd
import numpy as np
import random
from glob import glob
from sklearn.preprocessing import StandardScaler

def missing_value_generator(X, missing_rate, seed=0):
    """
    Introduce column-wise heterogeneous missing values that sum to a global missing rate.
    
    Parameters:
        X (np.ndarray): Input data (2D array).
        missing_rate (float): Overall missing rate (0–100).
        seed (int): Random seed.

    Returns:
        X_missing (np.ndarray): Data with missing values (as np.nan).
    """
    np.random.seed(seed)
    X = np.asarray(X)
    n_rows, n_cols = X.shape

    # Step 1: Generate per-column missing rates (heterogeneous)
    raw = np.random.exponential(scale=1.0, size=n_cols)
    col_missing_rates = raw / raw.sum() * missing_rate  # Normalize to match overall missing rate

    # Step 2: Apply missingness column-wise
    X_missing = X.copy().astype(float)
    for col in range(n_cols):
        rate = col_missing_rates[col] / 100
        n_missing = int(rate * n_rows)
        if n_missing == 0:
            continue
        missing_rows = np.random.choice(n_rows, n_missing, replace=False)
        X_missing[missing_rows, col] = np.nan

    return X_missing


# def generate_column_missing_rates(n_cols, overall_missing_rate, seed=0):
#     """
#     Generate heterogeneous missing percentages for columns summing to the overall rate.
    
#     Parameters:
#         n_cols (int): Number of columns.
#         overall_missing_rate (float): Total missing rate for the dataset (0–100).
#         seed (int): Random seed.

#     Returns:
#         List[float]: Per-column missing percentages.
#     """
#     np.random.seed(seed)

#     # Generate random weights for each column
#     raw = np.random.exponential(scale=1.0, size=n_cols)

#     # Normalize to sum to the overall rate
#     scaled = raw / raw.sum() * overall_missing_rate

#     return scaled.tolist()

# def missing_value_generator(X, missing_rate, seed):
#     row_num = X.shape[0]
#     column_num = X.shape[1]
#     missing_value_average_each_row = column_num * (missing_rate / 100)

#     np.random.seed(seed)
#     poisson_dist = poisson.rvs(mu=missing_value_average_each_row, size=row_num, random_state=seed)
#     poisson_dist = np.clip(poisson_dist, 0, X.shape[1] - 1)

#     column_idx = np.arange(column_num)
#     X_missing = X.copy().astype(float)
#     for i in range(row_num):
#         missing_idx = np.random.choice(column_idx, poisson_dist[i], replace=False)
#         for j in missing_idx:
#             X_missing[i, j] = np.nan

#     return X_missing

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def run_cross_val_imputation(X_full, X_missing, n_splits=30, seed=42, scheme='ascending'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses = []
    fold = 1

    # Create mask once: True where missing in X_missing
    missing_mask = np.isnan(X_missing)

    for train_idx, test_idx in kf.split(X_full):
        X_train_full, X_test_full = X_full[train_idx], X_full[test_idx]
        X_train_missing, X_test_missing = X_missing[train_idx], X_missing[test_idx]
        train_mask = missing_mask[train_idx]
        test_mask = missing_mask[test_idx]

        # Imputation model (replace with your own)
        imputer = ChainImputer(max_epochs=60, hidden_dim=128, imputation_scheme=scheme)
        # imputer = KNNImputer(n_neighbors=5)
        imputer.fit_transform(X_train_missing)

        X_test_imputed = imputer.transform(X_test_missing)

        if np.isnan(X_test_imputed).any():
            print(f"[ERROR] NaNs found after imputation! Fold {fold}")
            print("Number of NaNs:", np.isnan(X_test_imputed).sum())
            nan_mask = np.isnan(X_test_imputed)
            print("NaN indices (first few):", np.argwhere(nan_mask)[:10])
            fallback = SimpleImputer(strategy='mean')
            X_test_imputed = fallback.fit_transform(X_test_imputed)
        
        # RMSE only on missing values in test set
        y_true = X_test_full[test_mask]
        y_pred = X_test_imputed[test_mask]

        fold_rmse = rmse(y_true, y_pred)
        print(f"Fold {fold} RMSE: {fold_rmse:.4f}")
        rmses.append(fold_rmse)
        fold += 1

    print(f"Average RMSE over {n_splits} folds: {np.mean(rmses):.4f}")
    return rmses

# if __name__ == "__main__":
#     # Load dataset from CSV
#     df = pd.read_csv("preprocessing/1_breast.csv")  # Put your filename here
#     X_full = df.values  # Assuming all numeric data; otherwise, preprocess

#     # Create missing values once
#     missing_rate = 20  # percent
#     seed = 42
#     X_missing = missing_value_generator(X_full, missing_rate, seed)

#     for scheme in {'ascending', 'random', 'descending'}:

#         run_cross_val_imputation(X_full, X_missing, n_splits=30, seed=seed)




# Dummy function placeholders
# def missing_value_generator(X, missing_rate, seed): ...
# def run_cross_val_imputation(X_full, X_missing, n_splits=30, seed=42, scheme='ascending'): ...

def format_float(x):
    return f"{x:.4f}"

if __name__ == "__main__":
    dataset_dir = "other"
    output_csv = "imputation_results_correct_scaled.csv"
    missing_rate = 80  # percent
    seed = 42
    n_splits = 10

    result_rows = []

    for filepath in glob(os.path.join(dataset_dir, "*.csv")):
        print(filepath)
        df = pd.read_csv(filepath)
        X_full = df.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_full)

        X_missing = missing_value_generator(X_scaled, missing_rate, seed)
        missing_percents = np.mean(np.isnan(X_missing), axis=0) * 100
        print("Missing value percentage per column:")
        print(missing_percents)


        scores_dict = {}
        for scheme in ['ascending', 'random', 'descending']:
            scores = run_cross_val_imputation(X_scaled, X_missing, n_splits=n_splits, seed=seed, scheme=scheme)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            scores_dict[scheme] = (mean_score, std_score)

        # Baseline values
        baseline_mean, baseline_std = scores_dict['ascending']

        # Store results
        for scheme in ['ascending', 'random', 'descending']:
            mean_val, std_val = scores_dict[scheme]
            gap = mean_val - baseline_mean if scheme != 'ascending' else 0.0
            percent_gap = (gap / baseline_mean * 100) if scheme != 'ascending' else 0.0

            result_rows.append({
                'filename': os.path.basename(filepath),
                'scheme': scheme,
                'mean': format_float(mean_val),
                'std': format_float(std_val),
                'gap_from_ascending': format_float(gap),
                'percent_gap_from_ascending': format_float(percent_gap)
            })
        results_df = pd.DataFrame(result_rows)
        results_df.to_csv(output_csv, header=True, index=False)

    # Write results to CSV
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(f"all_{output_csv}", index=False)