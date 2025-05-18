import numpy as np
import torch
import torch.nn as nn
import random

class ChainImputer:
    def __init__(self, max_epochs=10, hidden_dim=64, lr=0.001, imputation_scheme='ascending',target_column=None):
        self.max_epochs = max_epochs
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.target_column = target_column
        self.imputation_scheme = imputation_scheme
        self.models = {}
        self.feature_sets = {}
        self.imputation_order = []
        self.prefilled_values = {}
        self.losses = {}

    def fit_transform(self, data):
        random.seed(42)
        data = np.array(data)
        n_samples, n_features = data.shape
        F = list(range(n_features))
        if self.imputation_scheme == 'ascending':
            F = sorted(F, key=lambda f: np.isnan(data[:, f]).sum())
        elif self.imputation_scheme == 'descending':
            F = sorted(F, key=lambda f: np.isnan(data[:, f]).sum(), reverse=True)
        elif self.imputation_scheme == 'random':
            random.shuffle(F)

        if self.target_column is not None:
            F.remove(self.target_column)
        S = []

        # Step 1: Add complete columns to S
        for f in F:
            if not np.isnan(data[:, f]).any():
                S.append(f)
                self.prefilled_values[f] = None  # Track prefilled
        # Step 2: If S is empty, mean impute one random feature
        if len(S) == 0:
            f = np.random.choice(F)
            col = data[:, f]
            mean_val = np.nanmean(col)
            col[np.isnan(col)] = mean_val
            data[:, f] = col
            S.append(f)
            F.remove(f)
            self.prefilled_values[f] = mean_val  # Track fallback impute

        # Step 3: Train models for other features
        
        for f in F:
            if f in S:
                continue
            target_col = data[:, f]
            mask = ~np.isnan(target_col)
            if mask.sum() == 0:
                continue  # skip if no observed data

            input_cols = data[:, S][mask]
            output = target_col[mask]

            model = nn.Sequential(
                nn.Linear(len(S), self.hidden_dim),
                nn.Sigmoid(),
                nn.Linear(self.hidden_dim, 1)
            )

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

            input_tensor = torch.tensor(input_cols, dtype=torch.float32)
            output_tensor = torch.tensor(output, dtype=torch.float32).unsqueeze(1)
            
            loss_store = []
            for epoch in range(self.max_epochs):
                model.train()
                optimizer.zero_grad()
                preds = model(input_tensor)
                loss = criterion(preds, output_tensor)
                loss_store.append(loss.detach())
                loss.backward()
                optimizer.step()
        
            self.losses[f] = loss_store
            self.models[f] = model
            self.feature_sets[f] = list(S)
            self.imputation_order.append(f)

            nan_mask = np.isnan(target_col)
            if nan_mask.sum() > 0:
                input_missing = data[nan_mask][:, S]
                input_missing_tensor = torch.tensor(input_missing, dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    preds = model(input_missing_tensor).squeeze().numpy()
                data[nan_mask, f] = preds

            S.append(f)

        return data

    def transform(self, data):
        data = np.array(data)
        data_imputed = data.copy()

        # Fill in the columns that were prefilled during fit_transform
        for f, mean_val in self.prefilled_values.items():
            if mean_val is not None:
                col = data_imputed[:, f]
                col[np.isnan(col)] = mean_val
                data_imputed[:, f] = col

        # Apply learned models in the same order
        for f in self.imputation_order:
            model = self.models[f]
            S = self.feature_sets[f]
            target_col = data_imputed[:, f]
            nan_mask = np.isnan(target_col)
            if nan_mask.sum() > 0:
                input_missing = data_imputed[nan_mask][:, S]
                input_missing_tensor = torch.tensor(input_missing, dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    preds = model(input_missing_tensor).squeeze().numpy()
                data_imputed[nan_mask, f] = preds

        return data_imputed