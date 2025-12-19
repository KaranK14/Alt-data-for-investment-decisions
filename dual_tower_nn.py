"""
Dual-tower Neural Network model implementation - tabular+embeddings only.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Dict
import warnings

from ..config import (
    NN_PARAMS, N_ITERATIONS, OVERSAMPLE_RATIOS, RANDOM_SEED
)
from ..imbalance import oversample_minority


def convert_tabular_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to numeric types for neural network preprocessing.
    
    Handles:
    - Object dtype columns: converts to numeric using LabelEncoder or numeric conversion
    - Boolean columns: converts to int (0/1)
    - Missing values: fills with 0
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric directly first
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
            except:
                # If that fails, use LabelEncoder
                le = LabelEncoder()
                # Handle NaN by converting to string 'nan' first
                col_series = df_clean[col].astype(str)
                df_clean[col] = le.fit_transform(col_series)
        elif df_clean[col].dtype == 'bool':
            # Convert boolean to int
            df_clean[col] = df_clean[col].astype(int)
        elif pd.api.types.is_categorical_dtype(df_clean[col]):
            # Convert category codes to numeric
            df_clean[col] = df_clean[col].cat.codes
        else:
            # For numeric types, just fill NaN
            df_clean[col] = df_clean[col].fillna(0)
    
    # Final fillna for any remaining NaN
    df_clean = df_clean.fillna(0)
    
    # Ensure all columns are numeric
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    return df_clean


class DualTowerDataset(Dataset):
    """Dataset class for dual-tower NN."""
    
    def __init__(self, X_tabular: np.ndarray, X_embedding: np.ndarray, y):
        # Convert to numpy arrays if they're pandas Series/DataFrames
        if hasattr(X_tabular, 'values'):
            X_tabular = X_tabular.values
        if hasattr(X_embedding, 'values'):
            X_embedding = X_embedding.values
        if hasattr(y, 'values'):
            y = y.values
        elif hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        
        # Convert to numpy arrays if not already
        X_tabular = np.asarray(X_tabular, dtype=np.float32)
        X_embedding = np.asarray(X_embedding, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        self.X_tabular = torch.FloatTensor(X_tabular)
        self.X_embedding = torch.FloatTensor(X_embedding)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'tabular': self.X_tabular[idx],
            'embedding': self.X_embedding[idx],
            'label': self.y[idx]
        }


class DualTowerNN(nn.Module):
    """Dual-tower neural network architecture."""
    
    def __init__(
        self,
        tabular_dim: int,
        embedding_dim: int,
        tabular_tower: list = None,
        embedding_tower: list = None,
        combined_units: int = 128,
        classifier_units: int = 64,
        dropout_rate: float = 0.3
    ):
        super(DualTowerNN, self).__init__()
        
        if tabular_tower is None:
            tabular_tower = NN_PARAMS["tabular_tower"]
        if embedding_tower is None:
            embedding_tower = NN_PARAMS["embedding_tower"]
        
        # Tabular tower
        tab_layers = []
        input_dim = tabular_dim
        for out_dim in tabular_tower:
            tab_layers.extend([
                nn.Linear(input_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = out_dim
        
        # Embedding tower
        emb_layers = []
        input_dim = embedding_dim
        for out_dim in embedding_tower:
            emb_layers.extend([
                nn.Linear(input_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = out_dim
        
        self.tabular_tower = nn.Sequential(*tab_layers)
        self.embedding_tower = nn.Sequential(*emb_layers)
        
        # Combined classifier
        tower_output_dim = tabular_tower[-1] + embedding_tower[-1]
        self.classifier = nn.Sequential(
            nn.Linear(tower_output_dim, combined_units),
            nn.BatchNorm1d(combined_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_units, classifier_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_units, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tabular, embedding):
        tab_out = self.tabular_tower(tabular)
        emb_out = self.embedding_tower(embedding)
        combined = torch.cat([tab_out, emb_out], dim=1)
        output = self.classifier(combined)
        # Squeeze and ensure output is between 0 and 1 (for numerical stability)
        output = output.squeeze()
        # Clamp to [0, 1] to ensure valid probability range
        output = torch.clamp(output, 0.0, 1.0)
        return output


def train_evaluate_dual_tower_nn(
    X_train_tab: pd.DataFrame,
    X_train_emb: pd.DataFrame,
    y_train: np.ndarray,
    X_val_tab: pd.DataFrame,
    X_val_emb: pd.DataFrame,
    y_val: np.ndarray,
    X_test_tab: pd.DataFrame,
    X_test_emb: pd.DataFrame,
    y_test: np.ndarray,
    n_iterations: int = N_ITERATIONS,
    oversample_ratios: list = None,
    random_state: int = RANDOM_SEED,
    device: str = None
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate dual-tower neural network."""
    if oversample_ratios is None:
        oversample_ratios = OVERSAMPLE_RATIOS
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"Dual-Tower Neural Network Training")
    print(f"{'='*80}")
    print(f"  Device: {device}")
    print(f"  Iterations per ratio: {n_iterations}")
    print(f"  Oversample ratios: {oversample_ratios}")
    
    # Preprocessing: Convert tabular features to numeric first
    print(f"  Converting tabular features to numeric...")
    X_train_tab_numeric = convert_tabular_to_numeric(X_train_tab)
    X_val_tab_numeric = convert_tabular_to_numeric(X_val_tab)
    X_test_tab_numeric = convert_tabular_to_numeric(X_test_tab)
    
    # Scale tabular features (MinMaxScaler)
    scaler_tab = MinMaxScaler()
    X_train_tab_scaled = scaler_tab.fit_transform(X_train_tab_numeric)
    X_val_tab_scaled = scaler_tab.transform(X_val_tab_numeric)
    X_test_tab_scaled = scaler_tab.transform(X_test_tab_numeric)
    
    scaler_emb = StandardScaler()
    X_train_emb_scaled = scaler_emb.fit_transform(X_train_emb)
    X_val_emb_scaled = scaler_emb.transform(X_val_emb)
    X_test_emb_scaled = scaler_emb.transform(X_test_emb)
    
    results = {}
    
    for ratio in oversample_ratios:
        print(f"\n  Oversample ratio: {ratio}")
        print(f"  {'-'*78}")
        
        metrics_list = []
        
        for iteration in range(n_iterations):
            # Oversample
            X_train_combined = np.hstack([X_train_tab_scaled, X_train_emb_scaled])
            X_train_combined_df = pd.DataFrame(
                X_train_combined,
                columns=list(X_train_tab.columns) + list(X_train_emb.columns)
            )
            
            X_train_oversampled, y_train_oversampled = oversample_minority(
                X_train_combined_df, y_train,
                target_ratio=ratio,
                random_state=random_state + iteration
            )
            
            n_tab = X_train_tab_scaled.shape[1]
            X_train_tab_os = X_train_oversampled.iloc[:, :n_tab].values
            X_train_emb_os = X_train_oversampled.iloc[:, n_tab:].values
            
            # Create datasets
            train_dataset = DualTowerDataset(X_train_tab_os, X_train_emb_os, y_train_oversampled)
            val_dataset = DualTowerDataset(X_val_tab_scaled, X_val_emb_scaled, y_val)
            test_dataset = DualTowerDataset(X_test_tab_scaled, X_test_emb_scaled, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=NN_PARAMS["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=NN_PARAMS["batch_size"], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=NN_PARAMS["batch_size"], shuffle=False)
            
            # Initialize model
            model = DualTowerNN(
                tabular_dim=X_train_tab_scaled.shape[1],
                embedding_dim=X_train_emb_scaled.shape[1]
            ).to(device)
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=NN_PARAMS["learning_rate"],
                weight_decay=NN_PARAMS["weight_decay"]
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )
            
            # Training loop
            best_val_auc = 0
            patience_counter = 0
            best_threshold = 0.5
            
            for epoch in range(NN_PARAMS["epochs"]):
                # Train
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    tab = batch['tabular'].to(device)
                    emb = batch['embedding'].to(device)
                    label = batch['label'].to(device)
                    
                    output = model(tab, emb)
                    # Ensure output and label have same shape
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                
                # Validate
                model.eval()
                val_proba = []
                val_labels = []
                with torch.no_grad():
                    for batch in val_loader:
                        tab = batch['tabular'].to(device)
                        emb = batch['embedding'].to(device)
                        label = batch['label'].to(device)
                        
                        output = model(tab, emb)
                        # Ensure output is 1D array for numpy conversion
                        if isinstance(output, torch.Tensor):
                            output_np = output.cpu().numpy()
                            if output_np.ndim == 0:
                                output_np = np.array([output_np])
                            val_proba.append(output_np.flatten())
                        else:
                            val_proba.append(np.array([output]).flatten())
                        val_labels.append(label.cpu().numpy().flatten())
                
                val_proba = np.concatenate(val_proba)
                val_labels = np.concatenate(val_labels)
                val_auc = roc_auc_score(val_labels, val_proba)
                
                scheduler.step(val_auc)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # Find best threshold
                    thresholds = np.arange(0.1, 1.0, 0.05)
                    best_f1 = 0
                    for thresh in thresholds:
                        pred = (val_proba >= thresh).astype(int)
                        f1 = f1_score(val_labels, pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = thresh
                else:
                    patience_counter += 1
                    if patience_counter >= NN_PARAMS["early_stopping_patience"]:
                        break
            
            # Evaluate on test set
            model.eval()
            test_proba = []
            test_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    tab = batch['tabular'].to(device)
                    emb = batch['embedding'].to(device)
                    label = batch['label'].to(device)
                    
                    output = model(tab, emb)
                    # Ensure output is 1D array for numpy conversion
                    if isinstance(output, torch.Tensor):
                        output_np = output.cpu().numpy()
                        if output_np.ndim == 0:
                            output_np = np.array([output_np])
                        test_proba.append(output_np.flatten())
                    else:
                        test_proba.append(np.array([output]).flatten())
                    test_labels.append(label.cpu().numpy().flatten())
            
            test_proba = np.concatenate(test_proba)
            test_labels = np.concatenate(test_labels)
            test_pred = (test_proba >= best_threshold).astype(int)
            
            metrics = {
                "accuracy": accuracy_score(test_labels, test_pred),
                "precision": precision_score(test_labels, test_pred, zero_division=0),
                "recall": recall_score(test_labels, test_pred, zero_division=0),
                "f1": f1_score(test_labels, test_pred, zero_division=0),
                "roc_auc": roc_auc_score(test_labels, test_proba),
            }
            
            metrics_list.append(metrics)
            
            if (iteration + 1) % 5 == 0:
                print(f"    Completed iteration {iteration + 1}/{n_iterations}")
        
        metrics_df = pd.DataFrame(metrics_list)
        results[f"oversample_{ratio}"] = {
            "accuracy_mean": metrics_df["accuracy"].mean(),
            "accuracy_std": metrics_df["accuracy"].std(),
            "precision_mean": metrics_df["precision"].mean(),
            "precision_std": metrics_df["precision"].std(),
            "recall_mean": metrics_df["recall"].mean(),
            "recall_std": metrics_df["recall"].std(),
            "f1_mean": metrics_df["f1"].mean(),
            "f1_std": metrics_df["f1"].std(),
            "roc_auc_mean": metrics_df["roc_auc"].mean(),
            "roc_auc_std": metrics_df["roc_auc"].std(),
        }
        
        print(f"\n    Test Set Results (mean ± std):")
        print(f"      Accuracy:  {results[f'oversample_{ratio}']['accuracy_mean']:.4f} ± {results[f'oversample_{ratio}']['accuracy_std']:.4f}")
        print(f"      Precision: {results[f'oversample_{ratio}']['precision_mean']:.4f} ± {results[f'oversample_{ratio}']['precision_std']:.4f}")
        print(f"      Recall:    {results[f'oversample_{ratio}']['recall_mean']:.4f} ± {results[f'oversample_{ratio}']['recall_std']:.4f}")
        print(f"      F1:        {results[f'oversample_{ratio}']['f1_mean']:.4f} ± {results[f'oversample_{ratio}']['f1_std']:.4f}")
        print(f"      ROC-AUC:   {results[f'oversample_{ratio}']['roc_auc_mean']:.4f} ± {results[f'oversample_{ratio}']['roc_auc_std']:.4f}")
    
    return results

