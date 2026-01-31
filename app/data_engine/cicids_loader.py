"""
CICIDS2017 Dataset Loader
Handles loading, cleaning, and preprocessing of CICIDS2017 dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import glob


class CICIDSLoader:
    """Loader for CICIDS2017 dataset with preprocessing."""
    
    def __init__(self, data_dir='data/cicids2017'):
        """
        Initialize CICIDS loader.
        
        Args:
            data_dir: Directory containing CICIDS2017 CSV files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, train_split=0.8):
        """
        Load and preprocess CICIDS2017 dataset.
        
        Args:
            train_split: Ratio of data to use for training
            
        Returns:
            X_train, y_train, X_test, y_test: Preprocessed data
        """
        print("Loading CICIDS2017 dataset...")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load all CSV files
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                dfs.append(df)
                print(f"  Loaded: {os.path.basename(csv_file)} ({len(df)} rows)")
            except Exception as e:
                print(f"  Warning: Failed to load {csv_file}: {e}")
        
        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)
        print(f"Total samples: {len(df)}")
        
        # Clean column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Preprocess
        X, y = self._preprocess(df, fit=True)
        
        # Split into train/test
        split_idx = int(len(X) * train_split)
        indices = np.random.permutation(len(X))
        
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"Preprocessed shape: {X.shape}")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def _preprocess(self, df, fit=False):
        """
        Preprocess CICIDS data: clean, encode, normalize.
        
        Args:
            df: Raw dataframe
            fit: Whether to fit scalers
            
        Returns:
            X, y: Features and labels
        """
        df = df.copy()
        
        # Find label column (varies in naming)
        label_col = None
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Could not find label column in CICIDS data")
        
        # Extract labels and convert to binary
        labels = df[label_col].astype(str).str.strip().str.lower()
        y = np.array([0 if 'benign' in label or 'normal' in label else 1 for label in labels])
        
        # Drop label and non-feature columns
        drop_cols = [label_col]
        for col in df.columns:
            # Drop timestamp, IP, port columns
            if any(x in col.lower() for x in ['timestamp', 'time', 'ip', 'port', 'label']):
                drop_cols.append(col)
        
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        # Clean numeric data
        # Replace inf with nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with column median or 0
        df = df.fillna(0)
        
        # Convert to numpy array
        X = df.values.astype(np.float32)
        
        # Remove any remaining non-finite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_columns = df.columns.tolist()
        else:
            X = self.scaler.transform(X)
        
        print(f"  CICIDS preprocessed: {X.shape}, Normal: {np.sum(y==0)}, Attack: {np.sum(y==1)}")
        
        return X, y
    
    def get_feature_names(self):
        """Return feature column names."""
        return self.feature_columns if self.feature_columns else []


if __name__ == '__main__':
    # Test loader
    loader = CICIDSLoader()
    try:
        X_train, y_train, X_test, y_test = loader.load_data()
        print(f"\n✓ CICIDS Loader Test Passed")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Features: {len(loader.get_feature_names())}")
    except FileNotFoundError:
        print("⚠ CICIDS2017 data files not found. Place CSV files in data/cicids2017/")
    except Exception as e:
        print(f"✗ Error: {e}")
