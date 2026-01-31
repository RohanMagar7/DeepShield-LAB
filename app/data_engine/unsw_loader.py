"""
UNSW-NB15 Dataset Loader
Handles loading, cleaning, and preprocessing of UNSW-NB15 dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class UNSWLoader:
    """Loader for UNSW-NB15 dataset with preprocessing."""
    
    # Categorical columns in UNSW-NB15
    CATEGORICAL_COLS = ['proto', 'service', 'state']
    
    def __init__(self, data_dir='data/unsw-nb15'):
        """
        Initialize UNSW loader.
        
        Args:
            data_dir: Directory containing UNSW-NB15 CSV files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, train_file='UNSW_NB15_training-set', 
                  test_file='UNSW_NB15_testing-set'):
        """
        Load and preprocess UNSW-NB15 dataset.
        
        Args:
            train_file: Training data filename (without extension)
            test_file: Testing data filename (without extension)
            
        Returns:
            X_train, y_train, X_test, y_test: Preprocessed data
        """
        print("Loading UNSW-NB15 dataset...")
        
        # Load train and test files (support both .csv and .parquet)
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        # Try loading as parquet first, then csv
        try:
            if os.path.exists(train_path + '.parquet'):
                train_df = pd.read_parquet(train_path + '.parquet')
            elif os.path.exists(train_path + '.csv'):
                train_df = pd.read_csv(train_path + '.csv', low_memory=False)
            else:
                raise FileNotFoundError(f"No UNSW training file found: {train_path}")
                
            if os.path.exists(test_path + '.parquet'):
                test_df = pd.read_parquet(test_path + '.parquet')
            elif os.path.exists(test_path + '.csv'):
                test_df = pd.read_csv(test_path + '.csv', low_memory=False)
            else:
                raise FileNotFoundError(f"No UNSW test file found: {test_path}")
        except ImportError:
            print("⚠ Parquet support not available, trying CSV...")
            train_df = pd.read_csv(train_path + '.csv', low_memory=False)
            test_df = pd.read_csv(test_path + '.csv', low_memory=False)
        
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # Preprocess
        X_train, y_train = self._preprocess(train_df, fit=True)
        X_test, y_test = self._preprocess(test_df, fit=False)
        
        print(f"Preprocessed shape: {X_train.shape}")
        return X_train, y_train, X_test, y_test
    
    def _preprocess(self, df, fit=False):
        """
        Preprocess UNSW data: clean, encode, normalize.
        
        Args:
            df: Raw dataframe
            fit: Whether to fit encoders/scalers
            
        Returns:
            X, y: Features and labels
        """
        df = df.copy()
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Find label column
        label_col = 'label' if 'label' in df.columns else None
        if label_col is None:
            # Try alternative names
            for col in df.columns:
                if 'attack' in col.lower() or 'class' in col.lower():
                    label_col = col
                    break
        
        if label_col is None:
            raise ValueError("Could not find label column in UNSW data")
        
        # Extract binary labels (0=normal, 1=attack)
        y = df[label_col].astype(int).values
        
        # Drop label and non-feature columns
        drop_cols = [label_col]
        for col in df.columns:
            # Drop ID, timestamp, IP address columns
            if any(x in col.lower() for x in ['id', 'time', 'srcip', 'dstip', 'attack_cat']):
                drop_cols.append(col)
        
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Handle categorical columns
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ else -1
                    )
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        # Clean numeric data
        df = df.replace([np.inf, -np.inf], np.nan)
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
        
        print(f"  UNSW preprocessed: {X.shape}, Normal: {np.sum(y==0)}, Attack: {np.sum(y==1)}")
        
        return X, y
    
    def get_feature_names(self):
        """Return feature column names."""
        return self.feature_columns if self.feature_columns else []


if __name__ == '__main__':
    # Test loader
    loader = UNSWLoader()
    try:
        X_train, y_train, X_test, y_test = loader.load_data()
        print(f"\n✓ UNSW Loader Test Passed")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Features: {len(loader.get_feature_names())}")
    except FileNotFoundError:
        print("⚠ UNSW-NB15 data files not found. Place files in data/unsw-nb15/")
    except Exception as e:
        print(f"✗ Error: {e}")
