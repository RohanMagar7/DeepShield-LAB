"""
NSL-KDD Dataset Loader
Handles loading, cleaning, and preprocessing of NSL-KDD dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class KDDLoader:
    """Loader for NSL-KDD dataset with preprocessing."""
    
    # NSL-KDD column names (41 features + 1 label + difficulty)
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # Categorical columns that need encoding
    CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
    
    def __init__(self, data_dir='data/nsl-kdd'):
        """
        Initialize KDD loader.
        
        Args:
            data_dir: Directory containing KDDTrain+.txt and KDDTest+.txt
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, train_file='KDDTrain+.txt', test_file='KDDTest+.txt'):
        """
        Load and preprocess NSL-KDD dataset.
        
        Args:
            train_file: Training data filename
            test_file: Testing data filename
            
        Returns:
            X_train, y_train, X_test, y_test: Preprocessed data
        """
        print("Loading NSL-KDD dataset...")
        
        # Load train and test files
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        train_df = pd.read_csv(train_path, names=self.COLUMN_NAMES, header=None)
        test_df = pd.read_csv(test_path, names=self.COLUMN_NAMES, header=None)
        
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # Preprocess
        X_train, y_train = self._preprocess(train_df, fit=True)
        X_test, y_test = self._preprocess(test_df, fit=False)
        
        print(f"Preprocessed shape: {X_train.shape}")
        return X_train, y_train, X_test, y_test
    
    def _preprocess(self, df, fit=False):
        """
        Preprocess KDD data: clean, encode, normalize.
        
        Args:
            df: Raw dataframe
            fit: Whether to fit encoders/scalers
            
        Returns:
            X, y: Features and labels
        """
        df = df.copy()
        
        # Remove difficulty column (not a feature)
        if 'difficulty' in df.columns:
            df = df.drop('difficulty', axis=1)
        
        # Extract labels and convert to binary (normal=0, attack=1)
        labels = df['label'].values
        y = np.array([0 if 'normal' in str(label).lower() else 1 for label in labels])
        
        # Drop label column
        df = df.drop('label', axis=1)
        
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
        
        # Clean numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace inf with nan, then fill with 0
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Convert to numpy array
        X = df.values.astype(np.float32)
        
        # Normalize
        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_columns = df.columns.tolist()
        else:
            X = self.scaler.transform(X)
        
        # Final check for any remaining NaN or Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  KDD preprocessed: {X.shape}, Normal: {np.sum(y==0)}, Attack: {np.sum(y==1)}")
        
        return X, y
    
    def get_feature_names(self):
        """Return feature column names."""
        return self.feature_columns if self.feature_columns else self.COLUMN_NAMES[:-2]


if __name__ == '__main__':
    # Test loader
    loader = KDDLoader()
    try:
        X_train, y_train, X_test, y_test = loader.load_data()
        print(f"\n✓ KDD Loader Test Passed")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Features: {len(loader.get_feature_names())}")
    except FileNotFoundError:
        print("⚠ NSL-KDD data files not found. Place files in data/nsl-kdd/")
    except Exception as e:
        print(f"✗ Error: {e}")
