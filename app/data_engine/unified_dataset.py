"""
Unified Dataset
Combines multiple IDS datasets into a single unified dataset with aligned features.
"""

import numpy as np
from app.data_engine.kdd_loader import KDDLoader
from app.data_engine.cicids_loader import CICIDSLoader
from app.data_engine.unsw_loader import UNSWLoader
from app.data_engine.feature_aligner import FeatureAligner


class UnifiedDataset:
    """
    Unified dataset combining NSL-KDD, CICIDS2017, and UNSW-NB15.
    All features are aligned to 41 dimensions using PCA.
    """
    
    def __init__(self, data_dirs=None, target_dim=41):
        """
        Initialize unified dataset.
        
        Args:
            data_dirs: Dictionary with dataset paths
                      {'kdd': 'data/nsl-kdd', 'cicids': 'data/cicids2017', 'unsw': 'data/unsw-nb15'}
            target_dim: Target feature dimensionality (default 41)
        """
        if data_dirs is None:
            data_dirs = {
                'kdd': 'data/nsl-kdd',
                'cicids': 'data/cicids2017',
                'unsw': 'data/unsw-nb15'
            }
        
        self.data_dirs = data_dirs
        self.target_dim = target_dim
        self.aligner = FeatureAligner(target_dim=target_dim)
        
        self.loaders = {
            'kdd': KDDLoader(data_dirs['kdd']),
            'cicids': CICIDSLoader(data_dirs['cicids']),
            'unsw': UNSWLoader(data_dirs['unsw'])
        }
        
        self.datasets = {}
        
    def load_all(self, datasets=['kdd', 'cicids', 'unsw']):
        """
        Load and align all specified datasets.
        
        Args:
            datasets: List of dataset names to load
            
        Returns:
            X_train, y_train, X_test, y_test: Combined aligned data
        """
        print("\n" + "="*60)
        print("Loading and Aligning Multi-Dataset IDS Data")
        print("="*60)
        
        train_data = []
        test_data = []
        
        for dataset_name in datasets:
            if dataset_name not in self.loaders:
                print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
                continue
            
            try:
                print(f"\n[{dataset_name.upper()}]")
                X_train, y_train, X_test, y_test = self._load_and_align(dataset_name)
                
                train_data.append((X_train, y_train))
                test_data.append((X_test, y_test))
                
                self.datasets[dataset_name] = {
                    'train': (X_train, y_train),
                    'test': (X_test, y_test)
                }
                
            except FileNotFoundError as e:
                print(f"⚠ Skipping {dataset_name}: Data files not found")
            except Exception as e:
                print(f"✗ Error loading {dataset_name}: {e}")
        
        if not train_data:
            raise RuntimeError("No datasets were successfully loaded")
        
        # Combine all datasets
        print(f"\n{'='*60}")
        print("Combining Datasets")
        print("="*60)
        
        X_train = np.vstack([X for X, y in train_data])
        y_train = np.concatenate([y for X, y in train_data])
        
        X_test = np.vstack([X for X, y in test_data])
        y_test = np.concatenate([y for X, y in test_data])
        
        # Shuffle
        train_idx = np.random.permutation(len(X_train))
        test_idx = np.random.permutation(len(X_test))
        
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        X_test, y_test = X_test[test_idx], y_test[test_idx]
        
        print(f"\nFinal Combined Dataset:")
        print(f"  Train: {X_train.shape} (Normal: {np.sum(y_train==0)}, Attack: {np.sum(y_train==1)})")
        print(f"  Test:  {X_test.shape} (Normal: {np.sum(y_test==0)}, Attack: {np.sum(y_test==1)})")
        print(f"  Feature Dimension: {self.target_dim}D")
        print("="*60 + "\n")
        
        return X_train, y_train, X_test, y_test
    
    def _load_and_align(self, dataset_name):
        """
        Load and align a single dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            X_train, y_train, X_test, y_test: Aligned data
        """
        loader = self.loaders[dataset_name]
        
        # Load raw data
        if dataset_name == 'cicids':
            X_train, y_train, X_test, y_test = loader.load_data(train_split=0.8)
        else:
            X_train, y_train, X_test, y_test = loader.load_data()
        
        # Fit aligner on training data
        print(f"Aligning features to {self.target_dim}D...")
        self.aligner.fit(X_train, dataset_name)
        
        # Transform both train and test
        X_train_aligned = self.aligner.transform(X_train, dataset_name)
        X_test_aligned = self.aligner.transform(X_test, dataset_name)
        
        print(f"  Aligned: {X_train_aligned.shape}")
        
        return X_train_aligned, y_train, X_test_aligned, y_test
    
    def get_dataset(self, dataset_name):
        """
        Get a specific dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dict with 'train' and 'test' tuples
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded")
        return self.datasets[dataset_name]
    
    def save_aligner(self, filepath='models/feature_aligner.pkl'):
        """
        Save feature aligner to disk.
        
        Args:
            filepath: Path to save aligner
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.aligner.save(filepath)
    
    def load_aligner(self, filepath='models/feature_aligner.pkl'):
        """
        Load feature aligner from disk.
        
        Args:
            filepath: Path to aligner file
        """
        self.aligner = FeatureAligner.load(filepath)
    
    def get_aligner(self):
        """Return the feature aligner instance."""
        return self.aligner


if __name__ == '__main__':
    # Test unified dataset
    print("Testing Unified Dataset...")
    
    try:
        dataset = UnifiedDataset()
        X_train, y_train, X_test, y_test = dataset.load_all()
        
        print(f"\n✓ Unified Dataset Test Passed")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
    except RuntimeError as e:
        print(f"\n⚠ Test skipped: {e}")
        print("  Place dataset files in data/ directory to test")
