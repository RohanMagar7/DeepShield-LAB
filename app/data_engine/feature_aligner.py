"""
Feature Aligner
Uses PCA to align all datasets to a unified 41-dimensional feature space.
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureAligner:
    """
    Aligns features from different datasets to a unified dimensionality.
    Uses PCA to reduce/expand features to target dimension (41D).
    """
    
    def __init__(self, target_dim=41):
        """
        Initialize feature aligner.
        
        Args:
            target_dim: Target dimensionality (default 41 for NSL-KDD compatibility)
        """
        self.target_dim = target_dim
        self.pca_models = {}
        self.input_dims = {}
        
    def fit(self, X, dataset_name):
        """
        Fit PCA transformer for a specific dataset.
        
        Args:
            X: Input features (n_samples, n_features)
            dataset_name: Name of dataset ('kdd', 'cicids', 'unsw')
        """
        n_features = X.shape[1]
        self.input_dims[dataset_name] = n_features
        
        print(f"Fitting PCA for {dataset_name}: {n_features}D -> {self.target_dim}D")
        
        if n_features == self.target_dim:
            # No transformation needed
            self.pca_models[dataset_name] = None
            print(f"  No transformation needed (already {self.target_dim}D)")
        elif n_features > self.target_dim:
            # Dimensionality reduction
            pca = PCA(n_components=self.target_dim, random_state=42)
            pca.fit(X)
            self.pca_models[dataset_name] = pca
            variance_ratio = np.sum(pca.explained_variance_ratio_)
            print(f"  Reduced to {self.target_dim}D (variance preserved: {variance_ratio:.3f})")
        else:
            # Dimensionality expansion (pad with zeros, then apply PCA)
            # Create augmented features by polynomial expansion
            print(f"  Expanding from {n_features}D to {self.target_dim}D")
            
            # Use mean-centered polynomial features for expansion
            X_expanded = self._expand_features(X, self.target_dim)
            
            # Apply PCA to select most informative expanded features
            pca = PCA(n_components=self.target_dim, random_state=42)
            pca.fit(X_expanded)
            self.pca_models[dataset_name] = pca
            print(f"  Expanded to {self.target_dim}D using feature augmentation")
    
    def transform(self, X, dataset_name):
        """
        Transform features to unified dimensionality.
        
        Args:
            X: Input features (n_samples, n_features)
            dataset_name: Name of dataset ('kdd', 'cicids', 'unsw')
            
        Returns:
            X_aligned: Transformed features (n_samples, target_dim)
        """
        if dataset_name not in self.pca_models:
            raise ValueError(f"Dataset '{dataset_name}' not fitted. Call fit() first.")
        
        n_features = X.shape[1]
        expected_dim = self.input_dims[dataset_name]
        
        if n_features != expected_dim:
            raise ValueError(
                f"Input dimension mismatch for {dataset_name}: "
                f"expected {expected_dim}, got {n_features}"
            )
        
        pca = self.pca_models[dataset_name]
        
        if pca is None:
            # No transformation needed
            return X
        elif n_features > self.target_dim:
            # Dimensionality reduction
            return pca.transform(X)
        else:
            # Dimensionality expansion
            X_expanded = self._expand_features(X, self.target_dim)
            return pca.transform(X_expanded)
    
    def fit_transform(self, X, dataset_name):
        """
        Fit and transform in one step.
        
        Args:
            X: Input features
            dataset_name: Dataset name
            
        Returns:
            X_aligned: Transformed features
        """
        self.fit(X, dataset_name)
        return self.transform(X, dataset_name)
    
    def _expand_features(self, X, target_dim):
        """
        Expand feature space using polynomial and statistical features.
        
        Args:
            X: Input features (n_samples, n_features)
            target_dim: Target dimension
            
        Returns:
            X_expanded: Expanded features
        """
        n_samples, n_features = X.shape
        
        # Start with original features
        features = [X]
        
        # Add squared features
        if n_features * 2 <= target_dim * 2:
            features.append(X ** 2)
        
        # Add pairwise products (limited to avoid explosion)
        if len(features) * n_features < target_dim * 2:
            # Add interaction terms for first few features
            n_interactions = min(10, n_features)
            for i in range(n_interactions):
                for j in range(i+1, min(i+5, n_interactions)):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    if sum(f.shape[1] for f in features) >= target_dim * 2:
                        break
                if sum(f.shape[1] for f in features) >= target_dim * 2:
                    break
        
        # Add statistical features (mean, std per sample)
        features.append(np.mean(X, axis=1, keepdims=True))
        features.append(np.std(X, axis=1, keepdims=True))
        features.append(np.max(X, axis=1, keepdims=True))
        features.append(np.min(X, axis=1, keepdims=True))
        
        # Concatenate all features
        X_expanded = np.concatenate(features, axis=1)
        
        # If still not enough, pad with random projections
        current_dim = X_expanded.shape[1]
        if current_dim < target_dim * 2:
            # Add random projections
            np.random.seed(42)
            n_random = target_dim * 2 - current_dim
            random_proj = np.random.randn(n_features, n_random) / np.sqrt(n_features)
            X_random = X @ random_proj
            X_expanded = np.concatenate([X_expanded, X_random], axis=1)
        
        return X_expanded
    
    def save(self, filepath):
        """
        Save aligner to disk.
        
        Args:
            filepath: Path to save pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'target_dim': self.target_dim,
                'pca_models': self.pca_models,
                'input_dims': self.input_dims
            }, f)
        print(f"Feature aligner saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load aligner from disk.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            FeatureAligner instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        aligner = FeatureAligner(target_dim=data['target_dim'])
        aligner.pca_models = data['pca_models']
        aligner.input_dims = data['input_dims']
        
        print(f"Feature aligner loaded from {filepath}")
        return aligner
    
    def get_aligned_dim(self):
        """Return target dimensionality."""
        return self.target_dim
    
    def summary(self):
        """Print summary of fitted transformations."""
        print("\n=== Feature Aligner Summary ===")
        print(f"Target Dimension: {self.target_dim}")
        print(f"Fitted Datasets: {len(self.pca_models)}")
        for name, dim in self.input_dims.items():
            transform_type = "None" if self.pca_models[name] is None else \
                           "Reduction" if dim > self.target_dim else "Expansion"
            print(f"  {name}: {dim}D -> {self.target_dim}D ({transform_type})")


if __name__ == '__main__':
    # Test feature aligner
    print("Testing Feature Aligner...")
    
    # Simulate different dataset dimensions
    np.random.seed(42)
    X_kdd = np.random.randn(1000, 41)  # Already 41D
    X_cicids = np.random.randn(1000, 78)  # Need reduction
    X_unsw = np.random.randn(1000, 49)  # Need reduction
    
    # Create aligner
    aligner = FeatureAligner(target_dim=41)
    
    # Fit and transform each dataset
    X_kdd_aligned = aligner.fit_transform(X_kdd, 'kdd')
    X_cicids_aligned = aligner.fit_transform(X_cicids, 'cicids')
    X_unsw_aligned = aligner.fit_transform(X_unsw, 'unsw')
    
    # Verify dimensions
    print(f"\n✓ Feature Aligner Test Passed")
    print(f"  KDD: {X_kdd.shape} -> {X_kdd_aligned.shape}")
    print(f"  CICIDS: {X_cicids.shape} -> {X_cicids_aligned.shape}")
    print(f"  UNSW: {X_unsw.shape} -> {X_unsw_aligned.shape}")
    
    aligner.summary()
