"""
Test Feature Alignment
Verifies that all datasets (NSL-KDD, CICIDS2017, UNSW-NB15) are correctly aligned to 41 dimensions.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data_engine.kdd_loader import KDDLoader
from app.data_engine.cicids_loader import CICIDSLoader
from app.data_engine.unsw_loader import UNSWLoader
from app.data_engine.feature_aligner import FeatureAligner


def test_individual_loaders():
    """Test individual dataset loaders."""
    print("\n" + "="*70)
    print("TEST 1: Individual Dataset Loaders")
    print("="*70)
    
    loaders = {
        'NSL-KDD': KDDLoader(),
        'CICIDS2017': CICIDSLoader(),
        'UNSW-NB15': UNSWLoader()
    }
    
    results = {}
    
    for name, loader in loaders.items():
        print(f"\n[{name}]")
        try:
            if name == 'CICIDS2017':
                X_train, y_train, X_test, y_test = loader.load_data(train_split=0.8)
            else:
                X_train, y_train, X_test, y_test = loader.load_data()
            
            print(f"  ✓ Loaded successfully")
            print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"    Features: {X_train.shape[1]}")
            print(f"    Normal: {np.sum(y_train==0)}, Attack: {np.sum(y_train==1)}")
            
            # Verify data quality
            assert not np.isnan(X_train).any(), "Contains NaN"
            assert not np.isinf(X_train).any(), "Contains Inf"
            assert len(X_train) > 0, "Empty dataset"
            
            results[name] = {
                'status': 'PASS',
                'train_shape': X_train.shape,
                'test_shape': X_test.shape
            }
            
        except FileNotFoundError:
            print(f"  ⚠ Data files not found (skipped)")
            results[name] = {'status': 'SKIP', 'reason': 'Data not found'}
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = {'status': 'FAIL', 'error': str(e)}
    
    return results


def test_feature_alignment():
    """Test feature alignment to 41D."""
    print("\n" + "="*70)
    print("TEST 2: Feature Alignment to 41D")
    print("="*70)
    
    # Create synthetic datasets with different dimensions
    np.random.seed(42)
    
    datasets = {
        'kdd': np.random.randn(100, 41),
        'cicids': np.random.randn(100, 78),
        'unsw': np.random.randn(100, 49)
    }
    
    print(f"\nOriginal dimensions:")
    for name, X in datasets.items():
        print(f"  {name}: {X.shape}")
    
    # Create feature aligner
    aligner = FeatureAligner(target_dim=41)
    
    # Fit and transform each dataset
    aligned_datasets = {}
    
    for name, X in datasets.items():
        print(f"\n[{name.upper()}]")
        X_aligned = aligner.fit_transform(X, name)
        aligned_datasets[name] = X_aligned
        
        # Verify alignment
        assert X_aligned.shape[1] == 41, f"Expected 41D, got {X_aligned.shape[1]}D"
        assert not np.isnan(X_aligned).any(), "Contains NaN after alignment"
        assert not np.isinf(X_aligned).any(), "Contains Inf after alignment"
        
        print(f"  ✓ Aligned: {X.shape} -> {X_aligned.shape}")
    
    # Verify all datasets are 41D
    print(f"\n{'='*70}")
    print("Verification:")
    all_pass = True
    for name, X_aligned in aligned_datasets.items():
        is_41d = X_aligned.shape[1] == 41
        status = "✓ PASS" if is_41d else "✗ FAIL"
        print(f"  {name}: {X_aligned.shape[1]}D - {status}")
        all_pass = all_pass and is_41d
    
    return all_pass


def test_unified_dataset():
    """Test unified dataset with real data."""
    print("\n" + "="*70)
    print("TEST 3: Unified Dataset Integration")
    print("="*70)
    
    from app.data_engine.unified_dataset import UnifiedDataset
    
    try:
        dataset = UnifiedDataset(target_dim=41)
        X_train, y_train, X_test, y_test = dataset.load_all()
        
        # Verify dimensions
        assert X_train.shape[1] == 41, f"Expected 41D, got {X_train.shape[1]}D"
        assert X_test.shape[1] == 41, f"Expected 41D, got {X_test.shape[1]}D"
        
        # Verify data quality
        assert not np.isnan(X_train).any(), "Train data contains NaN"
        assert not np.isinf(X_train).any(), "Train data contains Inf"
        assert not np.isnan(X_test).any(), "Test data contains NaN"
        assert not np.isinf(X_test).any(), "Test data contains Inf"
        
        print(f"\n✓ Unified Dataset Test PASSED")
        print(f"  Train: {X_train.shape}")
        print(f"  Test:  {X_test.shape}")
        print(f"  All features aligned to 41D")
        
        return True
        
    except RuntimeError as e:
        print(f"\n⚠ Test skipped: {e}")
        print("  Place dataset files in data/ directory to test")
        return None
    
    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_input_compatibility():
    """Test that aligned features are compatible with CNN-LSTM model."""
    print("\n" + "="*70)
    print("TEST 4: Model Input Compatibility")
    print("="*70)
    
    from app.models.cnn_lstm import CNNLSTM
    
    # Create model
    model = CNNLSTM(input_dim=41)
    model.build_model()
    
    # Test with synthetic data
    np.random.seed(42)
    X_test = np.random.randn(10, 41)
    
    # Reshape for model input
    X_reshaped = X_test.reshape(-1, 1, 41)
    
    # Test prediction
    predictions = model.get_model().predict(X_reshaped, verbose=0)
    
    assert predictions.shape == (10, 1), f"Unexpected output shape: {predictions.shape}"
    assert not np.isnan(predictions).any(), "Predictions contain NaN"
    
    print(f"\n✓ Model Compatibility Test PASSED")
    print(f"  Input shape: (batch, 1, 41)")
    print(f"  Output shape: (batch, 1)")
    print(f"  Sample predictions: {predictions[:3].flatten()}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "FEATURE ALIGNMENT TESTS")
    print(" "*15 + "DeepShield-LAB Test Suite")
    print("="*70)
    
    test_results = []
    
    # Test 1: Individual loaders
    loader_results = test_individual_loaders()
    test_results.append(('Individual Loaders', loader_results))
    
    # Test 2: Feature alignment
    alignment_pass = test_feature_alignment()
    test_results.append(('Feature Alignment', 'PASS' if alignment_pass else 'FAIL'))
    
    # Test 3: Unified dataset
    unified_result = test_unified_dataset()
    if unified_result is None:
        test_results.append(('Unified Dataset', 'SKIP'))
    else:
        test_results.append(('Unified Dataset', 'PASS' if unified_result else 'FAIL'))
    
    # Test 4: Model compatibility
    try:
        model_compat = test_model_input_compatibility()
        test_results.append(('Model Compatibility', 'PASS' if model_compat else 'FAIL'))
    except Exception as e:
        print(f"\n✗ Model Compatibility Test FAILED: {e}")
        test_results.append(('Model Compatibility', 'FAIL'))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in test_results:
        if isinstance(result, dict):
            # Loader results
            for loader, status in result.items():
                if isinstance(status, dict):
                    print(f"  {loader:20s} - {status['status']}")
                else:
                    print(f"  {loader:20s} - {status}")
        else:
            print(f"  {test_name:20s} - {result}")
    
    print("="*70 + "\n")
    
    # Overall result
    all_pass = all(
        r == 'PASS' or r == 'SKIP' or (isinstance(r, dict) and 
        all(s.get('status', 'FAIL') in ['PASS', 'SKIP'] for s in r.values()))
        for _, r in test_results
    )
    
    if all_pass:
        print("✓ All tests PASSED (or skipped)")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
