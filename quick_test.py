#!/usr/bin/env python3
"""
Quick test script - verifies all components work without full training
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

print("=" * 70)
print("DeepShield-LAB Quick Test")
print("=" * 70)

# Test 1: Data loaders
print("\n[1/5] Testing Data Loaders...")
try:
    from app.data_engine.kdd_loader import KDDLoader
    from app.data_engine.unsw_loader import UNSWLoader
    
    kdd = KDDLoader()
    X_kdd_train, y_kdd_train, X_kdd_test, y_kdd_test = kdd.load_data()
    print(f"  ✓ KDD: {X_kdd_train.shape}")
    
    unsw = UNSWLoader()
    X_unsw_train, y_unsw_train, X_unsw_test, y_unsw_test = unsw.load_data()
    print(f"  ✓ UNSW: {X_unsw_train.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Feature alignment
print("\n[2/5] Testing Feature Alignment...")
try:
    from app.data_engine.feature_aligner import FeatureAligner
    
    aligner = FeatureAligner(target_dim=41)
    
    X_kdd_aligned = aligner.fit_transform(X_kdd_train, 'kdd')
    X_unsw_aligned = aligner.fit_transform(X_unsw_train, 'unsw')
    
    assert X_kdd_aligned.shape[1] == 41, f"KDD not 41D: {X_kdd_aligned.shape[1]}"
    assert X_unsw_aligned.shape[1] == 41, f"UNSW not 41D: {X_unsw_aligned.shape[1]}"
    
    print(f"  ✓ KDD aligned to 41D")
    print(f"  ✓ UNSW aligned to 41D")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 3: CNN-LSTM model
print("\n[3/5] Testing CNN-LSTM Model...")
try:
    from app.models.cnn_lstm import CNNLSTM
    
    model = CNNLSTM(input_dim=41)
    model.build_model()
    
    # Test prediction
    X_test = np.random.randn(10, 41)
    preds, confs = model.predict(X_test)
    
    assert len(preds) == 10, f"Wrong prediction count: {len(preds)}"
    assert all(0 <= c <= 1 for c in confs), "Confidence out of range"
    
    print(f"  ✓ Model built: {model.model.count_params():,} params")
    print(f"  ✓ Predictions work (sample: {preds[:3]})")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 4: DQN agent
print("\n[4/5] Testing DQN Agent...")
try:
    from app.rl.dqn_agent import DQNAgent
    
    agent = DQNAgent(state_size=42, action_size=3)
    
    # Test state construction
    features = np.random.randn(41)
    confidence = 0.85
    state = agent.get_state(features, confidence)
    
    assert state.shape == (42,), f"Wrong state shape: {state.shape}"
    
    # Test action
    action = agent.act(state, training=False)
    assert action in [0, 1, 2], f"Invalid action: {action}"
    
    print(f"  ✓ DQN agent initialized")
    print(f"  ✓ State construction works (42D)")
    print(f"  ✓ Action selection works")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 5: Flow extractor
print("\n[5/5] Testing Flow Extractor...")
try:
    from realtime.flow_extractor import FlowExtractor
    from datetime import datetime
    
    extractor = FlowExtractor()
    
    # Add test packet
    packet = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.1',
        'src_port': 12345,
        'dst_port': 80,
        'protocol': 6,
        'length': 64,
        'flags': 0x02,
        'timestamp': datetime.now()
    }
    
    extractor.add_packet(packet)
    flow_id = extractor._get_flow_id(packet)
    features = extractor.extract_features(flow_id)
    
    assert features is not None, "Feature extraction failed"
    assert len(features) == 41, f"Wrong feature count: {len(features)}"
    
    print(f"  ✓ Flow extractor works")
    print(f"  ✓ Features extracted: 41D")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nNext steps:")
print("  1. Full training: python train/train_hybrid.py")
print("  2. Real-time:    sudo python realtime/sniffer.py")
print("=" * 70)
