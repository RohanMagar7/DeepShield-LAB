"""
Real-Time Intrusion Detector
Applies trained CNN-LSTM and DQN models to detect network intrusions.
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.cnn_lstm import CNNLSTM
from app.rl.dqn_agent import DQNAgent
from app.data_engine.feature_aligner import FeatureAligner
from sklearn.preprocessing import StandardScaler


class IntrusionDetector:
    """
    Real-time intrusion detector using CNN-LSTM + DQN.
    """
    
    def __init__(self, 
                 model_path='models/deepshield_cnn_lstm.h5',
                 dqn_path='models/dqn_weights.h5',
                 aligner_path='models/feature_aligner.pkl',
                 dataset_name='kdd'):
        """
        Initialize detector.
        
        Args:
            model_path: Path to CNN-LSTM model
            dqn_path: Path to DQN weights
            aligner_path: Path to feature aligner
            dataset_name: Dataset type for feature alignment
        """
        self.model_path = model_path
        self.dqn_path = dqn_path
        self.aligner_path = aligner_path
        self.dataset_name = dataset_name
        
        self.cnn_lstm = None
        self.dqn_agent = None
        self.aligner = None
        self.scaler = StandardScaler()
        self.is_loaded = False
        
    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        
        # Load CNN-LSTM
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"CNN-LSTM model not found: {self.model_path}")
        
        self.cnn_lstm = CNNLSTM(input_dim=41)
        self.cnn_lstm.load(self.model_path)
        print(f"  ✓ CNN-LSTM loaded from {self.model_path}")
        
        # Load DQN
        if os.path.exists(self.dqn_path):
            self.dqn_agent = DQNAgent(state_size=42, action_size=3)
            self.dqn_agent.load(self.dqn_path)
            print(f"  ✓ DQN loaded from {self.dqn_path}")
        else:
            print(f"  ⚠ DQN weights not found, using default threshold")
            self.dqn_agent = None
        
        # Load feature aligner
        if os.path.exists(self.aligner_path):
            self.aligner = FeatureAligner.load(self.aligner_path)
            print(f"  ✓ Feature aligner loaded from {self.aligner_path}")
        else:
            print(f"  ⚠ Feature aligner not found, assuming 41D input")
            self.aligner = None
        
        self.is_loaded = True
        print("✓ All models loaded successfully\n")
    
    def preprocess_features(self, features):
        """
        Preprocess raw features for model input.
        
        Args:
            features: Raw feature vector
            
        Returns:
            Preprocessed features (41D)
        """
        # Ensure numpy array
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Clean features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        features = self.scaler.fit_transform(features)
        
        # Align features if needed
        if self.aligner is not None and features.shape[1] != 41:
            features = self.aligner.transform(features, self.dataset_name)
        
        # Ensure 41D
        if features.shape[1] != 41:
            # Pad or truncate
            if features.shape[1] < 41:
                padding = np.zeros((1, 41 - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :41]
        
        return features.flatten()
    
    def detect(self, features, use_dqn=True):
        """
        Detect intrusion from flow features.
        
        Args:
            features: Flow feature vector
            use_dqn: Whether to use DQN for threshold adjustment
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Preprocess features
        features = self.preprocess_features(features)
        
        # Get CNN-LSTM prediction
        features_reshaped = features.reshape(1, -1)
        predictions, confidences = self.cnn_lstm.predict(features_reshaped)
        
        prediction = predictions[0]
        confidence = confidences[0]
        
        # Apply DQN threshold adjustment if available
        if use_dqn and self.dqn_agent is not None:
            state = self.dqn_agent.get_state(features, confidence)
            action = self.dqn_agent.act(state, training=False)
            threshold = self.dqn_agent.adjust_threshold(action)
            
            # Re-classify with adjusted threshold
            prediction = 1 if confidence >= threshold else 0
        else:
            threshold = 0.5
        
        # Determine attack type (simplified)
        if prediction == 1:
            if confidence > 0.9:
                threat_level = "HIGH"
            elif confidence > 0.7:
                threat_level = "MEDIUM"
            else:
                threat_level = "LOW"
        else:
            threat_level = "NONE"
        
        return {
            'is_attack': bool(prediction),
            'confidence': float(confidence),
            'threshold': float(threshold),
            'threat_level': threat_level,
            'label': 'ATTACK' if prediction == 1 else 'NORMAL'
        }
    
    def detect_batch(self, features_batch, use_dqn=True):
        """
        Detect intrusions in batch.
        
        Args:
            features_batch: Array of feature vectors (n_samples, n_features)
            use_dqn: Whether to use DQN
            
        Returns:
            List of detection results
        """
        results = []
        for features in features_batch:
            result = self.detect(features, use_dqn=use_dqn)
            results.append(result)
        return results
    
    def get_statistics(self):
        """
        Get detector statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'model_loaded': self.is_loaded,
            'cnn_lstm_available': self.cnn_lstm is not None,
            'dqn_available': self.dqn_agent is not None,
            'aligner_available': self.aligner is not None,
        }
        
        if self.dqn_agent:
            stats['current_threshold'] = self.dqn_agent.get_threshold()
            stats['epsilon'] = self.dqn_agent.epsilon
        
        return stats


if __name__ == '__main__':
    # Test detector
    print("Testing Intrusion Detector...")
    
    try:
        # Initialize detector
        detector = IntrusionDetector()
        detector.load_models()
        
        # Test with synthetic features
        test_features = np.random.randn(41)
        result = detector.detect(test_features)
        
        print(f"\n✓ Detector Test Passed")
        print(f"  Detection result: {result['label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Threshold: {result['threshold']:.3f}")
        print(f"  Threat level: {result['threat_level']}")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Test skipped: {e}")
        print("  Train models first with: python train/train_hybrid.py")
