"""
Hybrid Training Pipeline
Trains CNN-LSTM model with DQN agent for adaptive intrusion detection.
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data_engine.unified_dataset import UnifiedDataset
from app.models.cnn_lstm import CNNLSTM
from app.rl.dqn_agent import DQNAgent
from sklearn.model_selection import train_test_split


class HybridTrainer:
    """
    Hybrid training pipeline combining CNN-LSTM and DQN.
    """
    
    def __init__(self, input_dim=41):
        """
        Initialize hybrid trainer.
        
        Args:
            input_dim: Feature dimensionality
        """
        self.input_dim = input_dim
        self.dataset = None
        self.cnn_lstm = None
        self.dqn_agent = None
        
    def load_data(self, datasets=['kdd', 'cicids', 'unsw']):
        """
        Load and prepare unified dataset.
        
        Args:
            datasets: List of datasets to load
            
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        print("\n" + "="*70)
        print("LOADING DATASETS")
        print("="*70)
        
        self.dataset = UnifiedDataset(target_dim=self.input_dim)
        X_train_full, y_train_full, X_test, y_test = self.dataset.load_all(datasets)
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full
        )
        
        print(f"\nData Split:")
        print(f"  Training:   {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Testing:    {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_cnn_lstm(self, X_train, y_train, X_val, y_val,
                       epochs=50, batch_size=128):
        """
        Train CNN-LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Trained model
        """
        print("\n" + "="*70)
        print("TRAINING CNN-LSTM MODEL")
        print("="*70)
        
        self.cnn_lstm = CNNLSTM(input_dim=self.input_dim)
        self.cnn_lstm.build_model()
        
        history = self.cnn_lstm.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path='models/deepshield_cnn_lstm.h5'
        )
        
        return self.cnn_lstm
    
    def train_dqn(self, X_train, y_train, episodes=10, batch_size=256):
        """
        Train DQN agent for adaptive thresholding.
        
        Args:
            X_train: Training features
            y_train: Training labels
            episodes: Number of training episodes
            batch_size: Batch size per episode
            
        Returns:
            Trained DQN agent
        """
        print("\n" + "="*70)
        print("TRAINING DQN AGENT")
        print("="*70)
        
        if self.cnn_lstm is None:
            raise ValueError("CNN-LSTM must be trained first")
        
        # Initialize DQN agent
        self.dqn_agent = DQNAgent(state_size=42, action_size=3)
        
        print(f"\nTraining DQN for {episodes} episodes...")
        print(f"  Batch size: {batch_size}")
        print(f"  Initial epsilon: {self.dqn_agent.epsilon}")
        
        for episode in range(episodes):
            # Sample random batch
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # Get CNN-LSTM predictions
            predictions, confidences = self.cnn_lstm.predict(X_batch)
            
            # Train DQN on this episode
            avg_reward, avg_loss = self.dqn_agent.train_episode(
                X_batch, y_batch, predictions, confidences
            )
            
            # Update target network periodically
            if (episode + 1) % 5 == 0:
                self.dqn_agent.update_target_model()
            
            print(f"  Episode {episode+1}/{episodes}: "
                  f"Reward={avg_reward:.2f}, Loss={avg_loss:.4f}, "
                  f"Epsilon={self.dqn_agent.epsilon:.3f}, "
                  f"Threshold={self.dqn_agent.get_threshold():.3f}")
        
        print(f"\n✓ DQN Training Completed")
        print(f"  Final epsilon: {self.dqn_agent.epsilon:.3f}")
        print(f"  Final threshold: {self.dqn_agent.get_threshold():.3f}")
        
        return self.dqn_agent
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate hybrid model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        # Evaluate CNN-LSTM
        print("\n[CNN-LSTM Evaluation]")
        cnn_lstm_metrics = self.cnn_lstm.evaluate(X_test, y_test)
        
        # Evaluate with DQN-adjusted thresholds
        print("\n[Hybrid CNN-LSTM + DQN Evaluation]")
        self._evaluate_with_dqn(X_test, y_test)
        
        return cnn_lstm_metrics
    
    def _evaluate_with_dqn(self, X_test, y_test):
        """
        Evaluate using DQN-adjusted thresholds.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        # Get CNN-LSTM predictions
        _, confidences = self.cnn_lstm.predict(X_test)
        
        # Apply DQN threshold adjustment
        predictions = []
        self.dqn_agent.reset_threshold()
        
        for i in range(len(X_test)):
            # Get state
            state = self.dqn_agent.get_state(X_test[i], confidences[i])
            
            # Get action (no exploration)
            action = self.dqn_agent.act(state, training=False)
            
            # Adjust threshold
            threshold = self.dqn_agent.adjust_threshold(action)
            
            # Make prediction
            prediction = 1 if confidences[i] >= threshold else 0
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn}, FP: {fp}")
        print(f"  FN: {fn}, TP: {tp}")
        print(f"\nDetection Metrics:")
        print(f"  Detection Rate: {detection_rate:.4f}")
        print(f"  False Alarm Rate: {false_alarm_rate:.4f}")
    
    def save_models(self):
        """Save all trained models."""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Save CNN-LSTM
        self.cnn_lstm.save('models/deepshield_cnn_lstm.h5')
        
        # Save DQN
        self.dqn_agent.save('models/dqn_weights.h5')
        
        # Save feature aligner
        self.dataset.save_aligner('models/feature_aligner.pkl')
        
        print("\n✓ All models saved successfully")


def main():
    """Main training function."""
    print("\n" + "="*70)
    print(" "*15 + "DEEPSHIELD-LAB TRAINING")
    print(" "*10 + "Real-Time Adaptive Intrusion Detection")
    print("="*70)
    
    # Configuration
    INPUT_DIM = 41
    CNN_LSTM_EPOCHS = 50
    CNN_LSTM_BATCH_SIZE = 128
    DQN_EPISODES = 10
    DQN_BATCH_SIZE = 256
    
    # Use available datasets (will skip missing ones)
    DATASETS = ['kdd', 'cicids', 'unsw']
    
    try:
        # Initialize trainer
        trainer = HybridTrainer(input_dim=INPUT_DIM)
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_data(DATASETS)
        
        # Train CNN-LSTM
        trainer.train_cnn_lstm(
            X_train, y_train, X_val, y_val,
            epochs=CNN_LSTM_EPOCHS,
            batch_size=CNN_LSTM_BATCH_SIZE
        )
        
        # Train DQN
        trainer.train_dqn(
            X_train, y_train,
            episodes=DQN_EPISODES,
            batch_size=DQN_BATCH_SIZE
        )
        
        # Evaluate
        trainer.evaluate(X_test, y_test)
        
        # Save models
        trainer.save_models()
        
        print("\n" + "="*70)
        print(" "*20 + "TRAINING COMPLETE!")
        print("="*70)
        print("\nTrained models saved to models/ directory:")
        print("  - deepshield_cnn_lstm.h5")
        print("  - dqn_weights.h5")
        print("  - feature_aligner.pkl")
        print("\nRun real-time detection with: sudo python realtime/sniffer.py")
        print("="*70 + "\n")
        
    except RuntimeError as e:
        print(f"\n✗ Training failed: {e}")
        print("\nPlease ensure datasets are placed in the data/ directory:")
        print("  - data/nsl-kdd/")
        print("  - data/cicids2017/")
        print("  - data/unsw-nb15/")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
