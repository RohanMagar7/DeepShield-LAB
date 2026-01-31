"""
Deep Q-Network (DQN) Agent for Adaptive Intrusion Detection
Learns optimal detection thresholds to minimize false alarms while maintaining high detection rate.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from collections import deque
import random


class DQNAgent:
    """
    DQN agent for adaptive threshold adjustment in IDS.
    
    State: 41 features + detection confidence
    Action: Adjust threshold (0: decrease, 1: maintain, 2: increase)
    Reward: Based on detection accuracy and false alarm rate
    """
    
    def __init__(self, state_size=42, action_size=3, learning_rate=0.001):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state space (41 features + 1 confidence)
            action_size: Number of actions (3: decrease/maintain/increase threshold)
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        # Replay memory
        self.memory = deque(maxlen=10000)
        
        # Q-Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Threshold tracking
        self.current_threshold = 0.5
        self.threshold_min = 0.1
        self.threshold_max = 0.9
        self.threshold_step = 0.05
        
    def _build_model(self):
        """
        Build neural network for Q-value estimation.
        
        Returns:
            Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action
        """
        if training and np.random.random() <= self.epsilon:
            # Exploration
            return random.randrange(self.action_size)
        
        # Exploitation
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train on batch of experiences from replay memory.
        
        Returns:
            loss: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Compute target Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def get_state(self, features, confidence):
        """
        Construct state from features and confidence.
        
        Args:
            features: Network flow features (41D)
            confidence: Detection confidence from CNN-LSTM
            
        Returns:
            state: Combined state vector
        """
        # Ensure features are 41D
        if len(features) != 41:
            raise ValueError(f"Expected 41 features, got {len(features)}")
        
        # Combine features with confidence
        state = np.concatenate([features, [confidence]])
        return state
    
    def adjust_threshold(self, action):
        """
        Adjust detection threshold based on action.
        
        Args:
            action: Action to take (0: decrease, 1: maintain, 2: increase)
            
        Returns:
            new_threshold: Updated threshold
        """
        if action == 0:  # Decrease
            self.current_threshold = max(
                self.threshold_min,
                self.current_threshold - self.threshold_step
            )
        elif action == 2:  # Increase
            self.current_threshold = min(
                self.threshold_max,
                self.current_threshold + self.threshold_step
            )
        # action == 1: maintain (no change)
        
        return self.current_threshold
    
    def compute_reward(self, true_label, prediction, confidence):
        """
        Compute reward based on detection accuracy and confidence.
        
        Args:
            true_label: Ground truth (0=normal, 1=attack)
            prediction: Model prediction (0=normal, 1=attack)
            confidence: Prediction confidence
            
        Returns:
            reward: Computed reward
        """
        # Correct detection
        if true_label == prediction:
            if true_label == 1:
                # True positive (attack detected)
                reward = 10.0 * confidence
            else:
                # True negative (normal correctly classified)
                reward = 5.0 * confidence
        else:
            if prediction == 1:
                # False positive (false alarm)
                reward = -15.0 * confidence
            else:
                # False negative (missed attack)
                reward = -20.0 * confidence
        
        return reward
    
    def train_episode(self, X_batch, y_batch, predictions, confidences):
        """
        Train on a batch of samples (one episode).
        
        Args:
            X_batch: Feature batch (n_samples, 41)
            y_batch: True labels
            predictions: Model predictions
            confidences: Prediction confidences
            
        Returns:
            avg_reward: Average reward for episode
            avg_loss: Average training loss
        """
        total_reward = 0.0
        total_loss = 0.0
        
        for i in range(len(X_batch)):
            # Get state
            state = self.get_state(X_batch[i], confidences[i])
            
            # Choose action
            action = self.act(state, training=True)
            
            # Adjust threshold
            new_threshold = self.adjust_threshold(action)
            
            # Re-classify with new threshold
            new_prediction = 1 if confidences[i] >= new_threshold else 0
            
            # Compute reward
            reward = self.compute_reward(y_batch[i], new_prediction, confidences[i])
            total_reward += reward
            
            # Next state (same features, potentially different confidence)
            next_state = state  # In this simplified version
            
            # Store experience
            done = (i == len(X_batch) - 1)
            self.remember(state, action, reward, next_state, done)
            
            # Train
            loss = self.replay()
            total_loss += loss
        
        avg_reward = total_reward / len(X_batch)
        avg_loss = total_loss / len(X_batch)
        
        return avg_reward, avg_loss
    
    def save(self, filepath='models/dqn_weights.h5'):
        """
        Save DQN model weights.
        
        Args:
            filepath: Path to save weights
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"DQN weights saved to {filepath}")
    
    def load(self, filepath='models/dqn_weights.h5'):
        """
        Load DQN model weights.
        
        Args:
            filepath: Path to weights file
        """
        self.model.load_weights(filepath)
        self.update_target_model()
        print(f"DQN weights loaded from {filepath}")
    
    def get_threshold(self):
        """Return current threshold."""
        return self.current_threshold
    
    def reset_threshold(self):
        """Reset threshold to default."""
        self.current_threshold = 0.5


if __name__ == '__main__':
    # Test DQN agent
    print("Testing DQN Agent...")
    
    # Create agent
    agent = DQNAgent(state_size=42, action_size=3)
    
    print(f"\n✓ DQN Agent Test Passed")
    print(f"  State size: {agent.state_size}")
    print(f"  Action size: {agent.action_size}")
    print(f"  Initial threshold: {agent.current_threshold}")
    print(f"  Epsilon: {agent.epsilon}")
    
    # Test state construction
    features = np.random.randn(41)
    confidence = 0.85
    state = agent.get_state(features, confidence)
    print(f"  State shape: {state.shape}")
    
    # Test action selection
    action = agent.act(state, training=False)
    print(f"  Sample action: {action}")
    
    # Test threshold adjustment
    new_threshold = agent.adjust_threshold(action)
    print(f"  Adjusted threshold: {new_threshold}")
