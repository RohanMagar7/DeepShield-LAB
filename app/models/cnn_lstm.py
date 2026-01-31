"""
CNN-LSTM Model for Intrusion Detection
Combines CNN for spatial feature extraction and LSTM for temporal pattern recognition.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class CNNLSTM:
    """
    Hybrid CNN-LSTM model for binary intrusion detection.
    
    Architecture:
    - Input: (batch, 1, 41) - Sequential features
    - CNN layers: Extract spatial patterns
    - LSTM layers: Capture temporal dependencies
    - Dense layers: Binary classification
    """
    
    def __init__(self, input_dim=41, model_name='deepshield_cnn_lstm'):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_dim: Number of input features (default 41)
            model_name: Name for the model
        """
        self.input_dim = input_dim
        self.model_name = model_name
        self.model = None
        self.history = None
        
    def build_model(self, 
                   cnn_filters=[64, 32],
                   lstm_units=[64, 32],
                   dense_units=[128, 64],
                   dropout_rate=0.3):
        """
        Build CNN-LSTM architecture.
        
        Args:
            cnn_filters: List of CNN filter sizes
            lstm_units: List of LSTM unit sizes
            dense_units: List of dense layer sizes
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Keras model
        """
        print("Building CNN-LSTM Model...")
        print(f"  Input shape: (1, {self.input_dim})")
        
        # Input layer: (batch, timesteps, features)
        inputs = layers.Input(shape=(1, self.input_dim), name='input')
        
        # CNN layers for spatial feature extraction
        x = inputs
        for i, filters in enumerate(cnn_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=1,  # 1D convolution over features
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        print(f"  CNN Layers: {cnn_filters}")
        
        # LSTM layers for temporal pattern recognition
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)  # Last LSTM doesn't return sequences
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate / 2,
                name=f'lstm_{i+1}'
            )(x)
            if not return_sequences:
                x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        print(f"  LSTM Layers: {lstm_units}")
        
        # Dense layers for classification
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        print(f"  Dense Layers: {dense_units}")
        
        # Output layer: Binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        
        print(f"\nModel Summary:")
        print(f"  Total Parameters: {model.count_params():,}")
        print(f"  Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def get_model(self):
        """Return the model instance."""
        if self.model is None:
            self.build_model()
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val,
              epochs=50,
              batch_size=128,
              checkpoint_path='models/deepshield_cnn_lstm.h5',
              verbose=1):
        """
        Train the CNN-LSTM model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Reshape input to (batch, timesteps=1, features)
        X_train_reshaped = X_train.reshape(-1, 1, self.input_dim)
        X_val_reshaped = X_val.reshape(-1, 1, self.input_dim)
        
        print(f"\nTraining CNN-LSTM Model...")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        print(f"  Input shape: {X_train_reshaped.shape}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        
        # Create directory for checkpoint
        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        
        print(f"\n✓ Training completed")
        print(f"  Best val_loss: {min(history.history['val_loss']):.4f}")
        print(f"  Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return history
    
    def evaluate(self, X_test, y_test, batch_size=128):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Reshape input
        X_test_reshaped = X_test.reshape(-1, 1, self.input_dim)
        
        print(f"\nEvaluating Model...")
        print(f"  Test samples: {len(X_test)}")
        
        # Evaluate
        results = self.model.evaluate(X_test_reshaped, y_test, 
                                      batch_size=batch_size, 
                                      verbose=0)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
            print(f"  {name}: {value:.4f}")
        
        # Additional metrics
        y_pred_proba = self.model.predict(X_test_reshaped, batch_size=batch_size, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Detection rate and false alarm rate
        tn, fp, fn, tp = cm.ravel()
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nDetection Metrics:")
        print(f"  Detection Rate: {detection_rate:.4f}")
        print(f"  False Alarm Rate: {false_alarm_rate:.4f}")
        
        metrics['detection_rate'] = detection_rate
        metrics['false_alarm_rate'] = false_alarm_rate
        
        return metrics
    
    def predict(self, X, threshold=0.5, batch_size=128):
        """
        Predict on new data.
        
        Args:
            X: Input features (n_samples, n_features)
            threshold: Classification threshold
            batch_size: Batch size
            
        Returns:
            predictions: Binary predictions
            probabilities: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Reshape input
        X_reshaped = X.reshape(-1, 1, self.input_dim)
        
        # Predict probabilities
        probabilities = self.model.predict(X_reshaped, batch_size=batch_size, verbose=0)
        
        # Apply threshold
        predictions = (probabilities > threshold).astype(int).flatten()
        
        return predictions, probabilities.flatten()
    
    def save(self, filepath='models/deepshield_cnn_lstm.h5'):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/deepshield_cnn_lstm.h5'):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built")
        self.model.summary()


if __name__ == '__main__':
    # Test CNN-LSTM model
    print("Testing CNN-LSTM Model...")
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 41)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 41)
    y_val = np.random.randint(0, 2, 200)
    X_test = np.random.randn(200, 41)
    y_test = np.random.randint(0, 2, 200)
    
    # Build and train model
    model = CNNLSTM(input_dim=41)
    model.build_model()
    model.summary()
    
    print("\n✓ CNN-LSTM Model Test Passed")
    print(f"  Model built successfully")
    print(f"  Input shape: (1, 41)")
    print(f"  Output: Binary classification")
