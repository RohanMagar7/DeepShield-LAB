# DeepShield-LAB
## Real-Time Adaptive Intrusion Detection System

An industry-grade IDS combining Deep Learning (CNN-LSTM) and Reinforcement Learning (DQN) for adaptive threat detection with multi-dataset training.

## 🔥 Features

- **Multi-Dataset Training**: NSL-KDD, CICIDS2017, UNSW-NB15
- **Unified Feature Space**: PCA-based alignment to 41 dimensions
- **Hybrid Architecture**: CNN-LSTM for pattern detection + DQN for adaptive thresholding
- **Real-Time Detection**: Live network traffic analysis using Scapy
- **Zero-Day Capable**: Generalized threat detection across datasets
- **False Alarm Reduction**: RL-based adaptive decision making

## 📁 Project Structure

```
DeepShield-LAB/
├── app/
│   ├── data_engine/         # Dataset loaders and feature alignment
│   │   ├── kdd_loader.py
│   │   ├── cicids_loader.py
│   │   ├── unsw_loader.py
│   │   ├── feature_aligner.py
│   │   └── unified_dataset.py
│   ├── models/              # Deep learning models
│   │   └── cnn_lstm.py
│   └── rl/                  # Reinforcement learning
│       └── dqn_agent.py
├── train/                   # Training pipeline
│   └── train_hybrid.py
├── realtime/                # Real-time detection engine
│   ├── flow_extractor.py
│   ├── detector.py
│   └── sniffer.py
├── tests/                   # Test suite
│   └── test_alignment.py
├── models/                  # Saved models (created during training)
├── data/                    # Datasets (user-provided)
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# Clone or navigate to project directory
cd DeepShield-LAB

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Setup

Download and place datasets in the `data/` directory:

1. **NSL-KDD**: Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/nsl-kdd/`
2. **CICIDS2017**: Place CSV files in `data/cicids2017/`
3. **UNSW-NB15**: Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` in `data/unsw-nb15/`

Example structure:
```
data/
├── nsl-kdd/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── cicids2017/
│   └── *.csv
└── unsw-nb15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

## 🎯 Usage

### 1. Train the Model

```bash
python train/train_hybrid.py
```

This will:
- Load and preprocess all three datasets
- Align features to unified 41D space using PCA
- Train CNN-LSTM model for binary classification
- Train DQN agent for adaptive thresholding
- Save models to `models/` directory:
  - `models/deepshield_cnn_lstm.h5`
  - `models/feature_aligner.pkl`
  - `models/dqn_weights.h5`

### 2. Test Feature Alignment

```bash
python tests/test_alignment.py
```

Verifies that all three datasets are correctly mapped to 41 dimensions.

### 3. Real-Time Detection

```bash
# Run as root/sudo for packet capture
sudo python realtime/sniffer.py
```

This will:
- Capture live network traffic
- Extract flow features
- Apply CNN-LSTM + DQN detection
- Display alerts in real-time

## 🏗️ Architecture

### Data Engine
- **Loaders**: Clean, normalize, and preprocess NSL-KDD, CICIDS2017, UNSW-NB15
- **Feature Aligner**: PCA-based dimensionality reduction to unified 41D space
- **Unified Dataset**: Combined training/testing data from multiple sources

### CNN-LSTM Model
- **Input**: (1, 41) - Sequential features
- **CNN Layers**: Extract spatial patterns from traffic features
- **LSTM Layers**: Capture temporal dependencies
- **Output**: Binary classification (Normal/Attack)

### DQN Agent
- **State Space**: 41 features + detection confidence
- **Action Space**: Threshold adjustment (increase/decrease/maintain)
- **Reward**: Based on detection accuracy and false alarm rate
- **Goal**: Minimize false positives while maintaining high detection rate

### Real-Time Engine
- **Flow Extractor**: Aggregates packet features into network flows
- **Detector**: Applies trained models to classify traffic
- **Sniffer**: Captures live packets using Scapy

## 📈 Performance Metrics

The system tracks:
- **Detection Rate**: True positive rate
- **False Alarm Rate**: False positive rate
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Zero-Day Detection**: Performance on unseen attack types

## 🔬 Technical Details

### Feature Alignment
- Original feature counts: KDD (41), CICIDS (80+), UNSW (49)
- Target dimensionality: 41 features
- Method: PCA with variance preservation
- Ensures consistent input shape across all datasets

### Model Configuration
- **CNN-LSTM Input Shape**: (1, 41)
- **DQN State Size**: 41
- **Training**: Mixed-dataset batches for generalization
- **Optimization**: Adam optimizer with learning rate scheduling

### Real-Time Processing
- **Flow Window**: 5-second aggregation
- **Feature Extraction**: Statistical features from packet headers
- **Inference Latency**: < 10ms per flow
- **Scalability**: Multi-threaded packet processing

## 🛡️ Security Considerations

- **Root Privileges**: Required for packet capture (use with caution)
- **Network Interface**: Configure appropriate interface in sniffer
- **Data Privacy**: Captures network metadata only (no payload)
- **Production Deployment**: Implement proper logging and alerting

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
python tests/test_alignment.py
```

## 📝 License

For educational and research purposes.

## 🤝 Contributing

This is a reference implementation. For production use, consider:
- Distributed training for large datasets
- Model versioning and A/B testing
- Integration with SIEM systems
- Cloud deployment options
- Continuous learning pipelines

## 📚 References

- NSL-KDD: Network Security Dataset
- CICIDS2017: Canadian Institute for Cybersecurity IDS Dataset
- UNSW-NB15: University of New South Wales Network-Based Dataset

## ⚠️ Disclaimer

This tool is for authorized security testing and research only. Users are responsible for compliance with applicable laws and regulations.
# DeepShield-LAB
# DeepShield-LAB
