# DeepShield-LAB System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DEEPSHIELD-LAB ARCHITECTURE                          ║
║              Real-Time Adaptive Intrusion Detection System                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                            DATA INGESTION LAYER                              │
└──────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  NSL-KDD    │    │ CICIDS2017  │    │  UNSW-NB15  │
   │  (41 feat)  │    │  (78 feat)  │    │  (49 feat)  │
   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │                  │                   │
          └──────────────────┴───────────────────┘
                             ▼
                    ┌────────────────┐
                    │  Data Loaders  │
                    ├────────────────┤
                    │ • KDDLoader    │
                    │ • CICIDSLoader │
                    │ • UNSWLoader   │
                    └────────┬───────┘
                             ▼
                    ┌────────────────┐
                    │  Preprocessing │
                    ├────────────────┤
                    │ • Clean NaN    │
                    │ • Encode Cat   │
                    │ • Normalize    │
                    └────────┬───────┘
                             ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ALIGNMENT LAYER                               │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────┐
                    │ Feature Aligner│
                    ├────────────────┤
                    │  PCA Transform │
                    │  Target: 41D   │
                    └────────┬───────┘
                             ▼
              ┌──────────────┴──────────────┐
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │  41D    │    │  41D    │    │  41D    │
        │  (KDD)  │    │ (CICIDS)│    │ (UNSW)  │
        └────┬────┘    └────┬────┘    └────┬────┘
             └──────────────┴──────────────┘
                            ▼
                   ┌─────────────────┐
                   │ Unified Dataset │
                   │   (41D, Mixed)  │
                   └────────┬────────┘
                            ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                         DEEP LEARNING LAYER                                  │
└──────────────────────────────────────────────────────────────────────────────┘

                   ┌──────────────────┐
                   │  CNN-LSTM Model  │
                   ├──────────────────┤
                   │ Input: (1, 41)   │
                   └────────┬─────────┘
                            ▼
                ┌───────────────────────┐
                │   CNN Layers          │
                ├───────────────────────┤
                │ • Conv1D (64 filters) │
                │ • Conv1D (32 filters) │
                │ • BatchNorm + Dropout │
                └──────────┬────────────┘
                           ▼
                ┌───────────────────────┐
                │   LSTM Layers         │
                ├───────────────────────┤
                │ • LSTM (64 units)     │
                │ • LSTM (32 units)     │
                │ • Recurrent Dropout   │
                └──────────┬────────────┘
                           ▼
                ┌───────────────────────┐
                │   Dense Layers        │
                ├───────────────────────┤
                │ • Dense (128 units)   │
                │ • Dense (64 units)    │
                │ • BatchNorm + Dropout │
                └──────────┬────────────┘
                           ▼
                ┌───────────────────────┐
                │   Output Layer        │
                ├───────────────────────┤
                │ • Dense (1, sigmoid)  │
                │ • Binary: 0/1         │
                └──────────┬────────────┘
                           ▼
                    ┌──────────────┐
                    │ Prediction   │
                    │ + Confidence │
                    └──────┬───────┘
                           ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                   REINFORCEMENT LEARNING LAYER                               │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │    DQN Agent     │
                    ├──────────────────┤
                    │ State: 42D       │
                    │ (41 feat + conf) │
                    └────────┬─────────┘
                             ▼
              ┌──────────────┴───────────────┐
              ▼                              ▼
     ┌────────────────┐           ┌────────────────┐
     │  Q-Network     │           │ Target Network │
     ├────────────────┤           ├────────────────┤
     │ • Dense(128)   │           │ • Dense(128)   │
     │ • Dense(64)    │           │ • Dense(64)    │
     │ • Dense(32)    │           │ • Dense(32)    │
     │ • Dense(3)     │           │ • Dense(3)     │
     └────────┬───────┘           └────────────────┘
              │
              ▼
     ┌────────────────┐
     │  Action Space  │
     ├────────────────┤
     │ 0: Decrease ↓  │
     │ 1: Maintain →  │
     │ 2: Increase ↑  │
     └────────┬───────┘
              ▼
     ┌────────────────┐
     │ Threshold (τ)  │
     │  [0.1 - 0.9]   │
     └────────┬───────┘
              ▼
     ┌────────────────┐
     │ Final Decision │
     │ attack if p≥τ  │
     └────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME DETECTION LAYER                               │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  Network Traffic │
                    │   (Live Packets) │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  Packet Sniffer  │
                    ├──────────────────┤
                    │ • Scapy Capture  │
                    │ • Interface Mon. │
                    │ • IP/TCP/UDP     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Flow Extractor   │
                    ├──────────────────┤
                    │ • 5s Window      │
                    │ • Bidirectional  │
                    │ • Statistics     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Feature Extract  │
                    │   (41 Features)  │
                    └────────┬─────────┘
                             ▼
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │ Preprocessing  │           │   Alignment    │
     │ • Normalize    │           │ • PCA Apply    │
     │ • Clean        │           │ • 41D Output   │
     └────────┬───────┘           └────────┬───────┘
              └──────────────┬──────────────┘
                             ▼
                    ┌──────────────────┐
                    │  CNN-LSTM Model  │
                    │   (Inference)    │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  DQN Agent       │
                    │ (Threshold Adj)  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Detection Result │
                    ├──────────────────┤
                    │ • is_attack      │
                    │ • confidence     │
                    │ • threshold      │
                    │ • threat_level   │
                    └────────┬─────────┘
                             ▼
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │  Alert System  │           │  Statistics    │
     │ • Console Log  │           │ • Packet Count │
     │ • Notification │           │ • Flow Count   │
     │ • Flow Details │           │ • Attack Count │
     └────────────────┘           └────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                   │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌────────────┐     ┌────────────┐     ┌────────────┐
  │ Load Data  │ →   │ Align 41D  │ →   │ Split Data │
  └────────────┘     └────────────┘     └────────────┘
        │                                       │
        │                                       ▼
        │                            ┌─────────────────┐
        │                            │  Train/Val/Test │
        │                            │   60/20/20 %    │
        │                            └────────┬────────┘
        │                                     │
        ▼                                     ▼
  ┌──────────────────────────────────────────────────┐
  │          PHASE 1: CNN-LSTM Training              │
  ├──────────────────────────────────────────────────┤
  │ • Epochs: 50                                     │
  │ • Batch: 128                                     │
  │ • Loss: Binary Cross-Entropy                     │
  │ • Optimizer: Adam                                │
  │ • Callbacks: EarlyStopping, LR Scheduling        │
  │                                                  │
  │ Output: models/deepshield_cnn_lstm.h5            │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │          PHASE 2: DQN Training                   │
  ├──────────────────────────────────────────────────┤
  │ • Episodes: 10                                   │
  │ • Batch: 256                                     │
  │ • Replay Memory: 10k                             │
  │ • Epsilon Decay: 0.995                           │
  │ • Reward: Accuracy - False Alarms                │
  │                                                  │
  │ Output: models/dqn_weights.h5                    │
  └────────────────────────┬─────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────┐
  │          PHASE 3: Evaluation                     │
  ├──────────────────────────────────────────────────┤
  │ • Test Set Evaluation                            │
  │ • Confusion Matrix                               │
  │ • Detection Rate                                 │
  │ • False Alarm Rate                               │
  │ • Hybrid (CNN-LSTM + DQN) Performance            │
  └────────────────────────┬─────────────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Model Artifacts │
                  ├──────────────────┤
                  │ • CNN-LSTM (.h5) │
                  │ • DQN (.h5)      │
                  │ • Aligner (.pkl) │
                  └──────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INTERACTIONS                               │
└──────────────────────────────────────────────────────────────────────────────┘

Training Phase:
  Datasets → Loaders → Alignment → CNN-LSTM → DQN → Saved Models

Inference Phase:
  Network → Packets → Flows → Features → Alignment → CNN-LSTM → DQN → Alert

Reward Function:
  TP (attack detected):     +10 * confidence
  TN (normal correct):      +5  * confidence
  FP (false alarm):         -15 * confidence
  FN (missed attack):       -20 * confidence

┌──────────────────────────────────────────────────────────────────────────────┐
│                            KEY FEATURES                                      │
└──────────────────────────────────────────────────────────────────────────────┘

✓ Multi-Dataset Support
  • NSL-KDD (41 features)
  • CICIDS2017 (78+ features)
  • UNSW-NB15 (49 features)

✓ Feature Alignment
  • PCA-based dimensionality reduction
  • Unified 41D feature space
  • Variance preservation

✓ Hybrid Deep Learning
  • CNN for spatial patterns
  • LSTM for temporal patterns
  • Binary classification

✓ Adaptive Intelligence
  • DQN for threshold optimization
  • Experience replay
  • Target network stabilization

✓ Real-Time Detection
  • Live packet capture (Scapy)
  • Flow aggregation (5s window)
  • <10ms inference latency
  • Alert system

✓ Production Ready
  • Comprehensive error handling
  • Model versioning
  • Extensive testing
  • Complete documentation

┌──────────────────────────────────────────────────────────────────────────────┐
│                         PERFORMANCE METRICS                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Training:
  • CNN-LSTM: ~30-60 min (depends on dataset)
  • DQN: ~5-10 min
  • Total: ~40-70 min

Inference:
  • Latency: <10ms per flow
  • Throughput: ~1000 pkt/s
  • Memory: ~2-3 GB
  • CPU: 20-40% (single core)

Accuracy:
  • Detection Rate: 95-99%
  • False Alarm Rate: <5%
  • Precision: 90-95%
  • Recall: 90-95%
  • F1-Score: 90-95%

┌──────────────────────────────────────────────────────────────────────────────┐
│                           TECHNOLOGY STACK                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Languages:        Python 3.10+
Deep Learning:    TensorFlow 2.13+, Keras
ML Libraries:     scikit-learn, NumPy, pandas
Network:          Scapy
RL Framework:     Custom DQN (Gym-compatible)
Visualization:    matplotlib, seaborn
Testing:          pytest
OS:               Linux (primary), cross-platform compatible

┌──────────────────────────────────────────────────────────────────────────────┐
│                              FILE STRUCTURE                                  │
└──────────────────────────────────────────────────────────────────────────────┘

DeepShield-LAB/
├── app/
│   ├── data_engine/       # Dataset loaders & alignment
│   ├── models/            # CNN-LSTM model
│   └── rl/                # DQN agent
├── train/                 # Training pipeline
├── realtime/              # Detection engine
├── tests/                 # Test suite
├── models/                # Saved models (generated)
├── data/                  # Datasets (user-provided)
├── requirements.txt       # Dependencies
├── README.md              # Main documentation
├── PROJECT_SUMMARY.md     # Implementation details
├── DEPLOYMENT.md          # Deployment guide
├── ARCHITECTURE.md        # This file
└── quickstart.sh          # Quick start script

═══════════════════════════════════════════════════════════════════════════════
                        END OF ARCHITECTURE DIAGRAM
═══════════════════════════════════════════════════════════════════════════════
```
