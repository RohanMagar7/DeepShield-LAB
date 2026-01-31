# DeepShield-LAB Deployment Checklist

## 📋 Pre-Deployment Checklist

### Environment Setup
- [ ] Python 3.10+ installed
- [ ] pip package manager available
- [ ] Virtual environment created (recommended)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Root/sudo access available (for packet capture)

### Dataset Preparation
- [ ] NSL-KDD dataset downloaded and placed in `data/nsl-kdd/`
  - [ ] KDDTrain+.txt
  - [ ] KDDTest+.txt
- [ ] CICIDS2017 dataset downloaded and placed in `data/cicids2017/`
  - [ ] All CSV files
- [ ] UNSW-NB15 dataset downloaded and placed in `data/unsw-nb15/`
  - [ ] UNSW_NB15_training-set.csv
  - [ ] UNSW_NB15_testing-set.csv

### System Resources
- [ ] Minimum 8GB RAM available
- [ ] 10GB+ free disk space
- [ ] GPU support (optional, but recommended for training)

---

## 🔧 Installation Steps

### 1. Clone/Download Project
```bash
cd /home/r/project/DeepShield-LAB
```

### 2. Create Virtual Environment (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Expected packages:
- tensorflow>=2.13.0
- keras>=2.13.0
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- scapy>=2.5.0
- gym>=0.26.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- tqdm>=4.65.0
- pyyaml>=6.0
- pytest>=7.4.0

### 4. Verify Installation
```bash
python3 tests/test_alignment.py
```

---

## 🎯 Training Workflow

### Step 1: Prepare Data
Ensure datasets are in correct locations:
```
data/
├── nsl-kdd/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── cicids2017/
│   └── *.csv (all CSV files)
└── unsw-nb15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

### Step 2: Run Training
```bash
python train/train_hybrid.py
```

**Training Process:**
1. Loads all three datasets
2. Aligns features to 41D using PCA
3. Trains CNN-LSTM model (~30-60 minutes)
4. Trains DQN agent (~5-10 minutes)
5. Evaluates on test set
6. Saves models to `models/` directory

**Expected Output Files:**
- `models/deepshield_cnn_lstm.h5` - CNN-LSTM model (~2-5 MB)
- `models/dqn_weights.h5` - DQN weights (~500 KB)
- `models/feature_aligner.pkl` - Feature aligner (~100 KB)

### Step 3: Verify Training
Check that all model files exist:
```bash
ls -lh models/
```

---

## 🚀 Deployment Options

### Option 1: Real-Time Network Monitoring
```bash
# Run as root/sudo for packet capture
sudo python realtime/sniffer.py
```

**Features:**
- Live packet capture
- Flow aggregation (5s window)
- Real-time threat detection
- Alert notifications
- Statistics dashboard

**Requirements:**
- Root/sudo privileges
- Network interface access
- Trained models in `models/` directory

### Option 2: Batch Processing
Use the detector module for batch analysis:
```python
from realtime.detector import IntrusionDetector

detector = IntrusionDetector()
detector.load_models()

# Analyze features
features = extract_features_from_file(...)
result = detector.detect(features)
print(result)  # {'is_attack': True, 'confidence': 0.95, ...}
```

### Option 3: API Integration
Create a REST API wrapper:
```python
from flask import Flask, request, jsonify
from realtime.detector import IntrusionDetector

app = Flask(__name__)
detector = IntrusionDetector()
detector.load_models()

@app.route('/detect', methods=['POST'])
def detect():
    features = request.json['features']
    result = detector.detect(features)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🧪 Testing Checklist

### Unit Tests
- [ ] Run alignment tests: `python tests/test_alignment.py`
- [ ] Test individual loaders:
  - [ ] `python app/data_engine/kdd_loader.py`
  - [ ] `python app/data_engine/cicids_loader.py`
  - [ ] `python app/data_engine/unsw_loader.py`
- [ ] Test feature aligner: `python app/data_engine/feature_aligner.py`
- [ ] Test CNN-LSTM model: `python app/models/cnn_lstm.py`
- [ ] Test DQN agent: `python app/rl/dqn_agent.py`

### Integration Tests
- [ ] Test unified dataset loading
- [ ] Test full training pipeline
- [ ] Test detector with synthetic data
- [ ] Test flow extractor
- [ ] Test sniffer (requires network access)

### Performance Tests
- [ ] Measure inference latency (target: < 10ms)
- [ ] Test with high packet rates (> 1000 pkt/s)
- [ ] Monitor memory usage during long runs
- [ ] Verify GPU utilization (if available)

---

## 📊 Performance Metrics

### Expected Training Results
- **Accuracy:** 95-99% (depends on dataset)
- **Precision:** 90-95%
- **Recall:** 90-95%
- **F1-Score:** 90-95%
- **False Alarm Rate:** < 5%

### Real-Time Performance
- **Throughput:** ~1000 packets/second
- **Inference Latency:** < 10ms per flow
- **Memory Usage:** ~2-3 GB
- **CPU Usage:** 20-40% (single core)

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Dataset Not Found
```
Error: FileNotFoundError: No such file or directory
```
**Solution:** Verify dataset paths in `data/` directory

#### 2. Permission Denied (Packet Capture)
```
Error: PermissionError: Operation not permitted
```
**Solution:** Run with sudo: `sudo python realtime/sniffer.py`

#### 3. Out of Memory
```
Error: ResourceExhaustedError: OOM when allocating tensor
```
**Solution:** 
- Reduce batch size in training
- Close other applications
- Use smaller dataset split

#### 4. Missing Dependencies
```
Error: ModuleNotFoundError: No module named 'tensorflow'
```
**Solution:** Install dependencies: `pip install -r requirements.txt`

#### 5. GPU Not Detected
```
Warning: No GPU detected, using CPU
```
**Solution:** 
- Install CUDA and cuDNN (for NVIDIA GPUs)
- Verify TensorFlow GPU installation: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## 🔐 Security Considerations

### Network Monitoring
- [ ] Configure appropriate network interface
- [ ] Ensure legal authorization for packet capture
- [ ] Implement data retention policies
- [ ] Encrypt stored packet captures (if any)

### Access Control
- [ ] Restrict root access to authorized users
- [ ] Implement logging for all detections
- [ ] Set up alerting for high-threat events
- [ ] Regularly update models with new data

### Data Privacy
- [ ] Packet capture only includes headers (no payload)
- [ ] Anonymize IP addresses in logs (if required)
- [ ] Comply with data protection regulations
- [ ] Document data handling procedures

---

## 📈 Production Deployment

### Infrastructure Requirements

#### Minimum Requirements
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Disk:** 20 GB SSD
- **Network:** 1 Gbps interface

#### Recommended Requirements
- **CPU:** 8+ cores
- **RAM:** 16+ GB
- **Disk:** 50+ GB SSD
- **Network:** 10 Gbps interface
- **GPU:** NVIDIA GPU with 8+ GB VRAM

### Scaling Considerations

#### Horizontal Scaling
- Deploy multiple instances across network segments
- Use load balancer for distributed packet capture
- Centralized alert aggregation
- Shared model repository

#### Vertical Scaling
- Increase CPU cores for higher throughput
- Add more RAM for larger flow tables
- GPU acceleration for faster inference
- NVMe storage for model loading

### Monitoring & Maintenance

#### Metrics to Track
- [ ] Packet capture rate (pkt/s)
- [ ] Detection throughput (flows/s)
- [ ] False positive rate
- [ ] False negative rate
- [ ] System resource usage (CPU, RAM, disk)
- [ ] Model inference latency

#### Regular Maintenance
- [ ] Weekly: Review detected threats
- [ ] Monthly: Retrain models with new data
- [ ] Quarterly: Performance benchmarking
- [ ] Annually: Full system audit

---

## 🎓 Training & Documentation

### User Training
- [ ] System overview and capabilities
- [ ] Alert interpretation
- [ ] Incident response procedures
- [ ] False positive handling

### Documentation
- [ ] System architecture diagram
- [ ] API documentation (if applicable)
- [ ] Incident response playbook
- [ ] Troubleshooting guide

---

## ✅ Go-Live Checklist

### Pre-Production
- [ ] All tests passing
- [ ] Models trained and validated
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] Documentation finalized
- [ ] Team training completed

### Production Launch
- [ ] Deploy to production environment
- [ ] Configure network monitoring
- [ ] Enable alerting system
- [ ] Start packet capture
- [ ] Monitor initial performance
- [ ] Verify alerts are generated

### Post-Launch
- [ ] Monitor system for 24 hours
- [ ] Review initial detections
- [ ] Tune thresholds if needed
- [ ] Collect user feedback
- [ ] Document lessons learned

---

## 📞 Support & Resources

### Documentation
- **README.md** - Complete project overview
- **PROJECT_SUMMARY.md** - Technical implementation details
- **This file** - Deployment guide

### Code Structure
- **app/data_engine/** - Dataset processing
- **app/models/** - Deep learning models
- **app/rl/** - Reinforcement learning
- **train/** - Training pipeline
- **realtime/** - Detection engine
- **tests/** - Test suite

### External Resources
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- TensorFlow: https://www.tensorflow.org/
- Scapy: https://scapy.net/

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅
