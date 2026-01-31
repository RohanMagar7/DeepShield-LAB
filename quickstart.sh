#!/bin/bash

# DeepShield-LAB Quick Start Script
# Automated setup and testing

echo "=================================================="
echo "     DeepShield-LAB Quick Start"
echo "     Real-Time Adaptive Intrusion Detection"
echo "=================================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Create virtual environment (optional)
echo ""
echo "[2/5] Setting up virtual environment (optional)..."
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

echo "  To activate: source venv/bin/activate"

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
echo "  This may take a few minutes..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Dependencies installed successfully"
else
    echo "  ⚠ Error installing dependencies"
    echo "  Run manually: pip install -r requirements.txt"
fi

# Check for datasets
echo ""
echo "[4/5] Checking for datasets..."

datasets_found=0

if [ -d "data/nsl-kdd" ] && [ -f "data/nsl-kdd/KDDTrain+.txt" ]; then
    echo "  ✓ NSL-KDD dataset found"
    datasets_found=$((datasets_found + 1))
else
    echo "  ✗ NSL-KDD dataset not found"
    echo "    Place files in: data/nsl-kdd/"
fi

if [ -d "data/cicids2017" ] && [ "$(ls -A data/cicids2017/*.csv 2>/dev/null)" ]; then
    echo "  ✓ CICIDS2017 dataset found"
    datasets_found=$((datasets_found + 1))
else
    echo "  ✗ CICIDS2017 dataset not found"
    echo "    Place CSV files in: data/cicids2017/"
fi

if [ -d "data/unsw-nb15" ] && [ -f "data/unsw-nb15/UNSW_NB15_training-set.csv" ]; then
    echo "  ✓ UNSW-NB15 dataset found"
    datasets_found=$((datasets_found + 1))
else
    echo "  ✗ UNSW-NB15 dataset not found"
    echo "    Place files in: data/unsw-nb15/"
fi

if [ $datasets_found -eq 0 ]; then
    echo ""
    echo "  ⚠ No datasets found. Please download and place datasets:"
    echo "     - NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html"
    echo "     - CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html"
    echo "     - UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
fi

# Run tests
echo ""
echo "[5/5] Running tests..."
python3 tests/test_alignment.py
test_result=$?

echo ""
echo "=================================================="
echo "     Quick Start Complete"
echo "=================================================="
echo ""

if [ $datasets_found -gt 0 ]; then
    echo "Next steps:"
    echo "  1. Train models:"
    echo "     python train/train_hybrid.py"
    echo ""
    echo "  2. Run real-time detection (requires sudo):"
    echo "     sudo python realtime/sniffer.py"
else
    echo "Next steps:"
    echo "  1. Download and place datasets in data/ directory"
    echo "  2. Run: python train/train_hybrid.py"
    echo "  3. Run: sudo python realtime/sniffer.py"
fi

echo ""
echo "For detailed documentation, see README.md"
echo "=================================================="
