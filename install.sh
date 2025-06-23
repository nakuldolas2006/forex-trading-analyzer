#!/bin/bash
# Forex Trading Analyzer Installation Script

echo "Installing Forex Trading Analyzer..."

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p recordings logs data templates

echo "Installation complete!"
echo "To run the application:"
echo "streamlit run app.py --server.port 5000"
