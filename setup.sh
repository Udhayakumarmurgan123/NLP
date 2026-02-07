#!/bin/bash

# Energy Forecasting Project Setup Script
# This script sets up the entire project environment

echo "=========================================="
echo "Energy Consumption Forecasting - Setup"
echo "=========================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo ""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "1. Activate virtual environment: venv\\Scripts\\activate"
else
    echo "1. Activate virtual environment: source venv/bin/activate"
fi
echo "2. Run Streamlit: streamlit run app.py"
echo ""
echo "=========================================="
