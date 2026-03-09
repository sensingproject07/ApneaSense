#!/bin/bash
# ===============================
# ApneaSense - Environment Setup (macOS/Linux)
# ===============================

# Check for Python 3.12
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION != *"3."* ]]; then
    echo "Please install Python 3.xx before running this script."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt
python -m ipykernel install --user --name apneasense --display-name "Python (ApneaSense)"

echo ""
echo "==============================="
echo "Setup complete!"
echo "Activate environment: source .venv/bin/activate"
echo "==============================="