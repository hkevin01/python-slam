#!/bin/bash
# Development setup script for Python SLAM project

set -e

echo "Setting up Python SLAM development environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Python SLAM Environment Variables
PYTHONPATH=\${PWD}/src
ROS_DOMAIN_ID=42
ROS_DISTRO=humble

# Development settings
DEBUG=True
LOG_LEVEL=INFO

# SLAM parameters
MAX_FEATURES=1000
LOOP_CLOSURE_THRESHOLD=0.7
MAP_RESOLUTION=0.05

# Camera calibration (example values - update with actual calibration)
CAMERA_FX=525.0
CAMERA_FY=525.0
CAMERA_CX=319.5
CAMERA_CY=239.5
CAMERA_K1=0.0
CAMERA_K2=0.0
CAMERA_P1=0.0
CAMERA_P2=0.0
CAMERA_K3=0.0
EOF
    echo "Created .env file with default values"
fi

echo "Development environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To run SLAM pipeline:"
echo "  make run-slam"
echo ""
echo "To build Docker containers:"
echo "  make docker-build"
