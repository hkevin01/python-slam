# Installation Guide

This guide provides detailed instructions for installing Python-SLAM on various platforms.

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: 64-bit processor (x86_64 or ARM64)
- Memory: 4GB RAM
- Storage: 2GB available disk space
- Display: 1024x768 resolution

**Recommended Requirements:**
- CPU: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- Memory: 8GB+ RAM
- GPU: NVIDIA GPU with CUDA support, AMD GPU with ROCm, or Apple Silicon
- Storage: 5GB+ available disk space (SSD recommended)
- Display: 1920x1080+ resolution

### Software Requirements

**Operating Systems:**
- Linux: Ubuntu 20.04+, CentOS 8+, Arch Linux
- macOS: 10.15+ (Catalina or later)
- Windows: Windows 10/11 (with WSL2 recommended)

**Python Version:**
- Python 3.8 or higher
- pip (Python package installer)

## Quick Installation

### Automated Installation (Recommended)

The easiest way to install Python-SLAM is using the provided installation script:

```bash
# Clone the repository
git clone https://github.com/your-username/python-slam.git
cd python-slam

# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

The script will:
1. Detect your system configuration
2. Install system dependencies
3. Create a Python virtual environment
4. Install Python packages
5. Configure GPU support (if available)
6. Set up ROS2 integration (optional)
7. Run initial configuration

### Manual Installation

If you prefer manual installation or need custom configuration:

#### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    build-essential cmake git \
    libgl1-mesa-dev libglu1-mesa-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    qtbase5-dev qt5-qmake \
    libeigen3-dev libopencv-dev
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install -y \
    python3 python3-pip \
    gcc gcc-c++ cmake git \
    mesa-libGL-devel mesa-libGLU-devel \
    libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel \
    qt5-qtbase-devel \
    eigen3-devel opencv-devel
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3 cmake git qt@5 eigen opencv
```

#### 2. Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv_python_slam

# Activate virtual environment
source venv_python_slam/bin/activate  # Linux/macOS
# or
venv_python_slam\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### 3. Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install -r requirements-optional.txt

# Install development dependencies (if developing)
pip install -r requirements-dev.txt
```

#### 4. GPU Support (Optional)

**NVIDIA CUDA:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CuPy for CUDA operations
pip install cupy-cuda11x
```

**AMD ROCm:**
```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

**Apple Metal:**
```bash
# PyTorch with Metal Performance Shaders (included in standard PyTorch on macOS)
pip install torch torchvision torchaudio
```

#### 5. Configuration

```bash
# Run configuration wizard
python configure.py

# Or manually create configuration
cp config/default_config.json config/config.json
# Edit config/config.json as needed
```

## Platform-Specific Instructions

### Linux Installation

#### Ubuntu 20.04/22.04

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.9 python3.9-venv python3.9-dev \
    build-essential cmake git \
    libgl1-mesa-dev libglu1-mesa-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    qt6-base-dev qt6-tools-dev \
    libeigen3-dev libopencv-dev \
    pkg-config

# For NVIDIA GPU support
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Clone and install
git clone https://github.com/your-username/python-slam.git
cd python-slam
./install.sh
```

#### Arch Linux

```bash
# Install dependencies
sudo pacman -S python python-pip cmake git \
    mesa qt6-base qt6-tools \
    eigen opencv \
    cuda cuda-tools  # For NVIDIA GPU

# Clone and install
git clone https://github.com/your-username/python-slam.git
cd python-slam
./install.sh
```

### macOS Installation

#### Using Homebrew

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew dependencies
brew install python@3.11 cmake git qt@6 eigen opencv

# For Apple Silicon Macs, ensure native Python
arch -arm64 brew install python@3.11

# Clone and install
git clone https://github.com/your-username/python-slam.git
cd python-slam
./install.sh
```

#### Using MacPorts

```bash
# Install MacPorts dependencies
sudo port install python311 py311-pip cmake git-tools \
    qt6 eigen3 opencv4

# Clone and install
git clone https://github.com/your-username/python-slam.git
cd python-slam
./install.sh
```

### Windows Installation

#### Using WSL2 (Recommended)

```bash
# Install WSL2 and Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Ubuntu installation instructions
sudo apt update
# ... (follow Ubuntu steps above)
```

#### Native Windows Installation

```powershell
# Install Python from python.org or Microsoft Store
# Install Visual Studio Build Tools
# Install Qt6 from qt.io

# Clone repository
git clone https://github.com/your-username/python-slam.git
cd python-slam

# Install Python dependencies
pip install -r requirements.txt

# Manual configuration required for Windows
python configure.py
```

## ROS2 Integration (Optional)

If you want to use ROS2 integration features:

### ROS2 Humble Installation

**Ubuntu 22.04:**
```bash
# Add ROS2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2
sudo apt update
sudo apt install -y ros-humble-desktop-full

# Install additional packages
sudo apt install -y ros-humble-nav2-bringup ros-humble-slam-toolbox

# Source ROS2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**Other Platforms:**
Follow the official ROS2 installation guide for your platform: https://docs.ros.org/en/humble/Installation.html

### Configure ROS2 Integration

```bash
# Enable ROS2 in configuration
python configure.py --enable-ros2

# Or manually edit config
# Set "enable_integration": true in ros2 section of config.json
```

## Verification

After installation, verify that everything is working:

```bash
# Activate virtual environment
source venv_python_slam/bin/activate

# Check dependencies
python tests/run_tests.py --check-deps

# Run basic tests
python tests/run_tests.py --categories comprehensive

# Test GUI (if display available)
python python_slam_main.py --mode gui --test

# Test GPU acceleration
python -c "from python_slam.gpu_acceleration.gpu_detector import GPUDetector; print(GPUDetector().detect_all_gpus())"
```

Expected output should show:
- ✓ All dependencies available
- ✓ Tests passing
- ✓ GPU detection (if GPU available)
- ✓ GUI launching successfully

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Problem: "ModuleNotFoundError: No module named 'python_slam'"
# Solution: Ensure virtual environment is activated and PYTHONPATH is set
source venv_python_slam/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Qt/GUI Issues
```bash
# Problem: "qt.qpa.plugin: Could not load the Qt platform plugin"
# Solution: Install additional Qt dependencies
sudo apt install -y qt6-qpa-plugins qt6-wayland  # Linux
brew install qt6                                  # macOS
```

#### 3. GPU Not Detected
```bash
# Problem: GPU not detected despite being available
# Solution: Check drivers and libraries
nvidia-smi                    # NVIDIA
rocm-smi                      # AMD
system_profiler SPDisplaysDataType  # macOS

# Reinstall GPU-specific PyTorch
pip uninstall torch torchvision torchaudio
# Follow GPU-specific installation above
```

#### 4. OpenGL Issues
```bash
# Problem: OpenGL errors in 3D visualization
# Solution: Update graphics drivers and install Mesa
sudo apt install -y mesa-utils libgl1-mesa-glx  # Linux
glxinfo | grep "OpenGL version"                  # Check OpenGL version
```

#### 5. Permission Issues
```bash
# Problem: Permission denied errors
# Solution: Fix permissions
chmod +x install.sh configure.py
sudo chown -R $USER:$USER ~/.local/share/python-slam
```

### Performance Issues

#### Memory Usage
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
print(f'Total memory: {psutil.virtual_memory().total / 1024**3:.1f} GB')
"

# Reduce memory usage in config
# Set lower values for:
# - slam.max_features
# - gpu.memory_limit_mb
# - benchmarking.max_parallel_jobs
```

#### CPU Performance
```bash
# Check CPU info
python -c "
import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'CPU frequency: {psutil.cpu_freq().current:.0f} MHz')
"

# Enable CPU optimization in config
# Set embedded.enable_optimization: true
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look in `logs/` directory for error messages
2. **Run diagnostics**: `python python_slam_main.py --diagnose`
3. **Check GitHub Issues**: Search for similar problems
4. **Create an Issue**: Provide system info, error messages, and reproduction steps

### System Information Script

Create a system info script for troubleshooting:

```bash
# Save as system_info.py
python -c "
import sys, platform, subprocess
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch: Not installed')

try:
    from PyQt6.QtCore import QT_VERSION_STR
    print(f'PyQt6: {QT_VERSION_STR}')
except ImportError:
    try:
        import PySide6
        print(f'PySide6: {PySide6.__version__}')
    except ImportError:
        print('Qt: Not available')
"
```

This will help diagnose system-specific issues.
