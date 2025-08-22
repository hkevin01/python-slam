#!/bin/bash
"""
Python-SLAM Installation Script

This script sets up the complete Python-SLAM environment with all dependencies
and optional components (GPU acceleration, ROS2, embedded optimization).
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_SLAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PYTHON_SLAM_DIR/venv"
INSTALL_LOG="$PYTHON_SLAM_DIR/install.log"

# System detection
ARCH=$(uname -m)
OS=$(uname -s)
DISTRO=""

if [[ "$OS" == "Linux" ]]; then
    if command -v lsb_release &> /dev/null; then
        DISTRO=$(lsb_release -si)
    elif [[ -f /etc/os-release ]]; then
        DISTRO=$(grep ^ID= /etc/os-release | cut -d= -f2 | tr -d '"')
    fi
elif [[ "$OS" == "Darwin" ]]; then
    DISTRO="macOS"
fi

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_command() {
    echo "[$(date)] $1" >> "$INSTALL_LOG"
    eval "$1" >> "$INSTALL_LOG" 2>&1
}

check_requirements() {
    echo_info "Checking system requirements..."
    
    # Check Python 3.8+
    if ! command -v python3 &> /dev/null; then
        echo_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
        echo_error "Python 3.8+ is required (found $PYTHON_VERSION)"
        exit 1
    fi
    
    echo_success "Python $PYTHON_VERSION found"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        echo_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        echo_warning "Git not found - some features may not work"
    fi
    
    echo_success "System requirements check passed"
}

install_system_dependencies() {
    echo_info "Installing system dependencies..."
    
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$DISTRO" == "ubuntu" ]] || [[ "$DISTRO" == "debian" ]]; then
            echo_info "Installing dependencies for Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                pkg-config \
                libopencv-dev \
                python3-dev \
                python3-venv \
                python3-pip \
                libgl1-mesa-dev \
                libglu1-mesa-dev \
                freeglut3-dev \
                libeigen3-dev \
                libsuitesparse-dev \
                qtbase5-dev \
                qttools5-dev \
                libpython3-dev
                
        elif [[ "$DISTRO" == "fedora" ]] || [[ "$DISTRO" == "centos" ]] || [[ "$DISTRO" == "rhel" ]]; then
            echo_info "Installing dependencies for Fedora/CentOS/RHEL..."
            sudo dnf install -y \
                gcc-c++ \
                cmake \
                pkgconfig \
                opencv-devel \
                python3-devel \
                mesa-libGL-devel \
                mesa-libGLU-devel \
                freeglut-devel \
                eigen3-devel \
                suitesparse-devel \
                qt5-qtbase-devel \
                qt5-qttools-devel
        else
            echo_warning "Unknown Linux distribution: $DISTRO"
            echo_warning "Please install system dependencies manually"
        fi
        
    elif [[ "$OS" == "Darwin" ]]; then
        echo_info "Installing dependencies for macOS..."
        if ! command -v brew &> /dev/null; then
            echo_error "Homebrew is required for macOS installation"
            echo_info "Please install Homebrew from https://brew.sh"
            exit 1
        fi
        
        brew install \
            cmake \
            pkg-config \
            opencv \
            eigen \
            suite-sparse \
            qt@5
    else
        echo_error "Unsupported operating system: $OS"
        exit 1
    fi
    
    echo_success "System dependencies installed"
}

create_virtual_environment() {
    echo_info "Creating Python virtual environment..."
    
    if [[ -d "$VENV_DIR" ]]; then
        echo_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove existing environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            echo_info "Using existing virtual environment"
            return
        fi
    fi
    
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    echo_success "Virtual environment created at $VENV_DIR"
}

install_python_dependencies() {
    echo_info "Installing Python dependencies..."
    
    source "$VENV_DIR/bin/activate"
    
    # Core dependencies
    pip install numpy scipy matplotlib pandas
    pip install opencv-python opencv-contrib-python
    pip install scikit-learn scikit-image
    pip install pillow
    pip install psutil
    
    # GUI dependencies
    echo_info "Installing GUI dependencies..."
    pip install PyQt6 || {
        echo_warning "PyQt6 installation failed, trying PySide6..."
        pip install PySide6
    }
    pip install PyOpenGL PyOpenGL_accelerate
    
    # 3D visualization
    pip install vtk
    pip install mayavi || echo_warning "Mayavi installation failed (optional)"
    
    # Machine learning
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || {
        echo_warning "PyTorch installation failed, installing CPU version..."
        pip install torch torchvision torchaudio
    }
    
    # Optional GPU acceleration
    echo_info "Installing optional GPU dependencies..."
    if [[ "$ARCH" == "x86_64" ]]; then
        # Try CUDA version of PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
            echo_warning "CUDA PyTorch installation failed"
        
        # CuPy for CUDA
        pip install cupy-cuda11x || echo_warning "CuPy installation failed"
    fi
    
    # ROCm for AMD GPUs (Linux only)
    if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2 || \
            echo_warning "ROCm PyTorch installation failed"
    fi
    
    # Scientific computing
    pip install numba
    pip install cython
    
    # Development dependencies
    pip install pytest pytest-cov
    pip install black flake8 mypy
    pip install sphinx sphinx-rtd-theme
    
    echo_success "Python dependencies installed"
}

install_ros2_dependencies() {
    if [[ "$1" != "--enable-ros2" ]]; then
        echo_info "Skipping ROS2 installation (use --enable-ros2 to enable)"
        return
    fi
    
    echo_info "Installing ROS2 dependencies..."
    
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$DISTRO" == "ubuntu" ]]; then
            # Add ROS2 repository
            sudo apt update && sudo apt install curl gnupg lsb-release
            sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
            
            sudo apt update
            sudo apt install -y ros-humble-desktop ros-humble-nav2-bringup
            
            # Install Python ROS2 packages
            source "$VENV_DIR/bin/activate"
            pip install rclpy nav2-msgs geometry-msgs sensor-msgs
        else
            echo_warning "ROS2 installation not automated for $DISTRO"
            echo_info "Please install ROS2 manually: https://docs.ros.org/en/humble/Installation.html"
        fi
    else
        echo_warning "ROS2 installation not available for $OS"
    fi
    
    echo_success "ROS2 dependencies installed"
}

setup_environment() {
    echo_info "Setting up environment..."
    
    # Create environment activation script
    cat > "$PYTHON_SLAM_DIR/activate_env.sh" << 'EOF'
#!/bin/bash
# Python-SLAM Environment Activation Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Python-SLAM environment activated"
    
    # Add src to Python path
    export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
    
    # ROS2 setup (if available)
    if [[ -f "/opt/ros/humble/setup.bash" ]]; then
        source /opt/ros/humble/setup.bash
        echo "ROS2 environment sourced"
    fi
else
    echo "Virtual environment not found at $VENV_DIR"
    echo "Please run install.sh first"
fi
EOF
    
    chmod +x "$PYTHON_SLAM_DIR/activate_env.sh"
    
    # Create desktop launcher (Linux only)
    if [[ "$OS" == "Linux" ]] && command -v desktop-file-install &> /dev/null; then
        cat > "$PYTHON_SLAM_DIR/python-slam.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Python-SLAM
Comment=Comprehensive SLAM System with GUI
Icon=$PYTHON_SLAM_DIR/assets/icon.png
Exec=$PYTHON_SLAM_DIR/python_slam_main.py --mode gui
Terminal=false
Categories=Science;Education;
EOF
    fi
    
    # Create run scripts
    cat > "$PYTHON_SLAM_DIR/run_gui.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"
python "$SCRIPT_DIR/python_slam_main.py" --mode gui "$@"
EOF
    
    cat > "$PYTHON_SLAM_DIR/run_benchmark.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"
python "$SCRIPT_DIR/python_slam_main.py" --mode benchmark "$@"
EOF
    
    cat > "$PYTHON_SLAM_DIR/run_headless.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"
python "$SCRIPT_DIR/python_slam_main.py" --mode headless "$@"
EOF
    
    chmod +x "$PYTHON_SLAM_DIR/run_"*.sh
    
    echo_success "Environment setup completed"
}

run_tests() {
    echo_info "Running installation tests..."
    
    source "$VENV_DIR/bin/activate"
    
    # Test Python imports
    python3 -c "
import sys
sys.path.insert(0, '$PYTHON_SLAM_DIR/src')

try:
    from python_slam.gui.main_window import SlamMainWindow
    print('GUI components: OK')
except Exception as e:
    print(f'GUI components: FAILED - {e}')

try:
    from python_slam.benchmarking.benchmark_runner import BenchmarkRunner
    print('Benchmarking: OK')
except Exception as e:
    print(f'Benchmarking: FAILED - {e}')

try:
    from python_slam.gpu_acceleration import GPUManager
    print('GPU acceleration: OK')
except Exception as e:
    print(f'GPU acceleration: FAILED - {e}')

try:
    from python_slam.embedded_optimization.arm_optimization import ARMOptimizer
    print('ARM optimization: OK')
except Exception as e:
    print(f'ARM optimization: FAILED - {e}')

print('Installation test completed')
"
    
    echo_success "Installation tests completed"
}

print_summary() {
    echo
    echo_success "Python-SLAM installation completed!"
    echo
    echo_info "Installation Summary:"
    echo "  - Installation directory: $PYTHON_SLAM_DIR"
    echo "  - Virtual environment: $VENV_DIR"
    echo "  - Architecture: $ARCH"
    echo "  - Operating system: $OS ($DISTRO)"
    echo
    echo_info "To get started:"
    echo "  1. Activate environment: source activate_env.sh"
    echo "  2. Run GUI: ./run_gui.sh"
    echo "  3. Run benchmarks: ./run_benchmark.sh"
    echo "  4. Run headless: ./run_headless.sh"
    echo
    echo_info "For more information, see README.md"
    echo
}

# Main installation function
main() {
    echo_info "Python-SLAM Installation Script"
    echo_info "==============================="
    echo
    
    # Parse arguments
    ENABLE_ROS2=""
    SKIP_SYSTEM_DEPS=""
    SKIP_TESTS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --enable-ros2)
                ENABLE_ROS2="--enable-ros2"
                shift
                ;;
            --skip-system-deps)
                SKIP_SYSTEM_DEPS="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --enable-ros2       Install ROS2 dependencies"
                echo "  --skip-system-deps  Skip system dependency installation"
                echo "  --skip-tests        Skip installation tests"
                echo "  -h, --help          Show this help"
                exit 0
                ;;
            *)
                echo_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Create log file
    echo "Python-SLAM Installation Log - $(date)" > "$INSTALL_LOG"
    
    # Run installation steps
    check_requirements
    
    if [[ "$SKIP_SYSTEM_DEPS" != "true" ]]; then
        install_system_dependencies
    fi
    
    create_virtual_environment
    install_python_dependencies
    install_ros2_dependencies $ENABLE_ROS2
    setup_environment
    
    if [[ "$SKIP_TESTS" != "true" ]]; then
        run_tests
    fi
    
    print_summary
}

# Run main function
main "$@"
