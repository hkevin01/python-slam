#!/usr/bin/env python3
"""
Python-SLAM Configuration Wizard

Interactive configuration wizard for setting up Python-SLAM with optimal
settings for your system and use case.
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colored_print(text: str, color: str = Colors.END) -> None:
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.END}")

def get_user_input(prompt: str, default: str = "", validation_func=None) -> str:
    """Get user input with validation."""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{prompt}: ").strip()

        if validation_func:
            if validation_func(user_input):
                return user_input
            else:
                colored_print("Invalid input. Please try again.", Colors.RED)
        else:
            return user_input

def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_text = "Y/n" if default else "y/N"
    response = get_user_input(f"{prompt} ({default_text})", "y" if default else "n")
    return response.lower() in ['y', 'yes', '1', 'true'] if response else default

def detect_system_info() -> Dict[str, Any]:
    """Detect system information."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # Detect GPU
    gpu_info = detect_gpu()
    info.update(gpu_info)

    # Detect memory
    try:
        import psutil
        info["memory_gb"] = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        info["memory_gb"] = 8  # Default assumption

    return info

def detect_gpu() -> Dict[str, Any]:
    """Detect available GPU hardware."""
    gpu_info = {
        "nvidia_gpu": False,
        "amd_gpu": False,
        "apple_gpu": False,
        "gpu_names": []
    }

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info["nvidia_gpu"] = True
            gpu_info["gpu_names"].extend(result.stdout.strip().split('\n'))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for AMD GPU
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info["amd_gpu"] = True
            # Parse rocm-smi output for GPU names
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'GPU' in line:
                    parts = line.split()
                    if len(parts) > 2:
                        gpu_info["gpu_names"].append(' '.join(parts[2:]))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for Apple GPU (macOS)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "Metal" in result.stdout:
                gpu_info["apple_gpu"] = True
                # Extract Apple GPU names
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Chipset Model:' in line:
                        gpu_name = line.split(':')[1].strip()
                        gpu_info["gpu_names"].append(gpu_name)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return gpu_info

def configure_general_settings(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Configure general Python-SLAM settings."""
    colored_print("\n=== General Configuration ===", Colors.BOLD)

    config = {}

    # System optimization level
    print("\nOptimization Level:")
    print("1. Performance (maximum speed, higher resource usage)")
    print("2. Balanced (good performance with reasonable resource usage)")
    print("3. Power Save (minimize resource usage for embedded/battery systems)")

    optimization_choice = get_user_input(
        "Choose optimization level (1-3)",
        "2",
        lambda x: x in ['1', '2', '3']
    )

    optimization_map = {
        '1': 'performance',
        '2': 'balanced',
        '3': 'power_save'
    }
    config["optimization_level"] = optimization_map[optimization_choice]

    # Logging level
    log_level = get_user_input(
        "Logging level (DEBUG/INFO/WARNING/ERROR)",
        "INFO",
        lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    config["log_level"] = log_level.upper()

    # Enable components
    config["enable_gui"] = get_yes_no("Enable GUI interface", True)
    config["enable_benchmarking"] = get_yes_no("Enable benchmarking system", True)
    config["enable_real_time_monitoring"] = get_yes_no("Enable real-time performance monitoring", True)

    return config

def configure_slam_settings() -> Dict[str, Any]:
    """Configure SLAM algorithm settings."""
    colored_print("\n=== SLAM Configuration ===", Colors.BOLD)

    config = {}

    # Feature detector
    print("\nFeature Detector:")
    print("1. SIFT (high quality, slower)")
    print("2. ORB (fast, good for real-time)")
    print("3. SURF (balanced quality and speed)")
    print("4. FAST (very fast, basic features)")

    detector_choice = get_user_input(
        "Choose feature detector (1-4)",
        "2",
        lambda x: x in ['1', '2', '3', '4']
    )

    detector_map = {
        '1': 'SIFT',
        '2': 'ORB',
        '3': 'SURF',
        '4': 'FAST'
    }
    config["feature_detector"] = detector_map[detector_choice]

    # Maximum features
    max_features = get_user_input(
        "Maximum features to extract per frame",
        "1000",
        lambda x: x.isdigit() and int(x) > 0
    )
    config["max_features"] = int(max_features)

    # Loop closure
    config["enable_loop_closure"] = get_yes_no("Enable loop closure detection", True)

    # Bundle adjustment
    config["enable_bundle_adjustment"] = get_yes_no("Enable bundle adjustment", True)

    if config["enable_bundle_adjustment"]:
        ba_iterations = get_user_input(
            "Bundle adjustment iterations",
            "100",
            lambda x: x.isdigit() and int(x) > 0
        )
        config["bundle_adjustment_iterations"] = int(ba_iterations)

    return config

def configure_gpu_settings(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Configure GPU acceleration settings."""
    colored_print("\n=== GPU Acceleration Configuration ===", Colors.BOLD)

    config = {}

    # Check if any GPU is available
    has_gpu = system_info.get("nvidia_gpu") or system_info.get("amd_gpu") or system_info.get("apple_gpu")

    if not has_gpu:
        colored_print("No GPU detected. GPU acceleration will be disabled.", Colors.YELLOW)
        config["enable_gpu"] = False
        return config

    # Display detected GPUs
    if system_info.get("gpu_names"):
        print(f"\nDetected GPUs: {', '.join(system_info['gpu_names'])}")

    config["enable_gpu"] = get_yes_no("Enable GPU acceleration", True)

    if not config["enable_gpu"]:
        return config

    # Preferred backend
    available_backends = []
    if system_info.get("nvidia_gpu"):
        available_backends.append("CUDA")
    if system_info.get("amd_gpu"):
        available_backends.append("ROCm")
    if system_info.get("apple_gpu"):
        available_backends.append("Metal")

    if len(available_backends) > 1:
        print(f"\nAvailable GPU backends: {', '.join(available_backends)}")
        backend_choice = get_user_input(
            f"Preferred backend ({'/'.join(available_backends)}) or AUTO for automatic",
            "AUTO"
        )
        if backend_choice.upper() != "AUTO":
            config["preferred_backend"] = backend_choice

    # Mixed precision
    config["enable_mixed_precision"] = get_yes_no("Enable mixed precision (FP16/FP32)", True)

    # Memory management
    memory_limit = get_user_input(
        "GPU memory limit (MB, 0 for unlimited)",
        "0",
        lambda x: x.isdigit()
    )
    if int(memory_limit) > 0:
        config["memory_limit_mb"] = int(memory_limit)

    return config

def configure_ros2_settings() -> Dict[str, Any]:
    """Configure ROS2 integration settings."""
    colored_print("\n=== ROS2 Integration Configuration ===", Colors.BOLD)

    config = {}

    # Check if ROS2 is available
    ros2_available = False
    try:
        result = subprocess.run(['ros2', '--version'],
                              capture_output=True, text=True, timeout=5)
        ros2_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if not ros2_available:
        colored_print("ROS2 not detected. ROS2 integration will be disabled.", Colors.YELLOW)
        config["enable_ros2"] = False
        return config

    config["enable_ros2"] = get_yes_no("Enable ROS2 Nav2 integration", False)

    if not config["enable_ros2"]:
        return config

    # ROS2 domain ID
    domain_id = get_user_input(
        "ROS2 domain ID",
        "0",
        lambda x: x.isdigit() and 0 <= int(x) <= 101
    )
    config["ros_domain_id"] = int(domain_id)

    # Navigation settings
    config["enable_navigation"] = get_yes_no("Enable autonomous navigation", True)
    config["enable_mapping"] = get_yes_no("Enable real-time mapping", True)
    config["enable_localization"] = get_yes_no("Enable localization", True)

    return config

def configure_embedded_settings(system_info: Dict[str, Any]) -> Dict[str, Any]:
    """Configure embedded system optimization."""
    colored_print("\n=== Embedded System Configuration ===", Colors.BOLD)

    config = {}

    # Check if running on ARM
    is_arm = system_info["arch"].lower() in ['arm', 'aarch64', 'armv7', 'armv8']

    if is_arm:
        colored_print(f"ARM architecture detected: {system_info['arch']}", Colors.GREEN)
        config["enable_arm_optimization"] = get_yes_no("Enable ARM-specific optimizations", True)
    else:
        colored_print(f"Non-ARM architecture: {system_info['arch']}", Colors.BLUE)
        config["enable_arm_optimization"] = get_yes_no("Enable ARM optimizations (for cross-compilation)", False)

    if not config["enable_arm_optimization"]:
        return config

    # NEON SIMD
    config["enable_neon"] = get_yes_no("Enable ARM NEON SIMD optimizations", True)

    # Cache optimization
    config["enable_cache_optimization"] = get_yes_no("Enable cache optimization", True)

    # Power management
    config["enable_power_management"] = get_yes_no("Enable power management", True)

    if config["enable_power_management"]:
        power_budget = get_user_input(
            "Power budget (watts, 0 for unlimited)",
            "5.0",
            lambda x: x.replace('.', '').isdigit()
        )
        if float(power_budget) > 0:
            config["power_budget_watts"] = float(power_budget)

    # Memory constraints
    memory_limit = get_user_input(
        f"Memory limit (MB, detected: {system_info.get('memory_gb', 'unknown')}GB)",
        "512",
        lambda x: x.isdigit() and int(x) > 0
    )
    config["memory_limit_mb"] = int(memory_limit)

    # Real-time constraints
    config["enable_real_time"] = get_yes_no("Enable real-time scheduling", False)

    if config["enable_real_time"]:
        max_processing_time = get_user_input(
            "Maximum processing time per frame (ms)",
            "50",
            lambda x: x.isdigit() and int(x) > 0
        )
        config["max_processing_time_ms"] = int(max_processing_time)

        target_fps = get_user_input(
            "Target frame rate (FPS)",
            "20",
            lambda x: x.isdigit() and int(x) > 0
        )
        config["target_fps"] = int(target_fps)

    return config

def configure_gui_settings() -> Dict[str, Any]:
    """Configure GUI settings."""
    colored_print("\n=== GUI Configuration ===", Colors.BOLD)

    config = {}

    # Theme
    print("\nGUI Theme:")
    print("1. Dark (modern dark theme)")
    print("2. Light (clean light theme)")
    print("3. Auto (follow system theme)")

    theme_choice = get_user_input(
        "Choose theme (1-3)",
        "1",
        lambda x: x in ['1', '2', '3']
    )

    theme_map = {
        '1': 'dark',
        '2': 'light',
        '3': 'auto'
    }
    config["theme"] = theme_map[theme_choice]

    # Real-time updates
    config["enable_real_time_updates"] = get_yes_no("Enable real-time visualization updates", True)

    # Update frequency
    if config["enable_real_time_updates"]:
        update_rate = get_user_input(
            "GUI update rate (Hz)",
            "30",
            lambda x: x.isdigit() and 1 <= int(x) <= 60
        )
        config["update_rate_hz"] = int(update_rate)

    # 3D visualization
    config["enable_3d_visualization"] = get_yes_no("Enable 3D visualization", True)

    if config["enable_3d_visualization"]:
        config["enable_point_cloud_rendering"] = get_yes_no("Enable point cloud rendering", True)
        config["enable_trajectory_visualization"] = get_yes_no("Enable trajectory visualization", True)
        config["enable_keyframe_visualization"] = get_yes_no("Enable keyframe visualization", True)

    # Performance monitoring
    config["enable_performance_dashboard"] = get_yes_no("Enable performance monitoring dashboard", True)

    return config

def configure_benchmarking_settings() -> Dict[str, Any]:
    """Configure benchmarking settings."""
    colored_print("\n=== Benchmarking Configuration ===", Colors.BOLD)

    config = {}

    # Enable parallel execution
    config["enable_parallel_execution"] = get_yes_no("Enable parallel benchmark execution", True)

    # Timeout
    timeout = get_user_input(
        "Benchmark timeout (seconds)",
        "300",
        lambda x: x.isdigit() and int(x) > 0
    )
    config["timeout_seconds"] = int(timeout)

    # Output format
    print("\nBenchmark output formats:")
    print("1. JSON (machine readable)")
    print("2. CSV (spreadsheet compatible)")
    print("3. HTML (web report)")
    print("4. All formats")

    format_choice = get_user_input(
        "Choose output format (1-4)",
        "4",
        lambda x: x in ['1', '2', '3', '4']
    )

    format_map = {
        '1': ['json'],
        '2': ['csv'],
        '3': ['html'],
        '4': ['json', 'csv', 'html']
    }
    config["output_formats"] = format_map[format_choice]

    # Evaluation metrics
    config["enable_ate_evaluation"] = get_yes_no("Enable ATE (Absolute Trajectory Error) evaluation", True)
    config["enable_rpe_evaluation"] = get_yes_no("Enable RPE (Relative Pose Error) evaluation", True)
    config["enable_processing_metrics"] = get_yes_no("Enable processing performance metrics", True)
    config["enable_memory_profiling"] = get_yes_no("Enable memory profiling", True)

    return config

def save_configuration(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to file."""
    try:
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        colored_print(f"Configuration saved to {config_path}", Colors.GREEN)
        return True

    except Exception as e:
        colored_print(f"Failed to save configuration: {e}", Colors.RED)
        return False

def load_existing_configuration(config_path: str) -> Optional[Dict[str, Any]]:
    """Load existing configuration if available."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            colored_print(f"Loaded existing configuration from {config_path}", Colors.BLUE)
            return config
    except Exception as e:
        colored_print(f"Failed to load existing configuration: {e}", Colors.YELLOW)

    return None

def main():
    """Main configuration wizard."""
    colored_print("Python-SLAM Configuration Wizard", Colors.BOLD + Colors.BLUE)
    colored_print("=" * 40, Colors.BLUE)

    # Detect system
    colored_print("\nDetecting system configuration...", Colors.BLUE)
    system_info = detect_system_info()

    print(f"Operating System: {system_info['os']} ({system_info['arch']})")
    print(f"Python Version: {system_info['python_version']}")
    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"Memory: {system_info.get('memory_gb', 'unknown')} GB")

    if system_info.get('gpu_names'):
        print(f"GPUs: {', '.join(system_info['gpu_names'])}")
    else:
        print("GPUs: None detected")

    # Configuration file path
    config_dir = Path.cwd() / "config"
    config_path = config_dir / "python_slam_config.json"

    # Check for existing configuration
    existing_config = load_existing_configuration(str(config_path))
    if existing_config:
        use_existing = get_yes_no("Use existing configuration as starting point", True)
        if not use_existing:
            existing_config = None

    # Start configuration process
    config = existing_config or {}

    # Configure each section
    general_config = configure_general_settings(system_info)
    config.update(general_config)

    slam_config = configure_slam_settings()
    config["slam"] = slam_config

    gpu_config = configure_gpu_settings(system_info)
    config["gpu"] = gpu_config

    ros2_config = configure_ros2_settings()
    config["ros2"] = ros2_config

    embedded_config = configure_embedded_settings(system_info)
    config["embedded"] = embedded_config

    if config.get("enable_gui", True):
        gui_config = configure_gui_settings()
        config["gui"] = gui_config

    if config.get("enable_benchmarking", True):
        benchmark_config = configure_benchmarking_settings()
        config["benchmarking"] = benchmark_config

    # Save configuration
    colored_print("\n=== Configuration Summary ===", Colors.BOLD)
    print(json.dumps(config, indent=2))

    save_config = get_yes_no(f"\nSave configuration to {config_path}", True)

    if save_config:
        if save_configuration(config, str(config_path)):
            colored_print("\nConfiguration wizard completed successfully!", Colors.GREEN)
            colored_print(f"To use this configuration, run:", Colors.BLUE)
            colored_print(f"python src/python_slam_main.py --config {config_path}", Colors.BOLD)
        else:
            colored_print("\nConfiguration wizard completed with errors.", Colors.RED)
    else:
        colored_print("\nConfiguration not saved.", Colors.YELLOW)
        print("Configuration JSON:")
        print(json.dumps(config, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        colored_print("\n\nConfiguration wizard cancelled by user.", Colors.YELLOW)
        sys.exit(0)
    except Exception as e:
        colored_print(f"\nConfiguration wizard failed: {e}", Colors.RED)
        sys.exit(1)
