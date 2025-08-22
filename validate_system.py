#!/usr/bin/env python3
"""
Quick validation script for Python-SLAM system.
"""

import sys
import os
import traceback

def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_core_dependencies():
    """Check core dependencies."""
    dependencies = {
        "numpy": "NumPy",
        "torch": "PyTorch", 
        "matplotlib": "Matplotlib"
    }
    
    results = {}
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} available")
            results[module] = True
        except ImportError:
            print(f"❌ {name} not available")
            results[module] = False
    
    return results

def check_gui_dependencies():
    """Check GUI dependencies."""
    gui_available = False
    
    try:
        import PyQt6
        print(f"✅ PyQt6 {PyQt6.QtCore.PYQT_VERSION_STR} available")
        gui_available = True
    except ImportError:
        print("⚠️  PyQt6 not available")
    
    try:
        import PySide6
        print(f"✅ PySide6 {PySide6.__version__} available")
        gui_available = True
    except ImportError:
        print("⚠️  PySide6 not available")
    
    if not gui_available:
        print("❌ No GUI backend available")
    
    return gui_available

def check_gpu_support():
    """Check GPU support."""
    gpu_available = False
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} device(s)")
            gpu_available = True
        else:
            print("⚠️  CUDA not available")
            
        # Check if MPS (Metal Performance Shaders) is available on macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Metal Performance Shaders available")
            gpu_available = True
    except ImportError:
        print("❌ PyTorch not available for GPU check")
    
    if not gpu_available:
        print("⚠️  No GPU acceleration available (will use CPU)")
    
    return gpu_available

def check_project_structure():
    """Check project structure."""
    expected_dirs = [
        "src",
        "tests", 
        "docs",
        "config"
    ]
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    for dir_name in expected_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    return True

def test_basic_imports():
    """Test basic imports."""
    print("\nTesting basic imports...")
    
    # Add src to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    sys.path.insert(0, src_path)
    
    test_imports = [
        ("python_slam.gpu_acceleration.gpu_detector", "GPUDetector"),
        ("python_slam.benchmarking.benchmark_metrics", "ProcessingMetrics"),
        ("python_slam.gui.utils", "MaterialDesignManager"),
    ]
    
    success_count = 0
    for module, class_name in test_imports:
        try:
            module_obj = __import__(module, fromlist=[class_name])
            getattr(module_obj, class_name)
            print(f"✅ {module}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module}.{class_name}: {e}")
    
    return success_count == len(test_imports)

def test_main_system():
    """Test main system import."""
    print("\nTesting main system...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    try:
        import python_slam_main
        print("✅ Main system module imports successfully")
        
        # Test configuration creation
        config = python_slam_main.create_default_config()
        print("✅ Default configuration creation works")
        
        return True
    except Exception as e:
        print(f"❌ Main system test failed: {e}")
        return False

def main():
    """Run validation."""
    print("Python-SLAM System Validation")
    print("=" * 50)
    
    results = []
    
    # Check Python version
    results.append(check_python_version())
    
    print("\n" + "-" * 30)
    print("Core Dependencies")
    print("-" * 30)
    core_deps = check_core_dependencies()
    results.append(all(core_deps.values()))
    
    print("\n" + "-" * 30)
    print("GUI Dependencies")
    print("-" * 30)
    gui_available = check_gui_dependencies()
    # GUI is optional, so don't fail validation
    
    print("\n" + "-" * 30)
    print("GPU Support")
    print("-" * 30)
    gpu_available = check_gpu_support()
    # GPU is optional, so don't fail validation
    
    print("\n" + "-" * 30)
    print("Project Structure")
    print("-" * 30)
    results.append(check_project_structure())
    
    print("\n" + "-" * 30)
    print("Module Imports")
    print("-" * 30)
    results.append(test_basic_imports())
    
    print("\n" + "-" * 30)
    print("Main System")
    print("-" * 30)
    results.append(test_main_system())
    
    print("\n" + "=" * 50)
    print("Validation Summary")
    print("=" * 50)
    
    if all(results):
        print("✅ All critical components validated successfully!")
        print("🚀 Python-SLAM is ready to use!")
        
        if gui_available:
            print("💻 GUI components available")
        else:
            print("⚠️  GUI components not available (headless mode only)")
            
        if gpu_available:
            print("🔥 GPU acceleration available")
        else:
            print("⚠️  GPU acceleration not available (CPU mode)")
            
        return True
    else:
        print("❌ Some critical components failed validation")
        print("Please check the installation and try again")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
