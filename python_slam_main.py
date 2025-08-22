#!/usr/bin/env python3
"""
Python-SLAM Unified Entry Point

This is the main entry point for the complete Python-SLAM system,
integrating GUI, benchmarking, GPU acceleration, ROS2 Nav2, and embedded optimization.
"""

import sys
import os
import argparse
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # GUI Components
    from python_slam.gui.main_window import SlamMainWindow
    from python_slam.gui.utils import MaterialDesignManager
    
    # Benchmarking
    from python_slam.benchmarking.benchmark_runner import BenchmarkRunner, BenchmarkConfig
    
    # GPU Acceleration
    from python_slam.gpu_acceleration import GPUManager, get_gpu_accelerator, is_gpu_available
    
    # ROS2 Integration (optional)
    try:
        from python_slam.ros2_nav2_integration.nav2_bridge import Nav2Bridge
        ROS2_AVAILABLE = True
    except ImportError:
        ROS2_AVAILABLE = False
    
    # Embedded Optimization
    from python_slam.embedded_optimization.arm_optimization import ARMOptimizer
    
    # Core SLAM
    from python_slam.basic_slam_pipeline import BasicSLAMPipeline
    
    # PyQt6/PySide6 for GUI
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
        GUI_BACKEND = "PyQt6"
    except ImportError:
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QTimer
            GUI_BACKEND = "PySide6"
        except ImportError:
            GUI_BACKEND = None
            
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('python_slam.log')
    ]
)
logger = logging.getLogger(__name__)

class PythonSLAMSystem:
    """Main Python-SLAM system integrating all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # System components
        self.slam_pipeline = None
        self.gui_app = None
        self.main_window = None
        self.gpu_manager = None
        self.arm_optimizer = None
        self.nav2_bridge = None
        self.benchmark_runner = None
        
        # System state
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance monitoring
        self.performance_monitor = None
        self.start_time = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self, mode: str = "full") -> bool:
        """Initialize the Python-SLAM system."""
        logger.info("Initializing Python-SLAM system...")
        self.start_time = time.time()
        
        try:
            # Initialize core SLAM pipeline
            if not self._initialize_slam_pipeline():
                logger.error("Failed to initialize SLAM pipeline")
                return False
            
            # Initialize GPU acceleration
            if self.config.get("enable_gpu", True):
                self._initialize_gpu_acceleration()
            
            # Initialize ARM optimization
            if self.config.get("enable_arm_optimization", True):
                self._initialize_arm_optimization()
            
            # Initialize GUI (if requested)
            if mode in ["full", "gui"] and GUI_BACKEND:
                if not self._initialize_gui():
                    if mode == "gui":
                        logger.error("GUI initialization failed in GUI mode")
                        return False
            
            # Initialize ROS2 Nav2 (if available and requested)
            if mode in ["full", "ros2"] and ROS2_AVAILABLE and self.config.get("enable_ros2", False):
                self._initialize_ros2_nav2()
            
            # Initialize benchmarking
            if mode in ["full", "benchmark"]:
                self._initialize_benchmarking()
            
            # Start performance monitoring
            self._start_performance_monitoring()
            
            logger.info(f"Python-SLAM system initialized successfully in {mode} mode")
            self.running = True
            return True
        
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _initialize_slam_pipeline(self) -> bool:
        """Initialize the core SLAM pipeline."""
        try:
            slam_config = self.config.get("slam", {})
            self.slam_pipeline = BasicSLAMPipeline(slam_config)
            logger.info("SLAM pipeline initialized")
            return True
        except Exception as e:
            logger.error(f"SLAM pipeline initialization failed: {e}")
            return False
    
    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration."""
        try:
            if is_gpu_available():
                self.gpu_manager = GPUManager()
                if self.gpu_manager.initialize_accelerators():
                    available_backends = self.gpu_manager.get_available_backends()
                    logger.info(f"GPU acceleration initialized: {[b.value for b in available_backends]}")
                else:
                    logger.warning("GPU acceleration initialization failed")
            else:
                logger.info("No GPU acceleration available")
        except Exception as e:
            logger.error(f"GPU acceleration initialization failed: {e}")
    
    def _initialize_arm_optimization(self):
        """Initialize ARM processor optimization."""
        try:
            from python_slam.embedded_optimization.arm_optimization import ARMConfig
            arm_config = ARMConfig(**self.config.get("arm", {}))
            self.arm_optimizer = ARMOptimizer(arm_config)
            logger.info("ARM optimization initialized")
        except Exception as e:
            logger.error(f"ARM optimization initialization failed: {e}")
    
    def _initialize_gui(self) -> bool:
        """Initialize the GUI application."""
        try:
            if not GUI_BACKEND:
                logger.error("No GUI backend available (PyQt6/PySide6)")
                return False
            
            self.gui_app = QApplication(sys.argv)
            
            # Apply Material Design theme
            material_manager = MaterialDesignManager()
            material_manager.apply_theme(self.gui_app, "dark")
            
            # Create main window
            self.main_window = SlamMainWindow(
                slam_system=self.slam_pipeline,
                gpu_manager=self.gpu_manager
            )
            
            logger.info(f"GUI initialized with {GUI_BACKEND}")
            return True
        except Exception as e:
            logger.error(f"GUI initialization failed: {e}")
            return False
    
    def _initialize_ros2_nav2(self):
        """Initialize ROS2 Nav2 integration."""
        try:
            import rclpy
            rclpy.init()
            
            self.nav2_bridge = Nav2Bridge(slam_system=self.slam_pipeline)
            
            # Start ROS2 spinning in a separate thread
            self.ros2_thread = threading.Thread(
                target=self._ros2_spin_thread,
                daemon=True
            )
            self.ros2_thread.start()
            
            logger.info("ROS2 Nav2 integration initialized")
        except Exception as e:
            logger.error(f"ROS2 Nav2 initialization failed: {e}")
    
    def _initialize_benchmarking(self):
        """Initialize benchmarking system."""
        try:
            benchmark_config = BenchmarkConfig(**self.config.get("benchmark", {}))
            self.benchmark_runner = BenchmarkRunner(benchmark_config)
            logger.info("Benchmarking system initialized")
        except Exception as e:
            logger.error(f"Benchmarking initialization failed: {e}")
    
    def _start_performance_monitoring(self):
        """Start system performance monitoring."""
        try:
            self.performance_monitor = threading.Thread(
                target=self._performance_monitor_thread,
                daemon=True
            )
            self.performance_monitor.start()
            logger.info("Performance monitoring started")
        except Exception as e:
            logger.error(f"Performance monitoring failed to start: {e}")
    
    def _ros2_spin_thread(self):
        """ROS2 spinning thread."""
        try:
            import rclpy
            while self.running and not self.shutdown_event.is_set():
                rclpy.spin_once(self.nav2_bridge, timeout_sec=0.1)
        except Exception as e:
            logger.error(f"ROS2 spin thread error: {e}")
    
    def _performance_monitor_thread(self):
        """Performance monitoring thread."""
        try:
            import psutil
            
            while self.running and not self.shutdown_event.is_set():
                # Monitor system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Log performance metrics
                if cpu_percent > 80:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                
                if memory.percent > 80:
                    logger.warning(f"High memory usage: {memory.percent}%")
                
                # Update GUI if available
                if self.main_window:
                    # This would update the metrics dashboard
                    pass
                
                time.sleep(5)  # Monitor every 5 seconds
        
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
    
    def run(self, mode: str = "full"):
        """Run the Python-SLAM system."""
        if not self.running:
            logger.error("System not initialized")
            return False
        
        try:
            logger.info(f"Starting Python-SLAM system in {mode} mode")
            
            if mode == "gui" and self.main_window:
                # Run GUI event loop
                self.main_window.show()
                return self.gui_app.exec()
            
            elif mode == "benchmark" and self.benchmark_runner:
                # Run benchmarks
                self._run_benchmarks()
                return True
            
            elif mode == "headless":
                # Run in headless mode
                self._run_headless()
                return True
            
            elif mode == "full":
                # Run full system
                if self.main_window:
                    self.main_window.show()
                    return self.gui_app.exec()
                else:
                    self._run_headless()
                    return True
            
            else:
                logger.error(f"Unknown run mode: {mode}")
                return False
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            return True
        except Exception as e:
            logger.error(f"System run error: {e}")
            return False
    
    def _run_benchmarks(self):
        """Run comprehensive benchmarks."""
        logger.info("Starting comprehensive benchmarks...")
        
        # SLAM benchmarks
        if self.benchmark_runner:
            results = self.benchmark_runner.run_benchmarks()
            logger.info("SLAM benchmarks completed")
        
        # GPU benchmarks
        if self.gpu_manager:
            gpu_results = self.gpu_manager.run_benchmark()
            logger.info("GPU benchmarks completed")
        
        # ARM benchmarks
        if self.arm_optimizer:
            arm_results = self.arm_optimizer.benchmark_operations()
            logger.info("ARM benchmarks completed")
        
        # Save results
        self._save_benchmark_results({
            "slam": results if 'results' in locals() else {},
            "gpu": gpu_results if 'gpu_results' in locals() else {},
            "arm": arm_results if 'arm_results' in locals() else {}
        })
    
    def _run_headless(self):
        """Run in headless mode without GUI."""
        logger.info("Running in headless mode - press Ctrl+C to stop")
        
        try:
            while not self.shutdown_event.is_set():
                # Process SLAM data if available
                if self.slam_pipeline:
                    # This would process incoming sensor data
                    pass
                
                time.sleep(0.1)  # 10 Hz update rate
        
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown the Python-SLAM system."""
        logger.info("Shutting down Python-SLAM system...")
        
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Shutdown components
            if self.nav2_bridge:
                self.nav2_bridge.shutdown()
            
            if self.gpu_manager:
                self.gpu_manager.cleanup()
            
            if self.slam_pipeline:
                # self.slam_pipeline.shutdown()
                pass
            
            # Shutdown ROS2
            if ROS2_AVAILABLE and hasattr(self, 'nav2_bridge'):
                try:
                    import rclpy
                    rclpy.shutdown()
                except:
                    pass
            
            # Calculate uptime
            if self.start_time:
                uptime = time.time() - self.start_time
                logger.info(f"System uptime: {uptime:.2f} seconds")
            
            logger.info("Python-SLAM system shutdown completed")
        
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "running": self.running,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "components": {}
        }
        
        # SLAM status
        if self.slam_pipeline:
            status["components"]["slam"] = {"initialized": True, "active": True}
        
        # GPU status
        if self.gpu_manager:
            status["components"]["gpu"] = self.gpu_manager.get_accelerator_status()
        
        # ARM status
        if self.arm_optimizer:
            status["components"]["arm"] = self.arm_optimizer.get_performance_stats()
        
        # ROS2 status
        if self.nav2_bridge:
            status["components"]["ros2"] = self.nav2_bridge.get_status().__dict__
        
        return status

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for Python-SLAM."""
    return {
        "enable_gpu": True,
        "enable_arm_optimization": True,
        "enable_ros2": False,
        "slam": {
            "feature_detector": "SIFT",
            "max_features": 1000,
            "enable_loop_closure": True
        },
        "gpu": {
            "preferred_backend": None,
            "enable_mixed_precision": True
        },
        "arm": {
            "enable_neon": True,
            "optimization_level": "balanced"
        },
        "benchmark": {
            "enable_parallel_execution": True,
            "timeout_seconds": 300
        },
        "gui": {
            "theme": "dark",
            "enable_real_time_updates": True
        }
    }

def main():
    """Main entry point for Python-SLAM."""
    parser = argparse.ArgumentParser(description="Python-SLAM Unified System")
    
    parser.add_argument("--mode", choices=["full", "gui", "headless", "benchmark", "ros2"],
                       default="full", help="Run mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--enable-gpu", action="store_true", default=True,
                       help="Enable GPU acceleration")
    parser.add_argument("--disable-gpu", action="store_false", dest="enable_gpu",
                       help="Disable GPU acceleration")
    parser.add_argument("--enable-ros2", action="store_true", default=False,
                       help="Enable ROS2 Nav2 integration")
    parser.add_argument("--benchmark-only", action="store_true",
                       help="Run benchmarks only")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            config = create_default_config()
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config["enable_gpu"] = args.enable_gpu
    config["enable_ros2"] = args.enable_ros2
    
    # Determine run mode
    run_mode = "benchmark" if args.benchmark_only else args.mode
    
    # Create and run system
    try:
        system = PythonSLAMSystem(config)
        
        if system.initialize(mode=run_mode):
            success = system.run(mode=run_mode)
            exit_code = 0 if success else 1
        else:
            logger.error("System initialization failed")
            exit_code = 1
        
        system.shutdown()
        
    except Exception as e:
        logger.error(f"System error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
