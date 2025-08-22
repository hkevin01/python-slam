#!/usr/bin/env python3
"""
Test Runner for Python-SLAM Test Suite

This script provides a comprehensive test runner for the Python-SLAM project
with options for running different test categories and generating detailed reports.
"""

import sys
import os
import unittest
import argparse
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonSLAMTestRunner:
    """Comprehensive test runner for Python-SLAM."""
    
    def __init__(self, test_dir: str = None):
        """Initialize test runner."""
        if test_dir is None:
            test_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.test_dir = Path(test_dir)
        self.project_root = self.test_dir.parent
        self.src_dir = self.project_root / "src"
        
        # Add paths for imports
        sys.path.insert(0, str(self.src_dir))
        sys.path.insert(0, str(self.project_root))
        
        # Test categories
        self.test_categories = {
            "comprehensive": "test_comprehensive.py",
            "gpu": "test_gpu_acceleration.py",
            "gui": "test_gui_components.py",
            "benchmarking": "test_benchmarking.py",
            "integration": "test_integration.py"
        }
        
        # Results storage
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        dependencies = {
            "numpy": False,
            "torch": False,
            "pyqt6": False,
            "pyside6": False,
            "matplotlib": False,
            "opencv": False,
            "psutil": False
        }
        
        # Check numpy
        try:
            import numpy
            dependencies["numpy"] = True
            logger.info(f"✓ NumPy {numpy.__version__}")
        except ImportError:
            logger.warning("✗ NumPy not available")
        
        # Check PyTorch
        try:
            import torch
            dependencies["torch"] = True
            logger.info(f"✓ PyTorch {torch.__version__}")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
            else:
                logger.info("• CUDA not available")
                
        except ImportError:
            logger.warning("✗ PyTorch not available")
        
        # Check GUI backends
        try:
            import PyQt6
            dependencies["pyqt6"] = True
            logger.info(f"✓ PyQt6 {PyQt6.QtCore.PYQT_VERSION_STR}")
        except ImportError:
            logger.info("• PyQt6 not available")
        
        try:
            import PySide6
            dependencies["pyside6"] = True
            logger.info(f"✓ PySide6 {PySide6.__version__}")
        except ImportError:
            logger.info("• PySide6 not available")
        
        # Check matplotlib
        try:
            import matplotlib
            dependencies["matplotlib"] = True
            logger.info(f"✓ Matplotlib {matplotlib.__version__}")
        except ImportError:
            logger.warning("✗ Matplotlib not available")
        
        # Check OpenCV
        try:
            import cv2
            dependencies["opencv"] = True
            logger.info(f"✓ OpenCV {cv2.__version__}")
        except ImportError:
            logger.info("• OpenCV not available")
        
        # Check psutil
        try:
            import psutil
            dependencies["psutil"] = True
            logger.info(f"✓ psutil {psutil.__version__}")
        except ImportError:
            logger.info("• psutil not available")
        
        return dependencies
    
    def run_test_category(self, category: str, verbosity: int = 2) -> unittest.TestResult:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        test_file = self.test_categories[category]
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        logger.info(f"Running {category} tests from {test_file}")
        
        # Load tests from file
        loader = unittest.TestLoader()
        
        # Import the test module
        spec = __import__(test_file[:-3])  # Remove .py extension
        suite = loader.loadTestsFromModule(spec)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            stream=sys.stdout,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        elapsed_time = time.time() - start_time
        
        # Store results
        self.results[category] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            "elapsed_time": elapsed_time,
            "was_successful": result.wasSuccessful()
        }
        
        return result
    
    def run_all_tests(self, categories: List[str] = None, verbosity: int = 2) -> bool:
        """Run all tests or specified categories."""
        if categories is None:
            categories = list(self.test_categories.keys())
        
        logger.info("Starting Python-SLAM test suite")
        logger.info("="*60)
        
        # Check dependencies
        logger.info("Checking dependencies...")
        dependencies = self.check_dependencies()
        
        missing_critical = []
        if not dependencies["numpy"]:
            missing_critical.append("numpy")
        if not dependencies["torch"]:
            missing_critical.append("torch")
        
        if missing_critical:
            logger.error(f"Critical dependencies missing: {missing_critical}")
            logger.error("Please install missing dependencies before running tests")
            return False
        
        logger.info("="*60)
        
        self.start_time = time.time()
        overall_success = True
        
        for category in categories:
            try:
                logger.info(f"\n{'='*20} {category.upper()} TESTS {'='*20}")
                result = self.run_test_category(category, verbosity)
                
                if not result.wasSuccessful():
                    overall_success = False
                    logger.warning(f"{category} tests had failures/errors")
                else:
                    logger.info(f"✓ {category} tests passed")
                    
            except Exception as e:
                logger.error(f"Failed to run {category} tests: {e}")
                overall_success = False
                
                # Record failed category
                self.results[category] = {
                    "tests_run": 0,
                    "failures": 0,
                    "errors": 1,
                    "skipped": 0,
                    "success_rate": 0.0,
                    "elapsed_time": 0.0,
                    "was_successful": False,
                    "error": str(e)
                }
        
        self.end_time = time.time()
        
        # Generate summary
        self.generate_summary()
        
        return overall_success
    
    def generate_summary(self):
        """Generate test summary report."""
        if not self.results:
            logger.warning("No test results to summarize")
            return
        
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        logger.info("\n" + "="*60)
        logger.info("TEST SUITE SUMMARY")
        logger.info("="*60)
        
        total_tests = sum(r["tests_run"] for r in self.results.values())
        total_failures = sum(r["failures"] for r in self.results.values())
        total_errors = sum(r["errors"] for r in self.results.values())
        total_skipped = sum(r["skipped"] for r in self.results.values())
        total_passed = total_tests - total_failures - total_errors - total_skipped
        
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Total tests run: {total_tests}")
        logger.info(f"Passed: {total_passed}")
        logger.info(f"Failed: {total_failures}")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Skipped: {total_skipped}")
        
        if total_tests > 0:
            success_rate = total_passed / total_tests * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("\nCategory breakdown:")
        for category, result in self.results.items():
            status = "✓" if result["was_successful"] else "✗"
            logger.info(f"  {status} {category:15} {result['tests_run']:3d} tests, "
                       f"{result['success_rate']*100:5.1f}% success, "
                       f"{result['elapsed_time']:6.2f}s")
        
        # Check for any failures
        failed_categories = [cat for cat, result in self.results.items() if not result["was_successful"]]
        if failed_categories:
            logger.warning(f"\nCategories with failures: {', '.join(failed_categories)}")
        else:
            logger.info("\n✓ All test categories passed!")
    
    def save_report(self, output_file: str = None):
        """Save detailed test report to file."""
        if output_file is None:
            output_file = self.test_dir / "test_report.json"
        
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "python_version": sys.version,
            "test_results": self.results
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Detailed report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def run_coverage_analysis(self) -> bool:
        """Run test coverage analysis if coverage.py is available."""
        try:
            import coverage
            
            logger.info("Running coverage analysis...")
            
            # Initialize coverage
            cov = coverage.Coverage(source=[str(self.src_dir)])
            cov.start()
            
            # Run tests
            success = self.run_all_tests(verbosity=1)
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            # Generate coverage report
            logger.info("\nCoverage Report:")
            cov.report(show_missing=True)
            
            # Save HTML report
            html_dir = self.test_dir / "coverage_html"
            cov.html_report(directory=str(html_dir))
            logger.info(f"HTML coverage report saved to {html_dir}")
            
            return success
            
        except ImportError:
            logger.warning("coverage.py not available, skipping coverage analysis")
            return self.run_all_tests()
    
    def run_performance_benchmark(self):
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        try:
            # Import and run performance tests
            from test_integration import TestPerformanceIntegration
            
            suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceIntegration)
            runner = unittest.TextTestRunner(verbosity=2)
            
            result = runner.run(suite)
            return result.wasSuccessful()
            
        except ImportError as e:
            logger.warning(f"Performance benchmarks not available: {e}")
            return True

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Python-SLAM Test Runner")
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["comprehensive", "gpu", "gui", "benchmarking", "integration", "all"],
        default=["all"],
        help="Test categories to run"
    )
    
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test output verbosity"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage analysis"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for test report"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = PythonSLAMTestRunner()
    
    # Check dependencies only
    if args.check_deps:
        runner.check_dependencies()
        return 0
    
    # Determine test categories
    categories = None
    if "all" not in args.categories:
        categories = args.categories
    
    # Run tests
    try:
        if args.coverage:
            success = runner.run_coverage_analysis()
        else:
            success = runner.run_all_tests(categories, args.verbosity)
        
        # Run performance benchmarks if requested
        if args.performance:
            runner.run_performance_benchmark()
        
        # Save report
        if args.output:
            runner.save_report(args.output)
        else:
            runner.save_report()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
