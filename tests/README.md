# Python-SLAM Test Suite

This directory contains a comprehensive test suite for the Python-SLAM project.

## Test Structure

### Test Categories

1. **Comprehensive Tests** (`test_comprehensive.py`)
   - Core SLAM functionality
   - System integration tests
   - Basic smoke tests

2. **GPU Acceleration Tests** (`test_gpu_acceleration.py`)
   - GPU detection and management
   - CUDA, ROCm, and Metal backend testing
   - Accelerated operations validation
   - Performance benchmarking

3. **GUI Component Tests** (`test_gui_components.py`)
   - PyQt6/PySide6 interface testing
   - Material Design styling
   - 3D visualization components
   - Control panels and metrics dashboard

4. **Benchmarking Tests** (`test_benchmarking.py`)
   - Trajectory evaluation metrics (ATE, RPE)
   - Processing performance metrics
   - Dataset loading and validation
   - Benchmark report generation

5. **Integration Tests** (`test_integration.py`)
   - Component interaction testing
   - Data pipeline validation
   - Performance monitoring
   - Error handling and recovery
   - Scalability testing

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --categories gpu benchmarking

# Run with coverage analysis
python tests/run_tests.py --coverage

# Check dependencies
python tests/run_tests.py --check-deps
```

### Test Runner Options

```bash
python tests/run_tests.py [OPTIONS]

Options:
  --categories CATEGORIES   Test categories to run (comprehensive, gpu, gui, benchmarking, integration, all)
  --verbosity VERBOSITY     Test output verbosity (0, 1, 2)
  --coverage               Run with coverage analysis
  --performance            Run performance benchmarks
  --output OUTPUT          Output file for test report
  --check-deps             Only check dependencies and exit
```

### Examples

```bash
# Run only GPU and benchmarking tests with high verbosity
python tests/run_tests.py --categories gpu benchmarking --verbosity 2

# Run all tests with coverage and save report
python tests/run_tests.py --coverage --output test_results.json

# Run performance benchmarks
python tests/run_tests.py --performance

# Check if all dependencies are available
python tests/run_tests.py --check-deps
```

## Test Requirements

### Core Dependencies
- Python 3.8+
- NumPy
- PyTorch (for GPU acceleration tests)

### Optional Dependencies
- PyQt6 or PySide6 (for GUI tests)
- Matplotlib (for visualization tests)
- OpenCV (for computer vision tests)
- psutil (for system monitoring tests)
- coverage.py (for coverage analysis)

### GPU Testing
- NVIDIA GPU + CUDA (for CUDA tests)
- AMD GPU + ROCm (for ROCm tests)
- Apple Silicon (for Metal tests)

Note: GPU tests will automatically skip if the corresponding hardware/software is not available.

## Test Output

### Console Output
The test runner provides detailed console output including:
- Dependency checking results
- Test execution progress
- Individual test results
- Summary statistics
- Performance metrics

### Test Reports
- JSON report with detailed results (`test_report.json`)
- Coverage reports (if `--coverage` used)
- Performance benchmark results

### Example Output
```
==================== COMPREHENSIVE TESTS ====================
test_basic_slam_pipeline (__main__.TestPythonSLAMCore) ... ok
test_feature_extraction (__main__.TestPythonSLAMCore) ... ok
...

============================================================
TEST SUITE SUMMARY
============================================================
Total execution time: 45.23s
Total tests run: 89
Passed: 85
Failed: 2
Errors: 0
Skipped: 2
Success rate: 95.5%

Category breakdown:
  ✓ comprehensive    25 tests,  96.0% success,   8.45s
  ✓ gpu             18 tests, 100.0% success,  12.34s
  ✗ gui             15 tests,  86.7% success,   5.67s
  ✓ benchmarking    21 tests, 100.0% success,  11.23s
  ✓ integration     10 tests, 100.0% success,   7.54s
```

## Continuous Integration

### GitHub Actions
The test suite is designed to work with GitHub Actions. Example workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run tests
      run: python tests/run_tests.py --coverage
```

### Local Development
For local development, you can run tests automatically on file changes using tools like `pytest-watch` or `watchdog`.

## Writing New Tests

### Test Structure
Follow the existing test structure:

```python
import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestNewComponent(unittest.TestCase):
    """Test new component functionality."""
    
    def setUp(self):
        """Set up test environment."""
        pass
    
    def tearDown(self):
        """Clean up test environment."""
        pass
    
    def test_component_functionality(self):
        """Test specific functionality."""
        try:
            from python_slam.new_component import NewComponent
            component = NewComponent()
            self.assertIsNotNone(component)
        except ImportError:
            self.skipTest("New component not available")

if __name__ == "__main__":
    unittest.main(verbosity=2)
```

### Best Practices
1. **Use `skipTest()` for optional dependencies**
2. **Clean up resources in `tearDown()`**
3. **Use meaningful test names**
4. **Test both success and failure cases**
5. **Mock external dependencies when possible**
6. **Include performance tests for critical paths**

### Adding Tests to Runner
To add new test categories, update the `test_categories` dictionary in `run_tests.py`:

```python
self.test_categories = {
    "new_category": "test_new_category.py",
    # ... existing categories
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify src directory structure

2. **GPU Test Failures**
   - Check GPU drivers and libraries
   - Verify CUDA/ROCm installation
   - Tests should skip gracefully if GPU unavailable

3. **GUI Test Failures**
   - May require display server (use Xvfb on headless systems)
   - Check PyQt6/PySide6 installation
   - Some tests may need to be skipped in CI environments

4. **Memory Issues**
   - Large datasets may cause memory issues
   - Reduce test data size if needed
   - Ensure proper cleanup in tearDown methods

### Debug Mode
For debugging test failures, increase verbosity and run specific test files:

```bash
python -m unittest tests.test_gpu_acceleration.TestGPUDetector.test_cuda_detection -v
```

## Contributing

When contributing new features:
1. Write corresponding tests
2. Ensure all existing tests pass
3. Add integration tests for component interactions
4. Update documentation if test structure changes
5. Consider performance implications

## Performance Monitoring

The test suite includes performance monitoring:
- Execution time tracking
- Memory usage monitoring
- GPU utilization metrics
- Benchmark comparisons

Use the `--performance` flag to run additional performance tests that measure system capabilities and identify potential bottlenecks.
