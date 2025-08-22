#!/usr/bin/env python3
"""
Simple Test Launcher for Python-SLAM

Quick test launcher with common test scenarios.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main test launcher."""
    test_dir = Path(__file__).parent
    test_runner = test_dir / "run_tests.py"
    
    print("Python-SLAM Test Launcher")
    print("=" * 50)
    print("1. Run all tests")
    print("2. Run quick tests (comprehensive only)")
    print("3. Run GPU tests")
    print("4. Run GUI tests")
    print("5. Run benchmarking tests")
    print("6. Run integration tests")
    print("7. Check dependencies")
    print("8. Run with coverage")
    print("9. Custom test selection")
    print("0. Exit")
    print("=" * 50)
    
    while True:
        try:
            choice = input("Select test option (0-9): ").strip()
            
            if choice == "0":
                print("Exiting...")
                break
            elif choice == "1":
                cmd = [sys.executable, str(test_runner)]
            elif choice == "2":
                cmd = [sys.executable, str(test_runner), "--categories", "comprehensive"]
            elif choice == "3":
                cmd = [sys.executable, str(test_runner), "--categories", "gpu"]
            elif choice == "4":
                cmd = [sys.executable, str(test_runner), "--categories", "gui"]
            elif choice == "5":
                cmd = [sys.executable, str(test_runner), "--categories", "benchmarking"]
            elif choice == "6":
                cmd = [sys.executable, str(test_runner), "--categories", "integration"]
            elif choice == "7":
                cmd = [sys.executable, str(test_runner), "--check-deps"]
            elif choice == "8":
                cmd = [sys.executable, str(test_runner), "--coverage"]
            elif choice == "9":
                print("\nAvailable categories: comprehensive, gpu, gui, benchmarking, integration")
                categories = input("Enter categories (space-separated): ").strip().split()
                if categories:
                    cmd = [sys.executable, str(test_runner), "--categories"] + categories
                else:
                    print("No categories specified, skipping...")
                    continue
            else:
                print(f"Invalid choice: {choice}")
                continue
            
            print(f"\nRunning: {' '.join(cmd)}")
            print("-" * 50)
            
            # Run the command
            try:
                result = subprocess.run(cmd, cwd=test_dir.parent)
                print("-" * 50)
                if result.returncode == 0:
                    print("✓ Tests completed successfully")
                else:
                    print("✗ Tests completed with failures")
            except KeyboardInterrupt:
                print("\n⚠ Test execution interrupted")
            except Exception as e:
                print(f"✗ Error running tests: {e}")
            
            print("\nPress Enter to continue...")
            input()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
