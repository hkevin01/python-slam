# Testing Strategy

## Overview
Testing is performed at the unit, integration, and pipeline levels. Automated tests are run via GitHub Actions CI.

## Unit Tests
- Each core module (feature extraction, pose estimation, mapping) has dedicated tests in `tests/`.

## Integration Tests
- The basic SLAM pipeline is validated with sample data and images.

## CI/CD
- All tests are run automatically on push and pull request via GitHub Actions.

## Benchmarking
- Results are compared against expected outputs and performance metrics.

## Troubleshooting
- See `docs/troubleshooting.md` for common issues and solutions.
