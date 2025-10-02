# Python-SLAM Project Structure

This document describes the organized structure of the Python-SLAM project after cleanup and reorganization.

## Root Directory Structure

```text
python-slam/
├── .copilot/              # GitHub Copilot configuration
├── .github/               # GitHub workflows and templates
├── .vscode/               # VS Code workspace settings
├── assets/                # Static assets and media files
├── config/                # Configuration files
│   ├── build/             # Build and dependency configuration
│   │   ├── CMakeLists.txt
│   │   ├── Makefile
│   │   ├── package.xml
│   │   ├── requirements.txt
│   │   └── setup.py
│   └── env/               # Environment configuration
│       ├── .env           # Default environment variables
│       └── .env.multi     # Multi-container environment
├── data/                  # Data files and datasets
├── docker/                # Docker containerization
│   ├── Dockerfile         # Main application container
│   ├── Dockerfile.backend # Backend service container
│   ├── Dockerfile.visualization # Visualization container
│   ├── docker-compose.yml # Standard compose configuration
│   ├── docker-compose.multi.yml # Multi-container setup
│   ├── cyclonedx.xml      # Software Bill of Materials
│   └── entrypoint.sh      # Container entry point script
├── docs/                  # Documentation
│   ├── archive/           # Archived documentation versions
│   │   ├── README_*.md    # Previous README versions
│   │   ├── COMPLETION_SUMMARY*.md # Implementation summaries
│   │   └── IMPLEMENTATION_*.md # Implementation documents
│   ├── api/               # API documentation
│   ├── configuration/     # Configuration guides
│   ├── design/            # Design documents
│   ├── diagrams/          # Architecture diagrams
│   ├── procedures/        # Operational procedures
│   ├── requirements/      # Requirements documentation
│   └── testing/           # Testing documentation
├── launch/                # ROS2 launch files
├── resource/              # ROS2 resource files
├── rviz/                  # RViz configuration files
├── scripts/               # Build and utility scripts
│   ├── build_ros2.sh      # ROS2 build script
│   ├── dev.sh             # Development environment setup
│   ├── install.sh         # Installation script
│   ├── launch_slam.sh     # SLAM launch script
│   ├── run-multi.sh       # Multi-container runner
│   ├── setup.sh           # General setup script
│   ├── setup_dev.sh       # Development setup
│   ├── test-multi.sh      # Multi-container testing
│   └── test_pyslam_*.py   # PySlam integration tests
├── src/                   # Source code
│   ├── python_slam/       # Main Python package
│   └── python_slam_main.py # Main application entry point
├── tests/                 # Test suite
│   ├── test_comprehensive.py # Comprehensive test suite (NASA STD-8739.8)
│   ├── test_framework.py  # Framework tests
│   └── test_slam_modules.py # Module-specific tests
├── tools/                 # Development and validation tools
│   └── validation/        # System validation tools
│       ├── configure.py   # Configuration wizard
│       └── validate_system.py # System validation script
├── CHANGELOG.md           # Version history and changes
├── README.md              # Main project documentation
└── TODO.md                # Project todo list
```

## Organization Principles

### 1. Configuration Management

- **config/build/**: Build system configuration (CMake, Make, setuptools, requirements)
- **config/env/**: Environment variables and runtime configuration
- **docker/**: All Docker-related files consolidated in one location

### 2. Documentation Strategy

- **docs/**: Active documentation with organized subdirectories
- **docs/archive/**: Historical versions and deprecated documentation
- Separation of API docs, design docs, and user guides

### 3. Code Organization

- **src/**: All source code including main application entry point
- **tests/**: Comprehensive test suite with NASA STD-8739.8 compliance
- **tools/**: Development utilities and validation scripts

### 4. Operational Files

- **scripts/**: All shell scripts for build, setup, and deployment
- **launch/**: ROS2 launch configurations
- **rviz/**: Visualization configurations

### 5. Asset Management

- **assets/**: Static files and media
- **data/**: Datasets and runtime data
- **resource/**: ROS2 package resources

## Benefits of This Structure

1. **Clear Separation of Concerns**: Each directory has a specific purpose
2. **Easy Navigation**: Developers can quickly find relevant files
3. **Build System Clarity**: All build configs in config/build/
4. **Documentation Organization**: Active vs archived documentation
5. **Container Management**: All Docker files in one location
6. **Tool Accessibility**: Development tools in dedicated directory
7. **Clean Root**: Minimal files in root directory for better readability

## Maintenance Guidelines

- Keep root directory clean with only essential files
- Archive old documentation versions in docs/archive/
- Place new configuration files in appropriate config/ subdirectories
- Add new tools to tools/ with appropriate subdirectories
- Maintain this structure documentation when making organizational changes

## File Reference Updates

After reorganization, scripts and configurations may need path updates:

- Main application: `src/python_slam_main.py`
- Requirements: `config/build/requirements.txt`
- Docker compose: `docker/docker-compose.yml`
- Environment: `config/env/.env`
- Build configs: `config/build/`
