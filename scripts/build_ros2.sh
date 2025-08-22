#!/bin/bash
# ROS 2 build script for Python SLAM

set -e

echo "Building Python SLAM with ROS 2..."

# Source ROS 2 environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "Sourced ROS 2 Humble"
else
    echo "Warning: ROS 2 Humble not found. Please install ROS 2 Humble."
    exit 1
fi

# Create workspace if it doesn't exist
WORKSPACE_DIR="$(dirname $(pwd))/ros2_ws"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Creating ROS 2 workspace at $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR/src"
fi

# Link current package to workspace
PACKAGE_LINK="$WORKSPACE_DIR/src/python_slam"
if [ ! -L "$PACKAGE_LINK" ]; then
    echo "Linking package to ROS 2 workspace..."
    ln -s "$(pwd)" "$PACKAGE_LINK"
fi

# Build the workspace
cd "$WORKSPACE_DIR"
echo "Building ROS 2 workspace..."
colcon build --packages-select python_slam --symlink-install

# Source the workspace
echo "Sourcing workspace..."
source "$WORKSPACE_DIR/install/setup.bash"

echo "ROS 2 build complete!"
echo ""
echo "To use the built package:"
echo "  source $WORKSPACE_DIR/install/setup.bash"
echo ""
echo "To launch SLAM:"
echo "  ros2 launch python_slam slam_launch.py"
