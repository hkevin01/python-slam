#!/bin/bash
# Launch script for Python SLAM with ROS 2

set -e

# Default parameters
USE_RVIZ="true"
CAMERA_TOPIC="/camera/image_raw"
CAMERA_INFO_TOPIC="/camera/camera_info"
MAP_FRAME="map"
BASE_FRAME="base_link"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-rviz)
            USE_RVIZ="false"
            shift
            ;;
        --camera-topic)
            CAMERA_TOPIC="$2"
            shift 2
            ;;
        --camera-info-topic)
            CAMERA_INFO_TOPIC="$2"
            shift 2
            ;;
        --map-frame)
            MAP_FRAME="$2"
            shift 2
            ;;
        --base-frame)
            BASE_FRAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-rviz              Don't launch RViz"
            echo "  --camera-topic TOPIC    Camera image topic (default: /camera/image_raw)"
            echo "  --camera-info-topic TOPIC Camera info topic (default: /camera/camera_info)"
            echo "  --map-frame FRAME       Map frame ID (default: map)"
            echo "  --base-frame FRAME      Base frame ID (default: base_link)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Source ROS 2 environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo "Error: ROS 2 Humble not found. Please install ROS 2 Humble."
    exit 1
fi

# Source workspace if it exists
WORKSPACE_DIR="$(dirname $(pwd))/ros2_ws"
if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
    echo "Sourced workspace from $WORKSPACE_DIR"
else
    echo "Warning: ROS 2 workspace not found. Run scripts/build_ros2.sh first."
fi

echo "Launching Python SLAM with the following parameters:"
echo "  Camera topic: $CAMERA_TOPIC"
echo "  Camera info topic: $CAMERA_INFO_TOPIC"
echo "  Map frame: $MAP_FRAME"
echo "  Base frame: $BASE_FRAME"
echo "  Use RViz: $USE_RVIZ"
echo ""

# Launch SLAM
ros2 launch python_slam slam_launch.py \
    camera_topic:="$CAMERA_TOPIC" \
    camera_info_topic:="$CAMERA_INFO_TOPIC" \
    map_frame:="$MAP_FRAME" \
    base_frame:="$BASE_FRAME" \
    use_rviz:="$USE_RVIZ"
