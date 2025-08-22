#!/bin/bash
# Defense SLAM System Entrypoint Script
# Handles secure initialization and environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Print classification banner
print_classification_banner() {
    local level=${CLASSIFICATION_LEVEL:-UNCLASSIFIED}
    local color=""

    case "$level" in
        "UNCLASSIFIED") color=$GREEN ;;
        "CONFIDENTIAL") color=$BLUE ;;
        "SECRET") color=$YELLOW ;;
        "TOP SECRET") color=$RED ;;
        *) color=$RED ;;
    esac

    echo -e "${color}"
    echo "=========================================="
    echo "  CLASSIFICATION: $level"
    echo "  DEFENSE SLAM SYSTEM v1.0.0"
    echo "  $(date +'%Y-%m-%d %H:%M:%S UTC')"
    echo "=========================================="
    echo -e "${NC}"
}

# Validate environment
validate_environment() {
    log "Validating environment configuration..."

    # Check ROS2 environment
    if [ -z "$ROS_DISTRO" ]; then
        error "ROS_DISTRO not set"
        exit 1
    fi

    # Check workspace
    if [ ! -d "/workspace" ]; then
        error "Workspace directory not found"
        exit 1
    fi

    # Check classification level
    case "${CLASSIFICATION_LEVEL:-UNCLASSIFIED}" in
        "UNCLASSIFIED"|"CONFIDENTIAL"|"SECRET"|"TOP SECRET")
            log "Classification level: $CLASSIFICATION_LEVEL"
            ;;
        *)
            warn "Invalid classification level, defaulting to UNCLASSIFIED"
            export CLASSIFICATION_LEVEL="UNCLASSIFIED"
            ;;
    esac

    log "Environment validation complete"
}

# Setup ROS2 environment
setup_ros_environment() {
    log "Setting up ROS2 environment..."

    # Source ROS2
    if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
        source "/opt/ros/$ROS_DISTRO/setup.bash"
        log "Sourced ROS2 $ROS_DISTRO"
    else
        error "ROS2 $ROS_DISTRO setup file not found"
        exit 1
    fi

    # Source workspace if built
    if [ -f "/workspace/install/setup.bash" ]; then
        source "/workspace/install/setup.bash"
        log "Sourced workspace"
    else
        warn "Workspace not built, building now..."
        cd /workspace
        colcon build --packages-select python_slam
        source "/workspace/install/setup.bash"
        log "Built and sourced workspace"
    fi
}

# Initialize defense systems
initialize_defense_systems() {
    log "Initializing defense systems..."

    # Create necessary directories
    mkdir -p /workspace/logs/defense
    mkdir -p /workspace/config/defense
    mkdir -p /workspace/missions

    # Set environment variables for defense modules
    export UCI_ENABLED=${UCI_ENABLED:-false}
    export PX4_ENABLED=${PX4_ENABLED:-false}
    export OMS_ENABLED=${OMS_ENABLED:-false}
    export DEFENSE_MODE=${DEFENSE_MODE:-true}

    # Create default configuration if not exists
    if [ ! -f "/workspace/config/defense/slam_config.yaml" ]; then
        cat > /workspace/config/defense/slam_config.yaml << EOF
# Defense SLAM Configuration
classification_level: "$CLASSIFICATION_LEVEL"
defense_mode: $DEFENSE_MODE
uci_enabled: $UCI_ENABLED
px4_enabled: $PX4_ENABLED
oms_enabled: $OMS_ENABLED

# Network Configuration
uci_command_port: 5555
uci_telemetry_port: 5556
px4_connection: "udp://:14540"

# SLAM Parameters
max_features: 1000
keyframe_distance: 1.0
processing_frequency: 30.0
state_publish_frequency: 50.0
map_publish_frequency: 1.0

# Security Settings
enable_encryption: true
log_classification: "$CLASSIFICATION_LEVEL"
audit_enabled: true
EOF
    fi

    log "Defense systems initialized"
}

# Check system health
check_system_health() {
    log "Performing system health check..."

    # Check ROS2 daemon
    if ! ros2 daemon status > /dev/null 2>&1; then
        warn "ROS2 daemon not running, starting..."
        ros2 daemon start
    fi

    # Check network connectivity for PX4
    if [ "$PX4_ENABLED" = "true" ]; then
        log "PX4 integration enabled, checking connectivity..."
        # Additional PX4 checks would go here
    fi

    # Check UCI ports
    if [ "$UCI_ENABLED" = "true" ]; then
        log "UCI interface enabled, checking ports..."
        # Additional UCI checks would go here
    fi

    log "System health check complete"
}

# Wait for dependencies
wait_for_dependencies() {
    log "Waiting for dependencies..."

    # Wait for ROS2 master
    local timeout=30
    local count=0

    while ! ros2 node list > /dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            error "Timeout waiting for ROS2 master"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
    done

    log "Dependencies ready"
}

# Main execution
main() {
    print_classification_banner
    validate_environment
    setup_ros_environment
    initialize_defense_systems
    check_system_health
    wait_for_dependencies

    log "Defense SLAM system ready, executing command: $*"

    # Execute the main command
    exec "$@"
}

# Trap signals for graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."

    # Stop any running processes gracefully
    if pgrep -f "slam_node" > /dev/null; then
        log "Stopping SLAM node..."
        pkill -TERM -f "slam_node"
        sleep 2
    fi

    # Additional cleanup
    log "Cleanup complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Run main function with all arguments
main "$@"
