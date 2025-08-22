#!/bin/bash
# Docker development scripts for Python SLAM

# Function to run commands in development container
run_in_dev() {
    docker-compose exec slam-dev "$@"
}

# Function to run commands in development container with ROS sourced
run_ros() {
    docker-compose exec slam-dev bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && $*"
}

case "$1" in
    "setup")
        echo "🐳 Setting up Python SLAM development environment..."
        docker-compose build slam-dev
        docker-compose up -d slam-dev
        echo "✅ Development environment ready!"
        ;;

    "build")
        echo "🔨 Building ROS 2 workspace..."
        run_in_dev colcon build --symlink-install
        ;;

    "test")
        echo "🧪 Running tests..."
        run_in_dev pytest tests/ -v
        ;;

    "slam")
        echo "🚀 Starting SLAM system..."
        run_ros "ros2 launch python_slam slam_launch.py"
        ;;

    "shell")
        echo "🖥️ Opening development shell..."
        docker-compose exec dev bash
        ;;

    "stop")
        echo "🛑 Stopping development environment..."
        docker-compose down
        ;;

    "clean")
        echo "🧹 Cleaning up containers and images..."
        docker-compose down -v
        docker system prune -f
        ;;

    "logs")
        echo "📋 Showing container logs..."
        docker-compose logs -f dev
        ;;

    *)
        echo "🐳 Python SLAM Docker Development Tools"
        echo "======================================="
        echo ""
        echo "Usage: $0 {setup|build|test|slam|shell|stop|clean|logs}"
        echo ""
        echo "Commands:"
        echo "  setup   - Build and start development environment"
        echo "  build   - Build ROS 2 workspace in container"
        echo "  test    - Run test suite in container"
        echo "  slam    - Start SLAM system"
        echo "  shell   - Open development shell in container"
        echo "  stop    - Stop development environment"
        echo "  clean   - Clean up containers and images"
        echo "  logs    - Show container logs"
        echo ""
        echo "Examples:"
        echo "  $0 setup     # Initial setup"
        echo "  $0 build     # Build the project"
        echo "  $0 shell     # Start development session"
        echo "  $0 slam      # Run SLAM system"
        ;;
esac
