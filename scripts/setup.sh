#!/bin/bash
# Setup script for Python SLAM project - Docker-based workflow

echo "🐳 Python SLAM Docker Setup"
echo "=========================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose build

# Start development environment
echo "🚀 Starting development environment..."
docker-compose up -d dev

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  docker-compose exec dev bash                    # Enter development container"
echo "  docker-compose exec dev colcon build            # Build ROS 2 workspace"
echo "  docker-compose exec dev source install/setup.bash && ros2 launch python_slam slam_launch.py"
echo "  docker-compose exec dev pytest tests/          # Run tests"
echo "  docker-compose down                             # Stop containers"
echo ""
echo "📚 See README.md for detailed documentation"
