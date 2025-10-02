#!/bin/bash
# Multi-Container SLAM Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Python SLAM Multi-Container Setup ===${NC}"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Check if config/env/.env.multi exists
if [ ! -f config/env/.env.multi ]; then
    echo -e "${YELLOW}Warning: config/env/.env.multi not found, creating from template${NC}"
    cp config/env/.env.multi.template config/env/.env.multi 2>/dev/null || echo "ROS_DOMAIN_ID=0" > config/env/.env.multi
fi

# Setup X11 forwarding for GUI
echo -e "${YELLOW}Setting up X11 forwarding for visualization...${NC}"
xhost +local:docker 2>/dev/null || echo "Warning: Could not setup X11 forwarding"

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  up                 Start all services (backend + visualization)"
    echo "  backend            Start only the SLAM backend"
    echo "  visualization      Start only the visualization GUI"
    echo "  dev                Start development containers"
    echo "  down               Stop all containers"
    echo "  logs               Show logs from all containers"
    echo "  status             Show status of all containers"
    echo "  build              Build all images"
    echo ""
    echo "Options:"
    echo "  --rebuild          Force rebuild of images"
    echo "  --no-cache         Build without using cache"
    echo "  --verbose          Show verbose output"
}

# Parse arguments
COMMAND=${1:-up}
REBUILD=false
NO_CACHE=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND=$1
            fi
            shift
            ;;
    esac
done

# Docker compose file
COMPOSE_FILE="docker-compose.multi.yml"
ENV_FILE="config/env/.env.multi"

# Build if needed
if [ "$REBUILD" = true ] || [ "$COMMAND" = "build" ]; then
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE build $NO_CACHE $VERBOSE
fi

# Execute command
case $COMMAND in
    up)
        echo -e "${GREEN}Starting SLAM backend and visualization...${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d slam-backend
        echo -e "${YELLOW}Waiting for backend to start...${NC}"
        sleep 5
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up slam-visualization
        ;;
    backend)
        echo -e "${GREEN}Starting SLAM backend only...${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up slam-backend
        ;;
    visualization)
        echo -e "${GREEN}Starting visualization only...${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up slam-visualization
        ;;
    dev)
        echo -e "${GREEN}Starting development containers...${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up slam-dev-backend slam-dev-visualization
        ;;
    down)
        echo -e "${YELLOW}Stopping all containers...${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE down
        ;;
    logs)
        echo -e "${GREEN}Showing logs...${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE logs -f
        ;;
    status)
        echo -e "${GREEN}Container status:${NC}"
        docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE ps
        ;;
    build)
        echo -e "${GREEN}Build complete${NC}"
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_usage
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"
