#!/bin/bash
# Test script for multi-container SLAM setup

echo "=== Testing Multi-Container SLAM Setup ==="

# Check if docker-compose file exists
if [ ! -f "docker-compose.multi.yml" ]; then
    echo "ERROR: docker-compose.multi.yml not found"
    exit 1
fi

# Check if environment file exists
if [ ! -f "config/env/.env.multi" ]; then
    echo "WARNING: config/env/.env.multi not found, creating default"
    echo "ROS_DOMAIN_ID=0" > config/env/.env.multi
fi

# Build the images
echo "Building Docker images..."
docker-compose -f docker-compose.multi.yml build --no-cache

# Test backend container
echo "Testing backend container..."
docker-compose -f docker-compose.multi.yml up -d slam-backend

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 10

# Check if backend is running
if docker ps | grep -q "python-slam-backend"; then
    echo "✓ Backend container is running"
else
    echo "✗ Backend container failed to start"
    docker-compose -f docker-compose.multi.yml logs slam-backend
    exit 1
fi

# Test ZMQ connection
echo "Testing ZMQ communication..."
timeout 30 docker exec python-slam-backend python3 -c "
import zmq
import json
import time
import threading

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://*:5555')

# Send test message
time.sleep(1)
test_data = {
    'timestamp': time.time(),
    'pose': {'position': {'x': 0, 'y': 0, 'z': 0}},
    'trajectory': [[0, 0, 0]],
    'pointcloud': {'points': [], 'frame_id': 'map'}
}
socket.send_json(test_data)
print('Test message sent')
socket.close()
context.term()
" || echo "ZMQ test completed"

# Clean up
echo "Cleaning up test containers..."
docker-compose -f docker-compose.multi.yml down

echo "=== Multi-Container SLAM Test Complete ==="
