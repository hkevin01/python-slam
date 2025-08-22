# Multi-Container SLAM Architecture

This document describes the multi-container architecture for the Python SLAM system, which separates the ROS2 SLAM processing backend from the PyQt5 visualization frontend.

## Architecture Overview

The system is now split into two main containers:

1. **SLAM Backend** (`slam-backend`): Handles all ROS2 SLAM processing
2. **SLAM Visualization** (`slam-visualization`): Provides PyQt5 GUI for visualization

## Benefits of Multi-Container Setup

- **Separation of Concerns**: Backend processing is isolated from GUI rendering
- **Scalability**: Backend can run on different hardware than visualization
- **Flexibility**: Visualization can connect to remote SLAM backends
- **Development**: Easier to develop and debug individual components
- **Performance**: GUI doesn't impact SLAM processing performance
- **Deployment**: Backend can run headless on robots while visualization runs on operator stations

## Container Details

### SLAM Backend Container
- **Image**: Built from `docker/Dockerfile.backend`
- **Purpose**: ROS2 SLAM processing, sensor fusion, mapping
- **Exposed Ports**:
  - `14540`: PX4 connection
  - `5555`: UCI command port / ZMQ visualization data
  - `5556`: UCI telemetry port
- **Key Components**:
  - SLAM node
  - Feature extraction
  - Pose estimation
  - Mapping
  - Loop closure
  - Visualization bridge (ZMQ publisher)

### SLAM Visualization Container
- **Image**: Built from `docker/Dockerfile.visualization`
- **Purpose**: PyQt5 GUI for real-time visualization
- **Features**:
  - 3D point cloud visualization
  - Camera trajectory tracking
  - Performance metrics
  - Real-time status monitoring
- **Connection**: Connects to backend via ZMQ (tcp://slam-backend:5555)

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- X11 forwarding for GUI (Linux/macOS)

### Starting the System

1. **Complete System** (Backend + Visualization):
   ```bash
   ./run-multi.sh up
   ```

2. **Backend Only**:
   ```bash
   ./run-multi.sh backend
   ```

3. **Visualization Only** (requires running backend):
   ```bash
   ./run-multi.sh visualization
   ```

4. **Development Mode**:
   ```bash
   ./run-multi.sh dev
   ```

### Configuration

Environment variables are configured in `.env.multi`:
```bash
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
SLAM_BACKEND_ADDRESS=tcp://slam-backend:5555
ENABLE_PX4=false
ENABLE_UCI=false
```

## Communication Architecture

### ROS2 Communication (Backend Internal)
- All SLAM nodes communicate via ROS2 topics
- Uses CycloneDX DDS implementation
- Custom network configuration in `config/cyclonedx.xml`

### Backend-to-Visualization Communication
- **Protocol**: ZeroMQ (ZMQ)
- **Pattern**: Publisher/Subscriber
- **Port**: 5555
- **Data Format**: JSON

#### ZMQ Message Format
```json
{
  "timestamp": 1234567890.123,
  "pose": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  },
  "trajectory": [[x1, y1, z1], [x2, y2, z2], ...],
  "pointcloud": {
    "points": [[x1, y1, z1], ...],
    "frame_id": "map"
  },
  "map": {
    "width": 1024,
    "height": 1024,
    "resolution": 0.05,
    "data": [...]
  }
}
```

## Development

### Backend Development
```bash
# Start backend development container
docker-compose -f docker-compose.multi.yml up slam-dev-backend

# Attach to container
docker exec -it python-slam-dev-backend bash

# Build and test
cd /workspace/src/python_slam
colcon build
source install/setup.bash
ros2 launch python_slam slam_backend_launch.py
```

### Visualization Development
```bash
# Start visualization development container
docker-compose -f docker-compose.multi.yml up slam-dev-visualization

# Attach to container
docker exec -it python-slam-dev-visualization bash

# Run visualization
python3 src/python_slam/gui/slam_visualizer.py
```

## Networking

### Docker Network
- **Network**: `slam-network` (172.20.0.0/16)
- **DNS**: Containers can resolve each other by name
- **Ports**: Only necessary ports are exposed to host

### ROS2 Domain Configuration
- **Domain ID**: 0 (configurable via ROS_DOMAIN_ID)
- **Discovery**: Uses CycloneDX with custom peer configuration
- **QoS**: Optimized profiles for different data types

## Monitoring and Debugging

### Container Status
```bash
./run-multi.sh status
```

### Logs
```bash
# All containers
./run-multi.sh logs

# Specific container
docker-compose -f docker-compose.multi.yml logs slam-backend
docker-compose -f docker-compose.multi.yml logs slam-visualization
```

### ROS2 Debugging
```bash
# List nodes
docker exec python-slam-backend ros2 node list

# Topic information
docker exec python-slam-backend ros2 topic list
docker exec python-slam-backend ros2 topic info /slam/pose

# Echo topics
docker exec python-slam-backend ros2 topic echo /slam/pose
```

### ZMQ Connection Testing
```bash
# Test ZMQ connection from visualization container
docker exec python-slam-visualization python3 -c "
import zmq
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://slam-backend:5555')
socket.subscribe('')
socket.setsockopt(zmq.RCVTIMEO, 5000)
try:
    message = socket.recv_json()
    print('ZMQ connection working:', message.keys())
except:
    print('ZMQ connection failed')
"
```

## Troubleshooting

### Common Issues

1. **GUI not displaying**:
   - Check X11 forwarding: `xhost +local:docker`
   - Verify DISPLAY environment variable

2. **Backend not connecting**:
   - Check ROS2 domain ID matches
   - Verify network connectivity between containers

3. **ZMQ connection failed**:
   - Ensure backend is fully started before visualization
   - Check if backend is publishing data

4. **Performance issues**:
   - Limit point cloud size in visualization
   - Adjust update frequencies
   - Monitor container resource usage

### Performance Tuning

1. **Backend Performance**:
   - Adjust SLAM processing frequency
   - Optimize feature detection parameters
   - Use appropriate QoS profiles

2. **Visualization Performance**:
   - Limit maximum points displayed
   - Reduce update frequency
   - Adjust rendering quality

3. **Network Performance**:
   - Use appropriate ZMQ message sizes
   - Consider data compression for large point clouds
   - Optimize ROS2 QoS settings

## Security Considerations

1. **Network Isolation**: Containers run in isolated network
2. **Port Exposure**: Only necessary ports exposed to host
3. **User Privileges**: Containers run with minimal privileges
4. **Data Encryption**: Consider TLS for production deployments

## Future Enhancements

1. **Multi-Backend Support**: Connect visualization to multiple SLAM backends
2. **Cloud Deployment**: Support for Kubernetes deployment
3. **Data Recording**: Built-in recording and playback capabilities
4. **Remote Access**: Web-based visualization interface
5. **Distributed Computing**: Split SLAM processing across multiple containers


