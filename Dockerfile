# Defense-Oriented Multi-stage Dockerfile for Python SLAM with ROS 2, PX4, and UCI support
FROM ros:humble-ros-base-jammy as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble
ENV COLCON_WS=/workspace
ENV SHELL=/bin/bash
ENV CLASSIFICATION_LEVEL=UNCLASSIFIED
ENV DEFENSE_MODE=true

# Install system dependencies (enhanced for defense operations)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    python3-pyqt5 \
    python3-zmq \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libeigen3-dev \
    libopencv-dev \
    libpcl-dev \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-visualization-msgs \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Install defense-oriented Python packages
RUN pip3 install --no-cache-dir \
    mavsdk==1.4.10 \
    mavsdk-server==1.4.10 \
    pyzmq==25.1.0 \
    cryptography==41.0.3 \
    asyncio-mqtt==0.16.1 \
    lxml==4.9.2 \
    pyproj==3.6.0 \
    xmltodict==0.13.0

# Initialize rosdep (skip if already exists)
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi && rosdep update

# Set up workspace
WORKDIR ${COLCON_WS}
COPY . ${COLCON_WS}/src/python_slam/

# Install Python dependencies
RUN cd ${COLCON_WS}/src/python_slam && \
    pip3 install --no-cache-dir -r requirements.txt

# Development stage with additional tools
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    gdb \
    valgrind \
    htop \
    tree \
    tmux \
    bash-completion \
    python3-pytest \
    python3-pytest-cov \
    python3-flake8 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development packages (defense-enhanced)
RUN pip3 install --no-cache-dir \
    ipython \
    jupyter \
    pre-commit \
    mypy \
    isort \
    black \
    pylint \
    pytest-asyncio \
    pytest-mock

# Set up development environment
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${COLCON_WS}/install/setup.bash" >> ~/.bashrc && \
    echo "export PYTHONPATH=${COLCON_WS}/src/python_slam/src:\$PYTHONPATH" >> ~/.bashrc

# Production stage (defense-enhanced)
FROM base as production

# Create defense user for security
RUN groupadd -r defense && useradd -r -g defense defense

# Create configuration directories
RUN mkdir -p /workspace/config/defense \
    /workspace/logs/defense \
    /workspace/missions \
    /workspace/keys

# Build the ROS 2 workspace
RUN cd ${COLCON_WS} && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --packages-select python_slam

# Set proper permissions
RUN chown -R defense:defense /workspace && \
    chmod 750 /workspace/config/defense \
    /workspace/logs/defense \
    /workspace/missions

# Source the workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${COLCON_WS}/install/setup.bash" >> ~/.bashrc

# Expose defense ports
EXPOSE 5555 5556 14540 14541

# Health check for defense monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ros2 node list | grep -q slam_node || exit 1

# Switch to defense user
USER defense

# Set the default command
CMD ["ros2", "launch", "python_slam", "slam_launch.py"]

# Runtime stage for minimal deployment
FROM ros:humble-ros-core-jammy as runtime

ENV ROS_DISTRO=humble
ENV COLCON_WS=/workspace

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    python3-numpy \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    && rm -rf /var/lib/apt/lists/*

# Copy built workspace from production stage
COPY --from=production ${COLCON_WS}/install ${COLCON_WS}/install

# Source ROS 2 and workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${COLCON_WS}/install/setup.bash" >> ~/.bashrc

WORKDIR ${COLCON_WS}
CMD ["ros2", "launch", "python_slam", "slam_launch.py"]
