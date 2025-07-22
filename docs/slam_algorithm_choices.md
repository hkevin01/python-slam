# SLAM Algorithm Choices

## Selected Approach
- Visual SLAM using ORB feature extraction and Essential Matrix pose estimation.
- Chosen for simplicity, speed, and compatibility with aerial drone hardware.

## Alternatives Considered
- ORB-SLAM3: Full-featured, but more complex and resource-intensive.
- GraphSLAM: Good for pose graph optimization, but less real-time.
- RGB-D SLAM: Requires depth sensors, not always available on drones.

## Rationale
- The basic pipeline is modular and can be extended with IMU, loop closure, and semantic mapping as needed.
