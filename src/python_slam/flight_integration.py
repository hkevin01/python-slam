#!/usr/bin/env python3
"""
Flight Integration Module for Python SLAM
Integrates SLAM with drone flight control systems
"""

import numpy as np
from typing import Optional, List, Tuple
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32


class FlightIntegration:
    """
    Flight integration class for aerial drone SLAM systems.
    Provides safe flight control integration with SLAM localization.
    """

    def __init__(self, max_velocity: float = 2.0, safety_distance: float = 1.0):
        """
        Initialize flight integration system.

        Args:
            max_velocity: Maximum allowed velocity (m/s)
            safety_distance: Safety distance from obstacles (m)
        """
        self.max_velocity = max_velocity
        self.safety_distance = safety_distance

        # Flight state
        self.current_pose: Optional[PoseStamped] = None
        self.target_pose: Optional[PoseStamped] = None
        self.planned_path: Optional[Path] = None
        self.emergency_stop_active = False

        # Control parameters
        self.position_gain = 1.0
        self.velocity_gain = 0.5
        self.altitude_gain = 2.0
        self.yaw_gain = 1.0

        # Safety parameters
        self.min_altitude = 0.5  # meters
        self.max_altitude = 50.0  # meters
        self.obstacle_detection_range = 3.0  # meters

        # Flight modes
        self.altitude_control_enabled = True
        self.autonomous_navigation_enabled = False

    def update_pose(self, pose: PoseStamped):
        """
        Update current pose from SLAM.

        Args:
            pose: Current robot pose from SLAM
        """
        self.current_pose = pose

        # Check safety conditions
        self._check_safety_conditions()

    def update_path(self, path: Path):
        """
        Update planned flight path.

        Args:
            path: Planned flight path
        """
        self.planned_path = path

        # Set next target from path
        if len(path.poses) > 0:
            self.target_pose = path.poses[0]

    def update(self, current_pose: PoseStamped) -> Optional[Twist]:
        """
        Main update function - generates flight control commands.

        Args:
            current_pose: Current pose from SLAM

        Returns:
            Velocity command or None if emergency stop
        """
        try:
            self.update_pose(current_pose)

            if self.emergency_stop_active:
                return self._emergency_stop_command()

            if self.target_pose is None:
                return self._hover_command()

            # Generate control command
            cmd_vel = self._generate_velocity_command()

            # Apply safety limits
            cmd_vel = self._apply_safety_limits(cmd_vel)

            return cmd_vel

        except Exception as e:
            print(f"Error in flight integration update: {e}")
            return self._emergency_stop_command()

    def _generate_velocity_command(self) -> Twist:
        """Generate velocity command to reach target."""
        cmd_vel = Twist()

        if self.current_pose is None or self.target_pose is None:
            return cmd_vel

        try:
            # Calculate position error
            current_pos = np.array([
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ])

            target_pos = np.array([
                self.target_pose.pose.position.x,
                self.target_pose.pose.position.y,
                self.target_pose.pose.position.z
            ])

            position_error = target_pos - current_pos

            # Generate velocity commands
            cmd_vel.linear.x = self.position_gain * position_error[0]
            cmd_vel.linear.y = self.position_gain * position_error[1]

            # Altitude control
            if self.altitude_control_enabled:
                cmd_vel.linear.z = self.altitude_gain * position_error[2]

            # Yaw control (simplified)
            cmd_vel.angular.z = 0.0

            return cmd_vel

        except Exception as e:
            print(f"Error generating velocity command: {e}")
            return Twist()

    def _apply_safety_limits(self, cmd_vel: Twist) -> Twist:
        """Apply safety limits to velocity command."""
        try:
            # Limit linear velocities
            linear_speed = np.sqrt(cmd_vel.linear.x**2 + cmd_vel.linear.y**2 + cmd_vel.linear.z**2)
            if linear_speed > self.max_velocity:
                scale = self.max_velocity / linear_speed
                cmd_vel.linear.x *= scale
                cmd_vel.linear.y *= scale
                cmd_vel.linear.z *= scale

            # Altitude limits
            if self.current_pose is not None:
                current_altitude = self.current_pose.pose.position.z

                if current_altitude <= self.min_altitude and cmd_vel.linear.z < 0:
                    cmd_vel.linear.z = 0.0

                if current_altitude >= self.max_altitude and cmd_vel.linear.z > 0:
                    cmd_vel.linear.z = 0.0

            return cmd_vel

        except Exception as e:
            print(f"Error applying safety limits: {e}")
            return Twist()

    def _emergency_stop_command(self) -> Twist:
        """Generate emergency stop command."""
        cmd_vel = Twist()
        # All velocities to zero
        return cmd_vel

    def _hover_command(self) -> Twist:
        """Generate hover command."""
        cmd_vel = Twist()

        # Maintain current altitude if altitude control enabled
        if self.altitude_control_enabled and self.current_pose is not None:
            target_altitude = max(self.min_altitude, self.current_pose.pose.position.z)
            altitude_error = target_altitude - self.current_pose.pose.position.z
            cmd_vel.linear.z = self.altitude_gain * altitude_error

        return cmd_vel

    def _check_safety_conditions(self):
        """Check and update safety conditions."""
        try:
            if self.current_pose is None:
                return

            # Check altitude limits
            altitude = self.current_pose.pose.position.z
            if altitude < self.min_altitude * 0.8 or altitude > self.max_altitude * 1.1:
                self.emergency_stop_active = True
                print(f"Emergency stop: Altitude violation ({altitude:.2f}m)")
                return

            # Check for rapid position changes (potential SLAM failure)
            # Implementation would track position changes over time

            # Reset emergency stop if conditions are safe
            if self.emergency_stop_active:
                if self.min_altitude <= altitude <= self.max_altitude:
                    self.emergency_stop_active = False
                    print("Emergency stop cleared")

        except Exception as e:
            print(f"Error checking safety: {e}")
            self.emergency_stop_active = True

    def emergency_stop(self):
        """Activate emergency stop."""
        self.emergency_stop_active = True
        print("Emergency stop activated manually")

    def clear_emergency_stop(self):
        """Clear emergency stop."""
        self.emergency_stop_active = False
        print("Emergency stop cleared manually")

    def check_safety(self, pose: PoseStamped) -> bool:
        """
        Check if current pose is safe for flight.

        Args:
            pose: Current pose to check

        Returns:
            True if pose is safe
        """
        try:
            altitude = pose.pose.position.z

            # Check altitude bounds
            if altitude < self.min_altitude or altitude > self.max_altitude:
                return False

            # Additional safety checks could be added here

            return True

        except Exception as e:
            print(f"Error in safety check: {e}")
            return False

    def get_altitude_command(self) -> Optional[float]:
        """
        Get altitude command for altitude controller.

        Returns:
            Target altitude or None
        """
        if self.target_pose is not None and self.altitude_control_enabled:
            return self.target_pose.pose.position.z

        return None

    def set_target_pose(self, pose: PoseStamped):
        """Set target pose for navigation."""
        self.target_pose = pose

    def enable_altitude_control(self, enable: bool):
        """Enable/disable altitude control."""
        self.altitude_control_enabled = enable

    def enable_autonomous_navigation(self, enable: bool):
        """Enable/disable autonomous navigation."""
        self.autonomous_navigation_enabled = enable

    def get_flight_status(self) -> dict:
        """Get current flight status."""
        status = {
            'emergency_stop': self.emergency_stop_active,
            'altitude_control': self.altitude_control_enabled,
            'autonomous_nav': self.autonomous_navigation_enabled,
            'has_target': self.target_pose is not None,
            'has_path': self.planned_path is not None,
        }

        if self.current_pose is not None:
            status['current_altitude'] = self.current_pose.pose.position.z
            status['altitude_safe'] = self.check_safety(self.current_pose)

        return status


if __name__ == "__main__":
    print("Flight Integration Module - Demo")

    # Create flight integrator
    flight = FlightIntegration(max_velocity=1.5, safety_distance=1.0)

    # Create test pose
    pose = PoseStamped()
    pose.pose.position.x = 1.0
    pose.pose.position.y = 2.0
    pose.pose.position.z = 3.0

    # Set target
    target = PoseStamped()
    target.pose.position.x = 2.0
    target.pose.position.y = 3.0
    target.pose.position.z = 3.0

    flight.set_target_pose(target)

    # Update
    cmd_vel = flight.update(pose)

    if cmd_vel:
        print(f"Generated velocity command: x={cmd_vel.linear.x:.2f}, y={cmd_vel.linear.y:.2f}, z={cmd_vel.linear.z:.2f}")

    # Get status
    status = flight.get_flight_status()
    print(f"Flight status: {status}")

    print("Flight integration module demo complete!")
