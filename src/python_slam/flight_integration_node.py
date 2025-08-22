#!/usr/bin/env python3
"""
Flight Integration Node for ROS 2
Integrates SLAM with drone flight control
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Path

from .flight_integration import FlightIntegration as FlightIntegrationCore


class FlightIntegrationNode(Node):
    """ROS 2 node for flight integration."""

    def __init__(self):
        super().__init__('flight_integration_node')

        # Initialize flight integration core
        self.flight_controller = FlightIntegrationCore()

        # Declare parameters
        self._declare_parameters()

        # Set up QoS
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.control_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/slam/pose',
            self.pose_callback,
            self.sensor_qos
        )

        self.path_sub = self.create_subscription(
            Path,
            '/flight/planned_path',
            self.path_callback,
            self.sensor_qos
        )

        self.emergency_sub = self.create_subscription(
            Bool,
            '/flight/emergency_stop',
            self.emergency_callback,
            self.control_qos
        )

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            self.control_qos
        )

        self.altitude_cmd_pub = self.create_publisher(
            Float32,
            '/flight/altitude_command',
            self.control_qos
        )

        self.safety_status_pub = self.create_publisher(
            Bool,
            '/flight/safety_status',
            self.sensor_qos
        )

        # Create timer for flight control updates
        control_frequency = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.control_timer = self.create_timer(1.0 / control_frequency, self.flight_control_update)

        # State variables
        self.current_pose = None
        self.planned_path = None
        self.emergency_stop = False

        self.get_logger().info('Flight Integration Node initialized')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('altitude_control', True)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('safety_distance', 1.0)
        self.declare_parameter('emergency_landing_enabled', True)
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('target_altitude', 2.0)
        self.declare_parameter('path_following_enabled', True)

    def pose_callback(self, msg: PoseStamped):
        """Update current pose from SLAM."""
        self.current_pose = msg
        self.flight_controller.update_pose(msg)

    def path_callback(self, msg: Path):
        """Update planned flight path."""
        self.planned_path = msg
        self.flight_controller.update_path(msg)

    def emergency_callback(self, msg: Bool):
        """Handle emergency stop commands."""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.get_logger().warn('Emergency stop activated!')
            self.flight_controller.emergency_stop()

    def flight_control_update(self):
        """Update flight control commands."""
        try:
            if self.current_pose is None:
                return

            # Check safety status
            is_safe = self.flight_controller.check_safety(self.current_pose)
            safety_msg = Bool()
            safety_msg.data = is_safe
            self.safety_status_pub.publish(safety_msg)

            if not is_safe or self.emergency_stop:
                # Emergency stop - publish zero velocity
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                return

            # Generate flight commands
            velocity_cmd = self.flight_controller.update(self.current_pose)

            if velocity_cmd is not None:
                self.cmd_vel_pub.publish(velocity_cmd)

            # Altitude control if enabled
            if self.get_parameter('altitude_control').get_parameter_value().bool_value:
                altitude_cmd = self.flight_controller.get_altitude_command()
                if altitude_cmd is not None:
                    altitude_msg = Float32()
                    altitude_msg.data = altitude_cmd
                    self.altitude_cmd_pub.publish(altitude_msg)

        except Exception as e:
            self.get_logger().error(f'Error in flight control update: {str(e)}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = FlightIntegrationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
