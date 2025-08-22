#!/usr/bin/env python3
"""
PX4 Bridge Node - ROS2 Bridge for PX4 Flight Controller Integration
Provides interface between SLAM system and PX4 autopilot for UAS operations
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import asyncio
import threading
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Bool
import sys
import os

# Add the PX4 integration module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python_slam', 'px4_integration'))

try:
    from px4_interface import PX4Interface, UASState
except ImportError as e:
    print(f"Warning: PX4 interface not available: {e}")
    PX4Interface = None
    UASState = None


class PX4BridgeNode(Node):
    """ROS2 bridge node for PX4 integration"""

    def __init__(self):
        super().__init__('px4_bridge')

        # Parameters
        self.declare_parameter('px4_connection', 'udp://:14540')
        self.declare_parameter('classification_level', 'UNCLASSIFIED')

        self.px4_connection = self.get_parameter('px4_connection').value
        self.classification_level = self.get_parameter('classification_level').value

        # QoS profiles for defense operations
        self.mission_critical_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        self.real_time_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=5
        )

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/px4/pose',
            self.mission_critical_qos
        )

        self.velocity_pub = self.create_publisher(
            Twist,
            '/px4/velocity',
            self.real_time_qos
        )

        self.gps_pub = self.create_publisher(
            NavSatFix,
            '/px4/gps',
            self.mission_critical_qos
        )

        self.status_pub = self.create_publisher(
            String,
            '/px4/status',
            self.mission_critical_qos
        )

        # Subscribers
        self.cmd_pose_sub = self.create_subscription(
            PoseStamped,
            '/slam/target_pose',
            self.target_pose_callback,
            self.mission_critical_qos
        )

        self.cmd_velocity_sub = self.create_subscription(
            Twist,
            '/slam/cmd_vel',
            self.cmd_velocity_callback,
            self.real_time_qos
        )

        self.emergency_sub = self.create_subscription(
            Bool,
            '/slam/emergency_stop',
            self.emergency_callback,
            self.mission_critical_qos
        )

        # Initialize PX4 interface
        self.px4_interface = None
        self.connected = False

        if PX4Interface is not None:
            try:
                self.px4_interface = PX4Interface(self.px4_connection)
                self.get_logger().info(f"PX4 Bridge initialized with connection: {self.px4_connection}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize PX4 interface: {e}")
        else:
            self.get_logger().warning("PX4Interface not available - running in simulation mode")

        # Start async tasks
        self.executor_thread = threading.Thread(target=self.run_async_tasks, daemon=True)
        self.executor_thread.start()

        # Timers
        self.status_timer = self.create_timer(0.1, self.publish_status)  # 10Hz
        self.telemetry_timer = self.create_timer(0.02, self.publish_telemetry)  # 50Hz

        self.get_logger().info(f"PX4 Bridge Node started (Classification: {self.classification_level})")

    def run_async_tasks(self):
        """Run async PX4 tasks in separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if self.px4_interface:
                loop.run_until_complete(self.px4_interface.connect())
                self.connected = True
                loop.run_until_complete(self.px4_interface.monitor_telemetry())
        except Exception as e:
            self.get_logger().error(f"PX4 async task error: {e}")
        finally:
            loop.close()

    def target_pose_callback(self, msg: PoseStamped):
        """Handle target pose commands from SLAM"""
        if not self.connected or not self.px4_interface:
            return

        try:
            # Convert ROS pose to PX4 coordinates
            target_position = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]

            # Send waypoint to PX4 (async operation)
            asyncio.run_coroutine_threadsafe(
                self.px4_interface.set_position_ned(*target_position),
                self.px4_interface.loop
            )

            self.get_logger().debug(f"Sent target pose to PX4: {target_position}")

        except Exception as e:
            self.get_logger().error(f"Failed to send target pose: {e}")

    def cmd_velocity_callback(self, msg: Twist):
        """Handle velocity commands from SLAM"""
        if not self.connected or not self.px4_interface:
            return

        try:
            # Convert ROS twist to PX4 velocity
            velocity = [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.z
            ]

            # Send velocity command to PX4
            asyncio.run_coroutine_threadsafe(
                self.px4_interface.set_velocity_ned(*velocity),
                self.px4_interface.loop
            )

        except Exception as e:
            self.get_logger().error(f"Failed to send velocity command: {e}")

    def emergency_callback(self, msg: Bool):
        """Handle emergency stop commands"""
        if msg.data and self.connected and self.px4_interface:
            try:
                # Trigger emergency landing
                asyncio.run_coroutine_threadsafe(
                    self.px4_interface.emergency_land(),
                    self.px4_interface.loop
                )
                self.get_logger().warning("Emergency stop activated - initiating landing")
            except Exception as e:
                self.get_logger().error(f"Failed to execute emergency stop: {e}")

    def publish_status(self):
        """Publish PX4 status information"""
        status_msg = String()

        if self.connected and self.px4_interface:
            try:
                state = self.px4_interface.get_current_state()
                if state:
                    status_msg.data = f"CONNECTED|{state.flight_mode}|{state.armed}|{state.in_air}"
                else:
                    status_msg.data = "CONNECTED|UNKNOWN|UNKNOWN|UNKNOWN"
            except Exception as e:
                status_msg.data = f"ERROR|{str(e)}"
        else:
            status_msg.data = "DISCONNECTED"

        self.status_pub.publish(status_msg)

    def publish_telemetry(self):
        """Publish PX4 telemetry data"""
        if not self.connected or not self.px4_interface:
            return

        try:
            state = self.px4_interface.get_current_state()
            if not state:
                return

            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = state.position[0]
            pose_msg.pose.position.y = state.position[1]
            pose_msg.pose.position.z = state.position[2]
            pose_msg.pose.orientation.w = state.attitude[0]
            pose_msg.pose.orientation.x = state.attitude[1]
            pose_msg.pose.orientation.y = state.attitude[2]
            pose_msg.pose.orientation.z = state.attitude[3]
            self.pose_pub.publish(pose_msg)

            # Publish velocity
            vel_msg = Twist()
            vel_msg.linear.x = state.velocity[0]
            vel_msg.linear.y = state.velocity[1]
            vel_msg.linear.z = state.velocity[2]
            self.velocity_pub.publish(vel_msg)

            # Publish GPS if available
            if state.gps_position:
                gps_msg = NavSatFix()
                gps_msg.header.stamp = pose_msg.header.stamp
                gps_msg.header.frame_id = "gps"
                gps_msg.latitude = state.gps_position[0]
                gps_msg.longitude = state.gps_position[1]
                gps_msg.altitude = state.gps_position[2]
                self.gps_pub.publish(gps_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish telemetry: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PX4BridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"PX4 Bridge Node error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
