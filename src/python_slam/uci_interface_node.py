#!/usr/bin/env python3
"""
UCI Interface Node - Universal Command Interface for Defense Applications
Provides standardized command and control interface for defense systems
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import threading
import json
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import sys
import os

# Add the UCI integration module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python_slam', 'uci_integration'))

try:
    from uci_interface import UCIInterface, TelemetryData, ThreatData
except ImportError as e:
    print(f"Warning: UCI interface not available: {e}")
    UCIInterface = None
    TelemetryData = None
    ThreatData = None


class UCIInterfaceNode(Node):
    """ROS2 node for UCI defense interface"""

    def __init__(self):
        super().__init__('uci_interface')

        # Parameters
        self.declare_parameter('command_port', 5555)
        self.declare_parameter('telemetry_port', 5556)
        self.declare_parameter('classification_level', 'UNCLASSIFIED')
        self.declare_parameter('node_id', 'SLAM_UCI_NODE')

        self.command_port = self.get_parameter('command_port').value
        self.telemetry_port = self.get_parameter('telemetry_port').value
        self.classification_level = self.get_parameter('classification_level').value
        self.node_id = self.get_parameter('node_id').value

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
        self.command_pub = self.create_publisher(
            String,
            '/uci/commands',
            self.mission_critical_qos
        )

        self.status_pub = self.create_publisher(
            String,
            '/uci/status',
            self.mission_critical_qos
        )

        self.threat_pub = self.create_publisher(
            String,
            '/uci/threats',
            self.mission_critical_qos
        )

        self.mission_pub = self.create_publisher(
            String,
            '/uci/mission',
            self.mission_critical_qos
        )

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/slam/pose',
            self.pose_callback,
            self.real_time_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/slam/odometry',
            self.odometry_callback,
            self.real_time_qos
        )

        self.status_sub = self.create_subscription(
            String,
            '/slam/status',
            self.slam_status_callback,
            self.mission_critical_qos
        )

        # Initialize UCI interface
        self.uci_interface = None
        self.connected = False

        if UCIInterface is not None:
            try:
                self.uci_interface = UCIInterface(
                    command_port=self.command_port,
                    telemetry_port=self.telemetry_port,
                    node_id=self.node_id,
                    classification_level=self.classification_level
                )
                self.get_logger().info(f"UCI Interface initialized on ports {self.command_port}/{self.telemetry_port}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize UCI interface: {e}")
        else:
            self.get_logger().warning("UCIInterface not available - running in simulation mode")

        # Start UCI interface in separate thread
        if self.uci_interface:
            self.uci_thread = threading.Thread(target=self.run_uci_interface, daemon=True)
            self.uci_thread.start()

        # Current state
        self.current_pose = None
        self.current_velocity = None
        self.slam_status = "INITIALIZING"

        # Timers
        self.telemetry_timer = self.create_timer(0.1, self.publish_telemetry)  # 10Hz
        self.command_timer = self.create_timer(0.05, self.process_commands)    # 20Hz

        self.get_logger().info(f"UCI Interface Node started (Classification: {self.classification_level})")

    def run_uci_interface(self):
        """Run UCI interface in separate thread"""
        try:
            if self.uci_interface:
                self.uci_interface.start()
                self.connected = True
                self.get_logger().info("UCI interface started successfully")
        except Exception as e:
            self.get_logger().error(f"UCI interface error: {e}")

    def pose_callback(self, msg: PoseStamped):
        """Handle pose updates from SLAM"""
        self.current_pose = msg

    def odometry_callback(self, msg: Odometry):
        """Handle odometry updates from SLAM"""
        self.current_velocity = msg.twist.twist

    def slam_status_callback(self, msg: String):
        """Handle SLAM status updates"""
        self.slam_status = msg.data

    def publish_telemetry(self):
        """Publish telemetry data to UCI interface"""
        if not self.connected or not self.uci_interface:
            return

        try:
            # Create telemetry data
            if self.current_pose and TelemetryData:
                telemetry = TelemetryData(
                    timestamp=self.get_clock().now().seconds_nanoseconds()[0],
                    position=[
                        self.current_pose.pose.position.x,
                        self.current_pose.pose.position.y,
                        self.current_pose.pose.position.z
                    ],
                    orientation=[
                        self.current_pose.pose.orientation.w,
                        self.current_pose.pose.orientation.x,
                        self.current_pose.pose.orientation.y,
                        self.current_pose.pose.orientation.z
                    ],
                    velocity=[
                        self.current_velocity.linear.x if self.current_velocity else 0.0,
                        self.current_velocity.linear.y if self.current_velocity else 0.0,
                        self.current_velocity.linear.z if self.current_velocity else 0.0
                    ],
                    status=self.slam_status,
                    confidence=0.95,  # Default confidence
                    classification=self.classification_level
                )

                # Send telemetry through UCI interface
                self.uci_interface.send_telemetry(telemetry)

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                "node_id": self.node_id,
                "status": "OPERATIONAL" if self.connected else "DISCONNECTED",
                "classification": self.classification_level,
                "slam_status": self.slam_status,
                "timestamp": self.get_clock().now().seconds_nanoseconds()[0]
            })
            self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish telemetry: {e}")

    def process_commands(self):
        """Process incoming UCI commands"""
        if not self.connected or not self.uci_interface:
            return

        try:
            # Check for incoming commands
            commands = self.uci_interface.get_pending_commands()

            for command in commands:
                self.process_uci_command(command)

        except Exception as e:
            self.get_logger().error(f"Failed to process commands: {e}")

    def process_uci_command(self, command):
        """Process a single UCI command"""
        try:
            command_type = command.get('type', 'UNKNOWN')

            # Publish command to ROS topics
            command_msg = String()
            command_msg.data = json.dumps(command)
            self.command_pub.publish(command_msg)

            if command_type == 'MOVE_TO_POSITION':
                # Handle position command
                target = command.get('target', {})
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = target.get('x', 0.0)
                pose_msg.pose.position.y = target.get('y', 0.0)
                pose_msg.pose.position.z = target.get('z', 0.0)

                # Publish to appropriate topic (to be consumed by other nodes)
                # This would typically go to a navigation or path planning node

            elif command_type == 'EMERGENCY_STOP':
                # Handle emergency stop
                emergency_msg = Bool()
                emergency_msg.data = True
                # Would publish to emergency topic

            elif command_type == 'SET_MISSION':
                # Handle mission assignment
                mission_msg = String()
                mission_msg.data = json.dumps(command.get('mission', {}))
                self.mission_pub.publish(mission_msg)

            elif command_type == 'THREAT_ALERT':
                # Handle threat information
                threat_msg = String()
                threat_msg.data = json.dumps(command.get('threat', {}))
                self.threat_pub.publish(threat_msg)

            self.get_logger().info(f"Processed UCI command: {command_type}")

        except Exception as e:
            self.get_logger().error(f"Failed to process UCI command: {e}")

    def shutdown(self):
        """Clean shutdown of UCI interface"""
        if self.uci_interface:
            try:
                self.uci_interface.stop()
                self.get_logger().info("UCI interface stopped")
            except Exception as e:
                self.get_logger().error(f"Error stopping UCI interface: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = UCIInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"UCI Interface Node error: {e}")
    finally:
        if 'node' in locals():
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
