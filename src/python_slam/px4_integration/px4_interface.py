"""
PX4 Integration for UAS SLAM Applications
Provides interface to PX4-based flight controllers for autonomous navigation
"""

import asyncio
import numpy as np
import threading
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

try:
    from mavsdk import System
    from mavsdk.offboard import (PositionNedYaw, VelocityBodyYawspeed,
                                 VelocityNedYaw, AttitudeRate)
    from mavsdk.telemetry import Position, Attitude, VelocityNed
    MAVSDK_AVAILABLE = True
except ImportError:
    MAVSDK_AVAILABLE = False
    print("MAVSDK not available. PX4 integration will be disabled.")

class FlightMode(Enum):
    """Flight modes for UAS operations"""
    MANUAL = "MANUAL"
    STABILIZE = "STABILIZE"
    ALTITUDE_HOLD = "ALTITUDE_HOLD"
    POSITION_HOLD = "POSITION_HOLD"
    OFFBOARD = "OFFBOARD"
    AUTO_MISSION = "AUTO_MISSION"
    AUTO_RTL = "AUTO_RTL"
    AUTO_LAND = "AUTO_LAND"

@dataclass
class UASState:
    """UAS state information"""
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray  # Roll, pitch, yaw
    armed: bool
    flight_mode: FlightMode
    battery_voltage: float
    gps_satellites: int
    timestamp: float

class PX4Interface:
    """Interface for PX4-based flight controllers"""

    def __init__(self, system_address: str = "udp://:14540"):
        if not MAVSDK_AVAILABLE:
            raise ImportError("MAVSDK not available for PX4 integration")

        self.drone = System()
        self.system_address = system_address
        self.is_connected = False
        self.is_armed = False
        self.current_state = None

        # Telemetry data
        self.position_data = None
        self.attitude_data = None
        self.velocity_data = None

        # Control loops
        self.telemetry_task = None
        self.control_task = None

        # Command queue for offboard control
        self.command_queue = asyncio.Queue()

        # Safety parameters
        self.max_velocity = 5.0  # m/s
        self.max_acceleration = 2.0  # m/s²
        self.safety_altitude_min = 2.0  # m
        self.safety_altitude_max = 100.0  # m

    async def connect(self) -> bool:
        """Connect to PX4 flight controller"""
        try:
            await self.drone.connect(system_address=self.system_address)

            # Wait for connection
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print(f"PX4 Connected to {self.system_address}")
                    self.is_connected = True
                    break

            if self.is_connected:
                # Start telemetry monitoring
                self.telemetry_task = asyncio.create_task(self._telemetry_loop())
                return True

        except Exception as e:
            print(f"Failed to connect to PX4: {e}")

        return False

    async def disconnect(self):
        """Disconnect from PX4"""
        if self.telemetry_task:
            self.telemetry_task.cancel()
        if self.control_task:
            self.control_task.cancel()
        self.is_connected = False

    async def _telemetry_loop(self):
        """Continuous telemetry monitoring"""
        try:
            # Start telemetry streams
            position_task = asyncio.create_task(self._position_stream())
            attitude_task = asyncio.create_task(self._attitude_stream())
            velocity_task = asyncio.create_task(self._velocity_stream())

            # Wait for all streams
            await asyncio.gather(position_task, attitude_task, velocity_task)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Telemetry loop error: {e}")

    async def _position_stream(self):
        """Monitor position telemetry"""
        async for position in self.drone.telemetry.position():
            self.position_data = position

    async def _attitude_stream(self):
        """Monitor attitude telemetry"""
        async for attitude in self.drone.telemetry.attitude_euler():
            self.attitude_data = attitude

    async def _velocity_stream(self):
        """Monitor velocity telemetry"""
        async for velocity in self.drone.telemetry.velocity_ned():
            self.velocity_data = velocity

    def get_current_state(self) -> Optional[UASState]:
        """Get current UAS state"""
        if not all([self.position_data, self.attitude_data, self.velocity_data]):
            return None

        return UASState(
            position=np.array([
                self.position_data.latitude_deg,
                self.position_data.longitude_deg,
                self.position_data.absolute_altitude_m
            ]),
            velocity=np.array([
                self.velocity_data.north_m_s,
                self.velocity_data.east_m_s,
                self.velocity_data.down_m_s
            ]),
            attitude=np.array([
                self.attitude_data.roll_deg,
                self.attitude_data.pitch_deg,
                self.attitude_data.yaw_deg
            ]),
            armed=self.is_armed,
            flight_mode=FlightMode.OFFBOARD,  # Simplified
            battery_voltage=0.0,  # Would need battery telemetry
            gps_satellites=0,  # Would need GPS telemetry
            timestamp=time.time()
        )

    async def arm(self) -> bool:
        """Arm the vehicle"""
        try:
            await self.drone.action.arm()
            self.is_armed = True
            return True
        except Exception as e:
            print(f"Failed to arm: {e}")
            return False

    async def disarm(self) -> bool:
        """Disarm the vehicle"""
        try:
            await self.drone.action.disarm()
            self.is_armed = False
            return True
        except Exception as e:
            print(f"Failed to disarm: {e}")
            return False

    async def takeoff(self, altitude: float = 20.0) -> bool:
        """Takeoff to specified altitude"""
        try:
            # Check safety limits
            if altitude < self.safety_altitude_min or altitude > self.safety_altitude_max:
                print(f"Takeoff altitude {altitude}m outside safety limits")
                return False

            await self.drone.action.set_takeoff_altitude(altitude)
            await self.drone.action.takeoff()
            return True
        except Exception as e:
            print(f"Failed to takeoff: {e}")
            return False

    async def land(self) -> bool:
        """Land the vehicle"""
        try:
            await self.drone.action.land()
            return True
        except Exception as e:
            print(f"Failed to land: {e}")
            return False

    async def return_to_launch(self) -> bool:
        """Return to launch position"""
        try:
            await self.drone.action.return_to_launch()
            return True
        except Exception as e:
            print(f"Failed to RTL: {e}")
            return False

    async def start_offboard_mode(self) -> bool:
        """Start offboard control mode"""
        try:
            # Set initial setpoint
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(0.0, 0.0, 0.0, 0.0)
            )

            # Start offboard mode
            await self.drone.offboard.start()

            # Start control loop
            self.control_task = asyncio.create_task(self._control_loop())
            return True
        except Exception as e:
            print(f"Failed to start offboard mode: {e}")
            return False

    async def stop_offboard_mode(self) -> bool:
        """Stop offboard control mode"""
        try:
            await self.drone.offboard.stop()
            if self.control_task:
                self.control_task.cancel()
            return True
        except Exception as e:
            print(f"Failed to stop offboard mode: {e}")
            return False

    async def _control_loop(self):
        """Main control loop for offboard mode"""
        try:
            while True:
                # Process command queue
                try:
                    command = await asyncio.wait_for(
                        self.command_queue.get(),
                        timeout=0.05
                    )
                    await self._execute_command(command)
                except asyncio.TimeoutError:
                    # No new commands, maintain current setpoint
                    pass

                await asyncio.sleep(0.02)  # 50 Hz control loop

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Control loop error: {e}")

    async def _execute_command(self, command: Dict):
        """Execute a control command"""
        cmd_type = command.get('type')

        if cmd_type == 'position':
            await self._set_position_setpoint(command)
        elif cmd_type == 'velocity':
            await self._set_velocity_setpoint(command)
        elif cmd_type == 'trajectory':
            await self._execute_trajectory(command)

    async def _set_position_setpoint(self, command: Dict):
        """Set position setpoint"""
        north = command.get('north', 0.0)
        east = command.get('east', 0.0)
        down = command.get('down', 0.0)
        yaw = command.get('yaw', 0.0)

        # Safety checks
        if abs(down) < self.safety_altitude_min:
            down = -self.safety_altitude_min
        elif abs(down) > self.safety_altitude_max:
            down = -self.safety_altitude_max

        await self.drone.offboard.set_position_ned(
            PositionNedYaw(north, east, down, yaw)
        )

    async def _set_velocity_setpoint(self, command: Dict):
        """Set velocity setpoint"""
        vx = command.get('vx', 0.0)
        vy = command.get('vy', 0.0)
        vz = command.get('vz', 0.0)
        yaw_rate = command.get('yaw_rate', 0.0)

        # Apply velocity limits
        velocity_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if velocity_mag > self.max_velocity:
            scale = self.max_velocity / velocity_mag
            vx *= scale
            vy *= scale
            vz *= scale

        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, yaw_rate)
        )

    async def send_position_command(self, north: float, east: float,
                                  down: float, yaw: float = 0.0):
        """Send position command to queue"""
        command = {
            'type': 'position',
            'north': north,
            'east': east,
            'down': down,
            'yaw': yaw
        }
        await self.command_queue.put(command)

    async def send_velocity_command(self, vx: float, vy: float,
                                  vz: float, yaw_rate: float = 0.0):
        """Send velocity command to queue"""
        command = {
            'type': 'velocity',
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'yaw_rate': yaw_rate
        }
        await self.command_queue.put(command)

    async def execute_waypoint_mission(self, waypoints: List[Tuple[float, float, float, float]],
                                     tolerance: float = 1.0) -> bool:
        """Execute a waypoint mission"""
        try:
            for i, wp in enumerate(waypoints):
                north, east, down, yaw = wp

                print(f"Flying to waypoint {i+1}/{len(waypoints)}: "
                      f"N={north:.1f}, E={east:.1f}, D={down:.1f}")

                # Send position command
                await self.send_position_command(north, east, down, yaw)

                # Wait for arrival
                await self._wait_for_position(north, east, down, tolerance)

            print("Waypoint mission completed")
            return True

        except Exception as e:
            print(f"Mission execution failed: {e}")
            return False

    async def _wait_for_position(self, target_north: float, target_east: float,
                               target_down: float, tolerance: float):
        """Wait until vehicle reaches target position"""
        while True:
            if self.position_data:
                # Convert to NED coordinates (simplified)
                current_north = 0.0  # Would need proper coordinate conversion
                current_east = 0.0
                current_down = -self.position_data.relative_altitude_m

                distance = np.sqrt(
                    (current_north - target_north)**2 +
                    (current_east - target_east)**2 +
                    (current_down - target_down)**2
                )

                if distance < tolerance:
                    break

            await asyncio.sleep(0.1)

    def emergency_stop(self):
        """Emergency stop - switch to RTL mode"""
        asyncio.create_task(self.return_to_launch())

    def set_safety_parameters(self, max_velocity: float = None,
                            max_acceleration: float = None,
                            altitude_min: float = None,
                            altitude_max: float = None):
        """Set safety parameters"""
        if max_velocity is not None:
            self.max_velocity = max_velocity
        if max_acceleration is not None:
            self.max_acceleration = max_acceleration
        if altitude_min is not None:
            self.safety_altitude_min = altitude_min
        if altitude_max is not None:
            self.safety_altitude_max = altitude_max

class PX4SLAMIntegration:
    """Integration layer between SLAM and PX4"""

    def __init__(self, px4_interface: PX4Interface):
        self.px4 = px4_interface
        self.slam_pose = None
        self.target_pose = None
        self.control_mode = "POSITION_HOLD"

    def update_slam_pose(self, pose: np.ndarray):
        """Update current pose from SLAM system"""
        self.slam_pose = pose

    def set_target_pose(self, pose: np.ndarray):
        """Set target pose for navigation"""
        self.target_pose = pose

    async def navigate_to_pose(self, target_pose: np.ndarray,
                             tolerance: float = 0.5) -> bool:
        """Navigate to target pose using SLAM feedback"""
        if self.slam_pose is None:
            print("No SLAM pose available")
            return False

        # Convert pose to NED coordinates
        target_north = target_pose[0, 3]
        target_east = target_pose[1, 3]
        target_down = -target_pose[2, 3]  # Convert to NED

        # Extract yaw from rotation matrix
        yaw = np.arctan2(target_pose[1, 0], target_pose[0, 0])

        # Send command to PX4
        await self.px4.send_position_command(
            target_north, target_east, target_down, yaw
        )

        return True

    async def slam_guided_navigation(self, waypoints: List[np.ndarray]) -> bool:
        """Execute navigation using SLAM pose feedback"""
        for i, waypoint in enumerate(waypoints):
            print(f"Navigating to waypoint {i+1}/{len(waypoints)}")

            success = await self.navigate_to_pose(waypoint)
            if not success:
                print(f"Failed to navigate to waypoint {i+1}")
                return False

            # Wait for arrival using SLAM pose
            await self._wait_for_slam_arrival(waypoint)

        return True

    async def _wait_for_slam_arrival(self, target_pose: np.ndarray,
                                   tolerance: float = 0.5):
        """Wait for arrival using SLAM pose feedback"""
        while True:
            if self.slam_pose is not None:
                # Calculate distance to target
                current_pos = self.slam_pose[:3, 3]
                target_pos = target_pose[:3, 3]
                distance = np.linalg.norm(current_pos - target_pos)

                if distance < tolerance:
                    break

            await asyncio.sleep(0.1)
