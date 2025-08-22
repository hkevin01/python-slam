"""
UCI (Universal Command and Control Interface) and OMS (Open Mission Systems) Integration
Provides defense-oriented command and control capabilities for UAS SLAM operations
"""

import json
import time
import threading
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("ZMQ not available. UCI interface will be disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class UCIMessageType(Enum):
    """UCI message types for defense applications"""
    COMMAND = "COMMAND"
    STATUS = "STATUS"
    TELEMETRY = "TELEMETRY"
    SENSOR = "SENSOR"
    MISSION = "MISSION"
    THREAT = "THREAT"
    INTELLIGENCE = "INTELLIGENCE"
    NAVIGATION = "NAVIGATION"
    HEALTH = "HEALTH"
    EMERGENCY = "EMERGENCY"

class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class MissionPhase(Enum):
    """Mission execution phases"""
    PLANNING = "PLANNING"
    PREPARATION = "PREPARATION"
    EXECUTION = "EXECUTION"
    COMPLETION = "COMPLETION"
    ABORT = "ABORT"
    EMERGENCY = "EMERGENCY"

@dataclass
class UCIMessage:
    """UCI message structure for defense communications"""
    message_id: str
    message_type: UCIMessageType
    priority: Priority
    timestamp: float
    source: str
    destination: str
    classification: str  # UNCLASSIFIED, CONFIDENTIAL, SECRET, etc.
    payload: Dict[str, Any]
    sequence_number: int = 0
    acknowledgment_required: bool = False

@dataclass
class TelemetryData:
    """Standardized telemetry data structure"""
    position: List[float]  # [lat, lon, alt] or [x, y, z]
    orientation: List[float]  # [roll, pitch, yaw]
    velocity: List[float]  # [vx, vy, vz]
    angular_velocity: List[float]  # [wx, wy, wz]
    timestamp: float
    confidence: float  # 0.0 to 1.0
    source: str  # "SLAM", "GPS", "INS", etc.

@dataclass
class ThreatData:
    """Threat detection data"""
    threat_id: str
    threat_type: str  # "RADAR", "MISSILE", "AIRCRAFT", etc.
    position: List[float]
    velocity: List[float]
    confidence: float
    severity: int  # 1-5 scale
    timestamp: float

class UCIInterface:
    """Universal Command and Control Interface for UAS operations"""

    def __init__(self, node_id: str = "SLAM_NODE",
                 command_port: int = 5555,
                 telemetry_port: int = 5556,
                 classification: str = "UNCLASSIFIED"):

        if not ZMQ_AVAILABLE:
            raise ImportError("ZMQ not available for UCI interface")

        self.node_id = node_id
        self.classification = classification
        self.context = zmq.Context()

        # Command channel (REQ/REP pattern)
        self.command_socket = self.context.socket(zmq.REP)
        self.command_socket.bind(f"tcp://*:{command_port}")

        # Telemetry channel (PUB/SUB pattern)
        self.telemetry_socket = self.context.socket(zmq.PUB)
        self.telemetry_socket.bind(f"tcp://*:{telemetry_port}")

        # Subscriber for external messages
        self.subscriber_socket = self.context.socket(zmq.SUB)
        self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Message handling
        self.message_handlers: Dict[UCIMessageType, Callable] = {}
        self.sequence_counter = 0
        self.running = False

        # Threading
        self.command_thread = None
        self.telemetry_thread = None

    def start(self):
        """Start UCI interface threads"""
        self.running = True

        # Start command handling thread
        self.command_thread = threading.Thread(
            target=self._command_handler_loop,
            daemon=True
        )
        self.command_thread.start()

        # Start subscriber thread for external messages
        self.telemetry_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True
        )
        self.telemetry_thread.start()

        print(f"UCI Interface started for node {self.node_id}")

    def stop(self):
        """Stop UCI interface"""
        self.running = False

        if self.command_thread:
            self.command_thread.join(timeout=1.0)
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=1.0)

        self.command_socket.close()
        self.telemetry_socket.close()
        self.subscriber_socket.close()
        self.context.term()

    def register_handler(self, message_type: UCIMessageType,
                        handler: Callable[[UCIMessage], Dict]):
        """Register message handler"""
        self.message_handlers[message_type] = handler

    def _command_handler_loop(self):
        """Handle incoming command messages"""
        while self.running:
            try:
                # Wait for command with timeout
                if self.command_socket.poll(1000):  # 1 second timeout
                    message_data = self.command_socket.recv_json()

                    # Parse UCI message
                    uci_msg = self._parse_uci_message(message_data)

                    # Handle message
                    response = self._handle_message(uci_msg)

                    # Send response
                    self.command_socket.send_json(response)

            except Exception as e:
                print(f"Command handler error: {e}")
                # Send error response
                error_response = {
                    "status": "ERROR",
                    "message": str(e),
                    "timestamp": time.time()
                }
                try:
                    self.command_socket.send_json(error_response)
                except:
                    pass

    def _subscriber_loop(self):
        """Handle incoming telemetry/status messages"""
        while self.running:
            try:
                if self.subscriber_socket.poll(1000):  # 1 second timeout
                    message_data = self.subscriber_socket.recv_json()
                    uci_msg = self._parse_uci_message(message_data)
                    self._handle_message(uci_msg)

            except Exception as e:
                print(f"Subscriber error: {e}")

    def _parse_uci_message(self, data: Dict) -> UCIMessage:
        """Parse UCI message from JSON data"""
        return UCIMessage(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=UCIMessageType(data['message_type']),
            priority=Priority(data.get('priority', Priority.NORMAL.value)),
            timestamp=data.get('timestamp', time.time()),
            source=data['source'],
            destination=data['destination'],
            classification=data.get('classification', 'UNCLASSIFIED'),
            payload=data['payload'],
            sequence_number=data.get('sequence_number', 0),
            acknowledgment_required=data.get('acknowledgment_required', False)
        )

    def _handle_message(self, message: UCIMessage) -> Dict:
        """Handle UCI message and return response"""
        try:
            if message.message_type in self.message_handlers:
                response_data = self.message_handlers[message.message_type](message)
            else:
                response_data = {
                    "status": "UNSUPPORTED",
                    "message": f"No handler for {message.message_type.value}"
                }

            response = {
                "message_id": str(uuid.uuid4()),
                "response_to": message.message_id,
                "status": "SUCCESS",
                "timestamp": time.time(),
                "source": self.node_id,
                "destination": message.source,
                "data": response_data
            }

            return response

        except Exception as e:
            return {
                "message_id": str(uuid.uuid4()),
                "response_to": message.message_id,
                "status": "ERROR",
                "error": str(e),
                "timestamp": time.time(),
                "source": self.node_id,
                "destination": message.source
            }

    def send_telemetry(self, telemetry: TelemetryData):
        """Send telemetry update"""
        message = UCIMessage(
            message_id=str(uuid.uuid4()),
            message_type=UCIMessageType.TELEMETRY,
            priority=Priority.NORMAL,
            timestamp=time.time(),
            source=self.node_id,
            destination="GCS",
            classification=self.classification,
            payload=asdict(telemetry)
        )

        self._publish_message(message)

    def send_status(self, status_code: str, status_text: str,
                   priority: Priority = Priority.NORMAL):
        """Send status update"""
        message = UCIMessage(
            message_id=str(uuid.uuid4()),
            message_type=UCIMessageType.STATUS,
            priority=priority,
            timestamp=time.time(),
            source=self.node_id,
            destination="GCS",
            classification=self.classification,
            payload={
                "status_code": status_code,
                "status_text": status_text,
                "node_health": "OPERATIONAL"
            }
        )

        self._publish_message(message)

    def send_threat_alert(self, threat: ThreatData):
        """Send threat detection alert"""
        message = UCIMessage(
            message_id=str(uuid.uuid4()),
            message_type=UCIMessageType.THREAT,
            priority=Priority.CRITICAL,
            timestamp=time.time(),
            source=self.node_id,
            destination="ALL",
            classification="CONFIDENTIAL",  # Threat data typically classified
            payload=asdict(threat)
        )

        self._publish_message(message)

    def send_emergency(self, emergency_type: str, description: str):
        """Send emergency message"""
        message = UCIMessage(
            message_id=str(uuid.uuid4()),
            message_type=UCIMessageType.EMERGENCY,
            priority=Priority.EMERGENCY,
            timestamp=time.time(),
            source=self.node_id,
            destination="ALL",
            classification=self.classification,
            payload={
                "emergency_type": emergency_type,
                "description": description,
                "location": "UNKNOWN",
                "severity": "HIGH"
            }
        )

        self._publish_message(message)

    def _publish_message(self, message: UCIMessage):
        """Publish message via telemetry channel"""
        try:
            message_data = {
                "message_id": message.message_id,
                "message_type": message.message_type.value,
                "priority": message.priority.value,
                "timestamp": message.timestamp,
                "source": message.source,
                "destination": message.destination,
                "classification": message.classification,
                "payload": message.payload,
                "sequence_number": self.sequence_counter
            }

            self.sequence_counter += 1
            self.telemetry_socket.send_json(message_data)

        except Exception as e:
            print(f"Failed to publish message: {e}")

class OMSAdapter:
    """Open Mission Systems adapter for defense applications"""

    def __init__(self):
        self.mission_state = MissionPhase.PLANNING
        self.current_waypoint = 0
        self.waypoints: List[Dict] = []
        self.mission_metadata: Dict = {}
        self.constraints: Dict = {}

    def load_mission_xml(self, mission_file: str) -> bool:
        """Load OMS-compliant mission file"""
        try:
            tree = ET.parse(mission_file)
            root = tree.getroot()

            # Parse mission metadata
            self.mission_metadata = {
                "mission_id": root.get("id", "unknown"),
                "mission_name": root.get("name", "Unnamed Mission"),
                "classification": root.get("classification", "UNCLASSIFIED"),
                "priority": root.get("priority", "NORMAL"),
                "start_time": root.get("start_time"),
                "end_time": root.get("end_time")
            }

            # Parse waypoints
            self.waypoints.clear()
            for wp in root.findall('.//Waypoint'):
                waypoint = {
                    "id": int(wp.get("id", 0)),
                    "type": wp.get("type", "TRANSIT"),
                    "latitude": float(wp.find("Latitude").text),
                    "longitude": float(wp.find("Longitude").text),
                    "altitude": float(wp.find("Altitude").text),
                    "speed": float(wp.find("Speed").text) if wp.find("Speed") is not None else 5.0,
                    "loiter_time": float(wp.find("LoiterTime").text) if wp.find("LoiterTime") is not None else 0.0,
                    "heading": float(wp.find("Heading").text) if wp.find("Heading") is not None else 0.0
                }

                # Parse actions
                actions = []
                for action in wp.findall(".//Action"):
                    actions.append({
                        "type": action.get("type"),
                        "parameters": {child.tag: child.text for child in action}
                    })
                waypoint["actions"] = actions

                self.waypoints.append(waypoint)

            # Parse constraints
            self.constraints = {
                "max_altitude": 120.0,  # Default AGL limit
                "min_altitude": 30.0,
                "max_speed": 15.0,  # m/s
                "no_fly_zones": [],
                "geofence": None
            }

            constraints_elem = root.find(".//Constraints")
            if constraints_elem is not None:
                for constraint in constraints_elem:
                    if constraint.tag == "MaxAltitude":
                        self.constraints["max_altitude"] = float(constraint.text)
                    elif constraint.tag == "MinAltitude":
                        self.constraints["min_altitude"] = float(constraint.text)
                    elif constraint.tag == "MaxSpeed":
                        self.constraints["max_speed"] = float(constraint.text)

            print(f"Loaded mission: {self.mission_metadata['mission_name']} "
                  f"with {len(self.waypoints)} waypoints")

            return True

        except Exception as e:
            print(f"Failed to load mission file: {e}")
            return False

    def validate_mission(self) -> Tuple[bool, List[str]]:
        """Validate mission against constraints and regulations"""
        errors = []

        # Check waypoint constraints
        for wp in self.waypoints:
            if wp["altitude"] > self.constraints["max_altitude"]:
                errors.append(f"Waypoint {wp['id']} altitude exceeds maximum")
            if wp["altitude"] < self.constraints["min_altitude"]:
                errors.append(f"Waypoint {wp['id']} altitude below minimum")
            if wp["speed"] > self.constraints["max_speed"]:
                errors.append(f"Waypoint {wp['id']} speed exceeds maximum")

        # Check for required metadata
        if not self.mission_metadata.get("mission_id"):
            errors.append("Mission ID required")

        return len(errors) == 0, errors

    def get_next_waypoint(self) -> Optional[Dict]:
        """Get next waypoint in mission sequence"""
        if self.current_waypoint < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint]
            self.current_waypoint += 1
            return wp
        return None

    def get_current_waypoint(self) -> Optional[Dict]:
        """Get current active waypoint"""
        if 0 <= self.current_waypoint < len(self.waypoints):
            return self.waypoints[self.current_waypoint]
        return None

    def reset_mission(self):
        """Reset mission to beginning"""
        self.current_waypoint = 0
        self.mission_state = MissionPhase.PREPARATION

    def abort_mission(self):
        """Abort current mission"""
        self.mission_state = MissionPhase.ABORT

    def complete_mission(self):
        """Mark mission as completed"""
        self.mission_state = MissionPhase.COMPLETION

    def get_mission_progress(self) -> float:
        """Get mission completion percentage"""
        if not self.waypoints:
            return 0.0
        return (self.current_waypoint / len(self.waypoints)) * 100.0

    def export_mission_status(self) -> Dict:
        """Export current mission status"""
        return {
            "mission_metadata": self.mission_metadata,
            "mission_state": self.mission_state.value,
            "current_waypoint": self.current_waypoint,
            "total_waypoints": len(self.waypoints),
            "progress_percent": self.get_mission_progress(),
            "constraints": self.constraints,
            "timestamp": time.time()
        }

class DefenseSLAMInterface:
    """Integration interface for defense SLAM operations"""

    def __init__(self, node_id: str = "SLAM_DEFENSE_NODE"):
        self.node_id = node_id
        self.uci_interface = None
        self.oms_adapter = OMSAdapter()

        # Defense-specific state
        self.threat_level = "GREEN"  # GREEN, YELLOW, ORANGE, RED
        self.operational_mode = "NORMAL"  # NORMAL, STEALTH, DEFENSIVE, OFFENSIVE
        self.classification_level = "UNCLASSIFIED"

    def initialize_uci(self, command_port: int = 5555,
                      telemetry_port: int = 5556) -> bool:
        """Initialize UCI interface"""
        try:
            self.uci_interface = UCIInterface(
                self.node_id,
                command_port,
                telemetry_port,
                self.classification_level
            )

            # Register handlers
            self.uci_interface.register_handler(
                UCIMessageType.COMMAND,
                self._handle_command
            )
            self.uci_interface.register_handler(
                UCIMessageType.MISSION,
                self._handle_mission
            )

            self.uci_interface.start()
            return True

        except Exception as e:
            print(f"Failed to initialize UCI: {e}")
            return False

    def _handle_command(self, message: UCIMessage) -> Dict:
        """Handle UCI command messages"""
        command = message.payload.get("command")

        if command == "SET_THREAT_LEVEL":
            self.threat_level = message.payload.get("level", "GREEN")
            return {"status": "ACKNOWLEDGED", "new_threat_level": self.threat_level}

        elif command == "SET_OPERATIONAL_MODE":
            self.operational_mode = message.payload.get("mode", "NORMAL")
            return {"status": "ACKNOWLEDGED", "new_mode": self.operational_mode}

        elif command == "REQUEST_STATUS":
            return {
                "threat_level": self.threat_level,
                "operational_mode": self.operational_mode,
                "mission_status": self.oms_adapter.export_mission_status()
            }

        else:
            return {"status": "UNKNOWN_COMMAND", "command": command}

    def _handle_mission(self, message: UCIMessage) -> Dict:
        """Handle UCI mission messages"""
        mission_command = message.payload.get("mission_command")

        if mission_command == "LOAD_MISSION":
            mission_data = message.payload.get("mission_data")
            # Process mission data
            return {"status": "MISSION_LOADED"}

        elif mission_command == "START_MISSION":
            self.oms_adapter.mission_state = MissionPhase.EXECUTION
            return {"status": "MISSION_STARTED"}

        elif mission_command == "ABORT_MISSION":
            self.oms_adapter.abort_mission()
            return {"status": "MISSION_ABORTED"}

        else:
            return {"status": "UNKNOWN_MISSION_COMMAND", "command": mission_command}

    def send_slam_telemetry(self, position: List[float], orientation: List[float],
                           velocity: List[float], confidence: float):
        """Send SLAM telemetry via UCI"""
        if self.uci_interface:
            telemetry = TelemetryData(
                position=position,
                orientation=orientation,
                velocity=velocity,
                angular_velocity=[0.0, 0.0, 0.0],  # Would come from IMU
                timestamp=time.time(),
                confidence=confidence,
                source="SLAM"
            )
            self.uci_interface.send_telemetry(telemetry)

    def report_threat_detection(self, threat_type: str, position: List[float],
                              confidence: float, severity: int):
        """Report threat detection via UCI"""
        if self.uci_interface:
            threat = ThreatData(
                threat_id=str(uuid.uuid4()),
                threat_type=threat_type,
                position=position,
                velocity=[0.0, 0.0, 0.0],  # Unknown velocity
                confidence=confidence,
                severity=severity,
                timestamp=time.time()
            )
            self.uci_interface.send_threat_alert(threat)

    def emergency_response(self, emergency_type: str, description: str):
        """Send emergency response via UCI"""
        if self.uci_interface:
            self.uci_interface.send_emergency(emergency_type, description)

    def shutdown(self):
        """Shutdown defense interface"""
        if self.uci_interface:
            self.uci_interface.stop()
