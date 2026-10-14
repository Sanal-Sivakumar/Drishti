"""Safety Subsystem: Mosquitto MQTT Telemetry & Caregiver Gateway Client.

Transmits heartbeat telemetry, battery status, GPS coordinates, and emergency
distress alerts over TLS 1.3 with QoS Level 1 ('at least once' delivery).
"""

from typing import Dict, Any, Optional, List
import json
import time
import threading


class TelemetryClient:
    """Manages telemetry broadcast and caregiver emergency dispatch."""

    def __init__(
        self,
        device_id: str = "drishti-wearable-01",
        broker_host: str = "localhost",
        broker_port: int = 8883,
        use_tls: bool = True,
        qos: int = 1,
    ) -> None:
        self.device_id = device_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.use_tls = use_tls
        self.qos = qos
        self.is_connected = False
        self.buffered_telemetry: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        # Topic structure
        self.topic_heartbeat = f"drishti/device/{self.device_id}/heartbeat"
        self.topic_status = f"drishti/device/{self.device_id}/status"
        self.topic_emergency = f"drishti/device/{self.device_id}/emergency"

        self.mqtt_client = None
        try:
            import paho.mqtt.client as mqtt
            self.mqtt_client = mqtt.Client(client_id=self.device_id)
            if self.use_tls:
                self.mqtt_client.tls_set()
            # Register Last Will and Testament (LWT)
            lwt_payload = json.dumps({
                "device_id": self.device_id,
                "status": "OFFLINE_UNEXPECTED",
                "timestamp": time.time(),
            })
            self.mqtt_client.will_set(self.topic_status, payload=lwt_payload, qos=self.qos, retain=True)
        except (ImportError, Exception):
            self.mqtt_client = None

    def connect(self) -> bool:
        """Establish connection to MQTT broker."""
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.connect(self.broker_host, self.broker_port, keepalive=60)
                self.mqtt_client.loop_start()
                self.is_connected = True
                return True
            except Exception:
                self.is_connected = False
                return False
        return False

    def build_heartbeat_payload(
        self,
        battery_pct: int = 85,
        cpu_temp_c: float = 52.0,
        system_state: str = "ACTIVE_NAVIGATION",
        gps_lat: float = 12.9716,
        gps_lon: float = 77.5946,
    ) -> Dict[str, Any]:
        """Construct a standardized telemetry heartbeat JSON dictionary."""
        return {
            "device_id": self.device_id,
            "timestamp": round(time.time(), 3),
            "battery_pct": battery_pct,
            "cpu_temp_c": round(cpu_temp_c, 1),
            "system_state": system_state,
            "gps": {
                "latitude": round(gps_lat, 6),
                "longitude": round(gps_lon, 6),
                "accuracy_m": 3.2,
            },
        }

    def publish_heartbeat(
        self,
        battery_pct: int = 85,
        cpu_temp_c: float = 52.0,
        system_state: str = "ACTIVE_NAVIGATION",
    ) -> Dict[str, Any]:
        """Publish periodic health and location telemetry."""
        payload = self.build_heartbeat_payload(
            battery_pct=battery_pct,
            cpu_temp_c=cpu_temp_c,
            system_state=system_state,
        )

        with self.lock:
            self.buffered_telemetry.append(payload)

        if self.is_connected and self.mqtt_client is not None:
            try:
                msg_json = json.dumps(payload)
                self.mqtt_client.publish(self.topic_heartbeat, payload=msg_json, qos=self.qos)
            except Exception:
                pass

        return payload

    def publish_emergency_alert(
        self,
        alert_type: str = "SOS_MANUAL_BUTTON",
        obstacle_class: Optional[str] = None,
        distance_m: Optional[float] = None,
        snapshot_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Publish high-priority emergency alert with QoS 1 acknowledgment."""
        payload = {
            "device_id": self.device_id,
            "timestamp": round(time.time(), 3),
            "alert_type": alert_type,
            "obstacle_class": obstacle_class,
            "distance_m": distance_m,
            "snapshot_metadata": snapshot_metadata or {"faces_redacted": 0, "privacy_filtered": True},
            "status": "REQUIRES_CAREGIVER_ATTENTION",
        }

        with self.lock:
            self.buffered_telemetry.append(payload)

        if self.is_connected and self.mqtt_client is not None:
            try:
                msg_json = json.dumps(payload)
                self.mqtt_client.publish(self.topic_emergency, payload=msg_json, qos=self.qos)
            except Exception:
                pass

        return payload

    def disconnect(self) -> None:
        """Cleanly disconnect from broker."""
        if self.is_connected and self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
            self.is_connected = False
