"""Bounded MQTT telemetry with authentication, persistence and PUBACK tracking."""

from __future__ import annotations

import json
import os
import ssl
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from config.logging_config import get_logger
from safety.system_metrics import SystemMetricsProvider

logger = get_logger("safety.telemetry")


class TelemetryClient:
    """Publish real local metrics while retaining a bounded reconnect buffer."""

    def __init__(
        self,
        device_id: str = "drishti-wearable-01",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        use_tls: bool = False,
        ca_certs: Optional[str] = None,
        qos: int = 1,
        username: Optional[str] = None,
        password: Optional[str] = None,
        buffer_limit: int = 500,
        persistence_path: Optional[str] = None,
        metrics_provider: Optional[SystemMetricsProvider] = None,
        require_auth: bool = True,
    ) -> None:
        self.device_id = device_id
        self.broker_host, self.broker_port = broker_host, int(broker_port)
        self.use_tls, self.ca_certs, self.qos = bool(use_tls), ca_certs, int(qos)
        self.username = username or os.environ.get("DRISHTI_MQTT_USERNAME")
        self.password = password or os.environ.get("DRISHTI_MQTT_PASSWORD")
        self.buffer_limit = max(1, int(buffer_limit))
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.metrics_provider = metrics_provider or SystemMetricsProvider()
        self.require_auth = bool(require_auth)
        self.is_connected = False
        self.buffered_telemetry: Deque[Dict[str, Any]] = deque(maxlen=self.buffer_limit)
        self.pending_acks: Dict[int, str] = {}
        self.lock = threading.RLock()
        self._connected_event = threading.Event()
        base = f"drishti/device/{self.device_id}"
        self.topic_heartbeat, self.topic_status = f"{base}/heartbeat", f"{base}/status"
        self.topic_emergency, self.topic_events = f"{base}/emergency", f"{base}/events"
        self.mqtt_client = None
        self._load_persisted()
        self._build_client()

    def _build_client(self) -> None:
        try:
            import paho.mqtt.client as mqtt
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=self.device_id)
            except (ImportError, AttributeError):
                client = mqtt.Client(client_id=self.device_id)
            if self.username:
                client.username_pw_set(self.username, self.password)
            if self.use_tls:
                client.tls_set(
                    ca_certs=self.ca_certs if self.ca_certs and os.path.exists(self.ca_certs) else None,
                    tls_version=ssl.PROTOCOL_TLS_CLIENT,
                )
                client.tls_insecure_set(False)
            client.on_connect, client.on_disconnect, client.on_publish = (
                self._on_connect, self._on_disconnect, self._on_publish
            )
            client.will_set(
                self.topic_status,
                json.dumps({"device_id": self.device_id, "status": "OFFLINE_UNEXPECTED"}),
                qos=self.qos,
                retain=True,
            )
            self.mqtt_client = client
        except Exception as exc:
            logger.warning("MQTT client unavailable: %s", exc)

    def connect(self, wait_s: float = 1.0) -> bool:
        if self.mqtt_client is None:
            return False
        if self.require_auth and (not self.username or not self.password):
            logger.warning("Telemetry authentication is required; MQTT credentials are missing.")
            return False
        try:
            self._connected_event.clear()
            self.mqtt_client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.mqtt_client.loop_start()
            self._connected_event.wait(timeout=max(0.0, wait_s))
        except Exception as exc:
            logger.info("Telemetry offline; retaining bounded local buffer: %s", exc)
            self.is_connected = False
        return self.is_connected

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        try:
            self.is_connected = int(reason_code) == 0
        except (TypeError, ValueError):
            self.is_connected = str(reason_code).lower() in ("success", "0")
        if self.is_connected:
            self._connected_event.set()
            self._replay_buffer()

    def _on_disconnect(self, client: Any, userdata: Any, *args: Any) -> None:
        self.is_connected = False
        self._connected_event.clear()

    def _on_publish(self, client: Any, userdata: Any, mid: int, *args: Any) -> None:
        with self.lock:
            event_id = self.pending_acks.pop(int(mid), None)
            if event_id:
                self.buffered_telemetry = deque(
                    (x for x in self.buffered_telemetry if x.get("_event_id") != event_id),
                    maxlen=self.buffer_limit,
                )
                self._persist()

    def _append(self, payload: Dict[str, Any], topic: str) -> Dict[str, Any]:
        entry = {**payload, "_event_id": uuid.uuid4().hex, "_topic": topic}
        with self.lock:
            self.buffered_telemetry.append(entry)
            self._persist()
        self._publish_entry(entry)
        return payload

    def _publish_entry(self, entry: Dict[str, Any]) -> None:
        if not self.is_connected or self.mqtt_client is None:
            return
        public = {k: v for k, v in entry.items() if not k.startswith("_")}
        try:
            info = self.mqtt_client.publish(entry["_topic"], json.dumps(public), qos=self.qos)
            if self.qos:
                with self.lock:
                    self.pending_acks[int(info.mid)] = str(entry["_event_id"])
        except Exception as exc:
            logger.warning("Telemetry publish failed; retaining event: %s", exc)
            self.is_connected = False

    def _replay_buffer(self) -> None:
        with self.lock:
            entries = list(self.buffered_telemetry)
        for entry in entries:
            self._publish_entry(entry)

    def build_heartbeat_payload(
        self,
        battery_pct: Optional[int] = None,
        cpu_temp_c: Optional[float] = None,
        system_state: str = "ACTIVE_NAVIGATION",
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        readiness: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        measured = self.metrics_provider.snapshot()
        gps = measured.get("gps")
        if gps_lat is not None and gps_lon is not None:
            gps = {"latitude": round(gps_lat, 6), "longitude": round(gps_lon, 6)}
        return {
            "device_id": self.device_id,
            "timestamp": round(time.time(), 3),
            "battery_pct": battery_pct if battery_pct is not None else measured.get("battery_pct"),
            "cpu_temp_c": round(cpu_temp_c, 1) if cpu_temp_c is not None else measured.get("cpu_temp_c"),
            "system_state": system_state,
            "gps": gps,
            "readiness": readiness,
        }

    def publish_heartbeat(
        self,
        battery_pct: Optional[int] = None,
        cpu_temp_c: Optional[float] = None,
        system_state: str = "ACTIVE_NAVIGATION",
        readiness: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._append(
            self.build_heartbeat_payload(battery_pct, cpu_temp_c, system_state, readiness=readiness),
            self.topic_heartbeat,
        )

    def publish_event(self, event_type: str = "USER_NARRATION_REQUEST", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "device_id": self.device_id, "timestamp": round(time.time(), 3),
            "event_type": event_type, "alert_type": event_type,
            "details": details or {}, "status": "INFORMATIONAL",
        }
        return self._append(payload, self.topic_events)

    def publish_emergency_alert(self, alert_type: str = "SOS_MANUAL_BUTTON", obstacle_class: Optional[str] = None, distance_m: Optional[float] = None, snapshot_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "device_id": self.device_id, "timestamp": round(time.time(), 3),
            "alert_type": alert_type, "obstacle_class": obstacle_class,
            "distance_m": distance_m,
            "snapshot_metadata": snapshot_metadata or {"faces_redacted": None, "privacy_filtered": False},
            "status": "REQUIRES_CAREGIVER_ATTENTION",
        }
        return self._append(payload, self.topic_emergency)

    def _load_persisted(self) -> None:
        if self.persistence_path is None or not self.persistence_path.exists():
            return
        try:
            for line in self.persistence_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if isinstance(item, dict) and "_topic" in item:
                    self.buffered_telemetry.append(item)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load telemetry buffer: %s", exc)

    def _persist(self) -> None:
        if self.persistence_path is None:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.persistence_path.with_suffix(".tmp")
            temporary.write_text(
                "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in self.buffered_telemetry),
                encoding="utf-8",
            )
            temporary.replace(self.persistence_path)
        except OSError as exc:
            logger.warning("Could not persist telemetry buffer: %s", exc)

    def disconnect(self) -> None:
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
        self.is_connected = False
        self._connected_event.clear()
