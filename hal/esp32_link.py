"""Hardware Abstraction Layer: ESP32 USB-serial bridge.

The ESP32 DevKit V1 acts as a deterministic sensor MCU. It owns the
VL53L0X I2C ranging loop and the two tactile buttons, and streams compact
CSV frames over USB serial. This module owns the host side of that
contract: framing, parsing, staleness tracking, button latching, health
metrics, and automatic reconnection.

Frame grammar (newline terminated, ASCII, 921600 8N1)::

    I,<fw>,<model>                              boot banner
    D,<seq>,<t_ms>,<mm_f>,<mm_raw>,<st>,<btn>   range sample (~30-50 Hz)
    B,<id>,<t_ms>,<edge>                        button edge (1=press)
    S,<hz>,<invalid_pct>,<i2c_ok>,<link_ok>      1 Hz status
    E,<code>,<msg>                              error

Host commands::

    V                firmware banner request (sent on every connect)
    P                heartbeat, every 500 ms
    T,<ms>           set measurement timing budget
    R                soft reset

The link never blocks the perception loop: a daemon reader thread owns the
serial port and publishes the latest immutable sample. ``get_distance``
style queries are O(1).
"""

from __future__ import annotations

import glob
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Optional

from config.logging_config import get_logger

logger = get_logger("hal.esp32")

# Ordered by preference: stable by-id symlinks first, then raw device nodes.
_PORT_PATTERNS = (
    "/dev/serial/by-id/*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/tty.usbserial*",
    "/dev/tty.wchusbserial*",
)


@dataclass(frozen=True)
class TofSample:
    """One immutable VL53L0X range sample received from the ESP32."""

    seq: int
    device_ms: int
    mm_filtered: int
    mm_raw: int
    status: int
    button_mask: int
    recv_monotonic: float


@dataclass(frozen=True)
class ButtonEvent:
    """A latched button press edge received from the ESP32."""

    button_id: int
    device_ms: int
    recv_monotonic: float


@dataclass
class LinkStatus:
    """Snapshot of the serial link health, suitable for logs and the viewer."""

    connected: bool = False
    port: str = ""
    firmware: str = ""
    sensor_model: str = ""
    hz: float = 0.0
    invalid_pct: float = 0.0
    seq_drops: int = 0
    reconnects: int = 0
    samples: int = 0
    i2c_ok: bool = True
    device_link_ok: bool = True
    last_error: str = ""
    age_ms: float = float("inf")


class Esp32Link:
    """Threaded USB-serial transport for the Drishti ESP32 sensor MCU."""

    def __init__(
        self,
        port: str = "auto",
        baud: int = 921600,
        stale_ms: float = 120.0,
        reconnect_s: float = 1.0,
        read_timeout_s: float = 0.2,
        serial_factory: Optional[Callable[[str], Any]] = None,
        heartbeat_interval_s: float = 0.5,
    ) -> None:
        self._port_override = port or "auto"
        self.baud = int(baud)
        self.stale_ms = float(stale_ms)
        self.reconnect_s = float(reconnect_s)
        self.read_timeout_s = float(read_timeout_s)
        self._serial_factory = serial_factory
        self._heartbeat_interval_s = float(heartbeat_interval_s)

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial: Any = None
        self._port = ""
        self._connected = False
        self._last_error = ""

        self._latest: Optional[TofSample] = None
        self._buttons: Deque[ButtonEvent] = deque(maxlen=64)
        self._last_seq: Optional[int] = None
        self._last_rx = 0.0
        self._sample_times: Deque[float] = deque(maxlen=120)
        self._samples = 0
        self._invalid = 0
        self._seq_drops = 0
        self._reconnects = 0
        self._button_mask = 0

        self._firmware = ""
        self._sensor_model = ""
        self._device_hz = 0.0
        self._device_invalid_pct = 0.0
        self._i2c_ok = True
        self._device_link_ok = True
        self._consecutive_timeouts = 0
        self._last_recovery_reset = 0.0

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the background reader thread (idempotent)."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, name="esp32-link", daemon=True)
        self._thread.start()
        logger.info("ESP32 link reader started (port=%s, baud=%d).", self._port_override, self.baud)

    def stop(self) -> None:
        """Stop the reader thread and close the serial port."""
        with self._lock:
            self._running = False
            ser = self._serial
            self._serial = None
            self._connected = False
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._thread = None
        logger.info("ESP32 link reader stopped.")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def port(self) -> str:
        with self._lock:
            return self._port

    def request_reset(self) -> None:
        """Send a soft-reset command ('R') to the ESP32 MCU."""
        with self._lock:
            ser = self._serial
        if ser is not None:
            try:
                ser.write(b"R\n")
                logger.info("Sent soft-reset request to ESP32.")
            except Exception as exc:
                logger.warning("Failed to send soft-reset to ESP32: %s", exc)

    # ------------------------------------------------------------------ reader

    def _run(self) -> None:
        last_heartbeat = 0.0
        while True:
            with self._lock:
                if not self._running:
                    return
                ser = self._serial

            if ser is None:
                ser = self._open_serial()
                if ser is None:
                    time.sleep(self.reconnect_s)
                    continue
                last_heartbeat = 0.0

            now = time.monotonic()
            if now - last_heartbeat >= self._heartbeat_interval_s:
                try:
                    ser.write(b"P\n")
                    last_heartbeat = now
                except Exception as exc:
                    if not self._is_running():
                        return
                    self._drop_connection(str(exc))
                    continue

            try:
                raw = ser.readline()
            except Exception as exc:
                if not self._is_running():
                    return
                self._drop_connection(str(exc))
                continue

            if raw:
                self.feed_line(raw)

    def _candidate_ports(self) -> List[str]:
        found: List[str] = []
        seen = set()
        for pattern in _PORT_PATTERNS:
            for path in sorted(glob.glob(pattern)):
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                found.append(path)
        return found

    def _open_serial(self) -> Any:
        if self._serial_factory is None:
            try:
                import serial  # noqa: F401  (availability probe)
            except ImportError:
                self._record_error("pyserial is not installed (pip install pyserial)")
                return None

        if self._port_override != "auto":
            ports = [self._port_override]
        else:
            ports = self._candidate_ports()

        if not ports:
            self._record_error("no serial port found (is the ESP32 plugged in?)")
            return None

        errors: List[str] = []
        for port in ports:
            try:
                if self._serial_factory is not None:
                    ser = self._serial_factory(port)
                else:
                    import serial

                    ser = serial.Serial(port=port, baudrate=self.baud, timeout=self.read_timeout_s)
                with self._lock:
                    self._serial = ser
                    self._port = port
                    self._connected = True
                    self._reconnects += 1
                    self._last_error = ""
                # Ask for the firmware banner on every (re)connect; the MCU's
                # boot-time banner may have been sent before we opened the port.
                try:
                    ser.write(b"V\n")
                except Exception:
                    pass
                logger.info("ESP32 link connected on %s @ %d baud.", port, self.baud)
                return ser
            except Exception as exc:
                errors.append(f"{port}: {exc}")

        self._record_error("; ".join(errors[:3]))
        return None

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._connected = False
            self._last_error = message
        logger.debug("ESP32 link unavailable: %s", message)

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    def _drop_connection(self, reason: str) -> None:
        with self._lock:
            if not self._running:
                return
            ser = self._serial
            self._serial = None
            self._connected = False
            self._last_error = reason
            self._buttons.clear()
            self._last_seq = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        logger.warning("ESP32 link lost (%s); reconnecting...", reason)

    # ------------------------------------------------------------------ parsing

    def feed_line(self, raw: Any, now: Optional[float] = None) -> None:
        """Parse a single protocol line.

        Public so that tests and ``tools/fake_esp32.py`` can drive the link
        without a physical serial port.
        """
        if isinstance(raw, (bytes, bytearray)):
            text = raw.decode("ascii", errors="ignore")
        else:
            text = str(raw)
        text = text.strip()
        if not text:
            return

        now = time.monotonic() if now is None else float(now)
        parts = text.split(",")
        tag = parts[0]
        try:
            if tag == "D":
                self._parse_data(parts, now)
            elif tag == "B":
                self._parse_button(parts, now)
            elif tag == "I":
                self._parse_info(parts)
            elif tag == "S":
                self._parse_status(parts)
            elif tag == "E":
                self._parse_error(parts)
            else:
                logger.debug("Ignoring unknown ESP32 frame: %r", text)
        except (ValueError, IndexError):
            logger.debug("Malformed ESP32 frame: %r", text)

    def _parse_data(self, parts: List[str], now: float) -> None:
        if len(parts) < 7:
            raise ValueError("short data frame")
        seq = int(parts[1])
        device_ms = int(parts[2])
        mm_filtered = int(parts[3])
        mm_raw = int(parts[4])
        status = int(parts[5])
        button_mask = int(parts[6])

        with self._lock:
            if self._last_seq is not None and seq > self._last_seq + 1:
                self._seq_drops += seq - self._last_seq - 1
            self._last_seq = seq
            self._samples += 1
            if status != 0:
                self._invalid += 1
            if status == 254:
                self._consecutive_timeouts += 1
            elif status == 0:
                self._consecutive_timeouts = 0

            if self._consecutive_timeouts >= 25 and (now - self._last_recovery_reset) > 5.0:
                self._last_recovery_reset = now
                self._consecutive_timeouts = 0
                logger.warning("ESP32 ToF hardware timeout persistent (status 254); issuing auto-recovery reset.")
                ser = self._serial
                if ser is not None:
                    try:
                        ser.write(b"R\n")
                    except Exception:
                        pass
            self._sample_times.append(now)
            self._button_mask = button_mask
            self._last_rx = now
            self._latest = TofSample(
                seq=seq,
                device_ms=device_ms,
                mm_filtered=mm_filtered,
                mm_raw=mm_raw,
                status=status,
                button_mask=button_mask,
                recv_monotonic=now,
            )

    def _parse_button(self, parts: List[str], now: float) -> None:
        if len(parts) < 4:
            raise ValueError("short button frame")
        button_id = int(parts[1])
        edge = int(parts[3])
        if edge != 1 or button_id not in (1, 2, 3):
            return
        with self._lock:
            self._buttons.append(
                ButtonEvent(
                    button_id=button_id,
                    device_ms=int(parts[2]),
                    recv_monotonic=now,
                )
            )
            self._last_rx = now

    def _parse_info(self, parts: List[str]) -> None:
        with self._lock:
            self._firmware = parts[1] if len(parts) > 1 else ""
            self._sensor_model = parts[2] if len(parts) > 2 else ""

    def _parse_status(self, parts: List[str]) -> None:
        with self._lock:
            if len(parts) > 1:
                self._device_hz = float(parts[1])
            if len(parts) > 2:
                self._device_invalid_pct = float(parts[2])
            if len(parts) > 3:
                self._i2c_ok = bool(int(parts[3]))
            if len(parts) > 4:
                self._device_link_ok = bool(int(parts[4]))

    def _parse_error(self, parts: List[str]) -> None:
        message = ",".join(parts[1:]) if len(parts) > 1 else "unknown ESP32 error"
        with self._lock:
            self._last_error = message
        logger.warning("ESP32 firmware error: %s", message)

    # ------------------------------------------------------------------ queries

    def latest(self) -> Optional[TofSample]:
        """Return the most recent sample (immutable), or None."""
        with self._lock:
            return self._latest

    def age_ms(self, now: Optional[float] = None) -> float:
        """Age of the latest sample in milliseconds (``inf`` when empty)."""
        sample = self.latest()
        if sample is None:
            return float("inf")
        current = time.monotonic() if now is None else float(now)
        return max(0.0, (current - sample.recv_monotonic) * 1000.0)

    def take_button_press(self, button_id: int) -> bool:
        """Pop one latched press event for ``button_id`` (FIFO)."""
        with self._lock:
            for index, event in enumerate(self._buttons):
                if event.button_id == button_id:
                    del self._buttons[index]
                    return True
        return False

    def pop_button_events(self) -> List[ButtonEvent]:
        """Drain and return all pending button events (used by tests/diagnostics)."""
        with self._lock:
            events = list(self._buttons)
            self._buttons.clear()
        return events

    def status(self, now: Optional[float] = None) -> LinkStatus:
        """Return a consistent health snapshot."""
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            times = list(self._sample_times)
            if len(times) >= 5 and times[-1] - times[0] >= 0.25:
                hz = (len(times) - 1) / (times[-1] - times[0])
            else:
                hz = 0.0
            invalid_pct = (100.0 * self._invalid / self._samples) if self._samples else 0.0
            age = float("inf")
            if self._latest is not None:
                age = max(0.0, (current - self._latest.recv_monotonic) * 1000.0)
            return LinkStatus(
                connected=self._connected,
                port=self._port,
                firmware=self._firmware,
                sensor_model=self._sensor_model,
                hz=round(hz, 2),
                invalid_pct=round(invalid_pct, 2),
                seq_drops=self._seq_drops,
                reconnects=self._reconnects,
                samples=self._samples,
                i2c_ok=self._i2c_ok,
                device_link_ok=self._device_link_ok,
                last_error=self._last_error,
                age_ms=round(age, 1) if age != float("inf") else float("inf"),
            )
