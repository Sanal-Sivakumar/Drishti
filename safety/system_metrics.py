"""Best-effort local health metrics without fabricated placeholder values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _read_number(path: Path, scale: float = 1.0) -> Optional[float]:
    try:
        return float(path.read_text(encoding="utf-8").strip()) / scale
    except (OSError, ValueError):
        return None


class SystemMetricsProvider:
    """Read battery, CPU temperature and optional GPS from the local device."""

    def snapshot(self) -> Dict[str, Any]:
        battery = None
        for candidate in Path("/sys/class/power_supply").glob("*/capacity"):
            battery = _read_number(candidate)
            if battery is not None:
                break
        cpu_temp = None
        for candidate in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            value = _read_number(candidate, scale=1000.0)
            if value is not None and 0.0 < value < 150.0:
                cpu_temp = value
                break
        gps = None
        try:
            lat, lon = os.environ.get("DRISHTI_GPS_LAT"), os.environ.get("DRISHTI_GPS_LON")
            if lat is not None and lon is not None:
                gps = {"latitude": float(lat), "longitude": float(lon)}
        except ValueError:
            gps = None
        return {
            "battery_pct": round(battery) if battery is not None else None,
            "cpu_temp_c": round(cpu_temp, 1) if cpu_temp is not None else None,
            "gps": gps,
        }
