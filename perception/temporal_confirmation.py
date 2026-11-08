"""Short-horizon confirmation for narration, independent of urgent cues."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _direction(azimuth_deg: float) -> str:
    if azimuth_deg < -15.0:
        return "left"
    if azimuth_deg > 15.0:
        return "right"
    return "ahead"


class TemporalDetectionConfirmer:
    """Require consecutive class/direction observations before narration."""

    def __init__(self, required_frames: int = 2, max_distance_delta_m: float = 0.6) -> None:
        self.required_frames = max(1, int(required_frames))
        self.max_distance_delta_m = float(max_distance_delta_m)
        self._tracks: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def update(self, obstacles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        for obstacle in obstacles:
            key = (
                str(obstacle.get("class_name", "obstacle")),
                _direction(float(obstacle.get("azimuth_deg", 0.0))),
            )
            seen.add(key)
            previous = self._tracks.get(key)
            distance = float(obstacle.get("distance_m", 999.0))
            consecutive = 1
            if previous is not None and abs(float(previous["distance_m"]) - distance) <= self.max_distance_delta_m:
                consecutive = int(previous["count"]) + 1
            self._tracks[key] = {"count": consecutive, "distance_m": distance, "obstacle": dict(obstacle)}

        for key in list(self._tracks):
            if key not in seen:
                del self._tracks[key]

        confirmed = []
        for track in self._tracks.values():
            if track["count"] >= self.required_frames:
                item = dict(track["obstacle"])
                item["confirmation_frames"] = int(track["count"])
                confirmed.append(item)
        return confirmed
