"""Safety readiness state for the Drishti navigation pipeline.

Navigation output is only authoritative when every required production
component is healthy.  Synthetic devices remain useful for CI and demos, but
are never silently promoted to production sensor data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List


class ReadinessLevel(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"
    SIMULATION = "simulation"


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str
    critical: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "critical": self.critical}


@dataclass
class SystemReadiness:
    mode: str
    issues: List[ReadinessIssue] = field(default_factory=list)

    @property
    def level(self) -> ReadinessLevel:
        if self.mode in ("synthetic", "simulation"):
            return ReadinessLevel.SIMULATION
        if any(issue.critical for issue in self.issues):
            return ReadinessLevel.NOT_READY
        if self.issues:
            return ReadinessLevel.DEGRADED
        return ReadinessLevel.READY

    @property
    def navigation_authoritative(self) -> bool:
        return self.level == ReadinessLevel.READY

    def extend(self, issues: Iterable[ReadinessIssue]) -> None:
        self.issues.extend(issues)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "level": self.level.value,
            "navigation_authoritative": self.navigation_authoritative,
            "issues": [issue.as_dict() for issue in self.issues],
        }
