"""Headless-friendly MJPEG debug viewer for the Drishti perception loop.

The laptop venv uses ``opencv-python-headless`` (and the Pi may run without a
display), so a native ``cv2.imshow`` window is not available. Instead this
module serves the annotated frames as a browser-viewable MJPEG stream:

    http://127.0.0.1:8080/          live view + layout
    http://127.0.0.1:8080/stream    raw MJPEG stream
    http://127.0.0.1:8080/snapshot  single JPEG

It is deliberately dependency-free apart from OpenCV and the standard library.
The perception thread only encodes the newest frame; the HTTP threads read the
latest JPEG, so a slow browser can never stall the 20-30 FPS hazard loop.
"""

from __future__ import annotations

import threading
import time
import json
from http import server
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config.logging_config import get_logger

logger = get_logger("perception.viewer")

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drishti Debug Viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #101214; color: #e8eaed;
         font-family: ui-sans-serif, system-ui, sans-serif; }
  header { padding: 14px 20px; border-bottom: 1px solid #2a2f35;
           display: flex; align-items: baseline; gap: 16px; }
  h1 { font-size: 18px; margin: 0; letter-spacing: .04em; }
  header span { color: #8a939d; font-size: 13px; }
  main { padding: 16px 20px 28px; }
  img { width: min(100%, 1100px); border-radius: 8px;
        border: 1px solid #2a2f35; background: #000; display: block; }
  .legend { display: flex; flex-wrap: wrap; gap: 18px; margin: 12px 0 0;
            font-size: 13px; color: #aab3bd; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
         margin-right: 6px; vertical-align: middle; }
  code { color: #9ecbff; }
</style>
</head>
<body>
<header>
  <h1>DRISHTI · LIVE DEBUG VIEW</h1>
  <span>single VL53L0X centre ray · YOLO boxes · fused distance</span>
</header>
<main>
  <img src="/stream" alt="Drishti annotated camera stream">
  <div class="legend">
    <span><i class="dot" style="background:#ff3b30"></i>primary hazard</span>
    <span><i class="dot" style="background:#32d74b"></i>ToF-fused</span>
    <span><i class="dot" style="background:#ffd60a"></i>geometric prior</span>
    <span><code>[ToF]</code> box overlaps the centre ray</span>
  </div>
</main>
</body>
</html>
"""


def _draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple,
) -> None:
    """Draw a compact label with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 4, y)
    x = max(0, min(x, canvas.shape[1] - tw - 6))
    cv2.rectangle(
        canvas,
        (x, y - th - baseline - 4),
        (x + tw + 6, y + baseline - 2),
        (12, 14, 16),
        -1,
    )
    cv2.putText(canvas, text, (x + 3, y - 3), font, scale, colour, thickness, cv2.LINE_AA)


class _ViewerServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _StreamHandler(server.BaseHTTPRequestHandler):
    server_version = "DrishtiViewer/1.0"

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        viewer: "DebugViewer" = self.server.viewer  # type: ignore[attr-defined]

        if self.path in ("/", "/index.html"):
            body = _INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/stream"):
            self._stream(viewer)
            return

        if self.path.startswith("/snapshot"):
            jpeg = viewer.latest_jpeg()
            if jpeg is None:
                self.send_error(503, "No frame captured yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if self.path.startswith("/metrics"):
            body = json.dumps(viewer.latest_metrics(), separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def _stream(self, viewer: "DebugViewer") -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        last_jpeg: Optional[bytes] = None
        while not viewer.is_stopping:
            jpeg = viewer.latest_jpeg()
            if jpeg is None or jpeg is last_jpeg:
                time.sleep(0.02)
                continue
            last_jpeg = jpeg
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            time.sleep(0.01)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("viewer HTTP: " + fmt, *args)


class DebugViewer:
    """Serve annotated perception frames over HTTP as MJPEG."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        jpeg_quality: int = 80,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._stopping = threading.Event()
        self._server: Optional[_ViewerServer] = None
        self._thread: Optional[threading.Thread] = None
        self._frames = 0
        self._latest_result: Dict[str, Any] = {}

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _ViewerServer((self.host, self.port), _StreamHandler)
        self._server.viewer = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="drishti-viewer",
            daemon=True,
        )
        self._thread.start()
        logger.info("Debug viewer listening on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        self._stopping.set()
        server_instance = self._server
        self._server = None
        if server_instance is not None:
            server_instance.shutdown()
            server_instance.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    @property
    def is_stopping(self) -> bool:
        return self._stopping.is_set()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ------------------------------------------------------------------ frames

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def latest_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {"frames": self._frames, "perception": dict(self._latest_result)}

    def update(
        self,
        frame: Optional[np.ndarray],
        fast_result: Optional[Dict[str, Any]] = None,
        status_lines: Optional[List[str]] = None,
    ) -> None:
        """Annotate and publish the newest frame (never blocks on HTTP)."""
        if frame is None or frame.size == 0:
            return

        annotated = self.annotate(frame, fast_result, status_lines)
        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self._lock:
            self._jpeg = buffer.tobytes()
            self._frames += 1
            self._latest_result = dict(fast_result or {})

    @staticmethod
    def annotate(
        frame: np.ndarray,
        fast_result: Optional[Dict[str, Any]] = None,
        status_lines: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, fusion state, ToF ray and HUD onto a copy."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        result = fast_result or {}
        obstacles = list(result.get("obstacles") or [])
        primary = result.get("primary_hazard")

        # Single centre ToF ray (the VL53L0X is not a camera).
        cv2.line(
            canvas,
            (width // 2, height // 2 - 18),
            (width // 2, height // 2 + 18),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (width // 2, height // 2), 9, (255, 255, 255), 1, cv2.LINE_AA)
        _draw_label(canvas, "ToF ray", width // 2 + 14, height // 2 - 6, (230, 230, 230))

        for obstacle in obstacles:
            bbox = obstacle.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in bbox)
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))

            is_primary = primary is not None and obstacle is primary
            fused = obstacle.get("fusion_source") == "tof_laser_fusion"

            if is_primary:
                colour = (48, 57, 255)  # red (BGR)
                thickness = 2
            elif fused:
                colour = (75, 215, 50)  # green
                thickness = 1
            else:
                colour = (10, 214, 255)  # amber
                thickness = 1

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

            label = (
                f'{obstacle.get("class_name", "?")} '
                f'{float(obstacle.get("confidence", 0.0)):.2f} '
                f'{float(obstacle.get("distance_m", 0.0)):.2f} m'
            )
            if obstacle.get("in_tof_cone"):
                label += " [ToF]"
            _draw_label(canvas, label, x1, y1 - 6, colour)

        # HUD
        lines = list(status_lines or [])
        if not lines:
            lines = [f"obstacles: {len(obstacles)}"]
        line_height = 20
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (width, min(height, 8 + line_height * len(lines))),
            (8, 10, 12),
            -1,
        )
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)

        y = 18
        for index, line in enumerate(lines):
            colour = (255, 255, 255) if index == 0 else (200, 210, 220)
            cv2.putText(
                canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA
            )
            y += line_height

        if not obstacles:
            cv2.putText(
                canvas,
                "PATH CLEAR",
                (width - 150, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (75, 215, 50),
                2,
                cv2.LINE_AA,
            )

        return canvas
