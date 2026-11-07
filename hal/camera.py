"""Hardware Abstraction Layer: Camera Subsystem.

Provides unified interface for both physical cameras (V4L2/libcamera/OpenCV)
and headless synthetic frame generators for testing.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import os
import shutil
import subprocess
import time
import numpy as np
from config.logging_config import get_logger

logger = get_logger("hal.camera")


class BaseCameraSource(ABC):
    """Abstract base class for all camera input sources."""

    @property
    def is_synthetic(self) -> bool:
        """Flag indicating whether the camera source is a synthetic generator."""
        return False

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, BGR_image_array)
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera hardware resources."""
        pass

    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Return camera properties (resolution, target FPS, backend)."""
        pass


class SyntheticCameraSource(BaseCameraSource):
    """Deterministic, headless camera simulator for testing without physical sensors."""

    @property
    def is_synthetic(self) -> bool:
        return True

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_index = 0
        self.is_open = True
        self.start_time = time.time()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Generate a synthetic test frame with an animated target obstacle."""
        if not self.is_open:
            return False, None

        # Create gradient background (dark gray to deep blue)
        y_coords = np.linspace(30, 80, self.height, dtype=np.uint8)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = y_coords[:, None]  # Blue channel
        frame[:, :, 1] = 40                 # Green channel
        frame[:, :, 2] = 40                 # Red channel

        # Animate a simulated obstacle (moving horizontally)
        t = self.frame_index * (1.0 / self.fps)
        center_x = int(self.width / 2 + (self.width / 4) * np.sin(2 * np.pi * 0.2 * t))
        center_y = int(self.height / 2 + (self.height / 8) * np.cos(2 * np.pi * 0.2 * t))
        box_w, box_h = 100, 140

        x1 = max(0, center_x - box_w // 2)
        y1 = max(0, center_y - box_h // 2)
        x2 = min(self.width, center_x + box_w // 2)
        y2 = min(self.height, center_y + box_h // 2)

        # Draw simulated obstacle (warm amber rectangle)
        frame[y1:y2, x1:x2, :] = [20, 180, 240]

        # Draw a small high-contrast "face" or "marker" on top of the obstacle
        face_w, face_h = 40, 40
        fx1 = max(0, center_x - face_w // 2)
        fy1 = max(0, y1 + 10)
        fx2 = min(self.width, fx1 + face_w)
        fy2 = min(self.height, fy1 + face_h)
        frame[fy1:fy2, fx1:fx2, :] = [180, 210, 255]

        self.frame_index += 1
        return True, frame

    def release(self) -> None:
        self.is_open = False

    def get_properties(self) -> Dict[str, Any]:
        return {
            "backend": "synthetic",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_index,
        }


class OpenCVCameraSource(BaseCameraSource):
    """Camera driver using OpenCV V4L2 / GStreamer for physical webcams."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

        try:
            import cv2
            self.cv2 = cv2
            self.cap = cv2.VideoCapture(device_index)
            if self.cap.isOpened():
                # MJPG keeps many laptop webcams at 30 FPS instead of silently
                # dropping to 10 FPS with the default YUYV format.
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                logger.info(
                    "OpenCV camera %d opened: %dx%d @ %.0f FPS requested.",
                    device_index, width, height, fps,
                )
            else:
                logger.debug("OpenCV VideoCapture device index %d could not be opened.", device_index)
        except ImportError:
            logger.debug("OpenCV (cv2) not installed; OpenCVCameraSource disabled.")
            self.cap = None
        except Exception as e:
            logger.warning("Error initializing OpenCV camera device %d: %s", device_index, e)
            self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_properties(self) -> Dict[str, Any]:
        return {
            "backend": "opencv",
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_opened": self.cap.isOpened() if self.cap else False,
        }


class V4L2CameraSource(BaseCameraSource):
    """Direct V4L2 raw video stream driver for Linux/laptop webcams without OpenCV dependency."""

    def __init__(self, device_path: str = "/dev/video0", width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3
        self.proc = None
        self.is_open = False

        if not os.path.exists(device_path):
            logger.debug("V4L2 camera path %s does not exist on filesystem.", device_path)
            return

        if not shutil.which("ffmpeg"):
            logger.debug("ffmpeg executable not found in PATH; V4L2 raw stream disabled.")
            return

        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-f", "v4l2",
            "-video_size", f"{width}x{height}",
            "-i", device_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-"
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_bytes * 2
            )
            self.is_open = True
            logger.info("Spawned V4L2 ffmpeg capture stream for %s (%dx%d @ %d FPS).", device_path, width, height, fps)
        except Exception as e:
            logger.warning("Failed to spawn ffmpeg V4L2 stream for %s: %s", device_path, e)
            self.proc = None
            self.is_open = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open or self.proc is None or self.proc.stdout is None:
            return False, None
        try:
            raw = self.proc.stdout.read(self.frame_bytes)
            if len(raw) == self.frame_bytes:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                return True, frame
            return False, None
        except Exception as e:
            logger.debug("V4L2 read_frame exception: %s", e)
            return False, None

    def release(self) -> None:
        self.is_open = False
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    logger.warning("V4L2 ffmpeg process did not terminate within timeout; killing PID %d.", self.proc.pid)
                    self.proc.kill()
                    self.proc.wait(timeout=0.5)
            except Exception as e:
                logger.warning("Error releasing V4L2 process: %s", e)
            self.proc = None

    def __enter__(self) -> "V4L2CameraSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def get_properties(self) -> Dict[str, Any]:
        return {
            "backend": "v4l2_ffmpeg",
            "device_path": self.device_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_open": self.is_open,
        }


def _frame_is_color(frame: Optional[np.ndarray]) -> bool:
    """Heuristic: an IR / Windows Hello camera yields near-equal BGR channels."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
        return False
    means = frame.reshape(-1, 3).mean(axis=0)
    std_diff = abs(float(np.std(frame[:, :, 0]) - np.std(frame[:, :, 2])))
    return float(means.max() - means.min()) >= 4.0 or std_diff >= 3.0


def _probe_opencv_cameras(
    width: int, height: int, fps: int, device_index: Optional[int]
) -> Optional[OpenCVCameraSource]:
    """Open the requested index, or probe indices 0..4 preferring a colour camera."""
    if device_index is not None and device_index >= 0:
        cam = OpenCVCameraSource(device_index=device_index, width=width, height=height, fps=fps)
        success, frame = cam.read_frame()
        if success and frame is not None and frame.size > 0:
            return cam
        cam.release()
        return None

    first_working: Optional[int] = None
    colour_index: Optional[int] = None

    for index in range(5):
        cam = OpenCVCameraSource(device_index=index, width=width, height=height, fps=fps)
        # Read a couple of frames to let sensor auto-exposure and white balance settle
        success = False
        frame = None
        for _ in range(3):
            s, f = cam.read_frame()
            if s and f is not None:
                success, frame = s, f
        if success and frame is not None and frame.size > 0:
            if first_working is None:
                first_working = index
            if _frame_is_color(frame):
                colour_index = index
                cam.release()
                break
        cam.release()

    chosen = colour_index if colour_index is not None else first_working
    if chosen is None:
        return None

    cam = OpenCVCameraSource(device_index=chosen, width=width, height=height, fps=fps)
    success, _ = cam.read_frame()
    if success:
        if colour_index is None:
            logger.warning(
                "Only a monochrome/IR-capable camera was found (index %d); "
                "object detection quality may be degraded.", chosen,
            )
        logger.info("Camera probe selected index %d.", chosen)
        return cam
    cam.release()
    return None


def create_camera_source(
    backend: str = "auto",
    device_path: str = "/dev/video0",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    device_index: Optional[int] = None,
) -> BaseCameraSource:
    """Instantiate camera driver with automatic graceful fallback to synthetic generator.

    Args:
        device_index: Explicit OpenCV index. ``None`` probes 0..4 and prefers a
            colour camera, skipping IR / Windows Hello devices.
    """
    if backend == "synthetic":
        return SyntheticCameraSource(width=width, height=height, fps=fps)

    # 1. Try OpenCV if explicitly requested or auto
    if backend in ("opencv", "auto"):
        try:
            cam = _probe_opencv_cameras(width, height, fps, device_index)
            if cam is not None:
                logger.info("Successfully acquired hardware camera via OpenCV backend.")
                return cam
        except Exception as e:
            logger.debug("OpenCV camera acquisition probe failed: %s", e)

    # 2. Try V4L2 ffmpeg stream
    if backend in ("v4l2", "auto"):
        try:
            cam = V4L2CameraSource(device_path=device_path, width=width, height=height, fps=fps)
            success, frame = cam.read_frame()
            if success and frame is not None and frame.size > 0:
                logger.info("Successfully acquired hardware camera via V4L2 raw stream (%s).", device_path)
                return cam
            cam.release()
        except Exception as e:
            logger.debug("V4L2 camera acquisition probe failed for %s: %s", device_path, e)

    # 3. Graceful fallback to synthetic camera simulator
    logger.warning(
        "Physical camera [%s] unavailable or yielded empty frames. "
        "Engaging autonomous SyntheticCameraSource fallback per system resiliency spec.",
        device_path,
    )
    return SyntheticCameraSource(width=width, height=height, fps=fps)


