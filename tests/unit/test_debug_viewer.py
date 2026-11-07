"""Unit tests for the headless MJPEG debug viewer."""

import urllib.request

import numpy as np

from perception.debug_viewer import DebugViewer


def _synthetic_result():
    primary = {
        "class_name": "chair",
        "confidence": 0.91,
        "distance_m": 1.2,
        "azimuth_deg": -4.0,
        "bbox": [80, 60, 220, 200],
        "fusion_source": "tof_laser_fusion",
        "in_tof_cone": True,
    }
    secondary = {
        "class_name": "person",
        "confidence": 0.72,
        "distance_m": 2.4,
        "azimuth_deg": 18.0,
        "bbox": [230, 30, 300, 210],
        "fusion_source": "geometric_perspective_prior",
        "in_tof_cone": False,
    }
    return {
        "obstacles": [primary, secondary],
        "primary_hazard": primary,
        "obstacles_count": 2,
        "latency_ms": {"total_cycle": 32.0},
    }


def test_annotate_draws_boxes_on_copy():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = _synthetic_result()

    annotated = DebugViewer.annotate(frame, result, ["cycle 1 | 31 FPS"])

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert int(annotated.sum()) > 0
    # The original frame must not be mutated.
    assert int(frame.sum()) == 0


def test_annotate_handles_empty_result():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    annotated = DebugViewer.annotate(frame, {}, [])
    assert annotated.shape == frame.shape


def test_viewer_serves_latest_snapshot():
    viewer = DebugViewer(host="127.0.0.1", port=0)
    viewer.start()
    try:
        viewer.update(np.zeros((60, 80, 3), dtype=np.uint8), {})
        jpeg = viewer.latest_jpeg()
        assert jpeg is not None
        assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI

        port = viewer._server.server_address[1]  # type: ignore[union-attr]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot", timeout=2.0) as response:
            assert response.status == 200
            assert response.read()[:2] == b"\xff\xd8"
    finally:
        viewer.stop()
