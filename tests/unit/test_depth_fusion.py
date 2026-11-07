"""Unit tests for Camera-ToF Spatial Depth Fusion."""

import pytest
import numpy as np
from perception.depth_fusion import SpatialDepthFuser


def test_ray_angle_computation():
    """Verify pixel ray angles match pinhole geometry."""
    fuser = SpatialDepthFuser(focal_length_px=500.0, principal_point_x=320.0, principal_point_y=240.0)

    # Center pixel (cx, cy) -> 0 radians azimuth and elevation
    az, el = fuser.compute_ray_angles(320.0, 240.0)
    assert np.isclose(az, 0.0, atol=1e-5)
    assert np.isclose(el, 0.0, atol=1e-5)

    # Right pixel (u = 320 + 500 = 820) -> arctan(1.0) = 45 degrees = pi/4 rad
    az_right, _ = fuser.compute_ray_angles(820.0, 240.0)
    assert np.isclose(az_right, np.pi / 4, atol=1e-5)


def test_tof_cone_intersection():
    """Verify bounding box overlap check with ToF 27-degree acceptance cone."""
    fuser = SpatialDepthFuser(
        focal_length_px=500.0,
        principal_point_x=320.0,
        principal_point_y=240.0,
        tof_cone_half_angle_deg=13.5
    )

    # Centered obstacle [280, 200, 360, 280] -> definitely inside cone
    assert fuser.is_in_tof_cone(280.0, 360.0) is True

    # Far-left obstacle [20, 200, 80, 280] -> u around 50 -> theta = arctan((50-320)/500) = -28 deg -> outside 13.5 deg cone
    assert fuser.is_in_tof_cone(20.0, 80.0) is False


def test_fuse_detection_with_tof_and_fallback():
    """Verify metric fusion selects ToF distance inside cone and perspective prior outside cone."""
    fuser = SpatialDepthFuser(focal_length_px=500.0, principal_point_x=320.0, principal_point_y=240.0)

    # Inside cone with valid ToF reading
    fused_tof = fuser.fuse_detection(
        bbox=(280.0, 180.0, 360.0, 300.0),
        class_name="chair",
        confidence=0.85,
        tof_distance_m=1.45
    )
    assert fused_tof["fusion_source"] == "tof_laser_fusion"
    assert fused_tof["distance_m"] == 1.45
    assert fused_tof["in_tof_cone"] is True
    assert len(fused_tof["coords_3d_m"]) == 3

    # Outside cone -> should use geometric perspective fallback
    fused_fallback = fuser.fuse_detection(
        bbox=(10.0, 180.0, 70.0, 300.0),
        class_name="chair",
        confidence=0.80,
        tof_distance_m=1.45
    )
    assert fused_fallback["fusion_source"] == "geometric_perspective_prior"
    assert fused_fallback["in_tof_cone"] is False
    assert fused_fallback["distance_m"] > 0


def test_fuse_multiple_detections_disambiguation():
    """Verify that multiple obstacles in the cone receive differentiated distances."""
    fuser = SpatialDepthFuser(focal_length_px=500.0, principal_point_x=320.0, principal_point_y=240.0)

    # Obstacle 1: Centered chair [300, 200, 340, 280]
    # Obstacle 2: Secondary person also near center but larger [290, 100, 350, 380]
    raw_detections = [
        {"bbox": [300.0, 200.0, 340.0, 280.0], "class_name": "chair", "confidence": 0.8},
        {"bbox": [290.0, 100.0, 350.0, 380.0], "class_name": "person", "confidence": 0.9},
    ]

    fused = fuser.fuse_all_detections(raw_detections, tof_distance_m=1.10)
    assert len(fused) == 2
    # One should get ToF laser fusion, while the other maintains perspective prior
    sources = [f["fusion_source"] for f in fused]
    assert "tof_laser_fusion" in sources
    assert "geometric_perspective_prior" in sources


def test_tof_spot_pixel_parallax_and_point_in_bbox():
    """Verify baseline parallax projection and point-in-box hit testing."""
    fuser = SpatialDepthFuser(
        focal_length_px=500.0,
        principal_point_x=320.0,
        principal_point_y=240.0,
        tof_baseline_x_m=0.05,
        tof_baseline_y_m=-0.02,
        tof_hit_margin_px=4.0,
    )

    # At 1 m: u = 320 + 500*0.05 = 345, v = 240 + 500*(-0.02) = 230
    spot_u, spot_v = fuser.tof_spot_pixel(1.0)
    assert np.isclose(spot_u, 345.0)
    assert np.isclose(spot_v, 230.0)

    # At 2 m the parallax shrinks towards the principal point.
    far_u, far_v = fuser.tof_spot_pixel(2.0)
    assert np.isclose(far_u, 332.5)
    assert np.isclose(far_v, 235.0)

    assert fuser.bbox_contains_pixel((330.0, 210.0, 360.0, 250.0), spot_u, spot_v) is True
    assert fuser.bbox_contains_pixel((100.0, 100.0, 200.0, 200.0), spot_u, spot_v) is False
    # Margin expands the box by 4 px on every side.
    assert fuser.bbox_contains_pixel((341.0, 226.0, 344.0, 229.0), spot_u, spot_v) is True


def test_point_in_bbox_prefers_box_under_laser_spot():
    """Point-in-box must beat the plain cone fallback when the spot is offset."""
    fuser = SpatialDepthFuser(
        focal_length_px=500.0,
        principal_point_x=320.0,
        principal_point_y=240.0,
        tof_baseline_x_m=0.10,
        tof_hit_margin_px=2.0,
    )

    # The laser spot at 1 m is u = 320 + 500*0.10 = 370.
    # A is centred on the optical axis but does not cover the spot.
    # B is the small object actually sitting under the laser.
    raw_detections = [
        {"bbox": [300.0, 100.0, 340.0, 380.0], "class_name": "person", "confidence": 0.9},
        {"bbox": [360.0, 200.0, 380.0, 280.0], "class_name": "chair", "confidence": 0.8},
    ]

    fused = fuser.fuse_all_detections(raw_detections, tof_distance_m=1.0)

    assert fused[0]["fusion_source"] == "geometric_perspective_prior"
    assert fused[1]["fusion_source"] == "tof_laser_fusion"
    assert fused[1]["distance_m"] == 1.0


def test_no_tof_reading_keeps_geometric_priors():
    """A stale or missing ToF sample must not be fabricated."""
    fuser = SpatialDepthFuser(focal_length_px=500.0, principal_point_x=320.0, principal_point_y=240.0)
    raw_detections = [
        {"bbox": [300.0, 200.0, 340.0, 280.0], "class_name": "chair", "confidence": 0.8},
    ]

    fused = fuser.fuse_all_detections(raw_detections, tof_distance_m=None)

    assert len(fused) == 1
    assert fused[0]["fusion_source"] == "geometric_perspective_prior"
    assert fused[0]["distance_m"] > 0


