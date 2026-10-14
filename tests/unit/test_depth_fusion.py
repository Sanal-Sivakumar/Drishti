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

