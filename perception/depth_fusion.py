"""Perception Subsystem: Camera & Time-of-Flight (ToF) Spatial Depth Fusion.

Combines 2D visual bounding boxes from the object detector with 1D/multi-zone
Time-of-Flight metric distance measurements via ray-cone geometric projection.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SpatialDepthFuser:
    """Fuses 2D visual bounding boxes and ToF range readings into 3D spatial obstacle vectors."""

    def __init__(
        self,
        focal_length_px: float = 500.0,
        principal_point_x: float = 320.0,
        principal_point_y: float = 240.0,
        tof_cone_half_angle_deg: float = 13.5,
        class_priors_m: Optional[Dict[str, float]] = None,
    ) -> None:
        self.fx = focal_length_px
        self.fy = focal_length_px
        self.cx = principal_point_x
        self.cy = principal_point_y
        self.cone_half_angle_rad = np.radians(tof_cone_half_angle_deg)
        self.class_priors = class_priors_m or {
            "person": 1.70,
            "chair": 0.85,
            "door": 2.00,
            "table": 0.75,
            "car": 1.50,
            "stairs": 1.20,
        }

    def compute_ray_angles(self, u: float, v: float) -> Tuple[float, float]:
        """Compute horizontal azimuth and vertical elevation angles for pixel (u, v).
        
        Args:
            u: Horizontal pixel column.
            v: Vertical pixel row.
        
        Returns:
            Tuple[float, float]: (azimuth_rad, elevation_rad)
        """
        azimuth_rad = np.arctan((u - self.cx) / self.fx)
        elevation_rad = np.arctan((v - self.cy) / self.fy)
        return float(azimuth_rad), float(elevation_rad)

    def is_in_tof_cone(self, u_min: float, u_max: float) -> bool:
        """Evaluate if bounding box horizontal extent intersects the ToF sensor's conical aperture.
        
        Args:
            u_min: Left bounding box edge.
            u_max: Right bounding box edge.
        
        Returns:
            bool: True if bounding box intersects [-theta_FoV, +theta_FoV].
        """
        theta_min, _ = self.compute_ray_angles(u_min, self.cy)
        theta_max, _ = self.compute_ray_angles(u_max, self.cy)

        # Interval intersection test: [theta_min, theta_max] vs [-cone_half, +cone_half]
        overlap_left = max(theta_min, -self.cone_half_angle_rad)
        overlap_right = min(theta_max, self.cone_half_angle_rad)
        return bool(overlap_left <= overlap_right)

    def fuse_detection(
        self,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        confidence: float,
        tof_distance_m: Optional[float],
    ) -> Dict[str, Any]:
        """Fuse a single detected obstacle bounding box with ToF distance.
        
        Args:
            bbox: (u_min, v_min, u_max, v_max) in pixel coordinates.
            class_name: Semantic label (e.g., 'chair', 'person').
            confidence: Detector confidence score [0.0, 1.0].
            tof_distance_m: Metric distance reading from ToF sensor, or None.
        
        Returns:
            Dict[str, Any]: Spatial obstacle dictionary containing 3D coordinates,
                            azimuth angle, metric distance, and fusion method.
        """
        u_min, v_min, u_max, v_max = bbox
        u_center = (u_min + u_max) / 2.0
        v_center = (v_min + v_max) / 2.0
        bbox_height_px = max(1.0, v_max - v_min)

        azimuth_rad, elevation_rad = self.compute_ray_angles(u_center, v_center)
        in_cone = self.is_in_tof_cone(u_min, u_max)

        # Determine metric distance
        if in_cone and tof_distance_m is not None and tof_distance_m > 0:
            metric_distance = tof_distance_m
            fusion_source = "tof_laser_fusion"
        else:
            # Perspective prior fallback estimation: Z \approx (fy * H_real) / h_pixels
            prior_height = self.class_priors.get(class_name, 1.0)
            metric_distance = float((self.fy * prior_height) / bbox_height_px)
            # Clamp between 0.3m and 10m
            metric_distance = float(np.clip(metric_distance, 0.3, 10.0))
            fusion_source = "geometric_perspective_prior"

        # Calculate 3D Cartesian coordinates in camera frame:
        # X: Right (+), Y: Down (+), Z: Forward (+)
        x_3d = metric_distance * np.sin(azimuth_rad) * np.cos(elevation_rad)
        y_3d = metric_distance * np.sin(elevation_rad)
        z_3d = metric_distance * np.cos(azimuth_rad) * np.cos(elevation_rad)

        return {
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "bbox": [round(c, 1) for c in bbox],
            "azimuth_rad": round(azimuth_rad, 4),
            "azimuth_deg": round(np.degrees(azimuth_rad), 2),
            "distance_m": round(metric_distance, 3),
            "coords_3d_m": [round(x_3d, 3), round(y_3d, 3), round(z_3d, 3)],
            "in_tof_cone": in_cone,
            "fusion_source": fusion_source,
        }
