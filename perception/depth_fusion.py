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

    def is_in_tof_cone(
        self,
        u_min: float,
        u_max: float,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
    ) -> bool:
        """Evaluate if bounding box intersects the ToF sensor's conical aperture.
        
        Args:
            u_min: Left bounding box edge.
            u_max: Right bounding box edge.
            v_min: Optional top bounding box edge.
            v_max: Optional bottom bounding box edge.
        
        Returns:
            bool: True if bounding box intersects [-theta_FoV, +theta_FoV] horizontally and vertically.
        """
        theta_min, _ = self.compute_ray_angles(u_min, self.cy)
        theta_max, _ = self.compute_ray_angles(u_max, self.cy)

        # Horizontal overlap test
        overlap_h_left = max(theta_min, -self.cone_half_angle_rad)
        overlap_h_right = min(theta_max, self.cone_half_angle_rad)
        h_overlap = overlap_h_left <= overlap_h_right

        if not h_overlap:
            return False

        if v_min is not None and v_max is not None:
            _, phi_min = self.compute_ray_angles(self.cx, v_min)
            _, phi_max = self.compute_ray_angles(self.cx, v_max)
            overlap_v_top = max(min(phi_min, phi_max), -self.cone_half_angle_rad)
            overlap_v_bottom = min(max(phi_min, phi_max), self.cone_half_angle_rad)
            return bool(overlap_v_top <= overlap_v_bottom)

        return True

    def fuse_detection(
        self,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        confidence: float,
        tof_distance_m: Optional[float],
        force_tof: bool = False,
    ) -> Dict[str, Any]:
        """Fuse a single detected obstacle bounding box with ToF distance.
        
        Args:
            bbox: (u_min, v_min, u_max, v_max) in pixel coordinates.
            class_name: Semantic label (e.g., 'chair', 'person').
            confidence: Detector confidence score [0.0, 1.0].
            tof_distance_m: Metric distance reading from ToF sensor, or None.
            force_tof: If True, assign ToF distance when in cone.
        
        Returns:
            Dict[str, Any]: Spatial obstacle dictionary containing 3D coordinates,
                            azimuth angle, metric distance, and fusion method.
        """
        u_min, v_min, u_max, v_max = bbox
        u_center = (u_min + u_max) / 2.0
        v_center = (v_min + v_max) / 2.0
        bbox_height_px = max(1.0, v_max - v_min)

        azimuth_rad, elevation_rad = self.compute_ray_angles(u_center, v_center)
        in_cone = self.is_in_tof_cone(u_min, u_max, v_min, v_max)

        # Perspective prior fallback estimation: Z \approx (fy * H_real) / h_pixels
        prior_height = self.class_priors.get(class_name, 1.0)
        geom_distance = float((self.fy * prior_height) / bbox_height_px)
        geom_distance = float(np.clip(geom_distance, 0.3, 10.0))

        # Determine metric distance
        if (in_cone or force_tof) and tof_distance_m is not None and tof_distance_m > 0:
            metric_distance = tof_distance_m
            fusion_source = "tof_laser_fusion"
        else:
            metric_distance = geom_distance
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

    def fuse_all_detections(
        self,
        raw_detections: List[Dict[str, Any]],
        tof_distance_m: Optional[float] = None,
        zone_distances: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Fuse multiple simultaneous detections with multi-zone or single ToF range.
        
        Disambiguates multiple objects in the cone so that different objects at
        different depths are not erroneously assigned the same scalar distance.
        """
        if not raw_detections:
            return []

        # Find which detections fall inside the ToF cone
        cone_candidates = []
        for i, det in enumerate(raw_detections):
            bbox = det["bbox"]
            u_min, v_min, u_max, v_max = bbox
            if self.is_in_tof_cone(u_min, u_max, v_min, v_max):
                u_c = (u_min + u_max) / 2.0
                v_c = (v_min + v_max) / 2.0
                az, el = self.compute_ray_angles(u_c, v_c)
                angular_offset = az * az + el * el
                cone_candidates.append((i, angular_offset))

        # Sort candidate in cone closest to optical center
        primary_cone_idx = None
        if cone_candidates:
            cone_candidates.sort(key=lambda c: c[1])
            primary_cone_idx = cone_candidates[0][0]

        fused = []
        for i, det in enumerate(raw_detections):
            bbox = det["bbox"]
            c_name = det.get("class_name", "obstacle")
            conf = det.get("confidence", 0.5)

            # Only assign single scalar ToF distance to the primary centered obstacle in cone
            is_primary_cone = (i == primary_cone_idx)
            use_tof = tof_distance_m if is_primary_cone else None

            # If multi-zone is available (e.g. 4x4 array of 16 distances)
            if zone_distances and len(zone_distances) >= 16:
                u_c = (bbox[0] + bbox[2]) / 2.0
                v_c = (bbox[1] + bbox[3]) / 2.0
                col = int(np.clip(u_c / (self.cx * 2.0) * 4, 0, 3))
                row = int(np.clip(v_c / (self.cy * 2.0) * 4, 0, 3))
                zone_idx = row * 4 + col
                if zone_idx < len(zone_distances) and zone_distances[zone_idx] > 0:
                    use_tof = zone_distances[zone_idx]
                    is_primary_cone = True

            out = self.fuse_detection(
                bbox=tuple(bbox),
                class_name=c_name,
                confidence=conf,
                tof_distance_m=use_tof,
                force_tof=is_primary_cone and use_tof is not None,
            )
            fused.append(out)

        return fused
