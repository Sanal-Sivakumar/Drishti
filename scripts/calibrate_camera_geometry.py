#!/usr/bin/env python3
"""Calibrate camera intrinsics and camera-to-ToF alignment from real samples."""

from __future__ import annotations

import argparse
import csv
import math
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def calibrate_intrinsics(images: list[Path], columns: int, rows: int, square_m: float):
    object_grid = np.zeros((rows * columns, 3), np.float32)
    object_grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    object_points, image_points, size = [], [], None
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, (columns, rows))
        if not found:
            continue
        refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_grid)
        image_points.append(refined)
    if size is None or len(object_points) < 8:
        raise ValueError("At least eight usable chessboard images are required")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    return error, matrix, distortion, size, len(object_points)


def calibrate_alignment(csv_path: Path, fx: float, fy: float, cx: float, cy: float):
    baselines_x, baselines_y, offsets = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference = float(row["reference_m"])
            measured = float(row["tof_m"])
            u, v = float(row["spot_u"]), float(row["spot_v"])
            offsets.append(reference - measured)
            baselines_x.append((u - cx) * reference / fx)
            baselines_y.append((v - cy) * reference / fy)
    if len(offsets) < 5:
        raise ValueError("At least five alignment rows are required")
    return float(np.median(baselines_x)), float(np.median(baselines_y)), float(np.median(offsets)), len(offsets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Glob for real chessboard images")
    parser.add_argument("--alignment-csv", required=True,
                        help="CSV columns: reference_m,tof_m,spot_u,spot_v")
    parser.add_argument("--columns", type=int, default=9, help="Inner chessboard columns")
    parser.add_argument("--rows", type=int, default=6, help="Inner chessboard rows")
    parser.add_argument("--square-m", type=float, required=True)
    parser.add_argument("--output", default=str(ROOT / "config/geometry_calibration.yaml"))
    args = parser.parse_args()

    images = [Path(item) for item in sorted(glob.glob(args.images))]
    error, matrix, distortion, size, image_count = calibrate_intrinsics(
        images, args.columns, args.rows, args.square_m
    )
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    bx, by, offset, alignment_count = calibrate_alignment(
        Path(args.alignment_csv), fx, fy, cx, cy
    )
    fov = math.degrees(2.0 * math.atan(size[0] / (2.0 * fx)))
    payload = {
        "verified": bool(error <= 1.0 and image_count >= 8 and alignment_count >= 5),
        "frame_width": size[0], "frame_height": size[1],
        "camera_focal_length_x_px": float(fx), "camera_focal_length_y_px": float(fy),
        "principal_point_x": float(cx), "principal_point_y": float(cy),
        "camera_fov_h_deg": float(fov), "baseline_x_m": bx, "baseline_y_m": by,
        "tof_offset_m": offset, "reprojection_error_px": float(error),
        "distortion_coefficients": distortion.ravel().astype(float).tolist(),
        "sample_count": image_count, "alignment_sample_count": alignment_count,
    }
    output = Path(args.output)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(payload, sort_keys=False))
    return 0 if payload["verified"] else 2


if __name__ == "__main__":
    sys.exit(main())
