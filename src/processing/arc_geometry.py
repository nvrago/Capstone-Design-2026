"""
arc_geometry.py -- rigid-body transforms between camera and plate frames

the D405 rides on a carriage along a 300mm-radius arc. as the carriage
sweeps from 0 to 180 degrees, the camera traces a half-circle about the
arc center, always pointing inward (toward the arc center).

this module gives the mechanically-known transform between the camera
frame at any arc angle and the plate-centered frame. it does not require
registration or icp -- the transform is fully determined by the arc
angle and the arc geometry constants.

plate frame convention:
  - origin at plate top-center
  - +z up (away from plate surface, toward arc peak)
  - arc sweeps in the y-z plane (plate lies in the x-y plane)
  - +x is the arc's rotation axis
  - angle = 0 places the camera at +y (one side), looking in -y
  - angle = 90 places the camera at the arc peak, looking -z (down)
  - angle = 180 places the camera at -y, looking +y

camera frame convention (standard D405):
  - +z points into the scene (depth axis)
  - +x right, +y down in the image
  - origin at the camera optical center

if the physical arc rotates the opposite way or the "0-degree" home is
defined differently, flip the sign of angle_deg at the call site --
the math is symmetric.
"""

import numpy as np


def camera_position_in_plate(
    angle_deg: float,
    arc_radius_m: float,
    arc_center_z_m: float,
) -> np.ndarray:
    """position of the camera in plate coordinates at a given arc angle."""
    theta = np.radians(angle_deg)
    return np.array([
        0.0,
        arc_radius_m * np.cos(theta),
        arc_center_z_m + arc_radius_m * np.sin(theta),
    ], dtype=np.float64)


def camera_frame_in_plate(
    angle_deg: float,
    arc_radius_m: float,
    arc_center_z_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    return (x_cam, y_cam, z_cam, origin) all expressed in plate coords.
    the camera's z-axis points from the camera toward the arc center
    (which is the center of the plate horizontally, raised by 50mm).
    """
    theta = np.radians(angle_deg)

    # camera position in plate coords
    origin = camera_position_in_plate(angle_deg, arc_radius_m, arc_center_z_m)

    # arc center in plate coords
    arc_center = np.array([0.0, 0.0, arc_center_z_m], dtype=np.float64)

    # camera's z axis: from camera toward arc center (inward along radius)
    z_cam = arc_center - origin
    z_cam = z_cam / np.linalg.norm(z_cam)

    # camera's x axis: aligned with arc's rotation axis (+x in plate frame).
    # this stays constant as the camera sweeps along the arc.
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # camera's y axis: completes right-handed frame. for a standard camera
    # with +y pointing down in the image, y_cam should point "down" relative
    # to the camera's view. z_cam x x_cam gives a vector perpendicular to
    # both, in the correct handedness for opencv-style cameras.
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    return x_cam, y_cam, z_cam, origin


def transform_camera_to_plate(
    angle_deg: float,
    arc_radius_m: float,
    arc_center_z_m: float,
) -> np.ndarray:
    """
    4x4 homogeneous transform mapping points from camera frame to plate frame.

    usage:
        T = transform_camera_to_plate(angle_deg, R, Zc)
        plate_pts = (T @ np.hstack([cam_pts, ones]).T).T[:, :3]

    returns a c-contiguous float64 matrix safe to pass into open3d.
    """
    x_cam, y_cam, z_cam, origin = camera_frame_in_plate(
        angle_deg, arc_radius_m, arc_center_z_m
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_cam
    T[:3, 1] = y_cam
    T[:3, 2] = z_cam
    T[:3, 3] = origin

    return np.ascontiguousarray(T, dtype=np.float64)


def transform_plate_to_camera(
    angle_deg: float,
    arc_radius_m: float,
    arc_center_z_m: float,
) -> np.ndarray:
    """
    4x4 inverse transform: plate frame back to camera frame.
    used to return points to camera coords after plate-frame filtering.
    """
    T = transform_camera_to_plate(angle_deg, arc_radius_m, arc_center_z_m)
    # rigid transform inverse: R^T and -R^T @ t
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return np.ascontiguousarray(inv, dtype=np.float64)


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    apply a 4x4 transform to an (N, 3) array of points.
    returns a c-contiguous float64 (N, 3) array.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return pts
    homog = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])
    out = (T @ homog.T).T[:, :3]
    return np.ascontiguousarray(out, dtype=np.float64)