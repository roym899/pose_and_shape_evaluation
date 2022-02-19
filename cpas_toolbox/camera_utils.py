"""This module provides a pinhole camera class."""
from typing import Tuple

import numpy as np
import open3d as o3d


class Camera:
    """Pinhole camera parameters.

    This class allows conversion between different pixel conventions, i.e., pixel
    center at (0.5, 0.5) (as common in computer graphics), and (0, 0) as common in
    computer vision.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        s: float = 0.0,
        pixel_center: float = 0.0,
    ):
        """Initialize camera parameters.

        Note that the principal point is only fully defined in combination with
        pixel_center.

        The pixel_center defines the relation between continuous image plane
        coordinates and discrete pixel coordinates.

        A discrete image coordinate (x, y) will correspond to the continuous
        image coordinate (x + pixel_center, y + pixel_center). Normally pixel_center
        will be either 0 or 0.5. During calibration it depends on the convention
        the point features used to compute the calibration matrix.

        Note that if pixel_center == 0, the corresponding continuous coordinate
        interval for a pixel are [x-0.5, x+0.5). I.e., proper rounding has to be done
        to convert from continuous coordinate to the corresponding discrete coordinate.

        For pixel_center == 0.5, the corresponding continuous coordinate interval for a
        pixel are [x, x+1). I.e., floor is sufficient to convert from continuous
        coordinate to the corresponding discrete coordinate.

        Args:
            width: Number of pixels in horizontal direction.
            height: Number of pixels in vertical direction.
            fx: Horizontal focal length.
            fy: Vertical focal length.
            cx: Principal point x-coordinate.
            cy: Principal point y-coordinate.
            s: Skew.
            pixel_center: The center offset for the provided principal point.
        """
        # focal length
        self.fx = fx
        self.fy = fy

        # principal point
        self.cx = cx
        self.cy = cy

        self.pixel_center = pixel_center

        # skew
        self.s = s

        # image dimensions
        self.width = width
        self.height = height

    def get_o3d_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters():
        """Convert camera to Open3D pinhole camera parameters.

        Open3D camera is at (0,0,0) looking along positive z axis (i.e., positive z
        values are in front of camera). Open3D expects camera with pixel_center = 0
        and does not support skew.

        Returns:
            The pinhole camera parameters.
        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0)
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        params.extrinsic = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        return params

    def get_pinhole_camera_parameters(self, pixel_center: float) -> Tuple:
        """Convert camera to general camera parameters.

        Args:
            pixel_center:
                At which ratio of a square the pixel center should be for the resulting
                parameters. Typically 0 or 0.5. See class documentation for more info.
        Returns:
            - fx, fy: The horizontal and vertical focal length
            - cx, cy:
                The position of the principal point in continuous image plane
                coordinates considering the provided pixel center and the pixel center
                specified during the construction.
            - s: The skew.
        """
        cx_corrected = self.cx - self.pixel_center + pixel_center
        cy_corrected = self.cy - self.pixel_center + pixel_center
        return self.fx, self.fy, cx_corrected, cy_corrected, self.s
