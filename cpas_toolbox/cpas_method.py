"""This module defines the interface for categorical pose and shape estimation methods.

This module defines two classes: PredictionDict and CPASMethod. PredictionDict defines
the prediction produced by a CPASMethod. CPASMethod defines the interface used to
evaluate categorical pose and shape estimation methods.
"""
import sys
from abc import ABC
from typing import Optional

if sys.version_info[0] >= 3 and sys.version_info[1] >= 8:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import open3d as o3d
import torch

from cpas_toolbox import camera_utils


class PredictionDict(TypedDict):
    """Pose and shape prediction.

    Attributes:
        position:
            Position of object center in camera frame. OpenCV convention. Shape (3,).
        orientation:
            Orientation of object in camera frame. OpenCV convention.
            Scalar-last quaternion, shape (4,).
        extents:
            Bounding box side lengths, shape (3,).
        reconstructed_pointcloud:
            Metrically-scaled reconstructed pointcloud in object frame.
            None if method does not perform reconstruction.
        reconstructed_mesh:
            Metrically-scaled reconstructed mesh in object frame.
            None if method does not perform reconstruction.
    """

    position: torch.Tensor
    orientation: torch.Tensor
    extents: torch.Tensor
    reconstructed_pointcloud: Optional[torch.Tensor]
    reconstructed_mesh: Optional[o3d.geometry.TriangleMesh]


class CPASMethod(ABC):
    """Interface class for categorical pose and shape estimation methods."""

    def __init__(self, config: dict, camera: camera_utils.Camera) -> None:
        """Initialize categorical pose and shape estimation method.

        Args:
            config: Method configuration dictionary.
            camera: Camera used for the input image.
        """
        pass

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """Run a method to predict pose and shape of an object.

        Args:
            color_image: The color image, shape (H, W, 3), RGB, 0-1, float.
            depth_image: The depth image, shape (H, W), meters, float.
            instance_mask: Mask of object of interest. (H, W), bool.
            category_str: The category of the object.
        """
        pass
