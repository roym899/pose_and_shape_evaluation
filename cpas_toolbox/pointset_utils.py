"""Utility functions to handle pointsets."""
import torch
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from cpas_toolbox import camera_utils, quaternion_utils, utils


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Normalize pointset to have zero mean.

    Normalization will be performed along second last dimension.

    Args:
        points:
            The pointsets which will be normalized,
            shape (N, M, D) or shape (M, D), N pointsets with M points of dimension D.

    Return:
        normalized_points:
            The normalized pointset, same shape as points.
        centroids:
            The means of the pointclouds used to normalize points.
            Shape (N, D) or (D,), for (N, M, D) and (M, D) inputs, respectively.
    """
    centroids = torch.mean(points, dim=-2, keepdim=True)
    normalized_points = points - centroids
    return normalized_points, centroids.squeeze()


def depth_to_pointcloud(
    depth_image: torch.Tensor,
    camera: camera_utils.Camera,
    normalize: bool = False,
    mask: Optional[torch.Tensor] = None,
    convention: str = "opengl",
) -> torch.Tensor:
    """Convert depth image to pointcloud.

    Args:
        depth_image: The depth image to convert to pointcloud, shape (H,W).
        camera: The camera used to lift the points.
        normalize: Whether to normalize the pointcloud with 0 centroid.
        mask:
            Only points with mask != 0 will be added to pointcloud.
            No masking will be performed if None.
        convention:
            The camera frame convention to use. One of:
                "opengl": x right, y up, z back
                "opencv": x right, y down, z forward
    Returns:
        The pointcloud in the camera frame, in OpenGL convention, shape (N,3).
    """
    fx, fy, cx, cy, _ = camera.get_pinhole_camera_parameters(0.0)

    if mask is None:
        indices = torch.nonzero(depth_image, as_tuple=True)
    else:
        indices = torch.nonzero(depth_image * mask, as_tuple=True)
    depth_values = depth_image[indices]
    points = torch.cat(
        (
            indices[1][:, None].float(),
            indices[0][:, None].float(),
            depth_values[:, None],
        ),
        dim=1,
    )

    if convention == "opengl":
        final_points = torch.empty_like(points)
        final_points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx
        final_points[:, 1] = -(points[:, 1] - cy) * points[:, 2] / fy
        final_points[:, 2] = -points[:, 2]
    elif convention == "opencv":
        final_points = torch.empty_like(points)
        final_points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx
        final_points[:, 1] = (points[:, 1] - cy) * points[:, 2] / fy
        final_points[:, 2] = points[:, 2]
    else:
        raise ValueError(f"Unsupported camera convention {convention}.")

    if normalize:
        final_points, _ = normalize_points(final_points)

    return final_points


def change_transform_camera_convention(
    in_transform: torch.Tensor, in_convention: str, out_convention: str
) -> torch.Tensor:
    """Change the camera convention for a frame A -> camera frame transform.

    Args:
        in_transform:
            Transformtion matrix(es) from coordinate frame A to in_convention camera
            frame.  Shape (...,4,4).
        in_convention:
            Camera convention for the in_transform. One of "opengl", "opencv".
        out_convention:
            Camera convention for the returned transform. One of "opengl", "opencv".

    Returns:
        Transformtion matrix(es) from coordinate frame A to out_convention camera frame.
        Same shape as in_transform.
    """
    # check whether valild convention was provided
    if in_convention not in ["opengl", "opencv"]:
        raise ValueError(f"In camera convention {in_convention} not supported.")
    if out_convention not in ["opengl", "opencv"]:
        raise ValueError(f"Out camera convention {in_convention} not supported.")

    if in_convention == out_convention:
        return in_transform
    else:
        gl2cv_transform = torch.diag(
            in_transform.new_tensor([1.0, -1.0, -1.0, 1.0])
        )  # == cv2gl_transform
        return gl2cv_transform @ in_transform


def change_position_camera_convention(
    in_position: torch.Tensor,
    in_convention: str,
    out_convention: str,
) -> tuple:
    """Change the camera convention for a position in a camera frame.

    Args:
        in_position:
            Position(s) of coordinate frame A in in_convention camera frame.
            Shape (...,3).
        in_convention:
            Camera convention for the in_position. One of "opengl", "opencv".
        out_convention:
            Camera convention for the returned transform. One of "opengl", "opencv".

    Returns:
        Position(s) of coordinate frame A in out_convention camera frame. Shape (...,3).
    """
    # check whether valild convention was provided
    if in_convention not in ["opengl", "opencv"]:
        raise ValueError(f"In camera convention {in_convention} not supported.")
    if out_convention not in ["opengl", "opencv"]:
        raise ValueError(f"Out camera convention {in_convention} not supported.")

    if in_convention == out_convention:
        return in_position
    else:
        gl2cv = in_position.new_tensor([1.0, -1.0, -1.0])  # == cv2gl
        return gl2cv * in_position


def change_orientation_camera_convention(
    in_orientation_q: torch.Tensor,
    in_convention: str,
    out_convention: str,
) -> tuple:
    """Change the camera convention for an orientation in a camera frame.

    Orientation is represented as a quaternion, that rotates points from a
    coordinate frame A to a camera frame (if those frames had the same origin).

    Args:
        in_orientation_q:
            Quaternion(s) which transforms from coordinate frame A to in_convention
            camera frame. Scalar-last convention. Shape (...,4).
        in_convention:
            Camera convention for the in_transform. One of "opengl", "opencv".
        out_convention:
            Camera convention for the returned transform. One of "opengl", "opencv".

    Returns:
        Quaternion(s) which transforms from coordinate frame A to in_convention camera
        frame. Scalar-last convention. Same shape as in_orientation_q.
    """
    # check whether valild convention was provided
    if in_convention not in ["opengl", "opencv"]:
        raise ValueError(f"In camera convention {in_convention} not supported.")
    if out_convention not in ["opengl", "opencv"]:
        raise ValueError(f"Out camera convention {in_convention} not supported.")

    if in_convention == out_convention:
        return in_orientation_q
    else:
        # rotate 180deg around x direction
        gl2cv_q = in_orientation_q.new_tensor([1.0, 0, 0, 0])  # == cv2gl
        return quaternion_utils.quaternion_multiply(gl2cv_q, in_orientation_q)


def visualize_pointset(pointset: torch.Tensor, max_points: int = 1000) -> None:
    """Visualize pointset as 3D scatter plot.

    Args:
        pointset:
            The pointset to visualize. Either shape (N,3), xyz, or shape (N,6), xyzrgb.
        max_points:
            Maximum number of points.
            If N>max_points only a random subset will be shown.
    """
    pointset_np = pointset.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1, 1, 1))

    if len(pointset_np) > max_points:
        indices = np.random.choice(len(pointset_np), replace=False, size=max_points)
        pointset_np = pointset_np[indices]

    if pointset_np.shape[1] == 6:
        colors = pointset_np[:, 3:]
    else:
        colors = None

    ax.scatter(pointset_np[:, 0], pointset_np[:, 1], pointset_np[:, 2], c=colors)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    utils.set_axes_equal(ax)
    plt.show()
