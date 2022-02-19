"""Functions to handle transformations with quaternions.

Inspired by PyTorch3D, but using scalar-last convention and not enforcing scalar > 0.
https://github.com/facebookresearch/pytorch3d
"""
import torch


def quaternion_multiply(
    quaternions_1: torch.Tensor, quaternions_2: torch.Tensor
) -> torch.Tensor:
    """Multiply two quaternions representing rotations.

    Normal broadcasting rules apply.

    Args:
        quaternions_1:
            normalized quaternions of shape (..., 4), scalar-last convention
        quaternions_2:
            normalized quaternions of shape (..., 4), scalar-last convention
    Returns:
        Composition of passed quaternions.
    """
    ax, ay, az, aw = torch.unbind(quaternions_1, -1)
    bx, by, bz, bw = torch.unbind(quaternions_2, -1)
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ow = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((ox, oy, oz, ow), -1)


def quaternion_apply(quaternions: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate points by quaternions representing rotations.

    Normal broadcasting rules apply.

    Args:
        quaternions:
            normalized quaternions of shape (..., 4), scalar-last convention
        points:
            points of shape (..., 3)

    Returns:
        Points rotated by the rotations representing quaternions.
    """
    points_as_quaternions = points.new_zeros(points.shape[:-1] + (4,))
    points_as_quaternions[..., :-1] = points
    return quaternion_multiply(
        quaternion_multiply(quaternions, points_as_quaternions),
        quaternion_invert(quaternions),
    )[..., :-1]


def quaternion_invert(quaternions: torch.Tensor) -> torch.Tensor:
    """Invert quaternions representing orientations.

    Args:
        quaternions:
            The quaternions to invert, shape (..., 4), scalar-last convention.

    Returns:
        Inverted quaternions, same shape as quaternions.
    """
    return quaternions * quaternions.new_tensor([-1, -1, -1, 1])


def geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute geodesic distances between quaternions.

    Args:
        q1: First set of quaterions, shape (N,4).
        q2: Second set of quaternions, shape (N,4).

    Returns:
        Mean distance between the quaternions, scalar.
    """
    abs_q1q2 = torch.clip(torch.abs(torch.sum(q1 * q2, dim=1)), 0, 1)
    geodesic_distances = 2 * torch.acos(abs_q1q2)
    return geodesic_distances
