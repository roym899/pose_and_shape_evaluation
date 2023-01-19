"""Metrics for shape evaluation."""
from typing import Optional, Union

import numpy as np
import scipy.spatial
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation


def diameter(points: np.ndarray) -> float:
    """Compute largest Euclidean distance between any two points.

    Args:
        points_gt: set of true
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
    Returns:
        Ratio of reconstructed points with closest ground truth point closer than
        threshold (in p-norm).
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except scipy.spatial.qhull.QhullError:
        # fallback to brute force distance matrix
        return np.max(scipy.spatial.distance_matrix(points, points))

    # this is wasteful, if too slow implement rotating caliper method
    return np.max(
        scipy.spatial.distance_matrix(points[hull.vertices], points[hull.vertices])
    )


def mean_accuracy(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute accuracy metric.

    Accuracy metric is the same as the mean pointwise distance (or asymmetric chamfer
    distance) from rec to gt.

    See, for example, Occupancy Networks Learning 3D Reconstruction in Function Space,
    Mescheder et al., 2019.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide result by Euclidean extent of points_gt
    Returns:
        Arithmetic mean of p-norm from reconstructed points to closest (in p-norm)
        ground truth points.
    """
    kd_tree = scipy.spatial.KDTree(points_gt)
    d, _ = kd_tree.query(points_rec, p=p_norm)
    if normalize:
        return np.mean(d) / diameter(points_gt)
    else:
        return np.mean(d)


def mean_completeness(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute completeness metric.

    Completeness metric is the same as the mean pointwise distance (or asymmetric
    chamfer distance) from gt to rec.

    See, for example, Occupancy Networks Learning 3D Reconstruction in Function Space,
    Mescheder et al., 2019.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide result by Euclidean extent of points_gt
    Returns:
        Arithmetic mean of p-norm from ground truth points to closest (in p-norm)
        reconstructed points.
    """
    kd_tree = scipy.spatial.KDTree(points_rec)
    d, _ = kd_tree.query(points_gt, p=p_norm)
    if normalize:
        return np.mean(d) / diameter(points_gt)
    else:
        return np.mean(d)


def symmetric_chamfer(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute symmetric chamfer distance.

    There are various slightly different definitions for the chamfer distance.

    Note that completeness and accuracy are themselves sometimes referred to as
    chamfer distances, with symmetric chamfer distance being the combination of the two.

    Chamfer L1 in the literature (see, for example, Occupancy Networks Learning 3D
    Reconstruction in Function Space, Mescheder et al., 2019) refers to using
    arithmetic mean (note that this is actually differently scaled from L1) when
    combining accuracy and completeness.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide result by Euclidean extent of points_gt
    Returns:
        Arithmetic mean of accuracy and completeness metrics using the specified p-norm.
    """
    return (
        mean_completeness(points_gt, points_rec, p_norm=p_norm, normalize=normalize)
        + mean_accuracy(points_gt, points_rec, p_norm=p_norm, normalize=normalize)
    ) / 2


def normalized_average_distance(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
) -> float:
    """Compute the maximum of the directed normalized average distances.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)

    Returns:
        Maximum of normalized mean accuracy and mean completeness metrics using the
        specified p-norm.
    """
    return max(
        mean_completeness(points_gt, points_rec, p_norm=p_norm, normalize=True),
        mean_accuracy(points_gt, points_rec, p_norm=p_norm, normalize=True),
    )


def completeness_thresh(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    threshold: float,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute thresholded completion metric.

    See FroDO: From Detections to 3D Objects, Rünz et al., 2020.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        threshold: distance threshold to count a point as correct
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide distances by Euclidean extent of points_gt
    Returns:
        Ratio of ground truth points with closest reconstructed point closer than
        threshold (in p-norm).
    """
    kd_tree = scipy.spatial.KDTree(points_rec)
    d, _ = kd_tree.query(points_gt, p=p_norm)
    if normalize:
        return np.sum(d / diameter(points_gt) < threshold) / points_gt.shape[0]
    else:
        return np.sum(d < threshold) / points_gt.shape[0]


def accuracy_thresh(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    threshold: float,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute thresholded accuracy metric.

    See FroDO: From Detections to 3D Objects, Rünz et al., 2020.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        threshold: distance threshold to count a point as correct
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide distances by Euclidean extent of points_gt
    Returns:
        Ratio of reconstructed points with closest ground truth point closer than
        threshold (in p-norm).
    """
    kd_tree = scipy.spatial.KDTree(points_gt)
    d, _ = kd_tree.query(points_rec, p=p_norm)
    if normalize:
        return np.sum(d / diameter(points_gt) < threshold) / points_rec.shape[0]
    else:
        return np.sum(d < threshold) / points_rec.shape[0]


def reconstruction_fscore(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    threshold: float,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute reconstruction fscore.

    See What Do Single-View 3D Reconstruction Networks Learn, Tatarchenko, 2019

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        threshold: distance threshold to count a point as correct
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide distances by Euclidean extent of points_gt
    Returns:
        Harmonic mean of precision (thresholded accuracy) and recall (thresholded
        completeness).
    """
    recall = completeness_thresh(
        points_gt, points_rec, threshold, p_norm=p_norm, normalize=normalize
    )
    precision = accuracy_thresh(
        points_gt, points_rec, threshold, p_norm=p_norm, normalize=normalize
    )
    if recall < 1e-7 or precision < 1e-7:
        return 0
    return 2 / (1 / recall + 1 / precision)


def iou_3d_sampling(
    p1: np.ndarray,
    r1: Rotation,
    e1: np.ndarray,
    p2: np.ndarray,
    r2: Rotation,
    e2: np.ndarray,
    num_points: int = 10000,
) -> float:
    """Compute 3D IoU of oriented bounding boxes by sampling the smaller bounding box.

    Args:
        p1: Center position of first bounding box, shape (3,).
        r1: Orientation of first bounding box.
            This is the rotation that rotates points from bounding box to camera frame.
        e1: Extents (i.e., side lengths) of first bounding box, shape (3,).
        p2: Center position of second bounding box, shape (3,).
        r2: Orientation of second bounding box.
            This is the rotation that rotates points from bounding box to camera frame.
        e2: Extents (i.e., side lengths) of second bounding box, shape (3,).
        num_points: Number of points to sample in smaller bounding box.

    Returns:
        Approximate intersection-over-union for the two oriented bounding boxes.
    """
    # sample smaller volume to estimate intersection
    vol_1 = np.prod(e1)
    vol_2 = np.prod(e2)
    if vol_1 < vol_2:
        points_1_in_1 = e1 * np.random.rand(num_points, 3) - e1 / 2
        points_1_in_w = r1.apply(points_1_in_1) + p1
        points_1_in_2 = r2.inv().apply(points_1_in_w - p2)
        ratio_1_in_2 = (
            np.sum(
                np.all(points_1_in_2 < e2 / 2, axis=1)
                * np.all(-e2 / 2 < points_1_in_2, axis=1)
            )
            / num_points
        )
        intersection = ratio_1_in_2 * vol_1
    else:
        points_2_in_2 = e2 * np.random.rand(num_points, 3) - e2 / 2
        points_2_in_w = r2.apply(points_2_in_2) + p2
        points_2_in_1 = r1.inv().apply(points_2_in_w - p1)
        ratio_2_in_1 = (
            np.sum(
                np.all(points_2_in_1 < e1 / 2, axis=1)
                * np.all(-e1 / 2 < points_2_in_1, axis=1)
            )
            / num_points
        )
        intersection = ratio_2_in_1 * vol_2

    union = vol_1 + vol_2 - intersection

    return intersection / union


def iou_3d(
    p1: np.ndarray,
    r1: Rotation,
    e1: np.ndarray,
    p2: np.ndarray,
    r2: Rotation,
    e2: np.ndarray,
) -> float:
    """Compute 3D IoU of oriented bounding boxes analytically.

    Code partly based on https://github.com/google-research-datasets/Objectron/.
    Implementation uses half-space intersection instead of Sutherland-Hodgman algorithm.

    Args:
        p1: Center position of first bounding box, shape (3,).
        r1: Orientation of first bounding box.
            This is the rotation that rotates points from bounding box to camera frame.
        e1: Extents (i.e., side lengths) of first bounding box, shape (3,).
        p2: Center position of second bounding box, shape (3,).
        r2: Orientation of second bounding box.
            This is the rotation that rotates points from bounding box to camera frame.
        e2: Extents (i.e., side lengths) of second bounding box, shape (3,).

    Returns:
        Accurate intersection-over-union for the two oriented bounding boxes.
    """
    # create halfspaces
    halfspaces = np.zeros((12, 4))
    halfspaces[0:3, 0:3] = r1.as_matrix().T
    halfspaces[0:3, 3] = -halfspaces[0:3, 0:3] @ (r1.apply(e1 / 2) + p1)
    halfspaces[3:6, 0:3] = -halfspaces[0:3, 0:3]
    halfspaces[3:6, 3] = -halfspaces[3:6, 0:3] @ (r1.apply(-e1 / 2) + p1)
    halfspaces[6:9, 0:3] = r2.as_matrix().T
    halfspaces[6:9, 3] = -halfspaces[6:9, 0:3] @ (r2.apply(e2 / 2) + p2)
    halfspaces[9:12, 0:3] = -halfspaces[6:9, 0:3]
    halfspaces[9:12, 3] = -halfspaces[9:12, 0:3] @ (r2.apply(-e2 / 2) + p2)

    # try to find point inside both bounding boxes
    inside_point = _find_inside_point(p1, r1, e1, p2, r2, e2, halfspaces)
    if inside_point is None:
        return 0

    # create halfspace intersection and compute IoU
    hs = scipy.spatial.HalfspaceIntersection(halfspaces, inside_point)
    ch = scipy.spatial.ConvexHull(hs.intersections)
    intersection = ch.volume
    vol_1 = np.prod(e1)
    vol_2 = np.prod(e2)
    union = vol_1 + vol_2 - intersection
    return intersection / union


def _find_inside_point(
    p1: np.ndarray,
    r1: Rotation,
    e1: np.ndarray,
    p2: np.ndarray,
    r2: Rotation,
    e2: np.ndarray,
    halfspaces: np.ndarray,
    sample_points: int = 100,
) -> Union[np.ndarray, None]:
    """Find 3D point inside two oriented bounding boxes.

    Args:
        p1: Center position of first bounding box, shape (3,).
        r1: Orientation of first bounding box.
            This is the rotation that rotates points from bounding box to camera frame.
        e1: Extents (i.e., side lengths) of first bounding box, shape (3,).
        p2: Center position of second bounding box, shape (3,).
        r2: Orientation of second bounding box.
            This is the rotation that rotates points from bounding box to camera frame.
        e2: Extents (i.e., side lengths) of second bounding box, shape (3,).
        halfspaces: Halfspaces defining the bounding boxes.
        sample_points:
            Number of samples sampled from smaller bounding box to check initially.
            If none of the points is inside both bounding boxes a linear program will
            be solved.

    Returns:
        Point inside both oriented bounding boxes. None if there is no such point.
        Shape (3,).
    """
    vol_1 = np.prod(e1)
    vol_2 = np.prod(e2)
    if vol_1 < vol_2:
        points_1_in_1 = e1 * np.random.rand(sample_points, 3) - e1 / 2
        points_1_in_w = r1.apply(points_1_in_1) + p1
        points_1_in_2 = r2.inv().apply(points_1_in_w - p2)
        points_in = np.all(points_1_in_2 < e2 / 2, axis=1) * np.all(
            -e2 / 2 < points_1_in_2, axis=1
        )
        index = np.argmax(points_in)
        if points_in[index]:
            return points_1_in_w[index]
    else:
        points_2_in_2 = e2 * np.random.rand(sample_points, 3) - e2 / 2
        points_2_in_w = r2.apply(points_2_in_2) + p2
        points_2_in_1 = r1.inv().apply(points_2_in_w - p1)
        points_in = np.all(points_2_in_1 < e1 / 2, axis=1) * np.all(
            -e1 / 2 < points_2_in_1, axis=1
        )
        index = np.argmax(points_in)
        if points_in[index]:
            return points_2_in_w[index]

    # no points found, solve linear program to find intersection point
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], np.ones((halfspaces.shape[0], 1))))
    b = -halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    if res.fun > 0:  # no intersection
        return None
    return res.x[:3]


def correct_thresh(
    position_gt: np.ndarray,
    position_prediction: np.ndarray,
    orientation_gt: Rotation,
    orientation_prediction: Rotation,
    extent_gt: Optional[np.ndarray] = None,
    extent_prediction: Optional[np.ndarray] = None,
    points_gt: Optional[np.ndarray] = None,
    points_prediction: Optional[np.ndarray] = None,
    position_threshold: Optional[float] = None,
    degree_threshold: Optional[float] = None,
    iou_3d_threshold: Optional[float] = None,
    fscore_threshold: Optional[float] = None,
    nad_threshold: Optional[float] = None,
    rotational_symmetry_axis: Optional[int] = None,
) -> int:
    """Classify a pose prediction as correct or incorrect.

    Args:
        position_gt: Ground truth position, shape (3,).
        position_prediction: Predicted position, shape (3,).
        position_threshold: Position threshold in meters, no threshold if None.
        orientation_gt:
            Ground truth orientation.
            This is the rotation that rotates points from bounding box to camera frame.
        orientation_prediction:
            Predicted orientation.
            This is the rotation that rotates points from bounding box to camera frame.
        extent_gt:
            Bounding box extents, shape (3,).
            Only used if IoU threshold specified.
        extent_prediction:
            Bounding box extents, shape (3,).
            Only used if IoU threshold specified.
        point_gt: Set of true points, shape (N,3).
        points_rec: Set of reconstructed points, shape (M,3).
        degree_threshold: Orientation threshold in degrees, no threshold if None.
        iou_3d_threshold: 3D IoU threshold, no threshold if None.
        nad_threshold: Normalized average distance thresold, no threshold if None.
        rotational_symmetry_axis:
            Specify axis along which rotation is ignored. If None, no axis is ignored.
            0 for x-axis, 1 for y-axis, 2 for z-axis.

    Returns:
        1 if error is below all provided thresholds.  0 if error is above one provided
        threshold.
    """
    if position_threshold is not None:
        position_error = np.linalg.norm(position_gt - position_prediction)
        if position_error > position_threshold:
            return 0
    if degree_threshold is not None:
        rad_threshold = degree_threshold * np.pi / 180.0
        if rotational_symmetry_axis is not None:
            p = np.array([0.0, 0.0, 0.0])
            p[rotational_symmetry_axis] = 1.0
            p1 = orientation_gt.apply(p)
            p2 = orientation_prediction.apply(p)
            rad_error = np.arccos(p1 @ p2)
        else:
            rad_error = (orientation_gt * orientation_prediction.inv()).magnitude()
        if rad_error > rad_threshold:
            return 0
    if iou_3d_threshold is not None:
        if rotational_symmetry_axis is not None:
            max_iou = 0

            for r in np.linspace(0, np.pi, 100):
                p = np.array([0.0, 0.0, 0.0])
                p[rotational_symmetry_axis] = 1.0
                p *= r
                sym_rot = Rotation.from_rotvec(r)
                iou = iou_3d(
                    position_gt,
                    orientation_gt,
                    extent_gt,
                    position_prediction,
                    orientation_prediction * sym_rot,
                    extent_prediction,
                )
                max_iou = max(iou, max_iou)
            iou = max_iou
        else:
            iou = iou_3d(
                position_gt,
                orientation_gt,
                extent_gt,
                position_prediction,
                orientation_prediction,
                extent_prediction,
            )
        if iou < iou_3d_threshold:
            return 0
    if fscore_threshold is not None:
        # TODO make 0.01 a parameter
        fscore = reconstruction_fscore(points_gt, points_prediction, 0.01)
        if fscore < fscore_threshold:
            return 0
    if nad_threshold is not None:
        nad = normalized_average_distance(points_gt, points_prediction)
        if nad < nad_threshold:
            return 0
    return 1
