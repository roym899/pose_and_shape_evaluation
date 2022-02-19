"""Metrics for shape evaluation."""
from typing import Optional

import numpy as np
import scipy.spatial
from scipy.spatial.transform import Rotation


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
    rotational_symmetry_axis: Optional[int] = None,
) -> int:
    """Classify a pose prediction as correct or incorrect.

    Args:
        position_gt: ground truth position, expected shape (3,).
        position_prediction: predicted position, expected shape (3,).
        position_threshold: position threshold in meters, no threshold if None
        orientation_q_qt: ground truth orientation, scalar-last quaternion, shape (4,)
        orientation_q_prediction:
            predicted orientation, scalar-last quaternion, shape (4,)
        extent_gt:
            bounding box extents, shape (3,)
            only used if IoU threshold specified
        extent_prediction:
            bounding box extents, shape (3,)
            only used if IoU threshold specified
        point_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        degree_threshold: orientation threshold in degrees, no threshold if None
        iou_3d_threshold: 3D IoU threshold, no threshold if None
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
        raise NotImplementedError("3D IoU is not impemented yet.")
        # TODO implement 3D IoU
        # starting point for proper implementation: https://github.com/google-research-datasets/Objectron/blob/c06a65165a18396e1e00091981fd1652875c97b5/objectron/dataset/iou.py#L6
        pass
    if fscore_threshold is not None:
        fscore = reconstruction_fscore(points_gt, points_prediction, 0.01)
        if fscore < fscore_threshold:
            return 0
    return 1


def mean_accuracy(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute accuracy metric.

    Accuracy metric is the same as asymmetric chamfer distance from rec to gt.

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
        return np.mean(d) / extent(points_gt)
    else:
        return np.mean(d)


def mean_completeness(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute completeness metric.

    Completeness metric is the same as asymmetric chamfer distance from gt to rec.

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
        return np.mean(d) / extent(points_gt)
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
        return np.sum(d / extent(points_gt) < threshold) / points_gt.shape[0]
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
        return np.sum(d / extent(points_gt) < threshold) / points_rec.shape[0]
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


def extent(points: np.ndarray) -> float:
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
