"""Module for utility function related to NOCS dataset.

This module contains functions to find similarity transform from NOCS maps and
evaluation function for typical metrics on the NOCS datasets.

Aligning code by Srinath Sridhar:
    https://raw.githubusercontent.com/hughw19/NOCS_CVPR2019/master/aligning.py

Evaluation code by ... TODO
"""
import numpy as np


def estimate_similarity_transform(
    source: np.ndarray, target: np.ndarray, verbose: bool = False
) -> tuple:
    """Estimate similarity transform from source to target from point correspondences.

    Source and target are pairwise correponding pointsets, i.e., they include same
    number of points and the first point of source corresponds to the first point of
    target. RANSAC is used for outlier-robust estimation.

    A similarity transform is estimated (i.e., isotropic scale, rotation and
    translation) that transforms source points onto the target points.

    Note that the returned values fulfill the following equations
        transform @ source_points = scale * rotation_matrix @ source_points + position
    when ignoring homogeneous coordinate for left-hand side.

    Args:
        source: Source points that will be transformed, shape (N,3).
        target: Target points to which source will be aligned to, shape (N,3).
        verbose: If true additional information will be printed.

    Returns:
        position (np.ndarray): Translation to translate source to target, shape (3,).
        rotation_matrix (np.ndarray): Rotation to rotate source to target, shape (3,3).
        scale (float):
            Scaling factor along each axis, to scale source to target.
        transform (np.ndarray): Homogeneous transformation matrix, shape (4,4).
    """
    if len(source) < 5 or len(target) < 5:
        print("Pose estimation failed. Not enough point correspondences: ", len(source))
        return None, None, None, None

    # make points homogeneous
    source_hom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))  # 4,N
    target_hom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))  # 4,N

    # Auto-parameter selection based on source-target heuristics
    target_norm = np.mean(np.linalg.norm(target, axis=1))  # mean distance from origin
    source_norm = np.mean(np.linalg.norm(source, axis=1))
    ratio_ts = target_norm / source_norm
    ratio_st = source_norm / target_norm
    pass_t = ratio_st if (ratio_st > ratio_ts) else ratio_ts
    pass_t *= 0.01  # tighter bound
    stop_t = pass_t / 100
    n_iter = 100
    if verbose:
        print("Pass threshold: ", pass_t)
        print("Stop threshold: ", stop_t)
        print("Number of iterations: ", n_iter)

    source_inliers_hom, target_inliers_hom, best_inlier_ratio = _get_ransac_inliers(
        source_hom,
        target_hom,
        max_iterations=n_iter,
        pass_threshold=pass_t,
        stop_threshold=stop_t,
    )

    if best_inlier_ratio < 0.1:
        print("Pose estimation failed. Small inlier ratio: ", best_inlier_ratio)
        return None, None, None, None

    scales, rotation_matrix, position, out_transform = _estimate_similarity_umeyama(
        source_inliers_hom, target_inliers_hom
    )
    scale = scales[0]

    if verbose:
        print("BestInlierRatio:", best_inlier_ratio)
        print("Rotation:\n", rotation_matrix)
        print("Position:\n", position)
        print("Scales:", scales)

    return position, rotation_matrix, scale, out_transform


def _get_ransac_inliers(
    source_hom: np.ndarray,
    target_hom: np.ndarray,
    max_iterations: int = 100,
    pass_threshold: float = 200,
    stop_threshold: float = 1,
) -> tuple:
    """Apply RANSAC and return set of inliers.

    Args:
        source_hom: Homogeneous coordinates of source points, shape (4,N).
        target_hom: Homogeneous coordinates of target points, shape (4,N).
        max_iterations: Maximum number of RANSAC iterations.
        pass_threshold: Threshold at which a point correspondence is considered good.
        stop_threshold: If residual is below this threshold, RANSAC will stop early.

    Returns:
        source_inliers (np.ndarray):
            Homogeneous coordinates of inlier source points, shape (4,M).
        target_inliers (np.ndarray):
            Homogeneous coordinates of inlier target points, shape (4,M).
        inlier_ratio (float): Ratio of inliers and outliers.
    """
    best_residual = 1e10
    best_inlier_ratio = 0
    best_inlier_idx = np.arange(source_hom.shape[1])
    for _ in range(0, max_iterations):
        # Pick 5 random (but corresponding) points from source and target without repl.
        rand_idx = np.random.choice(
            source_hom.shape[1], size=5, replace=False
        )
        # rand_idx = np.random.randint(source_hom.shape[1], size=10)
        _, _, _, out_transform = _estimate_similarity_umeyama(
            source_hom[:, rand_idx], target_hom[:, rand_idx]
        )
        residual, inlier_ratio, inlier_idx = _evaluate_model(
            out_transform, source_hom, target_hom, pass_threshold
        )
        if residual < best_residual:
            best_residual = residual
            best_inlier_ratio = inlier_ratio
            best_inlier_idx = inlier_idx
        if best_residual < stop_threshold:
            break

        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('Inlier ratio: ', InlierRatio)

    return (
        source_hom[:, best_inlier_idx],
        target_hom[:, best_inlier_idx],
        best_inlier_ratio,
    )


def _evaluate_model(
    out_transform: np.ndarray,
    source_hom: np.ndarray,
    target_hom: np.ndarray,
    pass_threshold: float,
) -> tuple:
    """Evaluate transformation from source to target points.

    Args:
        out_transform:
            Transformation which will be applied to source points, shape (4,4).
        source_hom: Homogeneous coordinates of source points, shape (4,N).
        target_hom: Homogeneous coordinates of target points, shape (4,N).
        pass_threshold: Threshold at which a point correspondence is considered good.

    Returns:
        residual (float): The mean error between transformed source and target.
        inlier_ratio (float):
            Ratio between inliers and number of correspondences (i.e., N).
        inlier_idx (np.ndarray): Array containing the indices of inliers, shape (M,).
    """
    diff = target_hom - np.matmul(out_transform, source_hom)
    residual_vec = np.linalg.norm(diff[:3, :], axis=0)  # shape (N,)
    residual = np.linalg.norm(residual_vec)
    inlier_idx = np.where(residual_vec < pass_threshold)
    n_inliers = np.count_nonzero(inlier_idx)
    inlier_ratio = n_inliers / source_hom.shape[1]
    return residual, inlier_ratio, inlier_idx[0]


def _estimate_similarity_umeyama(
    source_hom: np.ndarray, target_hom: np.ndarray
) -> tuple:
    """Calculate similarity transform from 3D point correspondences.

    A similarity transform is calculated (i.e., isotropic scale, rotation and
    translation) that transforms source points such that squared Euclidean distance
    between transformed source points and target points is minimized.

    Original algorithm from Least-squares estimation of transformation parameters
    between two point patterns, Umeyama, 1991.
    http://web.stanford.edu/class/cs273/refs/umeyama.pdf

    The returned scale, rotation, translation, describe the same transformation as the
    homogeneous transformation matrix as (np notation):
        scale * rotation @ point + translation <=> transformation @ point_hom.

    Args:
        source_hom: Homogeneous coordinates of 5 source points, shape (4,M).
        target_hom: Homogeneous coordinates of 5 target points, shape (4,M).

    Returns:
        scales (np.ndarray):
            Scaling factors along each axis, to scale source to target, shape (3,).
            This will always be three times the same value, since similarity transforms
            only include isotropic scaling.
        rotation (np.ndarray): Rotation to rotate source to target, shape (3,).
        translation (np.ndarray): Translation to translate source to target, shape (3,).
        transform (np.ndarray): Homogeneous transformation matrix, shape (4,4).
    """
    source_centroid = np.mean(source_hom[:3, :], axis=1)  # shape (3,)
    target_centroid = np.mean(target_hom[:3, :], axis=1)  # shape (3,)
    n_points = source_hom.shape[1]

    centered_source = source_hom[:3, :] - source_centroid[:, None]  # shape (3, N)
    centered_target = target_hom[:3, :] - target_centroid[:, None]  # shape (3, N)

    cov_matrix = np.matmul(centered_target, np.transpose(centered_source)) / n_points

    if np.isnan(cov_matrix).any():
        print("nPoints:", n_points)
        print(source_hom.shape)
        print(target_hom.shape)
        raise RuntimeError("There are NANs in the input.")

    u, diag_values, vh = np.linalg.svd(cov_matrix, full_matrices=True)
    diag = np.diag(diag_values)
    s = np.eye(3)
    if np.linalg.det(cov_matrix) < 0.0:
        s[-1, -1] = -1

    rotation = u @ s @ vh

    var_p = np.var(source_hom[:3, :], axis=1).sum()
    if var_p == 0:
        print("Pose estimation failed: 0 variance in sampled points.""")
        print(source_hom)
        raise PoseEstimationError()
    scale_fact = 1 / var_p * np.trace(s @ diag)  # scale factor
    scales = np.array([scale_fact, scale_fact, scale_fact])
    scales_matrix = np.diag(scales)

    translation = target_centroid - scales_matrix @ rotation @ source_centroid

    # create homogeneous transformation from scale, rotation and translation
    out_transform = np.identity(4)
    out_transform[:3, :3] = scales_matrix @ rotation
    out_transform[:3, 3] = translation

    return scales, rotation, translation, out_transform


class PoseEstimationError(Exception):
    """Error if pose estimation encountered an error."""

    pass
