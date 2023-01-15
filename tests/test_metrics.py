"""Tests for cpas_toolbox.metrics module."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cpas_toolbox import metrics


def test_correct_thresh() -> None:
    """Test correct threshold metric."""
    np.random.seed(0)

    position_gt = np.array([0.0, 0.0, 0.0])
    position_prediction = np.array([0.0, 0.2, 0.0])
    # -> error 0.2
    orientation_gt = Rotation.from_euler("xyz", angles=[0.3, 0.3, 0.5])
    orientation_prediction = orientation_gt * Rotation.from_euler(
        "X", angles=10, degrees=True
    )
    # -> error 10deg, 0 deg if x is symmetry axis
    extent_gt = np.array([1.0, 1.0, 1.0])
    extent_prediction = np.array([1.1, 1.0, 1.0])

    kwargs = {
        "position_gt": position_gt,
        "position_prediction": position_prediction,
        "orientation_gt": orientation_gt,
        "orientation_prediction": orientation_prediction,
        "extent_gt": extent_gt,
        "extent_prediction": extent_prediction,
    }
    iou_3d_sampled = metrics.iou_3d_sampling(
        position_gt,
        orientation_gt,
        extent_gt,
        position_prediction,
        orientation_prediction,
        extent_prediction
    )

    assert metrics.correct_thresh(**kwargs) == 1

    result = metrics.correct_thresh(
        **kwargs,
        degree_threshold=9,
    )
    assert result == 0

    result = metrics.correct_thresh(
        **kwargs,
        degree_threshold=0.001,
        rotational_symmetry_axis=0,
    )
    assert result == 1

    result = metrics.correct_thresh(
        **kwargs,
        degree_threshold=0.001,
        rotational_symmetry_axis=1,
    )
    assert result == 0

    result = metrics.correct_thresh(
        **kwargs,
        position_threshold=0.15,
    )
    assert result == 0

    result = metrics.correct_thresh(
        **kwargs,
        position_threshold=0.25,
        degree_threshold=15,
    )
    assert result == 1

    result = metrics.correct_thresh(
        **kwargs,
        iou_3d_threshold=iou_3d_sampled - 0.01,
    )
    assert result == 1

    result = metrics.correct_thresh(
        **kwargs,
        iou_3d_threshold=iou_3d_sampled + 0.01,
    )
    assert result == 0


def test_mean_accuracy() -> None:
    """Test mean accuracy metric."""
    # Ground truth point that has no close neighbor is ignored in accuracy metric.
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [10, 10, 10]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0]])
    assert metrics.mean_accuracy(points_gt, points_rec) == pytest.approx(0.1)

    # Mean accuracy should be 0 if same set of points is passed twice
    np.random.seed(0)
    points = np.random.rand(1000, 3)
    assert metrics.mean_accuracy(points, points) == 0

    # Norm test
    points_gt = np.array([[1.2, 0.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]])
    points_rec = np.array([[0.0, 0.0, 0.0]])
    assert metrics.mean_accuracy(points_gt, points_rec, p_norm=1) == pytest.approx(1.2)
    assert metrics.mean_accuracy(points_gt, points_rec, p_norm=2) == pytest.approx(1)

    # Normalize test
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [10, 10, 10]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0]])
    assert metrics.mean_accuracy(
        points_gt, points_rec, normalize=True
    ) == pytest.approx(0.0057735)


def test_mean_completeness() -> None:
    """Test mean completeness metric."""
    # Reconstructed point that has no close neighbor is ignored in completeness metric.
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0], [10, 10, 10]])
    assert metrics.mean_completeness(points_gt, points_rec) == pytest.approx(0.1)

    # Reconstructed point that has no close neighbor is ignored in completeness metric.
    points_gt = np.array([[0.0, 0.0, 0.0]])
    points_rec = np.array([[0.1, 0.1, 0.1]])
    assert metrics.mean_completeness(points_gt, points_rec) == pytest.approx(
        np.sqrt(3 * 0.1 ** 2)
    )

    # Mean completeness should be 0 if same set of points is passed twice
    np.random.seed(0)
    points = np.random.rand(1000, 3)
    assert metrics.mean_completeness(points, points) == 0

    # Norm test
    points_gt = np.array([[0.0, 0.0, 0.0]])
    points_rec = np.array([[1.2, 0.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]])
    assert metrics.mean_completeness(points_gt, points_rec, p_norm=1) == pytest.approx(
        1.2
    )
    assert metrics.mean_completeness(points_gt, points_rec, p_norm=2) == pytest.approx(
        1
    )

    # Normalize test
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0], [10, 10, 10]])
    assert metrics.mean_completeness(
        points_gt, points_rec, normalize=True
    ) == pytest.approx(0.0577350269)


def test_symmtric_chamfer() -> None:
    """Test Chamfer L1 metric."""
    # Reconstructed point that has no close neighbor is ignored in completeness metric.
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0]])
    assert metrics.symmetric_chamfer(points_gt, points_rec) == pytest.approx(0.1)

    # Mean completeness should be 0 if same set of points is passed twice
    np.random.seed(0)
    points = np.random.rand(1000, 3)
    assert metrics.symmetric_chamfer(points, points) == 0

    # Normalize test
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 1.0]])
    assert metrics.symmetric_chamfer(
        points_gt, points_rec, normalize=True
    ) == pytest.approx(0.0577350269)


def test_completeness_thresh() -> None:
    """Test thresholded completion metric."""
    # First gt point has close enough neighbor, second does not
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 2.0], [10, 10, 10]])
    assert metrics.completeness_thresh(points_gt, points_rec, 0.2) == 0.5

    # Thresholded completeness should be 1 if same set of points is passed twice
    np.random.seed(0)
    points = np.random.rand(1000, 3)
    assert metrics.completeness_thresh(points, points, 0.1) == 1

    # Norm test, depending on norm both or only one point have close enough neighbor
    points_gt = np.array([[1.0, 0.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]])
    points_rec = np.array([[0.0, 0.0, 0.0]])
    assert metrics.completeness_thresh(
        points_gt, points_rec, 1.1, p_norm=1
    ) == pytest.approx(0.5)
    assert metrics.completeness_thresh(
        points_gt, points_rec, 1.1, p_norm=2
    ) == pytest.approx(1)

    # Normalize test
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 2.0], [10, 10, 10]])
    # 0.1 away, extent sqrt(3)
    assert (
        metrics.completeness_thresh(points_gt, points_rec, 0.05, normalize=True) == 0.0
    )  # 0.0866 < 0.1
    assert (
        metrics.completeness_thresh(points_gt, points_rec, 0.06, normalize=True) == 0.5
    )  # 0.1039 > 0.1


def test_accuracy_thresh() -> None:
    """Test thresholded accuracy metric."""
    # First rec point has close enough neighbor, second does not
    points_gt = np.array([[0.1, 0.0, 0.0], [1.0, 1.1, 2.0], [10, 10, 10]])
    points_rec = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert metrics.accuracy_thresh(points_gt, points_rec, 0.2) == 0.5

    # Thresholded accuracy should be 1 if same set of points is passed twice
    np.random.seed(0)
    points = np.random.rand(1000, 3)
    assert metrics.accuracy_thresh(points, points, 0.1) == 1

    # Norm test, depending on norm both or only one point have close enough neighbor
    points_gt = np.array([[0.0, 0.0, 0.0]])
    points_rec = np.array([[1.0, 0.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]])
    assert metrics.accuracy_thresh(
        points_gt, points_rec, 1.1, p_norm=1
    ) == pytest.approx(0.5)
    assert metrics.accuracy_thresh(
        points_gt, points_rec, 1.1, p_norm=2
    ) == pytest.approx(1)

    # Normalize test
    points_gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.1, 2.0], [10, 10, 10]])
    points_rec = np.array([[0.1, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert metrics.accuracy_thresh(points_gt, points_rec, 0.005, normalize=True) == 0.0
    assert metrics.accuracy_thresh(points_gt, points_rec, 0.006, normalize=True) == 0.5


def test_extent() -> None:
    """Test computation of point cloud extent."""
    points = np.array(
        [[-1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]
    )
    assert metrics.diameter(points) == 10.0

    # coplanar points
    points = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]
    )
    assert metrics.diameter(points) == 10.0

    # less than 4 points
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -5.0], [0.0, 0.0, 5.0]])
    assert metrics.diameter(points) == 10.0
