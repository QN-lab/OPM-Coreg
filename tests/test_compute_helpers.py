import numpy as np
import pytest

from src.compute_helpers import obtain_true_landmarks_from_marked_points


# ----------------------------
# Basic correctness
# ----------------------------


def test_exact_match():
    """Marked points exactly match landmarks."""
    landmarks = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    marked = landmarks.copy()

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    np.testing.assert_array_equal(result, landmarks)


def test_nearest_neighbor_mapping():
    """Each marked point maps to nearest landmark."""
    landmarks = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
        ]
    )

    marked = np.array(
        [
            [0.1, 0, 0],  # closer to first
            [9.8, 0, 0],  # closer to second
        ]
    )

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    expected = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_output_shape_matches_input():
    """Output must match shape of marked points."""
    landmarks = np.random.rand(5, 3)
    marked = np.random.rand(10, 3)

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    assert result.shape == marked.shape


# ----------------------------
# Determinism / tie-breaking
# ----------------------------


def test_tie_breaking_returns_first():
    """If equidistant landmarks exist, first is chosen."""
    landmarks = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
        ]
    )

    marked = np.array(
        [
            [1, 0, 0],  # equidistant
        ]
    )

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    # np.argmin returns first occurrence
    np.testing.assert_array_equal(result[0], landmarks[0])


# ----------------------------
# Multiple marked points mapping to same landmark
# ----------------------------


def test_many_to_one_mapping():
    """Multiple marked points can map to same landmark."""
    landmarks = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
        ]
    )

    marked = np.array(
        [
            [0.1, 0, 0],
            [0.2, 0, 0],
            [0.3, 0, 0],
        ]
    )

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    np.testing.assert_array_equal(result, expected)


# ----------------------------
# Input validation
# ----------------------------


@pytest.mark.parametrize(
    "bad_landmarks",
    [
        np.array([1, 2, 3]),  # 1D
        np.array([[1, 2]]),  # wrong second dim
        np.zeros((3, 4)),  # 4D points
    ],
)
def test_invalid_landmark_shape(bad_landmarks):
    marked = np.zeros((3, 3))

    with pytest.raises(ValueError, match="landmark_positions"):
        obtain_true_landmarks_from_marked_points(bad_landmarks, marked)


@pytest.mark.parametrize(
    "bad_marked",
    [
        np.array([1, 2, 3]),  # 1D
        np.array([[1, 2]]),  # wrong second dim
        np.zeros((3, 4)),  # 4D points
    ],
)
def test_invalid_marked_shape(bad_marked):
    landmarks = np.zeros((3, 3))

    with pytest.raises(ValueError, match="points_marked_from_source"):
        obtain_true_landmarks_from_marked_points(landmarks, bad_marked)


def test_empty_landmarks():
    """No landmarks should raise error."""
    landmarks = np.zeros((0, 3))
    marked = np.zeros((3, 3))

    with pytest.raises(ValueError, match="at least one landmark"):
        obtain_true_landmarks_from_marked_points(landmarks, marked)


# ----------------------------
# Type handling
# ----------------------------


def test_list_inputs_are_supported():
    """Python lists should be accepted and converted."""
    landmarks = [[0, 0, 0], [1, 0, 0]]
    marked = [[0.1, 0, 0]]

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)


def test_integer_inputs_cast_to_float():
    """Integer inputs should be safely cast to float."""
    landmarks = np.array([[0, 0, 0], [1, 0, 0]], dtype=int)
    marked = np.array([[0, 0, 0]], dtype=int)

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    assert result.dtype == float


def test_non_numeric_input_raises():
    """Non-numeric input should fail during np.asarray."""
    landmarks = [["a", "b", "c"]]
    marked = [[0, 0, 0]]

    with pytest.raises(ValueError):
        obtain_true_landmarks_from_marked_points(landmarks, marked)


# ----------------------------
# Numerical behavior
# ----------------------------


def test_large_coordinates():
    """Works with large coordinate values."""
    landmarks = np.array(
        [
            [1e6, 0, 0],
            [0, 1e6, 0],
        ]
    )
    marked = np.array(
        [
            [1e6 + 1, 0, 0],
        ]
    )

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    np.testing.assert_array_equal(result[0], landmarks[0])


def test_small_coordinates():
    """Works with very small coordinates."""
    landmarks = np.array(
        [
            [1e-6, 0, 0],
            [0, 1e-6, 0],
        ]
    )
    marked = np.array(
        [
            [1.1e-6, 0, 0],
        ]
    )

    result = obtain_true_landmarks_from_marked_points(landmarks, marked)

    np.testing.assert_array_equal(result[0], landmarks[0])
