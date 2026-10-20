import numpy as np
import pytest

from opm_coreg.compute import (
    compute_aux_to_device,
    compute_aux_to_scan,
    compute_dev_to_scan,
    compute_device_to_head,
    compute_head_to_mri_initial,
    compute_head_to_scan,
    obtain_true_landmarks_from_marked_points,
)

# ----------------------------
# Tests for compute_xx
# ----------------------------

# ==========================================================
# 1. Parametrized tests for point-based functions
# ==========================================================


@pytest.mark.parametrize(
    "func, src_attr, dst_attr, transform_attr, fixture_name",
    [
        (
            compute_aux_to_scan,
            "aux_alignment_points_from_source",
            "aux_alignment_points_from_target",
            "T_aux_to_scan",
            "gui_with_valid_aux_points",
        ),
        (
            compute_dev_to_scan,
            "sensor_alignment_points_from_source",
            "sensor_alignment_points_from_target",
            "T_dev_to_scan",
            "gui_with_valid_sensor_points",
        ),
        (
            compute_head_to_scan,
            "head_alignment_points_from_source",
            "head_alignment_points_from_target",
            "T_head_to_scan",
            "gui_with_valid_head_points",
        ),
    ],
)
def test_points_missing_or_insufficient(
    func, src_attr, dst_attr, transform_attr, fixture_name, request, gui
):
    """Test missing and insufficient source/target points."""
    setattr(gui, src_attr, [])
    setattr(gui, dst_attr, [np.zeros(3)] * 3)

    assert func(gui) is False
    assert not hasattr(gui, transform_attr)


@pytest.mark.parametrize(
    "func, transform_attr, fixture_name",
    [
        (compute_aux_to_scan, "T_aux_to_scan", "gui_with_valid_aux_points"),
        (compute_dev_to_scan, "T_dev_to_scan", "gui_with_valid_sensor_points"),
        (compute_head_to_scan, "T_head_to_scan", "gui_with_valid_head_points"),
    ],
)
def test_transform_success(
    func,
    transform_attr,
    fixture_name,
    request,
    mock_compute_transform_success,
    mock_validate_success,
):
    """Test successful transform computation."""
    gui = request.getfixturevalue(fixture_name)

    result = func(gui)

    assert result is True
    assert hasattr(gui, transform_attr)


@pytest.mark.parametrize(
    "func, transform_attr, fixture_name",
    [
        (compute_aux_to_scan, "T_aux_to_scan", "gui_with_valid_aux_points"),
        (compute_dev_to_scan, "T_dev_to_scan", "gui_with_valid_sensor_points"),
        (compute_head_to_scan, "T_head_to_scan", "gui_with_valid_head_points"),
    ],
)
def test_transform_failure(
    func, transform_attr, fixture_name, request, mock_compute_transform_failure
):
    """Test failure of compute_rigid_transform."""
    gui = request.getfixturevalue(fixture_name)

    result = func(gui)

    assert result is False
    assert not hasattr(gui, transform_attr)


@pytest.mark.parametrize(
    "func, transform_attr, fixture_name",
    [
        (compute_aux_to_scan, "T_aux_to_scan", "gui_with_valid_aux_points"),
        (compute_dev_to_scan, "T_dev_to_scan", "gui_with_valid_sensor_points"),
        (compute_head_to_scan, "T_head_to_scan", "gui_with_valid_head_points"),
    ],
)
def test_validation_failure(
    func,
    transform_attr,
    fixture_name,
    request,
    mock_compute_transform_success,
    mock_validate_failure,
):
    """Test invalid transform rejected by validation."""
    gui = request.getfixturevalue(fixture_name)

    result = func(gui)

    assert result is False
    assert not hasattr(gui, transform_attr)


# ==========================================================
# 2. Landmark-specific tests (only aux + dev)
# ==========================================================


@pytest.mark.parametrize(
    "func, fixture_name, transform_attr, landmark_fixture",
    [
        (
            compute_aux_to_scan,
            "gui_with_aux_landmarks",
            "T_aux_to_scan",
            "mock_landmarks_success",
        ),
        (
            compute_dev_to_scan,
            "gui_with_helmet_landmarks",
            "T_dev_to_scan",
            "mock_landmarks_success",
        ),
    ],
)
def test_landmarks_success(
    func,
    fixture_name,
    transform_attr,
    landmark_fixture,
    request,
    mock_compute_transform_success,
    mock_validate_success,
):
    gui = request.getfixturevalue(fixture_name)
    request.getfixturevalue(landmark_fixture)

    result = func(gui)

    assert result is True
    assert hasattr(gui, transform_attr)


@pytest.mark.parametrize(
    "func, fixture_name, transform_attr",
    [
        (compute_aux_to_scan, "gui_with_aux_landmarks", "T_aux_to_scan"),
        (compute_dev_to_scan, "gui_with_helmet_landmarks", "T_dev_to_scan"),
    ],
)
def test_landmarks_failure_fallback(
    func,
    fixture_name,
    transform_attr,
    request,
    mock_landmarks_failure,
    mock_compute_transform_success,
    mock_validate_success,
):
    gui = request.getfixturevalue(fixture_name)

    result = func(gui)

    assert result is True  # fallback path
    assert hasattr(gui, transform_attr)


# ==========================================================
# 3. Transform composition functions
# ==========================================================


@pytest.mark.parametrize(
    "func, out_attr",
    [
        (compute_aux_to_device, "T_aux_to_device"),
        (compute_device_to_head, "T_dev_to_head"),
    ],
)
def test_transform_composition_success(
    func, out_attr, gui_with_transforms, mock_validate_success
):
    result = func(gui_with_transforms)

    assert result is True
    assert hasattr(gui_with_transforms, out_attr)


@pytest.mark.parametrize(
    "func",
    [
        compute_aux_to_device,
        compute_device_to_head,
    ],
)
def test_transform_composition_missing_inputs(func, gui):
    result = func(gui)

    assert result is False


@pytest.mark.parametrize(
    "func, bad_attr",
    [
        (compute_aux_to_device, "T_dev_to_scan"),
        (compute_device_to_head, "T_head_to_scan"),
    ],
)
def test_transform_inversion_failure(func, bad_attr, gui_with_transforms):
    # Create singular matrix
    gui_with_transforms.__dict__[bad_attr] = np.zeros((4, 4))

    result = func(gui_with_transforms)

    assert result is False


@pytest.mark.parametrize(
    "func, out_attr",
    [
        (compute_aux_to_device, "T_aux_to_device"),
        (compute_device_to_head, "T_dev_to_head"),
    ],
)
def test_composition_validation_failure(
    func, out_attr, gui_with_transforms, mock_validate_failure
):
    result = func(gui_with_transforms)

    assert result is False
    assert not hasattr(gui_with_transforms, out_attr)


# ==========================================================
# 4. Head → MRI function
# ==========================================================


def test_head_to_mri_missing_fiducials(gui):
    gui.marked_fiducials_on_head_scan = {}
    gui.mri_fiducials = {}

    result = compute_head_to_mri_initial(gui)

    assert result is False


def test_head_to_mri_partial_fiducials(gui):
    gui.marked_fiducials_on_head_scan = {"NAS": np.zeros(3)}
    gui.mri_fiducials = {
        "NAS": np.zeros(3),
        "LPA": np.zeros(3),
        "RPA": np.zeros(3),
    }

    result = compute_head_to_mri_initial(gui)

    assert result is False


def test_head_to_mri_success(
    gui_with_fiducials, mock_compute_transform_success, mock_validate_success
):
    result = compute_head_to_mri_initial(gui_with_fiducials)

    assert result is True
    assert hasattr(gui_with_fiducials, "T_head_to_mri")
    assert hasattr(gui_with_fiducials, "head_to_mri_rms")
    assert hasattr(gui_with_fiducials, "head_to_mri_errors")


def test_head_to_mri_compute_failure(
    gui_with_fiducials, mock_compute_transform_failure
):
    result = compute_head_to_mri_initial(gui_with_fiducials)

    assert result is False


def test_head_to_mri_validation_failure(
    gui_with_fiducials, mock_compute_transform_success, mock_validate_failure
):
    result = compute_head_to_mri_initial(gui_with_fiducials)

    assert result is False
    assert not hasattr(gui_with_fiducials, "T_head_to_mri")


# ----------------------------
# Tests for obtain_true_landmarks_from_marked_points
# ----------------------------

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
