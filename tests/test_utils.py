import numpy as np
import pytest

from src.utils import compute_rigid_transform, validate_transform, vtk_matrix_from_numpy

# ----------------------------
# Tests for compute_rigid_transform
# ----------------------------

# ----------------------------
# Helpers
# ----------------------------


def make_rotation_matrix(axis, theta):
    """Generate rotation matrix using Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    I3 = np.eye(3)
    R = I3 + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


# ----------------------------
# Parametrized transformation tests
# ----------------------------


@pytest.mark.parametrize(
    "axis,theta,translation",
    [
        ([0, 0, 1], 0, [0, 0, 0]),  # identity
        ([0, 0, 1], np.pi / 2, [0, 0, 0]),  # pure rotation
        ([1, 0, 0], np.pi / 4, [1, 2, 3]),  # rotation + translation
        ([0, 1, 0], np.pi, [-5, 0, 2]),  # 180 rotation
        ([1, 1, 1], np.pi / 3, [10, -3, 4]),  # arbitrary axis
    ],
)
def test_parametrized_rigid_transform(axis, theta, translation):
    """Check recovery of multiple rotation + translation combinations."""
    R = make_rotation_matrix(axis, theta)
    t = np.array(translation)

    src = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    dst = (R @ src.T).T + t

    T, rms, errs = compute_rigid_transform(src, dst)

    np.testing.assert_allclose(T[:3, :3], R, atol=1e-6)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-6)
    assert rms < 1e-6
    np.testing.assert_allclose(errs, 0, atol=1e-6)


# ----------------------------
# Noise robustness
# ----------------------------


def test_with_noise():
    """Ensure RMS reflects added Gaussian noise."""
    np.random.seed(42)

    src = np.random.rand(50, 3)
    t = np.array([1.0, -2.0, 3.0])
    dst_clean = src + t

    noise = np.random.normal(scale=0.01, size=src.shape)
    dst = dst_clean + noise

    T, rms, errs = compute_rigid_transform(src, dst)

    assert 0 < rms < 0.05
    assert errs.shape == (50,)


# ----------------------------
# Minimal valid case
# ----------------------------


def test_minimum_points_valid():
    """Exactly 3 non-colinear points should work."""
    src = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    dst = src + np.array([1, 2, 3])

    T, rms, errs = compute_rigid_transform(src, dst)

    assert rms < 1e-6


# ----------------------------
# Type errors
# ----------------------------


@pytest.mark.parametrize(
    "bad_input",
    [
        "not an array",
        object(),
    ],
)
def test_type_errors_src(bad_input):
    with pytest.raises(TypeError):
        compute_rigid_transform(bad_input, [[0, 0, 0]])


@pytest.mark.parametrize(
    "bad_input",
    [
        "invalid",
        object(),
    ],
)
def test_type_errors_dst(bad_input):
    with pytest.raises(TypeError):
        compute_rigid_transform([[0, 0, 0]], bad_input)


# ----------------------------
# Shape validation
# ----------------------------


def test_invalid_shape_src():
    src = np.array([[1, 2], [3, 4]])
    dst = np.zeros((3, 3))

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


def test_invalid_shape_dst():
    src = np.zeros((3, 3))
    dst = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


def test_mismatched_shapes():
    src = np.zeros((3, 3))
    dst = np.zeros((4, 3))

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


def test_too_few_points():
    src = np.zeros((2, 3))
    dst = np.zeros((2, 3))

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


# ----------------------------
# Non-finite values
# ----------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_values(bad_value):
    src = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    src[0, 0] = bad_value
    dst = src.copy()

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


# ----------------------------
# Degenerate configurations
# ----------------------------


def test_identical_points_degenerate():
    """All points identical → rank deficient."""
    src = np.array([[1, 1, 1]] * 5)
    dst = np.array([[2, 2, 2]] * 5)

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


def test_colinear_points_degenerate():
    """Points on a line → insufficient rank."""
    t = np.linspace(0, 1, 5)
    src = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    dst = src + np.array([1, 2, 3])

    with pytest.raises(ValueError):
        compute_rigid_transform(src, dst)


# ----------------------------
# Reflection handling
# ----------------------------


def test_reflection_correction():
    """Ensure improper rotation is corrected."""
    src = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    # Reflect across X axis
    R_reflect = np.diag([-1, 1, 1])
    dst = (R_reflect @ src.T).T

    T, rms, errs = compute_rigid_transform(src, dst)

    det = np.linalg.det(T[:3, :3])

    # Should be a proper rotation
    assert pytest.approx(det, abs=1e-6) == 1.0


# ----------------------------
# Numerical stability tests
# ----------------------------


def test_large_coordinates_stability():
    """Test stability with very large coordinates (~1e6)."""
    scale = 1e6

    src = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        * scale
    )

    R = np.eye(3)
    t = np.array([1000, -2000, 500])  # translation smaller than coordinate scale

    dst = (R @ src.T).T + t

    T, rms, errs = compute_rigid_transform(src, dst)

    np.testing.assert_allclose(T[:3, :3], R, atol=1e-6)
    np.testing.assert_allclose(T[:3, 3], t, rtol=1e-6, atol=1e-3)

    assert rms < 1e-6 * scale  # scaled tolerance
    np.testing.assert_allclose(errs, 0, atol=1e-3)


def test_small_coordinates_stability():
    """Test stability with very small coordinates (~1e-6)."""
    scale = 1e-6

    src = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        * scale
    )

    theta = np.pi / 6
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    t = np.array([1e-6, -2e-6, 3e-6])  # same order of magnitude

    dst = (R @ src.T).T + t

    T, rms, errs = compute_rigid_transform(src, dst)

    np.testing.assert_allclose(T[:3, :3], R, atol=1e-6)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-9)

    assert rms < 1e-9
    np.testing.assert_allclose(errs, 0, atol=1e-9)


# ----------------------------
# Tests for vtk_matrix_from_numpy
# ----------------------------


def test_vtk_matrix_conversion():
    """Check values are copied correctly into vtkMatrix4x4."""
    import vtk

    T = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=float,
    )

    M = vtk_matrix_from_numpy(T)

    for i in range(4):
        for j in range(4):
            assert M.GetElement(i, j) == pytest.approx(T[i, j])


# ----------------------------
# Tests for validate_transform
# ----------------------------

# ----------------------------
# Helpers
# ----------------------------


def make_valid_transform():
    """Return a simple valid identity transform."""
    return np.eye(4)


def make_rotation_z(theta):
    """Rotation matrix about Z-axis."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# ----------------------------
# Valid cases
# ----------------------------


def test_identity_transform_valid():
    T = np.eye(4)
    valid, msg = validate_transform(T)
    assert valid is True
    assert msg == "OK"


def test_valid_rotation_and_translation():
    R = make_rotation_z(np.pi / 4)
    t = np.array([1, 2, 3])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    valid, msg = validate_transform(T)

    assert valid is True
    assert msg == "OK"


def test_with_small_numerical_noise_within_tolerance():
    T = make_valid_transform()
    T[0, 0] += 1e-5  # tiny deviation

    valid, msg = validate_transform(T, ortho_tol=1e-3)
    assert valid is True


# ----------------------------
# Shape validation
# ----------------------------


@pytest.mark.parametrize(
    "bad_shape",
    [
        np.eye(3),
        np.zeros((4, 3)),
        np.zeros((5, 4)),
        np.zeros((4, 4, 1)),
    ],
)
def test_invalid_shape(bad_shape):
    valid, msg = validate_transform(bad_shape)
    assert valid is False
    assert "Expected shape" in msg


# ----------------------------
# Non-finite values
# ----------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_transform_validity(bad_value):
    T = make_valid_transform()
    T[0, 0] = bad_value

    valid, msg = validate_transform(T)

    assert valid is False
    assert "Non-finite values" in msg


# ----------------------------
# Bottom row checks
# ----------------------------


def test_invalid_bottom_row():
    T = make_valid_transform()
    T[3, :] = [0, 0, 0, 0]  # wrong last element

    valid, msg = validate_transform(T)

    assert valid is False
    assert "Bottom row" in msg


def test_bottom_row_within_tolerance():
    T = make_valid_transform()
    T[3, 3] = 1 + 1e-7  # small perturbation

    valid, msg = validate_transform(T, finite_tol=1e-6)

    assert valid is True


# ----------------------------
# Orthonormality checks
# ----------------------------


def test_non_orthonormal_rotation():
    T = make_valid_transform()
    T[0, 0] = 2.0  # scaling breaks orthogonality

    valid, msg = validate_transform(T)

    assert valid is False
    assert "orthonormal" in msg


def test_orthonormal_within_tolerance():
    R = make_rotation_z(np.pi / 6)
    R[0, 0] += 1e-4  # slight perturbation

    T = make_valid_transform()
    T[:3, :3] = R

    valid, msg = validate_transform(T, ortho_tol=1e-3)

    assert valid is True


# ----------------------------
# Determinant checks
# ----------------------------


def test_reflection_matrix_rejected():
    T = make_valid_transform()
    T[0, 0] = -1  # reflection → det = -1

    valid, msg = validate_transform(T)

    assert valid is False
    assert "determinant" in msg


def test_determinant_within_tolerance():
    R = make_rotation_z(np.pi / 4)

    # introduce tiny scaling error
    R = R * 1.0005

    T = make_valid_transform()
    T[:3, :3] = R

    valid, msg = validate_transform(T, det_tol=1e-2)

    assert valid is True


# ----------------------------
# Combined failure prioritization
# ----------------------------


def test_failure_order_shape_checked_first():
    T = np.array([np.nan])  # wrong shape and non-finite

    valid, msg = validate_transform(T)

    assert valid is False
    assert "Expected shape" in msg


def test_failure_order_nonfinite_before_other_checks():
    T = make_valid_transform()
    T[0, 0] = np.nan

    valid, msg = validate_transform(T)

    assert valid is False
    assert "Non-finite" in msg
