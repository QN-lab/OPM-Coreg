import numpy as np
import pytest

from .compute_helpers import (
    extract_convex_hull_roi,
    extract_face_roi,
    extract_face_roi_from_head_scan,
    extract_irregular_roi_from_head_scan,
    run_icp_head_to_master_scan,
    run_icp_head_to_mri,
)

# ==========================================================
# 1. extract_face_roi (pure function)
# ==========================================================


@pytest.mark.parametrize(
    "polydata, fid_dict",
    [
        (None, {}),  # no polydata
        (object(), {}),  # missing fiducials
    ],
)
def test_extract_face_roi_early_fail(polydata, fid_dict):
    result = extract_face_roi(polydata, fid_dict)
    assert result is None


def test_extract_face_roi_zero_cells(
    fake_polydata_large,
    valid_fiducials,
    mock_pyvista_zero_cells,
):
    result = extract_face_roi(fake_polydata_large, valid_fiducials)
    assert result is None


def test_extract_face_roi_below_min_cells(
    fake_polydata_large,
    valid_fiducials,
    mock_pyvista_small_cells,
):
    result = extract_face_roi(fake_polydata_large, valid_fiducials, min_cells=50)
    assert result is None


def test_extract_face_roi_success(
    fake_polydata_large,
    valid_fiducials,
    mock_pyvista,
):
    result = extract_face_roi(fake_polydata_large, valid_fiducials, min_cells=10)
    assert result is not None


# ==========================================================
# 2. extract_convex_hull_roi
# ==========================================================


@pytest.mark.parametrize(
    "polydata, marked_points",
    [
        (None, None),
        (None, [np.zeros(3)] * 4),
        (object(), None),
    ],
)
def test_extract_convex_hull_roi_invalid_inputs(polydata, marked_points):
    result = extract_convex_hull_roi(polydata, marked_points)
    assert result is None


def test_extract_convex_hull_roi_insufficient_points(fake_polydata_large):
    points = [np.zeros(3)] * 3  # < 4
    result = extract_convex_hull_roi(fake_polydata_large, points)
    assert result is None


def test_extract_convex_hull_roi_delaunay_failure(
    fake_polydata_large,
    mock_delaunay_failure,
):
    points = [np.zeros(3)] * 4
    result = extract_convex_hull_roi(fake_polydata_large, points)
    assert result is None


def test_extract_convex_hull_roi_too_few_selected_points(
    fake_polydata_small,
    mock_delaunay_success,
):
    points = [np.zeros(3)] * 4

    result = extract_convex_hull_roi(
        fake_polydata_small,
        points,
        min_points=100,  # force failure
    )
    assert result is None


# ==========================================================
# 3. extract_face_roi_from_head_scan (GUI)
# ==========================================================


@pytest.mark.parametrize(
    "head_polydata, fiducials",
    [
        (None, {}),  # missing polydata
        (object(), {}),  # missing fiducials
    ],
)
def test_extract_face_roi_from_head_scan_invalid(gui, head_polydata, fiducials):
    gui.head_polydata = head_polydata
    gui.marked_fiducials_on_head_scan = fiducials

    result = extract_face_roi_from_head_scan(gui)

    assert result is False
    assert not hasattr(gui, "face_roi_in_head_frame")


def test_extract_face_roi_from_head_scan_roi_failure(
    gui_with_head_polydata,
    mock_extract_face_roi_failure,
):
    result = extract_face_roi_from_head_scan(gui_with_head_polydata)

    assert result is False
    assert not hasattr(gui_with_head_polydata, "face_roi_in_head_frame")


def test_extract_face_roi_from_head_scan_success(
    gui_with_head_polydata,
    mock_extract_face_roi_success,
):
    result = extract_face_roi_from_head_scan(gui_with_head_polydata)

    assert result is True
    assert hasattr(gui_with_head_polydata, "face_roi_in_head_frame")


# ==========================================================
# 4. extract_irregular_roi_from_head_scan (GUI)
# ==========================================================


@pytest.mark.parametrize(
    "head_polydata, points",
    [
        (None, None),
        (None, [np.zeros(3)] * 4),
        (object(), None),
    ],
)
def test_extract_irregular_roi_from_head_scan_invalid(gui, head_polydata, points):
    gui.head_polydata = head_polydata
    gui.head_alignment_points_from_source = points

    result = extract_irregular_roi_from_head_scan(gui)

    assert result is False
    assert not hasattr(gui, "irregular_roi_in_head_frame")


def test_extract_irregular_roi_from_head_scan_failure(
    gui_with_irregular_points,
    mock_extract_convex_failure,
):
    result = extract_irregular_roi_from_head_scan(gui_with_irregular_points)

    assert result is False
    assert not hasattr(gui_with_irregular_points, "irregular_roi_in_head_frame")


def test_extract_irregular_roi_from_head_scan_success(
    gui_with_irregular_points,
    mock_extract_convex_success,
):
    result = extract_irregular_roi_from_head_scan(gui_with_irregular_points)

    assert result is True
    assert hasattr(gui_with_irregular_points, "irregular_roi_in_head_frame")


# ==========================================================
# 5. run_icp_head_to_mri (False paths)
# ==========================================================


@pytest.mark.parametrize(
    "fixture_name",
    [
        "gui_empty",  # no transform
        "gui_with_mri_only",  # missing transform
    ],
)
def test_icp_head_to_mri_missing_transform(request, fixture_name):
    gui = request.getfixturevalue(fixture_name)

    result = run_icp_head_to_mri(gui)

    assert result is False


def test_icp_head_to_mri_missing_mri(gui_with_transform_only):
    result = run_icp_head_to_mri(gui_with_transform_only)

    assert result is False


@pytest.mark.parametrize("face_roi", [True, False])
def test_icp_head_to_mri_missing_roi(gui_with_no_roi, face_roi):
    result = run_icp_head_to_mri(gui_with_no_roi, face_roi=face_roi)

    assert result is False


# ==========================================================
# 6. run_icp_head_to_master_scan (False paths)
# ==========================================================


@pytest.mark.parametrize(
    "fixture_name",
    [
        "gui_master_scan_missing",  # no transform
        "gui_master_with_transform_only",  # no master scan
        "gui_master_no_roi",  # no ROI
    ],
)
def test_icp_head_to_master_scan_missing_inputs(request, fixture_name):
    gui = request.getfixturevalue(fixture_name)

    result = run_icp_head_to_master_scan(gui)

    # Function sometimes returns None instead of False → accept both
    assert result is False


# ==========================================================
# 7. Validation failure after ICP
# ==========================================================


def test_icp_head_to_mri_validation_failure(
    gui,
    mock_validate_failure,
    mock_vtk_safe,
    mock_surface_rms,
    mock_vtk_matrix,
):
    # Minimal valid setup to reach validation stage
    gui.T_head_to_mri = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    gui.mri_polydata = object()
    gui.face_roi_in_head_frame = object()

    result = run_icp_head_to_mri(gui)

    assert result is False


def test_icp_head_to_master_scan_validation_failure(
    gui,
    mock_validate_failure,
    mock_vtk_safe,
    mock_surface_rms,
    mock_vtk_matrix,
):
    gui.T_head_to_scan = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    gui.master_scan_polydata = object()
    gui.irregular_roi_in_head_frame = object()

    result = run_icp_head_to_master_scan(gui)

    assert result is False
