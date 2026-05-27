"""Helper functions to compute transforms based on GUI state and user inputs."""

import numpy as np
import vtk
from scipy.spatial import Delaunay

from src.display_helpers import (
    overlay_head_transformed,
    overlay_helmet_sensors_transformed,
    preview_head_on_mri,
    preview_roi_on_mri,
)
from src.file_io_helpers import (
    save_generated_transforms,
    save_marked_points_in_cache,
    save_transformed_sensors_csv,
)
from src.utils import (
    compute_rigid_transform,
    vtk_matrix_from_numpy,
)


def compute_aux_dev_to_scan(gui):
    """Compute transform from auxiliary device to master scan using the auxiliary device pairs."""
    if (
        not getattr(gui, "aux_alignment_points_from_source", None)
        or len(gui.aux_alignment_points_from_source) < 3
    ):
        gui.update_info("Need ≥3 auxiliary device pairs.")
        return
    src = np.vstack(gui.aux_alignment_points_from_source)
    dst = np.vstack(gui.aux_alignment_points_from_target)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_aux_dev_to_scan = T
    gui.update_info(
        f"Computed auxiliary device to master scan transform RMS={rms:.3f} mm."
    )

    # Save generated transform
    save_generated_transforms(gui, trans_to_save="aux_dev_to_scan")

    # Save marked points in cache after computing transform
    save_marked_points_in_cache(gui, file_to_save="aux_alignment_points_from_source")
    save_marked_points_in_cache(gui, file_to_save="aux_alignment_points_from_target")


def compute_dev_to_scan(gui):
    """Compute transform from device (sensor helmet) to master scan using the device pairs."""
    if (
        not getattr(gui, "sensor_alignment_points_from_source", None)
        or len(gui.sensor_alignment_points_from_source) < 3
    ):
        gui.update_info("Need ≥3 point pairs from sensor alignment step.")
        return
    src = np.vstack(gui.sensor_alignment_points_from_source)
    dst = np.vstack(gui.sensor_alignment_points_from_target)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_dev_to_scan = T
    gui.update_info(
        f"Computed device (helmet) to master scan transform RMS={rms:.3f} mm."
    )

    # Save generated transform
    save_generated_transforms(gui, trans_to_save="dev_to_scan")

    # Save marked points in cache after computing transform
    save_marked_points_in_cache(gui, file_to_save="sensor_alignment_points_from_source")
    save_marked_points_in_cache(gui, file_to_save="sensor_alignment_points_from_target")


def compute_head_to_scan(gui):
    """Compute transform from head scan to master scan using the head pairs."""
    if len(gui.head_alignment_points_from_source) < 3:
        gui.update_info("Need at least 3 head alignment pairs to compute transform.")
        return
    src = np.vstack(gui.head_alignment_points_from_source)
    dst = np.vstack(gui.head_alignment_points_from_target)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_head_to_scan = T
    gui.update_info(
        f"Computed head_to_scan RMS={rms:.3f} mm. Overlaying transformed head scan onto target window."
    )
    # overlay transformed head scan on target window (light orange)
    overlay_head_transformed(gui, T)

    # Save generated transform
    save_generated_transforms(gui, trans_to_save="head_to_scan")

    # Save marked points in cache after computing transform
    save_marked_points_in_cache(gui, file_to_save="head_alignment_points_from_source")
    save_marked_points_in_cache(gui, file_to_save="head_alignment_points_from_target")
    save_marked_points_in_cache(gui, file_to_save="marked_fiducials_on_head_scan")


def compute_aux_to_device(gui):
    """Compute transform from auxiliary device to device (sensor helmet).

    Using the aux->scan transform and dev->scan transform."""
    if gui.T_aux_dev_to_scan is None or gui.T_dev_to_scan is None:
        gui.update_info("Need Aux Alignment and Sensors Alignment steps transforms.")
        return

    T_scan_to_dev = np.linalg.inv(gui.T_dev_to_scan)
    T_aux_to_device = T_scan_to_dev @ gui.T_aux_dev_to_scan
    gui.T_aux_to_device = T_aux_to_device

    # Save generated transform
    save_generated_transforms(gui, trans_to_save="aux_to_device")


def compute_device_to_head(gui):
    """Compute transform from device (sensor helmet) to head using the dev->scan and head->scan transforms."""
    if gui.T_dev_to_scan is None or gui.T_head_to_scan is None:
        gui.update_info("Need Sensors Alignment and Head Alignment steps transforms.")
        return

    T_scan_to_head = np.linalg.inv(gui.T_head_to_scan)
    T_dev_to_head = T_scan_to_head @ gui.T_dev_to_scan
    gui.T_dev_to_head = T_dev_to_head

    # Save generated transform
    save_generated_transforms(gui, trans_to_save="dev_to_head")


def obtain_true_landmarks_from_marked_points(
    helmet_landmark_positions, sensor_alignment_points_from_source
):
    """Obtain a list of true landmark coordinates from points marked during sensor alignment step.

    Args:
        helmet_landmark_positions (list): Pre-defined true landmark coordinates
            (e.g., from CAD model or manufacturer specification), shape (n_landmarks, 3).
        sensor_alignment_points_from_source (list): User-marked points on the helmet STL,
            shape (n_marked_points, 3).

    Returns:
        list: True landmark coordinates corresponding to each user-marked point, in the same
            order as sensor_alignment_points_from_source. shape (n_marked_points, 3)
    """
    true_landmark_coords = []

    for marked_point in sensor_alignment_points_from_source:
        # Find closest landmark to marked point
        distances = [
            np.linalg.norm(marked_point - lm) for lm in helmet_landmark_positions
        ]
        closest_idx = np.argmin(distances)
        true_landmark_coords.append(helmet_landmark_positions[closest_idx])

    return true_landmark_coords


def compute_sensors_transform(gui):
    """Compute transform from helmet to master scan using the helmet pairs, then apply to sensors."""
    if len(gui.sensor_alignment_points_from_source) < 3:
        gui.update_info("Need ≥3 point pairs from sensor alignment step.")
        return

    # Compute device (helmet) to scan transform
    compute_dev_to_scan(gui)
    # Compute aux -> device if aux->scan and dev->scan exist
    compute_aux_to_device(gui)
    # Compute device -> head if head transform exists
    compute_device_to_head(gui)

    # overlay transformed STL + transformed sensors
    overlay_helmet_sensors_transformed(gui, gui.T_dev_to_scan)

    # save transformed sensors csv
    save_transformed_sensors_csv(gui)


def compute_head_to_mri_initial(gui):
    """Compute initial head to MRI transform using the fiducials from head alignment (head scan) and mri alignment (MRI)."""
    required = ["NAS", "LPA", "RPA"]

    if not all(fid in gui.marked_fiducials_on_head_scan for fid in required):
        gui.update_info("Head fiducials missing (incomplete head alignment step).")
        return

    if not all(fid in gui.mri_fiducials for fid in required):
        gui.update_info("MRI fiducials missing.")
        return

    src = np.array(
        [
            gui.marked_fiducials_on_head_scan["NAS"],
            gui.marked_fiducials_on_head_scan["LPA"],
            gui.marked_fiducials_on_head_scan["RPA"],
        ]
    )

    dst = np.array(
        [
            gui.mri_fiducials["NAS"],
            gui.mri_fiducials["LPA"],
            gui.mri_fiducials["RPA"],
        ]
    )

    T, rms, errs = compute_rigid_transform(src, dst)

    gui.T_head_to_mri = T
    gui.head_to_mri_rms = rms
    gui.head_to_mri_errors = errs

    gui.update_info(
        f"Head→MRI RMS = {rms:.2f} mm | "
        f"NAS={errs[0]:.2f}, "
        f"LPA={errs[1]:.2f}, "
        f"RPA={errs[2]:.2f}"
    )

    preview_head_on_mri(gui, T)


def compute_surface_rms(src_poly, dst_poly):
    """Compute RMS distance from each point in src_poly to closest point in dst_poly."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(dst_poly)
    locator.BuildLocator()

    src_points = src_poly.GetPoints()
    num_points = src_points.GetNumberOfPoints()

    squared_distances = []

    for i in range(num_points):
        src_point = src_points.GetPoint(i)
        closest_idx = locator.FindClosestPoint(src_point)
        closest_dst_point = dst_poly.GetPoint(closest_idx)
        squared_distances.append(
            np.sum((np.array(src_point) - np.array(closest_dst_point)) ** 2)
        )

    squared_distances = np.array(squared_distances)
    return np.sqrt(np.mean(squared_distances))


def run_icp_head_to_mri(gui):
    """Run ICP to refine head→MRI alignment using the face ROI."""
    if gui.T_head_to_mri is None:
        gui.update_info("Compute initial fiducial alignment first.")
        return

    if gui.head_polydata is None or gui.mri_polydata is None:
        gui.update_info("Need head and MRI surfaces loaded.")
        return

    if not all(
        fid in gui.marked_fiducials_on_head_scan
        for fid in ["NAS", "LPA", "RPA", "NOSE", "CHIN"]
    ):
        gui.update_info("Need 5 head fiducials.")
        return

    # --------------------------------------------------
    # Extract ROI in HEAD space
    # --------------------------------------------------
    roi_in_head_space = extract_face_roi(
        gui, gui.head_polydata, gui.marked_fiducials_on_head_scan
    )
    # # Create a list of marked points from the head fiducials
    # marked_points = list(gui.marked_fiducials_on_head_scan.values())
    # roi_in_head_space = extract_convex_hull_roi(
    #     gui.head_polydata, marked_points, expansion_factor=1.5
    # )

    if roi_in_head_space is None:
        gui.update_info("ROI extraction failed.")
        return

    # --------------------------------------------------
    # Transform ROI to MRI space
    # --------------------------------------------------
    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(gui.T_head_to_mri))

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetTransform(tf)
    tf_filter.SetInputData(roi_in_head_space)
    tf_filter.Update()

    roi_in_mri_space = tf_filter.GetOutput()

    # --------------------------------------------------
    # Initial RMS
    # --------------------------------------------------
    initial_rms = compute_surface_rms(roi_in_mri_space, gui.mri_polydata)

    # --------------------------------------------------
    # ICP
    # --------------------------------------------------
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(roi_in_mri_space)
    icp.SetTarget(gui.mri_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(50)
    icp.StartByMatchingCentroidsOff()
    icp.Modified()
    icp.Update()

    # Extract matrix
    T_icp = np.eye(4)
    icp_matrix = icp.GetMatrix()
    for row in range(4):
        for col in range(4):
            T_icp[row, col] = icp_matrix.GetElement(row, col)

    # Compose
    gui.T_head_to_mri = T_icp @ gui.T_head_to_mri

    # Save generated transform
    save_generated_transforms(gui, trans_to_save="head_to_mri")

    # --------------------------------------------------
    # Recompute RMS after ICP
    # --------------------------------------------------
    tf2 = vtk.vtkTransform()
    tf2.SetMatrix(vtk_matrix_from_numpy(gui.T_head_to_mri))

    tf_filter2 = vtk.vtkTransformPolyDataFilter()
    tf_filter2.SetTransform(tf2)
    tf_filter2.SetInputData(roi_in_head_space)
    tf_filter2.Update()

    roi_refined_in_mri_space = tf_filter2.GetOutput()

    final_rms = compute_surface_rms(roi_refined_in_mri_space, gui.mri_polydata)

    gui.update_info(f"ICP complete | ROI RMS: {initial_rms:.2f} → {final_rms:.2f} mm")

    # --------------------------------------------------
    # Preview full head
    # --------------------------------------------------
    preview_head_on_mri(gui, gui.T_head_to_mri)
    preview_roi_on_mri(gui, roi_refined_in_mri_space)

def extract_face_roi(gui, polydata, fid_dict, plane_offset_mm=15.0):
    """Extract face ROI using robust surface clipping."""
    
    import pyvista as pv

    required = ["NAS", "LPA", "RPA"]
    if not all(fid in fid_dict for fid in required):
        gui.update_info("ROI extraction: missing fiducials")
        return None

    nas  = fid_dict["NAS"]
    lpa  = fid_dict["LPA"]
    rpa  = fid_dict["RPA"]
    nose = fid_dict.get("NOSE", nas)
    chin = fid_dict.get("CHIN", nas)

    # ---- convert VTK → PyVista
    head_mesh = pv.wrap(polydata)

    pts = np.vstack([nas, lpa, rpa, nose])
    center = pts.mean(axis=0)

    radius = 0.75 * np.max(np.linalg.norm(pts - center, axis=1))

    normal = np.cross(lpa - nas, rpa - nas)
    normal /= np.linalg.norm(normal)

    if np.dot(normal, chin - nas) > 0:
        normal *= -1

    plane_origin = nas + plane_offset_mm * normal

    sphere = pv.Sphere(radius=radius, center=center)
    roi = head_mesh.clip_surface(sphere)

    if roi.n_cells == 0:
        gui.update_info("ROI too small — returning None")
        return None

    side = float(np.dot(nose - plane_origin, normal))
    invert = True if side < 0.0 else False

    roi = roi.clip(
        normal=list(normal),
        origin=list(plane_origin),
        invert=invert
    )

    if roi.n_cells < 500:
        gui.update_info("ROI too small — returning None")
        return None

    gui.update_info(f"ROI points: {roi.n_points}")

    return roi

def extract_convex_hull_roi(polydata, marked_points, expansion_factor=1.5):
    """
    Extract ROI containing all points inside the convex hull volume of user-marked points.

    Args:
        polydata: vtkPolyData of the full head surface
        marked_points: List of (x, y, z) points marked by user (N ≥ 4)
        expansion_factor: Factor to expand the hull outward (default 1.5)

    Returns:
        roi_poly: Extracted region as vtkPolyData, or None if too small
    """
    points_array = np.array(marked_points)

    if len(points_array) < 4:
        return None

    # 1. Compute center and expand points outward
    center = np.mean(points_array, axis=0)
    expanded_points = center + expansion_factor * (points_array - center)

    # 2. Build Delaunay triangulation for inside/outside test
    try:
        tri = Delaunay(expanded_points)
    except Exception:
        return None

    # 3. Test every point in polydata for inclusion in hull
    vtk_points = polydata.GetPoints()
    n_pts = vtk_points.GetNumberOfPoints()

    ids = vtk.vtkIdTypeArray()

    for i in range(n_pts):
        p = np.array(vtk_points.GetPoint(i))

        # Check if point is inside the convex hull
        simplex = tri.find_simplex(p)
        if simplex >= 0:
            ids.InsertNextValue(i)

    if ids.GetNumberOfTuples() < 100:
        return None

    # 4. Extract selected points
    selection_node = vtk.vtkSelectionNode()
    selection_node.SetFieldType(vtk.vtkSelectionNode.POINT)
    selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
    selection_node.SetSelectionList(ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selection_node)

    extract = vtk.vtkExtractSelection()
    extract.SetInputData(0, polydata)
    extract.SetInputData(1, selection)
    extract.Update()

    geo = vtk.vtkGeometryFilter()
    geo.SetInputConnection(extract.GetOutputPort())
    geo.Update()

    return geo.GetOutput()
