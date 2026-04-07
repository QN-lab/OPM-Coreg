"""Helper functions to compute transforms based on GUI state and user inputs."""


import numpy as np
import vtk

from src.display_helpers import (
    overlay_head_transformed,
    overlay_helmet_sensors_transformed,
    preview_head_on_mri,
)
from src.utils import (
    actor_from_polydata,
    compute_rigid_transform,
    vtk_matrix_from_numpy,
)
from src.file_io_helpers import save_transformed_sensors_csv, auto_save_transform


def compute_aux_dev_to_scan(gui):
    """Compute transform from auxiliary device to master scan using the auxiliary device pairs."""
    if not getattr(gui, "aux_dev_pairs_src", None) or len(gui.aux_dev_pairs_src) < 3:
        gui._update_info("Need ≥3 auxiliary device pairs.")
        return
    src = np.vstack(gui.aux_dev_pairs_src)
    dst = np.vstack(gui.aux_dev_pairs_dst)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_aux_dev_to_scan = T
    gui._update_info(
        f"Computed auxiliary device to master scan transform RMS={rms:.3f} mm."
    )
    # auto save available transforms
    auto_save_transform(gui)


def compute_dev_to_scan(gui):
    """Compute transform from device (sensor helmet) to master scan using the device pairs."""
    if (
        not getattr(gui, "ref_point_pairs_helmet_stl", None)
        or len(gui.ref_point_pairs_helmet_stl) < 3
    ):
        gui._update_info("Need ≥3 point pairs from sensor alignment step.")
        return
    src = np.vstack(gui.ref_point_pairs_helmet_stl)
    dst = np.vstack(gui.ref_point_pairs_scan)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_dev_to_scan = T
    gui._update_info(
        f"Computed device (helmet) to master scan transform RMS={rms:.3f} mm."
    )
    # auto save available transforms
    auto_save_transform(gui)


def compute_head_to_scan(gui):
    """Compute transform from head scan to master scan using the head pairs."""
    if len(gui.head_alignment_points_from_source) < 3:
        gui._update_info("Need at least 3 head alignment pairs to compute transform.")
        return
    src = np.vstack(gui.head_alignment_points_from_source)
    dst = np.vstack(gui.head_alignment_points_from_target)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_head_to_scan = T
    gui._update_info(
        f"Computed head_to_scan RMS={rms:.3f} mm. Overlaying transformed head scan onto target window."
    )
    # overlay transformed head scan on target window (light orange)
    overlay_head_transformed(gui, T)

    # auto save available transforms
    auto_save_transform(gui)


def compute_aux_to_device(gui):
    """Compute transform from auxiliary device to device (sensor helmet).

    Using the aux->scan transform and dev->scan transform."""
    if gui.T_aux_dev_to_scan is None or gui.T_dev_to_scan is None:
        gui._update_info("Need Aux Alignment and Sensors Alignment steps transforms.")
        return

    T_scan_to_dev = np.linalg.inv(gui.T_dev_to_scan)
    T_aux_to_device = T_scan_to_dev @ gui.T_aux_dev_to_scan
    gui.T_aux_to_device = T_aux_to_device

    # auto save available transforms
    auto_save_transform(gui)


def compute_device_to_head(gui):
    """Compute transform from device (sensor helmet) to head using the dev->scan and head->scan transforms."""
    if gui.T_dev_to_scan is None or gui.T_head_to_scan is None:
        gui._update_info("Need Sensors Alignment and Head Alignment steps transforms.")
        return

    T_scan_to_head = np.linalg.inv(gui.T_head_to_scan)
    T_dev_to_head = T_scan_to_head @ gui.T_dev_to_scan
    gui.T_dev_to_head = T_dev_to_head

    # auto save available transforms
    auto_save_transform(gui)


def obtain_true_landmarks_from_marked_points(
    helmet_landmark_positions, ref_point_pairs_helmet_stl
):
    """Obtain a list of true landmark coordinates from points marked during sensor alignment step.

    Args:
        helmet_landmark_positions (list): Pre-defined true landmark coordinates
            (e.g., from CAD model or manufacturer specification), shape (n_landmarks, 3).
        ref_point_pairs_helmet_stl (list): User-marked points on the helmet STL,
            shape (n_marked_points, 3).

    Returns:
        list: True landmark coordinates corresponding to each user-marked point, in the same
            order as ref_point_pairs_helmet_stl. shape (n_marked_points, 3)
    """
    true_landmark_coords = []

    for marked_point in ref_point_pairs_helmet_stl:
        # Find closest landmark to marked point
        distances = [
            np.linalg.norm(marked_point - lm) for lm in helmet_landmark_positions
        ]
        closest_idx = np.argmin(distances)
        true_landmark_coords.append(helmet_landmark_positions[closest_idx])

    return true_landmark_coords


def compute_sensors_transform(gui):
    """Compute transform from helmet to master scan using the helmet pairs, then apply to sensors."""
    if len(gui.ref_point_pairs_helmet_stl) < 3:
        gui._update_info("Need ≥3 point pairs from sensor alignment step.")
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

    # auto save available transforms
    auto_save_transform(gui)


def compute_head_to_mri_initial(gui):
    """Compute initial head to MRI transform using the fiducials from head alignment (head scan) and mri alignment (MRI)."""
    required = ["NAS", "LPA", "RPA"]

    if not all(fid in gui.head_fiducials for fid in required):
        gui._update_info("Head fiducials missing (incomplete head alignment step).")
        return

    if not all(fid in gui.mri_fiducials for fid in required):
        gui._update_info("MRI fiducials missing.")
        return

    src = np.array(
        [
            gui.head_fiducials["NAS"],
            gui.head_fiducials["LPA"],
            gui.head_fiducials["RPA"],
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

    print("Head→MRI RMS:", rms)
    print("Errors per fiducial:", errs)

    gui._update_info(
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
        gui._update_info("Compute initial fiducial alignment first.")
        return

    if gui.head_polydata is None or gui.mri_polydata is None:
        gui._update_info("Need head and MRI surfaces loaded.")
        return

    if not all(
        fid in gui.head_fiducials for fid in ["NAS", "LPA", "RPA", "NOSE", "CHIN"]
    ):
        gui._update_info("Need 5 head fiducials.")
        return

    # --------------------------------------------------
    # 1) Extract ROI in HEAD space
    # --------------------------------------------------
    roi_head_native = extract_face_roi(gui.head_polydata, gui.head_fiducials)

    if roi_head_native is None:
        gui._update_info("ROI extraction failed.")
        return

    # --------------------------------------------------
    # 2) Transform ROI to MRI space
    # --------------------------------------------------
    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(gui.T_head_to_mri))

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetTransform(tf)
    tf_filter.SetInputData(roi_head_native)
    tf_filter.Update()

    roi_head = tf_filter.GetOutput()

    # --------------------------------------------------
    # 3) Show ROI in source window (red)
    # --------------------------------------------------
    if hasattr(gui, "_roi_actor") and gui._roi_actor is not None:
        try:
            gui.ren_source.RemoveActor(gui._roi_actor)
        except Exception:
            pass

    gui._roi_actor = actor_from_polydata(roi_head, color=(1, 0, 0), opacity=0.7)

    gui.ren_source.AddActor(gui._roi_actor)
    gui.vtk_source.GetRenderWindow().Render()

    # --------------------------------------------------
    # 4) Initial RMS
    # --------------------------------------------------
    initial_rms = compute_surface_rms(roi_head, gui.mri_polydata)

    # --------------------------------------------------
    # 5) ICP
    # --------------------------------------------------
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(roi_head)
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

    # save available transforms after ICP refinement
    auto_save_transform(gui)

    # --------------------------------------------------
    # 6) Recompute RMS after ICP
    # --------------------------------------------------
    tf2 = vtk.vtkTransform()
    tf2.SetMatrix(vtk_matrix_from_numpy(gui.T_head_to_mri))

    tf_filter2 = vtk.vtkTransformPolyDataFilter()
    tf_filter2.SetTransform(tf2)
    tf_filter2.SetInputData(roi_head_native)
    tf_filter2.Update()

    roi_refined = tf_filter2.GetOutput()

    final_rms = compute_surface_rms(roi_refined, gui.mri_polydata)

    gui._update_info(f"ICP complete | ROI RMS: {initial_rms:.2f} → {final_rms:.2f} mm")

    # --------------------------------------------------
    # 7) Preview full head
    # --------------------------------------------------
    preview_head_on_mri(gui, gui.T_head_to_mri)


def extract_face_roi(polydata, fid_dict, plane_offset_mm=15.0):
    """Extract a region of interest (ROI) from the head scan polydata based on fiducials defining the face area, and return the subset polydata for registration to MRI."""
    required = ["NAS", "LPA", "RPA", "NOSE", "CHIN"]
    if not all(fid in fid_dict for fid in required):
        print("ROI extraction: missing fiducials")
        return None

    nas = fid_dict["NAS"]
    lpa = fid_dict["LPA"]
    rpa = fid_dict["RPA"]
    nose = fid_dict["NOSE"]
    chin = fid_dict["CHIN"]

    # --------------------------------------------------
    # 1) ROI center (face region)
    # --------------------------------------------------
    fiducial_points = np.vstack([nas, lpa, rpa, nose, chin])
    roi_center = np.mean(fiducial_points, axis=0)

    max_dist = np.max(np.linalg.norm(fiducial_points - roi_center, axis=1))
    roi_radius = 2.2 * max_dist  # larger radius to include LPA/RPA fully

    # --------------------------------------------------
    # 2) Define fiducial plane (NAS–LPA–RPA)
    # --------------------------------------------------
    v1 = lpa - nas
    v2 = rpa - nas
    plane_normal = np.cross(v1, v2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Make sure normal points upward (towards top of head)
    if np.dot(plane_normal, chin - nas) > 0:
        plane_normal = -plane_normal

    # Shift plane upward
    plane_point = nas + plane_offset_mm * plane_normal

    # --------------------------------------------------
    # 3) Select points
    # --------------------------------------------------
    points = polydata.GetPoints()
    n_pts = points.GetNumberOfPoints()

    ids = vtk.vtkIdTypeArray()

    for i in range(n_pts):
        p = np.array(points.GetPoint(i))

        # radial constraint
        if np.linalg.norm(p - roi_center) > roi_radius:
            continue

        # chop-off plane: keep only points BELOW shifted plane
        if np.dot(p - plane_point, plane_normal) > 0:
            continue

        ids.InsertNextValue(i)

    if ids.GetNumberOfTuples() < 500:
        print("ROI too small — returning None")
        return None

    # --------------------------------------------------
    # 4) Extract subset
    # --------------------------------------------------
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

    roi_poly = geo.GetOutput()

    print("ROI points:", roi_poly.GetNumberOfPoints())
    return roi_poly
