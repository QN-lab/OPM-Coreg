"""Helper functions for updating the VTK display in response to various events, such as loading files, computing transforms, or toggling options."""

import numpy as np
import vtk

from src.const import FIDUCIAL_COLORS
from src.utils import (
    actor_from_polydata,
    arrow_actor,
    sphere_actor,
    vtk_matrix_from_numpy,
)


def preview_aux_dev_on_master_scan(gui):
    """Merge the auxiliary device STL into the master scan on the target window using the T_aux_dev_to_scan transform."""
    if gui.aux_dev_polydata is None or gui.T_aux_dev_to_scan is None:
        gui.update_info("Need auxiliary device STL and T_aux_dev_to_scan first.")
        return
    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(gui.T_aux_dev_to_scan))
    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetTransform(tf)
    tfilter.SetInputData(gui.aux_dev_polydata)
    tfilter.Update()
    if gui.aux_dev_actor_transformed is not None:
        try:
            gui.ren_target.RemoveActor(gui.aux_dev_actor_transformed)
        except Exception:
            pass
    gui.aux_dev_actor_transformed = actor_from_polydata(
        tfilter.GetOutput(), color=(0.7, 1.0, 0.2), opacity=0.45
    )

    if gui.cb_show_aux_dev.isChecked():
        gui.ren_target.AddActor(gui.aux_dev_actor_transformed)
        gui.ren_target.ResetCameraClippingRange()
        gui.vtk_target.GetRenderWindow().Render()
        gui.update_info("Previewed auxiliary device on master scan.")


def preview_head_scan_on_master_scan(gui):
    """Overlay the head scan transformed by T onto the target window, along with transformed fiducials."""
    # remove old transformed head scan if any
    try:
        if (
            hasattr(gui, "head_actor_transformed")
            and gui.head_actor_transformed is not None
        ):
            gui.ren_target.RemoveActor(gui.head_actor_transformed)
    except Exception:
        pass
    if gui.head_polydata is None:
        gui.update_info(
            "Load head scan STL and compute T_head_to_scan (Head Alignment step) first."
        )
        return
    if gui.T_head_to_scan is None:
        gui.update_info("Compute T_head_to_scan (Head Alignment step) first.")
        return
    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(gui.T_head_to_scan))
    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetTransform(tf)
    tfilter.SetInputData(gui.head_polydata)
    tfilter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(tfilter.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.6, 0.3)
    actor.GetProperty().SetOpacity(0.45)
    gui.head_actor_transformed = actor

    if gui.cb_show_head.isChecked():
        gui.ren_target.AddActor(actor)
        gui.ren_target.ResetCameraClippingRange()
        gui.vtk_target.GetRenderWindow().Render()
        gui.update_info("Previewed head scan on master scan.")


def preview_helmet_stl_on_master_scan(gui, T):
    """Overlay the helmet STL transformed by T onto the target window.

    Args:
        gui: The main GUI object containing the VTK renderers and data.
        T: A 4x4 numpy array representing the rigid transform to apply to the sensor positions and axes for previewing on the target window.
    """
    try:
        if gui.helmet_stl_actor_target is not None:
            gui.ren_target.RemoveActor(gui.helmet_stl_actor_target)
    except Exception:
        pass

    if gui.helmet_stl_raw is None:
        return

    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(T))

    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetTransform(tf)
    tfilter.SetInputData(gui.helmet_stl_raw)
    tfilter.Update()

    gui.helmet_stl_actor_target = actor_from_polydata(
        tfilter.GetOutput(),
        color=(0.3, 0.8, 1.0),
        opacity=0.4,
    )

    if gui.cb_show_helmet.isChecked():
        gui.ren_target.AddActor(gui.helmet_stl_actor_target)
        gui.ren_target.ResetCameraClippingRange()
        gui.vtk_target.GetRenderWindow().Render()
        gui.update_info("Previewed helmet STL on master scan.")


def preview_sensors_on_master_scan(gui, T):
    """Overlay the sensor positions transformed by T onto the target window.

    Args:
        gui: The main GUI object containing the VTK renderers and data.
        T: A 4x4 numpy array representing the rigid transform to apply to the sensor positions and axes for previewing on the target window.
    """
    rotation = T[:3, :3]
    translation = T[:3, 3]

    for actor in gui.transformed_sensor_actors:
        try:
            gui.ren_target.RemoveActor(actor)
        except Exception:
            pass

    gui.transformed_sensor_actors = []

    if gui.sensor_positions is None:
        return

    pts_t = (rotation @ gui.sensor_positions.T).T + translation

    for p in pts_t:
        actor = sphere_actor(
            p,
            color=(0.6, 0.0, 0.6),
            radius=3.2,
        )
        gui.transformed_sensor_actors.append(actor)

    if gui.cb_show_sensors.isChecked():
        for actor in gui.transformed_sensor_actors:
            gui.ren_target.AddActor(actor)
        gui.ren_target.ResetCameraClippingRange()
        gui.vtk_target.GetRenderWindow().Render()
        gui.update_info("Previewed sensor positions on master scan.")


def preview_sensor_unit_vectors_on_master_scan(gui, T):
    """Overlay unit vectors for each sensor transformed by T on the target window, if sensor axes are available.

    Args:
        gui: The main GUI object containing the VTK renderers and data.
        T: A 4x4 numpy array representing the rigid transform to apply to the sensor positions and axes for previewing on the target window.
    """
    rotation = T[:3, :3]
    translation = T[:3, 3]

    for actor in gui.transformed_vector_actors:
        try:
            gui.ren_target.RemoveActor(actor)
        except Exception:
            pass

    gui.transformed_vector_actors = []

    if gui.sensor_positions is None:
        return

    if gui.sensor_axes is None:
        return

    pts_t = (rotation @ gui.sensor_positions.T).T + translation

    axes_t = np.empty_like(gui.sensor_axes)

    for i in range(3):
        axes_t[:, i, :] = (rotation @ gui.sensor_axes[:, i, :].T).T

    for p, vecs in zip(pts_t, axes_t):
        for vec, color in [
            (vecs[0], (1, 0, 0)),
            (vecs[1], (0, 1, 0)),
            (vecs[2], (0, 0, 1)),
        ]:
            ar = arrow_actor(
                p,
                vec,
                scale=10.0,
                color=color,
            )

            if ar is not None:
                gui.transformed_vector_actors.append(ar)

    if gui.cb_show_unit_vectors.isChecked():
        for actor in gui.transformed_vector_actors:
            gui.ren_target.AddActor(actor)
        gui.ren_target.ResetCameraClippingRange()
        gui.vtk_target.GetRenderWindow().Render()
        gui.update_info("Previewed sensor unit vectors on master scan.")


def preview_helmet_sensor_unit_vectors_on_master_scan(gui):
    """Preview the helmet STL, sensor positions, and unit vectors all together on the master scan in the target window."""
    if gui.T_dev_to_scan is None:
        gui.update_info("Need Device→Scan (Sensor Alignment step) first.")
        return

    preview_helmet_stl_on_master_scan(gui, gui.T_dev_to_scan)
    preview_sensors_on_master_scan(gui, gui.T_dev_to_scan)
    preview_sensor_unit_vectors_on_master_scan(gui, gui.T_dev_to_scan)


def draw_sensor_positions_on_source_window(gui):
    """Draw sensor positions as spheres on the source window."""
    for actor in gui.sensor_actors_source_window:
        try:
            gui.ren_source.RemoveActor(actor)
        except Exception:
            pass
    gui.sensor_actors_source_window = []
    for pnt in gui.sensor_positions:
        actor = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
        gui.ren_source.AddActor(actor)
        gui.sensor_actors_source_window.append(actor)
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()


def draw_sensor_vectors_on_source_window(gui):
    """Draw unit vectors for each sensor on the source window."""
    for actor in gui.sensor_vector_actors_source_window:
        try:
            gui.ren_source.RemoveActor(actor)
        except Exception:
            pass
    gui.sensor_vector_actors_source_window = []

    if gui.sensor_axes is not None and gui.sensor_positions is not None:
        for p, vecs in zip(gui.sensor_positions, gui.sensor_axes):
            for vec, color in [
                (vecs[0], (1, 0, 0)),
                (vecs[1], (0, 1, 0)),
                (vecs[2], (0, 0, 1)),
            ]:
                ar = arrow_actor(p, vec, scale=6.0, color=color)
                if ar is not None:
                    gui.ren_source.AddActor(ar)
                    gui.sensor_vector_actors_source_window.append(ar)
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()


def draw_mri_fiducials(gui):
    """Draw fiducials on the source window in MRI frame.

    This is called after loading MRI fiducials, and also after computing head→MRI transform to update the transformed head fiducials."""
    # Remove old actors
    if hasattr(gui, "mri_fiducial_actors"):
        for actor in gui.mri_fiducial_actors.values():
            try:
                gui.ren_source.RemoveActor(actor)
            except Exception:
                pass

    gui.mri_fiducial_actors = {}

    for lbl, pos in gui.mri_fiducials.items():
        actor = sphere_actor(pos, color=FIDUCIAL_COLORS.get(lbl, (1, 1, 0)), radius=3.5)

        gui.ren_source.AddActor(actor)
        gui.mri_fiducial_actors[lbl] = actor

    gui.vtk_source.GetRenderWindow().Render()


def preview_head_on_mri(gui):
    """Preview the head scan transformed by T overlaid on the MRI in the source window."""
    if gui.head_polydata is None:
        return

    if gui.T_head_to_mri is None:
        return

    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(gui.T_head_to_mri))

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(tf)
    filt.SetInputData(gui.head_polydata)
    filt.Update()

    try:
        if (
            hasattr(gui, "head_on_mri_actor_source_window")
            and gui.head_on_mri_actor_source_window is not None
        ):
            gui.ren_source.RemoveActor(gui.head_on_mri_actor_source_window)
    except Exception:
        pass
    gui.head_on_mri_actor_source_window = actor_from_polydata(
        filt.GetOutput(), color=(0.3, 0.6, 1.0), opacity=0.5
    )

    gui.ren_source.AddActor(gui.head_on_mri_actor_source_window)
    gui.vtk_source.GetRenderWindow().Render()


def preview_roi_on_mri(gui):
    """Preview the ROI surface overlaid on the MRI in the source window."""
    if (
        hasattr(gui, "roi_on_mri_actor_source_window")
        and gui.roi_on_mri_actor_source_window is not None
    ):
        try:
            gui.ren_source.RemoveActor(gui.roi_on_mri_actor_source_window)
        except Exception:
            pass

    # Check if the ROI in MRI frame exists
    if not hasattr(gui, "roi_in_mri_frame") or gui.roi_in_mri_frame is None:
        gui.update_info(
            "No ROI in MRI frame to preview. Compute Head→MRI transform and ROI extraction first."
        )
        return

    # Show ROI on MRI in source window (light red)
    gui.roi_on_mri_actor_source_window = actor_from_polydata(
        gui.roi_in_mri_frame, color=(1.0, 0.5, 0.5), opacity=0.6
    )

    gui.ren_source.AddActor(gui.roi_on_mri_actor_source_window)
    gui.vtk_source.GetRenderWindow().Render()


def preview_mri_on_master_scan(gui):
    """Preview the MRI mesh transformed to the scan frame overlaid on the master scan in the target window view."""
    if gui.T_head_to_mri is None:
        gui.update_info("Compute Head→MRI (MRI ALignment step) first.")
        return

    if gui.T_head_to_scan is None:
        gui.update_info("Need Head→Scan (Head Alignment step) first.")
        return

    if gui.mri_polydata is None:
        gui.update_info("Load MRI mesh first.")
        return

    # MRI is in MRI frame
    # We want MRI in SCAN frame (target window)

    # head -> mri  = T_head_to_mri
    # head -> scan = T_head_to_scan
    # So:

    # mri -> head
    T_mri_to_head = np.linalg.inv(gui.T_head_to_mri)

    # head -> scan
    T_head_to_scan = gui.T_head_to_scan

    # mri -> scan
    T_mri_to_scan = T_head_to_scan @ T_mri_to_head

    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(T_mri_to_scan))

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(tf)
    filt.SetInputData(gui.mri_polydata)
    filt.Update()

    # remove old if exists
    if gui.mri_actor_target_window is not None:
        gui.ren_target.RemoveActor(gui.mri_actor_target_window)

    gui.mri_actor_target_window = actor_from_polydata(
        filt.GetOutput(),
        color=(1.0, 0.6, 0.8),  # pink
        opacity=0.4,
    )

    if gui.cb_show_mri.isChecked():
        gui.ren_target.AddActor(gui.mri_actor_target_window)
        gui.ren_target.ResetCameraClippingRange()
        gui.vtk_target.GetRenderWindow().Render()
        gui.update_info("Previewed MRI on master scan (target window).")
