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


def merge_aux_dev_into_helmet(gui):
    """Merge the auxiliary device STL (e.g. coil) into the helmet STL on the RIGHT view using the aux_dev_to_scan transform."""
    if gui.coil_polydata is None or gui.aux_dev_to_scan is None:
        gui._update_info("Need coil STL and T_aux_dev_to_scan first.")
        return
    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(gui.aux_dev_to_scan))
    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetTransform(tf)
    tfilter.SetInputData(gui.coil_polydata)
    tfilter.Update()
    if gui.coil_actor_transformed is not None:
        try:
            gui.ren_right.RemoveActor(gui.coil_actor_transformed)
        except Exception:
            pass
    gui.coil_actor_transformed = actor_from_polydata(
        tfilter.GetOutput(), color=(0.7, 1.0, 0.2), opacity=0.45
    )
    gui.ren_right.AddActor(gui.coil_actor_transformed)
    gui.ren_right.ResetCameraClippingRange()
    gui.vtk_right.GetRenderWindow().Render()
    gui._update_info("Previewed coil on helmet.")


def overlay_head_transformed(gui, T):
    """Overlay the head scan transformed by T onto the RIGHT view, along with transformed fiducials.

    Args:
        gui: The main GUI object containing the VTK renderers and data.
        T: A 4x4 numpy array representing the rigid transform to apply to the head scan and fiducials.
    """
    # remove old transformed head scan if any
    try:
        if (
            hasattr(gui, "head_actor_transformed")
            and gui.head_actor_transformed is not None
        ):
            gui.ren_right.RemoveActor(gui.head_actor_transformed)
    except Exception:
        pass
    if gui.head_polydata is None:
        return
    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(T))
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
    gui.ren_right.AddActor(actor)
    # transform fiducials too and plot on right
    rotation = T[:3, :3]
    translation = T[:3, 3]
    for lbl, p in gui.head_fiducials.items():
        p_hel = rotation @ p + translation
        try:
            # if existing fid actor on right, remove it first
            if (
                hasattr(gui, "head_fiducial_actors_right")
                and lbl in gui.head_fiducial_actors_right
            ):
                gui.ren_right.RemoveActor(gui.head_fiducial_actors_right[lbl])
        except Exception:
            pass
        actor = sphere_actor(p_hel, color=FIDUCIAL_COLORS[lbl], radius=3.5)
        if not hasattr(gui, "head_fiducial_actors_right"):
            gui.head_fiducial_actors_right = {}
        gui.ren_right.AddActor(actor)
        gui.head_fiducial_actors_right[lbl] = actor
    gui.ren_right.ResetCameraClippingRange()
    gui.vtk_right.GetRenderWindow().Render()


def overlay_step3_transformed(gui, T):
    """Overlay the helmet STL and sensors transformed by T onto the RIGHT view.

    Args:
        gui: The main GUI object containing the VTK renderers and data.
        T: A 4x4 numpy array representing the rigid transform to apply to the helmet STL and sensor positions.
    """
    # transformed STL preview on RIGHT
    rotation = T[:3, :3]
    translation = T[:3, 3]
    try:
        if gui.helmet_stl_actor_overlay is not None:
            gui.ren_right.RemoveActor(gui.helmet_stl_actor_overlay)
    except Exception:
        pass
    if gui.helmet_stl_raw is not None:
        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix_from_numpy(T))
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(tf)
        tfilter.SetInputData(gui.helmet_stl_raw)
        tfilter.Update()
        gui.helmet_stl_actor_overlay = actor_from_polydata(
            tfilter.GetOutput(), color=(0.3, 0.8, 1.0), opacity=0.4
        )
        if gui.cb_stl.isChecked():
            gui.ren_right.AddActor(gui.helmet_stl_actor_overlay)
    # transformed sensors on right
    for actor in gui.transformed_sensor_actors:
        try:
            gui.ren_right.RemoveActor(actor)
        except Exception:
            pass
    for actor in gui.transformed_vector_actors:
        try:
            gui.ren_right.RemoveActor(actor)
        except Exception:
            pass
    gui.transformed_sensor_actors = []
    gui.transformed_vector_actors = []
    if gui.sensor_positions is not None:
        pts_t = (rotation @ gui.sensor_positions.T).T + translation
        for p in pts_t:
            actor = sphere_actor(p, color=(0.6, 0.0, 0.6), radius=3.2)
            gui.ren_right.AddActor(actor)
            gui.transformed_sensor_actors.append(actor)
        if gui.cb_vec.isChecked() and (gui.sensor_axes is not None):
            axes_t = np.empty_like(gui.sensor_axes)
            for i in range(3):
                axes_t[:, i, :] = (rotation @ gui.sensor_axes[:, i, :].T).T
            for p, vecs in zip(pts_t, axes_t):
                for vec, color in [
                    (vecs[0], (1, 0, 0)),
                    (vecs[1], (0, 1, 0)),
                    (vecs[2], (0, 0, 1)),
                ]:
                    ar = arrow_actor(p, vec, scale=10.0, color=color)
                    if ar is not None:
                        gui.ren_right.AddActor(ar)
                        gui.transformed_vector_actors.append(ar)
    gui.ren_right.ResetCameraClippingRange()
    gui.vtk_right.GetRenderWindow().Render()


def draw_sensor_vectors_left(gui):
    """Draw unit vectors for each sensor on the LEFT view, if toggled on and if sensor axes are available."""
    # rebuild left view: helmet_stl + sensor spheres + optional vectors
    gui.ren_left.RemoveAllViewProps()
    if gui.helmet_stl_actor_left is not None:
        gui.ren_left.AddActor(gui.helmet_stl_actor_left)
    gui.sensor_actors_left = []
    if gui.sensor_positions is not None:
        for pnt in gui.sensor_positions:
            actor = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
            gui.ren_left.AddActor(actor)
            gui.sensor_actors_left.append(actor)
    if gui.cb_vec.isChecked() and (gui.sensor_axes is not None):
        for p, vecs in zip(gui.sensor_positions, gui.sensor_axes):
            for vec, color in [
                (vecs[0], (1, 0, 0)),
                (vecs[1], (0, 1, 0)),
                (vecs[2], (0, 0, 1)),
            ]:
                ar = arrow_actor(p, vec, scale=6.0, color=color)
                if ar is not None:
                    gui.ren_left.AddActor(ar)
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()


def draw_mri_fiducials(gui):
    """Draw fiducials on the LEFT view in MRI frame.

    This is called after loading MRI fiducials, and also after computing head→MRI transform to update the transformed head fiducials."""
    # Remove old actors
    if hasattr(gui, "mri_fiducial_actors"):
        for actor in gui.mri_fiducial_actors.values():
            try:
                gui.ren_left.RemoveActor(actor)
            except Exception:
                pass

    gui.mri_fiducial_actors = {}

    for lbl, pos in gui.mri_fiducials.items():
        actor = sphere_actor(pos, color=FIDUCIAL_COLORS.get(lbl, (1, 1, 0)), radius=3.5)

        gui.ren_left.AddActor(actor)
        gui.mri_fiducial_actors[lbl] = actor

    gui.vtk_left.GetRenderWindow().Render()


def preview_head_on_mri(gui, T):
    """Preview the head scan transformed by T overlaid on the MRI in the LEFT view.

    Args:
        gui: The main GUI object containing the VTK renderers and data.
        T: A 4x4 numpy array representing the rigid transform to apply to the head scan for previewing on the MRI.
    """
    if gui.head_polydata is None:
        return

    tf = vtk.vtkTransform()
    tf.SetMatrix(vtk_matrix_from_numpy(T))

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(tf)
    filt.SetInputData(gui.head_polydata)
    filt.Update()

    actor = actor_from_polydata(filt.GetOutput(), color=(0.3, 0.6, 1.0), opacity=0.5)

    gui.ren_left.AddActor(actor)
    gui.vtk_left.GetRenderWindow().Render()


def preview_mri_on_scan(gui):
    """Preview the MRI mesh transformed to the scan frame overlaid on the master scan in the RIGHT view."""
    if gui.T_head_to_mri is None:
        gui._update_info("Compute Head→MRI first.")
        return

    if gui.T_head_to_scan is None:
        gui._update_info("Need Head→Scan (Step2) first.")
        return

    if gui.mri_polydata is None:
        gui._update_info("Load MRI mesh first.")
        return

    # MRI is in MRI frame
    # We want MRI in SCAN frame (RIGHT window)

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
    if gui.mri_actor_right is not None:
        gui.ren_right.RemoveActor(gui.mri_actor_right)

    gui.mri_actor_right = actor_from_polydata(
        filt.GetOutput(),
        color=(1.0, 0.6, 0.8),  # pink
        opacity=0.4,
    )

    gui.ren_right.AddActor(gui.mri_actor_right)
    gui.vtk_right.GetRenderWindow().Render()

    gui._update_info("Previewed MRI on master scan (RIGHT).")
