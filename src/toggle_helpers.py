"""Contains helper functions for toggling the visibility of various elements in the GUI."""

from src.display_helpers import (
    draw_sensor_vectors_source_window,
    overlay_step3_transformed,
)


def toggle_vectors(gui, state):
    """Toggle the visibility of the unit vectors in the source window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    gui.show_vectors = bool(state)
    # redraw source window vectors and transformed vectors if transform exists
    draw_sensor_vectors_source_window(gui)
    if gui.T_dev_to_scan is not None:
        overlay_step3_transformed(gui, gui.T_dev_to_scan)


def toggle_helmet_overlay(gui, state):
    """Toggle the visibility of the helmet STL overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.helmet_stl_actor_target is None:
        return
    if gui.cb_stl.isChecked():
        gui.ren_target.AddActor(gui.helmet_stl_actor_target)
    else:
        try:
            gui.ren_target.RemoveActor(gui.helmet_stl_actor_target)
        except Exception:
            pass
    gui.vtk_target.GetRenderWindow().Render()


def toggle_aux_dev_overlay(gui, state):
    """Toggle the visibility of the auxiliary device (TMS coil) overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.coil_actor_transformed is None:
        return
    if state:
        gui.ren_target.AddActor(gui.coil_actor_transformed)
    else:
        gui.ren_target.RemoveActor(gui.coil_actor_transformed)
    gui.vtk_target.GetRenderWindow().Render()


def toggle_head_overlay(gui, state):
    """Toggle the visibility of the head scan overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if not hasattr(gui, "head_actor_transformed") or gui.head_actor_transformed is None:
        return
    if state:
        gui.ren_target.AddActor(gui.head_actor_transformed)
    else:
        gui.ren_target.RemoveActor(gui.head_actor_transformed)
    gui.vtk_target.GetRenderWindow().Render()


def toggle_mri_overlay(gui, state):
    """Toggle the visibility of the MRI scan overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if not hasattr(gui, "mri_actor_target_window") or gui.mri_actor_target_window is None:
        return
    if state:
        gui.ren_target.AddActor(gui.mri_actor_target_window)
    else:
        gui.ren_target.RemoveActor(gui.mri_actor_target_window)
    gui.vtk_target.GetRenderWindow().Render()
