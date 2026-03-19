"""Contains helper functions for toggling the visibility of various elements in the GUI."""

from src.display_helpers import (
    draw_sensor_vectors_left,
    overlay_step3_transformed,
)


def toggle_vectors(gui, state):
    """Toggle the visibility of the unit vectors in the left view.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    gui.show_vectors = bool(state)
    # redraw left vectors and transformed vectors if transform exists
    draw_sensor_vectors_left(gui)
    if gui.T_dev_to_scan is not None:
        overlay_step3_transformed(gui, gui.T_dev_to_scan)


def toggle_helmet_overlay(gui, state):
    """Toggle the visibility of the helmet STL overlay in the right view.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.helmet_stl_actor_overlay is None:
        return
    if gui.cb_stl.isChecked():
        gui.ren_right.AddActor(gui.helmet_stl_actor_overlay)
    else:
        try:
            gui.ren_right.RemoveActor(gui.helmet_stl_actor_overlay)
        except Exception:
            pass
    gui.vtk_right.GetRenderWindow().Render()


def toggle_aux_dev_overlay(gui, state):
    """Toggle the visibility of the auxiliary device (TMS coil) overlay in the right view.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.coil_actor_transformed is None:
        return
    if state:
        gui.ren_right.AddActor(gui.coil_actor_transformed)
    else:
        gui.ren_right.RemoveActor(gui.coil_actor_transformed)
    gui.vtk_right.GetRenderWindow().Render()


def toggle_head_overlay(gui, state):
    """Toggle the visibility of the head scan overlay in the right view.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if not hasattr(gui, "head_actor_transformed") or gui.head_actor_transformed is None:
        return
    if state:
        gui.ren_right.AddActor(gui.head_actor_transformed)
    else:
        gui.ren_right.RemoveActor(gui.head_actor_transformed)
    gui.vtk_right.GetRenderWindow().Render()


def toggle_mri_overlay(gui, state):
    """Toggle the visibility of the MRI scan overlay in the right view.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if not hasattr(gui, "mri_actor_right") or gui.mri_actor_right is None:
        return
    if state:
        gui.ren_right.AddActor(gui.mri_actor_right)
    else:
        gui.ren_right.RemoveActor(gui.mri_actor_right)
    gui.vtk_right.GetRenderWindow().Render()
