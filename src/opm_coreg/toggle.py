"""Contains helper functions for toggling the visibility of various elements in the GUI."""


def toggle_unit_vectors_overlay(gui, state):
    """Toggle the visibility of the unit vectors in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.transformed_vector_actors == []:
        return
    if state:
        for actor in gui.transformed_vector_actors:
            gui.ren_target.AddActor(actor)
    else:
        for actor in gui.transformed_vector_actors:
            try:
                gui.ren_target.RemoveActor(actor)
            except Exception:
                pass
    gui.vtk_target.GetRenderWindow().Render()


def toggle_helmet_overlay(gui, state):
    """Toggle the visibility of the helmet STL overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.helmet_stl_actor_target is None:
        return
    if state:
        gui.ren_target.AddActor(gui.helmet_stl_actor_target)
    else:
        try:
            gui.ren_target.RemoveActor(gui.helmet_stl_actor_target)
        except Exception:
            pass
    gui.vtk_target.GetRenderWindow().Render()


def toggle_sensors_overlay(gui, state):
    """Toggle the visibility of the sensor positions overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.transformed_sensor_actors == []:
        return
    if state:
        for actor in gui.transformed_sensor_actors:
            gui.ren_target.AddActor(actor)
    else:
        for actor in gui.transformed_sensor_actors:
            try:
                gui.ren_target.RemoveActor(actor)
            except Exception:
                pass
    gui.vtk_target.GetRenderWindow().Render()


def toggle_aux_dev_overlay(gui, state):
    """Toggle the visibility of the auxiliary device overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if gui.aux_dev_actor_transformed is None:
        return
    if state:
        gui.ren_target.AddActor(gui.aux_dev_actor_transformed)
    else:
        gui.ren_target.RemoveActor(gui.aux_dev_actor_transformed)
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
    if (
        not hasattr(gui, "mri_actor_target_window")
        or gui.mri_actor_target_window is None
    ):
        return
    if state:
        gui.ren_target.AddActor(gui.mri_actor_target_window)
    else:
        gui.ren_target.RemoveActor(gui.mri_actor_target_window)
    gui.vtk_target.GetRenderWindow().Render()


def toggle_master_scan_overlay(gui, state):
    """Toggle the visibility of the master scan overlay in the target window.

    Args:
        gui: The main GUI object.
        state: The state of the checkbox.
    """
    if not hasattr(gui, "master_scan_actor") or gui.master_scan_actor is None:
        return
    if state:
        gui.ren_target.AddActor(gui.master_scan_actor)
    else:
        gui.ren_target.RemoveActor(gui.master_scan_actor)
    gui.vtk_target.GetRenderWindow().Render()
