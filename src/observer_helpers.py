"""Helper functions for managing VTK observers in the GUI."""


def add_left_observer(gui, evt, func):
    """Add an observer to the left interactor style and keep track of its ID.

    Args:
        gui: The GUI object containing the left interactor.
        evt: The event name to observe (e.g., "LeftButtonPressEvent").
        func: The callback function to call when the event occurs.
    """
    style = gui.left_iren.GetInteractorStyle()
    oid = style.AddObserver(evt, func)
    gui._observer_ids.append(("left_style", oid))


def add_right_observer(gui, evt, func):
    """Add an observer to the right interactor style and keep track of its ID.

    Args:
        gui: The GUI object containing the right interactor.
        evt: The event name to observe (e.g., "LeftButtonPressEvent").
        func: The callback function to call when the event occurs.
    """
    style = gui.right_iren.GetInteractorStyle()
    oid = style.AddObserver(evt, func)
    gui._observer_ids.append(("right_style", oid))


def remove_all_observers(gui):
    """Remove all observers that have been added to the interactor styles.

    Args:
        gui: The GUI object containing the interactor styles and observer IDs.
    """
    for side, oid in list(gui._observer_ids):
        try:
            if side == "left_style":
                gui.left_iren.GetInteractorStyle().RemoveObserver(oid)
            elif side == "right_style":
                gui.right_iren.GetInteractorStyle().RemoveObserver(oid)
        except Exception:
            pass
        gui._observer_ids.remove((side, oid))


def clear_left_renderer(gui):
    """Clear all actors from the left renderer and reset the camera.

    Args:
        gui: The GUI object containing the left renderer and VTK window.
    """
    gui.ren_left.RemoveAllViewProps()
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()


def clear_right_renderer_keep_scan(gui):
    """Clear all actors from the right renderer except the helmet (master scan) and reset the camera.

    Args:
        gui: The GUI object containing the right renderer, helmet actor, and VTK window.
    """
    gui.ren_right.RemoveAllViewProps()
    if gui.helmet_actor is not None:
        gui.ren_right.AddActor(gui.helmet_actor)
    gui.ren_right.ResetCameraClippingRange()
    gui.vtk_right.GetRenderWindow().Render()
