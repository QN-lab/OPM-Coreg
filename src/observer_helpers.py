"""Helper functions for managing VTK observers in the GUI."""


def add_source_observer(gui, evt, func):
    """Add an observer to the source interactor style and keep track of its ID.

    Args:
        gui: The GUI object containing the source interactor.
        evt: The event name to observe (e.g., "LeftButtonPressEvent").
        func: The callback function to call when the event occurs.
    """
    style = gui.iren_source.GetInteractorStyle()
    oid = style.AddObserver(evt, func)
    gui._observer_ids.append(("source_style", oid))


def add_target_observer(gui, evt, func):
    """Add an observer to the target interactor style and keep track of its ID.

    Args:
        gui: The GUI object containing the target interactor.
        evt: The event name to observe (e.g., "LeftButtonPressEvent").
        func: The callback function to call when the event occurs.
    """
    style = gui.iren_target.GetInteractorStyle()
    oid = style.AddObserver(evt, func)
    gui._observer_ids.append(("target_style", oid))


def remove_all_observers(gui):
    """Remove all observers that have been added to the interactor styles.

    Args:
        gui: The GUI object containing the interactor styles and observer IDs.
    """
    for side, oid in list(gui._observer_ids):
        try:
            if side == "source_style":
                gui.iren_source.GetInteractorStyle().RemoveObserver(oid)
            elif side == "target_style":
                gui.iren_target.GetInteractorStyle().RemoveObserver(oid)
        except Exception:
            pass
        gui._observer_ids.remove((side, oid))


def clear_source_renderer(gui):
    """Clear all actors from the source renderer and reset the camera.

    Args:
        gui: The GUI object containing the source renderer and VTK window.
    """
    gui.ren_source.RemoveAllViewProps()
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()


def clear_target_renderer_keep_scan(gui):
    """Clear all actors from the target renderer except the master scan and reset the camera.

    Args:
        gui: The GUI object containing the target renderer, master scan actor, and VTK window.
    """
    gui.ren_target.RemoveAllViewProps()
    if gui.master_scan_actor is not None:
        gui.ren_target.AddActor(gui.master_scan_actor)
    gui.ren_target.ResetCameraClippingRange()
    gui.vtk_target.GetRenderWindow().Render()
