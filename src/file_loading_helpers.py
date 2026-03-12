import csv
import os

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog

from src.const import PAIR_COLORS
from src.utils import (
    actor_from_polydata,
    load_mesh_to_polydata,
    sphere_actor,
)
from mne.io import read_fiducials
import vtk
from mne.io.constants import FIFF


def load_master_scan(gui):
    """Load master scan (with participant in the helmet) on the RIGHT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui,
        "Select master scan (with participant in the helmet)",
        "",
        "PLY/STL Files (*.ply *.stl)",
    )
    if not filepath:
        return
    gui.helmet_polydata = load_mesh_to_polydata(filepath)
    gui.helmet_filename = os.path.basename(filepath)
    if gui.helmet_actor is not None:
        try:
            gui.ren_right.RemoveActor(gui.helmet_actor)
        except Exception:
            pass
    gui.helmet_actor = actor_from_polydata(
        gui.helmet_polydata, color=(0.85, 0.85, 0.85), opacity=1.0
    )
    gui.ren_right.AddActor(gui.helmet_actor)
    gui.ren_right.ResetCamera()
    gui.vtk_right.GetRenderWindow().Render()
    gui._update_info(f"Loaded master scan: {gui.helmet_filename}")


def load_aux_dev_stl(gui):
    """Load auxiliary device STL (e.g. coil) on the LEFT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Select coil STL", "", "STL Files (*.stl)"
    )
    if not filepath:
        return
    gui.coil_polydata = load_mesh_to_polydata(filepath)
    if gui.coil_actor_left is not None:
        try:
            gui.ren_left.RemoveActor(gui.coil_actor_left)
        except Exception:
            pass
    gui.coil_actor_left = actor_from_polydata(
        gui.coil_polydata, color=(0.7, 1, 0.2), opacity=1.0
    )
    gui.ren_left.AddActor(gui.coil_actor_left)
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info("Loaded coil STL (left).")


def load_aux_dev_landmarks(gui):
    """Load auxiliary device (coil) landmarks CSV on the LEFT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Select coil landmarks CSV", "", "CSV Files (*.csv)"
    )
    if not filepath:
        return
    lm = {}
    with open(filepath, "r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
        for row in rows[1:]:
            if len(row) < 4:
                continue
            lab = row[0].strip()
            try:
                x, y, z = float(row[1]), float(row[2]), float(row[3])
            except Exception:
                continue
            lm[lab] = np.array([x, y, z], float)
    gui.coil_landmarks = lm
    # draw landmarks on left
    for actor in getattr(gui, "coil_landmark_actors_left", []):
        try:
            gui.ren_left.RemoveActor(actor)
        except Exception:
            pass
    gui.coil_landmark_actors_left = []
    for i, (lab, point_coords) in enumerate(gui.coil_landmarks.items()):
        selected_color = PAIR_COLORS[i % len(PAIR_COLORS)]
        actor = sphere_actor(point_coords, color=selected_color, radius=3.5)
        gui.ren_left.AddActor(actor)
        gui.coil_landmark_actors_left.append(actor)
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info(f"Loaded {len(gui.coil_landmarks)} coil landmarks.")


def load_head_scan(gui):
    """Load head scan (without helmet) on the LEFT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Select OUTSIDE head scan (LEFT)", "", "PLY/STL Files (*.ply *.stl)"
    )
    if not filepath:
        return
    gui.head_polydata = load_mesh_to_polydata(filepath)
    if gui.head_actor is not None:
        try:
            gui.ren_left.RemoveActor(gui.head_actor)
        except Exception:
            pass
    gui.head_actor = actor_from_polydata(
        gui.head_polydata, color=(0.8, 0.8, 0.95), opacity=1.0
    )
    gui.ren_left.AddActor(gui.head_actor)
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info(f"Loaded outside head scan: {os.path.basename(filepath)}")


def load_helmet_cad(gui):
    """Load helmet CAD (STL) on the LEFT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Load helmet STL", "", "STL/PLY Files (*.stl *.ply)"
    )
    if not filepath:
        return
    gui.helmet_stl_raw = load_mesh_to_polydata(filepath)
    if gui.helmet_stl_actor_left is not None:
        try:
            gui.ren_left.RemoveActor(gui.helmet_stl_actor_left)
        except Exception:
            pass
    gui.helmet_stl_actor_left = actor_from_polydata(
        gui.helmet_stl_raw, color=(0.7, 0.9, 1.0), opacity=0.8
    )
    gui.ren_left.AddActor(gui.helmet_stl_actor_left)
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info(f"Loaded helmet STL: {os.path.basename(filepath)}")


def load_sensor_csv(gui):
    """Load sensor positions + axes from CSV, display on LEFT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Load sensor CSV", "", "CSV Files (*.csv)"
    )
    if not filepath:
        return
    df = pd.read_csv(filepath)
    cols = [
        "X",
        "Y",
        "Z",
        "x_i",
        "x_j",
        "x_k",
        "y_i",
        "y_j",
        "y_k",
        "z_i",
        "z_j",
        "z_k",
    ]
    for column in cols:
        if column not in df.columns:
            gui._update_info(f"CSV missing column: {column}")
            return
    gui.sensor_df = df.copy()
    gui.sensor_csv_path = filepath
    gui.sensor_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    x = df[["x_i", "x_j", "x_k"]].to_numpy(dtype=float)
    y = df[["y_i", "y_j", "y_k"]].to_numpy(dtype=float)
    z = df[["z_i", "z_j", "z_k"]].to_numpy(dtype=float)
    gui.sensor_axes = np.stack([x, y, z], axis=1)
    # display original sensors on LEFT (on top of STL)
    for actor in gui.sensor_actors_left:
        try:
            gui.ren_left.RemoveActor(actor)
        except Exception:
            pass
    gui.sensor_actors_left = []
    for pnt in gui.sensor_positions:
        actor = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
        gui.ren_left.AddActor(actor)
        gui.sensor_actors_left.append(actor)
    # show vectors on left if toggled
    gui._draw_sensor_vectors_left()
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info(
        f"Loaded sensor CSV: {os.path.basename(filepath)} ({len(gui.sensor_positions)} sensors)."
    )


def load_mri_mesh(gui):
    """Load MRI mesh (e.g. scalp) on the LEFT, auto-detect unit scale."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Load MRI mesh", "", "PLY/STL Files (*.ply *.stl)"
    )
    if not filepath:
        return

    gui.mri_polydata = load_mesh_to_polydata(filepath)

    # -------------------------------------------------
    # AUTO-DETECT UNIT SCALE
    # -------------------------------------------------
    bounds = gui.mri_polydata.GetBounds()
    max_abs = max(abs(b) for b in bounds)

    print("MRI bounds before scaling:", bounds)

    # If values look like meters (~0.1), scale to mm
    if max_abs < 1.0:
        print("MRI appears to be in meters → scaling to mm")

        scale_tf = vtk.vtkTransform()
        scale_tf.Scale(1000.0, 1000.0, 1000.0)

        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetTransform(scale_tf)
        tf_filter.SetInputData(gui.mri_polydata)
        tf_filter.Update()

        gui.mri_polydata = tf_filter.GetOutput()

        bounds_after = gui.mri_polydata.GetBounds()
        print("MRI bounds after scaling:", bounds_after)

    else:
        print("MRI appears already in mm")

    # -------------------------------------------------
    # Display on left
    # -------------------------------------------------
    actor = actor_from_polydata(gui.mri_polydata, color=(0.85, 0.85, 0.95), opacity=0.7)

    gui.ren_left.RemoveAllViewProps()
    gui.ren_left.AddActor(actor)

    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()

    gui._update_info("Loaded MRI mesh (left).")


def load_mri_fiducials(gui):
    """Load MRI fiducials from FIF, display on LEFT."""
    path = QFileDialog.getOpenFileName(
        gui, "Load MRI Fiducials (.fif)", "", "FIF files (*.fif)"
    )[0]

    if not path:
        return

    fid, coord_frame = read_fiducials(path)

    gui.mri_fiducials = {}

    for f in fid:
        ident = f["ident"]
        coord_mm = f["r"] * 1000.0  # meters → mm

        if ident == FIFF.FIFFV_POINT_NASION:
            gui.mri_fiducials["NAS"] = coord_mm
        elif ident == FIFF.FIFFV_POINT_LPA:
            gui.mri_fiducials["LPA"] = coord_mm
        elif ident == FIFF.FIFFV_POINT_RPA:
            gui.mri_fiducials["RPA"] = coord_mm

    print("Loaded MRI fiducials:", gui.mri_fiducials)

    if len(gui.mri_fiducials) != 3:
        gui._update_info(f"Incomplete MRI fiducials: {list(gui.mri_fiducials.keys())}")
        return
    gui._draw_mri_fiducials()

    gui._update_info("Loaded MRI fiducials from FIF.")
