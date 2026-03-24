"""Helper functions to load files based on GUI actions, and update the GUI state accordingly."""

import csv
import mne
from mne.transforms import Transform
import os

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog

from src.const import PAIR_COLORS
from src.display_helpers import (
    draw_mri_fiducials,
    draw_sensor_vectors_left,
)
from src.utils import (
    actor_from_polydata,
    load_mesh_to_polydata,
    sphere_actor,
    save_transform,
    create_text_actor,
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
    draw_sensor_vectors_left(gui)
    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info(
        f"Loaded sensor CSV: {os.path.basename(filepath)} ({len(gui.sensor_positions)} sensors)."
    )


def load_helmet_landmark_csv(gui):
    """Load helmet landmark points CSV, display on LEFT."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Load helmet landmark CSV", "", "CSV Files (*.csv)"
    )
    if not filepath:
        return
    df = pd.read_csv(filepath)
    cols = [
        "X",
        "Y",
        "Z",
    ]
    for column in cols:
        if column not in df.columns:
            gui._update_info(f"CSV missing column: {column}")
            return
    gui.helmet_landamrk_csv_path = filepath
    gui.helmet_landmark_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)

    # Get labels if they exist, otherwise use index
    has_labels = "Label" in df.columns
    labels = df["Label"].tolist() if has_labels else [f"L{i+1}" for i in range(len(df))]

    # Clear existing actors
    for actor in gui.helmet_landmark_actors_left:
        try:
            gui.ren_left.RemoveActor(actor)
        except Exception:
            pass
    gui.helmet_landmark_actors_left = []
    
    # Clear existing text actors
    for actor in gui.helmet_landmark_text_actors_left:
        try:
            gui.ren_left.RemoveActor(actor)
        except Exception:
            pass
    gui.helmet_landmark_text_actors_left = []

    # Add sphere and text for each landmark
    for pnt, label in zip(gui.helmet_landmark_positions, labels):
        # Sphere actor
        sphere = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
        gui.ren_left.AddActor(sphere)
        gui.helmet_landmark_actors_left.append(sphere)
        
        # Text label
        text_actor = create_text_actor(label, pnt, color=(1, 1, 0), font_size=7, offset=(2, 2, 2))
        gui.ren_left.AddActor(text_actor)
        gui.helmet_landmark_text_actors_left.append(text_actor)

    gui.ren_left.ResetCamera()
    gui.vtk_left.GetRenderWindow().Render()
    gui._update_info(
        f"Loaded helmet landmark CSV: {os.path.basename(filepath)}."
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
    draw_mri_fiducials(gui)

    gui._update_info("Loaded MRI fiducials from FIF.")


def save_transformed_sensors_csv(gui, T, base):
    """Apply the given transform to the original sensor positions and axes, and save the transformed coordinates to a CSV file."""
    if gui.sensor_positions is None:
        return
    rotation = T[:3, :3]
    translation = T[:3, 3]
    pts_t = (rotation @ gui.sensor_positions.T).T + translation
    df_out = gui.sensor_df.copy()
    df_out[["X", "Y", "Z"]] = pts_t
    if gui.sensor_axes is not None:
        axes_t = np.empty_like(gui.sensor_axes)
        for i in range(3):
            axes_t[:, i, :] = (rotation @ gui.sensor_axes[:, i, :].T).T
        df_out["x_i"] = axes_t[:, 0, 0]
        df_out["x_j"] = axes_t[:, 0, 1]
        df_out["x_k"] = axes_t[:, 0, 2]
        df_out["y_i"] = axes_t[:, 1, 0]
        df_out["y_j"] = axes_t[:, 1, 1]
        df_out["y_k"] = axes_t[:, 1, 2]
        df_out["z_i"] = axes_t[:, 2, 0]
        df_out["z_j"] = axes_t[:, 2, 1]
        df_out["z_k"] = axes_t[:, 2, 2]
    out_dir = (
        os.path.dirname(gui.sensor_csv_path)
        if getattr(gui, "sensor_csv_path", None)
        else os.getcwd()
    )
    out_path = os.path.join(out_dir, base + "_sensors_to_scan_transformed.csv")
    df_out.to_csv(out_path, index=False)
    gui._update_info(f"Saved transformed sensors CSV: {out_path}")


def save_head_to_mri(gui):
    """Save the computed head to MRI transform to disk.

    Transforms are saved in custom format and MNE's .fif format, converting units as needed for compatibility."""
    if gui.T_head_to_mri is None:
        return

    base = os.path.splitext(gui.helmet_filename or "helmet")[0]

    # --- Convert mm → meters for MNE ---
    T_m = gui.T_head_to_mri.copy()
    T_m[:3, 3] /= 1000.0

    save_transform(gui.T_head_to_mri, base + "_head_to_mri_mm")

    trans = Transform("head", "mri", T_m)

    mne.write_trans(base + "-trans.fif", trans, overwrite=True)

    gui._update_info("Saved Head→MRI transform (mm internally, m in trans.fif).")


def auto_save_transform(gui):
    """Automatically save any available transforms to disk in mm units, using the helmet filename as a base for naming."""
    base = os.path.splitext(gui.helmet_filename or "helmet")[0]

    if gui.aux_dev_to_scan is not None:
        save_transform(gui.aux_dev_to_scan, base + "_aux_dev_to_scan_mm")

    if gui.T_head_to_scan is not None:
        save_transform(gui.T_head_to_scan, base + "_head_to_scan_mm")

    if gui.T_dev_to_scan is not None:
        save_transform(gui.T_dev_to_scan, base + "_dev_to_scan_mm")

    if gui.T_head_to_mri is not None:
        save_transform(gui.T_head_to_mri, base + "_head_to_mri_mm")

    gui._update_info("Auto-saved available transforms (mm).")
