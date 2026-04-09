"""Helper functions to load files based on GUI actions, and update the GUI state accordingly."""

import csv
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import vtk
from mne.io import read_fiducials
from mne.io.constants import FIFF
from mne.transforms import Transform
from PyQt5.QtWidgets import QFileDialog

from src.const import PAIR_COLORS
from src.display_helpers import (
    draw_mri_fiducials,
    draw_sensor_vectors_source_window,
)
from src.utils import (
    actor_from_polydata,
    create_text_actor,
    load_mesh_to_polydata,
    save_transform,
    sphere_actor,
)


def load_master_scan(gui):
    """Load master scan (with participant in the helmet) on the target window."""
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
            gui.ren_target.RemoveActor(gui.helmet_actor)
        except Exception:
            pass
    gui.helmet_actor = actor_from_polydata(
        gui.helmet_polydata, color=(0.85, 0.85, 0.85), opacity=1.0
    )
    gui.ren_target.AddActor(gui.helmet_actor)
    gui.ren_target.ResetCamera()
    gui.vtk_target.GetRenderWindow().Render()
    gui.update_info(f"Loaded master scan: {gui.helmet_filename}")


def load_aux_dev_stl(gui):
    """Load auxiliary device STL on the source window."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Select auxiliary device STL", "", "STL Files (*.stl)"
    )
    if not filepath:
        return
    gui.aux_dev_polydata = load_mesh_to_polydata(filepath)
    if gui.aux_dev_actor_source_window is not None:
        try:
            gui.ren_source.RemoveActor(gui.aux_dev_actor_source_window)
        except Exception:
            pass
    gui.aux_dev_actor_source_window = actor_from_polydata(
        gui.aux_dev_polydata, color=(0.7, 1, 0.2), opacity=1.0
    )
    gui.ren_source.AddActor(gui.aux_dev_actor_source_window)
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()
    gui.update_info("Loaded auxiliary device STL (source window).")


def load_aux_dev_landmarks(gui):
    """Load auxiliary device landmarks CSV on the source window."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Select auxiliary device landmarks CSV", "", "CSV Files (*.csv)"
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
    gui.aux_dev_landmarks = lm
    # draw landmarks on source window
    for actor in getattr(gui, "aux_dev_landmark_actors_source_window", []):
        try:
            gui.ren_source.RemoveActor(actor)
        except Exception:
            pass
    gui.aux_dev_landmark_actors_source_window = []
    for i, (lab, point_coords) in enumerate(gui.aux_dev_landmarks.items()):
        selected_color = PAIR_COLORS[i % len(PAIR_COLORS)]
        actor = sphere_actor(point_coords, color=selected_color, radius=3.5)
        gui.ren_source.AddActor(actor)
        gui.aux_dev_landmark_actors_source_window.append(actor)
    gui.vtk_source.GetRenderWindow().Render()
    gui.update_info(f"Loaded {len(gui.aux_dev_landmarks)} auxiliary device landmarks.")


def load_head_scan(gui):
    """Load head scan (without helmet) on the source window."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui,
        "Select OUTSIDE head scan (source window)",
        "",
        "PLY/STL Files (*.ply *.stl)",
    )
    if not filepath:
        return
    gui.head_polydata = load_mesh_to_polydata(filepath)
    if gui.head_actor is not None:
        try:
            gui.ren_source.RemoveActor(gui.head_actor)
        except Exception:
            pass
    gui.head_actor = actor_from_polydata(
        gui.head_polydata, color=(0.8, 0.8, 0.95), opacity=1.0
    )
    gui.ren_source.AddActor(gui.head_actor)
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()
    gui.update_info(f"Loaded outside head scan: {os.path.basename(filepath)}")


def load_helmet_cad(gui):
    """Load helmet CAD (STL) on the source window."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Load helmet STL", "", "STL/PLY Files (*.stl *.ply)"
    )
    if not filepath:
        return
    gui.helmet_stl_raw = load_mesh_to_polydata(filepath)
    if gui.helmet_stl_actor_source is not None:
        try:
            gui.ren_source.RemoveActor(gui.helmet_stl_actor_source)
        except Exception:
            pass
    gui.helmet_stl_actor_source = actor_from_polydata(
        gui.helmet_stl_raw, color=(0.7, 0.9, 1.0), opacity=0.8
    )
    gui.ren_source.AddActor(gui.helmet_stl_actor_source)
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()
    gui.update_info(f"Loaded helmet STL: {os.path.basename(filepath)}")


def load_sensor_csv(gui):
    """Load sensor positions + axes from CSV, display on source window."""
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
            gui.update_info(f"CSV missing column: {column}")
            return
    gui.sensor_df = df.copy()
    gui.sensor_csv_path = filepath
    gui.sensor_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    x = df[["x_i", "x_j", "x_k"]].to_numpy(dtype=float)
    y = df[["y_i", "y_j", "y_k"]].to_numpy(dtype=float)
    z = df[["z_i", "z_j", "z_k"]].to_numpy(dtype=float)
    gui.sensor_axes = np.stack([x, y, z], axis=1)
    # display original sensors on source window (on top of STL)
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
    # show vectors on source window if toggled
    draw_sensor_vectors_source_window(gui)
    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()
    gui.update_info(
        f"Loaded sensor CSV: {os.path.basename(filepath)} ({len(gui.sensor_positions)} sensors)."
    )


def load_helmet_landmark_csv(gui):
    """Load helmet landmark points CSV, display on source window."""
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
            gui.update_info(f"CSV missing column: {column}")
            return
    gui.helmet_landamrk_csv_path = filepath
    gui.helmet_landmark_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)

    # Get labels if they exist, otherwise use index
    has_labels = "Label" in df.columns
    labels = (
        df["Label"].tolist() if has_labels else [f"L{i + 1}" for i in range(len(df))]
    )

    # Clear existing actors
    for actor in gui.helmet_landmark_actors_source_window:
        try:
            gui.ren_source.RemoveActor(actor)
        except Exception:
            pass
    gui.helmet_landmark_actors_source_window = []

    # Clear existing text actors
    for actor in gui.helmet_landmark_text_actors_source_window:
        try:
            gui.ren_source.RemoveActor(actor)
        except Exception:
            pass
    gui.helmet_landmark_text_actors_source_window = []

    # Add sphere and text for each landmark
    for pnt, label in zip(gui.helmet_landmark_positions, labels):
        # Sphere actor
        sphere = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
        gui.ren_source.AddActor(sphere)
        gui.helmet_landmark_actors_source_window.append(sphere)

        # Text label
        text_actor = create_text_actor(
            label, pnt, color=(1, 1, 0), font_size=7, offset=(2, 2, 2)
        )
        gui.ren_source.AddActor(text_actor)
        gui.helmet_landmark_text_actors_source_window.append(text_actor)

    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()
    gui.update_info(f"Loaded helmet landmark CSV: {os.path.basename(filepath)}.")


def load_mri_mesh(gui):
    """Load MRI mesh (e.g. scalp) on the source window, auto-detect unit scale."""
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
    # Display on source window
    # -------------------------------------------------
    actor = actor_from_polydata(gui.mri_polydata, color=(0.85, 0.85, 0.95), opacity=0.7)

    gui.ren_source.RemoveAllViewProps()
    gui.ren_source.AddActor(actor)

    gui.ren_source.ResetCamera()
    gui.vtk_source.GetRenderWindow().Render()

    gui.update_info("Loaded MRI mesh (source window).")


def load_mri_fiducials(gui):
    """Load MRI fiducials from FIF, display on source window."""
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
        gui.update_info(f"Incomplete MRI fiducials: {list(gui.mri_fiducials.keys())}")
        return
    draw_mri_fiducials(gui)

    gui.update_info("Loaded MRI fiducials from FIF.")


def save_transformed_sensors_csv(gui):
    """Apply the given transform to the original sensor positions and axes, and save the transformed coordinates to a CSV file."""
    if gui.sensor_positions is None:
        return
    if gui.T_dev_to_scan is None:
        gui.update_info(
            "No device→scan transform available, cannot save transformed sensors."
        )
        return

    # Transforming sensor positions from device space to master scan space
    rotation = gui.T_dev_to_scan[:3, :3]
    translation = gui.T_dev_to_scan[:3, 3]
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

    # Use helmet filename as base for naming, or default to "master_scan_missing" if not available
    master_scan_basename = os.path.splitext(
        gui.helmet_filename or "master_scan_missing"
    )[0]

    # Create a participant subdirectory in results if it doesn't exist (root_dir/results/participant_name/)
    participant_result_dir = Path(gui.results_dir) / master_scan_basename
    participant_result_dir.mkdir(exist_ok=True)

    out_path = (
        participant_result_dir
        / f"{master_scan_basename}_sensors_to_scan_transformed.csv"
    )
    df_out.to_csv(out_path, index=False)
    gui.update_info(f"Saved transformed sensors CSV: {out_path}", info_update=False)


def save_head_to_mri_mne_fif(head_to_mri_transform_mm, out_base):
    """Save the head-to-MRI transform in MNE FIF format, converting from mm (internal) to m (FIF) units.

    The output file will be named {out_base}-trans.fif.
    This allows compatibility with MNE-Python for downstream processing such as source localization.

    Args:
        head_to_mri_transform_mm: 4x4 numpy array representing the head-to-MRI transform in mm units.
        out_base (Path): Base path (without extension) for the output FIF file.
    """
    # Convert from mm (internal) to m (MNE FIF)
    head_to_mri_transform_m = head_to_mri_transform_mm.copy()
    head_to_mri_transform_m[:3, 3] /= 1000.0

    # Create MNE Transform object and save to FIF
    trans = Transform("head", "mri", head_to_mri_transform_m)
    mne.write_trans(str(out_base) + "-trans.fif", trans, overwrite=True)


def save_dev_to_head_for_mne(dev_to_head_transform_mm, out_base):
    """Save the device-to-head transform in MNE compatible format, converting from mm (internal) to m (FIF) units.
    
    Args:
        dev_to_head_transform_mm: 4x4 numpy array representing the device-to-head transform in mm units.
        out_base (Path): Base path (without extension) for the output FIF file.
    """
    # Convert from mm (internal) to m (MNE FIF)
    dev_to_head_transform_m = dev_to_head_transform_mm.copy()
    dev_to_head_transform_m[:3, 3] /= 1000.0

    # Save the device_to_head transform without "mm" to signify it is directly usable in MNE
    save_transform(dev_to_head_transform_m, out_base)


def save_generated_transforms(gui, trans_to_save=None):
    """Save specified transforms to disk in mm units, using the helmet filename as a base for naming.

    This function is called after computing any transform, to ensure results are saved without requiring extra user action.
    The info box is not updated for each individual transform save, but the logfile is updated normally.
    
    Args:
        gui: The GUI object containing the transforms and state.
        trans_to_save: The name of the transform to save (e.g., "dev_to_scan"). If None, all available transforms are saved.
    """
    # Use helmet filename as base for naming, or default to "master_scan_missing" if not available
    master_scan_basename = os.path.splitext(
        gui.helmet_filename or "master_scan_missing"
    )[0]

    # Create a participant subdirectory in results if it doesn't exist (root_dir/results/participant_name/)
    participant_result_dir = Path(gui.results_dir) / master_scan_basename
    participant_result_dir.mkdir(exist_ok=True)

    # Create a dictionary of available transforms for easy iteration
    available_transforms = {
        "aux_dev_to_scan": gui.T_aux_dev_to_scan,
        "dev_to_scan": gui.T_dev_to_scan,
        "head_to_scan": gui.T_head_to_scan,
        "head_to_mri": gui.T_head_to_mri,
        "aux_to_device": gui.T_aux_to_device,
        "dev_to_head": gui.T_dev_to_head,
    }

    for transform_name, transform in available_transforms.items():
        # If a specific transform is specified to save, skip others
        if trans_to_save is not None and trans_to_save != transform_name:
            continue

        # Only save if the transform is available (not None)
        if transform is not None:
            save_transform(
                transform,
                participant_result_dir / f"{master_scan_basename}_{transform_name}_mm",
            )
            gui.update_info(
                f"Saved {transform_name} transform. Translation components are in mm units.", info_update=False
            )

            if transform_name == "head_to_mri":
                save_head_to_mri_mne_fif(
                    transform, participant_result_dir / f"{master_scan_basename}"
                )
                gui.update_info(
                    "Saved Head→MRI transform (Directly usable by MNE-Python for forward modeling).",
                    info_update=False,
                )

            if transform_name == "dev_to_head":
                save_dev_to_head_for_mne(
                    transform, participant_result_dir / f"{master_scan_basename}_{transform_name}"
                )
                gui.update_info(
                    "Saved Device→Head transform (Directly usable by MNE-Python for forward modeling).",
                    info_update=False,
                )
