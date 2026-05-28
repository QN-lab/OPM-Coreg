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
        "Select master scan file (with participant in the helmet)",
        "",
        "PLY/STL Files (*.ply *.stl)",
    )
    if not filepath:
        return
    gui.master_scan_polydata = load_mesh_to_polydata(filepath)
    gui.master_scan_filename = os.path.basename(filepath)
    if gui.master_scan_actor is not None:
        try:
            gui.ren_target.RemoveActor(gui.master_scan_actor)
        except Exception:
            pass
    gui.master_scan_actor = actor_from_polydata(
        gui.master_scan_polydata, color=(0.85, 0.85, 0.85), opacity=1.0
    )
    gui.ren_target.AddActor(gui.master_scan_actor)
    gui.ren_target.ResetCamera()
    gui.vtk_target.GetRenderWindow().Render()
    gui.update_info(f"Loaded master scan: {gui.master_scan_filename}")


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
        "Select head scan file (source window)",
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
    gui.update_info(f"Loaded head scan: {os.path.basename(filepath)}")


def load_helmet_cad(gui):
    """Load helmet design file (STL) on the source window."""
    filepath, _ = QFileDialog.getOpenFileName(
        gui, "Load helmet design file (STL)", "", "STL/PLY Files (*.stl *.ply)"
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

    gui.update_info(f"MRI bounds before scaling: {bounds}")

    # If values look like meters (~0.1), scale to mm
    if max_abs < 1.0:
        gui.update_info("MRI appears to be in meters → scaling to mm")

        scale_tf = vtk.vtkTransform()
        scale_tf.Scale(1000.0, 1000.0, 1000.0)

        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetTransform(scale_tf)
        tf_filter.SetInputData(gui.mri_polydata)
        tf_filter.Update()

        gui.mri_polydata = tf_filter.GetOutput()

        bounds_after = gui.mri_polydata.GetBounds()
        gui.update_info(f"MRI bounds after scaling: {bounds_after}")

    else:
        gui.update_info("MRI appears already in mm")

    # -------------------------------------------------
    # Display on source window
    # -------------------------------------------------
    try:
        if gui.mri_actor_source_window is not None:
            gui.ren_source.RemoveActor(gui.mri_actor_source_window)
    except Exception:
        pass
    gui.mri_actor_source_window = actor_from_polydata(
        gui.mri_polydata, color=(0.85, 0.85, 0.95), opacity=0.7
    )

    gui.ren_source.RemoveAllViewProps()
    gui.ren_source.AddActor(gui.mri_actor_source_window)

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

    gui.update_info(f"Loaded MRI fiducials: {gui.mri_fiducials}")
    if len(gui.mri_fiducials) != 3:
        gui.update_info(f"Incomplete MRI fiducials: {list(gui.mri_fiducials.keys())}")
        return
    draw_mri_fiducials(gui)


def save_transformed_sensors_csv(gui):
    """Apply the given transform to the original sensor positions and axes, and save the transformed coordinates to a CSV file."""
    if gui.sensor_positions is None:
        return
    if gui.T_dev_to_scan is None:
        gui.update_info(
            "No device→scan transform available, cannot save transformed sensors."
        )
        return

    # Transforming sensor positions from device frame to master scan frame
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

    # Use master scan filename as base for naming, or default to "master_scan_missing" if not available
    master_scan_basename = os.path.splitext(
        gui.master_scan_filename or "master_scan_missing"
    )[0]

    # Create a participant subdirectory in results if it doesn't exist (root_dir/results/participant_name/)
    participant_result_dir = Path(gui.results_dir) / master_scan_basename
    participant_result_dir.mkdir(exist_ok=True)

    out_path = (
        participant_result_dir
        / f"{master_scan_basename}_sensors_to_scan_transformed.csv"
    )
    df_out.to_csv(out_path, index=False)
    gui.update_info(f"Saved transformed sensors CSV: {out_path}")


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
    """Save specified transforms to disk in mm units, using the master scan filename as a base for naming.

    This function is called after computing any transform, to ensure results are saved without requiring extra user action.

    Args:
        gui: The GUI object containing the transforms and state.
        trans_to_save: The name of the transform to save (e.g., "dev_to_scan"). If None, all available transforms are saved.
    """
    # Use master scan filename as base for naming, or default to "master_scan_missing" if not available
    master_scan_basename = os.path.splitext(
        gui.master_scan_filename or "master_scan_missing"
    )[0]

    # Create a participant subdirectory in results if it doesn't exist (root_dir/results/participant_name/)
    # NOTE : do not create a new subdirectory for every new run of the GUI, as multiple participants could be processed in one run.
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
                f"Saved {transform_name} transform. Translation components are in mm units."
            )

            if transform_name == "head_to_mri":
                save_head_to_mri_mne_fif(
                    transform, participant_result_dir / f"{master_scan_basename}"
                )
                gui.update_info(
                    "Saved Head→MRI transform (Directly usable by MNE-Python for forward modeling)."
                )

            if transform_name == "dev_to_head":
                save_dev_to_head_for_mne(
                    transform,
                    participant_result_dir / f"{master_scan_basename}_{transform_name}",
                )
                gui.update_info(
                    "Saved Device→Head transform (Directly usable by MNE-Python for forward modeling)."
                )


def save_marked_points_in_cache(gui, file_to_save=None):
    """Save marked points in all steps as cached files.

    Cache directory is specific to each participant, and is located at root_dir/results/participant_name/cache/.

    Args:
        gui: The GUI object containing the points and state.
        file_to_save: The name of the specific points to save (e.g., "sensor_alignment_points_from_source"). If None, all available points are saved.
    """
    # Use master scan filename as base for naming, or default to "master_scan_missing" if not available
    master_scan_basename = os.path.splitext(
        gui.master_scan_filename or "master_scan_missing"
    )[0]

    # Create a participant subdirectory in results if it doesn't exist (root_dir/results/participant_name/)
    participant_result_dir = Path(gui.results_dir) / master_scan_basename
    participant_result_dir.mkdir(exist_ok=True)

    # Create a cache subdirectory for intermediate files
    cache_dir = participant_result_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Create dictionary of files to save
    files_to_save = {
        "aux_alignment_points_from_source": gui.aux_alignment_points_from_source,
        "aux_alignment_points_from_target": gui.aux_alignment_points_from_target,
        "sensor_alignment_points_from_source": gui.sensor_alignment_points_from_source,
        "sensor_alignment_points_from_target": gui.sensor_alignment_points_from_target,
        "head_alignment_points_from_source": gui.head_alignment_points_from_source,
        "head_alignment_points_from_target": gui.head_alignment_points_from_target,
        "marked_fiducials_on_head_scan": gui.marked_fiducials_on_head_scan,
    }

    for name, points in files_to_save.items():
        if file_to_save is not None and name != file_to_save:
            continue

        # Save all points that are not None and have length > 0
        if points is not None and len(points) > 0:
            np.save(cache_dir / f"{name}.npy", points)
            gui.update_info(f"Saved {name} to cache: {cache_dir / f'{name}.npy'}")


def load_cached_files(gui, cache_folder_path=None):
    """Load marked points from cached files if they exist, and update the GUI state.

    Args:
        gui: The GUI object to update with loaded points.
        cache_folder_path: The path to the cache folder where intermediate files are saved (root_dir/results/participant_name/cache/).
    """
    if cache_folder_path is None:
        gui.update_info("No cache folder path provided, cannot load cached files.")
        return

    files_to_load = {
        "aux_alignment_points_from_source": "Aux device alignment points (source)",
        "aux_alignment_points_from_target": "Aux device alignment points (target)",
        "sensor_alignment_points_from_source": "Sensor alignment points (source)",
        "sensor_alignment_points_from_target": "Sensor alignment points (target)",
        "head_alignment_points_from_source": "Head alignment points (source)",
        "head_alignment_points_from_target": "Head alignment points (target)",
        "marked_fiducials_on_head_scan": "Marked fiducials on head scan",
    }

    for name, description in files_to_load.items():
        file_path = cache_folder_path / f"{name}.npy"
        if file_path.exists():
            points = np.load(
                file_path, allow_pickle=True
            ).tolist()  # Use allow_pickle for loading dicts (e.g., marked_fiducials_on_head_scan)
            setattr(gui, name, points)
            gui.update_info(f"Loaded cached {description} from {file_path}.")


def load_transforms(gui, results_foler_path=None):
    """Load computed transforms from cache if they exist, and update the GUI state.

    Args:
        gui: The GUI object to update with loaded transforms.
        results_foler_path: The path to the participant's results folder where transforms are saved.
    """
    if results_foler_path is None:
        gui.update_info("No results folder path provided, cannot load transforms.")
        return

    transforms_to_load = {
        "aux_dev_to_scan": "Aux device to scan transform",
        "dev_to_scan": "Device to scan transform",
        "head_to_scan": "Head to scan transform",
        "head_to_mri": "Head to MRI transform",
        "aux_to_device": "Aux device to device transform",
        "dev_to_head": "Device to head transform",
    }

    for name, description in transforms_to_load.items():
        # Find a file in the results folder that ends with "{name}_mm.npy"
        file_path = None
        for f in results_foler_path.glob(f"*{name}_mm.npy"):
            file_path = f
            break

        # If such a file exists, load the transform and set it in the GUI
        if file_path is not None and file_path.exists():
            transform = np.load(file_path)
            setattr(
                gui, f"T_{name}", transform
            )  # Set the transform in the GUI (e.g., T_dev_to_scan)
            gui.update_info(f"Loaded cached {description} from {file_path}.")


def load_all_saved_data(gui):
    """Load all saved data (marked points + computed transforms) from the results folder specified by the user."""
    # Ask user to select the participant's results folder (root_dir/results/participant_name/)
    folder_path = QFileDialog.getExistingDirectory(
        gui, "Select participant's results folder", str(gui.results_dir)
    )
    if not folder_path:
        gui.update_info("No folder selected, cancelled loading saved data.")
        return
    folder_path = Path(folder_path)

    # Load computed transforms
    load_transforms(gui, results_foler_path=folder_path)

    # Load cached files (marked points)
    load_cached_files(gui, cache_folder_path=folder_path / "cache")
