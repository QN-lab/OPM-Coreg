"""Helper functions to compute transforms based on GUI state and user inputs."""

import os
import numpy as np
from src.display_helpers import (
    overlay_head_transformed,
    overlay_step3_transformed,
    preview_head_on_mri,
)
from src.utils import compute_rigid_transform, save_transform


def compute_aux_dev_to_scan(gui):
    """Compute transform from auxiliary device (e.g. coil) to master scan using the coil pairs."""
    if not getattr(gui, "coil_pairs_src", None) or len(gui.coil_pairs_src) < 3:
        gui._update_info("Need ≥3 coil pairs.")
        return
    src = np.vstack(gui.coil_pairs_src)
    dst = np.vstack(gui.coil_pairs_dst)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.aux_dev_to_scan = T
    gui._update_info(
        f"Computed coil to master scan transform RMS={rms:.3f} mm. Saved transform."
    )
    # auto save
    base = os.path.splitext(gui.helmet_filename or "helmet")[0]
    save_transform(T, base + "_aux_dev_to_scan")


def compute_aux_to_device(gui):
    """Compute transform from auxiliary device to device (e.g. coil to helmet).

    Using the aux->scan transform and dev->scan transform."""
    if gui.aux_dev_to_scan is None or gui.T_dev_to_scan is None:
        gui._update_info("Need Step1 and Step3 transforms.")
        return

    T_scan_to_dev = np.linalg.inv(gui.T_dev_to_scan)
    T_aux_to_device = T_scan_to_dev @ gui.aux_dev_to_scan

    base = os.path.splitext(gui.helmet_filename or "helmet")[0]
    save_transform(T_aux_to_device, base + "_aux_to_device")

    gui._update_info("Saved aux→device transform.")


def compute_head_to_scan(gui):
    """Compute transform from head scan to master scan using the head pairs."""
    if len(gui.step2_pairs_head) < 3:
        gui._update_info("Need at least 3 Step2 pairs to compute transform.")
        return
    src = np.vstack(gui.step2_pairs_head)
    dst = np.vstack(gui.step2_pairs_scan)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_head_to_scan = T
    gui._update_info(
        f"Computed head_to_scan RMS={rms:.3f} mm. Overlaying transformed head scan onto RIGHT."
    )
    # overlay transformed head scan on right (light orange)
    overlay_head_transformed(gui, T)
    # save transform
    base = os.path.splitext(gui.helmet_filename or "helmet")[0]
    save_transform(T, base + "_head_to_scan")


def compute_sensors_transform(gui):
    """Compute transform from helmet to master scan using the helmet pairs, then apply to sensors."""
    if len(gui.ref_point_pairs_helmet_stl) < 3:
        gui._update_info("Need ≥3 Step3 pairs.")
        return
    src = np.vstack(gui.ref_point_pairs_helmet_stl)
    dst = np.vstack(gui.ref_point_pairs_scan)
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_dev_to_scan = T
    # -------------------------------------------------
    # Compute aux -> device if aux transform exists
    # -------------------------------------------------
    if gui.aux_dev_to_scan is not None:
        T_scan_to_dev = np.linalg.inv(gui.T_dev_to_scan)
        T_aux_to_device = T_scan_to_dev @ gui.aux_dev_to_scan

        base = os.path.splitext(gui.helmet_filename or "helmet")[0]
        save_transform(T_aux_to_device, base + "_aux_to_device")

        gui._update_info("Saved aux→device transform.")
    # ---------------------------------------------------
    # Compute device -> head if Step2 is available
    # ---------------------------------------------------
    if gui.T_head_to_scan is not None:
        T_scan_to_head = np.linalg.inv(gui.T_head_to_scan)
        T_dev_to_head = T_scan_to_head @ gui.T_dev_to_scan
        gui.T_dev_to_head = T_dev_to_head

        base = os.path.splitext(gui.helmet_filename or "helmet")[0]
        save_transform(T_dev_to_head, base + "_dev_to_head")

        gui._update_info("Computed device→head and saved dev_head_t.")
    else:
        gui._update_info(
            "Computed dev_to_scan, but head_to_scan not available — cannot compute dev_to_head yet."
        )

    gui._update_info(
        f"Computed helmet→scan RMS={rms:.3f} mm. Previewing transformed STL & sensors on RIGHT."
    )
    # overlay transformed STL + transformed sensors
    overlay_step3_transformed(gui, T)
    # save transform automatically
    base = os.path.splitext(gui.helmet_filename or "helmet")[0]
    save_transform(T, base + "_T_dev_to_scan")
    # save transformed sensors csv
    gui._save_transformed_sensors_csv(T, base)


def compute_head_to_mri_initial(gui):
    """Compute initial head to MRI transform using the fiducials from step2 (head scan) and step4 (MRI)."""
    required = ["NAS", "LPA", "RPA"]

    if not all(fid in gui.head_fiducials for fid in required):
        gui._update_info("Head fiducials missing (Step2).")
        return

    if not all(fid in gui.mri_fiducials for fid in required):
        gui._update_info("MRI fiducials missing.")
        return

    src = np.array(
        [
            gui.head_fiducials["NAS"],
            gui.head_fiducials["LPA"],
            gui.head_fiducials["RPA"],
        ]
    )

    dst = np.array(
        [
            gui.mri_fiducials["NAS"],
            gui.mri_fiducials["LPA"],
            gui.mri_fiducials["RPA"],
        ]
    )

    T, rms, errs = compute_rigid_transform(src, dst)

    gui.T_head_to_mri = T
    gui.head_to_mri_rms = rms
    gui.head_to_mri_errors = errs

    print("Head→MRI RMS:", rms)
    print("Errors per fiducial:", errs)

    gui._update_info(
        f"Head→MRI RMS = {rms:.2f} mm | "
        f"NAS={errs[0]:.2f}, "
        f"LPA={errs[1]:.2f}, "
        f"RPA={errs[2]:.2f}"
    )

    preview_head_on_mri(gui, T)


def compute_helmet_to_mri(gui):
    """Compute transform from helmet to MRI using the head scan as an intermediate, then apply to coil/device."""
    # requires mri_fiducials and head scan fiducials transformed to master scan (from step2)
    if not gui.mri_fiducials or not gui.head_fiducials or gui.T_head_to_scan is None:
        gui._update_info(
            "Need MRI fiducials (step4) and head_to_scan transform (step2)."
        )
        return
    # map head scan fiducials to helmet using T_head_to_scan
    R_oh = gui.T_head_to_scan[:3, :3]
    t_oh = gui.T_head_to_scan[:3, 3]
    helmet_fids = {}
    for lbl, p in gui.head_fiducials.items():
        helmet_fids[lbl] = R_oh @ p + t_oh
    # order NAS,LPA,RPA
    src = np.vstack([helmet_fids[fid] for fid in ["NAS", "LPA", "RPA"]])
    dst = np.vstack([gui.mri_fiducials[fid] for fid in ["NAS", "LPA", "RPA"]])
    T, rms, errs = compute_rigid_transform(src, dst)
    gui.T_scan_to_mri = T
    gui._update_info(
        f"Computed scan_to_mri RMS={rms:.3f} mm. Applying to coil/device/head."
    )
    # apply final transforms and save
    gui._apply_final_transforms(T)
