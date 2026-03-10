#!/usr/bin/env python3
# tms_opm_all_steps_gui_fixed.py
#
# All-in-one GUI (fixed):
#  - Step1: coil -> master scan (select landmarks on master scan based on landmark locations in csv)
#  - Step2: scan of the head "outside" -> master scan (fiducials + pairwise)
#  - Step3: sensors & helmet STL -> master_scan (pairwise)  <- integrated from your working script
#  - Step4: helmet -> MRI (fiducial) ### check this

# TODO : move main GUI class to a separate file, and have a main.py that just launches the GUI (and can be extended to support CLI in the future)

import csv
import os
import sys

import mne
import numpy as np
import pandas as pd
import vtk
from mne.io import read_fiducials
from mne.io.constants import FIFF
from mne.transforms import Transform
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from src.const import FIDUCIAL_COLORS, PAIR_COLORS
from src.utils import (
    actor_from_polydata,
    arrow_actor,
    compute_rigid_transform,
    load_mesh_to_polydata,
    save_transform,
    sphere_actor,
    vtk_matrix_from_numpy,
)


# ---------------- Main GUI ----------------
class AllStepsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TMS-OPM Multi-step Registration (fixed)")
        self.resize(1800, 1000)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Renderers / interactors
        self.vtk_left = QVTKRenderWindowInteractor(self)
        self.ren_left = vtk.vtkRenderer()
        self.vtk_left.GetRenderWindow().AddRenderer(self.ren_left)
        self.left_iren = self.vtk_left.GetRenderWindow().GetInteractor()
        style_l = vtk.vtkInteractorStyleTrackballCamera()
        style_l.SetMotionFactor(4.0)
        self.left_iren.SetInteractorStyle(style_l)

        self.vtk_right = QVTKRenderWindowInteractor(self)
        self.ren_right = vtk.vtkRenderer()
        self.vtk_right.GetRenderWindow().AddRenderer(self.ren_right)
        self.right_iren = self.vtk_right.GetRenderWindow().GetInteractor()
        style_r = vtk.vtkInteractorStyleTrackballCamera()
        style_r.SetMotionFactor(4.0)
        self.right_iren.SetInteractorStyle(style_r)

        # Keep track of added observer IDs so we can remove them safely
        self._observer_ids = []

        # ===== shared state (per-step) =====
        self.mode = None
        self.picking_state = {}

        # Step1 coil to master scan
        self.helmet_polydata = None
        self.helmet_actor = None
        self.helmet_filename = None
        self.coil_polydata = None
        self.coil_actor_left = None
        self.coil_landmarks = {}
        self.coil_landmark_actors_left = []
        self.coil_expected_labels = []
        self.coil_pairs_src = []
        self.coil_pairs_dst = []
        self.coil_pick_index = 0
        self.aux_dev_to_scan = None
        self.coil_actor_transformed = None

        # Step2 head to master scan
        self.head_polydata = None
        self.head_actor = None
        self.head_fiducials = {}  # NAS,LPA,RPA
        self.head_fiducial_actors = {}
        self.step2_pairs_head = []
        self.step2_pairs_scan = []
        self.step2_pair_actors_left = []
        self.step2_pair_actors_right = []
        self.T_head_to_scan = None

        # Step3 helmet STL/sensor positions  to master scan (to get the device to scan transform)
        self.helmet_stl_raw = None
        self.helmet_stl_actor_left = None
        self.helmet_stl_actor_overlay = None
        self.sensor_df = None
        self.sensor_positions = None
        self.sensor_axes = None
        self.sensor_actors_left = []
        self.transformed_sensor_actors = []
        self.transformed_vector_actors = []
        self.ref_point_pairs_helmet_stl = []
        self.ref_point_pairs_scan = []
        self.T_dev_to_scan = None
        self.T_dev_to_head = None

        self.show_vectors = False
        self.show_helmet_stl = True
        self.sensor_csv_path = None
        # Step3 pairing actors (MUST exist before use)
        self.pair_actors_left = []
        self.pair_actors_right = []

        # Step4 (MRI)
        self.mri_polydata = None
        self.mri_fiducials = {}
        self.mri_fiducial_actors = {}
        self.T_head_to_mri = None
        self.mri_actor_right = None

        # Build UI
        self._build_ui()
        # initialize
        self.left_iren.Initialize()
        self.right_iren.Initialize()

    # ---------- UI ----------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        # two render windows
        view_layout = QHBoxLayout()
        view_layout.addWidget(self.vtk_left, 1)
        view_layout.addWidget(self.vtk_right, 1)
        main_layout.addLayout(view_layout, stretch=8)

        # bottom controls
        ctrl_layout = QHBoxLayout()
        # steps list
        steps_box = QGroupBox("Steps")
        steps_v = QVBoxLayout()
        for lbl, m in [
            ("Step 1: Coil to scan", "step1"),
            ("Step 2: Head to scan", "step2"),
            ("Step 3: Sensors to scan", "step3"),
            ("Step 4: Head to MRI", "step4"),
        ]:
            b = QPushButton(lbl)
            b.clicked.connect(lambda _, mm=m: self.set_mode(mm))
            steps_v.addWidget(b)
        steps_box.setLayout(steps_v)
        ctrl_layout.addWidget(steps_box)

        # dynamic step controls
        self.step_ctrl_box = QGroupBox("Step Controls")
        self.step_ctrl_layout = QGridLayout()
        self.step_ctrl_box.setLayout(self.step_ctrl_layout)
        ctrl_layout.addWidget(self.step_ctrl_box, stretch=1)

        # info box
        info_box = QGroupBox("Info / Save")
        info_v = QVBoxLayout()
        self.info_label = QLabel("Select layers to see:")
        self.info_label.setWordWrap(True)
        info_v.addWidget(self.info_label)

        self.cb_show_aux_dev = QCheckBox("TMS coil (Step 1)")
        self.cb_show_aux_dev.setChecked(True)
        self.cb_show_aux_dev.stateChanged.connect(self._toggle_aux_dev_overlay)
        info_v.addWidget(self.cb_show_aux_dev)

        self.cb_show_head = QCheckBox("Head scan (Step 2)")
        self.cb_show_head.setChecked(True)
        self.cb_show_head.stateChanged.connect(self._toggle_head_overlay)
        info_v.addWidget(self.cb_show_head)
        self.cb_stl = QCheckBox("Sensors (Step 3)")
        self.cb_stl.setChecked(True)
        self.cb_stl.stateChanged.connect(self._toggle_helmet_overlay)
        info_v.addWidget(self.cb_stl)
        self.cb_vec = QCheckBox("Unit vectors (Step 3)")
        self.cb_vec.stateChanged.connect(self._toggle_vectors)
        info_v.addWidget(self.cb_vec)

        self.cb_show_mri = QCheckBox("MRI (Step 4)")
        self.cb_show_mri.setChecked(True)
        self.cb_show_mri.stateChanged.connect(self._toggle_mri_overlay)
        info_v.addWidget(self.cb_show_mri)

        self.btn_save_all = QPushButton("Auto-save available transforms")
        self.btn_save_all.clicked.connect(self.auto_save_transform)
        info_v.addWidget(self.btn_save_all)
        info_box.setLayout(info_v)
        ctrl_layout.addWidget(info_box)

        main_layout.addLayout(ctrl_layout, stretch=2)

        bottom = QFrame()
        bottom.setStyleSheet("background-color:#222;")
        bottom_layout = QVBoxLayout(bottom)
        self.bottom_instr = QLabel(
            "Instructions: use SHIFT+click to pick points; rotate normally without SHIFT."
        )
        self.bottom_instr.setWordWrap(True)
        bottom_layout.addWidget(self.bottom_instr)
        main_layout.addWidget(bottom, stretch=1)

    def _update_info(self, txt):
        print(txt)
        self.info_label.setText(txt)

    # ---------- observer helpers ----------
    def _add_left_observer(self, evt, func):
        style = self.left_iren.GetInteractorStyle()
        oid = style.AddObserver(evt, func)
        self._observer_ids.append(("left_style", oid))
        return oid

    def _add_right_observer(self, evt, func):
        style = self.right_iren.GetInteractorStyle()
        oid = style.AddObserver(evt, func)
        self._observer_ids.append(("right_style", oid))
        return oid

    def _remove_all_observers(self):
        for side, oid in list(self._observer_ids):
            try:
                if side == "left_style":
                    self.left_iren.GetInteractorStyle().RemoveObserver(oid)
                elif side == "right_style":
                    self.right_iren.GetInteractorStyle().RemoveObserver(oid)
            except Exception:
                pass
            self._observer_ids.remove((side, oid))

    def _clear_left_renderer(self):
        self.ren_left.RemoveAllViewProps()
        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()

    def _clear_right_renderer_keep_scan(self):
        self.ren_right.RemoveAllViewProps()
        if self.helmet_actor is not None:
            self.ren_right.AddActor(self.helmet_actor)
        self.ren_right.ResetCameraClippingRange()
        self.vtk_right.GetRenderWindow().Render()

    # ---------- mode switching ----------
    def set_mode(self, m):

        # 1. Remove old observers
        self._remove_all_observers()

        # 2. Create fresh styles WITH motion factor
        style_l = vtk.vtkInteractorStyleTrackballCamera()
        style_l.SetMotionFactor(4.0)

        style_r = vtk.vtkInteractorStyleTrackballCamera()
        style_r.SetMotionFactor(4.0)

        self.left_iren.SetInteractorStyle(style_l)
        self.right_iren.SetInteractorStyle(style_r)

        self.mode = m
        self.picking_state = {}

        # 3. Reset renderers
        self._clear_left_renderer()
        self._clear_right_renderer_keep_scan()

        # 4. Step setup
        if m == "step1":
            self._populate_step1_controls()

        elif m == "step2":
            self._populate_step2_controls()
            self._add_left_observer("LeftButtonPressEvent", self._left_click_step2)
            self._add_right_observer("LeftButtonPressEvent", self._right_click_step2)

        elif m == "step3":
            self._populate_step3_controls()
            self._add_left_observer("LeftButtonPressEvent", self._left_click_step3)
            self._add_right_observer("LeftButtonPressEvent", self._right_click_step3)

        elif m == "step4":
            self._populate_step4_controls()
            self._add_left_observer("LeftButtonPressEvent", self._left_click_step4)

        self._update_info(f"Mode: {m}")

    def _clear_step_controls(self):
        for i in reversed(range(self.step_ctrl_layout.count())):
            w = self.step_ctrl_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

    # ----------------- STEP 1 -------------------
    def _populate_step1_controls(self):
        self._clear_step_controls()
        r = 0
        b1 = QPushButton("Load 3D master scan (RIGHT)")
        b1.clicked.connect(self.load_master_scan)
        self.step_ctrl_layout.addWidget(b1, r, 0)
        r += 1
        b2 = QPushButton("Load coil STL (LEFT)")
        b2.clicked.connect(self.load_aux_dev_stl)
        self.step_ctrl_layout.addWidget(b2, r, 0)
        r += 1
        b3 = QPushButton("Load coil landmarks CSV")
        b3.clicked.connect(self.load_aux_dev_landmarks)
        self.step_ctrl_layout.addWidget(b3, r, 0)
        r += 1
        b4 = QPushButton("Start picking coil landmarks in master scan")
        b4.clicked.connect(self.start_aux_dev_landmark_pairs)
        self.step_ctrl_layout.addWidget(b4, r, 0)
        r += 1
        b5 = QPushButton("Undo last coil pair")
        b5.clicked.connect(self.undo_last_aux_dev_pair)
        self.step_ctrl_layout.addWidget(b5, r, 0)
        r += 1
        b6 = QPushButton("Compute coil to master scan transform")
        b6.clicked.connect(self.compute_aux_dev_to_scan)
        self.step_ctrl_layout.addWidget(b6, r, 0)
        r += 1
        b7 = QPushButton("Preview coil on master scan")
        b7.clicked.connect(self.merge_aux_dev_into_helmet)
        self.step_ctrl_layout.addWidget(b7, r, 0)
        r += 1
        b8 = QPushButton("Restart Step 1")
        b8.clicked.connect(self.restart_step1)
        self.step_ctrl_layout.addWidget(b8, r, 0)

    def restart_step1(self):
        try:
            if self.coil_actor_left is not None:
                self.ren_left.RemoveActor(self.coil_actor_left)
        except:
            pass
        for a in getattr(self, "coil_landmark_actors_left", []):
            try:
                self.ren_left.RemoveActor(a)
            except:
                pass
        try:
            if self.coil_actor_transformed is not None:
                self.ren_right.RemoveActor(self.coil_actor_transformed)
        except:
            pass
        self.coil_polydata = None
        self.coil_actor_left = None
        self.coil_landmarks = {}
        self.coil_landmark_actors_left = []
        self.coil_pairs_src = []
        self.coil_pairs_dst = []
        self.coil_pick_index = 0
        self.aux_dev_to_scan = None
        self.coil_actor_transformed = None
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()
        self._update_info("Step1 restarted.")

    def load_master_scan(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select master scan (with participant in the helmet)",
            "",
            "PLY/STL Files (*.ply *.stl)",
        )
        if not p:
            return
        self.helmet_polydata = load_mesh_to_polydata(p)
        self.helmet_filename = os.path.basename(p)
        if self.helmet_actor is not None:
            try:
                self.ren_right.RemoveActor(self.helmet_actor)
            except:
                pass
        self.helmet_actor = actor_from_polydata(
            self.helmet_polydata, color=(0.85, 0.85, 0.85), opacity=1.0
        )
        self.ren_right.AddActor(self.helmet_actor)
        self.ren_right.ResetCamera()
        self.vtk_right.GetRenderWindow().Render()
        self._update_info(f"Loaded master scan: {self.helmet_filename}")

    def load_aux_dev_stl(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select coil STL", "", "STL Files (*.stl)"
        )
        if not p:
            return
        self.coil_polydata = load_mesh_to_polydata(p)
        if self.coil_actor_left is not None:
            try:
                self.ren_left.RemoveActor(self.coil_actor_left)
            except:
                pass
        self.coil_actor_left = actor_from_polydata(
            self.coil_polydata, color=(0.7, 1, 0.2), opacity=1.0
        )
        self.ren_left.AddActor(self.coil_actor_left)
        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()
        self._update_info("Loaded coil STL (left).")

    def load_aux_dev_landmarks(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select coil landmarks CSV", "", "CSV Files (*.csv)"
        )
        if not p:
            return
        lm = {}
        with open(p, "r", newline="") as f:
            r = csv.reader(f)
            rows = list(r)
            for row in rows[1:]:
                if len(row) < 4:
                    continue
                lab = row[0].strip()
                try:
                    x, y, z = float(row[1]), float(row[2]), float(row[3])
                except:
                    continue
                lm[lab] = np.array([x, y, z], float)
        self.coil_landmarks = lm
        # draw landmarks on left
        for a in getattr(self, "coil_landmark_actors_left", []):
            try:
                self.ren_left.RemoveActor(a)
            except:
                pass
        self.coil_landmark_actors_left = []
        for i, (lab, p3) in enumerate(self.coil_landmarks.items()):
            c = PAIR_COLORS[i % len(PAIR_COLORS)]
            ac = sphere_actor(p3, color=c, radius=3.5)
            self.ren_left.AddActor(ac)
            self.coil_landmark_actors_left.append(ac)
        self.vtk_left.GetRenderWindow().Render()
        self._update_info(f"Loaded {len(self.coil_landmarks)} coil landmarks.")

    def start_aux_dev_landmark_pairs(self):

        if not self.coil_landmarks or self.helmet_polydata is None:
            self._update_info(
                "Need coil landmarks and master scan (participant in helmet)."
            )
            return

        ordered = [
            lab
            for lab in ["HTR", "HBR", "HBL", "HTL", "STR", "SBR", "SBL", "STL"]
            if lab in self.coil_landmarks
        ]

        if not ordered:
            ordered = list(self.coil_landmarks.keys())

        self.coil_expected_labels = ordered
        self.coil_pairs_src = []
        self.coil_pairs_dst = []
        self.coil_pick_index = 0

        self._update_info(
            "Pick coil landmarks in the RIGHT window in order: "
            + ", ".join(self.coil_expected_labels)
        )

        def handler(obj, event):

            # 🔹 If not SHIFT → let VTK rotate normally
            if not self.right_iren.GetShiftKey():
                obj.OnLeftButtonDown()
                return

            # 🔹 If done picking → let VTK rotate
            if self.coil_pick_index >= len(self.coil_expected_labels):
                obj.OnLeftButtonDown()
                return

            x, y = self.right_iren.GetEventPosition()
            picker = vtk.vtkPropPicker()

            ok = picker.Pick(x, y, 0, self.ren_right)
            if not ok:
                self._update_info("No pick on right.")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), float)
            lab = self.coil_expected_labels[self.coil_pick_index]

            self.coil_pairs_src.append(self.coil_landmarks[lab])
            self.coil_pairs_dst.append(pos)

            c = PAIR_COLORS[self.coil_pick_index % len(PAIR_COLORS)]

            self.ren_right.AddActor(sphere_actor(pos, color=c, radius=3.5))

            self.vtk_right.GetRenderWindow().Render()

            self.coil_pick_index += 1

            if self.coil_pick_index >= len(self.coil_expected_labels):
                try:
                    self.right_iren.RemoveObserver(obs_id)
                except:
                    pass

                self._update_info("All coil pairs picked. Click 'Compute coil→helmet'.")
            else:
                self._update_info(
                    f"Picked {lab}. Next: "
                    f"{self.coil_expected_labels[self.coil_pick_index]}"
                )

            # 🔹 IMPORTANT: After handling SHIFT pick,
            # still forward event so camera doesn't freeze
            obj.OnLeftButtonDown()

        obs_id = self._add_right_observer("LeftButtonPressEvent", handler)

    def undo_last_aux_dev_pair(self):
        if not getattr(self, "coil_pairs_src", []):
            self._update_info("No coil pairs to undo.")
            return
        self.coil_pairs_src.pop()
        self.coil_pairs_dst.pop()
        self.coil_pick_index = max(0, self.coil_pick_index - 1)
        # redraw right: helmet + placed coil pair spheres + transformed coil if present
        self.ren_right.RemoveAllViewProps()
        if self.helmet_actor is not None:
            self.ren_right.AddActor(self.helmet_actor)
        for i, pos in enumerate(self.coil_pairs_dst):
            c = PAIR_COLORS[i % len(PAIR_COLORS)]
            self.ren_right.AddActor(sphere_actor(pos, color=c, radius=3.5))
        if self.coil_actor_transformed is not None:
            self.ren_right.AddActor(self.coil_actor_transformed)
        self.vtk_right.GetRenderWindow().Render()
        self._update_info("Undid last coil pair.")

    def compute_aux_dev_to_scan(self):
        if not getattr(self, "coil_pairs_src", None) or len(self.coil_pairs_src) < 3:
            self._update_info("Need ≥3 coil pairs.")
            return
        src = np.vstack(self.coil_pairs_src)
        dst = np.vstack(self.coil_pairs_dst)
        T, rms, errs = compute_rigid_transform(src, dst)
        self.aux_dev_to_scan = T
        self._update_info(
            f"Computed coil to master scan transform RMS={rms:.3f} mm. Saved transform."
        )
        # auto save
        base = os.path.splitext(self.helmet_filename or "helmet")[0]
        save_transform(T, base + "_aux_dev_to_scan")

    def compute_aux_to_device(self):
        if self.aux_dev_to_scan is None or self.T_dev_to_scan is None:
            self._update_info("Need Step1 and Step3 transforms.")
            return

        T_scan_to_dev = np.linalg.inv(self.T_dev_to_scan)
        T_aux_to_device = T_scan_to_dev @ self.aux_dev_to_scan

        base = os.path.splitext(self.helmet_filename or "helmet")[0]
        save_transform(T_aux_to_device, base + "_aux_to_device")

        self._update_info("Saved aux→device transform.")

    def merge_aux_dev_into_helmet(self):
        if self.coil_polydata is None or self.aux_dev_to_scan is None:
            self._update_info("Need coil STL and T_aux_dev_to_scan first.")
            return
        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix_from_numpy(self.aux_dev_to_scan))
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(tf)
        tfilter.SetInputData(self.coil_polydata)
        tfilter.Update()
        if self.coil_actor_transformed is not None:
            try:
                self.ren_right.RemoveActor(self.coil_actor_transformed)
            except:
                pass
        self.coil_actor_transformed = actor_from_polydata(
            tfilter.GetOutput(), color=(0.7, 1.0, 0.2), opacity=0.45
        )
        self.ren_right.AddActor(self.coil_actor_transformed)
        self.ren_right.ResetCameraClippingRange()
        self.vtk_right.GetRenderWindow().Render()
        self._update_info("Previewed coil on helmet.")

    # ---------------------------------------------------------------
    # ----------------- STEP 2 (Outside -> Helmet) -----------------
    # ---------------------------------------------------------------

    def _populate_step2_controls(self):
        self._clear_step_controls()
        r = 0
        b1 = QPushButton("Load OUTSIDE head (LEFT)")
        b1.clicked.connect(self.load_head_scan)
        self.step_ctrl_layout.addWidget(b1, r, 0)
        r += 1
        b2 = QPushButton("Load HELMET scan (RIGHT)")
        b2.clicked.connect(self.load_master_scan)
        self.step_ctrl_layout.addWidget(b2, r, 0)
        r += 1
        b3 = QPushButton("Start fiducial mode (pick NAS,LPA,RPA, Nose, Chin)")
        b3.clicked.connect(self.start_fiducial_mode_step2)
        self.step_ctrl_layout.addWidget(b3, r, 0)
        r += 1
        b4 = QPushButton("Undo last fiducial (step2)")
        b4.clicked.connect(self.undo_last_fiducial_step2)
        self.step_ctrl_layout.addWidget(b4, r, 0)
        r += 1
        b5 = QPushButton("Start pairwise mode (LEFT->RIGHT)")
        b5.clicked.connect(self.start_pairs_step2)
        self.step_ctrl_layout.addWidget(b5, r, 0)
        r += 1
        b6 = QPushButton("Undo last pair (step2)")
        b6.clicked.connect(self.undo_last_pair_step2)
        self.step_ctrl_layout.addWidget(b6, r, 0)
        r += 1
        b7 = QPushButton("Compute outside->helmet")
        b7.clicked.connect(self.compute_head_to_scan)
        self.step_ctrl_layout.addWidget(b7, r, 0)
        r += 1
        b8 = QPushButton("Restart Step 2")
        b8.clicked.connect(self.restart_step2)
        self.step_ctrl_layout.addWidget(b8, r, 0)

    def load_head_scan(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select OUTSIDE head scan (LEFT)", "", "PLY/STL Files (*.ply *.stl)"
        )
        if not p:
            return
        self.head_polydata = load_mesh_to_polydata(p)
        if self.head_actor is not None:
            try:
                self.ren_left.RemoveActor(self.head_actor)
            except:
                pass
        self.head_actor = actor_from_polydata(
            self.head_polydata, color=(0.8, 0.8, 0.95), opacity=1.0
        )
        self.ren_left.AddActor(self.head_actor)
        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()
        self._update_info(f"Loaded outside head scan: {os.path.basename(p)}")

    def start_fiducial_mode_step2(self):
        # enable picking fiducials on left: pick NAS,LPA,RPA in order (SHIFT+click)
        self.picking_state = {"mode": "fiducials", "next": 0}
        self._update_info(
            "Fiducial mode: SHIFT+click NAS → LPA → RPA → NOSE → CHIN (LEFT) (outside scan)."
        )

    def undo_last_fiducial_step2(self):
        if not self.head_fiducials:
            self._update_info("No fiducials to undo.")
            return
        # remove last inserted fiducial (order NAS,LPA,RPA)
        order = ["NAS", "LPA", "RPA", "NOSE", "CHIN"]
        for lbl in reversed(order):
            if lbl in self.head_fiducials:
                # remove actor
                actor = self.head_fiducial_actors.get(lbl)
                if actor is not None:
                    try:
                        self.ren_left.RemoveActor(actor)
                    except:
                        pass
                del self.head_fiducials[lbl]
                if lbl in self.head_fiducial_actors:
                    del self.head_fiducial_actors[lbl]
                self.vtk_left.GetRenderWindow().Render()
                self._update_info(f"Removed fiducial {lbl}.")
                return

    def start_pairs_step2(self):
        if self.head_polydata is None or self.helmet_polydata is None:
            self._update_info("Load head scan (left) and master scan (right) first.")
            return
        self.step2_pairs_head = []
        self.step2_pairs_scan = []
        self._clear_step2_pair_markers()
        self.picking_state = {"mode": "pairs", "await": "left"}
        self._update_info(
            "Pair mode: SHIFT+click LEFT (head scan) then SHIFT+click RIGHT (master scan)."
        )

    def _left_click_step2(self, obj, event):

        if not self.left_iren.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        # ---------- Fiducials ----------
        if ps.get("mode") == "fiducials":
            x, y = self.left_iren.GetEventPosition()
            picker = vtk.vtkPropPicker()
            ok = picker.Pick(x, y, 0, self.ren_left)

            if not ok:
                self._update_info("No pick on left (fiducial).")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), dtype=float)

            order = ["NAS", "LPA", "RPA", "NOSE", "CHIN"]

            for lbl in order:
                if lbl not in self.head_fiducials:
                    next_lbl = lbl
                    break
            else:
                self._update_info("All fiducials already set.")
                obj.OnLeftButtonDown()
                return

            self.head_fiducials[next_lbl] = pos

            actor = sphere_actor(pos, color=FIDUCIAL_COLORS[next_lbl], radius=3.5)

            self.ren_left.AddActor(actor)
            self.head_fiducial_actors[next_lbl] = actor
            self.vtk_left.GetRenderWindow().Render()

            self._update_info(f"Set fiducial {next_lbl} at {np.round(pos, 2)}")

            obj.OnLeftButtonDown()
            return

        # ---------- Pair picking ----------
        elif ps.get("mode") == "pairs" and ps.get("await") == "left":
            x, y = self.left_iren.GetEventPosition()
            picker = vtk.vtkPropPicker()
            ok = picker.Pick(x, y, 0, self.ren_left)

            if not ok:
                self._update_info("No pick on left (pair).")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), dtype=float)

            ps["left_pt"] = pos
            ps["await"] = "right"
            self.picking_state = ps

            self._update_info(
                "Picked left pair point; now pick matching point on RIGHT."
            )

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def _right_click_step2(self, obj, event):

        if not self.right_iren.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        if ps.get("mode") == "pairs" and ps.get("await") == "right":
            x, y = self.right_iren.GetEventPosition()
            picker = vtk.vtkPropPicker()
            ok = picker.Pick(x, y, 0, self.ren_right)

            if not ok:
                self._update_info("No pick on right (pair).")
                obj.OnLeftButtonDown()
                return

            pos_scan = np.array(picker.GetPickPosition(), dtype=float)
            pos_left = ps.get("left_pt")

            if pos_left is None:
                self._update_info("No left point stored.")
                obj.OnLeftButtonDown()
                return

            idx = len(self.step2_pairs_head)
            c = PAIR_COLORS[idx % len(PAIR_COLORS)]

            self.step2_pairs_head.append(pos_left)
            self.step2_pairs_scan.append(pos_scan)

            a_l = sphere_actor(pos_left, color=c, radius=3.0)
            a_r = sphere_actor(pos_scan, color=c, radius=3.0)

            self.ren_left.AddActor(a_l)
            self.ren_right.AddActor(a_r)

            self.step2_pair_actors_left.append(a_l)
            self.step2_pair_actors_right.append(a_r)

            self.vtk_left.GetRenderWindow().Render()
            self.vtk_right.GetRenderWindow().Render()

            self.picking_state = {"mode": "pairs", "await": "left"}

            self._update_info(f"Added Step2 pair #{idx + 1}.")

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def undo_last_pair_step2(self):
        if not self.step2_pairs_head:
            self._update_info("No Step2 pairs to undo.")
            return
        self.step2_pairs_head.pop()
        self.step2_pairs_scan.pop()
        if self.step2_pair_actors_left:
            a = self.step2_pair_actors_left.pop()
            self.ren_left.RemoveActor(a)
        if self.step2_pair_actors_right:
            b = self.step2_pair_actors_right.pop()
            self.ren_right.RemoveActor(b)
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()
        self._update_info(
            f"Undid last Step2 pair. {len(self.step2_pairs_head)} remain."
        )

    def _clear_step2_pair_markers(self):
        for a in self.step2_pair_actors_left:
            try:
                self.ren_left.RemoveActor(a)
            except:
                pass
        for a in self.step2_pair_actors_right:
            try:
                self.ren_right.RemoveActor(a)
            except:
                pass
        self.step2_pair_actors_left = []
        self.step2_pair_actors_right = []
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def compute_head_to_scan(self):
        if len(self.step2_pairs_head) < 3:
            self._update_info("Need at least 3 Step2 pairs to compute transform.")
            return
        src = np.vstack(self.step2_pairs_head)
        dst = np.vstack(self.step2_pairs_scan)
        T, rms, errs = compute_rigid_transform(src, dst)
        self.T_head_to_scan = T
        self._update_info(
            f"Computed head_to_scan RMS={rms:.3f} mm. Overlaying transformed head scan onto RIGHT."
        )
        # overlay transformed head scan on right (light orange)
        self._overlay_head_transformed(T)
        # save transform
        base = os.path.splitext(self.helmet_filename or "helmet")[0]
        save_transform(T, base + "_head_to_scan")

    def _overlay_head_transformed(self, T):
        # remove old transformed head scan if any
        try:
            if (
                hasattr(self, "head_actor_transformed")
                and self.head_actor_transformed is not None
            ):
                self.ren_right.RemoveActor(self.head_actor_transformed)
        except:
            pass
        if self.head_polydata is None:
            return
        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix_from_numpy(T))
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(tf)
        tfilter.SetInputData(self.head_polydata)
        tfilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(tfilter.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.6, 0.3)
        actor.GetProperty().SetOpacity(0.45)
        self.head_actor_transformed = actor
        self.ren_right.AddActor(actor)
        # transform fiducials too and plot on right
        R = T[:3, :3]
        t = T[:3, 3]
        for lbl, p in self.head_fiducials.items():
            p_hel = R @ p + t
            try:
                # if existing fid actor on right, remove it first
                if (
                    hasattr(self, "head_fiducial_actors_right")
                    and lbl in self.head_fiducial_actors_right
                ):
                    self.ren_right.RemoveActor(self.head_fiducial_actors_right[lbl])
            except:
                pass
            ac = sphere_actor(p_hel, color=FIDUCIAL_COLORS[lbl], radius=3.5)
            if not hasattr(self, "head_fiducial_actors_right"):
                self.head_fiducial_actors_right = {}
            self.ren_right.AddActor(ac)
            self.head_fiducial_actors_right[lbl] = ac
        self.ren_right.ResetCameraClippingRange()
        self.vtk_right.GetRenderWindow().Render()

    def restart_step2(self):
        # remove transformed actor and pair markers
        try:
            if (
                hasattr(self, "head_actor_transformed")
                and self.head_actor_transformed is not None
            ):
                self.ren_right.RemoveActor(self.head_actor_transformed)
        except:
            pass
        # remove right fiducials
        try:
            if hasattr(self, "head_fiducial_actors_right"):
                for a in self.head_fiducial_actors_right.values():
                    try:
                        self.ren_right.RemoveActor(a)
                    except:
                        pass
                self.head_fiducial_actors_right = {}
        except:
            pass
        # remove left fiducials
        for a in self.head_fiducial_actors.values():
            try:
                self.ren_left.RemoveActor(a)
            except:
                pass
        self.head_fiducials = {}
        self.head_fiducial_actors = {}
        self._clear_step2_pair_markers()
        self.step2_pairs_head = []
        self.step2_pairs_scan = []
        self.T_head_to_scan = None
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()
        self._update_info("Restarted Step2 (fiducials & pairs cleared).")

    # -------------------------------------------------------------------
    # ------------------ STEP 3 (Sensors & STL -> scan) -----------------
    # -------------------------------------------------------------------
    def _populate_step3_controls(self):
        self._clear_step_controls()
        r = 0
        b1 = QPushButton("Load helmet STL (LEFT)")
        b1.clicked.connect(self.load_helmet_cad)
        self.step_ctrl_layout.addWidget(b1, r, 0)
        r += 1
        b2 = QPushButton("Load master scan (RIGHT)")
        b2.clicked.connect(self.load_master_scan)
        self.step_ctrl_layout.addWidget(b2, r, 0)
        r += 1
        b3 = QPushButton("Load sensor locations CSV (plot LEFT)")
        b3.clicked.connect(self.load_sensor_csv)
        self.step_ctrl_layout.addWidget(b3, r, 0)
        r += 1
        b4 = QPushButton("Start helmet to master scan pair mode (LEFT->RIGHT)")
        b4.clicked.connect(self.start_pairs_step3)
        self.step_ctrl_layout.addWidget(b4, r, 0)
        r += 1
        b5 = QPushButton("Undo last pair (Step3)")
        b5.clicked.connect(self.undo_last_pair_step3)
        self.step_ctrl_layout.addWidget(b5, r, 0)
        r += 1
        b6 = QPushButton("Compute transform & save sensors")
        b6.clicked.connect(self.compute_sensors_transform)
        self.step_ctrl_layout.addWidget(b6, r, 0)
        r += 1
        b7 = QPushButton("Restart Step3")
        b7.clicked.connect(self.restart_step3)
        self.step_ctrl_layout.addWidget(b7, r, 0)

    def restart_step3(self):
        # remove actors (left/right)
        try:
            if self.helmet_stl_actor_left is not None:
                self.ren_left.RemoveActor(self.helmet_stl_actor_left)
        except:
            pass
        try:
            if self.helmet_stl_actor_overlay is not None:
                self.ren_right.RemoveActor(self.helmet_stl_actor_overlay)
        except:
            pass
        self.helmet_stl_raw = None
        self.helmet_stl_actor_left = None
        self.helmet_stl_actor_overlay = None
        # sensors
        for a in (
            self.sensor_actors_left
            + self.transformed_sensor_actors
            + self.transformed_vector_actors
        ):
            try:
                self.ren_left.RemoveActor(a)
                self.ren_right.RemoveActor(a)
            except:
                pass
        self.sensor_positions = None
        self.sensor_axes = None
        self.sensor_df = None
        self.sensor_csv_path = None
        self.sensor_actors_left = []
        self.transformed_sensor_actors = []
        self.transformed_vector_actors = []
        # pairs
        self.ref_point_pairs_helmet_stl = []
        self.ref_point_pairs_scan = []
        self.pair_actors_left = []
        self.pair_actors_right = []
        self.T_dev_to_scan = None
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()
        self._update_info("Restarted Step3.")

    def load_helmet_cad(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load helmet STL", "", "STL/PLY Files (*.stl *.ply)"
        )
        if not p:
            return
        self.helmet_stl_raw = load_mesh_to_polydata(p)
        if self.helmet_stl_actor_left is not None:
            try:
                self.ren_left.RemoveActor(self.helmet_stl_actor_left)
            except:
                pass
        self.helmet_stl_actor_left = actor_from_polydata(
            self.helmet_stl_raw, color=(0.7, 0.9, 1.0), opacity=0.8
        )
        self.ren_left.AddActor(self.helmet_stl_actor_left)
        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()
        self._update_info(f"Loaded helmet STL: {os.path.basename(p)}")

    def load_sensor_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load sensor CSV", "", "CSV Files (*.csv)"
        )
        if not p:
            return
        df = pd.read_csv(p)
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
        for c in cols:
            if c not in df.columns:
                self._update_info(f"CSV missing column: {c}")
                return
        self.sensor_df = df.copy()
        self.sensor_csv_path = p
        self.sensor_positions = df[["X", "Y", "Z"]].to_numpy(dtype=float)
        x = df[["x_i", "x_j", "x_k"]].to_numpy(dtype=float)
        y = df[["y_i", "y_j", "y_k"]].to_numpy(dtype=float)
        z = df[["z_i", "z_j", "z_k"]].to_numpy(dtype=float)
        self.sensor_axes = np.stack([x, y, z], axis=1)
        # display original sensors on LEFT (on top of STL)
        for a in self.sensor_actors_left:
            try:
                self.ren_left.RemoveActor(a)
            except:
                pass
        self.sensor_actors_left = []
        for pnt in self.sensor_positions:
            ac = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
            self.ren_left.AddActor(ac)
            self.sensor_actors_left.append(ac)
        # show vectors on left if toggled
        self._draw_sensor_vectors_left()
        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()
        self._update_info(
            f"Loaded sensor CSV: {os.path.basename(p)} ({len(self.sensor_positions)} sensors)."
        )

    def start_pairs_step3(self):
        if self.helmet_stl_raw is None or self.helmet_polydata is None:
            self._update_info(
                "Need helmet STL (left) and participant in the helmet scan (right) loaded."
            )
            return
        self.ref_point_pairs_helmet_stl = []
        self.ref_point_pairs_scan = []
        self._clear_step3_pair_markers()
        self.picking_state = {"mode": "step3_pairs", "await": "left"}
        self._update_info(
            "Step3: SHIFT+click helmet STL on LEFT then matching point on RIGHT (scan)."
        )

    def _left_click_step3(self, obj, event):

        if not self.left_iren.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        if ps.get("mode") == "step3_pairs" and ps.get("await") == "left":
            x, y = self.left_iren.GetEventPosition()
            picker = vtk.vtkPropPicker()

            ok = picker.Pick(x, y, 0, self.ren_left)
            if not ok:
                self._update_info("No pick on left (STL).")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), dtype=float)

            self.picking_state["left_pt"] = pos
            self.picking_state["await"] = "right"

            self._update_info(
                "Selected STL point; now SHIFT+click matching point on RIGHT."
            )

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def _right_click_step3(self, obj, event):

        if not self.right_iren.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        if ps.get("mode") == "step3_pairs" and ps.get("await") == "right":
            x, y = self.right_iren.GetEventPosition()
            picker = vtk.vtkPropPicker()

            ok = picker.Pick(x, y, 0, self.ren_right)
            if not ok:
                self._update_info("No pick on right (scan).")
                obj.OnLeftButtonDown()
                return

            pos_scan = np.array(picker.GetPickPosition(), dtype=float)
            pos_stl = ps.get("left_pt")

            if pos_stl is None:
                self._update_info("No left point stored.")
                obj.OnLeftButtonDown()
                return

            self.ref_point_pairs_helmet_stl.append(pos_stl)
            self.ref_point_pairs_scan.append(pos_scan)

            idx = len(self.ref_point_pairs_helmet_stl) - 1
            c = PAIR_COLORS[idx % len(PAIR_COLORS)]

            a_l = sphere_actor(pos_stl, color=c, radius=3.0)
            a_r = sphere_actor(pos_scan, color=c, radius=3.0)

            self.ren_left.AddActor(a_l)
            self.ren_right.AddActor(a_r)

            self.pair_actors_left.append(a_l)
            self.pair_actors_right.append(a_r)

            self.vtk_left.GetRenderWindow().Render()
            self.vtk_right.GetRenderWindow().Render()

            self.picking_state = {"mode": "step3_pairs", "await": "left"}

            self._update_info(f"Added Step3 pair #{idx + 1}.")

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def undo_last_pair_step3(self):
        if not self.ref_point_pairs_helmet_stl:
            self._update_info("No Step3 pairs to undo.")
            return
        self.ref_point_pairs_helmet_stl.pop()
        self.ref_point_pairs_scan.pop()
        if self.pair_actors_left:
            a = self.pair_actors_left.pop()
            self.ren_left.RemoveActor(a)
        if self.pair_actors_right:
            b = self.pair_actors_right.pop()
            self.ren_right.RemoveActor(b)
        # re-render and re-add transformed if present
        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()
        if self.T_dev_to_scan is not None:
            self._overlay_step3_transformed(self.T_dev_to_scan)
        self._update_info("Undid last Step3 pair.")

    def _clear_step3_pair_markers(self):
        if hasattr(self, "pair_actors_left"):
            for a in self.pair_actors_left:
                try:
                    self.ren_left.RemoveActor(a)
                except:
                    pass

        if hasattr(self, "pair_actors_right"):
            for a in self.pair_actors_right:
                try:
                    self.ren_right.RemoveActor(a)
                except:
                    pass

        self.pair_actors_left = []
        self.pair_actors_right = []

        self.vtk_left.GetRenderWindow().Render()
        self.vtk_right.GetRenderWindow().Render()

    def compute_sensors_transform(self):
        if len(self.ref_point_pairs_helmet_stl) < 3:
            self._update_info("Need ≥3 Step3 pairs.")
            return
        src = np.vstack(self.ref_point_pairs_helmet_stl)
        dst = np.vstack(self.ref_point_pairs_scan)
        T, rms, errs = compute_rigid_transform(src, dst)
        self.T_dev_to_scan = T
        # -------------------------------------------------
        # Compute aux -> device if aux transform exists
        # -------------------------------------------------
        if self.aux_dev_to_scan is not None:
            T_scan_to_dev = np.linalg.inv(self.T_dev_to_scan)
            T_aux_to_device = T_scan_to_dev @ self.aux_dev_to_scan

            base = os.path.splitext(self.helmet_filename or "helmet")[0]
            save_transform(T_aux_to_device, base + "_aux_to_device")

            self._update_info("Saved aux→device transform.")
        # ---------------------------------------------------
        # Compute device -> head if Step2 is available
        # ---------------------------------------------------
        if self.T_head_to_scan is not None:
            T_scan_to_head = np.linalg.inv(self.T_head_to_scan)
            T_dev_to_head = T_scan_to_head @ self.T_dev_to_scan
            self.T_dev_to_head = T_dev_to_head

            base = os.path.splitext(self.helmet_filename or "helmet")[0]
            save_transform(T_dev_to_head, base + "_dev_to_head")

            self._update_info("Computed device→head and saved dev_head_t.")
        else:
            self._update_info(
                "Computed dev_to_scan, but head_to_scan not available — cannot compute dev_to_head yet."
            )

        self._update_info(
            f"Computed helmet→scan RMS={rms:.3f} mm. Previewing transformed STL & sensors on RIGHT."
        )
        # overlay transformed STL + transformed sensors
        self._overlay_step3_transformed(T)
        # save transform automatically
        base = os.path.splitext(self.helmet_filename or "helmet")[0]
        save_transform(T, base + "_T_dev_to_scan")
        # save transformed sensors csv
        self._save_transformed_sensors_csv(T, base)

    def _overlay_step3_transformed(self, T):
        # transformed STL preview on RIGHT
        R = T[:3, :3]
        t = T[:3, 3]
        try:
            if self.helmet_stl_actor_overlay is not None:
                self.ren_right.RemoveActor(self.helmet_stl_actor_overlay)
        except:
            pass
        if self.helmet_stl_raw is not None:
            tf = vtk.vtkTransform()
            tf.SetMatrix(vtk_matrix_from_numpy(T))
            tfilter = vtk.vtkTransformPolyDataFilter()
            tfilter.SetTransform(tf)
            tfilter.SetInputData(self.helmet_stl_raw)
            tfilter.Update()
            self.helmet_stl_actor_overlay = actor_from_polydata(
                tfilter.GetOutput(), color=(0.3, 0.8, 1.0), opacity=0.4
            )
            if self.cb_stl.isChecked():
                self.ren_right.AddActor(self.helmet_stl_actor_overlay)
        # transformed sensors on right
        for a in self.transformed_sensor_actors:
            try:
                self.ren_right.RemoveActor(a)
            except:
                pass
        for a in self.transformed_vector_actors:
            try:
                self.ren_right.RemoveActor(a)
            except:
                pass
        self.transformed_sensor_actors = []
        self.transformed_vector_actors = []
        if self.sensor_positions is not None:
            pts_t = (R @ self.sensor_positions.T).T + t
            for p in pts_t:
                a = sphere_actor(p, color=(0.6, 0.0, 0.6), radius=3.2)
                self.ren_right.AddActor(a)
                self.transformed_sensor_actors.append(a)
            if self.cb_vec.isChecked() and (self.sensor_axes is not None):
                axes_t = np.empty_like(self.sensor_axes)
                for i in range(3):
                    axes_t[:, i, :] = (R @ self.sensor_axes[:, i, :].T).T
                for p, vecs in zip(pts_t, axes_t):
                    for vec, color in [
                        (vecs[0], (1, 0, 0)),
                        (vecs[1], (0, 1, 0)),
                        (vecs[2], (0, 0, 1)),
                    ]:
                        ar = arrow_actor(p, vec, scale=10.0, color=color)
                        if ar is not None:
                            self.ren_right.AddActor(ar)
                            self.transformed_vector_actors.append(ar)
        self.ren_right.ResetCameraClippingRange()
        self.vtk_right.GetRenderWindow().Render()

    def _save_transformed_sensors_csv(self, T, base):
        if self.sensor_positions is None:
            return
        R = T[:3, :3]
        t = T[:3, 3]
        pts_t = (R @ self.sensor_positions.T).T + t
        df_out = self.sensor_df.copy()
        df_out[["X", "Y", "Z"]] = pts_t
        if self.sensor_axes is not None:
            axes_t = np.empty_like(self.sensor_axes)
            for i in range(3):
                axes_t[:, i, :] = (R @ self.sensor_axes[:, i, :].T).T
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
            os.path.dirname(self.sensor_csv_path)
            if getattr(self, "sensor_csv_path", None)
            else os.getcwd()
        )
        out_path = os.path.join(out_dir, base + "_sensors_to_scan_transformed.csv")
        df_out.to_csv(out_path, index=False)
        self._update_info(f"Saved transformed sensors CSV: {out_path}")

    def _draw_sensor_vectors_left(self):
        # rebuild left view: helmet_stl + sensor spheres + optional vectors
        self.ren_left.RemoveAllViewProps()
        if self.helmet_stl_actor_left is not None:
            self.ren_left.AddActor(self.helmet_stl_actor_left)
        self.sensor_actors_left = []
        if self.sensor_positions is not None:
            for pnt in self.sensor_positions:
                a = sphere_actor(pnt, color=(0.2, 0.4, 0.9), radius=2.8)
                self.ren_left.AddActor(a)
                self.sensor_actors_left.append(a)
        if self.cb_vec.isChecked() and (self.sensor_axes is not None):
            for p, vecs in zip(self.sensor_positions, self.sensor_axes):
                for vec, color in [
                    (vecs[0], (1, 0, 0)),
                    (vecs[1], (0, 1, 0)),
                    (vecs[2], (0, 0, 1)),
                ]:
                    ar = arrow_actor(p, vec, scale=6.0, color=color)
                    if ar is not None:
                        self.ren_left.AddActor(ar)
        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()

    # toggles
    def _toggle_vectors(self, state):
        self.show_vectors = bool(state)
        # redraw left vectors and transformed vectors if transform exists
        self._draw_sensor_vectors_left()
        if self.T_dev_to_scan is not None:
            self._overlay_step3_transformed(self.T_dev_to_scan)

    def _toggle_helmet_overlay(self, state):
        if self.helmet_stl_actor_overlay is None:
            return
        if self.cb_stl.isChecked():
            self.ren_right.AddActor(self.helmet_stl_actor_overlay)
        else:
            try:
                self.ren_right.RemoveActor(self.helmet_stl_actor_overlay)
            except:
                pass
        self.vtk_right.GetRenderWindow().Render()

    def _toggle_aux_dev_overlay(self, state):
        if self.coil_actor_transformed is None:
            return
        if state:
            self.ren_right.AddActor(self.coil_actor_transformed)
        else:
            self.ren_right.RemoveActor(self.coil_actor_transformed)
        self.vtk_right.GetRenderWindow().Render()

    def _toggle_head_overlay(self, state):
        if (
            not hasattr(self, "head_actor_transformed")
            or self.head_actor_transformed is None
        ):
            return
        if state:
            self.ren_right.AddActor(self.head_actor_transformed)
        else:
            self.ren_right.RemoveActor(self.head_actor_transformed)
        self.vtk_right.GetRenderWindow().Render()

    def _toggle_mri_overlay(self, state):
        if not hasattr(self, "mri_actor_right") or self.mri_actor_right is None:
            return
        if state:
            self.ren_right.AddActor(self.mri_actor_right)
        else:
            self.ren_right.RemoveActor(self.mri_actor_right)
        self.vtk_right.GetRenderWindow().Render()

    # ---------------------------------------------------------
    # ----------------- STEP 4 (head -> MRI) -----------------
    # ---------------------------------------------------------
    def _populate_step4_controls(self):
        self._clear_step_controls()
        r = 0

        b1 = QPushButton("Load MRI mesh (LEFT)")
        b1.clicked.connect(self.load_mri_mesh)
        self.step_ctrl_layout.addWidget(b1, r, 0)
        r += 1

        b2 = QPushButton("Load MRI fiducials (.fif)")
        b2.clicked.connect(self.load_mri_fiducials)
        self.step_ctrl_layout.addWidget(b2, r, 0)
        r += 1

        b3 = QPushButton("Compute initial Head→MRI (base on step 2 fiducials)")
        b3.clicked.connect(self.compute_head_to_mri_initial)
        self.step_ctrl_layout.addWidget(b3, r, 0)
        r += 1

        b4 = QPushButton("Run ICP refinement")
        b4.clicked.connect(self.run_icp_head_to_mri)
        self.step_ctrl_layout.addWidget(b4, r, 0)
        r += 1

        b_preview = QPushButton("Preview MRI on master scan (RIGHT)")
        b_preview.clicked.connect(self.preview_mri_on_scan)
        self.step_ctrl_layout.addWidget(b_preview, r, 0)
        r += 1

        b5 = QPushButton("Save Head→MRI")
        b5.clicked.connect(self.save_head_to_mri)
        self.step_ctrl_layout.addWidget(b5, r, 0)

    def load_mri_mesh(self):

        p, _ = QFileDialog.getOpenFileName(
            self, "Load MRI mesh", "", "PLY/STL Files (*.ply *.stl)"
        )
        if not p:
            return

        self.mri_polydata = load_mesh_to_polydata(p)

        # -------------------------------------------------
        # AUTO-DETECT UNIT SCALE
        # -------------------------------------------------
        bounds = self.mri_polydata.GetBounds()
        max_abs = max(abs(b) for b in bounds)

        print("MRI bounds before scaling:", bounds)

        # If values look like meters (~0.1), scale to mm
        if max_abs < 1.0:
            print("MRI appears to be in meters → scaling to mm")

            scale_tf = vtk.vtkTransform()
            scale_tf.Scale(1000.0, 1000.0, 1000.0)

            tf_filter = vtk.vtkTransformPolyDataFilter()
            tf_filter.SetTransform(scale_tf)
            tf_filter.SetInputData(self.mri_polydata)
            tf_filter.Update()

            self.mri_polydata = tf_filter.GetOutput()

            bounds_after = self.mri_polydata.GetBounds()
            print("MRI bounds after scaling:", bounds_after)

        else:
            print("MRI appears already in mm")

        # -------------------------------------------------
        # Display on left
        # -------------------------------------------------
        actor = actor_from_polydata(
            self.mri_polydata, color=(0.85, 0.85, 0.95), opacity=0.7
        )

        self.ren_left.RemoveAllViewProps()
        self.ren_left.AddActor(actor)

        self.ren_left.ResetCamera()
        self.vtk_left.GetRenderWindow().Render()

        self._update_info("Loaded MRI mesh (left).")

    def _draw_mri_fiducials(self):

        # Remove old actors
        if hasattr(self, "mri_fiducial_actors"):
            for a in self.mri_fiducial_actors.values():
                try:
                    self.ren_left.RemoveActor(a)
                except:
                    pass

        self.mri_fiducial_actors = {}

        for lbl, pos in self.mri_fiducials.items():
            actor = sphere_actor(
                pos, color=FIDUCIAL_COLORS.get(lbl, (1, 1, 0)), radius=3.5
            )

            self.ren_left.AddActor(actor)
            self.mri_fiducial_actors[lbl] = actor

        self.vtk_left.GetRenderWindow().Render()

    def load_mri_fiducials(self):

        path = QFileDialog.getOpenFileName(
            self, "Load MRI Fiducials (.fif)", "", "FIF files (*.fif)"
        )[0]

        if not path:
            return

        fid, coord_frame = read_fiducials(path)

        self.mri_fiducials = {}

        for f in fid:
            ident = f["ident"]
            coord_mm = f["r"] * 1000.0  # meters → mm

            if ident == FIFF.FIFFV_POINT_NASION:
                self.mri_fiducials["NAS"] = coord_mm
            elif ident == FIFF.FIFFV_POINT_LPA:
                self.mri_fiducials["LPA"] = coord_mm
            elif ident == FIFF.FIFFV_POINT_RPA:
                self.mri_fiducials["RPA"] = coord_mm

        print("Loaded MRI fiducials:", self.mri_fiducials)

        if len(self.mri_fiducials) != 3:
            self._update_info(
                f"Incomplete MRI fiducials: {list(self.mri_fiducials.keys())}"
            )
            return
        self._draw_mri_fiducials()

        self._update_info("Loaded MRI fiducials from FIF.")

    def compute_head_to_mri_initial(self):

        required = ["NAS", "LPA", "RPA"]

        if not all(k in self.head_fiducials for k in required):
            self._update_info("Head fiducials missing (Step2).")
            return

        if not all(k in self.mri_fiducials for k in required):
            self._update_info("MRI fiducials missing.")
            return

        src = np.array(
            [
                self.head_fiducials["NAS"],
                self.head_fiducials["LPA"],
                self.head_fiducials["RPA"],
            ]
        )

        dst = np.array(
            [
                self.mri_fiducials["NAS"],
                self.mri_fiducials["LPA"],
                self.mri_fiducials["RPA"],
            ]
        )

        T, rms, errs = compute_rigid_transform(src, dst)

        self.T_head_to_mri = T
        self.head_to_mri_rms = rms
        self.head_to_mri_errors = errs

        print("Head→MRI RMS:", rms)
        print("Errors per fiducial:", errs)

        self._update_info(
            f"Head→MRI RMS = {rms:.2f} mm | "
            f"NAS={errs[0]:.2f}, "
            f"LPA={errs[1]:.2f}, "
            f"RPA={errs[2]:.2f}"
        )

        self._preview_head_on_mri(T)

    def _preview_head_on_mri(self, T):
        if self.head_polydata is None:
            return

        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix_from_numpy(T))

        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetTransform(tf)
        filt.SetInputData(self.head_polydata)
        filt.Update()

        actor = actor_from_polydata(
            filt.GetOutput(), color=(0.3, 0.6, 1.0), opacity=0.5
        )

        self.ren_left.AddActor(actor)
        self.vtk_left.GetRenderWindow().Render()

    def start_mri_fiducial_mode(self):
        self.mri_fiducials = {}
        self.mri_fiducial_actors = {}
        self._update_info("MRI fiducial mode: SHIFT+click NAS,LPA,RPA on LEFT (MRI).")

    def extract_face_roi(self, polydata, fid_dict, plane_offset_mm=15.0):

        required = ["NAS", "LPA", "RPA", "NOSE", "CHIN"]
        if not all(k in fid_dict for k in required):
            print("ROI extraction: missing fiducials")
            return None

        nas = fid_dict["NAS"]
        lpa = fid_dict["LPA"]
        rpa = fid_dict["RPA"]
        nose = fid_dict["NOSE"]
        chin = fid_dict["CHIN"]

        # --------------------------------------------------
        # 1) ROI center (face region)
        # --------------------------------------------------
        pts = np.vstack([nas, lpa, rpa, nose, chin])
        roi_center = np.mean(pts, axis=0)

        max_dist = np.max(np.linalg.norm(pts - roi_center, axis=1))
        roi_radius = 2.2 * max_dist  # larger radius to include LPA/RPA fully

        # --------------------------------------------------
        # 2) Define fiducial plane (NAS–LPA–RPA)
        # --------------------------------------------------
        v1 = lpa - nas
        v2 = rpa - nas
        plane_normal = np.cross(v1, v2)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Make sure normal points upward (towards top of head)
        if np.dot(plane_normal, chin - nas) > 0:
            plane_normal = -plane_normal

        # Shift plane upward
        plane_point = nas + plane_offset_mm * plane_normal

        # --------------------------------------------------
        # 3) Select points
        # --------------------------------------------------
        points = polydata.GetPoints()
        n_pts = points.GetNumberOfPoints()

        ids = vtk.vtkIdTypeArray()

        for i in range(n_pts):
            p = np.array(points.GetPoint(i))

            # radial constraint
            if np.linalg.norm(p - roi_center) > roi_radius:
                continue

            # chop-off plane: keep only points BELOW shifted plane
            if np.dot(p - plane_point, plane_normal) > 0:
                continue

            ids.InsertNextValue(i)

        if ids.GetNumberOfTuples() < 500:
            print("ROI too small — returning None")
            return None

        # --------------------------------------------------
        # 4) Extract subset
        # --------------------------------------------------
        selection_node = vtk.vtkSelectionNode()
        selection_node.SetFieldType(vtk.vtkSelectionNode.POINT)
        selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        selection_node.SetSelectionList(ids)

        selection = vtk.vtkSelection()
        selection.AddNode(selection_node)

        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, polydata)
        extract.SetInputData(1, selection)
        extract.Update()

        geo = vtk.vtkGeometryFilter()
        geo.SetInputConnection(extract.GetOutputPort())
        geo.Update()

        roi_poly = geo.GetOutput()

        print("ROI points:", roi_poly.GetNumberOfPoints())
        return roi_poly

    def compute_surface_rms(self, src_poly, dst_poly):
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(dst_poly)
        locator.BuildLocator()

        pts = src_poly.GetPoints()
        n = pts.GetNumberOfPoints()

        d2 = []

        for i in range(n):
            p = pts.GetPoint(i)
            idx = locator.FindClosestPoint(p)
            q = dst_poly.GetPoint(idx)
            d2.append(np.sum((np.array(p) - np.array(q)) ** 2))

        d2 = np.array(d2)
        return np.sqrt(np.mean(d2))

    def run_icp_head_to_mri(self):

        if self.T_head_to_mri is None:
            self._update_info("Compute initial fiducial alignment first.")
            return

        if self.head_polydata is None or self.mri_polydata is None:
            self._update_info("Need head and MRI surfaces loaded.")
            return

        if not all(
            k in self.head_fiducials for k in ["NAS", "LPA", "RPA", "NOSE", "CHIN"]
        ):
            self._update_info("Need 5 head fiducials.")
            return

        # --------------------------------------------------
        # 1) Extract ROI in HEAD space
        # --------------------------------------------------
        roi_head_native = self.extract_face_roi(self.head_polydata, self.head_fiducials)

        if roi_head_native is None:
            self._update_info("ROI extraction failed.")
            return

        # --------------------------------------------------
        # 2) Transform ROI to MRI space
        # --------------------------------------------------
        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix_from_numpy(self.T_head_to_mri))

        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetTransform(tf)
        tf_filter.SetInputData(roi_head_native)
        tf_filter.Update()

        roi_head = tf_filter.GetOutput()

        # --------------------------------------------------
        # 3) Show ROI in LEFT window (red)
        # --------------------------------------------------
        if hasattr(self, "_roi_actor") and self._roi_actor is not None:
            try:
                self.ren_left.RemoveActor(self._roi_actor)
            except:
                pass

        self._roi_actor = actor_from_polydata(roi_head, color=(1, 0, 0), opacity=0.7)

        self.ren_left.AddActor(self._roi_actor)
        self.vtk_left.GetRenderWindow().Render()

        # --------------------------------------------------
        # 4) Initial RMS
        # --------------------------------------------------
        initial_rms = self.compute_surface_rms(roi_head, self.mri_polydata)

        # --------------------------------------------------
        # 5) ICP
        # --------------------------------------------------
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(roi_head)
        icp.SetTarget(self.mri_polydata)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(50)
        icp.StartByMatchingCentroidsOff()
        icp.Modified()
        icp.Update()

        # Extract matrix
        T_icp = np.eye(4)
        m = icp.GetMatrix()
        for i in range(4):
            for j in range(4):
                T_icp[i, j] = m.GetElement(i, j)

        # Compose
        self.T_head_to_mri = T_icp @ self.T_head_to_mri

        # --------------------------------------------------
        # 6) Recompute RMS after ICP
        # --------------------------------------------------
        tf2 = vtk.vtkTransform()
        tf2.SetMatrix(vtk_matrix_from_numpy(self.T_head_to_mri))

        tf_filter2 = vtk.vtkTransformPolyDataFilter()
        tf_filter2.SetTransform(tf2)
        tf_filter2.SetInputData(roi_head_native)
        tf_filter2.Update()

        roi_refined = tf_filter2.GetOutput()

        final_rms = self.compute_surface_rms(roi_refined, self.mri_polydata)

        self._update_info(
            f"ICP complete | ROI RMS: {initial_rms:.2f} → {final_rms:.2f} mm"
        )

        # --------------------------------------------------
        # 7) Preview full head
        # --------------------------------------------------
        self._preview_head_on_mri(self.T_head_to_mri)

    def show_full_head_on_mri(self):

        if self.T_head_to_mri is None:
            return

        self._preview_head_on_mri(self.T_head_to_mri)

    def show_roi_on_mri(self):

        if not hasattr(self, "_roi_head_mri"):
            self._update_info("Run ICP first.")
            return

        # remove previous preview
        if hasattr(self, "_preview_actor"):
            try:
                self.ren_right.RemoveActor(self._preview_actor)
            except:
                pass

        actor = actor_from_polydata(self._roi_head_mri, color=(1, 0, 0), opacity=0.7)

        self._preview_actor = actor
        self.ren_right.AddActor(actor)
        self.vtk_right.GetRenderWindow().Render()

    def preview_mri_on_scan(self):

        if self.T_head_to_mri is None:
            self._update_info("Compute Head→MRI first.")
            return

        if self.T_head_to_scan is None:
            self._update_info("Need Head→Scan (Step2) first.")
            return

        if self.mri_polydata is None:
            self._update_info("Load MRI mesh first.")
            return

        # MRI is in MRI frame
        # We want MRI in SCAN frame (RIGHT window)

        # head -> mri  = T_head_to_mri
        # head -> scan = T_head_to_scan
        # So:

        # mri -> head
        T_mri_to_head = np.linalg.inv(self.T_head_to_mri)

        # head -> scan
        T_head_to_scan = self.T_head_to_scan

        # mri -> scan
        T_mri_to_scan = T_head_to_scan @ T_mri_to_head

        tf = vtk.vtkTransform()
        tf.SetMatrix(vtk_matrix_from_numpy(T_mri_to_scan))

        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetTransform(tf)
        filt.SetInputData(self.mri_polydata)
        filt.Update()

        # remove old if exists
        if self.mri_actor_right is not None:
            self.ren_right.RemoveActor(self.mri_actor_right)

        self.mri_actor_right = actor_from_polydata(
            filt.GetOutput(),
            color=(1.0, 0.6, 0.8),  # pink
            opacity=0.4,
        )

        self.ren_right.AddActor(self.mri_actor_right)
        self.vtk_right.GetRenderWindow().Render()

        self._update_info("Previewed MRI on master scan (RIGHT).")

    def save_head_to_mri(self):

        if self.T_head_to_mri is None:
            return

        base = os.path.splitext(self.helmet_filename or "helmet")[0]

        # --- Convert mm → meters for MNE ---
        T_m = self.T_head_to_mri.copy()
        T_m[:3, 3] /= 1000.0

        save_transform(self.T_head_to_mri, base + "_head_to_mri_mm")

        trans = Transform("head", "mri", T_m)

        mne.write_trans(base + "-trans.fif", trans, overwrite=True)

        self._update_info("Saved Head→MRI transform (mm internally, m in trans.fif).")

    def _left_click_step4(self, obj, event):

        # If not SHIFT → forward to Trackball (rotate)
        if not self.left_iren.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        # Safety: wrong mode → still forward
        if self.mode != "step4":
            obj.OnLeftButtonDown()
            return

        x, y = self.left_iren.GetEventPosition()
        picker = vtk.vtkPropPicker()

        ok = picker.Pick(x, y, 0, self.ren_left)
        if not ok:
            self._update_info("No pick on left (MRI).")
            obj.OnLeftButtonDown()  # allow normal interaction
            return

        pos = np.array(picker.GetPickPosition(), dtype=float)

        for lbl in ["NAS", "LPA", "RPA"]:
            if lbl not in self.mri_fiducials:
                self.mri_fiducials[lbl] = pos

                a = sphere_actor(pos, color=FIDUCIAL_COLORS[lbl], radius=3.5)

                self.ren_left.AddActor(a)
                self.mri_fiducial_actors[lbl] = a

                self.vtk_left.GetRenderWindow().Render()

                self._update_info(f"Picked MRI fiducial {lbl} at {np.round(pos, 2)}")

                obj.OnLeftButtonDown()  # allow camera interaction
                return

        # If all fiducials already picked → still forward
        obj.OnLeftButtonDown()

    def compute_helmet_to_mri(self):
        # requires mri_fiducials and head scan fiducials transformed to master scan (from step2)
        if (
            not self.mri_fiducials
            or not self.head_fiducials
            or self.T_head_to_scan is None
        ):
            self._update_info(
                "Need MRI fiducials (step4) and head_to_scan transform (step2)."
            )
            return
        # map head scan fiducials to helmet using T_head_to_scan
        R_oh = self.T_head_to_scan[:3, :3]
        t_oh = self.T_head_to_scan[:3, 3]
        helmet_fids = {}
        for lbl, p in self.head_fiducials.items():
            helmet_fids[lbl] = R_oh @ p + t_oh
        # order NAS,LPA,RPA
        src = np.vstack([helmet_fids[l] for l in ["NAS", "LPA", "RPA"]])
        dst = np.vstack([self.mri_fiducials[l] for l in ["NAS", "LPA", "RPA"]])
        T, rms, errs = compute_rigid_transform(src, dst)
        self.T_scan_to_mri = T
        self._update_info(
            f"Computed scan_to_mri RMS={rms:.3f} mm. Applying to coil/device/head."
        )
        # apply final transforms and save
        self._apply_final_transforms(T)

    def _apply_final_transforms(self, T_head_to_mri):

        base = os.path.splitext(self.helmet_filename or "helmet")[0]

        if self.T_head_to_scan is None:
            self._update_info("Need head→scan transform first.")
            return

        # --- Compute scan → head ---
        T_scan_to_head = np.linalg.inv(self.T_head_to_scan)

        # --- Save head → mri ---
        save_transform(T_head_to_mri, base + "_head_to_mri_mm")

        # -----------------------------------------
        # Coil → MRI
        # -----------------------------------------
        if self.aux_dev_to_scan is not None:
            coil_to_mri = T_head_to_mri @ T_scan_to_head @ self.aux_dev_to_scan
            save_transform(coil_to_mri, base + "_aux_dev_to_mri_mm")

        # -----------------------------------------
        # Device → MRI
        # -----------------------------------------
        if self.T_dev_to_scan is not None:
            dev_to_mri = T_head_to_mri @ T_scan_to_head @ self.T_dev_to_scan
            save_transform(dev_to_mri, base + "_dev_to_mri_mm")

        self._update_info("Saved coil/device/head → MRI transforms (mm).")

    # ---------- utility: auto-save available ----------
    def auto_save_transform(self):

        base = os.path.splitext(self.helmet_filename or "helmet")[0]

        if self.aux_dev_to_scan is not None:
            save_transform(self.aux_dev_to_scan, base + "_aux_dev_to_scan_mm")

        if self.T_head_to_scan is not None:
            save_transform(self.T_head_to_scan, base + "_head_to_scan_mm")

        if self.T_dev_to_scan is not None:
            save_transform(self.T_dev_to_scan, base + "_dev_to_scan_mm")

        if self.T_head_to_mri is not None:
            save_transform(self.T_head_to_mri, base + "_head_to_mri_mm")

        self._update_info("Auto-saved available transforms (mm).")


# ---------- run ----------
def main():
    app = QApplication(sys.argv)
    w = AllStepsGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
