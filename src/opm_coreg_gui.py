# TODO : remove magic numbers, compile constants.
# TODO : dont do any conversions for external reasons e.g. transform scaling to mm for MNE. Set one internal unit (mm).
# TODO : helmet/scan words are used interchangeably for 3D master scan, choose one.
# TODO : rethink about the button order and button labels on each step screen.
# TODO : make display messages consistent.


import numpy as np
import vtk
from PyQt5.QtWidgets import (
    QCheckBox,
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

from src.compute_helpers import (
    compute_aux_dev_to_scan,
    compute_head_to_mri_initial,
    compute_head_to_scan,
    compute_sensors_transform,
    run_icp_head_to_mri,
)
from src.const import FIDUCIAL_COLORS, PAIR_COLORS
from src.display_helpers import (
    merge_aux_dev_into_helmet,
    overlay_helmet_sensors_transformed,
    preview_mri_on_scan,
)
from src.file_io_helpers import (
    auto_save_transform,
    load_aux_dev_landmarks,
    load_aux_dev_stl,
    load_head_scan,
    load_helmet_cad,
    load_helmet_landmark_csv,
    load_master_scan,
    load_mri_fiducials,
    load_mri_mesh,
    load_sensor_csv,
    save_head_to_mri,
)
from src.observer_helpers import (
    add_source_observer,
    add_target_observer,
    clear_source_renderer,
    clear_target_renderer_keep_scan,
    remove_all_observers,
)
from src.toggle_helpers import (
    toggle_aux_dev_overlay,
    toggle_head_overlay,
    toggle_helmet_overlay,
    toggle_mri_overlay,
    toggle_vectors,
)
from src.utils import (
    sphere_actor,
)


class OPMCoregGUI(QWidget):
    def __init__(self):
        """Initialize the main GUI window, VTK renderers, and shared state variables for the multi-step registration process."""
        super().__init__()
        self.setWindowTitle("TMS-OPM Multi-step Registration (fixed)")
        self.resize(1800, 1000)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Renderers / interactors
        self.vtk_source = QVTKRenderWindowInteractor(self)
        self.ren_source = vtk.vtkRenderer()
        self.vtk_source.GetRenderWindow().AddRenderer(self.ren_source)
        self.iren_source = self.vtk_source.GetRenderWindow().GetInteractor()
        style_source = vtk.vtkInteractorStyleTrackballCamera()
        style_source.SetMotionFactor(4.0)
        self.iren_source.SetInteractorStyle(style_source)

        self.vtk_target = QVTKRenderWindowInteractor(self)
        self.ren_target = vtk.vtkRenderer()
        self.vtk_target.GetRenderWindow().AddRenderer(self.ren_target)
        self.iren_target = self.vtk_target.GetRenderWindow().GetInteractor()
        style_target = vtk.vtkInteractorStyleTrackballCamera()
        style_target.SetMotionFactor(4.0)
        self.iren_target.SetInteractorStyle(style_target)

        # Keep track of added observer IDs so we can remove them safely
        self._observer_ids = []

        # ===== shared state (per-step) =====
        self.mode = None
        self.picking_state = {}

        # Auxiliary device alignment step
        self.helmet_polydata = None
        self.helmet_actor = None
        self.helmet_filename = None
        self.aux_dev_polydata = None
        self.aux_dev_actor_source_window = None
        self.aux_dev_landmarks = {}
        self.aux_dev_landmark_actors_source_window = []
        self.aux_dev_expected_labels = []
        self.aux_dev_pairs_src = []
        self.aux_dev_pairs_dst = []
        self.aux_dev_pick_index = 0
        self.aux_dev_to_scan = None
        self.aux_dev_actor_transformed = None

        # Head alignment step
        self.head_polydata = None
        self.head_actor = None
        self.head_fiducials = {}  # NAS,LPA,RPA
        self.head_fiducial_actors = {}
        self.head_alignment_points_from_source = []
        self.head_alignment_points_from_target = []
        self.head_alignment_pair_actors_source = []
        self.head_alignment_pair_actors_target = []
        self.T_head_to_scan = None

        # Sensors alignment step (to get the device to scan transform)
        self.helmet_stl_raw = None
        self.helmet_stl_actor_source = None
        self.helmet_stl_actor_target = None
        self.sensor_df = None
        self.sensor_positions = None
        self.sensor_axes = None
        self.sensor_actors_source_window = []
        self.helmet_landmark_positions = None
        self.helmet_landmark_actors_source_window = []
        self.helmet_landmark_text_actors_source_window = []
        self.transformed_sensor_actors = []
        self.transformed_vector_actors = []
        self.ref_point_pairs_helmet_stl = []
        self.ref_point_pairs_scan = []
        self.T_dev_to_scan = None
        self.T_dev_to_head = None

        self.show_vectors = False
        self.show_helmet_stl = True
        self.sensor_csv_path = None
        self.helmet_landmark_csv_path = None
        # Sensors alignment pairing actors (MUST exist before use)
        self.pair_actors_source = []
        self.pair_actors_target = []

        # MRI alignment step
        self.mri_polydata = None
        self.mri_fiducials = {}
        self.mri_fiducial_actors = {}
        self.T_head_to_mri = None
        self.mri_actor_target_window = None

        # Build UI
        self._build_ui()
        # initialize
        self.iren_source.Initialize()
        self.iren_target.Initialize()

    # ---------- UI ----------
    def _build_ui(self):
        """Build the main UI layout.

        Left (source) pane: 3D view (auxiliary device STL for aux alignment, head scan for head alignment, helmet STL + sensors for sensor alignment, MRI for mri alignment)
        Right (target) pane: 3D view (master scan for all steps, with overlays)
        Bottom pane: controls that change based on step, plus info and toggles for overlays
        """
        main_layout = QVBoxLayout(self)
        # two render windows
        view_layout = QHBoxLayout()
        view_layout.addWidget(self.vtk_source, 1)
        view_layout.addWidget(self.vtk_target, 1)
        main_layout.addLayout(view_layout, stretch=8)

        # bottom controls
        ctrl_layout = QHBoxLayout()
        # steps list
        steps_box = QGroupBox("Steps")
        steps_v = QVBoxLayout()
        for button_label, mode in [
            ("Step 1: Aux device to scan", "align_aux"),
            ("Step 2: Head to scan", "align_head"),
            ("Step 3: Sensors to scan", "align_sensors"),
            ("Step 4: Head to MRI", "align_mri"),
        ]:
            button = QPushButton(button_label)
            button.clicked.connect(
                lambda _, selected_mode=mode: self.set_mode(selected_mode)
            )
            steps_v.addWidget(button)
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

        self.cb_show_aux_dev = QCheckBox("Aux device")
        self.cb_show_aux_dev.setChecked(True)
        self.cb_show_aux_dev.stateChanged.connect(
            lambda state: toggle_aux_dev_overlay(self, state)
        )
        info_v.addWidget(self.cb_show_aux_dev)

        self.cb_show_head = QCheckBox("Head scan")
        self.cb_show_head.setChecked(True)
        self.cb_show_head.stateChanged.connect(
            lambda state: toggle_head_overlay(self, state)
        )
        info_v.addWidget(self.cb_show_head)
        self.cb_stl = QCheckBox("Sensors")
        self.cb_stl.setChecked(True)
        self.cb_stl.stateChanged.connect(
            lambda state: toggle_helmet_overlay(self, state)
        )
        info_v.addWidget(self.cb_stl)
        self.cb_vec = QCheckBox("Unit vectors")
        self.cb_vec.stateChanged.connect(lambda state: toggle_vectors(self, state))
        info_v.addWidget(self.cb_vec)

        self.cb_show_mri = QCheckBox("MRI")
        self.cb_show_mri.setChecked(True)
        self.cb_show_mri.stateChanged.connect(
            lambda state: toggle_mri_overlay(self, state)
        )
        info_v.addWidget(self.cb_show_mri)

        self.btn_save_all = QPushButton("Auto-save available transforms")
        self.btn_save_all.clicked.connect(lambda: auto_save_transform(self))
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
        """Update the info label at the bottom with new text."""
        print(txt)
        self.info_label.setText(txt)

    # ---------- mode switching ----------
    def set_mode(self, selected_mode):
        """Switch to a different step/mode, resetting state and UI as needed."""
        # 1. Remove old observers
        remove_all_observers(self)

        # 2. Create fresh styles with motion factor
        style_source = vtk.vtkInteractorStyleTrackballCamera()
        style_source.SetMotionFactor(4.0)

        style_target = vtk.vtkInteractorStyleTrackballCamera()
        style_target.SetMotionFactor(4.0)

        self.iren_source.SetInteractorStyle(style_source)
        self.iren_target.SetInteractorStyle(style_target)

        self.mode = selected_mode
        self.picking_state = {}

        # 3. Reset renderers
        clear_source_renderer(self)
        clear_target_renderer_keep_scan(self)

        # 4. Step setup
        if selected_mode == "align_aux":
            self._populate_aux_alignment_step_controls()

        elif selected_mode == "align_head":
            self._populate_head_alignment_step_controls()
            add_source_observer(self, "LeftButtonPressEvent", self._source_click_head_alignment)
            add_target_observer(self, "LeftButtonPressEvent", self._target_click_head_alignment)

        elif selected_mode == "align_sensors":
            self._populate_sensors_alignment_step_controls()
            add_source_observer(self, "LeftButtonPressEvent", self._source_click_sensors_alignment)
            add_target_observer(self, "LeftButtonPressEvent", self._target_click_sensors_alignment)

        elif selected_mode == "align_mri":
            self._populate_mri_alignment_step_controls()
            add_source_observer(self, "LeftButtonPressEvent", self._source_click_mri_alignment)

        self._update_info(f"Mode: {selected_mode}")

    def _clear_step_controls(self):
        """Remove all widgets from the step controls layout."""
        for i in reversed(range(self.step_ctrl_layout.count())):
            w = self.step_ctrl_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

    # ----------------- Auxiliary device alignment step -------------------
    def _populate_aux_alignment_step_controls(self):
        """Set up the controls for Aux Alignment step: auxiliary device to master scan registration."""
        self._clear_step_controls()

        # Define data structure for buttons required in the current step
        buttons = [
            {
                "label": "Load 3D master scan (target window)",
                "callback": lambda: load_master_scan(self)
            },
            {
                "label": "Load aux device STL (source window)",
                "callback": lambda: load_aux_dev_stl(self)
            },
            {
                "label": "Load aux device landmarks CSV",
                "callback": lambda: load_aux_dev_landmarks(self)
            },
            {
                "label": "Start picking aux device landmarks in master scan",
                "callback": self.start_aux_dev_landmark_pairs
            },
            {
                "label": "Undo last aux device pair",
                "callback": self.undo_last_aux_dev_pair
            },
            {
                "label": "Compute aux device to master scan transform",
                "callback": lambda: compute_aux_dev_to_scan(self)
            },
            {
                "label": "Preview aux device on master scan",
                "callback": lambda: merge_aux_dev_into_helmet(self)
            },
            {
                "label": "Restart Aux Alignment step",
                "callback": self.restart_aux_alignment
            },
        ]

        # Place all the buttons in step controls window
        for r, btn in enumerate(buttons):
            b = QPushButton(btn["label"])
            b.clicked.connect(btn["callback"])
            self.step_ctrl_layout.addWidget(b, r, 0)

    def restart_aux_alignment(self):
        """Reset all Aux Alignment step related state and UI elements."""
        try:
            if self.aux_dev_actor_source_window is not None:
                self.ren_source.RemoveActor(self.aux_dev_actor_source_window)
        except Exception:
            pass
        for actor in getattr(self, "aux_dev_landmark_actors_source_window", []):
            try:
                self.ren_source.RemoveActor(actor)
            except Exception:
                pass
        try:
            if self.aux_dev_actor_transformed is not None:
                self.ren_target.RemoveActor(self.aux_dev_actor_transformed)
        except Exception:
            pass
        self.aux_dev_polydata = None
        self.aux_dev_actor_source_window = None
        self.aux_dev_landmarks = {}
        self.aux_dev_landmark_actors_source_window = []
        self.aux_dev_pairs_src = []
        self.aux_dev_pairs_dst = []
        self.aux_dev_pick_index = 0
        self.aux_dev_to_scan = None
        self.aux_dev_actor_transformed = None
        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()
        self._update_info("Aux Alignment restarted.")

    def start_aux_dev_landmark_pairs(self):
        """Start the process of picking aux device landmark pairs on the master scan (target window) based on loaded aux device landmarks CSV."""
        if not self.aux_dev_landmarks or self.helmet_polydata is None:
            self._update_info(
                "Need aux device landmarks and master scan (participant in helmet)."
            )
            return

        ordered = [
            lab
            for lab in ["HTR", "HBR", "HBL", "HTL", "STR", "SBR", "SBL", "STL"]
            if lab in self.aux_dev_landmarks
        ]

        if not ordered:
            ordered = list(self.aux_dev_landmarks.keys())

        self.aux_dev_expected_labels = ordered
        self.aux_dev_pairs_src = []
        self.aux_dev_pairs_dst = []
        self.aux_dev_pick_index = 0

        self._update_info(
            "Pick aux device landmarks in the target window in order: "
            + ", ".join(self.aux_dev_expected_labels)
        )

        def handler(obj, event):

            # 🔹 If not SHIFT → let VTK rotate normally
            if not self.iren_target.GetShiftKey():
                obj.OnLeftButtonDown()
                return

            # 🔹 If done picking → let VTK rotate
            if self.aux_dev_pick_index >= len(self.aux_dev_expected_labels):
                obj.OnLeftButtonDown()
                return

            x, y = self.iren_target.GetEventPosition()
            picker = vtk.vtkPropPicker()

            pick_success = picker.Pick(x, y, 0, self.ren_target)
            if not pick_success:
                self._update_info("No pick on target window.")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), float)
            lab = self.aux_dev_expected_labels[self.aux_dev_pick_index]

            self.aux_dev_pairs_src.append(self.aux_dev_landmarks[lab])
            self.aux_dev_pairs_dst.append(pos)

            selected_color = PAIR_COLORS[self.aux_dev_pick_index % len(PAIR_COLORS)]

            self.ren_target.AddActor(sphere_actor(pos, color=selected_color, radius=3.5))

            self.vtk_target.GetRenderWindow().Render()

            self.aux_dev_pick_index += 1

            if self.aux_dev_pick_index >= len(self.aux_dev_expected_labels):
                try:
                    self.iren_target.RemoveObserver(obs_id)
                except Exception:
                    pass

                self._update_info("All aux device pairs picked. Click 'Compute aux→helmet'.")
            else:
                self._update_info(
                    f"Picked {lab}. Next: "
                    f"{self.aux_dev_expected_labels[self.aux_dev_pick_index]}"
                )

            # 🔹 IMPORTANT: After handling SHIFT pick,
            # still forward event so camera doesn't freeze
            obj.OnLeftButtonDown()

        obs_id = add_target_observer(self, "LeftButtonPressEvent", handler)

    def undo_last_aux_dev_pair(self):
        """Undo the last picked aux device landmark pair, removing the marker from the target renderer and updating state."""
        if not getattr(self, "aux_dev_pairs_src", []):
            self._update_info("No aux device pairs to undo.")
            return
        self.aux_dev_pairs_src.pop()
        self.aux_dev_pairs_dst.pop()
        self.aux_dev_pick_index = max(0, self.aux_dev_pick_index - 1)
        # redraw target window: helmet + placed aux device pair spheres + transformed aux device if present
        self.ren_target.RemoveAllViewProps()
        if self.helmet_actor is not None:
            self.ren_target.AddActor(self.helmet_actor)
        for i, pos in enumerate(self.aux_dev_pairs_dst):
            selected_color = PAIR_COLORS[i % len(PAIR_COLORS)]
            self.ren_target.AddActor(sphere_actor(pos, color=selected_color, radius=3.5))
        if self.aux_dev_actor_transformed is not None:
            self.ren_target.AddActor(self.aux_dev_actor_transformed)
        self.vtk_target.GetRenderWindow().Render()
        self._update_info("Undid last aux device pair.")

    # ---------------------------------------------------------------
    # ----------------- head alignment step (Outside -> Helmet) -----------------
    # ---------------------------------------------------------------

    def _populate_head_alignment_step_controls(self):
        """Set up the controls for head alignment step: head scan to master scan registration using fiducials and pairwise points."""
        self._clear_step_controls()

        # Define data structure for buttons required in the current step
        buttons = [
            {
                "label": "Load OUTSIDE head (source window)",
                "callback": lambda: load_head_scan(self)
            },
            {
                "label": "Load HELMET scan (target window)",
                "callback": lambda: load_master_scan(self)
            },
            {
                "label": "Start fiducial mode (pick NAS,LPA,RPA, Nose, Chin)",
                "callback": self.start_fiducial_mode_for_head_alignment
            },
            {
                "label": "Undo last fiducial",
                "callback": self.undo_last_fiducial_for_head_alignment
            },
            {
                "label": "Start pairwise mode (LEFT->RIGHT)",
                "callback": self.start_pairs_for_head_alignment
            },
            {
                "label": "Undo last pair",
                "callback": self.undo_last_pair_for_head_alignment
            },
            {
                "label": "Compute outside->helmet",
                "callback": lambda: compute_head_to_scan(self)
            },
            {
                "label": "Restart Head Alignment",
                "callback": self.restart_head_alignment
            },
        ]

        # Place all the buttons in step controls window
        for r, btn in enumerate(buttons):
            b = QPushButton(btn["label"])
            b.clicked.connect(btn["callback"])
            self.step_ctrl_layout.addWidget(b, r, 0)

    def start_fiducial_mode_for_head_alignment(self):
        """Enable picking fiducials on the head scan (source window) in the order NAS, LPA, RPA, Nose, Chin."""
        self.picking_state = {"mode": "head_alignment_fiducials", "next": 0}
        self._update_info(
            "Fiducial mode: SHIFT+click NAS → LPA → RPA → NOSE → CHIN (source window) (outside scan)."
        )

    def undo_last_fiducial_for_head_alignment(self):
        """Undo the last picked fiducial on the head scan (source window), removing the marker and updating state."""
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
                        self.ren_source.RemoveActor(actor)
                    except Exception:
                        pass
                del self.head_fiducials[lbl]
                if lbl in self.head_fiducial_actors:
                    del self.head_fiducial_actors[lbl]
                self.vtk_source.GetRenderWindow().Render()
                self._update_info(f"Removed fiducial {lbl}.")
                return

    def start_pairs_for_head_alignment(self):
        """Enable picking pairwise points between the head scan (source window) and master scan (target window) for head alignment step."""
        if self.head_polydata is None or self.helmet_polydata is None:
            self._update_info("Load head scan (source window) and master scan (target window) first.")
            return
        self.head_alignment_points_from_source = []
        self.head_alignment_points_from_target = []
        self._clear_head_alignment_pair_markers()
        self.picking_state = {"mode": "head_alignment_pairs", "await": "source"}
        self._update_info(
            "Pair mode: SHIFT+click source window (head scan) then SHIFT+click target window (master scan)."
        )

    def _source_click_head_alignment(self, obj, event):
        """Handle left clicks in head alignment step mode for picking fiducials and pairwise points on the head scan (source window)."""
        if not self.iren_source.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        # ---------- Fiducials ----------
        if ps.get("mode") == "head_alignment_fiducials":
            x, y = self.iren_source.GetEventPosition()
            picker = vtk.vtkPropPicker()
            pick_success = picker.Pick(x, y, 0, self.ren_source)

            if not pick_success:
                self._update_info("No pick on source window (fiducial).")
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

            self.ren_source.AddActor(actor)
            self.head_fiducial_actors[next_lbl] = actor
            self.vtk_source.GetRenderWindow().Render()

            self._update_info(f"Set fiducial {next_lbl} at {np.round(pos, 2)}")

            obj.OnLeftButtonDown()
            return

        # ---------- Pair picking ----------
        elif ps.get("mode") == "head_alignment_pairs" and ps.get("await") == "source":
            x, y = self.iren_source.GetEventPosition()
            picker = vtk.vtkPropPicker()
            pick_success = picker.Pick(x, y, 0, self.ren_source)

            if not pick_success:
                self._update_info("No pick on source window (pair).")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), dtype=float)

            ps["source_pt"] = pos
            ps["await"] = "target"
            self.picking_state = ps

            self._update_info(
                "Picked source window pair point; now pick matching point on target window."
            )

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def _target_click_head_alignment(self, obj, event):
        """Handle left clicks in head alignment step mode for picking pairwise points on the master scan (target window) that correspond to previously picked points on the head scan (source window)."""
        if not self.iren_target.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        if ps.get("mode") == "head_alignment_pairs" and ps.get("await") == "target":
            x, y = self.iren_target.GetEventPosition()
            picker = vtk.vtkPropPicker()
            pick_success = picker.Pick(x, y, 0, self.ren_target)

            if not pick_success:
                self._update_info("No pick on target window (pair).")
                obj.OnLeftButtonDown()
                return

            pos_target = np.array(picker.GetPickPosition(), dtype=float)
            pos_source = ps.get("source_pt")

            if pos_source is None:
                self._update_info("No source window point stored.")
                obj.OnLeftButtonDown()
                return

            idx = len(self.head_alignment_points_from_source)
            selected_color = PAIR_COLORS[idx % len(PAIR_COLORS)]

            self.head_alignment_points_from_source.append(pos_source)
            self.head_alignment_points_from_target.append(pos_target)

            a_l = sphere_actor(pos_source, color=selected_color, radius=3.0)
            a_r = sphere_actor(pos_target, color=selected_color, radius=3.0)

            self.ren_source.AddActor(a_l)
            self.ren_target.AddActor(a_r)

            self.head_alignment_pair_actors_source.append(a_l)
            self.head_alignment_pair_actors_target.append(a_r)

            self.vtk_source.GetRenderWindow().Render()
            self.vtk_target.GetRenderWindow().Render()

            self.picking_state = {"mode": "head_alignment_pairs", "await": "source"}

            self._update_info(f"Added head alignment pair #{idx + 1}.")

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def undo_last_pair_for_head_alignment(self):
        """Undo the last picked pairwise point between the head scan and master scan in head alignment step, removing the markers from both renderers and updating state."""
        if not self.head_alignment_points_from_source:
            self._update_info("No head alignment pairs to undo.")
            return
        self.head_alignment_points_from_source.pop()
        self.head_alignment_points_from_target.pop()
        if self.head_alignment_pair_actors_source:
            actor = self.head_alignment_pair_actors_source.pop()
            self.ren_source.RemoveActor(actor)
        if self.head_alignment_pair_actors_target:
            actor = self.head_alignment_pair_actors_target.pop()
            self.ren_target.RemoveActor(actor)
        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()
        self._update_info(
            f"Undid last head alignment pair. {len(self.head_alignment_points_from_source)} remain."
        )

    def _clear_head_alignment_pair_markers(self):
        """Remove all pairwise point markers from both renderers in head alignment step."""
        for actor in self.head_alignment_pair_actors_source:
            try:
                self.ren_source.RemoveActor(actor)
            except Exception:
                pass
        for actor in self.head_alignment_pair_actors_target:
            try:
                self.ren_target.RemoveActor(actor)
            except Exception:
                pass
        self.head_alignment_pair_actors_source = []
        self.head_alignment_pair_actors_target = []
        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()

    def restart_head_alignment(self):
        """Reset all Head Alignment related state and UI elements, including fiducials, pairwise points, and any transformed actors."""
        # remove transformed actor and pair markers
        try:
            if (
                hasattr(self, "head_actor_transformed")
                and self.head_actor_transformed is not None
            ):
                self.ren_target.RemoveActor(self.head_actor_transformed)
        except Exception:
            pass
        # remove target window fiducials
        try:
            if hasattr(self, "head_fiducial_actors_target_window"):
                for actor in self.head_fiducial_actors_target_window.values():
                    try:
                        self.ren_target.RemoveActor(actor)
                    except Exception:
                        pass
                self.head_fiducial_actors_target_window = {}
        except Exception:
            pass
        # remove source window fiducials
        for actor in self.head_fiducial_actors.values():
            try:
                self.ren_source.RemoveActor(actor)
            except Exception:
                pass
        self.head_fiducials = {}
        self.head_fiducial_actors = {}
        self._clear_head_alignment_pair_markers()
        self.head_alignment_points_from_source = []
        self.head_alignment_points_from_target = []
        self.T_head_to_scan = None
        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()
        self._update_info("Restarted Head Alignment (fiducials & pairs cleared).")

    # -------------------------------------------------------------------
    # ------------------ sensor alignment step (Sensors & STL -> scan) -----------------
    # -------------------------------------------------------------------
    def _populate_sensors_alignment_step_controls(self):
        """Set up the controls for sensor alignment step: helmet STL and sensor positions to master scan registration using pairwise points."""
        self._clear_step_controls()
        
        # Define data structure for buttons required in the current step
        buttons = [
            {
                "label": "Load helmet STL (source window)",
                "callback": lambda: load_helmet_cad(self)
            },
            {
                "label": "Load sensor locations CSV (plot on source window)",
                "callback": lambda: load_sensor_csv(self)
            },
            {
                "label": "Load helmet landmarks CSV (plot on source window) (Optional)",
                "callback": lambda: load_helmet_landmark_csv(self)
            },
            {
                "label": "Load master scan (target window)",
                "callback": lambda: load_master_scan(self)
            },
            {
                "label": "Start helmet to master scan pair mode (LEFT->RIGHT)",
                "callback": self.start_pairs_for_sensor_alignment
            },
            {
                "label": "Undo last pair",
                "callback": self.undo_last_pair_from_sensor_alignment_step
            },
            {
                "label": "Compute transform & save sensors",
                "callback": lambda: compute_sensors_transform(self)
            },
            {
                "label": "Restart Sensor Alignment",
                "callback": self.restart_sensors_alignment
            },
        ]

        # Place all the buttons in step controls window
        for r, btn in enumerate(buttons):
            b = QPushButton(btn["label"])
            b.clicked.connect(btn["callback"])
            self.step_ctrl_layout.addWidget(b, r, 0)

    def restart_sensors_alignment(self):
        """Reset all Sensor Alignment related state and UI elements, including helmet STL, sensor positions, pairwise points, and any transformed actors."""
        # remove actors (source window/target window)
        try:
            if self.helmet_stl_actor_source is not None:
                self.ren_source.RemoveActor(self.helmet_stl_actor_source)
        except Exception:
            pass
        try:
            if self.helmet_stl_actor_target is not None:
                self.ren_target.RemoveActor(self.helmet_stl_actor_target)
        except Exception:
            pass
        self.helmet_stl_raw = None
        self.helmet_stl_actor_source = None
        self.helmet_stl_actor_target = None
        # sensors
        for actor in (
            self.sensor_actors_source_window
            + self.transformed_sensor_actors
            + self.transformed_vector_actors
            + self.helmet_landmark_actors_source_window
            + self.helmet_landmark_text_actors_source_window
        ):
            try:
                self.ren_source.RemoveActor(actor)
                self.ren_target.RemoveActor(actor)
            except Exception:
                pass
        self.sensor_positions = None
        self.sensor_axes = None
        self.sensor_df = None
        self.sensor_csv_path = None
        self.sensor_actors_source_window = []
        self.helmet_landmark_positions = None
        self.helmet_landmark_actors_source_window = []
        self.helmet_landmark_text_actors_source_window = []
        self.helmet_landmark_csv_path = None
        self.transformed_sensor_actors = []
        self.transformed_vector_actors = []
        # pairs
        self.ref_point_pairs_helmet_stl = []
        self.ref_point_pairs_scan = []
        self.pair_actors_source = []
        self.pair_actors_target = []
        self.T_dev_to_scan = None
        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()
        self._update_info("Restarted Sensor Alignment.")

    def start_pairs_for_sensor_alignment(self):
        """Enable picking pairwise points between the helmet STL/helmet landmark list (source window) and master scan (target window) for sensor alignment.

        Requires that both the helmet STL (or helmet landmark CSV) and master scan are loaded first."""
        if self.helmet_polydata is None or (self.helmet_stl_raw is None and self.helmet_landmark_positions is None):
            self._update_info(
                "Need helmet STL/helmet ladmark list (source window) and participant in the helmet scan (target window) loaded."
            )
            return
        self.ref_point_pairs_helmet_stl = []
        self.ref_point_pairs_scan = []
        self._clear_sensor_alignment_pair_markers()
        self.picking_state = {"mode": "sensor_alignment_pairs", "await": "source"}
        self._update_info(
            "sensor alignment step: SHIFT+click helmet STL on source window then matching point on target window (scan)."
        )

    def _source_click_sensors_alignment(self, obj, event):
        """Handle left clicks in sensor alignment step mode for picking points on the helmet STL (source window) that correspond to points on the master scan (target window) for device registration."""
        if not self.iren_source.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        if ps.get("mode") == "sensor_alignment_pairs" and ps.get("await") == "source":
            x, y = self.iren_source.GetEventPosition()
            picker = vtk.vtkPropPicker()

            pick_success = picker.Pick(x, y, 0, self.ren_source)
            if not pick_success:
                self._update_info("No pick on source window (STL).")
                obj.OnLeftButtonDown()
                return

            pos = np.array(picker.GetPickPosition(), dtype=float)

            self.picking_state["source_pt"] = pos
            self.picking_state["await"] = "target"

            self._update_info(
                "Selected point on the helmet; now SHIFT+click matching point on target window."
            )

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def _target_click_sensors_alignment(self, obj, event):
        """Handle left clicks in sensor alignment step mode for picking points on the master scan (target window) that correspond to previously picked points on the helmet STL (source window) for device registration."""
        if not self.iren_target.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        ps = self.picking_state or {}

        if ps.get("mode") == "sensor_alignment_pairs" and ps.get("await") == "target":
            x, y = self.iren_target.GetEventPosition()
            picker = vtk.vtkPropPicker()

            pick_success = picker.Pick(x, y, 0, self.ren_target)
            if not pick_success:
                self._update_info("No pick on target window (scan).")
                obj.OnLeftButtonDown()
                return

            pos_target = np.array(picker.GetPickPosition(), dtype=float)
            pos_source = ps.get("source_pt")

            if pos_source is None:
                self._update_info("No source window point stored.")
                obj.OnLeftButtonDown()
                return

            self.ref_point_pairs_helmet_stl.append(pos_source)
            self.ref_point_pairs_scan.append(pos_target)

            idx = len(self.ref_point_pairs_helmet_stl) - 1
            selected_color = PAIR_COLORS[idx % len(PAIR_COLORS)]

            a_l = sphere_actor(pos_source, color=selected_color, radius=3.0)
            a_r = sphere_actor(pos_target, color=selected_color, radius=3.0)

            self.ren_source.AddActor(a_l)
            self.ren_target.AddActor(a_r)

            self.pair_actors_source.append(a_l)
            self.pair_actors_target.append(a_r)

            self.vtk_source.GetRenderWindow().Render()
            self.vtk_target.GetRenderWindow().Render()

            self.picking_state = {"mode": "sensor_alignment_pairs", "await": "source"}

            self._update_info(f"Added sensor alignment pair #{idx + 1}.")

            obj.OnLeftButtonDown()
            return

        obj.OnLeftButtonDown()

    def undo_last_pair_from_sensor_alignment_step(self):
        """Undo the last picked pairwise point between the helmet STL and master scan in sensor alignment step, removing the markers from both renderers and updating state."""
        if not self.ref_point_pairs_helmet_stl:
            self._update_info("No sensor alignment pairs to undo.")
            return
        self.ref_point_pairs_helmet_stl.pop()
        self.ref_point_pairs_scan.pop()
        if self.pair_actors_source:
            actor = self.pair_actors_source.pop()
            self.ren_source.RemoveActor(actor)
        if self.pair_actors_target:
            actor = self.pair_actors_target.pop()
            self.ren_target.RemoveActor(actor)
        # re-render and re-add transformed if present
        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()
        if self.T_dev_to_scan is not None:
            overlay_helmet_sensors_transformed(self, self.T_dev_to_scan)
        self._update_info("Undid last sensor alignment pair.")

    def _clear_sensor_alignment_pair_markers(self):
        """Remove all pairwise point markers from both renderers in sensor alignment step."""
        if hasattr(self, "pair_actors_source"):
            for actor in self.pair_actors_source:
                try:
                    self.ren_source.RemoveActor(actor)
                except Exception:
                    pass

        if hasattr(self, "pair_actors_target"):
            for actor in self.pair_actors_target:
                try:
                    self.ren_target.RemoveActor(actor)
                except Exception:
                    pass

        self.pair_actors_source = []
        self.pair_actors_target = []

        self.vtk_source.GetRenderWindow().Render()
        self.vtk_target.GetRenderWindow().Render()

    # ---------------------------------------------------------
    # ----------------- mri alignment step (head -> MRI) -----------------
    # ---------------------------------------------------------
    def _populate_mri_alignment_step_controls(self):
        """Set up the controls for mri alignment step: head scan to MRI registration using fiducials and ICP refinement, with options to load MRI data and preview results."""
        self._clear_step_controls()

        # Define data structure for buttons required in the current step
        buttons = [
            {
                "label": "Load MRI mesh (source window)",
                "callback": lambda: load_mri_mesh(self)
            },
            {
                "label": "Load MRI fiducials (.fif)",
                "callback": lambda: load_mri_fiducials(self)
            },
            {
                "label": "Compute initial Head→MRI (base on head alignment step fiducials)",
                "callback": lambda: compute_head_to_mri_initial(self)
            },
            {
                "label": "Run ICP refinement",
                "callback": lambda: run_icp_head_to_mri(self)
            },
            {
                "label": "Preview MRI on master scan (target window)",
                "callback": lambda: preview_mri_on_scan(self)
            },
            {
                "label": "Save Head→MRI",
                "callback": lambda: save_head_to_mri(self)
            },
        ]

        # Place all the buttons in step controls window
        for r, btn in enumerate(buttons):
            b = QPushButton(btn["label"])
            b.clicked.connect(btn["callback"])
            self.step_ctrl_layout.addWidget(b, r, 0)

    def _source_click_mri_alignment(self, obj, event):
        """Handle left clicks in mri alignment step mode for picking fiducials on the MRI mesh (source window) that correspond to head scan fiducials."""
        # If not SHIFT → forward to Trackball (rotate)
        if not self.iren_source.GetShiftKey():
            obj.OnLeftButtonDown()
            return

        # Safety: wrong mode → still forward
        if self.mode != "align_mri":
            obj.OnLeftButtonDown()
            return

        x, y = self.iren_source.GetEventPosition()
        picker = vtk.vtkPropPicker()

        pick_success = picker.Pick(x, y, 0, self.ren_source)
        if not pick_success:
            self._update_info("No pick on source window (MRI).")
            obj.OnLeftButtonDown()  # allow normal interaction
            return

        pos = np.array(picker.GetPickPosition(), dtype=float)

        for lbl in ["NAS", "LPA", "RPA"]:
            if lbl not in self.mri_fiducials:
                self.mri_fiducials[lbl] = pos

                actor = sphere_actor(pos, color=FIDUCIAL_COLORS[lbl], radius=3.5)

                self.ren_source.AddActor(actor)
                self.mri_fiducial_actors[lbl] = actor

                self.vtk_source.GetRenderWindow().Render()

                self._update_info(f"Picked MRI fiducial {lbl} at {np.round(pos, 2)}")

                obj.OnLeftButtonDown()  # allow camera interaction
                return

        # If all fiducials already picked → still forward
        obj.OnLeftButtonDown()
