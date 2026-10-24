"""
gui_app.py
Redesigned Engineering Desktop Application for IMU Gesture Recognition & PowerPoint Control

Academic Project:
"Machine Learning-Based Hand Gesture Recognition Using a Wearable IMU Sensor for PowerPoint Control"
Supervisor: Dr. Khalid Ali Abbas
University of Basrah · College of Engineering · Computer Engineering Department

Phase 2 Engineering Upgrades:
1. One Intentional Gesture -> One Command (GestureStateMachine with neutral hysteresis and dwell time).
2. Holding a tilted posture CANNOT repeatedly trigger commands (Awaiting Neutral return).
3. Multi-factor motion quality and class margin rejection (MotionQualityRejector).
4. User Setup & Guided Profile Personalization Workflow (ParticipantProfileManager).
5. Reproducible Evaluation & Latency Breakdown (EvaluationEngine).
6. Model Bundle Compatibility (ModelBundle with metadata inspection and legacy support).
7. Decoupled Concepts: Prediction -> Mapped Command -> Eligibility -> Dispatch Result.
8. Strictly bounds replay counters (119/119, no off-by-one errors) and safe replay restart.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, Dict, List, Any, Tuple

import numpy as np

from src.theme import (
    apply_dark_theme, BG_DARK, PANEL_BG, SURFACE_BG, SURFACE_HOVER, BORDER_COLOR,
    ACCENT_CYAN, ACCENT_TEAL, TEXT_PRIMARY, TEXT_MUTED,
    AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, COLOR_NEUTRAL,
    FONT_FAMILY, FONT_MONO
)
from src.gesture_schema import CANONICAL_CLASSES, DEFAULT_POWERPOINT_COMMANDS, get_command_for_gesture
from src.data_pipeline import ZScoreNormalizer, FEATURE_COLS, WINDOW_SIZE
from src.replay_source import CsvReplaySource, ReplayTrial
from src.recognition_engine import GestureRecognitionEngine, RecognitionResult
from src.state_machine import GestureState, DecisionReason
from src.ui_widgets import WaveformCanvas, GestureIconCanvas, SlidePreviewWidget, Tooltip
from src.compatibility import ModelBundle, check_acquisition_model_compatibility
from src.profile_manager import ParticipantProfile, GuidedSetupWorkflow, NeutralReference
from src.evaluation import EvaluationEngine, RuleBasedBaseline


class ImuGestureControllerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("IMU Gesture Controller — University of Basrah")
        self.root.geometry("1200x840")
        self.root.minsize(1040, 720)

        # Thread-safe message queue
        self.ui_queue = queue.Queue()

        # Engine & Source instances
        self.engine: Optional[GestureRecognitionEngine] = None
        self.normalizer: Optional[ZScoreNormalizer] = None
        self.replay_source = CsvReplaySource()
        self.active_bundle: Optional[ModelBundle] = None

        # Guided Setup instance
        self.setup_workflow = GuidedSetupWorkflow()

        # Default paths
        self.bundle_dir_var = tk.StringVar(value=os.path.join(REPO_ROOT, "models"))
        self.cnn_path = os.path.join(REPO_ROOT, "models", "imu_gesture_cnn.keras")
        self.mlp_path = os.path.join(REPO_ROOT, "models", "imu_gesture_mlp.pkl")
        self.labels_path = os.path.join(REPO_ROOT, "models", "label_classes.npy")
        self.mean_path = os.path.join(REPO_ROOT, "models", "feature_mean.npy")
        self.std_path = os.path.join(REPO_ROOT, "models", "feature_std.npy")

        # Serial state
        self.ser = None
        self.serial_thread = None
        self.serial_stop_event = threading.Event()

        # Apply dark engineering instrument design system
        self.style = apply_dark_theme(self.root)

        # Build interface
        self._build_header()
        self._build_navigation_tabs()
        self._init_pipeline()
        self._poll_ui_queue()

        # Clean shutdown handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # =========================================================================
    # 1. Project Header & Identity
    # =========================================================================
    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        header.pack(fill="x")

        # Left: Identity & Academic Title
        left_box = tk.Frame(header, bg=BG_DARK)
        left_box.pack(side="left", fill="y")

        title_row = tk.Frame(left_box, bg=BG_DARK)
        title_row.pack(anchor="w")

        tk.Label(
            title_row,
            text="IMU GESTURE CONTROLLER",
            font=(FONT_FAMILY, 14, "bold"),
            bg=BG_DARK,
            fg=TEXT_PRIMARY
        ).pack(side="left")

        tk.Label(
            title_row,
            text="· v2.1 Advanced Engineering Instrumentation",
            font=(FONT_FAMILY, 9),
            bg=BG_DARK,
            fg=ACCENT_CYAN
        ).pack(side="left", padx=8)

        # Full academic project title
        tk.Label(
            left_box,
            text="Machine Learning-Based Hand Gesture Recognition Using a Wearable IMU Sensor for PowerPoint Control",
            font=(FONT_FAMILY, 9),
            bg=BG_DARK,
            fg=TEXT_PRIMARY
        ).pack(anchor="w", pady=(2, 1))

        # Academic credit
        credit_text = "Supervisor: Dr. Khalid Ali Abbas  ·  University of Basrah · College of Engineering · Computer Engineering Department"
        tk.Label(
            left_box,
            text=credit_text,
            font=(FONT_FAMILY, 8),
            bg=BG_DARK,
            fg=TEXT_MUTED
        ).pack(anchor="w")

        # Right: Operational Badges
        right_box = tk.Frame(header, bg=BG_DARK)
        right_box.pack(side="right", fill="y", padx=4)

        badges_frame = tk.Frame(right_box, bg=BG_DARK)
        badges_frame.pack(anchor="e", pady=(4, 0))

        # State Machine Badge
        self.lbl_sm_badge = tk.Label(
            badges_frame,
            text="SM: READY",
            font=(FONT_MONO, 8, "bold"),
            bg="#1E293B",
            fg=ACCENT_TEAL,
            padx=8,
            pady=3,
            relief="solid",
            borderwidth=1
        )
        self.lbl_sm_badge.pack(side="left", padx=3)
        Tooltip(self.lbl_sm_badge, "Intent State Machine: READY -> CAPTURING -> WAITING_FOR_NEUTRAL")

        # Mode Badge
        self.lbl_mode_badge = tk.Label(
            badges_frame,
            text="REPLAY MODE",
            font=(FONT_FAMILY, 8, "bold"),
            bg="#0E2A47",
            fg=ACCENT_CYAN,
            padx=8,
            pady=3,
            relief="solid",
            borderwidth=1
        )
        self.lbl_mode_badge.pack(side="left", padx=3)
        Tooltip(self.lbl_mode_badge, "Current signal source: CSV Replay")

        # Dispatch Safety Badge
        self.lbl_safety_badge = tk.Label(
            badges_frame,
            text="DRY RUN (ACTIVE)",
            font=(FONT_FAMILY, 8, "bold"),
            bg="#064E3B",
            fg=COLOR_SUCCESS,
            padx=8,
            pady=3,
            relief="solid",
            borderwidth=1
        )
        self.lbl_safety_badge.pack(side="left", padx=3)
        Tooltip(self.lbl_safety_badge, "Keystroke commands are simulated safely. No OS keys sent.")

        # Engine Status Badge
        self.lbl_engine_badge = tk.Label(
            badges_frame,
            text="STANDBY",
            font=(FONT_FAMILY, 8, "bold"),
            bg="#1E293B",
            fg=TEXT_MUTED,
            padx=8,
            pady=3,
            relief="solid",
            borderwidth=1
        )
        self.lbl_engine_badge.pack(side="left", padx=3)

    # =========================================================================
    # 2. Compact Navigation Tabs
    # =========================================================================
    def _build_navigation_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        # 5 Functional Views
        self.tab_overview = ttk.Frame(self.notebook, style="TFrame")
        self.tab_data = ttk.Frame(self.notebook, style="TFrame")
        self.tab_setup = ttk.Frame(self.notebook, style="TFrame")
        self.tab_settings = ttk.Frame(self.notebook, style="TFrame")
        self.tab_info = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.tab_overview, text="  1. Overview & Live Demo  ")
        self.notebook.add(self.tab_data, text="  2. Data & Replay  ")
        self.notebook.add(self.tab_setup, text="  3. User Setup & Guided Profiles  ")
        self.notebook.add(self.tab_settings, text="  4. Settings & Model Bundles  ")
        self.notebook.add(self.tab_info, text="  5. Project Documentation  ")

        self._build_overview_tab()
        self._build_data_tab()
        self._build_setup_tab()
        self._build_settings_tab()
        self._build_info_tab()

    # =========================================================================
    # TAB 1: Overview & Live Demonstration
    # =========================================================================
    def _build_overview_tab(self):
        left_col = tk.Frame(self.tab_overview, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 6))

        right_col = tk.Frame(self.tab_overview, bg=BG_DARK)
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0))

        # --- LEFT COLUMN: RECOGNITION CARD & SLIDE PREVIEW ---
        rec_card = tk.Frame(left_col, bg=PANEL_BG, padx=14, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        rec_card.pack(fill="x", pady=(0, 8))

        rec_header = tk.Frame(rec_card, bg=PANEL_BG)
        rec_header.pack(fill="x")
        tk.Label(rec_header, text="GESTURE RECOGNITION STATUS", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=ACCENT_CYAN).pack(side="left")

        self.lbl_rec_status = tk.Label(rec_header, text="Ready", font=(FONT_FAMILY, 8, "bold"), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_rec_status.pack(side="right")

        # Main gesture display row
        g_row = tk.Frame(rec_card, bg=PANEL_BG)
        g_row.pack(fill="x", pady=(10, 8))

        self.icon_canvas = GestureIconCanvas(g_row, size=52)
        self.icon_canvas.pack(side="left", padx=(0, 12))

        g_info = tk.Frame(g_row, bg=PANEL_BG)
        g_info.pack(side="left", fill="x", expand=True)

        self.lbl_predicted = tk.Label(g_info, text="WAITING", font=(FONT_FAMILY, 18, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY)
        self.lbl_predicted.pack(anchor="w")

        self.lbl_conf = tk.Label(g_info, text="Model Confidence: 0.0%", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_conf.pack(anchor="w")

        # Confidence bar
        self.conf_bar = ttk.Progressbar(rec_card, orient="horizontal", mode="determinate", length=220)
        self.conf_bar.pack(fill="x", pady=(2, 8))

        # Metrics grid
        grid_f = tk.Frame(rec_card, bg=PANEL_BG)
        grid_f.pack(fill="x")

        tk.Label(grid_f, text="Expected Gesture:", font=(FONT_FAMILY, 8), bg=PANEL_BG, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_expected = tk.Label(grid_f, text="None (Replay idle)", font=(FONT_FAMILY, 8, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY)
        self.lbl_expected.grid(row=0, column=1, sticky="w", padx=8, pady=2)

        tk.Label(grid_f, text="Mapped Command:", font=(FONT_FAMILY, 8), bg=PANEL_BG, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_cmd = tk.Label(grid_f, text="None", font=(FONT_FAMILY, 8, "bold"), bg=PANEL_BG, fg=ACCENT_CYAN)
        self.lbl_cmd.grid(row=1, column=1, sticky="w", padx=8, pady=2)

        tk.Label(grid_f, text="Dispatch Status:", font=(FONT_FAMILY, 8), bg=PANEL_BG, fg=TEXT_MUTED).grid(row=2, column=0, sticky="w", pady=2)
        self.lbl_dispatch = tk.Label(grid_f, text="Idle", font=(FONT_FAMILY, 8, "bold"), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_dispatch.grid(row=2, column=1, sticky="w", padx=8, pady=2)

        tk.Label(grid_f, text="Intent State Machine:", font=(FONT_FAMILY, 8), bg=PANEL_BG, fg=TEXT_MUTED).grid(row=3, column=0, sticky="w", pady=2)
        self.lbl_sm_detail = tk.Label(grid_f, text="READY (Re-armed)", font=(FONT_FAMILY, 8), bg=PANEL_BG, fg=ACCENT_TEAL)
        self.lbl_sm_detail.grid(row=3, column=1, sticky="w", padx=8, pady=2)

        # 2. Built-in Presentation Preview Deck
        self.slide_preview = SlidePreviewWidget(left_col, height=215)
        self.slide_preview.pack(fill="both", expand=True)

        # --- RIGHT COLUMN: SENSOR PLOTS, REPLAY CONTROLS, AND TABULAR LOG ---
        self.accel_canvas = WaveformCanvas(right_col, title="ACCELEROMETER", unit="g", y_min=-2.0, y_max=2.0, height=120)
        self.accel_canvas.pack(fill="x", pady=(0, 6))

        self.gyro_canvas = WaveformCanvas(right_col, title="GYROSCOPE", unit="deg/s", y_min=-250.0, y_max=250.0, height=120)
        self.gyro_canvas.pack(fill="x", pady=(0, 6))

        tk.Label(
            right_col,
            text="* Dual vector waveforms display current 51-sample window (~1.0s). X=Rose, Y=Emerald, Z=Sky.",
            font=(FONT_FAMILY, 7, "italic"),
            bg=BG_DARK,
            fg=TEXT_MUTED
        ).pack(anchor="w", pady=(0, 4))

        # Replay Quick Bar
        quick_ctrl = tk.Frame(right_col, bg=PANEL_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        quick_ctrl.pack(fill="x", pady=(0, 8))

        ttk.Button(quick_ctrl, text="▶ Play", width=7, command=self._start_replay).pack(side="left", padx=2)
        ttk.Button(quick_ctrl, text="⏸ Pause", width=7, command=self._pause_replay).pack(side="left", padx=2)
        ttk.Button(quick_ctrl, text="⏭ Step", width=7, command=self._step_replay).pack(side="left", padx=2)
        ttk.Button(quick_ctrl, text="⏹ Stop/Reset", width=10, command=self._stop_replay).pack(side="left", padx=2)

        ttk.Label(quick_ctrl, text="Speed:", style="Panel.TLabel").pack(side="left", padx=(10, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_slider = ttk.Scale(quick_ctrl, from_=0.2, to=3.0, variable=self.speed_var, command=self._on_speed_change, length=80)
        self.speed_slider.pack(side="left", padx=2)
        self.lbl_speed_val = ttk.Label(quick_ctrl, text="1.0x", style="PanelMuted.TLabel", width=4)
        self.lbl_speed_val.pack(side="left")

        self.lbl_progress = tk.Label(quick_ctrl, text="Trial 0 / 0", font=(FONT_MONO, 8, "bold"), bg=PANEL_BG, fg=ACCENT_CYAN)
        self.lbl_progress.pack(side="right", padx=4)

        # Tabular Event Log
        log_frame = tk.Frame(right_col, bg=PANEL_BG, padx=10, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        log_frame.pack(fill="both", expand=True)

        log_hdr = tk.Frame(log_frame, bg=PANEL_BG)
        log_hdr.pack(fill="x", pady=(0, 4))
        tk.Label(log_hdr, text="EVENT AUDIT LOG", font=(FONT_FAMILY, 8, "bold"), bg=PANEL_BG, fg=ACCENT_CYAN).pack(side="left")
        ttk.Button(log_hdr, text="Clear Log", command=self._clear_log).pack(side="right")

        self.overview_log = scrolledtext.ScrolledText(
            log_frame,
            height=6,
            font=(FONT_MONO, 8),
            bg="#0F172A",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            borderwidth=0
        )
        self.overview_log.pack(fill="both", expand=True)
        self.overview_log.insert(tk.END, f"{'TIME':<8} | {'TRIAL':<10} | {'EXPECTED':<8} | {'PREDICTED':<9} | {'CONF':<6} | {'ACTION / DISPATCH'}\n")
        self.overview_log.insert(tk.END, "-" * 80 + "\n")

    # =========================================================================
    # TAB 2: Data & Replay Management
    # =========================================================================
    def _build_data_tab(self):
        pad = tk.Frame(self.tab_data, bg=BG_DARK, padx=12, pady=10)
        pad.pack(fill="both", expand=True)

        src_bar = tk.Frame(pad, bg=PANEL_BG, padx=12, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        src_bar.pack(fill="x", pady=(0, 10))

        tk.Label(src_bar, text="Dataset Directory or CSV:", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=2)
        self.data_path_var = tk.StringVar(value=os.path.join(REPO_ROOT, "data", "raw_new"))
        self.ent_path = ttk.Entry(src_bar, textvariable=self.data_path_var, width=54)
        self.ent_path.grid(row=0, column=1, padx=6, sticky="ew")

        ttk.Button(src_bar, text="Browse Folder", command=self._browse_folder).grid(row=0, column=2, padx=2)
        ttk.Button(src_bar, text="Browse File", command=self._browse_file).grid(row=0, column=3, padx=2)
        ttk.Button(src_bar, text="Load Data", style="Accent.TButton", command=self._load_replay_dataset).grid(row=0, column=4, padx=6)

        stat_bar = tk.Frame(pad, bg=SURFACE_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        stat_bar.pack(fill="x", pady=(0, 10))

        self.lbl_data_stats = tk.Label(
            stat_bar,
            text="Dataset: Not loaded | Valid Trials: 0 | Quarantined: 0",
            font=(FONT_FAMILY, 9),
            bg=SURFACE_BG,
            fg=TEXT_PRIMARY
        )
        self.lbl_data_stats.pack(side="left")

        ttk.Button(stat_bar, text="Comprehensive Reproducible Evaluation", command=self._run_reproducible_evaluation).pack(side="right")

        sub_nb = ttk.Notebook(pad)
        sub_nb.pack(fill="both", expand=True)

        # Loaded Trials Inventory
        tab_trials = ttk.Frame(sub_nb, style="TFrame")
        sub_nb.add(tab_trials, text=" Loaded Trials Inventory ")

        cols = ("idx", "key", "gesture_raw", "gesture_canon", "samples", "status")
        self.tree_trials = ttk.Treeview(tab_trials, columns=cols, show="headings", height=14)
        self.tree_trials.heading("idx", text="#")
        self.tree_trials.heading("key", text="Trial Key")
        self.tree_trials.heading("gesture_raw", text="Raw Label")
        self.tree_trials.heading("gesture_canon", text="Canonical Label")
        self.tree_trials.heading("samples", text="Rows")
        self.tree_trials.heading("status", text="Validity Status")

        self.tree_trials.column("idx", width=45, anchor="c")
        self.tree_trials.column("key", width=240, anchor="w")
        self.tree_trials.column("gesture_raw", width=110, anchor="w")
        self.tree_trials.column("gesture_canon", width=110, anchor="w")
        self.tree_trials.column("samples", width=70, anchor="c")
        self.tree_trials.column("status", width=220, anchor="w")

        tree_scroll = ttk.Scrollbar(tab_trials, orient="vertical", command=self.tree_trials.yview)
        self.tree_trials.configure(yscrollcommand=tree_scroll.set)
        self.tree_trials.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # Quarantine Inspector
        tab_quarantine = ttk.Frame(sub_nb, style="TFrame")
        sub_nb.add(tab_quarantine, text=" Quarantine Inspector (Anomalous Trials) ")

        self.txt_quarantine = scrolledtext.ScrolledText(
            tab_quarantine,
            font=(FONT_MONO, 9),
            bg="#0F172A",
            fg=TEXT_PRIMARY,
            wrap="word",
            padx=10,
            pady=8
        )
        self.txt_quarantine.pack(fill="both", expand=True)

    # =========================================================================
    # TAB 3: User Setup & Guided Profiles
    # =========================================================================
    def _build_setup_tab(self):
        pad = tk.Frame(self.tab_setup, bg=BG_DARK, padx=16, pady=12)
        pad.pack(fill="both", expand=True)

        # Profile Configuration Frame
        p_frame = ttk.LabelFrame(pad, text="Participant Profile & Acquisition Settings", style="Panel.TLabelframe", padding=12)
        p_frame.pack(fill="x", pady=(0, 10))

        row1 = tk.Frame(p_frame, bg=PANEL_BG)
        row1.pack(fill="x", pady=4)

        tk.Label(row1, text="Participant ID:", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 4))
        self.part_id_var = tk.StringVar(value="P01")
        ttk.Entry(row1, textvariable=self.part_id_var, width=10).pack(side="left", padx=(0, 16))

        tk.Label(row1, text="Session ID:", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 4))
        self.sess_id_var = tk.StringVar(value="S01")
        ttk.Entry(row1, textvariable=self.sess_id_var, width=8).pack(side="left", padx=(0, 16))

        tk.Label(row1, text="Sensor Placement:", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 4))
        self.place_var = tk.StringVar(value="dorsal_wrist")
        ttk.Combobox(row1, textvariable=self.place_var, values=["dorsal_wrist", "palmar_wrist"], width=14, state="readonly").pack(side="left", padx=(0, 16))

        tk.Label(row1, text="Hand:", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 4))
        self.hand_var = tk.StringVar(value="dominant_right")
        ttk.Combobox(row1, textvariable=self.hand_var, values=["dominant_right", "dominant_left", "non_dominant"], width=14, state="readonly").pack(side="left")

        # Guided Calibration & Acquisition Card
        cal_frame = ttk.LabelFrame(pad, text="Guided Calibration & Gesture Recording Workflow", style="Panel.TLabelframe", padding=12)
        cal_frame.pack(fill="x", pady=(0, 10))

        c_row = tk.Frame(cal_frame, bg=PANEL_BG)
        c_row.pack(fill="x", pady=4)

        ttk.Button(c_row, text="1. Capture Neutral Posture Reference", command=self._capture_neutral_reference).pack(side="left", padx=(0, 10))
        self.lbl_neutral_status = tk.Label(c_row, text="Neutral Reference: Not recorded (Using factory gravity [0, 0, 1g])", font=(FONT_FAMILY, 8, "italic"), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_neutral_status.pack(side="left")

        g_row = tk.Frame(cal_frame, bg=PANEL_BG)
        g_row.pack(fill="x", pady=(10, 4))

        tk.Label(g_row, text="Guided Collection Step:", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=ACCENT_CYAN).pack(side="left", padx=(0, 8))
        self.lbl_guided_step = tk.Label(g_row, text="Ready. Click 'Start Guided Protocol' below.", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY)
        self.lbl_guided_step.pack(side="left")

        btn_row = tk.Frame(cal_frame, bg=PANEL_BG)
        btn_row.pack(fill="x", pady=(8, 4))

        ttk.Button(btn_row, text="▶ Start Guided Protocol", style="Accent.TButton", command=self._start_guided_protocol).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Record Next Repetition (Mock / Stream)", command=self._record_guided_rep).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Run Full Offline Mock Protocol", command=self._run_full_mock_protocol).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Save Profile & Dataset", command=self._save_profile_session).pack(side="left")

        # Guidance Documentation Box
        txt_guide = scrolledtext.ScrolledText(pad, font=(FONT_FAMILY, 9), bg="#0F172A", fg=TEXT_PRIMARY, height=9, padx=10, pady=8)
        txt_guide.pack(fill="both", expand=True)
        txt_guide.insert("1.0", """=== GUIDED SETUP & PERSONALIZATION PROTOCOL ===

1. NEUTRAL POSTURE REFERENCE:
   Instruct the user to sit or stand comfortably with their forearm in a relaxed presentation posture.
   The sensor measures resting gravitational tilt [ax, ay, az] and zero-rate gyroscope bias.
   Re-arming the state machine requires the user to return to this posture for >= 0.3s after gesturing.

2. GUIDED GESTURE REPETITIONS:
   Guides the user sequentially through 5 repetitions for each canonical gesture:
   - still (resting neutral baseline)
   - left (wrist tilt left / previous slide)
   - right (wrist tilt right / next slide)
   - up (wrist tilt up / start presentation)
   - down (wrist tilt down / blank screen)
   - shake (rapid horizontal agitation / exit presentation)

3. NON-DESTRUCTIVE STORAGE:
   Recordings are saved safely into 'data/profiles/<participant_id>/' alongside profile metadata JSON.
   Raw benchmark datasets in 'data/raw_new/' are strictly preserved.
""")
        txt_guide.config(state="disabled")

    # =========================================================================
    # TAB 4: Settings & Model Bundles
    # =========================================================================
    def _build_settings_tab(self):
        pad = tk.Frame(self.tab_settings, bg=BG_DARK, padx=12, pady=10)
        pad.pack(fill="both", expand=True)

        # 1. Model Bundle Configuration Box
        m_frame = ttk.LabelFrame(pad, text="Machine Learning Model Bundle & Metadata", style="Panel.TLabelframe", padding=12)
        m_frame.pack(fill="x", pady=(0, 10))

        b_row = tk.Frame(m_frame, bg=PANEL_BG)
        b_row.pack(fill="x", pady=(0, 6))

        tk.Label(b_row, text="Bundle Folder:", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 6))
        self.ent_bundle = ttk.Entry(b_row, textvariable=self.bundle_dir_var, width=50)
        self.ent_bundle.pack(side="left", padx=(0, 8))
        ttk.Button(b_row, text="Browse Bundle", command=self._browse_bundle_dir).pack(side="left", padx=2)
        ttk.Button(b_row, text="Load Bundle", command=self._load_selected_bundle).pack(side="left", padx=2)

        row1 = tk.Frame(m_frame, bg=PANEL_BG)
        row1.pack(fill="x", pady=(4, 6))

        tk.Label(row1, text="Selected Architecture:", font=(FONT_FAMILY, 9, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 8))
        self.model_choice_var = tk.StringVar(value="cnn")
        ttk.Radiobutton(row1, text="1D-CNN (Temporal Conv1D, Recommended)", variable=self.model_choice_var, value="cnn", command=self._switch_model).pack(side="left", padx=6)
        ttk.Radiobutton(row1, text="MLP Baseline (Dense Flat 306-D)", variable=self.model_choice_var, value="mlp", command=self._switch_model).pack(side="left", padx=6)

        self.lbl_model_meta = tk.Label(
            m_frame,
            text=f"CNN Path: {os.path.basename(self.cnn_path)} | Input: (51, 6) | Classes: {len(CANONICAL_CLASSES)}",
            font=(FONT_MONO, 8),
            bg=PANEL_BG,
            fg=TEXT_MUTED
        )
        self.lbl_model_meta.pack(anchor="w", pady=(2, 0))

        # 2. Safety & Filtering Parameters
        p_frame = ttk.LabelFrame(pad, text="Recognition Thresholds & Safety Controls", style="Panel.TLabelframe", padding=12)
        p_frame.pack(fill="x", pady=(0, 10))

        self.dry_run_var = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(
            p_frame,
            text="STRICT DRY RUN MODE (Safe presentation preview; suppresses OS keystroke dispatch)",
            variable=self.dry_run_var,
            command=self._on_dry_run_toggle
        )
        chk.pack(anchor="w", pady=(0, 8))

        ctrl_row = tk.Frame(p_frame, bg=PANEL_BG)
        ctrl_row.pack(fill="x", pady=4)

        tk.Label(ctrl_row, text="Confidence Threshold:", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 6))
        self.thresh_var = tk.DoubleVar(value=0.75)
        self.spin_thresh = ttk.Spinbox(ctrl_row, from_=0.50, to=0.95, increment=0.05, textvariable=self.thresh_var, width=5, command=self._apply_thresholds)
        self.spin_thresh.pack(side="left", padx=(0, 20))

        tk.Label(ctrl_row, text="Cooldown Duration (sec):", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 6))
        self.cooldown_var = tk.DoubleVar(value=1.2)
        self.spin_cool = ttk.Spinbox(ctrl_row, from_=0.2, to=3.0, increment=0.1, textvariable=self.cooldown_var, width=5, command=self._apply_thresholds)
        self.spin_cool.pack(side="left")

        # 3. Hardware Serial
        ser_frame = ttk.LabelFrame(pad, text="Hardware Serial Connection (ESP32 + MPU6050)", style="Panel.TLabelframe", padding=12)
        ser_frame.pack(fill="x", pady=(0, 10))

        s_row = tk.Frame(ser_frame, bg=PANEL_BG)
        s_row.pack(fill="x")

        tk.Label(s_row, text="COM Port:", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 4))
        self.port_var = tk.StringVar(value="COM6")
        self.ent_port = ttk.Entry(s_row, textvariable=self.port_var, width=10)
        self.ent_port.pack(side="left", padx=(0, 12))

        tk.Label(s_row, text="Baud Rate:", font=(FONT_FAMILY, 9), bg=PANEL_BG, fg=TEXT_PRIMARY).pack(side="left", padx=(0, 4))
        self.baud_var = tk.StringVar(value="115200")
        self.cbo_baud = ttk.Combobox(s_row, textvariable=self.baud_var, values=["115200", "9600", "57600"], width=9, state="readonly")
        self.cbo_baud.pack(side="left", padx=(0, 16))

        self.btn_serial = ttk.Button(s_row, text="Connect Hardware", command=self._toggle_serial_connection)
        self.btn_serial.pack(side="left", padx=4)

        self.btn_calib = ttk.Button(s_row, text="Calibrate IMU", command=self._calibrate_serial)
        self.btn_calib.pack(side="left", padx=4)

        self.lbl_ser_status = tk.Label(s_row, text="Hardware Disconnected (Replay Available)", font=(FONT_FAMILY, 8, "italic"), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_ser_status.pack(side="left", padx=12)

    # =========================================================================
    # TAB 5: Project Info & Academic Identity
    # =========================================================================
    def _build_info_tab(self):
        pad = tk.Frame(self.tab_info, bg=BG_DARK, padx=16, pady=12)
        pad.pack(fill="both", expand=True)

        txt_info = scrolledtext.ScrolledText(
            pad,
            font=(FONT_FAMILY, 9),
            bg="#0F172A",
            fg=TEXT_PRIMARY,
            wrap="word",
            padx=16,
            pady=12,
            spacing1=3,
            spacing3=4
        )
        txt_info.pack(fill="both", expand=True)

        info_content = f"""UNIVERSITY OF BASRAH · COLLEGE OF ENGINEERING
Department of Computer Engineering
Graduation Project Technical Demonstration

PROJECT TITLE:
“Machine Learning-Based Hand Gesture Recognition Using a Wearable IMU Sensor for PowerPoint Control”

SUPERVISOR:
Dr. Khalid Ali Abbas

ACADEMIC RESEARCH MOTIVATION:
Hands-free presentation navigation using a single wearable 6-axis IMU sensor (accelerometer + gyroscope) worn on the user's wrist. The system classifies multi-variate inertial time series into presentation navigation commands without requiring direct physical touch, presentation clickers, or optical line-of-sight tracking.

CORE HARDWARE & SENSING SPECIFICATION:
1. Microcontroller: ESP32 (32-bit dual-core, hardware I2C at 400 kHz Fast-Mode).
2. Inertial Sensor: MPU6050 6-axis MEMS IMU.
   - Accelerometer: ±2g full-scale range (Sensitivity: 16384 LSB/g). Captures static gravitational tilt & dynamic acceleration.
   - Gyroscope: ±250 dps full-scale range (Sensitivity: 131 LSB/dps). Captures rotational velocity & agitation.
3. Sampling Pipeline: Compensated 50 Hz interval acquisition (20 ms interval). Exactly 51 consecutive samples per gesture window (~1.02 seconds).

CANONICAL GESTURE TO POWERPOINT MAPPING:
- Tilt Left  ('left')  -> Previous Slide (Left Arrow Key)
- Tilt Right ('right') -> Next Slide (Right Arrow Key)
- Tilt Up    ('up')    -> Start Presentation from Beginning (F5 Key)
- Tilt Down  ('down')  -> Toggle Blank / Unblank Screen (B Key)
- Shake      ('shake') -> End Presentation / Cancel (Escape Key)
- Still      ('still') -> Resting Neutral Baseline (No Command)

PHASE 2 ARCHITECTURAL ENHANCEMENTS:
1. Intentional Gesture State Machine:
   Enforces 'One Intentional Gesture -> Exactly One Command'.
   Holding a tilted posture is blocked ('HOLDING_POSTURE_BLOCKED') until the user returns to the neutral resting posture.
   Includes hysteresis (2 consecutive windows), 0.30s dwell time, and 6.0s timeout recovery.

2. Motion Quality & Margin Rejection:
   Rejects ambiguous predictions where top-1 and top-2 softmax separation is < 0.20.
   Validates kinetic motion energy before accepting dynamic gestures like shake.

3. Reproducible Latency & Evaluation:
   Reports accuracy, macro F1, accepted vs suppressed command counts, and breakdown of model inference vs window buffering latency.
"""
        txt_info.insert("1.0", info_content)
        txt_info.config(state="disabled")

    # =========================================================================
    # Pipeline Initialization & Management
    # =========================================================================
    def _init_pipeline(self):
        try:
            if os.path.exists(self.mean_path) and os.path.exists(self.std_path):
                self.normalizer = ZScoreNormalizer.load(self.mean_path, self.std_path)
                self._log(f"[INIT] Loaded normalizer from {os.path.basename(self.mean_path)}")
            else:
                self.normalizer = None

            self._switch_model()
            self._load_replay_dataset()
        except Exception as e:
            self._log(f"[INIT ERROR] {e}")

    def _switch_model(self):
        choice = self.model_choice_var.get()
        target_path = self.cnn_path if choice == "cnn" else self.mlp_path

        if not os.path.exists(target_path):
            messagebox.showerror("Model Not Found", f"Model artifact not found:\n{target_path}")
            return

        try:
            self.engine = GestureRecognitionEngine(
                model_path=target_path,
                labels_path=self.labels_path,
                normalizer=self.normalizer,
                confidence_threshold=self.thresh_var.get(),
                shake_threshold=0.55,
                cooldown_seconds=self.cooldown_var.get(),
                dry_run=self.dry_run_var.get()
            )
            model_name = "1D-CNN (Keras)" if choice == "cnn" else "MLP (Scikit-Learn)"
            self.lbl_model_meta.config(text=f"Loaded Model: {model_name} | Path: {os.path.basename(target_path)} | Input: (51, 6)")
            self._log(f"[MODEL] Switched to {model_name}.")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed loading {choice}: {e}")

    def _browse_bundle_dir(self):
        fld = filedialog.askdirectory(initialdir=REPO_ROOT, title="Select Model Bundle Directory")
        if fld:
            self.bundle_dir_var.set(fld)
            self._load_selected_bundle()

    def _load_selected_bundle(self):
        b_dir = self.bundle_dir_var.get().strip()
        try:
            self.active_bundle = ModelBundle(b_dir, prefer_model=self.model_choice_var.get())
            self.normalizer = self.active_bundle.normalizer
            
            # Reconstruct engine with loaded bundle
            model_fname = "imu_gesture_cnn.keras" if self.active_bundle.model_type == "cnn" else "imu_gesture_mlp.pkl"
            m_path = os.path.join(b_dir, model_fname)
            if not os.path.exists(m_path):
                # Try fallback names
                m_path = os.path.join(b_dir, "cnn_model.keras" if self.active_bundle.model_type == "cnn" else "mlp_model.pkl")

            l_path = os.path.join(b_dir, "label_classes.npy")
            if not os.path.exists(l_path):
                l_path = self.labels_path

            self.engine = GestureRecognitionEngine(
                model_path=m_path,
                labels_path=l_path,
                normalizer=self.normalizer,
                confidence_threshold=self.thresh_var.get(),
                shake_threshold=0.55,
                cooldown_seconds=self.cooldown_var.get(),
                dry_run=self.dry_run_var.get()
            )
            meta_str = f"Bundle: {self.active_bundle.metadata.model_name} | Rate: {self.active_bundle.metadata.sampling_rate_hz} Hz"
            self.lbl_model_meta.config(text=meta_str)
            self._log(f"[BUNDLE] Successfully loaded bundle from {os.path.basename(b_dir)} ({meta_str})")
        except Exception as e:
            messagebox.showerror("Bundle Load Error", f"Could not load bundle:\n{e}")

    def _apply_thresholds(self):
        if self.engine:
            self.engine.confidence_threshold = self.thresh_var.get()
            self.engine.cooldown_seconds = self.cooldown_var.get()
            self.engine.state_machine.cooldown_seconds = self.cooldown_var.get()
            self._log(f"[CONFIG] Updated confidence threshold to {self.thresh_var.get():.2f}, cooldown to {self.cooldown_var.get():.2f}s")

    def _on_dry_run_toggle(self):
        dry = self.dry_run_var.get()
        if self.engine:
            self.engine.dry_run = dry
        if dry:
            self.lbl_safety_badge.config(text="DRY RUN (ACTIVE)", bg="#064E3B", fg=COLOR_SUCCESS)
            self._log("[SAFETY] Strict Dry-Run Active: Keystrokes simulated safely.")
        else:
            confirm = messagebox.askyesno(
                "Enable Live Keystroke Dispatch",
                "WARNING: Enabling live dispatch sends OS keyboard events (Left, Right, F5, Esc, B) "
                "to whichever application currently has active focus on your Windows desktop.\n\nProceed?"
            )
            if not confirm:
                self.dry_run_var.set(True)
                if self.engine: self.engine.dry_run = True
                return
            self.lbl_safety_badge.config(text="LIVE OUTPUT (CAUTION)", bg="#7F1D1D", fg="#FCA5A5")
            self._log("[SAFETY WARNING] Live Keystroke Output ENABLED! Directing keys to active desktop window.")

    # =========================================================================
    # Guided Setup Handlers
    # =========================================================================
    def _capture_neutral_reference(self):
        # Sample current window from replay or mock
        trial = self.replay_source.get_trial(0)
        window = trial.data if trial is not None else self.setup_workflow.generate_mock_trial("still")
        ref = self.setup_workflow.record_neutral_window(window)
        
        # Configure engine state machine with new neutral baseline
        if self.engine:
            self.engine.state_machine.config.ref_ax = ref.mean_ax_g
            self.engine.state_machine.config.ref_ay = ref.mean_ay_g
            self.engine.state_machine.config.ref_az = ref.mean_az_g

        status_text = f"Neutral Reference Recorded: [{ref.mean_ax_g:.2f}, {ref.mean_ay_g:.2f}, {ref.mean_az_g:.2f}] g"
        self.lbl_neutral_status.config(text=status_text, fg=COLOR_SUCCESS)
        self._log(f"[SETUP] {status_text}")
        messagebox.showinfo("Neutral Calibration", f"Neutral resting posture calibrated successfully!\n\nBaseline Gravity: {ref.mean_ax_g:.2f}X, {ref.mean_ay_g:.2f}Y, {ref.mean_az_g:.2f}Z g\nGyro Bias: {ref.bias_gx_dps:.2f}X, {ref.bias_gy_dps:.2f}Y, {ref.bias_gz_dps:.2f}Z deg/s")

    def _start_guided_protocol(self):
        self.setup_workflow = GuidedSetupWorkflow(
            profile=ParticipantProfile(
                participant_id=self.part_id_var.get().strip() or "P01",
                session_id=self.sess_id_var.get().strip() or "S01",
                sensor_placement=self.place_var.get(),
                hand=self.hand_var.get()
            ),
            repetitions_per_gesture=5
        )
        self.lbl_guided_step.config(text=f"Perform gesture: '{self.setup_workflow.current_gesture.upper()}' (Repetition 1/5)")
        self._log(f"[SETUP] Initialized guided protocol for Participant {self.setup_workflow.profile.participant_id}.")

    def _record_guided_rep(self):
        curr_g = self.setup_workflow.current_gesture
        if curr_g is None:
            messagebox.showinfo("Protocol Completed", "All scheduled gestures have been collected.")
            return

        mock_win = self.setup_workflow.generate_mock_trial(curr_g)
        ok, msg, info = self.setup_workflow.add_trial(mock_win)
        if ok:
            if info["is_workflow_completed"]:
                self.lbl_guided_step.config(text="All gestures collected! Ready to save.")
                messagebox.showinfo("Protocol Complete", "Guided protocol complete for all 6 canonical gestures.")
            else:
                self.lbl_guided_step.config(text=f"Perform gesture: '{info['next_gesture'].upper()}' (Repetition {info['next_rep']}/5)")
            self._log(f"[SETUP] {msg}")
        else:
            messagebox.showwarning("Quality Check Warning", msg)

    def _run_full_mock_protocol(self):
        self._start_guided_protocol()
        while not self.setup_workflow.is_completed:
            curr_g = self.setup_workflow.current_gesture
            win = self.setup_workflow.generate_mock_trial(curr_g)
            self.setup_workflow.add_trial(win)
        self.lbl_guided_step.config(text="Full Mock Protocol Completed (30 trials generated).")
        self._log("[SETUP] Successfully executed full mock protocol across all gestures.")
        messagebox.showinfo("Mock Execution", "Completed offline simulation of guided collection (30 trials generated). Click 'Save Profile' to export.")

    def _save_profile_session(self):
        try:
            prof_path, trials_path = self.setup_workflow.save_session()
            msg = f"Profile and trials safely saved!\n\nProfile JSON:\n{prof_path}\n\nTrials CSV:\n{trials_path}"
            self._log(f"[SETUP] Exported profile to {os.path.basename(prof_path)}")
            messagebox.showinfo("Saved Successfully", msg)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed saving session:\n{e}")

    # =========================================================================
    # Reproducible Evaluation Action
    # =========================================================================
    def _run_reproducible_evaluation(self):
        if not self.engine or len(self.replay_source.valid_trials) == 0:
            messagebox.showwarning("Evaluation", "Please load a dataset first.")
            return

        def _worker():
            evaluator = EvaluationEngine()
            trials = self.replay_source.valid_trials
            windows = [t.data for t in trials]
            labels = [t.canonical_gesture for t in trials]

            def model_fn(w):
                res = self.engine.process_window(w)
                suppressed = res.is_suppressed_cooldown or res.is_holding_blocked
                rejected = res.is_suppressed_threshold or (res.suppression_reason in ["INSUFFICIENT_CLASS_MARGIN", "INSUFFICIENT_MOTION_ENERGY"])
                return res.predicted_label, res.confidence, suppressed, rejected

            report = evaluator.evaluate_model(
                model_fn=model_fn,
                test_windows=windows,
                true_labels=labels,
                model_name=self.engine.model_type.upper()
            )
            summary_txt = evaluator.format_summary_table(report)
            self.ui_queue.put(("log", f"\n{summary_txt}\n"))
            self.ui_queue.put(("custom", lambda: messagebox.showinfo(f"Evaluation Results ({report.model_name})", f"Accuracy: {report.accuracy_pct}%\nMacro F1: {report.macro_f1}\nAccepted Commands: {report.accepted_commands}\nDuplicate Rate: {report.duplicate_command_rate_pct}%\nMean Latency: {report.inference_ms_mean} ms\n\nFull table printed to Event Log.")))

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # Replay Actions & Callbacks
    # =========================================================================
    def _browse_folder(self):
        fld = filedialog.askdirectory(initialdir=REPO_ROOT, title="Select Dataset Directory")
        if fld:
            self.data_path_var.set(fld)
            self._load_replay_dataset()

    def _browse_file(self):
        fl = filedialog.askopenfilename(initialdir=REPO_ROOT, filetypes=[("CSV Files", "*.csv")], title="Select Gesture CSV")
        if fl:
            self.data_path_var.set(fl)
            self._load_replay_dataset()

    def _load_replay_dataset(self):
        path = self.data_path_var.get().strip()
        if not os.path.exists(path):
            self._log(f"[REPLAY ERROR] Path does not exist: {path}")
            return

        try:
            valid_c, quaran_c = self.replay_source.load(path)
            self.lbl_data_stats.config(text=f"Dataset: {os.path.basename(path)} | Valid Trials: {valid_c} | Quarantined: {quaran_c}")
            self.lbl_progress.config(text=f"Trial 0 / {valid_c}")
            self._log(f"[REPLAY] Loaded dataset '{os.path.basename(path)}': {valid_c} valid trials, {quaran_c} quarantined.")

            # Update trials table
            for row in self.tree_trials.get_children():
                self.tree_trials.delete(row)

            for i, t in enumerate(self.replay_source.trials):
                status_txt = "VALID (51 samples)" if t.is_valid else f"QUARANTINED: {t.status_message}"
                self.tree_trials.insert("", "end", values=(i + 1, t.trial_key, t.raw_gesture, str(t.canonical_gesture), len(t.data), status_txt))

            # Update Quarantine Text Box
            self.txt_quarantine.delete("1.0", tk.END)
            if self.replay_source.quarantine_log:
                self.txt_quarantine.insert(tk.END, f"=== QUARANTINE AUDIT LOG ({len(self.replay_source.quarantine_log)} Anomalous Trials Found) ===\n\n")
                for q in self.replay_source.quarantine_log:
                    self.txt_quarantine.insert(tk.END, f"Trial Key : {q.get('trial_key')}\n")
                    self.txt_quarantine.insert(tk.END, f"File      : {os.path.basename(q.get('file', ''))}\n")
                    self.txt_quarantine.insert(tk.END, f"Raw Label : '{q.get('raw_gesture')}'\n")
                    self.txt_quarantine.insert(tk.END, f"Samples   : {q.get('rows', 'N/A')}\n")
                    self.txt_quarantine.insert(tk.END, f"Reason    : {q.get('reason')}\n")
                    self.txt_quarantine.insert(tk.END, "-" * 70 + "\n")
            else:
                self.txt_quarantine.insert(tk.END, "Zero anomalous or quarantined trials in current dataset. All trials conform to 51 samples.")

        except Exception as e:
            self._log(f"[REPLAY ERROR] {e}")

    def _start_replay(self):
        if self.replay_source.total_trials == 0:
            self._load_replay_dataset()
        if self.replay_source.total_trials == 0:
            messagebox.showwarning("Replay", "No trials loaded.")
            return

        self.lbl_engine_badge.config(text="REPLAYING", bg="#0E2A47", fg=ACCENT_CYAN)
        self.replay_source.set_speed(self.speed_var.get())
        self._log("[REPLAY] Playback started.")

        self.replay_source.start_replay(
            on_trial_emit=self._on_trial_emitted,
            on_complete=self._on_replay_complete,
            only_valid=True
        )

    def _pause_replay(self):
        self.replay_source.pause()
        self.lbl_engine_badge.config(text="PAUSED", bg="#78350F", fg=COLOR_WARNING)
        self._log("[REPLAY] Playback paused.")

    def _stop_replay(self):
        self.replay_source.reset()
        if self.engine:
            self.engine.reset_state_machine()
        self.lbl_engine_badge.config(text="STANDBY", bg="#1E293B", fg=TEXT_MUTED)
        self.lbl_progress.config(text=f"Trial 0 / {len(self.replay_source.valid_trials)}")
        self.lbl_sm_badge.config(text="SM: READY", bg="#1E293B", fg=ACCENT_TEAL)
        self._log("[REPLAY] Playback reset to beginning. State machine re-armed.")

    def _step_replay(self):
        trial = self.replay_source.step_next(only_valid=True)
        if trial:
            self._on_trial_emitted(trial, self.replay_source.current_index, len(self.replay_source.valid_trials))
        else:
            self.lbl_engine_badge.config(text="COMPLETED", bg="#064E3B", fg=COLOR_SUCCESS)
            total = len(self.replay_source.valid_trials)
            self.lbl_progress.config(text=f"Trial {total} / {total}")
            self._log(f"[REPLAY] Reached end of dataset ({total}/{total}).")

    def _on_speed_change(self, val):
        s = float(val)
        self.lbl_speed_val.config(text=f"{s:.1f}x")
        self.replay_source.set_speed(s)

    def _on_replay_complete(self):
        def _cb():
            self.lbl_engine_badge.config(text="COMPLETED", bg="#064E3B", fg=COLOR_SUCCESS)
            total_valid = len(self.replay_source.valid_trials)
            self.lbl_progress.config(text=f"Trial {total_valid} / {total_valid}")
            self._log(f"[REPLAY COMPLETE] Finished replaying all {total_valid} valid trials.")
        self.ui_queue.put(("custom", _cb))

    def _on_trial_emitted(self, trial: ReplayTrial, current_idx: Optional[int] = None, total_trials: Optional[int] = None):
        """Processes each trial emitted by ReplaySource."""
        if not trial.is_valid:
            return

        c_idx = current_idx if current_idx is not None else self.replay_source.current_index
        t_tot = total_trials if total_trials is not None else len(self.replay_source.valid_trials)

        if self.engine:
            res: RecognitionResult = self.engine.process_window(trial.data)

            # Clean separation: a correct prediction is ALWAYS 'Prediction Matched'
            rec_status, rec_color = res.get_recognition_status(expected_gesture=trial.canonical_gesture)
            disp_status, disp_color = res.get_dispatch_status()

            # Pass accepted action to internal slide deck preview
            if res.filtered_label not in ["still", "unknown"] and not res.is_suppressed_cooldown and not res.is_holding_blocked:
                action = res.command.get("action", "")
                self.ui_queue.put(("slide_action", action))

            self.ui_queue.put(("update_recognition", {
                "expected": f"{trial.canonical_gesture} (#{trial.trial_id})",
                "predicted": res.predicted_label.upper(),
                "conf": res.confidence,
                "cmd": res.command.get("description", "None"),
                "rec_status": rec_status,
                "rec_color": rec_color,
                "disp_status": disp_status,
                "disp_color": disp_color,
                "wave_accel": trial.data[:, 0:3],
                "wave_gyro": trial.data[:, 3:6],
                "trial_idx": c_idx,
                "total_trials": t_tot,
                "sm_state": res.state_machine_state,
                "sm_detail": res.status_detail
            }))

            # Tabular log line: TIME | TRIAL | EXPECTED | PREDICTED | CONF | ACTION
            t_str = time.strftime("%H:%M:%S")
            log_line = f"{t_str:<8} | #{c_idx:<8} | {str(trial.canonical_gesture):<8} | {res.predicted_label:<9} | {res.confidence*100:>5.1f}% | {disp_status}"
            self.ui_queue.put(("log", log_line))

    # =========================================================================
    # Serial Hardware Connection (ESP32 USB)
    # =========================================================================
    def _toggle_serial_connection(self):
        if self.ser and self.ser.is_open:
            self.serial_stop_event.set()
            try:
                self.ser.write(b"stop\n")
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.btn_serial.config(text="Connect Hardware")
            self.lbl_ser_status.config(text="Disconnected", fg=TEXT_MUTED)
            self.lbl_mode_badge.config(text="REPLAY MODE", bg="#0E2A47", fg=ACCENT_CYAN)
            self._log("[SERIAL] Hardware connection closed. Switched to Replay.")
            return

        import serial
        port = self.port_var.get().strip()
        baud = int(self.baud_var.get().strip())

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(1.5)
            self.ser.reset_input_buffer()
            self.serial_stop_event.clear()

            self.btn_serial.config(text="Disconnect")
            self.lbl_ser_status.config(text=f"Connected on {port} (115200)", fg=COLOR_SUCCESS)
            self.lbl_mode_badge.config(text="SERIAL HARDWARE", bg="#064E3B", fg=COLOR_SUCCESS)
            self._log(f"[SERIAL] Connected to ESP32 on {port}. Waiting for button trigger or 'start'...")

            self.serial_thread = threading.Thread(target=self._serial_reader, daemon=True)
            self.serial_thread.start()
        except Exception as e:
            messagebox.showerror("Serial Connection Failed", f"Could not connect to {port}:\n{e}")

    def _calibrate_serial(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Serial", "ESP32 is not connected.")
            return
        self._log("[SERIAL] Sending 'calibrate' command to firmware...")
        try:
            self.ser.write(b"calibrate\n")
        except Exception as e:
            self._log(f"[SERIAL ERROR] {e}")

    def _serial_reader(self):
        from src.data_pipeline import parse_sensor_line, WINDOW_SIZE
        buffer = []
        is_capturing = False

        while not self.serial_stop_event.is_set():
            if not self.ser or not self.ser.is_open:
                break
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if "# BUTTON_PRESSED" in line:
                    self.ui_queue.put(("log", "[HARDWARE] ESP32 button press detected! Starting capture..."))
                    self.ser.reset_input_buffer()
                    self.ser.write(b"start\n")
                    buffer.clear()
                    is_capturing = True
                    continue

                if is_capturing:
                    vals, ts, err = parse_sensor_line(line)
                    if vals is not None:
                        buffer.append(vals)
                        if len(buffer) >= WINDOW_SIZE:
                            self.ser.write(b"stop\n")
                            is_capturing = False
                            window = np.array(buffer, dtype=np.float32)
                            buffer.clear()

                            if self.engine:
                                res = self.engine.process_window(window)
                                rec_status, rec_color = res.get_recognition_status()
                                disp_status, disp_color = res.get_dispatch_status()

                                if res.filtered_label not in ["still", "unknown"] and not res.is_suppressed_cooldown and not res.is_holding_blocked:
                                    self.ui_queue.put(("slide_action", res.command.get("action", "")))

                                self.ui_queue.put(("update_recognition", {
                                    "expected": "Live Serial Capture",
                                    "predicted": res.predicted_label.upper(),
                                    "conf": res.confidence,
                                    "cmd": res.command.get("description", "None"),
                                    "rec_status": rec_status,
                                    "rec_color": rec_color,
                                    "disp_status": disp_status,
                                    "disp_color": disp_color,
                                    "wave_accel": window[:, 0:3],
                                    "wave_gyro": window[:, 3:6],
                                    "trial_idx": 1,
                                    "total_trials": 1,
                                    "sm_state": res.state_machine_state,
                                    "sm_detail": res.status_detail
                                }))
                                self.ui_queue.put(("log", f"[LIVE SERIAL] Predicted '{res.predicted_label}' (Conf: {res.confidence*100:.1f}%) -> {disp_status}"))
            except Exception:
                time.sleep(0.05)

    # =========================================================================
    # Main Thread Queue Poller
    # =========================================================================
    def _poll_ui_queue(self):
        try:
            while True:
                msg_type, payload = self.ui_queue.get_nowait()
                if msg_type == "update_recognition":
                    self.lbl_predicted.config(text=payload["predicted"])
                    self.icon_canvas.set_gesture(payload["predicted"], color=payload["rec_color"])
                    self.lbl_conf.config(text=f"Model Confidence: {payload['conf']*100:.1f}%")
                    self.conf_bar["value"] = payload["conf"] * 100
                    self.lbl_expected.config(text=payload["expected"])
                    self.lbl_cmd.config(text=payload["cmd"])
                    self.lbl_rec_status.config(text=payload["rec_status"], fg=payload["rec_color"])
                    self.lbl_dispatch.config(text=payload["disp_status"], fg=payload["disp_color"])
                    
                    sm_state = payload.get("sm_state", "READY")
                    sm_color = COLOR_SUCCESS if sm_state == "READY" else COLOR_WARNING
                    self.lbl_sm_badge.config(text=f"SM: {sm_state}", fg=sm_color)
                    self.lbl_sm_detail.config(text=f"{sm_state} | {payload.get('sm_detail', '')[:48]}...")

                    if payload.get("wave_accel") is not None:
                        self.accel_canvas.update_data(payload["wave_accel"])
                        self.gyro_canvas.update_data(payload["wave_gyro"])

                    self.lbl_progress.config(text=f"Trial {payload.get('trial_idx', 0)} / {payload.get('total_trials', 0)}")

                elif msg_type == "slide_action":
                    self.slide_preview.handle_gesture_action(payload)

                elif msg_type == "log":
                    self.overview_log.insert(tk.END, payload + "\n")
                    self.overview_log.see(tk.END)

                elif msg_type == "custom":
                    payload()

        except queue.Empty:
            pass

        self.root.after(35, self._poll_ui_queue)

    def _log(self, text: str):
        t_str = time.strftime("[%H:%M:%S] ")
        self.ui_queue.put(("log", t_str + text))

    def _clear_log(self):
        self.overview_log.delete("1.0", tk.END)
        self.overview_log.insert(tk.END, f"{'TIME':<8} | {'TRIAL':<10} | {'EXPECTED':<8} | {'PREDICTED':<9} | {'CONF':<6} | {'ACTION / DISPATCH'}\n")
        self.overview_log.insert(tk.END, "-" * 80 + "\n")

    def _on_close(self):
        if self.replay_source:
            self.replay_source.stop()
        if self.ser and self.ser.is_open:
            self.serial_stop_event.set()
            try:
                self.ser.write(b"stop\n")
                self.ser.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ImuGestureControllerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
