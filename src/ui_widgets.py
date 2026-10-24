"""
ui_widgets.py
Custom Engineering UI Components for IMU Gesture Controller

Widgets Included:
1. WaveformCanvas: High-performance vector oscilloscope plotting 3-axis IMU signals.
2. GestureIconCanvas: Crisp vector iconography for directional tilts, shake, and still.
3. SlidePreviewWidget: Built-in local slide deck simulation responding to gestures.
4. Tooltip: Context-sensitive hover hints for paths and badges.
"""

from typing import List, Optional, Tuple, Dict, Any
import tkinter as tk
from tkinter import ttk
import numpy as np

from src.theme import (
    BG_DARK, PANEL_BG, SURFACE_BG, SURFACE_HOVER, BORDER_COLOR,
    ACCENT_CYAN, ACCENT_TEAL, TEXT_PRIMARY, TEXT_MUTED,
    AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, COLOR_NEUTRAL,
    FONT_FAMILY, FONT_MONO
)


class Tooltip:
    """Hover tooltip for Tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#1E293B",
            foreground="#F8FAFC",
            relief="solid",
            borderwidth=1,
            font=(FONT_FAMILY, 8),
            padx=6,
            pady=3
        )
        label.pack()

    def hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class GestureIconCanvas(tk.Canvas):
    """
    Renders clean vector glyphs for recognized gestures (arrows, waveform, level line).
    """

    def __init__(self, parent, size: int = 48, **kwargs):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=SURFACE_BG,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            **kwargs
        )
        self.size = size
        self.set_gesture("still")

    def set_gesture(self, gesture: str, color: str = ACCENT_CYAN):
        self.delete("all")
        s = self.size
        pad = s * 0.22
        cx, cy = s / 2, s / 2

        g = (gesture or "").lower()

        if g == "left":
            # Left arrow
            self.create_line(pad, cy, s - pad, cy, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(pad + 8, cy - 8, pad, cy, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(pad + 8, cy + 8, pad, cy, fill=color, width=3, capstyle=tk.ROUND)
        elif g == "right":
            # Right arrow
            self.create_line(pad, cy, s - pad, cy, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(s - pad - 8, cy - 8, s - pad, cy, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(s - pad - 8, cy + 8, s - pad, cy, fill=color, width=3, capstyle=tk.ROUND)
        elif g == "up":
            # Up arrow
            self.create_line(cx, pad, cx, s - pad, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(cx - 8, pad + 8, cx, pad, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(cx + 8, pad + 8, cx, pad, fill=color, width=3, capstyle=tk.ROUND)
        elif g == "down":
            # Down arrow
            self.create_line(cx, pad, cx, s - pad, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(cx - 8, s - pad - 8, cx, s - pad, fill=color, width=3, capstyle=tk.ROUND)
            self.create_line(cx + 8, s - pad - 8, cx, s - pad, fill=color, width=3, capstyle=tk.ROUND)
        elif g == "shake":
            # Oscillating angular wave
            pts = []
            steps = 14
            for i in range(steps):
                x = pad + (s - 2 * pad) * (i / (steps - 1))
                y = cy + (10 if i % 2 == 0 else -10)
                pts.extend([x, y])
            self.create_line(pts, fill=color, width=2.5, smooth=True, capstyle=tk.ROUND)
        elif g == "still":
            # Stationary level line with center dot
            self.create_line(pad, cy, s - pad, cy, fill=color, width=2, dash=(4, 3))
            self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=color, outline="")
        else:
            # Question / Unknown mark
            self.create_text(cx, cy, text="?", fill=TEXT_MUTED, font=(FONT_FAMILY, 16, "bold"))


class WaveformCanvas(tk.Canvas):
    """
    Renders a 3-channel time-series signal (ax/ay/az or gx/gy/gz) over a 51-sample window.
    Features dark grid lines, range markers, zero-line, and axis legends.
    """

    def __init__(self, parent, title: str, unit: str, y_min: float = -2.0, y_max: float = 2.0, height: int = 110, **kwargs):
        super().__init__(
            parent,
            height=height,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            **kwargs
        )
        self.title = title
        self.unit = unit
        self.y_min = y_min
        self.y_max = y_max
        self.current_data: Optional[np.ndarray] = None  # Expected shape (51, 3)

        self.bind("<Configure>", lambda e: self.redraw())

    def update_data(self, data_3ch: Optional[np.ndarray]):
        """Sets new (51, 3) data array and triggers redraw."""
        self.current_data = data_3ch
        self.redraw()

    def clear(self):
        self.current_data = None
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()

        if w < 50 or h < 30:
            return

        pad_left = 48
        pad_right = 16
        pad_top = 22
        pad_bottom = 20

        plot_w = w - pad_left - pad_right
        plot_h = h - pad_top - pad_bottom

        # Background plot area
        self.create_rectangle(pad_left, pad_top, pad_left + plot_w, pad_top + plot_h, fill="#0F172A", outline=BORDER_COLOR)

        # Header Title and Unit
        self.create_text(
            pad_left, 10,
            text=f"{self.title}  [{self.unit}]",
            anchor="w",
            fill=TEXT_PRIMARY,
            font=(FONT_FAMILY, 8, "bold")
        )

        # Legend (X=Rose, Y=Emerald, Z=Sky)
        leg_x = w - pad_right - 105
        self.create_oval(leg_x, 9, leg_x + 6, 15, fill=AXIS_X_COLOR, outline="")
        self.create_text(leg_x + 9, 12, text="X", anchor="w", fill=TEXT_MUTED, font=(FONT_FAMILY, 7, "bold"))

        leg_x += 28
        self.create_oval(leg_x, 9, leg_x + 6, 15, fill=AXIS_Y_COLOR, outline="")
        self.create_text(leg_x + 9, 12, text="Y", anchor="w", fill=TEXT_MUTED, font=(FONT_FAMILY, 7, "bold"))

        leg_x += 28
        self.create_oval(leg_x, 9, leg_x + 6, 15, fill=AXIS_Z_COLOR, outline="")
        self.create_text(leg_x + 9, 12, text="Z", anchor="w", fill=TEXT_MUTED, font=(FONT_FAMILY, 7, "bold"))

        # Zero reference line
        def val_to_y(val):
            ratio = (val - self.y_min) / (self.y_max - self.y_min)
            ratio = max(0.0, min(1.0, ratio))
            return pad_top + plot_h * (1.0 - ratio)

        zero_y = val_to_y(0.0)
        self.create_line(pad_left, zero_y, pad_left + plot_w, zero_y, fill="#334155", dash=(3, 3))

        # Y-axis labels and tick lines
        self.create_text(pad_left - 4, pad_top + 4, text=f"+{self.y_max:g}", anchor="e", fill=TEXT_MUTED, font=(FONT_MONO, 7))
        self.create_text(pad_left - 4, zero_y, text="0", anchor="e", fill=TEXT_MUTED, font=(FONT_MONO, 7))
        self.create_text(pad_left - 4, pad_top + plot_h - 4, text=f"{self.y_min:g}", anchor="e", fill=TEXT_MUTED, font=(FONT_MONO, 7))

        # X-axis timeline markers (0, 25, 50 samples)
        self.create_text(pad_left, h - 8, text="0", anchor="c", fill=TEXT_MUTED, font=(FONT_MONO, 7))
        self.create_text(pad_left + plot_w / 2, h - 8, text="25", anchor="c", fill=TEXT_MUTED, font=(FONT_MONO, 7))
        self.create_text(pad_left + plot_w, h - 8, text="50", anchor="c", fill=TEXT_MUTED, font=(FONT_MONO, 7))

        # Plot waveforms if data exists
        if self.current_data is not None and len(self.current_data) > 1:
            data = self.current_data
            n_samples = len(data)
            dx = plot_w / max(1, n_samples - 1)

            colors = [AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR]

            for ch in range(min(3, data.shape[1])):
                pts = []
                for i in range(n_samples):
                    px = pad_left + i * dx
                    py = val_to_y(data[i, ch])
                    pts.extend([px, py])

                self.create_line(pts, fill=colors[ch], width=1.5, capstyle=tk.ROUND, joinstyle=tk.ROUND)
        else:
            # Idle placeholder message
            self.create_text(
                pad_left + plot_w / 2,
                pad_top + plot_h / 2,
                text="Waiting for Window Data (51 samples)",
                fill="#475569",
                font=(FONT_FAMILY, 8, "italic")
            )


class SlidePreviewWidget(ttk.Frame):
    """
    Built-In Presentation Preview.
    Maintains a 5-slide academic demonstration deck and responds to recognized gestures.
    Works entirely locally without PowerPoint installed or sending OS keystrokes.
    """

    SAMPLE_DECK: List[Dict[str, Any]] = [
        {
            "slide_num": 1,
            "title": "IMU Gesture Presentation Controller",
            "subtitle": "Graduation Project Demonstration",
            "bullets": [
                "Machine Learning-Based Hand Gesture Recognition",
                "Wearable 6-Axis IMU (ESP32 + MPU6050)",
                "Hands-Free Desktop & Slideshow Navigation",
                "University of Basrah · College of Engineering",
                "Supervisor: Dr. Khalid Ali Abbas"
            ],
            "accent": ACCENT_CYAN
        },
        {
            "slide_num": 2,
            "title": "System Architecture & Hardware",
            "subtitle": "Multivariate Time-Series Acquisition",
            "bullets": [
                "MPU6050: 3-Axis Accelerometer (±2g) + 3-Axis Gyroscope (±250 dps)",
                "I2C Bus at 400 kHz Fast-Mode with ESP32",
                "Compensated 50 Hz Hardware Sampling Interval",
                "Push-Button Trigger on GPIO 4 with 200 ms Debounce",
                "Serial Stream over USB (115200 baud)"
            ],
            "accent": ACCENT_TEAL
        },
        {
            "slide_num": 3,
            "title": "IMU Kinematics & Waveforms",
            "subtitle": "Physical Gesture Signatures",
            "bullets": [
                "Tilt Left / Right: Sustained Roll Deflection (ax acceleration shift)",
                "Tilt Up / Down: Pitch Deflection (ay acceleration shift)",
                "Shake: High-Energy Angular Rate Oscillation (gx, gy, gz)",
                "Still: Stationary Baseline with +1.0g Gravity on Z-Axis",
                "Fixed Window: Exactly 51 Timesteps (~1.0 Second Window)"
            ],
            "accent": "#F59E0B"
        },
        {
            "slide_num": 4,
            "title": "Deep Learning Recognition Pipeline",
            "subtitle": "1D-CNN vs. Baseline MLP",
            "bullets": [
                "Per-Axis Z-Score Normalization (Fitted strictly on Training Split)",
                "1D-CNN: Conv1D (32) -> MaxPool -> Conv1D (64) -> MaxPool -> Dense (64)",
                "Softmax Classification into 6 Canonical Gesture Classes",
                "Confidence Threshold Filtering (0.75 General, 0.55 Shake)",
                "1.2-Second Action Cooldown to Suppress Accidental Double-Triggers"
            ],
            "accent": "#A855F7"
        },
        {
            "slide_num": 5,
            "title": "Experimental Evaluation & Summary",
            "subtitle": "Graduation Results & Deliverables",
            "bullets": [
                "100% Test Accuracy on Held-Out Controlled Test Set (18 Trials)",
                "Heuristic Tilt Rule Baseline Achieves 95.8% (Validates ML Necessity for Dynamic Swipes)",
                "Hardware-Free Replay Engine with Quarantined Anomaly Handling",
                "Complete Reproducibility Metadata & Audit Provenance",
                "Next Phase: BLE Wireless Integration & Multi-User Wrist Study"
            ],
            "accent": COLOR_SUCCESS
        }
    ]

    def __init__(self, parent, on_slide_change=None, **kwargs):
        super().__init__(parent, style="Panel.TFrame", **kwargs)
        self.current_slide_idx: int = 0
        self.is_black_screen: bool = False
        self.on_slide_change = on_slide_change

        self._build_ui()
        self.render_slide()

    def _build_ui(self):
        # Header strip
        top_strip = ttk.Frame(self, style="Panel.TFrame")
        top_strip.pack(fill="x", padx=8, pady=(6, 4))

        ttk.Label(
            top_strip,
            text="Internal Presentation Preview",
            style="Panel.TLabel",
            font=(FONT_FAMILY, 9, "bold")
        ).pack(side="left")

        self.lbl_badge = tk.Label(
            top_strip,
            text="SIMULATED DECK",
            font=(FONT_FAMILY, 7, "bold"),
            bg="#0E2A47",
            fg=ACCENT_CYAN,
            padx=5,
            pady=1
        )
        self.lbl_badge.pack(side="left", padx=8)

        self.lbl_counter = ttk.Label(
            top_strip,
            text="Slide 1 / 5",
            style="PanelMuted.TLabel",
            font=(FONT_MONO, 8)
        )
        self.lbl_counter.pack(side="right")

        # Slide Surface (Canvas for 16:9 proportioned clean rendering)
        self.slide_canvas = tk.Canvas(
            self,
            bg="#0F172A",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            height=165
        )
        self.slide_canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.slide_canvas.bind("<Configure>", lambda e: self.render_slide())

        # Bottom Mini-Control Toolbar
        toolbar = ttk.Frame(self, style="Panel.TFrame")
        toolbar.pack(fill="x", padx=8, pady=(4, 6))

        ttk.Button(toolbar, text="⏮ Start", width=7, command=self.start_deck).pack(side="left", padx=2)
        ttk.Button(toolbar, text="◀ Prev", width=7, command=self.prev_slide).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Next ▶", width=7, command=self.next_slide).pack(side="left", padx=2)
        ttk.Button(toolbar, text="⏹ End", width=7, command=self.end_deck).pack(side="left", padx=2)
        ttk.Button(toolbar, text="⬛ Blank (B)", width=9, command=self.toggle_black_screen).pack(side="left", padx=6)

        ttk.Label(
            toolbar,
            text="Responds directly to recognized gestures",
            style="PanelMuted.TLabel",
            font=(FONT_FAMILY, 8, "italic")
        ).pack(side="right")

    def handle_gesture_action(self, action: str):
        """Dispatches an action string emitted by the recognition engine."""
        act = (action or "").lower()
        if act == "next_slide":
            self.next_slide()
        elif act == "previous_slide":
            self.prev_slide()
        elif act == "start_presentation":
            self.start_deck()
        elif act == "end_presentation":
            self.end_deck()
        elif act == "black_screen":
            self.toggle_black_screen()

    def next_slide(self):
        self.is_black_screen = False
        if self.current_slide_idx < len(self.SAMPLE_DECK) - 1:
            self.current_slide_idx += 1
            self.render_slide()

    def prev_slide(self):
        self.is_black_screen = False
        if self.current_slide_idx > 0:
            self.current_slide_idx -= 1
            self.render_slide()

    def start_deck(self):
        self.is_black_screen = False
        self.current_slide_idx = 0
        self.render_slide()

    def end_deck(self):
        self.is_black_screen = False
        self.current_slide_idx = len(self.SAMPLE_DECK) - 1
        self.render_slide()

    def toggle_black_screen(self):
        self.is_black_screen = not self.is_black_screen
        self.render_slide()

    def render_slide(self):
        self.slide_canvas.delete("all")
        w = self.slide_canvas.winfo_width()
        h = self.slide_canvas.winfo_height()

        if w < 50 or h < 30:
            return

        total = len(self.SAMPLE_DECK)
        curr = self.current_slide_idx
        data = self.SAMPLE_DECK[curr]
        self.lbl_counter.config(text=f"Slide {curr + 1} / {total}")

        # If Black Screen Active
        if self.is_black_screen:
            self.slide_canvas.create_rectangle(0, 0, w, h, fill="#000000", outline="")
            self.slide_canvas.create_text(
                w / 2, h / 2 - 8,
                text="[ SCREEN BLANKED - GESTURE: DOWN / KEY: B ]",
                fill="#64748B",
                font=(FONT_MONO, 9, "bold")
            )
            self.slide_canvas.create_text(
                w / 2, h / 2 + 12,
                text="Press Down or click 'Blank' to restore presentation",
                fill="#475569",
                font=(FONT_FAMILY, 8)
            )
            return

        # Slide Surface Background
        self.slide_canvas.create_rectangle(0, 0, w, h, fill="#131E30", outline="")

        # Accent top bar
        accent_color = data.get("accent", ACCENT_CYAN)
        self.slide_canvas.create_rectangle(0, 0, w, 4, fill=accent_color, outline="")

        # Slide Title
        self.slide_canvas.create_text(
            16, 20,
            text=data["title"],
            anchor="w",
            fill=TEXT_PRIMARY,
            font=(FONT_FAMILY, 11, "bold")
        )

        # Slide Subtitle
        self.slide_canvas.create_text(
            16, 38,
            text=data["subtitle"],
            anchor="w",
            fill=accent_color,
            font=(FONT_FAMILY, 8, "bold")
        )

        # Divider
        self.slide_canvas.create_line(16, 48, w - 16, 48, fill="#1E293B", width=1)

        # Bullets
        bullets = data.get("bullets", [])
        start_y = 62
        spacing = 18

        for i, b in enumerate(bullets):
            by = start_y + i * spacing
            if by > h - 20:
                break
            # Bullet point glyph
            self.slide_canvas.create_oval(20, by - 3, 25, by + 2, fill=accent_color, outline="")
            self.slide_canvas.create_text(
                32, by,
                text=b,
                anchor="w",
                fill=TEXT_MUTED if i > 0 else TEXT_PRIMARY,
                font=(FONT_FAMILY, 8)
            )

        # Footer indicator
        footer_text = f"University of Basrah · Department of Computer Engineering · Slide {curr + 1} of {total}"
        self.slide_canvas.create_text(
            w - 16, h - 10,
            text=footer_text,
            anchor="e",
            fill="#475569",
            font=(FONT_FAMILY, 7)
        )
