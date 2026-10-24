"""
theme.py
Design System and Dark Engineering Palette for IMU Gesture Controller

Aesthetic Direction:
- Refined engineering instrument aesthetic
- Deep navy background: #0B1220
- Slightly lighter panels: #131E30
- Elevated surfaces / cards: #1B2940
- Subtle borders: #263852
- Cyan accent: #38BDF8
- Teal accent: #2DD4BF
- Primary text: #E8EEF7
- Muted text: #94A3B8
- Semantic colors: Green (#10B981), Amber (#F59E0B), Red (#EF4444)
"""

import tkinter as tk
from tkinter import ttk

# --- Palette Constants ---
BG_DARK = "#0B1220"        # Main window background
PANEL_BG = "#131E30"       # Cards, containers, sidebars
SURFACE_BG = "#1B2940"     # Elevated cards, input fields, tables
SURFACE_HOVER = "#243654"  # Hover state
BORDER_COLOR = "#263852"   # Clean structural borders
BORDER_ACTIVE = "#38BDF8"  # Focused/active border

# Accents
ACCENT_CYAN = "#38BDF8"    # Primary interactive / X-axis / focus
ACCENT_TEAL = "#2DD4BF"    # Secondary interactive / Pitch / active
ACCENT_BLUE = "#3B82F6"

# Semantic Signals
COLOR_SUCCESS = "#10B981"  # Matched / Connected (Emerald)
COLOR_WARNING = "#F59E0B"  # Cooldown / Low Conf (Amber)
COLOR_DANGER = "#EF4444"   # Differed / Disconnected (Red)
COLOR_NEUTRAL = "#64748B"  # Inactive / Still (Slate)

# Typography Colors
TEXT_PRIMARY = "#E8EEF7"   # High contrast header and primary text
TEXT_MUTED = "#94A3B8"     # Secondary labels, subtitles, units
TEXT_DISABLED = "#475569"  # Disabled controls

# Sensor Axis Colors (Standardized throughout application)
AXIS_X_COLOR = "#F43F5E"   # Rose / Red for X axis (Roll / ax / gx)
AXIS_Y_COLOR = "#10B981"   # Emerald / Green for Y axis (Pitch / ay / gy)
AXIS_Z_COLOR = "#38BDF8"   # Sky Cyan / Blue for Z axis (Yaw / Gravity / az / gz)

# Typography Config
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"


def apply_dark_theme(root: tk.Tk) -> ttk.Style:
    """Configures ttk styles and root window for deep navy engineering theme."""
    root.configure(bg=BG_DARK)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # General Frame
    style.configure("TFrame", background=BG_DARK)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("Card.TFrame", background=SURFACE_BG, relief="flat")

    # Labels
    style.configure("TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 9))
    style.configure("Panel.TLabel", background=PANEL_BG, foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 9))
    style.configure("PanelMuted.TLabel", background=PANEL_BG, foreground=TEXT_MUTED, font=(FONT_FAMILY, 8))
    style.configure("Card.TLabel", background=SURFACE_BG, foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 9))
    style.configure("CardMuted.TLabel", background=SURFACE_BG, foreground=TEXT_MUTED, font=(FONT_FAMILY, 8))

    # Headers
    style.configure("HeaderTitle.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 15, "bold"))
    style.configure("HeaderSub.TLabel", background=BG_DARK, foreground=ACCENT_CYAN, font=(FONT_FAMILY, 9, "bold"))
    style.configure("HeaderAcademic.TLabel", background=BG_DARK, foreground=TEXT_MUTED, font=(FONT_FAMILY, 8))
    style.configure("Section.TLabel", background=PANEL_BG, foreground=TEXT_PRIMARY, font=(FONT_FAMILY, 10, "bold"))

    # LabelFrames
    style.configure("Panel.TLabelframe", background=PANEL_BG, bordercolor=BORDER_COLOR, lightcolor=BORDER_COLOR, darkcolor=BORDER_COLOR)
    style.configure("Panel.TLabelframe.Label", background=PANEL_BG, foreground=ACCENT_CYAN, font=(FONT_FAMILY, 9, "bold"))

    # Buttons
    style.configure(
        "TButton",
        background=SURFACE_BG,
        foreground=TEXT_PRIMARY,
        bordercolor=BORDER_COLOR,
        focusthickness=1,
        focuscolor=ACCENT_CYAN,
        padding=(8, 4),
        font=(FONT_FAMILY, 9)
    )
    style.map(
        "TButton",
        background=[("pressed", PANEL_BG), ("active", SURFACE_HOVER)],
        foreground=[("pressed", ACCENT_CYAN), ("active", TEXT_PRIMARY)],
        bordercolor=[("active", ACCENT_CYAN)]
    )

    style.configure(
        "Accent.TButton",
        background=ACCENT_CYAN,
        foreground=BG_DARK,
        font=(FONT_FAMILY, 9, "bold"),
        padding=(10, 5)
    )
    style.map(
        "Accent.TButton",
        background=[("pressed", "#0284C7"), ("active", "#7DD3FC")],
        foreground=[("active", BG_DARK)]
    )

    # Entry & Spinbox
    style.configure(
        "TEntry",
        fieldbackground=SURFACE_BG,
        foreground=TEXT_PRIMARY,
        bordercolor=BORDER_COLOR,
        insertcolor=TEXT_PRIMARY,
        padding=4
    )
    style.map("TEntry", bordercolor=[("focus", ACCENT_CYAN)])

    style.configure(
        "TSpinbox",
        fieldbackground=SURFACE_BG,
        foreground=TEXT_PRIMARY,
        bordercolor=BORDER_COLOR,
        arrowcolor=TEXT_PRIMARY,
        padding=3
    )

    # Notebook Tabs
    style.configure(
        "TNotebook",
        background=BG_DARK,
        bordercolor=BORDER_COLOR,
        tabmargins=[0, 0, 0, 0]
    )
    style.configure(
        "TNotebook.Tab",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        padding=(14, 6),
        font=(FONT_FAMILY, 9, "bold"),
        bordercolor=BORDER_COLOR
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", SURFACE_BG), ("active", SURFACE_HOVER)],
        foreground=[("selected", ACCENT_CYAN), ("active", TEXT_PRIMARY)]
    )

    # Progressbar
    style.configure(
        "Horizontal.TProgressbar",
        background=ACCENT_CYAN,
        troughcolor=SURFACE_BG,
        bordercolor=BORDER_COLOR,
        thickness=8
    )

    # Treeview
    style.configure(
        "Treeview",
        background=SURFACE_BG,
        fieldbackground=SURFACE_BG,
        foreground=TEXT_PRIMARY,
        bordercolor=BORDER_COLOR,
        font=(FONT_FAMILY, 8),
        rowheight=22
    )
    style.configure(
        "Treeview.Heading",
        background=PANEL_BG,
        foreground=ACCENT_CYAN,
        font=(FONT_FAMILY, 8, "bold"),
        bordercolor=BORDER_COLOR
    )
    style.map(
        "Treeview",
        background=[("selected", SURFACE_HOVER)],
        foreground=[("selected", ACCENT_CYAN)]
    )

    # Radiobuttons and Checkbuttons
    style.configure(
        "TRadiobutton",
        background=PANEL_BG,
        foreground=TEXT_PRIMARY,
        font=(FONT_FAMILY, 9)
    )
    style.map("TRadiobutton", background=[("active", PANEL_BG)])

    style.configure(
        "TCheckbutton",
        background=PANEL_BG,
        foreground=TEXT_PRIMARY,
        font=(FONT_FAMILY, 9)
    )
    style.map("TCheckbutton", background=[("active", PANEL_BG)])

    return style
