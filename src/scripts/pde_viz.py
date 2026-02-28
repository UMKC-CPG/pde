#!/usr/bin/env python3
"""
Visualization UI for PDE — PySide6 + matplotlib.

Layout
------
  Left  canvas : G(x) curves at the current T (and P), with lower convex
                 envelope and common tangent lines highlighted.
  Right canvas : T-x phase diagram (Fixed P mode) or P-x phase diagram
                 (Fixed T mode), revealed incrementally as the primary
                 slider moves.
  Bottom       : Mode selector (when system.has_pressure), temperature slider,
                 and optional pressure slider.

Strategy
--------
  The full phase diagram is pre-computed at startup over N_T_STEPS evenly-
  spaced temperatures from T_max to T_min (at fixed P_initial) and, when
  system.has_pressure, over N_P_STEPS pressures from P_max to P_min (at
  fixed T_initial).

  The T-x / P-x canvas draws all regions once into the figure, then hides
  the unrevealed portion behind a white cover rectangle.  Moving the primary
  slider shrinks the cover (O(1)).  Moving the secondary slider triggers a
  full recompute of the opposite sweep on release.

  The G-x canvas does a full redraw each time the primary slider changes,
  using the pre-computed result nearest to the requested value.
"""

import sys
import threading

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle as MplRectangle
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                                QColorDialog, QComboBox, QDialog, QGridLayout,
                                QHBoxLayout, QLabel, QMainWindow, QMenu,
                                QProgressBar, QPushButton, QRadioButton,
                                QSlider, QStackedWidget, QVBoxLayout, QWidget)

from pde_compute import compute_equilibrium


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_T_STEPS = 200     # temperature steps for the pre-computed T-x diagram
N_P_STEPS = 200     # pressure steps for the pre-computed P-x diagram

# One colour per phase (assigned by position in system.phases)
_COLOR_PALETTE = [
    '#4878CF',  # blue       — gas
    '#6ACC65',  # green      — liquid
    '#D65F5F',  # red        — alpha
    '#B47CC7',  # purple     — beta
    '#C4AD66',  # gold       — gamma
    '#77BEDB',  # sky blue   — delta
    '#8C8C8C',  # grey       — epsilon
    '#A2C7A5',  # sage       — zeta
]

_TWO_PHASE_COLOR = '#D8D8D8'    # light grey fill for two-phase regions
_TWO_PHASE_HATCH = '///'        # diagonal hatch overlay (matplotlib pattern string)

_PALETTES = {
    'Default':         ['#E8A020', '#3A7DBC', '#C94040', '#3AA85A',
                        '#C9A030', '#8A40C9', '#C97030', '#40B8C9'],
    'Colorblind-safe': ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7', '#999999'],
    'Pastel':          ['#FFB347', '#AED6F1', '#F1948A', '#A9DFBF',
                        '#F9E79F', '#D2B4DE', '#FAD7A0', '#A2D9CE'],
    'Dark':            ['#8B4513', '#1A3A5C', '#7B0000', '#1A5C2A',
                        '#6B5B00', '#4A1A6B', '#8B4500', '#006B7B'],
    # Tableau 10 — matplotlib's default color cycle (v2+)
    'Tableau':         ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728',
                        '#9467BD', '#8C564B', '#E377C2', '#7F7F7F'],
    # Solarized — from the popular terminal/editor color scheme
    'Solarized':       ['#268BD2', '#CB4B16', '#859900', '#2AA198',
                        '#D33682', '#6C71C4', '#B58900', '#657B83'],
    # Nord — cool blues and muted accents from the Nord theme
    'Nord':            ['#5E81AC', '#BF616A', '#A3BE8C', '#EBCB8B',
                        '#B48EAD', '#88C0D0', '#D08770', '#4C566A'],
    # Earthy — warm natural tones
    'Earthy':          ['#A0522D', '#6B8E23', '#8B7355', '#556B2F',
                        '#CD853F', '#708090', '#8B8B00', '#2E8B57'],
    # Vivid — high-saturation, high-contrast for presentations
    'Vivid':           ['#E6194B', '#3CB44B', '#4363D8', '#F58231',
                        '#911EB4', '#42D4F4', '#F032E6', '#BFEF45'],
    # Muted — desaturated, easy on the eyes for long sessions
    'Muted':           ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7',
                        '#C4AD66', '#77BEDB', '#8C8C8C', '#A2C7A5'],
}

# Hatch patterns available for the two-phase region.
# Keys are human-readable display names; values are matplotlib hatch strings.
_HATCH_OPTIONS = {
    'None':          '',
    'Diagonal':      '///',
    'Back-diagonal': 3 * '\\',
    'Vertical':      '|||',
    'Horizontal':    '---',
    'Cross':         'xxx',
    'Dots':          '...',
    'Plus':          '+++',
    'Stars':         '***',
}
# Reverse lookup: hatch string → display name
_HATCH_NAMES = {v: k for k, v in _HATCH_OPTIONS.items()}

# Legend location options shown in the right-click context menu (row-major,
# matching the spatial layout so the menu reads like a 3×3 grid of corners).
_LEGEND_LOCATIONS = [
    'upper left',  'upper center',  'upper right',
    'center left', 'center',        'center right',
    'lower left',  'lower center',  'lower right',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_map(phases):
    """Return {phase.name: color_str} using the palette."""
    return {p.name: _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
            for i, p in enumerate(phases)}


def _compute_ylim(precomputed):
    """Return (ymin, ymax) with 5 % margin over all G values in precomputed."""
    all_G = np.concatenate([G for r in precomputed
                             for (_, G, _) in r.phase_curves])
    margin = 0.05 * (all_G.max() - all_G.min())
    return (all_G.min() - margin, all_G.max() + margin)


# ---------------------------------------------------------------------------
# G-x canvas
# ---------------------------------------------------------------------------

class GxCanvas(FigureCanvasQTAgg):
    """Left panel: Gibbs energy curves vs. composition at the current T (and P)."""

    def __init__(self, system, y_lim=None, colors=None):
        self.system = system
        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self._colors = colors if colors is not None else _color_map(system.phases)
        self._y_lim = y_lim
        self._legend_loc = 'upper left'
        self._last_result = None

    def redraw(self, result):
        self._last_result = result
        ax = self.ax
        ax.cla()

        # Draw each phase's G(x) curve.
        for x, G, phase in result.phase_curves:
            c = self._colors[phase.name]
            if phase.is_point:
                ax.plot(x, G, 'o', color=c, markersize=9,
                        label=phase.name, zorder=3)
            else:
                ax.plot(x, G, '-', color=c, linewidth=2.5, label=phase.name)

        # Lower convex envelope (dashed black).
        ax.plot(result.hull_x, result.hull_G, 'k--',
                linewidth=1.5, alpha=0.55, zorder=2)

        # Common tangent line across each two-phase region.
        for r in result.two_phase_regions:
            i0 = np.argmin(np.abs(result.hull_x - r['x0']))
            i1 = np.argmin(np.abs(result.hull_x - r['x1']))
            ax.plot([result.hull_x[i0], result.hull_x[i1]],
                    [result.hull_G[i0], result.hull_G[i1]],
                    'k-', linewidth=2.0, zorder=4)

        ax.set_xlabel('Composition  x(B)')
        ax.set_ylabel('Gibbs Energy  G')
        ax.set_title(f'G vs x     T = {result.T:.1f} K   P = {result.P:.3g}')
        ax.set_xlim(-0.02, 1.02)
        if self._y_lim is not None:
            ax.set_ylim(self._y_lim)
        ax.legend(loc=self._legend_loc, fontsize=8)
        self.draw()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addSection('Legend position')
        for loc in _LEGEND_LOCATIONS:
            action = menu.addAction(loc)
            action.setCheckable(True)
            action.setChecked(loc == self._legend_loc)
            action.triggered.connect(lambda checked, l=loc: self._set_legend_loc(l))
        menu.exec(event.globalPos())

    def _set_legend_loc(self, loc):
        self._legend_loc = loc
        if self._last_result is not None:
            self.redraw(self._last_result)


# ---------------------------------------------------------------------------
# T-x canvas
# ---------------------------------------------------------------------------

class TxCanvas(FigureCanvasQTAgg):
    """Right panel: T-x phase diagram with incremental top-down revelation."""

    def __init__(self, system, precomputed, colors=None,
                 two_phase_color=_TWO_PHASE_COLOR, two_phase_hatch=_TWO_PHASE_HATCH):
        """
        Parameters
        ----------
        system          : System
        precomputed     : list[EqResult]  sorted by T descending (T_max first)
        colors          : dict {phase_name: color_str}  or None to use default
        two_phase_color : str  hex color for two-phase regions
        two_phase_hatch : str  matplotlib hatch pattern ('' = no hatch)
        """
        self.system = system
        self.precomputed = precomputed
        self._colors = colors if colors is not None else _color_map(system.phases)
        self._two_phase_color = two_phase_color
        self._two_phase_hatch = two_phase_hatch
        self._lowest_T = system.T_initial   # lowest temperature explored so far
        self._reveal_all = False
        self._legend_loc = 'upper left'

        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(system.T_initial)
        self.draw()

    def _setup_axes(self):
        ax = self.ax
        ax.set_xlabel('Composition  x(B)')
        ax.set_ylabel('Temperature  (K)')
        ax.set_title('T-x  Phase Diagram')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(self.system.T_min, self.system.T_max)

        # Legend: one entry per non-end-member phase, plus two-phase.
        handles = [
            Patch(facecolor=self._colors[p.name], label=p.name)
            for p in self.system.phases
            if p.phase_type != 'end_member'
        ]
        handles.append(Patch(facecolor=self._two_phase_color,
                             hatch=self._two_phase_hatch or None,
                             edgecolor='black' if self._two_phase_hatch else 'none',
                             linewidth=0,
                             label='two-phase'))
        ax.legend(handles=handles, loc=self._legend_loc, fontsize=8)

    def _draw_full_diagram(self):
        """Render all pre-computed regions into the axes (called once at init)."""
        ax = self.ax
        results = self.precomputed

        for i, result in enumerate(results):
            T_top = result.T
            T_bot = (results[i + 1].T
                     if i + 1 < len(results)
                     else self.system.T_min)
            dT = T_top - T_bot

            for r in result.regions:
                x0 = r['x0']
                width = r['x1'] - r['x0']

                if r['type'] == 'two_phase':
                    ax.broken_barh(
                        [(x0, width)], (T_bot, dT),
                        facecolors=self._two_phase_color,
                        edgecolors='black' if self._two_phase_hatch else 'none',
                        linewidth=0,
                        hatch=self._two_phase_hatch or None,
                        alpha=0.9)
                else:
                    phase = self.system.phases[r['phases'][0]]
                    if phase.phase_type == 'end_member':
                        continue    # zero-width; nothing meaningful to draw
                    ax.broken_barh(
                        [(x0, width)], (T_bot, dT),
                        facecolors=self._colors[phase.name],
                        edgecolors='none',
                        alpha=0.85)

    def _add_cover_and_cursor(self, T_initial):
        """Add the white cover rectangle and cursor line at T_initial."""
        cover_height = T_initial - self.system.T_min
        self._cover = MplRectangle(
            xy=(-0.1, self.system.T_min),
            width=1.2,
            height=cover_height,
            facecolor='white', edgecolor='none', zorder=5)
        self.ax.add_patch(self._cover)

        self._cursor_line = self.ax.axhline(
            T_initial, color='black',
            linewidth=2.0, linestyle='--', zorder=10)

    def reset(self, precomputed, current_T=None):
        """Redraw canvas with new precomputed data (after secondary P-slider move).

        current_T sets the cover position; defaults to T_initial so the diagram
        starts fully covered when no primary slider has been moved yet.
        The current _reveal_all state is preserved across resets.
        """
        if current_T is None:
            current_T = self.system.T_initial
        self.precomputed = precomputed
        self._lowest_T = current_T
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(current_T)
        if self._reveal_all:
            self._cover.set_height(0)
        self.draw()

    def set_cursor(self, T):
        """Move cursor to T; shrink the cover if T is a new minimum."""
        if T < self._lowest_T:
            self._lowest_T = T
            if not self._reveal_all:
                self._cover.set_height(T - self.system.T_min)
        self._cursor_line.set_ydata([T, T])
        self.draw()

    def set_reveal_all(self, flag):
        """Show or hide the cover rectangle regardless of the slider position."""
        self._reveal_all = flag
        self._cover.set_height(0 if flag else self._lowest_T - self.system.T_min)
        self.draw()

    def recolor(self):
        """Redraw the diagram with updated colors, preserving cover/reveal state."""
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(self._lowest_T)
        if self._reveal_all:
            self._cover.set_height(0)
        self.draw()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addSection('Legend position')
        for loc in _LEGEND_LOCATIONS:
            action = menu.addAction(loc)
            action.setCheckable(True)
            action.setChecked(loc == self._legend_loc)
            action.triggered.connect(lambda checked, l=loc: self._set_legend_loc(l))
        menu.exec(event.globalPos())

    def _set_legend_loc(self, loc):
        self._legend_loc = loc
        leg = self.ax.get_legend()
        if leg is not None:
            leg.set_loc(loc)
            self.draw()


# ---------------------------------------------------------------------------
# P-x canvas
# ---------------------------------------------------------------------------

class PxCanvas(FigureCanvasQTAgg):
    """Right panel (Fixed T mode): P-x phase diagram with incremental revelation."""

    def __init__(self, system, precomputed, colors=None,
                 two_phase_color=_TWO_PHASE_COLOR, two_phase_hatch=_TWO_PHASE_HATCH):
        """
        Parameters
        ----------
        system          : System
        precomputed     : list[EqResult]  sorted by P descending (P_max first)
        colors          : dict {phase_name: color_str}  or None to use default
        two_phase_color : str  hex color for two-phase regions
        two_phase_hatch : str  matplotlib hatch pattern ('' = no hatch)
        """
        self.system = system
        self.precomputed = precomputed
        self._colors = colors if colors is not None else _color_map(system.phases)
        self._two_phase_color = two_phase_color
        self._two_phase_hatch = two_phase_hatch
        self._lowest_P = system.P_initial   # lowest pressure explored so far
        self._reveal_all = False
        self._legend_loc = 'upper left'

        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(system.P_initial)
        self.draw()

    def _setup_axes(self):
        ax = self.ax
        ax.set_xlabel('Composition  x(B)')
        ax.set_ylabel('Pressure')
        ax.set_title('P-x  Phase Diagram')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(self.system.P_min, self.system.P_max)

        handles = [
            Patch(facecolor=self._colors[p.name], label=p.name)
            for p in self.system.phases
            if p.phase_type != 'end_member'
        ]
        handles.append(Patch(facecolor=self._two_phase_color,
                             hatch=self._two_phase_hatch or None,
                             edgecolor='black' if self._two_phase_hatch else 'none',
                             linewidth=0,
                             label='two-phase'))
        ax.legend(handles=handles, loc=self._legend_loc, fontsize=8)

    def _draw_full_diagram(self):
        """Render all pre-computed regions into the axes (called once at init)."""
        ax = self.ax
        results = self.precomputed

        for i, result in enumerate(results):
            P_top = result.P
            P_bot = (results[i + 1].P
                     if i + 1 < len(results)
                     else self.system.P_min)
            dP = P_top - P_bot

            for r in result.regions:
                x0 = r['x0']
                width = r['x1'] - r['x0']

                if r['type'] == 'two_phase':
                    ax.broken_barh(
                        [(x0, width)], (P_bot, dP),
                        facecolors=self._two_phase_color,
                        edgecolors='black' if self._two_phase_hatch else 'none',
                        linewidth=0,
                        hatch=self._two_phase_hatch or None,
                        alpha=0.9)
                else:
                    phase = self.system.phases[r['phases'][0]]
                    if phase.phase_type == 'end_member':
                        continue
                    ax.broken_barh(
                        [(x0, width)], (P_bot, dP),
                        facecolors=self._colors[phase.name],
                        edgecolors='none',
                        alpha=0.85)

    def _add_cover_and_cursor(self, P_initial):
        """Add the white cover rectangle and cursor line at P_initial."""
        cover_height = P_initial - self.system.P_min
        self._cover = MplRectangle(
            xy=(-0.1, self.system.P_min),
            width=1.2,
            height=cover_height,
            facecolor='white', edgecolor='none', zorder=5)
        self.ax.add_patch(self._cover)

        self._cursor_line = self.ax.axhline(
            P_initial, color='black',
            linewidth=2.0, linestyle='--', zorder=10)

    def reset(self, precomputed, current_P=None):
        """Redraw canvas with new precomputed data (after secondary T-slider move).

        current_P sets the cover position; defaults to P_initial so the diagram
        starts fully covered when no primary slider has been moved yet.
        The current _reveal_all state is preserved across resets.
        """
        if current_P is None:
            current_P = self.system.P_initial
        self.precomputed = precomputed
        self._lowest_P = current_P
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(current_P)
        if self._reveal_all:
            self._cover.set_height(0)
        self.draw()

    def set_cursor(self, P):
        """Move cursor to P; shrink the cover if P is a new minimum."""
        if P < self._lowest_P:
            self._lowest_P = P
            if not self._reveal_all:
                self._cover.set_height(P - self.system.P_min)
        self._cursor_line.set_ydata([P, P])
        self.draw()

    def set_reveal_all(self, flag):
        """Show or hide the cover rectangle regardless of the slider position."""
        self._reveal_all = flag
        self._cover.set_height(0 if flag else self._lowest_P - self.system.P_min)
        self.draw()

    def recolor(self):
        """Redraw the diagram with updated colors, preserving cover/reveal state."""
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(self._lowest_P)
        if self._reveal_all:
            self._cover.set_height(0)
        self.draw()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addSection('Legend position')
        for loc in _LEGEND_LOCATIONS:
            action = menu.addAction(loc)
            action.setCheckable(True)
            action.setChecked(loc == self._legend_loc)
            action.triggered.connect(lambda checked, l=loc: self._set_legend_loc(l))
        menu.exec(event.globalPos())

    def _set_legend_loc(self, loc):
        self._legend_loc = loc
        leg = self.ax.get_legend()
        if leg is not None:
            leg.set_loc(loc)
            self.draw()


# ---------------------------------------------------------------------------
# Background worker: full T-P-x grid
# ---------------------------------------------------------------------------

class FullGridWorker(QThread):
    """Compute the full N_T_STEPS × N_P_STEPS grid in a background thread.

    Supports pause/resume via a threading.Event and clean abort via a flag.
    """

    progress = Signal(int, int)   # (n_done, n_total)
    finished = Signal(object)     # grid: list[list[EqResult]]

    def __init__(self, system, T_values, P_values):
        super().__init__()
        self._system      = system
        self._T_values    = T_values
        self._P_values    = P_values
        self._run_event   = threading.Event()
        self._run_event.set()   # start in the running state
        self._abort       = False

    def pause(self):
        """Block the worker at the next loop iteration."""
        self._run_event.clear()

    def resume(self):
        """Unblock the worker."""
        self._run_event.set()

    def abort(self):
        """Ask the worker to stop; unblocks it if currently paused."""
        self._abort = True
        self._run_event.set()

    def run(self):
        n_total = len(self._T_values) * len(self._P_values)
        done = 0
        grid = []
        for T in self._T_values:        # outer: N_T_STEPS rows
            row = []
            for P in self._P_values:    # inner: N_P_STEPS cols
                self._run_event.wait()  # blocks here while paused
                if self._abort:
                    return
                row.append(compute_equilibrium(self._system, T, P))
                done += 1
                self.progress.emit(done, n_total)
            grid.append(row)
        self.finished.emit(grid)


# ---------------------------------------------------------------------------
# Color selection dialog
# ---------------------------------------------------------------------------

def _swatch_style(hex_color):
    """Return a stylesheet string for a color-swatch QPushButton."""
    return (f'background-color: {hex_color}; '
            'border: 1px solid #888; '
            'min-width: 48px; min-height: 20px;')


class ColorDialog(QDialog):
    """Non-modal dialog for choosing per-phase colors and palette presets."""

    colors_changed = Signal(dict, str, str)   # (phase_colors, two_phase_color, two_phase_hatch)

    def __init__(self, system, phase_colors, two_phase_color,
                 two_phase_hatch=_TWO_PHASE_HATCH, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Phase Colors')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)

        # Working copies (edits are applied immediately via the signal).
        self._phase_colors    = dict(phase_colors)
        self._two_phase_color = two_phase_color
        self._two_phase_hatch = two_phase_hatch

        # ---- palette row ----
        palette_lbl  = QLabel('Palette:')
        self._palette_cb  = QComboBox()
        for name in _PALETTES:
            self._palette_cb.addItem(name)
        self._palette_cb.setCurrentText('Muted')
        apply_btn = QPushButton('Apply palette')
        apply_btn.clicked.connect(self._on_apply_palette)

        palette_row = QHBoxLayout()
        palette_row.addWidget(palette_lbl)
        palette_row.addWidget(self._palette_cb)
        palette_row.addWidget(apply_btn)
        palette_row.addStretch()

        # ---- per-phase swatch grid (skip end_members) ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self._phase_btns = {}   # phase_name → QPushButton
        col = 0
        row = 0
        for phase in system.phases:
            if phase.phase_type == 'end_member':
                continue
            lbl = QLabel(phase.name)
            btn = QPushButton()
            btn.setStyleSheet(_swatch_style(self._phase_colors[phase.name]))
            btn.setFixedWidth(56)
            # Use default arg capture to bind phase.name correctly in closure.
            btn.clicked.connect(lambda checked, n=phase.name: self._on_phase_swatch(n))
            self._phase_btns[phase.name] = btn
            grid.addWidget(lbl, row, col * 2)
            grid.addWidget(btn, row, col * 2 + 1)
            col += 1
            if col >= 2:
                col = 0
                row += 1

        # ---- two-phase swatch + hatch selector ----
        two_phase_lbl = QLabel('Two-phase region:')
        self._two_phase_btn = QPushButton()
        self._two_phase_btn.setStyleSheet(_swatch_style(self._two_phase_color))
        self._two_phase_btn.setFixedWidth(56)
        self._two_phase_btn.clicked.connect(self._on_two_phase_swatch)

        hatch_lbl = QLabel('Hatch:')
        self._hatch_cb = QComboBox()
        for name in _HATCH_OPTIONS:
            self._hatch_cb.addItem(name)
        # Set combobox to the current hatch value.
        current_hatch_name = _HATCH_NAMES.get(self._two_phase_hatch, 'Diagonal')
        self._hatch_cb.setCurrentText(current_hatch_name)
        self._hatch_cb.currentTextChanged.connect(self._on_hatch_changed)

        two_phase_row = QHBoxLayout()
        two_phase_row.addWidget(two_phase_lbl)
        two_phase_row.addWidget(self._two_phase_btn)
        two_phase_row.addSpacing(16)
        two_phase_row.addWidget(hatch_lbl)
        two_phase_row.addWidget(self._hatch_cb)
        two_phase_row.addStretch()

        # ---- close button ----
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.close)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_btn)

        # ---- assemble ----
        root = QVBoxLayout()
        root.addLayout(palette_row)
        root.addLayout(grid)
        root.addLayout(two_phase_row)
        root.addLayout(close_row)
        self.setLayout(root)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _emit(self):
        """Emit colors_changed with current state."""
        self.colors_changed.emit(
            dict(self._phase_colors), self._two_phase_color, self._two_phase_hatch)

    def _on_apply_palette(self):
        name = self._palette_cb.currentText()
        colors = _PALETTES[name]
        # Apply to non-end-member phases in order.
        phases = [p for p in self.parent().system.phases
                  if p.phase_type != 'end_member']
        for i, phase in enumerate(phases):
            hex_color = colors[i % len(colors)]
            self._phase_colors[phase.name] = hex_color
            self._phase_btns[phase.name].setStyleSheet(_swatch_style(hex_color))
        self._emit()

    def _on_phase_swatch(self, phase_name):
        from PySide6.QtGui import QColor
        initial = QColor(self._phase_colors[phase_name])
        color = QColorDialog.getColor(initial, self, f'Color for {phase_name}')
        if not color.isValid():
            return
        hex_color = color.name()
        self._phase_colors[phase_name] = hex_color
        self._phase_btns[phase_name].setStyleSheet(_swatch_style(hex_color))
        self._emit()

    def _on_two_phase_swatch(self):
        from PySide6.QtGui import QColor
        initial = QColor(self._two_phase_color)
        color = QColorDialog.getColor(initial, self, 'Color for two-phase region')
        if not color.isValid():
            return
        self._two_phase_color = color.name()
        self._two_phase_btn.setStyleSheet(_swatch_style(self._two_phase_color))
        self._emit()

    def _on_hatch_changed(self, name):
        self._two_phase_hatch = _HATCH_OPTIONS[name]
        self._emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self, system, precomputed_Tx, precomputed_Px=None):
        super().__init__()
        self.system = system
        self.setWindowTitle('PDE — Phase Diagram Energy')

        self._precomputed_Tx = precomputed_Tx
        self._precomputed_Px = precomputed_Px

        self._T_arr = np.array([r.T for r in precomputed_Tx])
        # If P-x was pre-computed, use its actual P values; otherwise build the
        # array from linspace so the full-grid code paths have a valid _P_arr
        # even before the P-x diagram is first computed.
        if precomputed_Px is not None:
            self._P_arr = np.array([r.P for r in precomputed_Px])
        elif system.has_pressure:
            self._P_arr = np.linspace(system.P_max, system.P_min, N_P_STEPS)
        else:
            self._P_arr = None

        # Current mode: 'fixed_P' (T is primary) or 'fixed_T' (P is primary).
        self._mode = 'fixed_P'

        # Full T-P-x grid cache (None until background worker finishes).
        self._full_grid    = None
        self._worker       = None   # kept alive to prevent GC during thread run
        self._worker_state = 'idle'  # 'idle' | 'running' | 'paused' | 'done'

        # ---- shared color state (shared dict propagates to all canvases) ----
        self._colors          = _color_map(system.phases)
        self._two_phase_color = _TWO_PHASE_COLOR
        self._two_phase_hatch = _TWO_PHASE_HATCH
        self._color_dialog    = None   # single dialog instance

        # ---- G-x y-limits from the Tx precomputed data ----
        y_lim = _compute_ylim(precomputed_Tx)

        # ---- canvases ----
        self.gx_canvas = GxCanvas(system, y_lim, colors=self._colors)
        self.tx_canvas = TxCanvas(system, precomputed_Tx,
                                  colors=self._colors,
                                  two_phase_color=self._two_phase_color,
                                  two_phase_hatch=self._two_phase_hatch)
        self.px_canvas = (PxCanvas(system, precomputed_Px,
                                   colors=self._colors,
                                   two_phase_color=self._two_phase_color,
                                   two_phase_hatch=self._two_phase_hatch)
                          if precomputed_Px is not None else None)

        # ---- right-panel stacked widget ----
        self._right_stack = QStackedWidget()
        self._right_stack.addWidget(self.tx_canvas)        # index 0
        if self.px_canvas is not None:
            self._right_stack.addWidget(self.px_canvas)    # index 1
        self._right_stack.setCurrentIndex(0)

        # ---- T slider ----
        self.T_slider = QSlider(Qt.Horizontal)
        self.T_slider.setMinimum(int(system.T_min))
        self.T_slider.setMaximum(int(system.T_max))
        self.T_slider.setValue(int(system.T_initial))
        self.T_slider.setSingleStep(1)
        self.T_slider.setPageStep(10)
        self.T_label = QLabel(f'T = {system.T_initial:.0f} K')
        self.T_label.setMinimumWidth(90)

        # ---- P slider (only when has_pressure) ----
        self.P_slider = None
        self.P_label = None
        if system.has_pressure:
            self.P_slider = QSlider(Qt.Horizontal)
            self.P_slider.setMinimum(0)
            self.P_slider.setMaximum(N_P_STEPS - 1)
            P_range = max(system.P_max - system.P_min, 1e-30)
            P_frac = (system.P_initial - system.P_min) / P_range
            self.P_slider.setValue(int(round(P_frac * (N_P_STEPS - 1))))
            self.P_label = QLabel(f'P = {system.P_initial:.3g} {system.P_unit}'.strip())
            self.P_label.setMinimumWidth(90)

        # ---- mode selector (only when has_pressure) ----
        self._mode_group = None
        if system.has_pressure:
            radio_fixed_P = QRadioButton('Fixed P  (T-x)')
            radio_fixed_T = QRadioButton('Fixed T  (P-x)')
            radio_fixed_P.setChecked(True)
            self._mode_group = QButtonGroup()
            self._mode_group.addButton(radio_fixed_P, 0)
            self._mode_group.addButton(radio_fixed_T, 1)

        # ---- shared controls ----
        self._reveal_cb  = QCheckBox('Reveal all')
        self._colors_btn = QPushButton('Colors\u2026')

        # ---- pre-compute widgets (only when pressure is active) ----
        self._precompute_btn    = None
        self._precompute_bar    = None
        self._precompute_status = None
        if system.has_pressure:
            self._precompute_btn = QPushButton('Pre-compute full T-P-x')
            n_total = N_T_STEPS * N_P_STEPS
            self._precompute_bar = QProgressBar()
            self._precompute_bar.setRange(0, n_total)
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(False)
            self._precompute_status = QLabel('')

        # ---- layout ----
        # Top row: mode toggles (if pressure) | Reveal all | Colors… | precompute (if pressure)
        top_row = QHBoxLayout()
        if system.has_pressure:
            top_row.addWidget(radio_fixed_P)
            top_row.addWidget(radio_fixed_T)
            top_row.addSpacing(16)
        top_row.addWidget(self._reveal_cb)
        top_row.addWidget(self._colors_btn)
        if system.has_pressure:
            top_row.addSpacing(16)
            top_row.addWidget(self._precompute_btn)
            top_row.addWidget(self._precompute_bar, stretch=1)
            top_row.addWidget(self._precompute_status)
        top_row.addStretch()

        canvas_row = QHBoxLayout()
        canvas_row.addWidget(self.gx_canvas)
        canvas_row.addWidget(self._right_stack)

        T_slider_row = QHBoxLayout()
        T_slider_row.addWidget(QLabel(f'{system.T_min:.0f} K'))
        T_slider_row.addWidget(self.T_slider)
        T_slider_row.addWidget(QLabel(f'{system.T_max:.0f} K'))
        T_slider_row.addWidget(self.T_label)

        root = QVBoxLayout()
        root.addLayout(top_row)
        root.addLayout(canvas_row)
        root.addLayout(T_slider_row)

        if self.P_slider is not None:
            P_slider_row = QHBoxLayout()
            P_slider_row.addWidget(QLabel(f'{system.P_min:.3g} {system.P_unit}'.strip()))
            P_slider_row.addWidget(self.P_slider)
            P_slider_row.addWidget(QLabel(f'{system.P_max:.3g} {system.P_unit}'.strip()))
            P_slider_row.addWidget(self.P_label)
            root.addLayout(P_slider_row)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # ---- signal connections ----
        self.T_slider.valueChanged.connect(self._on_T_changed)
        self.T_slider.sliderReleased.connect(self._on_T_released)
        self.T_slider.actionTriggered.connect(self._on_T_action)
        if self.P_slider is not None:
            self.P_slider.valueChanged.connect(self._on_P_changed)
            self.P_slider.sliderReleased.connect(self._on_P_released)
            self.P_slider.actionTriggered.connect(self._on_P_action)
            self._mode_group.idClicked.connect(self._on_mode_changed)
        if self._precompute_btn is not None:
            self._precompute_btn.clicked.connect(self._on_precompute_clicked)
        self._reveal_cb.toggled.connect(self._on_reveal_all_toggled)
        self._colors_btn.clicked.connect(self._on_colors_clicked)

        # Render initial state.
        self._on_T_changed(int(system.T_initial))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _P_from_tick(self, tick):
        """Map slider tick 0…N_P_STEPS-1 → pressure value."""
        frac = tick / (N_P_STEPS - 1)
        return self.system.P_min + frac * (self.system.P_max - self.system.P_min)

    def _current_P(self):
        if self.P_slider is None:
            return 0.0
        return self._P_from_tick(self.P_slider.value())

    def _current_T(self):
        return float(self.T_slider.value())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_T_changed(self, T_int):
        T = float(T_int)
        self.T_label.setText(f'T = {T:.0f} K')
        if self._mode == 'fixed_P':
            # T is the primary slider — fast O(1) canvas update.
            idx = int(np.argmin(np.abs(self._T_arr - T)))
            self.gx_canvas.redraw(self._precomputed_Tx[idx])
            self.tx_canvas.set_cursor(T)
        elif self._full_grid is not None:
            # T is the secondary slider and grid is cached — instant update.
            T_idx = int(np.argmin(np.abs(self._T_arr - T)))
            new_Px = [self._full_grid[T_idx][i_P] for i_P in range(N_P_STEPS)]
            self._precomputed_Px = new_Px
            self._P_arr = np.array([r.P for r in new_Px])
            self.gx_canvas._y_lim = _compute_ylim(new_Px)
            P = self._current_P()
            self.px_canvas.reset(new_Px, current_P=P)
            idx = int(np.argmin(np.abs(self._P_arr - P)))
            self.gx_canvas.redraw(new_Px[idx])
            self.px_canvas.set_cursor(P)
        # Otherwise T is the secondary slider with no cache: label updates here;
        # canvas update (recompute) fires in _on_T_released.

    def _on_T_released(self):
        """Recompute P-x diagram when T slider is released in Fixed T mode."""
        if self._mode != 'fixed_T':
            return
        T = self._current_T()
        if self._full_grid is not None:
            T_idx = int(np.argmin(np.abs(self._T_arr - T)))
            new_Px = [self._full_grid[T_idx][i_P] for i_P in range(N_P_STEPS)]
        else:
            print(f'Recomputing P-x diagram at T = {T:.1f} K...', end=' ', flush=True)
            new_Px = precompute_Px_diagram(self.system, T)
            print('done.')
        self._precomputed_Px = new_Px
        self._P_arr = np.array([r.P for r in new_Px])
        self.gx_canvas._y_lim = _compute_ylim(new_Px)
        P = self._current_P()
        self.px_canvas.reset(new_Px, current_P=P)
        idx = int(np.argmin(np.abs(self._P_arr - P)))
        self.gx_canvas.redraw(new_Px[idx])
        self.px_canvas.set_cursor(P)

    def _on_P_changed(self, tick):
        P = self._P_from_tick(tick)
        self.P_label.setText(f'P = {P:.3g} {self.system.P_unit}'.strip())
        if self._mode == 'fixed_T':
            # P is the primary slider — fast O(1) canvas update.
            idx = int(np.argmin(np.abs(self._P_arr - P)))
            self.gx_canvas.redraw(self._precomputed_Px[idx])
            self.px_canvas.set_cursor(P)
        elif self._full_grid is not None:
            # P is the secondary slider and grid is cached — instant update.
            P_idx = int(np.argmin(np.abs(self._P_arr - P)))
            new_Tx = [self._full_grid[i_T][P_idx] for i_T in range(N_T_STEPS)]
            self._precomputed_Tx = new_Tx
            self._T_arr = np.array([r.T for r in new_Tx])
            self.gx_canvas._y_lim = _compute_ylim(new_Tx)
            T = self._current_T()
            self.tx_canvas.reset(new_Tx, current_T=T)
            idx = int(np.argmin(np.abs(self._T_arr - T)))
            self.gx_canvas.redraw(new_Tx[idx])
            self.tx_canvas.set_cursor(T)
        # Otherwise P is the secondary slider with no cache: label updates here;
        # canvas update (recompute) fires in _on_P_released.

    def _on_P_released(self):
        """Recompute T-x diagram when P slider is released in Fixed P mode."""
        if self._mode != 'fixed_P':
            return
        P = self._P_from_tick(self.P_slider.value())
        if self._full_grid is not None:
            P_idx = int(np.argmin(np.abs(self._P_arr - P)))
            new_Tx = [self._full_grid[i_T][P_idx] for i_T in range(N_T_STEPS)]
        else:
            print(f'Recomputing T-x diagram at P = {P:.3g}...', end=' ', flush=True)
            new_Tx = precompute_Tx_diagram(self.system, P)
            print('done.')
        self._precomputed_Tx = new_Tx
        self._T_arr = np.array([r.T for r in new_Tx])
        self.gx_canvas._y_lim = _compute_ylim(new_Tx)
        T = self._current_T()
        self.tx_canvas.reset(new_Tx, current_T=T)
        idx = int(np.argmin(np.abs(self._T_arr - T)))
        self.gx_canvas.redraw(new_Tx[idx])
        self.tx_canvas.set_cursor(T)

    def _on_T_action(self, action):
        """Trigger recompute on bar-click / key-press of the T slider.

        Qt fires sliderReleased only at the end of a drag; for discrete
        actions (page-step bar click, arrow keys, Home/End) it fires only
        actionTriggered.  isSliderDown() is True only while the thumb is
        held — False for bar clicks — so we use it to distinguish the two
        cases without fragile enum comparisons.  We defer via singleShot so
        that valueChanged has already updated the slider value before the
        released handler reads _current_T().
        """
        if self.T_slider.isSliderDown():
            return  # drag in progress — sliderReleased will handle it
        QTimer.singleShot(0, self._on_T_released)

    def _on_P_action(self, action):
        """Trigger recompute on bar-click / key-press of the P slider."""
        if self.P_slider.isSliderDown():
            return  # drag in progress — sliderReleased will handle it
        QTimer.singleShot(0, self._on_P_released)

    def _on_mode_changed(self, button_id):
        if button_id == 0:
            self._mode = 'fixed_P'
            self._right_stack.setCurrentIndex(0)
            T = self._current_T()
            self.tx_canvas.reset(self._precomputed_Tx, current_T=T)
            idx = int(np.argmin(np.abs(self._T_arr - T)))
            self.gx_canvas.redraw(self._precomputed_Tx[idx])
            self.tx_canvas.set_cursor(T)
        else:
            self._mode = 'fixed_T'
            if self.px_canvas is None:
                # Lazy first-time computation of the P-x diagram.
                T = self._current_T()
                print(f'Computing P-x diagram at T={T:.1f} K...',
                      end=' ', flush=True)
                new_Px = precompute_Px_diagram(self.system, T)
                print('done.')
                self._precomputed_Px = new_Px
                self._P_arr = np.array([r.P for r in new_Px])
                self.gx_canvas._y_lim = _compute_ylim(new_Px)
                self.px_canvas = PxCanvas(self.system, new_Px,
                                          colors=self._colors,
                                          two_phase_color=self._two_phase_color,
                                          two_phase_hatch=self._two_phase_hatch)
                self._right_stack.addWidget(self.px_canvas)
                self.px_canvas.set_reveal_all(self._reveal_cb.isChecked())
            self._right_stack.setCurrentIndex(1)
            P = self._current_P()
            self.px_canvas.reset(self._precomputed_Px, current_P=P)
            idx = int(np.argmin(np.abs(self._P_arr - P)))
            self.gx_canvas.redraw(self._precomputed_Px[idx])
            self.px_canvas.set_cursor(P)

    def _on_reveal_all_toggled(self, checked):
        """Show or hide the cover on all phase-diagram canvases."""
        self.tx_canvas.set_reveal_all(checked)
        if self.px_canvas is not None:
            self.px_canvas.set_reveal_all(checked)

    def _current_result(self):
        """Return the EqResult nearest to the current slider position."""
        if self._mode == 'fixed_P':
            idx = int(np.argmin(np.abs(self._T_arr - self._current_T())))
            return self._precomputed_Tx[idx]
        else:
            idx = int(np.argmin(np.abs(self._P_arr - self._current_P())))
            return self._precomputed_Px[idx]

    def _apply_colors(self, phase_colors, two_phase_color, two_phase_hatch):
        """Update shared color/hatch state and redraw all canvases live."""
        self._colors.update(phase_colors)
        self._two_phase_color = two_phase_color
        self._two_phase_hatch = two_phase_hatch
        self.tx_canvas._two_phase_color = two_phase_color
        self.tx_canvas._two_phase_hatch = two_phase_hatch
        if self.px_canvas is not None:
            self.px_canvas._two_phase_color = two_phase_color
            self.px_canvas._two_phase_hatch = two_phase_hatch
        self.tx_canvas.recolor()
        if self.px_canvas is not None:
            self.px_canvas.recolor()
        self.gx_canvas.redraw(self._current_result())

    def _on_colors_clicked(self):
        """Open (or raise) the color selection dialog."""
        if self._color_dialog is not None and self._color_dialog.isVisible():
            self._color_dialog.raise_()
            return
        self._color_dialog = ColorDialog(
            self.system, self._colors, self._two_phase_color,
            self._two_phase_hatch, parent=self)
        self._color_dialog.colors_changed.connect(self._apply_colors)
        self._color_dialog.show()

    def _on_precompute_clicked(self):
        """Toggle pre-computation: start → pause → resume → (done)."""
        if self._worker_state == 'idle':
            # Start a fresh computation.
            n_total = N_T_STEPS * N_P_STEPS
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(True)
            self._precompute_status.setText('Computing... 0%')
            T_values = np.linspace(self.system.T_max, self.system.T_min, N_T_STEPS)
            P_values = np.linspace(self.system.P_max, self.system.P_min, N_P_STEPS)
            self._worker = FullGridWorker(self.system, T_values, P_values)
            self._worker.progress.connect(self._on_grid_progress)
            self._worker.finished.connect(self._on_grid_ready)
            self._worker.start()
            self._worker_state = 'running'
            self._precompute_btn.setText('Pause Computation')

        elif self._worker_state == 'running':
            # Pause the running worker.
            self._worker.pause()
            self._worker_state = 'paused'
            self._precompute_btn.setText('Restart Computation')

        elif self._worker_state == 'paused':
            # Resume from where we left off.
            self._worker.resume()
            self._worker_state = 'running'
            self._precompute_btn.setText('Pause Computation')

    def _on_grid_progress(self, done, total):
        self._precompute_bar.setValue(done)
        pct = int(100 * done / total)
        self._precompute_status.setText(f'Computing... {pct}%')

    def _on_grid_ready(self, grid):
        self._full_grid = grid
        self._worker_state = 'done'
        self._precompute_bar.setVisible(False)
        n_total = N_T_STEPS * N_P_STEPS
        self._precompute_status.setText(f'Cached ({n_total:,} evaluations)')
        self._precompute_btn.setText('Full T-P-x cached')
        self._precompute_btn.setEnabled(False)

    def closeEvent(self, event):
        """Abort any running/paused background worker before closing."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def precompute_Tx_diagram(system, P=0.0, n_steps=N_T_STEPS):
    """Sweep T_max → T_min at fixed P; return list[EqResult] (T descending)."""
    T_values = np.linspace(system.T_max, system.T_min, n_steps)
    return [compute_equilibrium(system, T, P) for T in T_values]


# Backward-compatible alias.
precompute_diagram = precompute_Tx_diagram


def precompute_Px_diagram(system, T, n_steps=N_P_STEPS):
    """Sweep P_max → P_min at fixed T; return list[EqResult] (P descending)."""
    P_values = np.linspace(system.P_max, system.P_min, n_steps)
    return [compute_equilibrium(system, T, P) for P in P_values]


def launch_ui(system):
    """Pre-compute the phase diagram(s) and open the interactive window."""
    if system.has_pressure:
        print(
            f'Pre-computing T-x diagram at P={system.P_initial:.3g} '
            f'({N_T_STEPS} evaluations)...',
            end=' ', flush=True)
        precomputed_Tx = precompute_Tx_diagram(system, system.P_initial)
        precomputed_Px = None
        print('done.')
    else:
        print('Pre-computing phase diagram...', end=' ', flush=True)
        precomputed_Tx = precompute_Tx_diagram(system)
        precomputed_Px = None
        print('done.')

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(system, precomputed_Tx, precomputed_Px)
    window.resize(1200, 650)
    window.show()
    sys.exit(app.exec())
