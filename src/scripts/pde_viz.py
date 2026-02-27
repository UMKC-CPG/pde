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

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle as MplRectangle
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout,
                                QLabel, QMainWindow, QProgressBar, QPushButton,
                                QRadioButton, QSlider, QStackedWidget,
                                QVBoxLayout, QWidget)

from pde_compute import compute_equilibrium


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_T_STEPS = 200     # temperature steps for the pre-computed T-x diagram
N_P_STEPS = 200     # pressure steps for the pre-computed P-x diagram

# One colour per phase (assigned by position in system.phases)
_COLOR_PALETTE = [
    '#E8A020',  # amber  — gas
    '#3A7DBC',  # blue   — liquid
    '#C94040',  # red    — alpha
    '#3AA85A',  # green  — beta
    '#C9A030',  # gold   — gamma
    '#8A40C9',  # purple — delta
    '#C97030',  # orange — epsilon
    '#40B8C9',  # teal   — zeta
]

_TWO_PHASE_COLOR = '#D8D8D8'    # light grey fill for two-phase regions
_TWO_PHASE_HATCH = '///'        # diagonal hatch overlay


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

    def __init__(self, system, y_lim=None):
        self.system = system
        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self._colors = _color_map(system.phases)
        self._y_lim = y_lim

    def redraw(self, result):
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
        ax.legend(loc='upper right', fontsize=8)
        self.draw()


# ---------------------------------------------------------------------------
# T-x canvas
# ---------------------------------------------------------------------------

class TxCanvas(FigureCanvasQTAgg):
    """Right panel: T-x phase diagram with incremental top-down revelation."""

    def __init__(self, system, precomputed):
        """
        Parameters
        ----------
        system      : System
        precomputed : list[EqResult]  sorted by T descending (T_max first)
        """
        self.system = system
        self.precomputed = precomputed
        self._colors = _color_map(system.phases)
        self._lowest_T = system.T_initial   # lowest temperature explored so far

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
        handles.append(Patch(facecolor=_TWO_PHASE_COLOR,
                             hatch=_TWO_PHASE_HATCH,
                             edgecolor='grey',
                             label='two-phase'))
        ax.legend(handles=handles, loc='upper right', fontsize=8)

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
                        facecolors=_TWO_PHASE_COLOR,
                        edgecolors='grey',
                        hatch=_TWO_PHASE_HATCH,
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

    def reset(self, precomputed):
        """Redraw canvas with new precomputed data (after secondary P-slider move)."""
        self.precomputed = precomputed
        self._lowest_T = self.system.T_initial
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(self.system.T_initial)
        self.draw()

    def set_cursor(self, T):
        """Move cursor to T; shrink the cover if T is a new minimum."""
        if T < self._lowest_T:
            self._lowest_T = T
            self._cover.set_height(T - self.system.T_min)
        self._cursor_line.set_ydata([T, T])
        self.draw()


# ---------------------------------------------------------------------------
# P-x canvas
# ---------------------------------------------------------------------------

class PxCanvas(FigureCanvasQTAgg):
    """Right panel (Fixed T mode): P-x phase diagram with incremental revelation."""

    def __init__(self, system, precomputed):
        """
        Parameters
        ----------
        system      : System
        precomputed : list[EqResult]  sorted by P descending (P_max first)
        """
        self.system = system
        self.precomputed = precomputed
        self._colors = _color_map(system.phases)
        self._lowest_P = system.P_initial   # lowest pressure explored so far

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
        handles.append(Patch(facecolor=_TWO_PHASE_COLOR,
                             hatch=_TWO_PHASE_HATCH,
                             edgecolor='grey',
                             label='two-phase'))
        ax.legend(handles=handles, loc='upper right', fontsize=8)

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
                        facecolors=_TWO_PHASE_COLOR,
                        edgecolors='grey',
                        hatch=_TWO_PHASE_HATCH,
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

    def reset(self, precomputed):
        """Redraw canvas with new precomputed data (after secondary T-slider move)."""
        self.precomputed = precomputed
        self._lowest_P = self.system.P_initial
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(self.system.P_initial)
        self.draw()

    def set_cursor(self, P):
        """Move cursor to P; shrink the cover if P is a new minimum."""
        if P < self._lowest_P:
            self._lowest_P = P
            self._cover.set_height(P - self.system.P_min)
        self._cursor_line.set_ydata([P, P])
        self.draw()


# ---------------------------------------------------------------------------
# Background worker: full T-P-x grid
# ---------------------------------------------------------------------------

class FullGridWorker(QThread):
    """Compute the full N_T_STEPS × N_P_STEPS grid in a background thread."""

    progress = Signal(int, int)   # (n_done, n_total)
    finished = Signal(object)     # grid: list[list[EqResult]]

    def __init__(self, system, T_values, P_values):
        super().__init__()
        self._system   = system
        self._T_values = T_values
        self._P_values = P_values

    def run(self):
        n_total = len(self._T_values) * len(self._P_values)
        done = 0
        grid = []
        for T in self._T_values:        # outer: N_T_STEPS rows
            row = []
            for P in self._P_values:    # inner: N_P_STEPS cols
                row.append(compute_equilibrium(self._system, T, P))
                done += 1
                self.progress.emit(done, n_total)
            grid.append(row)
        self.finished.emit(grid)


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
        self._P_arr = (np.array([r.P for r in precomputed_Px])
                       if precomputed_Px is not None else None)

        # Current mode: 'fixed_P' (T is primary) or 'fixed_T' (P is primary).
        self._mode = 'fixed_P'

        # Full T-P-x grid cache (None until background worker finishes).
        self._full_grid = None
        self._worker    = None   # kept alive to prevent GC during thread run

        # ---- G-x y-limits from the Tx precomputed data ----
        y_lim = _compute_ylim(precomputed_Tx)

        # ---- canvases ----
        self.gx_canvas = GxCanvas(system, y_lim)
        self.tx_canvas = TxCanvas(system, precomputed_Tx)
        self.px_canvas = (PxCanvas(system, precomputed_Px)
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
        if system.has_pressure and precomputed_Px is not None:
            self.P_slider = QSlider(Qt.Horizontal)
            self.P_slider.setMinimum(0)
            self.P_slider.setMaximum(N_P_STEPS - 1)
            P_range = max(system.P_max - system.P_min, 1e-30)
            P_frac = (system.P_initial - system.P_min) / P_range
            self.P_slider.setValue(int(round(P_frac * (N_P_STEPS - 1))))
            self.P_label = QLabel(f'P = {system.P_initial:.3g}')
            self.P_label.setMinimumWidth(90)

        # ---- mode selector (only when has_pressure) ----
        self._mode_group = None
        mode_row = None
        if system.has_pressure and precomputed_Px is not None:
            radio_fixed_P = QRadioButton('Fixed P  (T-x)')
            radio_fixed_T = QRadioButton('Fixed T  (P-x)')
            radio_fixed_P.setChecked(True)
            self._mode_group = QButtonGroup()
            self._mode_group.addButton(radio_fixed_P, 0)
            self._mode_group.addButton(radio_fixed_T, 1)
            mode_row = QHBoxLayout()
            mode_row.addStretch()
            mode_row.addWidget(radio_fixed_P)
            mode_row.addWidget(radio_fixed_T)
            mode_row.addStretch()

        # ---- layout ----
        canvas_row = QHBoxLayout()
        canvas_row.addWidget(self.gx_canvas)
        canvas_row.addWidget(self._right_stack)

        T_slider_row = QHBoxLayout()
        T_slider_row.addWidget(QLabel(f'{system.T_min:.0f} K'))
        T_slider_row.addWidget(self.T_slider)
        T_slider_row.addWidget(QLabel(f'{system.T_max:.0f} K'))
        T_slider_row.addWidget(self.T_label)

        root = QVBoxLayout()
        if mode_row is not None:
            root.addLayout(mode_row)
        root.addLayout(canvas_row)
        root.addLayout(T_slider_row)

        if self.P_slider is not None:
            P_slider_row = QHBoxLayout()
            P_slider_row.addWidget(QLabel(f'{system.P_min:.3g}'))
            P_slider_row.addWidget(self.P_slider)
            P_slider_row.addWidget(QLabel(f'{system.P_max:.3g}'))
            P_slider_row.addWidget(self.P_label)
            root.addLayout(P_slider_row)

        # ---- pre-compute row (only when pressure is active) ----
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

            precompute_row = QHBoxLayout()
            precompute_row.addWidget(self._precompute_btn)
            precompute_row.addWidget(self._precompute_bar, stretch=1)
            precompute_row.addWidget(self._precompute_status)
            root.addLayout(precompute_row)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # ---- signal connections ----
        self.T_slider.valueChanged.connect(self._on_T_changed)
        self.T_slider.sliderReleased.connect(self._on_T_released)
        if self.P_slider is not None:
            self.P_slider.valueChanged.connect(self._on_P_changed)
            self.P_slider.sliderReleased.connect(self._on_P_released)
            self._mode_group.idClicked.connect(self._on_mode_changed)
        if self._precompute_btn is not None:
            self._precompute_btn.clicked.connect(self._on_precompute_clicked)

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
            self.px_canvas.reset(new_Px)
            P = self._current_P()
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
        self.px_canvas.reset(new_Px)
        P = self._current_P()
        idx = int(np.argmin(np.abs(self._P_arr - P)))
        self.gx_canvas.redraw(new_Px[idx])
        self.px_canvas.set_cursor(P)

    def _on_P_changed(self, tick):
        P = self._P_from_tick(tick)
        self.P_label.setText(f'P = {P:.3g}')
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
            self.tx_canvas.reset(new_Tx)
            T = self._current_T()
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
        self.tx_canvas.reset(new_Tx)
        T = self._current_T()
        idx = int(np.argmin(np.abs(self._T_arr - T)))
        self.gx_canvas.redraw(new_Tx[idx])
        self.tx_canvas.set_cursor(T)

    def _on_mode_changed(self, button_id):
        if button_id == 0:
            self._mode = 'fixed_P'
            self._right_stack.setCurrentIndex(0)
            # Reset T-x canvas to initial state, then reveal up to current T.
            self.tx_canvas.reset(self._precomputed_Tx)
            T = self._current_T()
            idx = int(np.argmin(np.abs(self._T_arr - T)))
            self.gx_canvas.redraw(self._precomputed_Tx[idx])
            self.tx_canvas.set_cursor(T)
        else:
            self._mode = 'fixed_T'
            self._right_stack.setCurrentIndex(1)
            # Reset P-x canvas to initial state, then reveal up to current P.
            self.px_canvas.reset(self._precomputed_Px)
            P = self._current_P()
            idx = int(np.argmin(np.abs(self._P_arr - P)))
            self.gx_canvas.redraw(self._precomputed_Px[idx])
            self.px_canvas.set_cursor(P)

    def _on_precompute_clicked(self):
        """Start background computation of the full T-P-x grid."""
        self._precompute_btn.setEnabled(False)
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

    def _on_grid_progress(self, done, total):
        self._precompute_bar.setValue(done)
        pct = int(100 * done / total)
        self._precompute_status.setText(f'Computing... {pct}%')

    def _on_grid_ready(self, grid):
        self._full_grid = grid
        self._precompute_bar.setVisible(False)
        n_total = N_T_STEPS * N_P_STEPS
        self._precompute_status.setText(f'Cached ({n_total:,} evaluations)')
        self._precompute_btn.setText('Full T-P-x cached')


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
            f'and P-x diagram at T={system.T_initial:.1f} K '
            f'({N_T_STEPS + N_P_STEPS} evaluations)...',
            end=' ', flush=True)
        precomputed_Tx = precompute_Tx_diagram(system, system.P_initial)
        precomputed_Px = precompute_Px_diagram(system, system.T_initial)
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
