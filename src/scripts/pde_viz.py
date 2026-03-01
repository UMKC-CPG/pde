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
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle as MplRectangle
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox,
                                QColorDialog, QComboBox, QDialog, QGridLayout,
                                QHBoxLayout, QLabel, QMainWindow, QMenu,
                                QProgressBar, QPushButton,
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


def _G_from_phase_data(pd, x, T):
    """Evaluate G(x, T) = H(x) − T·S(x) directly from a PhaseData object.

    Uses the same ascending-order coefficient convention as HSModel.
    x may be a scalar or a numpy array.  Only HS form is supported; returns
    zeros for any other form (callers should guard with energy_form check).
    """
    H_coeffs = pd.hs_H if pd.hs_H else [0.0]
    S_coeffs = pd.hs_S if pd.hs_S else [0.0]
    return (np.polynomial.polynomial.polyval(x, H_coeffs)
            - T * np.polynomial.polynomial.polyval(x, S_coeffs))


# ---------------------------------------------------------------------------
# Interactive edit-mode state
# ---------------------------------------------------------------------------

@dataclass
class _DragState:
    """All state for one in-progress G(x) handle drag.

    Fields are sized for future extension:
      snapshot  — PhaseData copy before the drag → enables Ctrl+Z undo (Phase 5)
      T_ref     — temperature at drag start → two-temperature H+S solve (Phase 8)
      P_ref     — pressure at drag start    → Phase 8 extension
      handle_idx — which of the 3 handles is being dragged;
                   −1 is reserved for whole-curve drag (Interaction B, future)
    """
    phase_name:   str
    handle_idx:   int     # 0 = left endpoint, 1 = midpoint, 2 = right endpoint
    y_press_data: float   # data-y at button_press_event (drag reference origin)
    snapshot:     object  # deep copy of PhaseData BEFORE the drag (for Phase 5 undo)
    T_ref:        float   # temperature when drag started (for Phase 8)
    P_ref:        float   # pressure when drag started (for Phase 8)


# ---------------------------------------------------------------------------
# G-x canvas
# ---------------------------------------------------------------------------

class GxCanvas(FigureCanvasQTAgg):
    """Left panel: Gibbs energy curves vs. composition at the current T (and P).

    Edit mode
    ---------
    When set_edit_mode('handles') is called (by MainWindow when the builder
    opens), diamond drag handles appear on each HS-form G(x) curve.  Dragging
    a handle vertically translates that phase's curve by ΔG and emits
    phase_edited(phase_name, updated_PhaseData).

    Edit modes stored in _edit_mode
    --------------------------------
    'off'     : normal display, no handles, matplotlib navigation active
    'handles' : 3 diamond handles per HS phase; vertical drag → translate (Phase 2)
    'direct'  : future — direct curve grab with modifier keys (Interaction B)
    'anchors' : future — anchor-point least-squares fitting (Interaction C)
    """

    # Emitted on drag release: (phase_name: str, updated_data: PhaseData)
    phase_edited = Signal(str, object)

    def __init__(self, system, y_lim=None, colors=None):
        self.system = system
        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self._colors = colors if colors is not None else _color_map(system.phases)
        self._y_lim = y_lim
        self._legend_loc = 'upper left'
        self._last_result = None

        # ---- edit-mode state ----
        # _edit_mode     : current mode string (see class docstring)
        # _live_phase_data: dict[name, PhaseData] kept live by handle drags;
        #                   G curves are recomputed from this dict in redraw()
        #                   so edits persist across T-slider moves (hull will lag
        #                   behind until the user clicks Apply in the builder)
        # _phase_line_artists / _phase_orig_y: matplotlib Line2D + its original
        #                   y-data, rebuilt each redraw() for use by _on_motion
        # _handle_info   : {phase_name: [{'x':, 'G':, 'artist':}, ...]}
        #                   rebuilt each redraw() from _live_phase_data
        # _drag_state    : _DragState while a drag is in progress, else None
        # _*_cid         : mpl event connection IDs (None when disconnected)
        self._edit_mode          = 'off'
        self._live_phase_data    = None
        self._phase_line_artists = {}
        self._phase_orig_y       = {}
        self._handle_info        = {}
        self._drag_state         = None
        self._press_cid          = None
        self._move_cid           = None
        self._release_cid        = None

    def redraw(self, result):
        self._last_result = result
        ax = self.ax
        ax.cla()

        # Reset per-draw artist storage (all artists destroyed by cla()).
        self._phase_line_artists.clear()
        self._phase_orig_y.clear()
        self._handle_info.clear()

        # Draw each phase's G(x) curve.
        for x, G, phase in result.phase_curves:
            c = self._colors[phase.name]

            # In edit mode, replace G with values from _live_phase_data so the
            # displayed curves always reflect handle drags even across T-slider
            # moves.  The convex hull will lag behind until Apply is clicked
            # (acceptable Phase 2 limitation; fixed in Phase 4 via live recompute).
            G_plot = G
            if (self._edit_mode != 'off'
                    and self._live_phase_data is not None
                    and not phase.is_point
                    and self.system.energy_form == 'HS'):
                pd = self._live_phase_data.get(phase.name)
                if pd is not None:
                    G_plot = _G_from_phase_data(pd, x, result.T)

            if phase.is_point:
                ax.plot(x, G_plot, 'o', color=c, markersize=9,
                        label=phase.name, zorder=3)
            else:
                line, = ax.plot(x, G_plot, '-', color=c,
                                linewidth=2.5, label=phase.name)
                if self._edit_mode != 'off':
                    self._phase_line_artists[phase.name] = line
                    self._phase_orig_y[phase.name] = np.asarray(G_plot).copy()

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

        # Draw drag handles on top of everything when edit mode is active.
        if self._edit_mode != 'off':
            self._draw_edit_overlay(result)

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

    # ------------------------------------------------------------------
    # Interactive edit mode (Phase 2 — handle-drag framework)
    # ------------------------------------------------------------------

    def set_edit_mode(self, mode: str, live_phase_data=None):
        """Switch between display-only and interactive editing modes.

        Parameters
        ----------
        mode : str
            'off'     — normal display, no handles (default)
            'handles' — 3 diamond handles per HS phase (Phase 2, current)
            'direct'  — reserved for future Interaction B (direct curve grab)
            'anchors' — reserved for future Interaction C (anchor-point fit)
        live_phase_data : dict[str, PhaseData] or None
            Initial live PhaseData per phase.  When None and switching into a
            non-off mode, data is initialised from self.system automatically.
            Passing live_phase_data explicitly lets the caller preserve edits
            across reloads (e.g. after MainWindow.reload_system).
        """
        if live_phase_data is not None:
            self._live_phase_data = dict(live_phase_data)

        if mode == self._edit_mode:
            # Same mode: refresh live data and redraw handles if supplied.
            if live_phase_data is not None and self._last_result is not None:
                self.redraw(self._last_result)
            return

        self._edit_mode  = mode
        self._drag_state = None

        if mode == 'off':
            # Disconnect all event handlers.
            for attr in ('_press_cid', '_move_cid', '_release_cid'):
                cid = getattr(self, attr)
                if cid is not None:
                    self.mpl_disconnect(cid)
                    setattr(self, attr, None)
            self._live_phase_data = None
            if self._last_result is not None:
                self.redraw(self._last_result)
            return

        # Entering a non-off mode — initialise live data if not provided.
        if self._live_phase_data is None:
            from pde_builder import SystemData
            sd = SystemData.from_system(self.system)
            self._live_phase_data = {pd.name: pd for pd in sd.phases}

        # Connect event handlers (guard against double-connection).
        if self._press_cid is None:
            self._press_cid   = self.mpl_connect('button_press_event',
                                                  self._on_press)
            self._move_cid    = self.mpl_connect('motion_notify_event',
                                                  self._on_motion)
            self._release_cid = self.mpl_connect('button_release_event',
                                                  self._on_release)

        if self._last_result is not None:
            self.redraw(self._last_result)

    def _draw_edit_overlay(self, result):
        """Draw diamond handles for each editable HS phase (called from redraw).

        Three handles per phase at x = xmin, midpoint, xmax.  G values are
        computed from _live_phase_data at result.T so they stay consistent
        with the overridden G curves drawn earlier in redraw().

        Fills self._handle_info so that _on_press / _on_motion / _on_release
        can find and move the correct artist by index.

        Extension notes for future phases:
          Phase 3  — add horizontal-drag tabs at the endpoints for xmin/xmax.
          Phase 4  — extend to N handles for degree-N−1 H polynomial fitting.
          'direct' — _draw_edit_overlay delegates to a separate overlay method.
          'anchors'— same; anchor positions are stored in a separate dict.
        """
        if self._live_phase_data is None:
            return
        T = result.T
        for phase in self.system.phases:
            if phase.is_point or self.system.energy_form != 'HS':
                continue
            pd = self._live_phase_data.get(phase.name)
            if pd is None:
                continue
            xmin = pd.xmin
            xmax = pd.xmax
            xmid = 0.5 * (xmin + xmax)
            handle_xs = [xmin, xmid, xmax]
            c = self._colors.get(phase.name, 'k')
            self._handle_info[phase.name] = []
            for hx in handle_xs:
                G_val = float(_G_from_phase_data(pd, hx, T))
                artist, = self.ax.plot(
                    hx, G_val, 'D',
                    color=c, markersize=10,
                    markeredgecolor='white', markeredgewidth=1.5,
                    zorder=10, clip_on=False)
                self._handle_info[phase.name].append(
                    {'x': hx, 'G': G_val, 'artist': artist})

    # --- drag event handlers -----------------------------------------------

    def _on_press(self, event):
        """Begin a handle drag when the user clicks near a diamond."""
        if event.inaxes is not self.ax or event.button != 1:
            return
        if event.ydata is None or not self._handle_info:
            return

        HIT_RADIUS_PX = 12
        best_dist  = HIT_RADIUS_PX
        best_phase = None
        best_idx   = None

        for phase_name, info_list in self._handle_info.items():
            for idx, info in enumerate(info_list):
                disp = self.ax.transData.transform((info['x'], info['G']))
                dist = np.hypot(event.x - disp[0], event.y - disp[1])
                if dist < best_dist:
                    best_dist  = dist
                    best_phase = phase_name
                    best_idx   = idx

        if best_phase is None:
            return

        import copy
        pd = (self._live_phase_data.get(best_phase)
              if self._live_phase_data else None)
        self._drag_state = _DragState(
            phase_name   = best_phase,
            handle_idx   = best_idx,
            y_press_data = event.ydata,
            snapshot     = copy.deepcopy(pd),   # stored for Ctrl+Z undo (Phase 5)
            T_ref        = self._last_result.T if self._last_result else 0.0,
            P_ref        = self._last_result.P if self._last_result else 0.0,
        )

    def _on_motion(self, event):
        """Translate the dragged phase's curve and handles during mouse motion.

        Provides live visual feedback only — PhaseData is not updated here.
        Phase 2: the entire G(x) curve shifts uniformly by ΔG (vertical only).
        Phase 3 TODO: horizontal drag of endpoint handles → xmin / xmax.
        Phase 4 TODO: per-handle drag with polynomial re-fit drawn live.
        """
        if self._drag_state is None or event.inaxes is not self.ax:
            return
        if event.ydata is None:
            return

        delta_G    = event.ydata - self._drag_state.y_press_data
        phase_name = self._drag_state.phase_name

        # Shift the phase curve line artist.
        line   = self._phase_line_artists.get(phase_name)
        orig_y = self._phase_orig_y.get(phase_name)
        if line is not None and orig_y is not None:
            line.set_ydata(orig_y + delta_G)

        # Shift all handle artists for this phase by the same ΔG.
        for info in self._handle_info.get(phase_name, []):
            info['artist'].set_ydata([info['G'] + delta_G])

        self.draw_idle()

    def _on_release(self, event):
        """Finalise the drag: apply to PhaseData and emit phase_edited.

        Passes the dragged handle's new G (and unchanged G for the others)
        to apply_handle_drag().  Non-dragged handles are at original positions
        so that Phase 4 can use them as polynomial constraints without any
        API change here.
        """
        if self._drag_state is None or event.button != 1:
            self._drag_state = None
            return

        ds               = self._drag_state
        self._drag_state = None

        delta_G = (event.ydata - ds.y_press_data
                   if event.ydata is not None else 0.0)

        if abs(delta_G) < 1e-12:
            # No net movement — reset visuals from the last clean result.
            if self._last_result is not None:
                self.redraw(self._last_result)
            return

        pd = (self._live_phase_data.get(ds.phase_name)
              if self._live_phase_data else None)
        if pd is None:
            return

        # Build the updated handle G positions:
        #   dragged handle → new target G
        #   other handles  → unchanged G  (constraints for Phase 4 polynomial fit)
        info_list = self._handle_info.get(ds.phase_name, [])
        handles_x = [info['x'] for info in info_list]
        handles_G = [info['G'] + (delta_G if i == ds.handle_idx else 0.0)
                     for i, info in enumerate(info_list)]

        from pde_builder import apply_handle_drag
        try:
            new_pd = apply_handle_drag(
                pd, ds.handle_idx, handles_x, handles_G,
                ds.T_ref, self.system.energy_form)
        except Exception:
            if self._last_result is not None:
                self.redraw(self._last_result)
            return

        if self._live_phase_data is not None:
            self._live_phase_data[ds.phase_name] = new_pd

        # Notify the main window — it will update the builder spinboxes and
        # trigger a live equilibrium recompute to refresh the hull.
        self.phase_edited.emit(ds.phase_name, new_pd)


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
        self._builder      = None   # BuilderWindow (single instance, persists across reloads)
        self._worker       = None
        self._worker_state = 'idle'
        self._color_dialog = None
        self._init_system_state(system, precomputed_Tx, precomputed_Px)
        self._build_central_widget()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_system_state(self, system, precomputed_Tx, precomputed_Px=None):
        """Set all non-widget instance variables from a new system + precomputed data."""
        self._n_steps = N_T_STEPS
        self.system = system
        self.setWindowTitle(system.title)

        self._precomputed_Tx = precomputed_Tx
        self._precomputed_Px = precomputed_Px

        self._T_arr = np.array([r.T for r in precomputed_Tx])
        if precomputed_Px is not None:
            self._P_arr = np.array([r.P for r in precomputed_Px])
        elif system.has_pressure:
            self._P_arr = np.linspace(system.P_max, system.P_min, self._n_steps)
        else:
            self._P_arr = None

        self._mode         = 'fixed_P'
        self._full_grid    = None
        self._worker       = None
        self._worker_state = 'idle'

        self._colors          = _color_map(system.phases)
        self._two_phase_color = _TWO_PHASE_COLOR
        self._two_phase_hatch = _TWO_PHASE_HATCH
        self._color_dialog    = None

    def _build_central_widget(self):
        """Build (or rebuild) all Qt widgets, canvases, layouts, signal connections.

        Safe to call multiple times — setCentralWidget() replaces the old
        central widget and Qt automatically deletes it along with all its
        children.
        """
        system = self.system

        # ---- G-x y-limits from the Tx precomputed data ----
        y_lim = _compute_ylim(self._precomputed_Tx)

        # ---- canvases ----
        self.gx_canvas = GxCanvas(system, y_lim, colors=self._colors)
        self.tx_canvas = TxCanvas(system, self._precomputed_Tx,
                                  colors=self._colors,
                                  two_phase_color=self._two_phase_color,
                                  two_phase_hatch=self._two_phase_hatch)
        self.px_canvas = (PxCanvas(system, self._precomputed_Px,
                                   colors=self._colors,
                                   two_phase_color=self._two_phase_color,
                                   two_phase_hatch=self._two_phase_hatch)
                          if self._precomputed_Px is not None else None)

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
        self.P_label  = None
        if system.has_pressure:
            self.P_slider = QSlider(Qt.Horizontal)
            self.P_slider.setMinimum(0)
            self.P_slider.setMaximum(self._n_steps - 1)
            P_range = max(system.P_max - system.P_min, 1e-30)
            P_frac = (system.P_initial - system.P_min) / P_range
            self.P_slider.setValue(int(round(P_frac * (self._n_steps - 1))))
            self.P_label = QLabel(f'P = {system.P_initial:.3g} {system.P_unit}'.strip())
            self.P_label.setMinimumWidth(90)

        # ---- mode selector (only when has_pressure) ----
        self._mode_combo = None
        if system.has_pressure:
            self._mode_combo = QComboBox()
            self._mode_combo.addItem('Fixed P  (T-x)')   # index 0
            self._mode_combo.addItem('Fixed T  (P-x)')   # index 1
            self._mode_combo.setCurrentIndex(0)

        # ---- shared controls ----
        self._reveal_cb  = QCheckBox('Reveal all')
        self._colors_btn = QPushButton('Colors\u2026')
        self._res_combo  = QComboBox()
        for _label, _n in [('Very Low (25)', 25), ('Low (50)', 50),
                            ('Medium (100)', 100), ('High (200)', 200),
                            ('Very High (500)', 500)]:
            self._res_combo.addItem(_label, _n)
        self._res_combo.setCurrentIndex(3)   # default: High (200)

        # ---- pre-compute widgets (only when pressure is active) ----
        self._precompute_btn    = None
        self._precompute_bar    = None
        self._precompute_status = None
        if system.has_pressure:
            self._precompute_btn = QPushButton('Pre-compute full T-P-x')
            n_total = self._n_steps ** 2
            self._precompute_bar = QProgressBar()
            self._precompute_bar.setRange(0, n_total)
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(False)
            self._precompute_status = QLabel('')

        # ---- Builder button ----
        self._builder_btn = QPushButton('Builder\u2026')
        self._builder_btn.clicked.connect(self._open_builder)

        # ---- layout ----
        # Top row: Builder… | mode combo (if pressure) | Reveal all | Colors… | Resolution | precompute (if pressure)
        top_row = QHBoxLayout()
        top_row.addWidget(self._builder_btn)
        top_row.addSpacing(16)
        if system.has_pressure:
            top_row.addWidget(self._mode_combo)
            top_row.addSpacing(16)
        top_row.addWidget(self._reveal_cb)
        top_row.addWidget(self._colors_btn)
        top_row.addSpacing(8)
        top_row.addWidget(QLabel('Resolution:'))
        top_row.addWidget(self._res_combo)
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
            self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        if self._precompute_btn is not None:
            self._precompute_btn.clicked.connect(self._on_precompute_clicked)
        self._reveal_cb.toggled.connect(self._on_reveal_all_toggled)
        self._colors_btn.clicked.connect(self._on_colors_clicked)
        self._res_combo.currentIndexChanged.connect(self._on_resolution_changed)

        # Render initial state.
        self._on_T_changed(int(system.T_initial))

    # ------------------------------------------------------------------
    # Builder integration
    # ------------------------------------------------------------------

    def reload_system(self, system):
        """Rebuild the entire UI for a new system (called by the builder on Apply)."""
        if self._worker is not None:
            self._worker.abort()
            self._worker.wait()
        if self._color_dialog is not None:
            self._color_dialog.close()
        print('Applying builder changes...', end=' ', flush=True)
        P = system.P_initial if system.has_pressure else 0.0
        precomputed_Tx = precompute_Tx_diagram(system, P)
        print('done.')
        self._init_system_state(system, precomputed_Tx, None)
        self._build_central_widget()

        # Re-activate edit mode on the fresh canvas if the builder is still open.
        if self._builder is not None and self._builder.isVisible():
            from pde_builder import SystemData
            sd        = SystemData.from_system(system)
            live_data = {pd.name: pd for pd in sd.phases}
            self.gx_canvas.set_edit_mode('handles', live_data)
            self.gx_canvas.phase_edited.connect(self._on_phase_edited)

    def _open_builder(self):
        """Open (or raise) the builder window, pre-populated with the current system.

        Also activates handle-drag edit mode on GxCanvas so the user can
        directly manipulate G(x) curves while the builder is open.
        """
        if self._builder is None or not self._builder.isVisible():
            from pde_builder import BuilderWindow, SystemData
            self._builder = BuilderWindow(system=self.system)
            self._builder.system_applied.connect(self.reload_system)
            self._builder.finished.connect(self._on_builder_closed)

            # Activate edit mode on the canvas with fresh live data.
            sd        = SystemData.from_system(self.system)
            live_data = {pd.name: pd for pd in sd.phases}
            self.gx_canvas.set_edit_mode('handles', live_data)

            # Connect phase_edited → live spinbox update + equilibrium refresh.
            # Disconnect first to guard against stale connections after reload.
            try:
                self.gx_canvas.phase_edited.disconnect(self._on_phase_edited)
            except RuntimeError:
                pass
            self.gx_canvas.phase_edited.connect(self._on_phase_edited)

        self._builder.raise_()
        self._builder.show()

    def _on_builder_closed(self, result=None):
        """Deactivate handle-drag edit mode when the builder window closes."""
        self.gx_canvas.set_edit_mode('off')

    def _on_phase_edited(self, name, new_pd):
        """Live-update after a G(x) handle drag.

        1. Pushes the new PhaseData into the builder's spinboxes so the two
           UIs stay in sync.
        2. Recomputes equilibrium with a temporary system that has the edited
           phase, giving a correct hull at the current T and P without waiting
           for the user to click Apply.

        The T-x / P-x canvases are NOT updated here — they remain valid for
        the pre-Apply system and are refreshed only when Apply is clicked.
        """
        # Sync builder spinboxes.
        if self._builder is not None and self._builder.isVisible():
            self._builder.update_phase_data(name, new_pd)

        # Build a temporary system with the edited phase.
        from pde_builder import SystemData
        sd = SystemData.from_system(self.system)
        for i, pd in enumerate(sd.phases):
            if pd.name == name:
                sd.phases[i] = new_pd
                break
        tmp_system = sd.to_system()

        T      = self._current_T()
        P      = self._current_P()
        result = compute_equilibrium(tmp_system, T, P)
        self.gx_canvas.redraw(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _P_from_tick(self, tick):
        """Map slider tick 0…n_steps-1 → pressure value."""
        frac = tick / (self._n_steps - 1)
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
            new_Px = [self._full_grid[T_idx][i_P] for i_P in range(self._n_steps)]
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
            new_Px = [self._full_grid[T_idx][i_P] for i_P in range(self._n_steps)]
        else:
            print(f'Recomputing P-x diagram at T = {T:.1f} K...', end=' ', flush=True)
            new_Px = precompute_Px_diagram(self.system, T, n_steps=self._n_steps)
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
            new_Tx = [self._full_grid[i_T][P_idx] for i_T in range(self._n_steps)]
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
            new_Tx = [self._full_grid[i_T][P_idx] for i_T in range(self._n_steps)]
        else:
            print(f'Recomputing T-x diagram at P = {P:.3g}...', end=' ', flush=True)
            new_Tx = precompute_Tx_diagram(self.system, P, n_steps=self._n_steps)
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
            n_total = self._n_steps ** 2
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(True)
            self._precompute_status.setText('Computing... 0%')
            T_values = np.linspace(self.system.T_max, self.system.T_min, self._n_steps)
            P_values = np.linspace(self.system.P_max, self.system.P_min, self._n_steps)
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
        n_total = self._n_steps ** 2
        self._precompute_status.setText(f'Cached ({n_total:,} evaluations)')
        self._precompute_btn.setText('Full T-P-x cached')
        self._precompute_btn.setEnabled(False)

    def _on_resolution_changed(self, idx):
        """Recompute diagrams at a new step count chosen from the resolution combo."""
        n_steps = self._res_combo.itemData(idx)
        if n_steps == self._n_steps:
            return

        # Abort any running background worker.
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait()
        self._worker = None
        self._worker_state = 'idle'
        self._full_grid = None

        self._n_steps = n_steps

        # Update P slider range to match the new step count (preserves current P value).
        if self.P_slider is not None:
            P_cur = self._current_P()
            self.P_slider.setMaximum(n_steps - 1)
            P_range = max(self.system.P_max - self.system.P_min, 1e-30)
            P_frac = (P_cur - self.system.P_min) / P_range
            self.P_slider.setValue(int(round(P_frac * (n_steps - 1))))

        # Reset precompute button and progress bar.
        if self._precompute_btn is not None:
            self._precompute_btn.setText('Pre-compute full T-P-x')
            self._precompute_btn.setEnabled(True)
            self._precompute_bar.setRange(0, n_steps ** 2)
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(False)
            self._precompute_status.setText('')

        # Recompute T-x at the current P.
        P = self._current_P() if self.P_slider is not None else 0.0
        T = self._current_T()
        self._precomputed_Tx = precompute_Tx_diagram(self.system, P, n_steps=n_steps)
        self._T_arr = np.array([r.T for r in self._precomputed_Tx])
        self.gx_canvas._y_lim = _compute_ylim(self._precomputed_Tx)

        # Recompute P-x only if it was already lazily computed.
        if self._precomputed_Px is not None:
            self._precomputed_Px = precompute_Px_diagram(self.system, T, n_steps=n_steps)
            self._P_arr = np.array([r.P for r in self._precomputed_Px])
        elif self.system.has_pressure:
            self._P_arr = np.linspace(self.system.P_max, self.system.P_min, n_steps)

        # Redraw all canvases.
        self.tx_canvas.reset(self._precomputed_Tx, current_T=T)
        if self._precomputed_Px is not None and self.px_canvas is not None:
            self.px_canvas.reset(self._precomputed_Px, current_P=P)
        nearest = int(np.argmin(np.abs(self._T_arr - T)))
        self.gx_canvas.redraw(self._precomputed_Tx[nearest])

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


def _make_default_system():
    """Return a minimal single-phase System for use when no input file is given."""
    from pde_energy import HSModel
    from pde_phase import Phase, System
    liq = Phase('liquid', 'liquid',
                HSModel([8.0, -2.0, 2.0], [0.01]),
                0.0, 1.0)
    return System(['A', 'B'], [liq], 'HS', 500, 1500, 1500, title='New System')


def launch_ui_empty():
    """Launch with a minimal default system and open the builder automatically."""
    system = _make_default_system()
    print('Pre-computing default phase diagram...', end=' ', flush=True)
    precomputed_Tx = precompute_Tx_diagram(system)
    print('done.')
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(system, precomputed_Tx)
    window.resize(1200, 650)
    window.show()
    window._open_builder()   # auto-open builder with default system
    sys.exit(app.exec())
