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
                                QColorDialog, QComboBox, QDialog,
                                QFileDialog, QFormLayout, QGridLayout,
                                QHBoxLayout, QLabel, QMainWindow, QMenu,
                                QProgressBar, QPushButton, QSpinBox,
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



def _G_from_phase_spec(spec, x, T, P=0.0,
                       R_gas=0.0, P_ref=1.0):
    """Evaluate G(x, T[, P]) from a PhaseSpec (HS form).

    Lightweight evaluator used during interactive
    editing.  Mirrors HSModel.gibbs():
        G = H(x) − T·S(x) [+ P·V(x)] [+ R·T·ln(P/P₀)]

    *x* may be scalar or numpy array.  Only HS-form
    specs are supported; polynomial phases should not
    reach this path (guarded by the caller).
    """
    H = spec.H_coeffs or [0.0]
    S = spec.S_coeffs or [0.0]
    pv = np.polynomial.polynomial.polyval
    G = pv(x, H) - T * pv(x, S)
    V = spec.V_coeffs
    if V and P != 0.0:
        G = G + P * pv(x, V)
    if (spec.ideal_gas and R_gas != 0.0
            and P > 0.0 and P_ref > 0.0):
        G = G + R_gas * T * np.log(P / P_ref)
    return G


# ---------------------------------------------------------------------------
# Interactive edit-mode state
# ---------------------------------------------------------------------------

@dataclass
class _DragState:
    """All state for one in-progress G(x) handle drag.

    Fields are sized for future extension:
      snapshot  — PhaseSpec copy before drag (Ctrl+Z)
      T_ref     — temperature at drag start
      P_ref     — pressure at drag start
      handle_idx — which handle; −1 = rigid shift
    """
    phase_name:   str
    handle_idx:   int     # 0 = left endpoint, 1 = midpoint, 2 = right endpoint
    y_press_data: float   # data-y at button_press_event (drag reference origin)
    snapshot:     object  # deep copy of PhaseSpec before drag (for undo)
    T_ref:        float   # temperature when drag started (for Phase 8)
    P_ref:        float   # pressure when drag started (for Phase 8)
    x_press_data: float = 0.0   # data-x at press (Phase 3 horizontal reference)
    x_press_px:   float = 0.0   # pixel-x at press (direction detection)
    y_press_px:   float = 0.0   # pixel-y at press (direction detection)
    drag_axis:    object = None  # None=undetermined | 'vertical' | 'horizontal'
    fit_target:   str   = 'H'   # 'H' or 'S'; 'S' when Shift held at press


# ---------------------------------------------------------------------------
# G-x canvas
# ---------------------------------------------------------------------------

class GxCanvas(FigureCanvasQTAgg):
    """Left panel: Gibbs energy curves vs. composition at the current T (and P).

    Edit mode
    ---------
    When set_edit_mode('handles') is called (by MainWindow when the builder
    opens), diamond drag handles appear on each HS-form G(x) curve.  Dragging
    a handle vertically fits H₀, H₁, H₂ via a 3-point quadratic solve and
    emits phase_edited(phase_name, updated_PhaseSpec).  Dragging an endpoint
    handle horizontally updates xmin or xmax and emits the same signal.

    Edit modes stored in _edit_mode
    --------------------------------
    'off'     : normal display, no handles, matplotlib navigation active
    'handles' : 3 diamond handles per HS phase; vertical drag → quadratic H fit (Phase 4);
                endpoint horizontal drag → xmin/xmax (Phase 3)
    'direct'  : future — direct curve grab with modifier keys (Interaction B)
    'anchors' : future — anchor-point least-squares fitting (Interaction C)
    """

    # Emitted on drag release: (phase_name, PhaseSpec)
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
        # _edit_mode : current mode string
        # _live_phase_specs : dict[name, PhaseSpec]
        #   kept live by handle drags; G curves are
        #   recomputed from specs in redraw() so edits
        #   persist across T-slider moves.
        # _handle_info : per-phase handle metadata
        # _drag_state  : _DragState or None
        self._edit_mode          = 'off'
        self._live_phase_specs    = None
        self._phase_line_artists = {}
        self._phase_orig_y       = {}
        self._phase_orig_x       = {}
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
        self._phase_orig_x.clear()
        self._handle_info.clear()

        # Draw each phase's G(x) curve.
        for x, G, phase in result.phase_curves:
            c = self._colors[phase.name]

            # In edit mode, replace G with values from _live_phase_specs so the
            # displayed curves always reflect handle drags even across T-slider
            # moves.  The convex hull will lag behind until Apply is clicked
            # (acceptable Phase 2 limitation; fixed in Phase 4 via live recompute).
            G_plot = G
            if (self._edit_mode != 'off'
                    and self._live_phase_specs is not None
                    and not phase.is_point
                    and self.system.energy_form == 'HS'):
                ps = self._live_phase_specs.get(
                    phase.name)
                # VLE gas: H/S are derived from liquid;
                # use precomputed G instead.
                if (ps is not None
                        and not ps.is_vle_gas):
                    # Patched phases: precomputed G
                    # already has the piecewise shape.
                    has_patch = (
                        (ps.patch_left_x is not None
                         and ps.patch_left_phase)
                        or (ps.patch_right_x
                            is not None
                            and ps.patch_right_phase))
                    if not has_patch:
                        G_plot = _G_from_phase_spec(
                            ps, x, result.T,
                            result.P,
                            self.system.R_gas,
                            self.system.P_ref)

            if phase.is_point:
                ax.plot(x, G_plot, 'o', color=c, markersize=9,
                        label=phase.name, zorder=3)
            else:
                line, = ax.plot(x, G_plot, '-', color=c,
                                linewidth=2.5, label=phase.name)
                if self._edit_mode != 'off':
                    self._phase_line_artists[phase.name] = line
                    self._phase_orig_y[phase.name] = np.asarray(G_plot).copy()
                    self._phase_orig_x[phase.name] = np.asarray(line.get_xdata()).copy()

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
        parts = []
        for f in self.system.fields:
            if f.name == 'temperature':
                val_str = f'{f.symbol} = {result.T:.3g}'
            elif f.name == 'pressure':
                val_str = f'{f.symbol} = {result.P:.3g}'
            else:
                continue
            if f.unit:
                val_str += f' {f.unit}'
            parts.append(val_str)
        ax.set_title('G vs x     ' + '   '.join(parts) if parts else 'G vs x')
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
    # Interactive edit mode (Phases 2–4 — handle-drag framework)
    # ------------------------------------------------------------------

    def set_edit_mode(self, mode: str,
                      live_specs=None):
        """Switch between display-only and interactive
        editing modes.

        Parameters
        ----------
        mode : str
            'off', 'handles', 'direct', 'anchors'
        live_specs : dict[str, PhaseSpec] or None
            Initial PhaseSpec dict.  When None and
            entering a non-off mode, specs must have
            been set by the caller (MainWindow always
            provides them).
        """
        if live_specs is not None:
            self._live_phase_specs = dict(
                live_specs)

        if mode == self._edit_mode:
            if (live_specs is not None
                    and self._last_result is not None):
                self.redraw(self._last_result)
            return

        self._edit_mode = mode
        self._drag_state = None

        if mode == 'off':
            for attr in ('_press_cid',
                         '_move_cid',
                         '_release_cid'):
                cid = getattr(self, attr)
                if cid is not None:
                    self.mpl_disconnect(cid)
                    setattr(self, attr, None)
            self._live_phase_specs = None
            if self._last_result is not None:
                self.redraw(self._last_result)
            return

        # Entering non-off mode.
        if self._live_phase_specs is None:
            self._live_phase_specs = {}

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
        computed from _live_phase_specs at result.T so they stay consistent
        with the overridden G curves drawn earlier in redraw().

        Fills self._handle_info so that _on_press / _on_motion / _on_release
        can find and move the correct artist by index.

        Extension notes:
          Phase 3  — DONE: horizontal drag of endpoint handles for xmin/xmax.
          Phase 4  — DONE: 3-point quadratic H fit on vertical drag.
          'direct' — _draw_edit_overlay delegates to a separate overlay method.
          'anchors'— same; anchor positions are stored in a separate dict.
        """
        if self._live_phase_specs is None:
            return
        T = result.T
        P = result.P
        R_gas = self.system.R_gas
        P_ref = self.system.P_ref
        for phase in self.system.phases:
            if phase.is_point or self.system.energy_form != 'HS':
                continue
            # Skip VLE gas phases — their curve is derived from the liquid;
            # dragging it independently is physically meaningless.
            if getattr(phase.energy_model, 'vle_params', None) is not None:
                continue
            # Skip piecewise-patched phases — handle drag would conflict with
            # the patch polynomial.
            from pde_energy import PiecewisePatchModel as _PPM
            if isinstance(phase.energy_model, _PPM):
                continue
            pd = self._live_phase_specs.get(phase.name)
            if pd is None:
                continue
            xmin = pd.xmin
            xmax = pd.xmax
            xmid = 0.5 * (xmin + xmax)
            handle_xs = [xmin, xmid, xmax]
            c = self._colors.get(phase.name, 'k')
            self._handle_info[phase.name] = []
            for hx in handle_xs:
                G_val = float(_G_from_phase_spec(pd, hx, T, P, R_gas, P_ref))
                artist, = self.ax.plot(
                    hx, G_val, 'D',
                    color=c, markersize=10,
                    markeredgecolor='white', markeredgewidth=1.5,
                    zorder=10, clip_on=False)
                self._handle_info[phase.name].append(
                    {'x': hx, 'G': G_val, 'artist': artist})

    # --- drag event handlers -----------------------------------------------

    def _on_press(self, event):
        """Begin a handle drag or rigid shift when the user clicks.

        Ctrl+click anywhere on a G(x) curve starts a rigid shift of the whole
        curve (handle_idx = −1); both vertical (G offset) and horizontal
        (x-range translation) motion are supported.  A plain click near a
        diamond handle starts the normal vertical/horizontal drag.
        """
        if event.inaxes is not self.ax or event.button != 1:
            return
        if event.ydata is None:
            return

        import copy
        mods = QApplication.keyboardModifiers()
        rigid_mode = bool(mods & Qt.ControlModifier)

        if rigid_mode:
            # Hit-test phase curves, not handles.
            HIT_RADIUS_PX = 8
            best_dist  = HIT_RADIUS_PX
            best_phase = None
            for phase_name, line in self._phase_line_artists.items():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                pts   = self.ax.transData.transform(
                    np.column_stack([xdata, ydata]))
                dist = np.hypot(pts[:, 0] - event.x,
                                pts[:, 1] - event.y).min()
                if dist < best_dist:
                    best_dist  = dist
                    best_phase = phase_name
            if best_phase is None:
                return
            pd = (self._live_phase_specs.get(best_phase)
                  if self._live_phase_specs else None)
            # Freeze axes autoscaling for the duration of the drag so that
            # set_xdata() in _on_motion doesn't trigger a y-axis rescale
            # that makes curves disappear.  redraw() (called on release via
            # phase_edited) resets the axes via cla(), restoring the default.
            self.ax.autoscale(enable=False)
            self._drag_state = _DragState(
                phase_name   = best_phase,
                handle_idx   = -1,
                y_press_data = event.ydata,
                snapshot     = copy.deepcopy(pd),
                T_ref        = self._last_result.T if self._last_result else 0.0,
                P_ref        = self._last_result.P if self._last_result else 0.0,
                x_press_data = event.xdata if event.xdata is not None else 0.0,
                x_press_px   = event.x,
                y_press_px   = event.y,
                drag_axis    = 'rigid',
                fit_target   = 'H',
            )
            return

        if not self._handle_info:
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

        pd = (self._live_phase_specs.get(best_phase)
              if self._live_phase_specs else None)
        shift_held = bool(mods & Qt.ShiftModifier)
        self._drag_state = _DragState(
            phase_name   = best_phase,
            handle_idx   = best_idx,
            y_press_data = event.ydata,
            snapshot     = copy.deepcopy(pd),   # stored for Ctrl+Z undo (Phase 5)
            T_ref        = self._last_result.T if self._last_result else 0.0,
            P_ref        = self._last_result.P if self._last_result else 0.0,
            x_press_data = event.xdata if event.xdata is not None else 0.0,
            x_press_px   = event.x,
            y_press_px   = event.y,
            drag_axis    = None,
            fit_target   = 'S' if shift_held else 'H',
        )
        # Visual hint: yellow edge for S-mode handles.
        if shift_held:
            self._handle_info[best_phase][best_idx]['artist'].set_markeredgecolor(
                'yellow')
            self.draw_idle()

    def _on_motion(self, event):
        """Live visual feedback during a G(x) handle drag.

        Phase 3: horizontal drag of endpoint handles (idx 0 or 2) updates
                 xmin/xmax — the curve is extended/contracted along x.
        Phase 4: vertical drag of any handle → 3-point quadratic H fit drawn
                 live; G(x) curve curvature changes with the dragged position.

        Direction is determined on first motion after press by comparing the
        magnitude of horizontal vs. vertical pixel displacement.  The midpoint
        handle (idx 1) always resolves to vertical regardless of motion direction.
        """
        if self._drag_state is None or event.inaxes is not self.ax:
            return
        if event.ydata is None or event.xdata is None:
            return

        ds = self._drag_state

        # ---- Step 1: direction detection (on first motion after press) ----
        if ds.drag_axis is None:
            dx_px = abs(event.x - ds.x_press_px)
            dy_px = abs(event.y - ds.y_press_px)
            if dx_px < 3 and dy_px < 3:
                return   # too small to determine direction yet
            if dx_px > dy_px and ds.handle_idx in (0, 2):
                ds.drag_axis = 'horizontal'
            else:
                ds.drag_axis = 'vertical'

        phase_name = ds.phase_name
        pd = (self._live_phase_specs.get(phase_name)
              if self._live_phase_specs else None)
        if pd is None:
            return
        line = self._phase_line_artists.get(phase_name)
        if line is None:
            return
        T     = ds.T_ref
        P     = ds.P_ref
        R_gas = self.system.R_gas
        P_ref = self.system.P_ref

        if ds.drag_axis == 'rigid':
            # ---- Rigid shift: translate whole curve + all handles in x and y ----
            delta_G  = event.ydata - ds.y_press_data
            delta_x  = event.xdata - ds.x_press_data
            orig_G   = self._phase_orig_y.get(phase_name)
            orig_x   = self._phase_orig_x.get(phase_name)
            if orig_G is None or orig_x is None:
                return
            line.set_xdata(orig_x + delta_x)
            line.set_ydata(orig_G + delta_G)
            for info in self._handle_info.get(phase_name, []):
                info['artist'].set_xdata([info['x'] + delta_x])
                info['artist'].set_ydata([info['G'] + delta_G])

        elif ds.drag_axis == 'vertical':
            # ---- Phase 4: live 3-point quadratic H fit ----
            delta_G   = event.ydata - ds.y_press_data
            info_list = self._handle_info.get(phase_name, [])
            handles_x = [info['x'] for info in info_list]
            handles_G = [info['G'] + (delta_G if i == ds.handle_idx else 0.0)
                         for i, info in enumerate(info_list)]
            from pde_builder import apply_handle_drag
            try:
                new_pd = apply_handle_drag(
                    pd, ds.handle_idx, handles_x, handles_G,
                    T, self.system.energy_form, P, R_gas, P_ref,
                    fit_target=ds.fit_target)
            except Exception:
                return
            xs = line.get_xdata()
            line.set_ydata(_G_from_phase_spec(new_pd, xs, T, P, R_gas, P_ref))
            # Update all 3 handle artists to their fitted G values.
            for info in info_list:
                fitted_G = float(
                    _G_from_phase_spec(new_pd, info['x'], T, P, R_gas, P_ref))
                info['artist'].set_ydata([fitted_G])

        else:
            # ---- Phase 3: horizontal drag → update xmin or xmax ----
            new_x = float(np.clip(event.xdata, 0.0, 1.0))
            if ds.handle_idx == 0:
                new_xmin = float(np.clip(new_x, 0.0, pd.xmax - 0.02))
                new_xmax = pd.xmax
            else:  # handle_idx == 2
                new_xmin = pd.xmin
                new_xmax = float(np.clip(new_x, pd.xmin + 0.02, 1.0))
            xs      = np.linspace(new_xmin, new_xmax, len(line.get_xdata()))
            G_vals  = _G_from_phase_spec(pd, xs, T, P, R_gas, P_ref)
            line.set_xdata(xs)
            line.set_ydata(G_vals)
            # Move dragged endpoint handle.
            dragged_x = new_xmin if ds.handle_idx == 0 else new_xmax
            dragged_G = float(
                _G_from_phase_spec(pd, dragged_x, T, P, R_gas, P_ref))
            self._handle_info[phase_name][ds.handle_idx]['artist'].set_xdata(
                [dragged_x])
            self._handle_info[phase_name][ds.handle_idx]['artist'].set_ydata(
                [dragged_G])
            # Move midpoint handle.
            xmid  = 0.5 * (new_xmin + new_xmax)
            G_mid = float(_G_from_phase_spec(pd, xmid, T, P, R_gas, P_ref))
            self._handle_info[phase_name][1]['artist'].set_xdata([xmid])
            self._handle_info[phase_name][1]['artist'].set_ydata([G_mid])

        self.draw_idle()

    def _on_release(self, event):
        """Finalise the drag: apply to PhaseSpec and emit phase_edited.

        Routes to apply_xrange_drag (Phase 3 horizontal) or apply_handle_drag
        (Phase 4 vertical) depending on the drag axis determined during motion.
        """
        if self._drag_state is None or event.button != 1:
            self._drag_state = None
            return

        ds               = self._drag_state
        self._drag_state = None

        # No movement detected — reset visuals and return.
        if ds.drag_axis is None:
            if self._last_result is not None:
                self.redraw(self._last_result)
            return

        pd = (self._live_phase_specs.get(ds.phase_name)
              if self._live_phase_specs else None)
        if pd is None:
            return

        if ds.drag_axis == 'rigid':
            # Rigid shift: apply delta_G to hs_H[0] and delta_x to xmin/xmax.
            delta_G = (event.ydata - ds.y_press_data
                       if event.ydata is not None else 0.0)
            delta_x = (event.xdata - ds.x_press_data
                       if event.xdata is not None else 0.0)
            if abs(delta_G) < 1e-12 and abs(delta_x) < 1e-12:
                if self._last_result is not None:
                    self.redraw(self._last_result)
                return
            from pde_builder import apply_rigid_shift
            try:
                new_pd = apply_rigid_shift(pd, delta_G, delta_x)
            except Exception:
                if self._last_result is not None:
                    self.redraw(self._last_result)
                return

        elif ds.drag_axis == 'horizontal':
            # Phase 3: apply xmin / xmax change.
            if event.xdata is None:
                if self._last_result is not None:
                    self.redraw(self._last_result)
                return
            new_x = float(np.clip(event.xdata, 0.0, 1.0))
            from pde_builder import apply_xrange_drag
            try:
                new_pd = apply_xrange_drag(pd, ds.handle_idx, new_x)
            except Exception:
                if self._last_result is not None:
                    self.redraw(self._last_result)
                return

        else:
            # Phase 4: apply vertical G shift with 3-point quadratic fit.
            delta_G = (event.ydata - ds.y_press_data
                       if event.ydata is not None else 0.0)
            if abs(delta_G) < 1e-12:
                if self._last_result is not None:
                    self.redraw(self._last_result)
                return
            # Build target handle G positions:
            #   dragged handle → new G;  others → original G (quadratic constraints)
            info_list = self._handle_info.get(ds.phase_name, [])
            handles_x = [info['x'] for info in info_list]
            handles_G = [info['G'] + (delta_G if i == ds.handle_idx else 0.0)
                         for i, info in enumerate(info_list)]
            from pde_builder import apply_handle_drag
            try:
                new_pd = apply_handle_drag(
                    pd, ds.handle_idx, handles_x, handles_G,
                    ds.T_ref, self.system.energy_form,
                    ds.P_ref, self.system.R_gas, self.system.P_ref,
                    fit_target=ds.fit_target)
            except Exception:
                if self._last_result is not None:
                    self.redraw(self._last_result)
                return

        if self._live_phase_specs is not None:
            self._live_phase_specs[ds.phase_name] = new_pd

        # Notify the main window — it will update the builder spinboxes and
        # trigger a live equilibrium recompute to refresh the hull.
        self.phase_edited.emit(ds.phase_name, new_pd)


# ---------------------------------------------------------------------------
# Sweep canvas  (unified replacement for the old TxCanvas / PxCanvas)
# ---------------------------------------------------------------------------

class SweepCanvas(FigureCanvasQTAgg):
    """Right panel: x-phase-diagram swept over one field, all others fixed.

    Parameters
    ----------
    primary_field   : Field   — the field shown on the Y axis
    precomputed     : list[EqResult]  sorted by primary field DESCENDING
    """

    def __init__(self, system, primary_field, precomputed, colors=None,
                 two_phase_color=_TWO_PHASE_COLOR, two_phase_hatch=_TWO_PHASE_HATCH):
        self.system = system
        self.primary_field = primary_field
        self.precomputed = precomputed
        self._colors = colors if colors is not None else _color_map(system.phases)
        self._two_phase_color = two_phase_color
        self._two_phase_hatch = two_phase_hatch
        self._lowest_val = primary_field.initial_val
        self._reveal_all = False
        self._legend_loc = 'upper left'

        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

        self._prim_values = np.array([r.field_values[primary_field.name] for r in precomputed])
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(primary_field.initial_val)
        self.draw()

    def _setup_axes(self):
        f = self.primary_field
        unit_str = f' ({f.unit})' if f.unit else ''
        self.ax.set_xlabel('Composition  x(B)')
        self.ax.set_ylabel(f'{f.name.capitalize()}{unit_str}')
        self.ax.set_title(f'{f.symbol}-x  Phase Diagram')
        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(f.min_val, f.max_val)
        handles = [
            Patch(facecolor=self._colors[p.name], label=p.name)
            for p in self.system.phases if p.phase_type != 'end_member'
        ]
        handles.append(Patch(facecolor=self._two_phase_color,
                             hatch=self._two_phase_hatch or None,
                             edgecolor='black' if self._two_phase_hatch else 'none',
                             linewidth=0,
                             label='two-phase'))
        self.ax.legend(handles=handles, loc=self._legend_loc, fontsize=8)

    def _draw_full_diagram(self):
        ax = self.ax
        results = self.precomputed
        prim_min = self.primary_field.min_val
        for i, result in enumerate(results):
            v_top = self._prim_values[i]
            v_bot = (self._prim_values[i + 1]
                     if i + 1 < len(results) else prim_min)
            dv = v_top - v_bot
            for r in result.regions:
                x0 = r['x0']
                width = r['x1'] - r['x0']
                if r['type'] == 'two_phase':
                    ax.broken_barh(
                        [(x0, width)], (v_bot, dv),
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
                        [(x0, width)], (v_bot, dv),
                        facecolors=self._colors[phase.name],
                        edgecolors='none',
                        alpha=0.85)

    def _add_cover_and_cursor(self, initial_val):
        f = self.primary_field
        self._cover = MplRectangle(
            xy=(-0.1, f.min_val), width=1.2,
            height=initial_val - f.min_val,
            facecolor='white', edgecolor='none', zorder=5)
        self.ax.add_patch(self._cover)
        self._cursor_line = self.ax.axhline(
            initial_val, color='black', linewidth=2.0, linestyle='--', zorder=10)

    def reset(self, precomputed, current_val=None):
        """Redraw with new precomputed data (after secondary-slider recompute)."""
        if current_val is None:
            current_val = self.primary_field.initial_val
        self.precomputed = precomputed
        self._prim_values = np.array([r.field_values[self.primary_field.name] for r in precomputed])
        self._lowest_val = current_val
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(current_val)
        if self._reveal_all:
            self._cover.set_height(0)
        self.draw()

    def set_cursor(self, val):
        """Move cursor to val; shrink cover if val is a new minimum."""
        if val < self._lowest_val:
            self._lowest_val = val
            if not self._reveal_all:
                self._cover.set_height(val - self.primary_field.min_val)
        self._cursor_line.set_ydata([val, val])
        self.draw()

    def set_reveal_all(self, flag):
        """Show or hide the cover rectangle regardless of the slider position."""
        self._reveal_all = flag
        self._cover.set_height(
            0 if flag else self._lowest_val - self.primary_field.min_val)
        self.draw()

    def recolor(self):
        """Redraw with updated colors, preserving cover/reveal state."""
        self.ax.cla()
        self._setup_axes()
        self._draw_full_diagram()
        self._add_cover_and_cursor(self._lowest_val)
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
    """Compute a full field0 × field1 grid in a
    background thread.

    Parameters
    ----------
    system    : System
    field0    : Field — outer loop (rows)
    f0_values : ndarray — values for field0
    field1    : Field — inner loop (columns)
    f1_values : ndarray — values for field1

    Emits grid[i0][i1] of EqResult on finished.
    Supports pause/resume and clean abort.
    """

    progress = Signal(int, int)
    finished = Signal(object)

    def __init__(self, system, field0, f0_values,
                 field1, f1_values):
        super().__init__()
        self._system = system
        self._field0 = field0
        self._field1 = field1
        self._f0_values = f0_values
        self._f1_values = f1_values
        self._run_event = threading.Event()
        self._run_event.set()
        self._abort = False

    def pause(self):
        """Block at the next loop iteration."""
        self._run_event.clear()

    def resume(self):
        """Unblock the worker."""
        self._run_event.set()

    def abort(self):
        """Ask the worker to stop."""
        self._abort = True
        self._run_event.set()

    def run(self):
        n0 = len(self._f0_values)
        n1 = len(self._f1_values)
        n_total = n0 * n1
        done = 0
        grid = []
        f0_name = self._field0.name
        f1_name = self._field1.name
        for v0 in self._f0_values:
            row = []
            for v1 in self._f1_values:
                self._run_event.wait()
                if self._abort:
                    return
                fv = {f0_name: v0,
                      f1_name: v1}
                row.append(
                    compute_equilibrium(
                        self._system, fv))
                done += 1
                self.progress.emit(
                    done, n_total)
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

    def __init__(self, system, precomputed_Tx,
                 precomputed_Px=None, spec=None):
        super().__init__()
        self._builder = None
        self._worker = None
        self._worker_state = 'idle'
        self._color_dialog = None
        self._system_spec = spec
        self._init_system_state(
            system, precomputed_Tx, precomputed_Px)
        self._build_central_widget()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_system_state(self, system, precomputed_Tx, precomputed_Px=None):
        """Set all non-widget instance variables from a new system + precomputed data."""
        self._n_steps = N_T_STEPS
        self.system = system
        self.setWindowTitle(system.title)

        # Generalised sweep state — one entry per field in system.fields.
        # _precomputed[i]: list[EqResult] for sweeping field i at other fields' initial
        # _field_arr[i]:   numpy array of primary field values for _precomputed[i].
        # None entries are filled lazily on the first mode switch to field i.
        n_fields = len(system.fields)
        self._precomputed = [None] * n_fields
        self._field_arr   = [None] * n_fields

        self._precomputed[0] = precomputed_Tx
        self._field_arr[0]   = np.array([r.field_values[system.fields[0].name] for r in precomputed_Tx])

        if n_fields > 1:
            f1 = system.fields[1]
            if precomputed_Px is not None:
                self._precomputed[1] = precomputed_Px
                self._field_arr[1]   = np.array([r.field_values[f1.name] for r in precomputed_Px])
            else:
                # Placeholder until lazy computation.
                self._field_arr[1] = np.linspace(f1.max_val, f1.min_val, self._n_steps)

        # Which field is the primary sweep axis (Y axis of the sweep canvas).
        self._primary_idx = 0

        self._full_grid    = None
        self._worker       = None
        self._worker_state = 'idle'
        self._viz3d_window = None

        # Preserve user-chosen colors across reloads.
        # Only assign fresh defaults for phases that
        # were not present before (or on first init).
        prev = getattr(self, '_colors', None)
        if prev is None:
            self._colors = _color_map(system.phases)
        else:
            fresh = _color_map(system.phases)
            self._colors = {
                p.name: prev.get(p.name,
                                 fresh[p.name])
                for p in system.phases}
        if not hasattr(self, '_two_phase_color'):
            self._two_phase_color = _TWO_PHASE_COLOR
        if not hasattr(self, '_two_phase_hatch'):
            self._two_phase_hatch = _TWO_PHASE_HATCH
        self._color_dialog = None

    def _build_central_widget(self):
        """Build (or rebuild) all Qt widgets, canvases, layouts, signal connections.

        Safe to call multiple times — setCentralWidget() replaces the old
        central widget and Qt automatically deletes it along with all its
        children.
        """
        system = self.system

        # ---- G-x y-limits ----
        y_lim = _compute_ylim(self._precomputed[0])

        # ---- G-x canvas ----
        self.gx_canvas = GxCanvas(system, y_lim, colors=self._colors)

        # ---- primary sweep canvas ----
        prim_field = system.fields[self._primary_idx]
        prim_canvas = SweepCanvas(
            system, prim_field, self._precomputed[self._primary_idx],
            colors=self._colors,
            two_phase_color=self._two_phase_color,
            two_phase_hatch=self._two_phase_hatch)
        self._sweep_canvases = {self._primary_idx: prim_canvas}

        # ---- right-panel stacked widget ----
        self._right_stack = QStackedWidget()
        self._right_stack.addWidget(prim_canvas)
        self._right_stack.setCurrentIndex(0)

        # ---- field sliders (one per field) ----
        self._field_sliders = []
        self._field_labels  = []
        for field in system.fields:
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self._n_steps - 1)
            val_range = max(field.max_val - field.min_val, 1e-30)
            frac = (field.initial_val - field.min_val) / val_range
            slider.setValue(int(round(frac * (self._n_steps - 1))))
            slider.setSingleStep(1)
            slider.setPageStep(10)
            unit_str = f' {field.unit}' if field.unit else ''
            label = QLabel(f'{field.symbol} = {field.initial_val:.3g}{unit_str}')
            label.setMinimumWidth(90)
            self._field_sliders.append(slider)
            self._field_labels.append(label)

        # ---- mode selector (only when multiple fields) ----
        self._mode_combo = None
        if len(system.fields) > 1:
            self._mode_combo = QComboBox()
            for f in system.fields:
                self._mode_combo.addItem(f'{f.symbol}-x diagram')
            self._mode_combo.setCurrentIndex(self._primary_idx)

        # ---- shared controls ----
        self._reveal_cb  = QCheckBox('Reveal all')
        self._colors_btn = QPushButton('Colors\u2026')
        self._res_combo  = QComboBox()
        for _lbl, _n in [('Very Low (25)', 25), ('Low (50)', 50),
                          ('Medium (100)', 100), ('High (200)', 200),
                          ('Very High (500)', 500)]:
            self._res_combo.addItem(_lbl, _n)
        self._res_combo.setCurrentIndex(3)   # default: High (200)

        # ---- pre-compute / 3D widgets (only when multiple fields) ----
        self._precompute_btn    = None
        self._precompute_bar    = None
        self._precompute_status = None
        self._viz3d_btn         = None
        if len(system.fields) > 1:
            f0s = system.fields[0].symbol
            f1s = system.fields[1].symbol
            grid_label = (
                f'{f0s}-{f1s}-x')
            self._precompute_btn = QPushButton(
                f'Pre-compute full {grid_label}')
            n_total = self._n_steps ** 2
            self._precompute_bar = QProgressBar()
            self._precompute_bar.setRange(
                0, n_total)
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(False)
            self._precompute_status = QLabel('')
            self._viz3d_btn = QPushButton(
                '3D View\u2026')
            self._viz3d_btn.setEnabled(
                self._full_grid is not None)
            self._viz3d_btn.setToolTip(
                f'Opens 3D {grid_label} phase '
                f'diagram.\nRequires full grid '
                f'\u2014 click \u201cPre-compute '
                f'full {grid_label}\u201d first.')

        # ---- Export button ----
        self._export_btn = QPushButton('Export\u2026')
        self._export_btn.setToolTip(
            'Export phase diagram to HDF5 + XDMF for ParaView.')

        # ---- Builder button ----
        self._builder_btn = QPushButton('Builder\u2026')
        self._builder_btn.clicked.connect(self._open_builder)

        # ---- top row layout ----
        top_row = QHBoxLayout()
        top_row.addWidget(self._builder_btn)
        top_row.addSpacing(8)
        top_row.addWidget(self._export_btn)
        top_row.addSpacing(16)
        if self._viz3d_btn is not None:
            top_row.addWidget(self._viz3d_btn)
            top_row.addSpacing(8)
        if self._mode_combo is not None:
            top_row.addWidget(self._mode_combo)
            top_row.addSpacing(16)
        top_row.addWidget(self._reveal_cb)
        top_row.addWidget(self._colors_btn)
        top_row.addSpacing(8)
        top_row.addWidget(QLabel('Resolution:'))
        top_row.addWidget(self._res_combo)
        if self._precompute_btn is not None:
            top_row.addSpacing(16)
            top_row.addWidget(self._precompute_btn)
            top_row.addWidget(self._precompute_bar, stretch=1)
            top_row.addWidget(self._precompute_status)
        top_row.addStretch()

        canvas_row = QHBoxLayout()
        canvas_row.addWidget(self.gx_canvas)
        canvas_row.addWidget(self._right_stack)

        root = QVBoxLayout()
        root.addLayout(top_row)
        root.addLayout(canvas_row)

        # One slider row per field.
        for field, slider, label in zip(
                system.fields, self._field_sliders, self._field_labels):
            unit_str = f' {field.unit}' if field.unit else ''
            row = QHBoxLayout()
            row.addWidget(QLabel(f'{field.min_val:.3g}{unit_str}'))
            row.addWidget(slider)
            row.addWidget(QLabel(f'{field.max_val:.3g}{unit_str}'))
            row.addWidget(label)
            root.addLayout(row)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # ---- signal connections ----
        for i, slider in enumerate(self._field_sliders):
            slider.valueChanged.connect(
                lambda tick, fi=i: self._on_slider_changed(fi, tick))
            slider.sliderReleased.connect(
                lambda fi=i: self._on_slider_released(fi))
            slider.actionTriggered.connect(
                lambda action, fi=i: self._on_slider_action(fi, action))
        if self._mode_combo is not None:
            self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        if self._precompute_btn is not None:
            self._precompute_btn.clicked.connect(self._on_precompute_clicked)
        if self._viz3d_btn is not None:
            self._viz3d_btn.clicked.connect(self._on_3d_view_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._reveal_cb.toggled.connect(self._on_reveal_all_toggled)
        self._colors_btn.clicked.connect(self._on_colors_clicked)
        self._res_combo.currentIndexChanged.connect(self._on_resolution_changed)

        # Render initial state.
        self._on_slider_changed(0, self._field_sliders[0].value())

    # ------------------------------------------------------------------
    # Builder integration
    # ------------------------------------------------------------------

    def reload_system(self, system, spec=None):
        """Rebuild the UI for a new system (called by
        builder on Apply).

        Parameters
        ----------
        system : System  — live runtime objects
        spec   : SystemSpec or None — canonical spec;
            stored for builder/edit-mode reuse.
        """
        if self._worker is not None:
            self._worker.abort()
            self._worker.wait()
        if self._color_dialog is not None:
            self._color_dialog.close()
        if self._viz3d_window is not None:
            self._viz3d_window.close()
            self._viz3d_window = None
        if spec is not None:
            self._system_spec = spec
        print('Applying builder changes...',
              end=' ', flush=True)
        fixed_vals = {}
        if len(system.fields) > 1:
            try:
                fixed_vals = {
                    f.name: self._current_val(i)
                    for i, f in enumerate(
                        self.system.fields)
                    if i != 0
                    if i < len(self._field_sliders)}
            except (IndexError, AttributeError):
                fixed_vals = {
                    f.name: f.initial_val
                    for f in system.fields[1:]}
        precomputed_Tx = precompute_sweep_diagram(
            system, 0, fixed_vals)
        print('done.')
        self._init_system_state(
            system, precomputed_Tx, None)
        self._build_central_widget()

        # Re-activate edit mode if builder open.
        if (self._builder is not None
                and self._builder.isVisible()
                and not getattr(
                    self._builder,
                    '_closing', False)):
            live = {
                ps.name: ps
                for ps in self._system_spec.phases}
            self.gx_canvas.set_edit_mode(
                'handles', live)
            self.gx_canvas.phase_edited.connect(
                self._on_phase_edited)

    def _open_builder(self):
        """Open (or raise) the builder window.

        Pre-populates with the current SystemSpec and
        activates handle-drag edit mode on GxCanvas.
        """
        if (self._builder is None
                or not self._builder.isVisible()):
            from pde_builder import BuilderWindow
            self._builder = BuilderWindow(
                spec=self._system_spec)
            self._builder.system_applied.connect(
                self.reload_system)
            self._builder.patch_changed.connect(
                self._on_patch_changed)
            self._builder.finished.connect(
                self._on_builder_closed)

            live = {
                ps.name: ps
                for ps in self._system_spec.phases}
            self.gx_canvas.set_edit_mode(
                'handles', live)
            self.gx_canvas.phase_edited.connect(
                self._on_phase_edited)

        self._builder.raise_()
        self._builder.show()

    def _on_3d_view_clicked(self):
        """Open (or raise) the 3-D phase diagram."""
        if (self._viz3d_window is not None
                and self._viz3d_window.isVisible()):
            self._viz3d_window.raise_()
            return
        from pde_3d import (
            PhaseDiagram3D, Viz3DWindow)
        f0 = self.system.fields[0]
        f1 = self.system.fields[1]
        f0_arr = np.linspace(
            f0.max_val, f0.min_val,
            self._n_steps)
        f1_arr = np.linspace(
            f1.max_val, f1.min_val,
            self._n_steps)
        diagram = PhaseDiagram3D.from_grid(
            self._full_grid,
            f0, f0_arr, f1, f1_arr,
            self.system)
        self._viz3d_window = Viz3DWindow(
            diagram, colors=self._colors)
        self._viz3d_window.show()

    def _on_export_clicked(self):
        """Open a dialog to configure and run HDF5 + XDMF export."""
        dlg = QDialog(self)
        dlg.setWindowTitle('Export Phase Diagram')
        layout = QVBoxLayout()

        # --- Mode selector ---
        has_pressure = self.system.has_pressure
        mode_combo = QComboBox()
        mode_combo.addItem('2D slice (field-x)')
        if has_pressure:
            mode_combo.addItem('3D T-P-x volume')
            mode_combo.setCurrentIndex(1)
        layout.addWidget(mode_combo)

        # --- Stacked options for each mode ---
        stack = QStackedWidget()

        # -- Page 0: 2D slice options --
        page_2d = QWidget()
        form_2d = QFormLayout()

        nx_spin_2d = QSpinBox()
        nx_spin_2d.setRange(10, 2000)
        nx_spin_2d.setValue(200)
        form_2d.addRow('Composition points (n_x):', nx_spin_2d)

        nf_spin = QSpinBox()
        nf_spin.setRange(10, 2000)
        nf_spin.setValue(200)
        form_2d.addRow('Field points (n_field):', nf_spin)

        field_combo = QComboBox()
        for f in self.system.fields:
            field_combo.addItem(f'{f.symbol} ({f.name})')
        field_combo.setCurrentIndex(self._primary_idx)
        form_2d.addRow('Sweep field:', field_combo)

        page_2d.setLayout(form_2d)
        stack.addWidget(page_2d)

        # -- Page 1: 3D T-P-x options --
        page_3d = QWidget()
        form_3d = QFormLayout()

        nx_spin_3d = QSpinBox()
        nx_spin_3d.setRange(10, 500)
        nx_spin_3d.setValue(
            self._n_steps if has_pressure else 100)
        form_3d.addRow('Composition points (n_x):', nx_spin_3d)

        nt_spin = QSpinBox()
        nt_spin.setRange(10, 500)
        nt_spin.setValue(self._n_steps if has_pressure else 100)
        form_3d.addRow('Temperature points (n_T):', nt_spin)

        np_spin = QSpinBox()
        np_spin.setRange(10, 500)
        np_spin.setValue(self._n_steps if has_pressure else 50)
        form_3d.addRow('Pressure points (n_P):', np_spin)

        if has_pressure:
            T_f = self.system.T_field
            P_f = self.system.P_field
            range_label = QLabel(
                f'T: {T_f.min_val}\u2013{T_f.max_val} {T_f.unit},  '
                f'P: {P_f.min_val}\u2013{P_f.max_val} {P_f.unit}')
            range_label.setStyleSheet('color: grey; font-style: italic;')
            form_3d.addRow('Ranges:', range_label)

            if self._full_grid is not None:
                cache_label = QLabel(
                    f'Pre-computed grid available '
                    f'({self._n_steps}\u00d7{self._n_steps})')
                cache_label.setStyleSheet(
                    'color: green; font-style: italic;')
                form_3d.addRow('Cache:', cache_label)

        page_3d.setLayout(form_3d)
        stack.addWidget(page_3d)

        if has_pressure:
            stack.setCurrentIndex(1)

        mode_combo.currentIndexChanged.connect(stack.setCurrentIndex)
        layout.addWidget(stack)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        export_btn = QPushButton('Export')
        cancel_btn = QPushButton('Cancel')
        btn_row.addStretch()
        btn_row.addWidget(export_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        dlg.setLayout(layout)
        cancel_btn.clicked.connect(dlg.reject)

        def do_export():
            mode = mode_combo.currentIndex()
            dlg.accept()

            import re
            title = self.system.title or 'phase_diagram'
            default_name = re.sub(r'[^\w\-.]', '_', title).strip('_').lower()
            if mode == 1:
                default_name += '_3d'
            default_name += '.hdf5'
            path, _ = QFileDialog.getSaveFileName(
                self, 'Save HDF5 file', default_name,
                'HDF5 Files (*.hdf5);;All Files (*)')
            if not path:
                return

            if mode == 0:
                self._do_export_2d(path, nx_spin_2d.value(),
                                   nf_spin.value(),
                                   field_combo.currentIndex())
            else:
                self._do_export_3d(path, nx_spin_3d.value(),
                                   nt_spin.value(), np_spin.value())

        export_btn.clicked.connect(do_export)
        dlg.exec()

    def _do_export_2d(self, path, n_x, n_field, pfi):
        """Run the 2D field-x export with a progress bar."""
        fixed = {}
        for i, f in enumerate(self.system.fields):
            if i != pfi:
                fixed[f.name] = self._current_val(i)

        from pde_export import export_binary_Tx

        bar = QProgressBar(self)
        bar.setRange(0, n_field)
        bar.setFormat('Exporting 2D... %v / %m')
        self.statusBar().addWidget(bar, 1)
        bar.show()

        def on_progress(done, total):
            bar.setValue(done)
            QApplication.processEvents()

        try:
            xdmf = export_binary_Tx(
                self.system, path,
                n_x=n_x, n_field=n_field,
                primary_field_idx=pfi,
                fixed_field_values=fixed,
                progress_cb=on_progress)
            self.statusBar().showMessage(f'Exported: {xdmf}', 8000)
        except Exception as e:
            self.statusBar().showMessage(f'Export failed: {e}', 8000)
        finally:
            self.statusBar().removeWidget(bar)

    def _do_export_3d(self, path, n_x, n_T, n_P):
        """Run the 3D T-P-x export with a progress bar."""
        from pde_export import export_3d_TPx

        # Reuse the cached grid when dimensions match.
        grid = None
        if (self._full_grid is not None
                and len(self._full_grid) == n_T
                and len(self._full_grid[0]) == n_P):
            grid = self._full_grid

        total = n_T * n_P
        bar = QProgressBar(self)
        bar.setRange(0, total)
        fmt = ('Writing HDF5...' if grid
               else 'Exporting 3D...')
        bar.setFormat(f'{fmt} %v / %m')
        self.statusBar().addWidget(bar, 1)
        bar.show()

        def on_progress(done, total):
            bar.setValue(done)
            QApplication.processEvents()

        try:
            xdmf = export_3d_TPx(
                self.system, path,
                n_x=n_x, n_T=n_T, n_P=n_P,
                progress_cb=on_progress,
                precomputed_grid=grid)
            msg = f'Exported: {xdmf}'
            if grid:
                msg += ' (used cached grid)'
            self.statusBar().showMessage(msg, 8000)
        except Exception as e:
            self.statusBar().showMessage(
                f'Export failed: {e}', 8000)
        finally:
            self.statusBar().removeWidget(bar)

    def _on_builder_closed(self, result=None):
        """Deactivate handle-drag edit mode when the builder window closes."""
        self.gx_canvas.set_edit_mode('off')
        try:
            self.gx_canvas.phase_edited.disconnect(self._on_phase_edited)
        except RuntimeError:
            pass

    def _on_phase_edited(self, name, new_spec):
        """Live-update after a G(x) handle drag.

        1. Pushes the new PhaseSpec into the builder
           spinboxes so the two UIs stay in sync.
        2. Recomputes equilibrium from the updated
           SystemSpec so the hull is correct at the
           current T and P.

        Sweep canvases are NOT updated — they remain
        valid for the pre-Apply system.
        """
        if (self._builder is not None
                and self._builder.isVisible()):
            self._builder.update_phase_spec(
                name, new_spec)

        # Update the live SystemSpec.
        import copy
        for i, ps in enumerate(
                self._system_spec.phases):
            if ps.name == name:
                self._system_spec.phases[i] = (
                    new_spec)
                break

        # Build a temporary system with current T.
        build = copy.deepcopy(self._system_spec)
        t = self._current_val(0)
        for fs in build.fields:
            if fs.name == 'temperature':
                fs.initial_val = t
                break
        tmp_system = build.to_system()
        fv = {f.name: self._current_val(i)
              for i, f in enumerate(
                  self.system.fields)}
        result = compute_equilibrium(
            tmp_system, fv)
        self.gx_canvas.redraw(result)


    def _on_patch_changed(self, patched_system):
        """Lightweight GxCanvas redraw when a builder patch slider moves.

        The preview system already contains the PiecewisePatchModel for the
        patched phase.  We recompute equilibrium at the current T/P and redraw
        the G-x canvas without touching the sweep canvases or self.system.
        """
        fv = {f.name: self._current_val(i)
              for i, f in enumerate(self.system.fields)}
        try:
            result = compute_equilibrium(
                patched_system, fv)
            self.gx_canvas.redraw(result)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _val_from_tick(self, field_idx, tick):
        """Map slider tick 0..n_steps-1 → field value."""
        f = self.system.fields[field_idx]
        frac = tick / (self._n_steps - 1)
        return f.min_val + frac * (f.max_val - f.min_val)

    def _current_val(self, field_idx):
        """Return current slider value for the given field index."""
        return self._val_from_tick(field_idx, self._field_sliders[field_idx].value())

    def _current_result(self):
        """Return the EqResult nearest to the current slider positions."""
        prim_arr = self._field_arr[self._primary_idx]
        prim_val = self._current_val(self._primary_idx)
        if prim_arr is not None and self._precomputed[self._primary_idx] is not None:
            idx = int(np.argmin(np.abs(prim_arr - prim_val)))
            return self._precomputed[self._primary_idx][idx]
        fv = {f.name: self._current_val(i)
              for i, f in enumerate(
                  self.system.fields)}
        return compute_equilibrium(
            self.system, fv)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider_changed(self, field_idx, tick):
        """Handle slider movement — fast O(1) update for primary, label for secondary."""
        val = self._val_from_tick(field_idx, tick)
        f = self.system.fields[field_idx]
        unit_str = f' {f.unit}' if f.unit else ''
        self._field_labels[field_idx].setText(f'{f.symbol} = {val:.3g}{unit_str}')
        # Keep builder in sync with current T for patch polynomial computation.
        if field_idx == 0 and self._builder is not None and self._builder.isVisible():
            self._builder.set_current_T(val)

        if field_idx == self._primary_idx:
            # Primary slider — O(1) lookup.
            prim_arr = self._field_arr[self._primary_idx]
            data = self._precomputed[self._primary_idx]
            if prim_arr is not None and data is not None:
                idx = int(np.argmin(np.abs(prim_arr - val)))
                self.gx_canvas.redraw(data[idx])
                sweep = self._sweep_canvases.get(self._primary_idx)
                if sweep is not None:
                    sweep.set_cursor(val)
        elif self._full_grid is not None:
            # Secondary slider with cached full grid — instant primary sweep update.
            self._rebuild_primary_from_grid(field_idx, val)

    def _rebuild_primary_from_grid(self, sec_field_idx, sec_val):
        """Re-slice the cached grid after a secondary-field slider move."""
        prim_idx = self._primary_idx
        n = self._n_steps
        sec_arr = self._field_arr[sec_field_idx]
        if sec_arr is None:
            return
        sec_i = int(np.argmin(np.abs(sec_arr - sec_val)))

        # Slice the grid: fix sec dimension, vary prim dimension.
        # Currently only handles the T(primary)↔P(secondary) case (N=2 fields).
        if prim_idx == 0 and sec_field_idx == 1:
            new_data = [self._full_grid[i_T][sec_i] for i_T in range(n)]
        elif prim_idx == 1 and sec_field_idx == 0:
            new_data = [self._full_grid[sec_i][i_P] for i_P in range(n)]
        else:
            return

        self._precomputed[prim_idx] = new_data
        pf = self.system.fields[prim_idx]
        self._field_arr[prim_idx] = np.array([
            r.field_values[pf.name]
            for r in new_data])
        self.gx_canvas._y_lim = (
            _compute_ylim(new_data))
        prim_val = self._current_val(prim_idx)
        sweep = self._sweep_canvases.get(prim_idx)
        if sweep is not None:
            sweep.reset(new_data, current_val=prim_val)
        idx = int(np.argmin(np.abs(self._field_arr[prim_idx] - prim_val)))
        self.gx_canvas.redraw(new_data[idx])
        if sweep is not None:
            sweep.set_cursor(prim_val)

    def _on_slider_released(self, field_idx):
        """Handle slider release — trigger recompute for secondary sliders."""
        if field_idx == self._primary_idx:
            return  # primary release has no extra action
        if self._full_grid is not None:
            # Grid cached — already handled instantly in _on_slider_changed.
            return
        # Recompute primary sweep at the new secondary value.
        prim_idx = self._primary_idx
        prim_field = self.system.fields[prim_idx]
        sec_val = self._current_val(field_idx)
        sec_field = self.system.fields[field_idx]
        fixed_vals = {f.name: self._current_val(i)
                      for i, f in enumerate(self.system.fields)
                      if i != prim_idx}
        print(f'Recomputing {prim_field.symbol}-x diagram at '
              f'{sec_field.symbol} = {sec_val:.3g}'
              + (f' {sec_field.unit}' if sec_field.unit else '') + '...',
              end=' ', flush=True)
        new_data = precompute_sweep_diagram(
            self.system, prim_idx, fixed_vals, n_steps=self._n_steps)
        print('done.')
        self._precomputed[prim_idx] = new_data
        self._field_arr[prim_idx] = np.array([
            r.field_values[prim_field.name]
            for r in new_data])
        self.gx_canvas._y_lim = _compute_ylim(new_data)
        prim_val = self._current_val(prim_idx)
        sweep = self._sweep_canvases.get(prim_idx)
        if sweep is not None:
            sweep.reset(new_data, current_val=prim_val)
        idx = int(np.argmin(np.abs(self._field_arr[prim_idx] - prim_val)))
        self.gx_canvas.redraw(new_data[idx])
        if sweep is not None:
            sweep.set_cursor(prim_val)

    def _on_slider_action(self, field_idx, action):
        """Trigger recompute on bar-click / key-press (actionTriggered fires before
        sliderReleased for discrete steps like arrow keys and bar clicks)."""
        if self._field_sliders[field_idx].isSliderDown():
            return  # drag in progress — sliderReleased will handle it
        QTimer.singleShot(0, lambda: self._on_slider_released(field_idx))

    def _on_mode_changed(self, new_primary_idx):
        """Switch which field is the primary sweep axis (Y axis of sweep canvas)."""
        if new_primary_idx == self._primary_idx:
            return
        self._primary_idx = new_primary_idx
        prim_field = self.system.fields[new_primary_idx]
        prim_val = self._current_val(new_primary_idx)

        # Lazy: compute primary sweep for new field if not yet done.
        if self._precomputed[new_primary_idx] is None:
            fixed_vals = {f.name: self._current_val(i)
                          for i, f in enumerate(self.system.fields)
                          if i != new_primary_idx}
            print(f'Computing {prim_field.symbol}-x diagram...', end=' ', flush=True)
            new_data = precompute_sweep_diagram(
                self.system, new_primary_idx, fixed_vals, self._n_steps)
            print('done.')
            self._precomputed[new_primary_idx] = new_data
            self._field_arr[new_primary_idx] = (
                np.array([
                    r.field_values[prim_field.name]
                    for r in new_data]))
            self.gx_canvas._y_lim = _compute_ylim(new_data)

        # Create SweepCanvas for new primary if not already existing.
        if new_primary_idx not in self._sweep_canvases:
            canvas = SweepCanvas(
                self.system, prim_field, self._precomputed[new_primary_idx],
                colors=self._colors,
                two_phase_color=self._two_phase_color,
                two_phase_hatch=self._two_phase_hatch)
            self._sweep_canvases[new_primary_idx] = canvas
            self._right_stack.addWidget(canvas)

        canvas = self._sweep_canvases[new_primary_idx]
        self._right_stack.setCurrentWidget(canvas)
        canvas.reset(self._precomputed[new_primary_idx], current_val=prim_val)
        idx = int(np.argmin(np.abs(self._field_arr[new_primary_idx] - prim_val)))
        self.gx_canvas.redraw(self._precomputed[new_primary_idx][idx])
        canvas.set_cursor(prim_val)

    def _on_reveal_all_toggled(self, checked):
        """Show or hide the cover on all sweep canvases."""
        for canvas in self._sweep_canvases.values():
            canvas.set_reveal_all(checked)

    def _apply_colors(self, phase_colors, two_phase_color, two_phase_hatch):
        """Update shared color/hatch state and redraw all canvases live."""
        self._colors.update(phase_colors)
        self._two_phase_color = two_phase_color
        self._two_phase_hatch = two_phase_hatch
        for canvas in self._sweep_canvases.values():
            canvas._two_phase_color = two_phase_color
            canvas._two_phase_hatch = two_phase_hatch
            canvas.recolor()
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
            n_total = self._n_steps ** 2
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(True)
            self._precompute_status.setText('Computing... 0%')
            f0 = self.system.fields[0]
            f1 = self.system.fields[1]
            f0_vals = np.linspace(
                f0.max_val, f0.min_val,
                self._n_steps)
            f1_vals = np.linspace(
                f1.max_val, f1.min_val,
                self._n_steps)
            self._worker = FullGridWorker(
                self.system, f0, f0_vals,
                f1, f1_vals)
            self._worker.progress.connect(self._on_grid_progress)
            self._worker.finished.connect(self._on_grid_ready)
            self._worker.start()
            self._worker_state = 'running'
            self._precompute_btn.setText('Pause Computation')
        elif self._worker_state == 'running':
            self._worker.pause()
            self._worker_state = 'paused'
            self._precompute_btn.setText('Restart Computation')
        elif self._worker_state == 'paused':
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
        self._precompute_status.setText(
            f'Cached ({n_total:,} evaluations)')
        self._precompute_btn.setText(
            'Full grid cached')
        self._precompute_btn.setEnabled(False)
        if self._viz3d_btn is not None:
            self._viz3d_btn.setEnabled(True)

    def _on_resolution_changed(self, idx):
        """Recompute diagrams at a new step count chosen from the resolution combo."""
        n_steps = self._res_combo.itemData(idx)
        if n_steps == self._n_steps:
            return

        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait()
        self._worker = None
        self._worker_state = 'idle'
        self._full_grid = None
        self._n_steps = n_steps

        # Update all sliders' max range, preserving current field values.
        for i, (field, slider) in enumerate(
                zip(self.system.fields, self._field_sliders)):
            cur_val = self._val_from_tick(i, slider.value())
            slider.setMaximum(n_steps - 1)
            val_range = max(field.max_val - field.min_val, 1e-30)
            frac = (cur_val - field.min_val) / val_range
            slider.setValue(int(round(frac * (n_steps - 1))))

        # Reset precompute/3D widgets.
        if self._precompute_btn is not None:
            self._precompute_btn.setText('Pre-compute full T-P-x')
            self._precompute_btn.setEnabled(True)
            self._precompute_bar.setRange(0, n_steps ** 2)
            self._precompute_bar.setValue(0)
            self._precompute_bar.setVisible(False)
            self._precompute_status.setText('')
        if self._viz3d_btn is not None:
            self._viz3d_btn.setEnabled(False)

        # Recompute primary sweep.
        prim_idx = self._primary_idx
        fixed_vals = {f.name: self._current_val(i)
                      for i, f in enumerate(self.system.fields)
                      if i != prim_idx}
        self._precomputed[prim_idx] = precompute_sweep_diagram(
            self.system, prim_idx, fixed_vals, n_steps=n_steps)
        _pf = self.system.fields[prim_idx]
        self._field_arr[prim_idx] = np.array([
            r.field_values[_pf.name]
            for r in self._precomputed[prim_idx]])
        self.gx_canvas._y_lim = _compute_ylim(self._precomputed[prim_idx])

        # Recompute already-computed secondary sweeps.
        for i in range(len(self.system.fields)):
            if i == prim_idx:
                continue
            if self._precomputed[i] is not None:
                fixed = {f.name: self._current_val(j)
                         for j, f in enumerate(self.system.fields) if j != i}
                self._precomputed[i] = precompute_sweep_diagram(
                    self.system, i, fixed, n_steps=n_steps)
                _fi = self.system.fields[i]
                self._field_arr[i] = np.array([
                    r.field_values[_fi.name]
                    for r in self._precomputed[i]])
            else:
                f = self.system.fields[i]
                self._field_arr[i] = np.linspace(f.max_val, f.min_val, n_steps)

        # Redraw all sweep canvases.
        prim_val = self._current_val(prim_idx)
        for i, canvas in self._sweep_canvases.items():
            if self._precomputed[i] is not None:
                cur = self._current_val(i)
                canvas.reset(self._precomputed[i], current_val=cur)

        nearest = int(np.argmin(
            np.abs(self._field_arr[prim_idx] - prim_val)))
        self.gx_canvas.redraw(self._precomputed[prim_idx][nearest])

    def closeEvent(self, event):
        """Abort any running/paused background worker before closing."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait()
        if self._builder is not None:
            self._builder.close()
        if self._viz3d_window is not None:
            self._viz3d_window.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def precompute_sweep_diagram(system, primary_field_idx, fixed_field_values,
                              n_steps=N_T_STEPS):
    """Sweep primary field from max to min at fixed other field values.

    Parameters
    ----------
    system              : System
    primary_field_idx   : int — index into system.fields for the swept field
    fixed_field_values  : dict {field.name: value} for all non-primary fields
    n_steps             : int — number of sweep steps

    Returns list[EqResult] sorted by primary field descending.
    """
    pf = system.fields[primary_field_idx]
    prim_values = np.linspace(pf.max_val, pf.min_val, n_steps)
    fv = dict(fixed_field_values)
    results = []
    for v in prim_values:
        fv[pf.name] = v
        results.append(
            compute_equilibrium(system, fv))
    return results


def precompute_Tx_diagram(system, P=0.0, n_steps=N_T_STEPS):
    """Sweep T_max → T_min at fixed P; return list[EqResult] (T descending)."""
    return precompute_sweep_diagram(system, 0, {'pressure': P}, n_steps)


# Backward-compatible alias.
precompute_diagram = precompute_Tx_diagram


def precompute_Px_diagram(system, T, n_steps=N_P_STEPS):
    """Sweep P_max → P_min at fixed T; return list[EqResult] (P descending)."""
    p_idx = next((i for i, f in enumerate(system.fields)
                  if f.name == 'pressure'), 1)
    return precompute_sweep_diagram(system, p_idx, {'temperature': T}, n_steps)


def launch_ui(system, spec=None):
    """Pre-compute the phase diagram(s) and open the
    interactive window.

    Parameters
    ----------
    system : System — runtime objects
    spec   : SystemSpec or None — canonical spec for
        the builder; when None the builder will not
        have a spec until the user loads one.
    """
    # Ternary+ systems are not yet supported by
    # the interactive visualisation (see TODO C-11).
    # Show a diagnostic and exit gracefully.
    if system.n_components > 2:
        print(
            f'\nSystem has {system.n_components}'
            f' components '
            f'({", ".join(system.components)}).'
            f'\nTernary+ interactive '
            f'visualisation is not yet '
            f'implemented (C-11).'
            f'\nThe computation pipeline works'
            f' — use the Python API:\n'
            f'  from pde_input import '
            f'parse_system\n'
            f'  from pde_compute import '
            f'compute_equilibrium\n'
            f'  system = parse_system(infile)\n'
            f'  eq = compute_equilibrium('
            f'system, {{\"temperature\": T}})\n')
        sys.exit(0)

    if len(system.fields) > 1:
        f0 = system.fields[0]
        f1 = system.fields[1]
        print(
            f'Pre-computing {f0.symbol}-x diagram'
            f' at {f1.symbol}'
            f' = {f1.initial_val:.3g}'
            + (f' {f1.unit}' if f1.unit else '')
            + f' ({N_T_STEPS} evaluations)...',
            end=' ', flush=True)
        precomputed_Tx = precompute_sweep_diagram(
            system, 0,
            {f1.name: f1.initial_val})
        precomputed_Px = None
        print('done.')
    else:
        print('Pre-computing phase diagram...',
              end=' ', flush=True)
        precomputed_Tx = precompute_Tx_diagram(
            system)
        precomputed_Px = None
        print('done.')

    app = (QApplication.instance()
           or QApplication(sys.argv))
    window = MainWindow(
        system, precomputed_Tx,
        precomputed_Px, spec=spec)
    window.resize(1200, 650)
    window.show()
    sys.exit(app.exec())


def _make_default_system():
    """Return (System, SystemSpec) for a minimal
    single-phase system used when no input file is
    given.
    """
    from pde_phase import (
        FieldSpec, PhaseSpec, SystemSpec)
    spec = SystemSpec(
        title='New System',
        components=['A', 'B'],
        energy_form='HS',
        fields=[FieldSpec(
            name='temperature', symbol='T',
            unit='K', min_val=500,
            max_val=1500, initial_val=1500)],
        phases=[PhaseSpec(
            name='liquid', phase_type='liquid',
            xmin=0.0, xmax=1.0,
            model_type='HS',
            model_params={
                'H_coeffs': [8.0, -2.0, 2.0],
                'S_coeffs': [0.01],
            })])
    return spec.to_system(), spec


def launch_ui_empty():
    """Launch with a minimal default system and open
    the builder automatically.
    """
    system, spec = _make_default_system()
    print('Pre-computing default phase diagram...',
          end=' ', flush=True)
    precomputed_Tx = precompute_Tx_diagram(system)
    print('done.')
    app = (QApplication.instance()
           or QApplication(sys.argv))
    window = MainWindow(
        system, precomputed_Tx, spec=spec)
    window.resize(1200, 650)
    window.show()
    window._open_builder()
    sys.exit(app.exec())
