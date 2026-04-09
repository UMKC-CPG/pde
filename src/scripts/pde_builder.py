#!/usr/bin/env python3
"""
Graphical builder for PDE input systems.

Provides
--------
  BuilderWindow — QDialog editor that works directly with
      SystemSpec / PhaseSpec.  Emits system_applied(System,
      SystemSpec) on Apply.
  Fitting functions — apply_handle_drag, apply_rigid_shift,
      apply_xrange_drag (operate on PhaseSpec objects).

Usage from pde_viz.py::

    from pde_builder import BuilderWindow
    builder = BuilderWindow(spec=system_spec)
    builder.system_applied.connect(
        main_window.reload_system)
    builder.show()
"""

import os
import sys

_script_dir = os.path.dirname(
    os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

from pde_phase import (
    FieldSpec, PhaseSpec, SystemSpec,
)


# -----------------------------------------------------------
# Fitting layer — pure Python, no Qt
# -----------------------------------------------------------


def apply_handle_drag(phase_spec, drag_handle_idx,
                      handles_x, handles_G, T,
                      energy_form, P=0.0,
                      R_gas=0.0, P_ref=1.0,
                      fit_target='H'):
    """Return updated PhaseSpec after a vertical
    G(x) handle drag.

    *fit_target* selects which polynomial is fitted:
      'H'  (default) — adjust H₀, H₁, H₂; keep S.
      'S'            — adjust S₀, S₁, S₂; keep H.

    Higher-order coefficients (index >= 3) are kept.

    Parameters
    ----------
    phase_spec      : PhaseSpec — not modified
    drag_handle_idx : int       — 0=left, 1=mid, 2=right
    handles_x       : list[float] — handle x positions
    handles_G       : list[float] — target full G values
    T               : float     — temperature at drag
    energy_form     : str       — 'HS' or 'polynomial'
    P               : float     — pressure (default 0)
    R_gas           : float     — gas constant (default 0)
    P_ref           : float     — reference P (default 1)
    fit_target      : str       — 'H' or 'S' (default 'H')

    Returns
    -------
    PhaseSpec — deep copy with updated coefficients.
    """
    import numpy as np

    new_spec = copy.deepcopy(phase_spec)

    if energy_form != 'HS':
        return new_spec

    xs = np.asarray(handles_x, dtype=float)

    # Remove PV and ideal-gas contributions from
    # target G values to isolate H - T*S.
    net_G = np.asarray(
        handles_G, dtype=float).copy()
    V = phase_spec.V_coeffs
    if V and P != 0.0:
        V_vals = np.polynomial.polynomial.polyval(
            xs, np.asarray(V))
        net_G = net_G - P * V_vals
    if (phase_spec.ideal_gas and R_gas != 0.0
            and P > 0.0 and P_ref > 0.0):
        net_G -= R_gas * T * np.log(P / P_ref)

    # Vandermonde [1, x, x²] for each handle.
    A = np.column_stack(
        [np.ones(3), xs, xs ** 2])

    if fit_target == 'S':
        # -- S-mode: keep H, adjust S₀, S₁, S₂ --
        # G = H - T·S  →  S = (H - G) / T
        if abs(T) < 1e-10:
            return new_spec
        H_all = np.asarray(
            phase_spec.H_coeffs or [0.0])
        H_vals = np.polynomial.polynomial.polyval(
            xs, H_all)
        S_target = (H_vals - net_G) / T

        S_all = np.asarray(
            phase_spec.S_coeffs or [0.0])
        S_high = (S_all[3:]
                  if len(S_all) > 3
                  else np.array([]))

        # Subtract high-order S at handle positions.
        S_high_at_xs = np.zeros(3)
        for k, s_k in enumerate(S_high):
            S_high_at_xs += s_k * xs ** (k + 3)
        S_low_target = S_target - S_high_at_xs

        try:
            S_low_new = np.linalg.solve(
                A, S_low_target)
            new_spec.S_coeffs = (
                list(S_low_new) + list(S_high))
        except np.linalg.LinAlgError:
            # Degenerate — uniform S₀ shift.
            x_drag = handles_x[drag_handle_idx]
            H_at = float(
                np.polynomial.polynomial.polyval(
                    x_drag, H_all))
            net_G_at = float(
                net_G[drag_handle_idx])
            S_want = (H_at - net_G_at) / T
            S_now = float(
                np.polynomial.polynomial.polyval(
                    x_drag, S_all))
            new_S = list(
                phase_spec.S_coeffs or [0.0])
            if not new_S:
                new_S = [0.0]
            new_S[0] += S_want - S_now
            new_spec.S_coeffs = new_S

    else:
        # -- H-mode: keep S, adjust H₀, H₁, H₂ --
        # H = net_G + T·S
        S_arr = np.asarray(
            phase_spec.S_coeffs or [0.0])
        S_vals = np.polynomial.polynomial.polyval(
            xs, S_arr)
        H_target = net_G + T * S_vals

        H_all = np.asarray(
            phase_spec.H_coeffs or [0.0])
        H_high = (H_all[3:]
                  if len(H_all) > 3
                  else np.array([]))

        # Subtract high-order H at handle positions.
        H_high_at_xs = np.zeros(3)
        for k, h_k in enumerate(H_high):
            H_high_at_xs += h_k * xs ** (k + 3)
        H_low_target = H_target - H_high_at_xs

        try:
            H_low_new = np.linalg.solve(
                A, H_low_target)
            new_spec.H_coeffs = (
                list(H_low_new) + list(H_high))
        except np.linalg.LinAlgError:
            # Degenerate — uniform H₀ shift.
            H_old = np.asarray(
                phase_spec.H_coeffs or [0.0])
            S_old = np.asarray(
                phase_spec.S_coeffs or [0.0])
            x_drag = handles_x[drag_handle_idx]
            pv = np.polynomial.polynomial.polyval
            G_old = (pv(x_drag, H_old)
                     - T * pv(x_drag, S_old))
            if V and P != 0.0:
                G_old += P * pv(
                    x_drag, np.asarray(V))
            if (phase_spec.ideal_gas
                    and R_gas != 0.0
                    and P > 0.0
                    and P_ref > 0.0):
                G_old += (R_gas * T
                          * np.log(P / P_ref))
            delta_G = (handles_G[drag_handle_idx]
                       - G_old)
            new_H = list(
                phase_spec.H_coeffs or [0.0])
            new_H[0] += delta_G
            new_spec.H_coeffs = new_H

    return new_spec


def apply_xrange_drag(phase_spec, handle_idx,
                      new_x):
    """Return updated PhaseSpec after a horizontal
    endpoint-handle drag.

    Adjusts xmin (handle_idx == 0) or xmax
    (handle_idx == 2) to *new_x*, clamped to keep a
    minimum 0.02 separation and stay within [0, 1].

    Parameters
    ----------
    phase_spec : PhaseSpec — not modified in place
    handle_idx : int       — 0 = left, 2 = right
    new_x      : float     — new x from the drag

    Returns
    -------
    PhaseSpec — deep copy with updated xmin or xmax.
    """
    import numpy as np
    new_spec = copy.deepcopy(phase_spec)
    if handle_idx == 0:
        new_spec.xmin = float(np.clip(
            new_x, 0.0,
            phase_spec.xmax - 0.02))
    elif handle_idx == 2:
        new_spec.xmax = float(np.clip(
            new_x, phase_spec.xmin + 0.02, 1.0))
    return new_spec


def _shift_poly_coeffs(coeffs, dx):
    """Return ascending-order coefficients of p(x − dx).

    Given coefficients [c0, c1, …, cn] representing p(x) = Σ cᵢ xⁱ, returns
    the coefficients of p(x − dx) via numpy polynomial composition.  The curve
    shape is preserved; only the x-domain is translated.
    """
    import numpy as np
    if not coeffs or dx == 0.0:
        return list(coeffs) if coeffs else []
    # np.poly1d uses descending-degree order (highest power first).
    p = np.poly1d(list(reversed(coeffs)))
    q = np.poly1d([1.0, -dx])          # represents (x − dx)
    result = p(q)                       # polynomial composition p(x − dx)
    return list(reversed(result.c.tolist()))


def apply_rigid_shift(phase_spec, delta_G,
                      delta_x=0.0):
    """Return updated PhaseSpec after a rigid G(x)
    shift (vertical and/or horizontal).

    *delta_G* shifts H₀, translating the entire G(x,T)
    curve vertically.  *delta_x* translates the x-range
    and reparameterises H/S via p(x) → p(x − δ).

    Parameters
    ----------
    phase_spec : PhaseSpec — not modified in place
    delta_G    : float     — vertical shift (G units)
    delta_x    : float     — horizontal translation

    Returns
    -------
    PhaseSpec — deep copy with updated coefficients.
    """
    import numpy as np
    new_spec = copy.deepcopy(phase_spec)
    # Vertical shift: add delta_G to constant H term.
    H = list(new_spec.H_coeffs or [0.0])
    H[0] = H[0] + float(delta_G)
    new_spec.H_coeffs = H
    if delta_x != 0.0:
        width = phase_spec.xmax - phase_spec.xmin
        clamped_dx = float(np.clip(
            delta_x, -phase_spec.xmin,
            1.0 - phase_spec.xmax))
        if clamped_dx != 0.0:
            new_spec.xmin = (
                phase_spec.xmin + clamped_dx)
            new_spec.xmax = (
                new_spec.xmin + width)
            # Reparameterise: p(x) → p(x − dx).
            new_spec.H_coeffs = (
                _shift_poly_coeffs(
                    new_spec.H_coeffs,
                    clamped_dx))
            S = new_spec.S_coeffs
            if S:
                new_spec.S_coeffs = (
                    _shift_poly_coeffs(
                        S, clamped_dx))
    return new_spec


def _resolve_right_patch_chain(target_spec,
                               specs_by_name,
                               _seen=None):
    """Follow the right-patch chain from *target_spec*
    to find the root PhaseSpec.

    Returns (root_spec, xmax_eff).
    """
    if _seen is None:
        _seen = set()
    if (target_spec.patch_right_phase
            and target_spec.patch_right_x is not None
            and target_spec.name not in _seen):
        _seen.add(target_spec.name)
        nxt = specs_by_name.get(
            target_spec.patch_right_phase)
        if nxt is not None:
            return _resolve_right_patch_chain(
                nxt, specs_by_name, _seen)
    return target_spec, target_spec.xmax


def _resolve_left_patch_chain(target_spec,
                              specs_by_name,
                              _seen=None):
    """Follow the left-patch chain from *target_spec*
    to find the root PhaseSpec.

    Returns (root_spec, xmin_eff).
    """
    if _seen is None:
        _seen = set()
    if (target_spec.patch_left_phase
            and target_spec.patch_left_x is not None
            and target_spec.name not in _seen):
        _seen.add(target_spec.name)
        nxt = specs_by_name.get(
            target_spec.patch_left_phase)
        if nxt is not None:
            return _resolve_left_patch_chain(
                nxt, specs_by_name, _seen)
    return target_spec, target_spec.xmin


def compute_left_patch_H(phase_spec, target_spec,
                         x_cut, T=0.0,
                         specs_by_name=None):
    """Compute quadratic H coefficients Q(x) for the
    left patch region.

    Returns [q0, q1, q2] satisfying:
      1. Q(x_cut)  = H(x_cut)           — value
      2. Q'(x_cut) = H'(x_cut)          — slope
      3. G_patch'(xmin_eff) = G_root'(xmin_eff) — G
         slope match at the chain-root left edge.

    Condition 3 in H-space:
      Q'(xmin) = H_root'(xmin)
        + T·(S_phase'(xmin) − S_root'(xmin))

    Parameters
    ----------
    phase_spec   : PhaseSpec — phase being patched
    target_spec  : PhaseSpec — target for slope match
    x_cut        : float     — cut-off composition
    T            : float     — reference temperature
    specs_by_name : dict|None — for chain resolution

    Returns
    -------
    list[float] — [q0, q1, q2]
    """
    import numpy as np
    pv = np.polynomial.polynomial.polyval
    pd = np.polynomial.polynomial.polyder

    if specs_by_name is not None:
        root, xmin = _resolve_left_patch_chain(
            target_spec, specs_by_name)
    else:
        root = target_spec
        xmin = phase_spec.xmin

    H   = np.asarray(
        phase_spec.H_coeffs or [0.0])
    S   = np.asarray(
        phase_spec.S_coeffs or [0.0])
    H_t = np.asarray(
        root.H_coeffs or [0.0])
    S_t = np.asarray(
        root.S_coeffs or [0.0])

    H_val      = float(pv(x_cut, H))
    dH_at_cut  = (float(pv(x_cut, pd(H)))
                  if len(H) > 1 else 0.0)
    dH_t_xmin  = (float(pv(xmin, pd(H_t)))
                  if len(H_t) > 1 else 0.0)
    dS_xmin    = (float(pv(xmin, pd(S)))
                  if len(S) > 1 else 0.0)
    dS_t_xmin  = (float(pv(xmin, pd(S_t)))
                  if len(S_t) > 1 else 0.0)

    slope_target = (dH_t_xmin
                    + float(T)
                    * (dS_xmin - dS_t_xmin))

    dx = x_cut - xmin
    if abs(dx) < 1e-10:
        h = list(H)
        return (h + [0.0]
                * max(0, 3 - len(h)))[:3]

    q2 = (dH_at_cut - slope_target) / (2.0 * dx)
    q1 = slope_target - 2.0 * q2 * xmin
    q0 = H_val - q1 * x_cut - q2 * x_cut ** 2
    return [q0, q1, q2]


def compute_right_patch_H(phase_spec, target_spec,
                          x_cut, T=0.0,
                          specs_by_name=None):
    """Compute quadratic H coefficients Q(x) for the
    right patch region.

    Symmetric to compute_left_patch_H but matches the
    G slope at xmax_eff (the chain-root right edge).

    Parameters
    ----------
    phase_spec   : PhaseSpec — phase being patched
    target_spec  : PhaseSpec — target for slope match
    x_cut        : float     — cut-off composition
    T            : float     — reference temperature
    specs_by_name : dict|None — for chain resolution

    Returns
    -------
    list[float] — [q0, q1, q2]
    """
    import numpy as np
    pv = np.polynomial.polynomial.polyval
    pd = np.polynomial.polynomial.polyder

    if specs_by_name is not None:
        root, xmax = _resolve_right_patch_chain(
            target_spec, specs_by_name)
    else:
        root = target_spec
        xmax = phase_spec.xmax

    H   = np.asarray(
        phase_spec.H_coeffs or [0.0])
    S   = np.asarray(
        phase_spec.S_coeffs or [0.0])
    H_t = np.asarray(
        root.H_coeffs or [0.0])
    S_t = np.asarray(
        root.S_coeffs or [0.0])

    H_val      = float(pv(x_cut, H))
    dH_at_cut  = (float(pv(x_cut, pd(H)))
                  if len(H) > 1 else 0.0)
    dH_t_xmax  = (float(pv(xmax, pd(H_t)))
                  if len(H_t) > 1 else 0.0)
    dS_xmax    = (float(pv(xmax, pd(S)))
                  if len(S) > 1 else 0.0)
    dS_t_xmax  = (float(pv(xmax, pd(S_t)))
                  if len(S_t) > 1 else 0.0)

    slope_target = (dH_t_xmax
                    + float(T)
                    * (dS_xmax - dS_t_xmax))

    dx = xmax - x_cut
    if abs(dx) < 1e-10:
        h = list(H)
        return (h + [0.0]
                * max(0, 3 - len(h)))[:3]

    q2 = (slope_target - dH_at_cut) / (2.0 * dx)
    q1 = dH_at_cut - 2.0 * q2 * x_cut
    q0 = H_val - q1 * x_cut - q2 * x_cut ** 2
    return [q0, q1, q2]


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------

_PATCH_STEPS = 1000   # integer resolution for patch composition sliders


class _FloatSpinBox(QDoubleSpinBox):
    """General-purpose float spinbox: range ±1e9, 6 decimals, step 0.001."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-1e9, 1e9)
        self.setDecimals(6)
        self.setSingleStep(0.001)
        self.setMinimumWidth(70)


class CoeffRowWidget(QWidget):
    """One labelled row of coefficient spinboxes with dynamic +/- buttons.

    If *coeff_name* is given (e.g. ``'H'``), Unicode subscript headers
    (H₀, H₁, H₂, …) are shown above each spinbox so the user can see which
    coefficient of the polynomial each spinbox represents.

    Layout without coeff_name:
        [label]  [c0]  [c1]  …  [+]  [−]

    Layout with coeff_name='H', label='H(x) =':
        H(x) =    H₀        H₁        H₂       [+] [−]
                [spinbox] [spinbox] [spinbox]
    """

    _SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

    def __init__(self, label, coeffs=None, coeff_name='', parent=None):
        super().__init__(parent)
        self._spinboxes  = []
        self._sub_labels = []
        self._coeff_name = coeff_name

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        lbl = QLabel(label)
        lbl.setMinimumWidth(52)
        outer.addWidget(lbl, 0, Qt.AlignBottom)

        # Spin grid: row 0 = subscript labels (when coeff_name is set),
        #            row 1 = spinboxes  (row 0 when coeff_name is empty)
        self._spin_container = QWidget()
        self._spin_layout = QGridLayout(self._spin_container)
        self._spin_layout.setContentsMargins(0, 0, 0, 0)
        self._spin_layout.setSpacing(2)
        outer.addWidget(self._spin_container)

        add_btn = QPushButton('+')
        rem_btn = QPushButton('\u2212')   # minus sign
        add_btn.setFixedWidth(24)
        rem_btn.setFixedWidth(24)
        add_btn.setToolTip('Append coefficient')
        rem_btn.setToolTip('Remove last coefficient')
        add_btn.clicked.connect(lambda: self._add_spinbox(0.0))
        rem_btn.clicked.connect(self._remove_spinbox)
        outer.addWidget(add_btn, 0, Qt.AlignBottom)
        outer.addWidget(rem_btn, 0, Qt.AlignBottom)

        for c in (coeffs or [0.0]):
            self._add_spinbox(float(c))

    def _add_spinbox(self, value=0.0):
        col = len(self._spinboxes)
        if self._coeff_name:
            sub_text = self._coeff_name + str(col).translate(self._SUB)
            lbl = QLabel(sub_text)
            lbl.setAlignment(Qt.AlignCenter)
            self._spin_layout.addWidget(lbl, 0, col)
            self._sub_labels.append(lbl)
            spin_row = 1
        else:
            spin_row = 0
        sb = _FloatSpinBox()
        sb.setValue(value)
        self._spin_layout.addWidget(sb, spin_row, col)
        self._spinboxes.append(sb)

    def _remove_spinbox(self):
        if len(self._spinboxes) > 1:
            sb = self._spinboxes.pop()
            self._spin_layout.removeWidget(sb)
            sb.setParent(None)
            if self._coeff_name and self._sub_labels:
                lbl = self._sub_labels.pop()
                self._spin_layout.removeWidget(lbl)
                lbl.setParent(None)

    def get_coeffs(self):
        return [sb.value() for sb in self._spinboxes]

    def set_coeffs(self, coeffs):
        while len(self._spinboxes) > max(len(coeffs), 1):
            self._remove_spinbox()
        while len(self._spinboxes) < len(coeffs):
            self._add_spinbox(0.0)
        for sb, c in zip(self._spinboxes, coeffs):
            sb.setValue(float(c))


class PolyPhaseCoeffWidget(QWidget):
    """Grid widget for polynomial coefficients.

    Grid structure (0-indexed in the QGridLayout):
      (0, 0)     = empty corner
      (0, j+1)   = 'a_j' header  for j in 0..n_T-1
      (i+1, 0)   = 'x^i:' label  for i in 0..n_x-1
      (i+1, j+1) = spinbox        for coefficient x^i * T^j

    Rows (x-terms) and columns (T-powers) are dynamically resizable.
    Always at least 1 row and 1 column.
    """

    def __init__(self, coeffs=None, parent=None):
        super().__init__(parent)
        self._spinboxes = []   # [i][j] → QDoubleSpinBox
        self._x_labels  = []   # [i]    → QLabel
        self._T_labels  = []   # [j]    → QLabel
        self._n_x = 0
        self._n_T = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(2)
        root.addWidget(self._grid_widget)

        ctrl = QHBoxLayout()
        add_x = QPushButton('+ x term')
        rem_x = QPushButton('\u2212 x term')
        add_T = QPushButton('+ T power')
        rem_T = QPushButton('\u2212 T power')
        for btn in (add_x, rem_x, add_T, rem_T):
            ctrl.addWidget(btn)
        ctrl.addStretch()
        add_x.clicked.connect(self._add_x_row)
        rem_x.clicked.connect(self._remove_x_row)
        add_T.clicked.connect(self._add_T_col)
        rem_T.clicked.connect(self._remove_T_col)
        root.addLayout(ctrl)

        self.set_coeffs(coeffs if coeffs is not None else [[0.0]])

    # -- internal mutation --------------------------------------------------

    def _add_T_col(self):
        j = self._n_T
        lbl = QLabel(f'a{j}')
        lbl.setAlignment(Qt.AlignCenter)
        self._grid.addWidget(lbl, 0, j + 1)
        self._T_labels.append(lbl)
        for i in range(self._n_x):
            sb = _FloatSpinBox()
            self._grid.addWidget(sb, i + 1, j + 1)
            self._spinboxes[i].append(sb)
        self._n_T += 1

    def _remove_T_col(self):
        if self._n_T <= 1:
            return
        self._n_T -= 1
        lbl = self._T_labels.pop()
        self._grid.removeWidget(lbl)
        lbl.setParent(None)
        for i in range(self._n_x):
            sb = self._spinboxes[i].pop()
            self._grid.removeWidget(sb)
            sb.setParent(None)

    def _add_x_row(self):
        i = self._n_x
        lbl = QLabel(f'x^{i}:')
        self._grid.addWidget(lbl, i + 1, 0)
        self._x_labels.append(lbl)
        row = []
        for j in range(self._n_T):
            sb = _FloatSpinBox()
            self._grid.addWidget(sb, i + 1, j + 1)
            row.append(sb)
        self._spinboxes.append(row)
        self._n_x += 1

    def _remove_x_row(self):
        if self._n_x <= 1:
            return
        self._n_x -= 1
        lbl = self._x_labels.pop()
        self._grid.removeWidget(lbl)
        lbl.setParent(None)
        row = self._spinboxes.pop()
        for sb in row:
            self._grid.removeWidget(sb)
            sb.setParent(None)

    # -- public API ---------------------------------------------------------

    def get_coeffs(self):
        """Return list[list[float]] indexed by [x_power][T_power]."""
        return [[sb.value() for sb in row] for row in self._spinboxes]

    def set_coeffs(self, coeffs):
        """Set all coefficients, resizing grid as needed."""
        if not coeffs:
            coeffs = [[0.0]]
        n_x = len(coeffs)
        n_T = max((len(row) for row in coeffs), default=1) or 1

        # Adjust T columns first (so x-row additions get the right column count).
        while self._n_T > n_T:
            self._remove_T_col()
        while self._n_T < n_T:
            self._add_T_col()

        # Adjust x rows.
        while self._n_x > n_x:
            self._remove_x_row()
        while self._n_x < n_x:
            self._add_x_row()

        # Fill values.
        for i, row in enumerate(coeffs):
            for j, val in enumerate(row):
                if j < self._n_T:
                    self._spinboxes[i][j].setValue(float(val))


class PhaseEditorWidget(QFrame):
    """Editor for one phase.  Contains a header row and an energy content area.

    Signals
    -------
    remove_requested : emitted when the ✕ button is clicked.
    """

    remove_requested = Signal()
    patch_changed = Signal()   # emitted when any patch slider or combo changes

    def __init__(self, energy_form='HS', parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self._energy_form = energy_form
        self._vle_mode = False   # True when gas VLE page is active

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ---- header row ----
        header = QHBoxLayout()
        header.setSpacing(6)

        header.addWidget(QLabel('Name:'))
        self._name_edit = QLineEdit('phase')
        self._name_edit.setMaximumWidth(110)
        header.addWidget(self._name_edit)

        header.addWidget(QLabel('Type:'))
        self._type_combo = QComboBox()
        for t in ('solid', 'liquid', 'gas', 'end_member'):
            self._type_combo.addItem(t)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        header.addWidget(self._type_combo)

        header.addWidget(QLabel('x:'))
        self._xmin_sb = QDoubleSpinBox()
        self._xmin_sb.setRange(0.0, 1.0)
        self._xmin_sb.setDecimals(4)
        self._xmin_sb.setSingleStep(0.01)
        self._xmin_sb.setMaximumWidth(72)
        header.addWidget(self._xmin_sb)

        header.addWidget(QLabel('\u2013'))   # en-dash

        self._xmax_sb = QDoubleSpinBox()
        self._xmax_sb.setRange(0.0, 1.0)
        self._xmax_sb.setDecimals(4)
        self._xmax_sb.setSingleStep(0.01)
        self._xmax_sb.setValue(1.0)
        self._xmax_sb.setMaximumWidth(72)
        header.addWidget(self._xmax_sb)

        self._ideal_gas_cb = QCheckBox('Ideal gas')
        header.addWidget(self._ideal_gas_cb)

        header.addStretch()

        rem_btn = QPushButton('\u2715')   # ✕
        rem_btn.setFixedWidth(26)
        rem_btn.setToolTip('Remove phase')
        rem_btn.clicked.connect(self.remove_requested)
        header.addWidget(rem_btn)

        root.addLayout(header)

        # ---- stacked content: index 0 = HS, index 1 = poly, index 2 = VLE ----
        self._stack = QStackedWidget()

        # HS page (index 0)
        hs_widget = QWidget()
        hs_layout = QVBoxLayout(hs_widget)
        hs_layout.setContentsMargins(0, 0, 0, 0)
        hs_layout.setSpacing(4)
        # Primary equation label (text updated dynamically by _update_eq_label)
        self._hint_lbl = QLabel()
        self._hint_lbl.setStyleSheet('color: #777; font-style: italic;')
        hs_layout.addWidget(self._hint_lbl)
        self._H_row = CoeffRowWidget('H(x) =', [0.0], coeff_name='H')
        self._S_row = CoeffRowWidget('S(x) =', [0.0], coeff_name='S')
        self._V_enable_cb = QCheckBox('Enable PV term:')
        self._V_row = CoeffRowWidget('V(x) =', [0.0], coeff_name='V')
        self._V_row.setVisible(False)
        self._V_enable_cb.toggled.connect(self._V_row.setVisible)
        self._V_enable_cb.toggled.connect(self._update_eq_label)
        self._ideal_gas_cb.toggled.connect(self._update_eq_label)
        hs_layout.addWidget(self._H_row)
        hs_layout.addWidget(self._S_row)
        hs_layout.addWidget(self._V_enable_cb)
        hs_layout.addWidget(self._V_row)

        # ---- Patch controls ----
        self._patch_left_phase_name = ''   # requested target, used by update_patch_phase_list
        self._patch_right_phase_name = ''

        self._left_patch_cb = QCheckBox('Apply left patch')
        self._left_patch_container = QWidget()
        lp_layout = QHBoxLayout(self._left_patch_container)
        lp_layout.setContentsMargins(16, 0, 0, 0)
        lp_layout.setSpacing(6)
        self._left_patch_slider = QSlider(Qt.Horizontal)
        self._left_patch_slider.setRange(0, _PATCH_STEPS)
        self._left_patch_slider.setValue(0)
        self._left_patch_val_lbl = QLabel('0.0000')
        self._left_patch_val_lbl.setMinimumWidth(52)
        self._left_patch_phase_combo = QComboBox()
        self._left_patch_phase_combo.setMinimumWidth(80)
        lp_layout.addWidget(self._left_patch_slider)
        lp_layout.addWidget(self._left_patch_val_lbl)
        lp_layout.addWidget(self._left_patch_phase_combo)
        self._left_patch_container.setVisible(False)
        self._left_patch_cb.toggled.connect(self._left_patch_container.setVisible)
        self._left_patch_cb.toggled.connect(lambda _: self.patch_changed.emit())
        self._left_patch_slider.valueChanged.connect(self._update_patch_labels)
        self._left_patch_slider.valueChanged.connect(
            lambda _: self.patch_changed.emit())
        self._left_patch_phase_combo.currentIndexChanged.connect(
            self._on_left_patch_combo_changed)

        self._right_patch_cb = QCheckBox('Apply right patch')
        self._right_patch_container = QWidget()
        rp_layout = QHBoxLayout(self._right_patch_container)
        rp_layout.setContentsMargins(16, 0, 0, 0)
        rp_layout.setSpacing(6)
        self._right_patch_slider = QSlider(Qt.Horizontal)
        self._right_patch_slider.setRange(0, _PATCH_STEPS)
        self._right_patch_slider.setValue(_PATCH_STEPS)
        self._right_patch_val_lbl = QLabel('1.0000')
        self._right_patch_val_lbl.setMinimumWidth(52)
        self._right_patch_phase_combo = QComboBox()
        self._right_patch_phase_combo.setMinimumWidth(80)
        rp_layout.addWidget(self._right_patch_slider)
        rp_layout.addWidget(self._right_patch_val_lbl)
        rp_layout.addWidget(self._right_patch_phase_combo)
        self._right_patch_container.setVisible(False)
        self._right_patch_cb.toggled.connect(self._right_patch_container.setVisible)
        self._right_patch_cb.toggled.connect(lambda _: self.patch_changed.emit())
        self._right_patch_slider.valueChanged.connect(self._update_patch_labels)
        self._right_patch_slider.valueChanged.connect(
            lambda _: self.patch_changed.emit())
        self._right_patch_phase_combo.currentIndexChanged.connect(
            self._on_right_patch_combo_changed)

        hs_layout.addWidget(self._left_patch_cb)
        hs_layout.addWidget(self._left_patch_container)
        hs_layout.addWidget(self._right_patch_cb)
        hs_layout.addWidget(self._right_patch_container)

        self._stack.addWidget(hs_widget)   # index 0
        self._update_eq_label()            # set initial text

        # Connect xmin/xmax spinboxes to update patch slider labels when range changes.
        self._xmin_sb.valueChanged.connect(self._update_patch_labels)
        self._xmax_sb.valueChanged.connect(self._update_patch_labels)

        # Poly page (index 1)
        poly_page = QWidget()
        poly_layout = QVBoxLayout(poly_page)
        poly_layout.setContentsMargins(0, 0, 0, 0)
        poly_layout.setSpacing(4)
        poly_hint = QLabel(
            'G(x,T) = c\u2080(T) + c\u2081(T)\u00b7x + c\u2082(T)\u00b7x\u00b2 + \u2026'
            '   \u2502   '
            'c\u1d62(T) = a\u2080 + a\u2081\u00b7T + a\u2082\u00b7T\u00b2 + \u2026')
        poly_hint.setStyleSheet('color: #777; font-style: italic;')
        poly_layout.addWidget(poly_hint)
        self._poly_widget = PolyPhaseCoeffWidget()
        poly_layout.addWidget(self._poly_widget)
        self._stack.addWidget(poly_page)  # index 1

        # VLE page (index 2)
        vle_page = QWidget()
        vle_layout = QVBoxLayout(vle_page)
        vle_layout.setContentsMargins(0, 0, 0, 0)
        vle_layout.setSpacing(4)
        vle_hint = QLabel(
            'G\u1d33\u1d43\u02e2(x,T) = G\u2097\u1d35\u1d63(x,T) + \u0394H(x) \u2212 T\u00b7\u0394S(x)\n'
            '\u0394H(x) = L\u2090\u00b7(1\u2212x)\u00b2 + L\u1d2e\u00b7x\u00b2   '
            '   \u0394S(x) = (L\u2090/T\u2090)\u00b7(1\u2212x)\u00b2 + (L\u1d2e/T\u1d2e)\u00b7x\u00b2')
        vle_hint.setStyleSheet('color: #777; font-style: italic;')
        vle_layout.addWidget(vle_hint)

        vle_grid = QWidget()
        vle_grid_layout = QGridLayout(vle_grid)
        vle_grid_layout.setSpacing(6)
        vle_grid_layout.setContentsMargins(0, 0, 0, 0)

        self._T_bp_A_sb = _FloatSpinBox()
        self._T_bp_A_sb.setRange(0.0, 5000.0)
        self._T_bp_A_sb.setDecimals(2)
        self._T_bp_A_sb.setSingleStep(1.0)
        self._T_bp_A_sb.setValue(350.0)

        self._T_bp_B_sb = _FloatSpinBox()
        self._T_bp_B_sb.setRange(0.0, 5000.0)
        self._T_bp_B_sb.setDecimals(2)
        self._T_bp_B_sb.setSingleStep(1.0)
        self._T_bp_B_sb.setValue(400.0)

        self._L_A_sb = _FloatSpinBox()
        self._L_A_sb.setRange(0.0, 1e9)
        self._L_A_sb.setDecimals(4)
        self._L_A_sb.setSingleStep(0.1)
        self._L_A_sb.setValue(1.0)

        self._L_B_sb = _FloatSpinBox()
        self._L_B_sb.setRange(0.0, 1e9)
        self._L_B_sb.setDecimals(4)
        self._L_B_sb.setSingleStep(0.1)
        self._L_B_sb.setValue(1.0)

        vle_grid_layout.addWidget(QLabel('T_bp_A (K):'), 0, 0)
        vle_grid_layout.addWidget(self._T_bp_A_sb,       0, 1)
        vle_grid_layout.addWidget(QLabel('T_bp_B (K):'), 0, 2)
        vle_grid_layout.addWidget(self._T_bp_B_sb,       0, 3)
        vle_grid_layout.addWidget(QLabel('L_A:'),         1, 0)
        vle_grid_layout.addWidget(self._L_A_sb,           1, 1)
        vle_grid_layout.addWidget(QLabel('L_B:'),         1, 2)
        vle_grid_layout.addWidget(self._L_B_sb,           1, 3)
        vle_layout.addWidget(vle_grid)

        custom_hs_btn = QPushButton('Use custom H/S instead\u2026')
        custom_hs_btn.setToolTip(
            'Switch to the raw H/S coefficient page for full manual control')
        custom_hs_btn.clicked.connect(self._switch_to_hs_page)
        vle_layout.addWidget(custom_hs_btn)
        vle_layout.addStretch()
        self._stack.addWidget(vle_page)   # index 2

        # CALPHAD page (index 3)
        calphad_page = QWidget()
        calphad_lay = QVBoxLayout(calphad_page)
        calphad_lay.setContentsMargins(0, 0, 0, 0)
        calphad_lay.setSpacing(4)
        calphad_hint = QLabel(
            'G(x, T, P) from assessed TDB '
            'database.\nAll thermodynamic '
            'parameters come from the TDB '
            'file; no manual coefficients.')
        calphad_hint.setStyleSheet(
            'color: #777; font-style: italic;')
        calphad_lay.addWidget(calphad_hint)
        calphad_row = QHBoxLayout()
        calphad_row.addWidget(
            QLabel('TDB phase name:'))
        self._calphad_phase_edit = QLineEdit(
            'LIQUID')
        self._calphad_phase_edit.setMaximumWidth(
            160)
        self._calphad_phase_edit.setToolTip(
            'Phase identifier in the TDB file'
            ', e.g. LIQUID, FCC_A1, HCP_A3')
        calphad_row.addWidget(
            self._calphad_phase_edit)
        calphad_row.addStretch()
        calphad_lay.addLayout(calphad_row)
        calphad_lay.addStretch()
        self._stack.addWidget(
            calphad_page)                 # index 3

        root.addWidget(self._stack)
        self.set_energy_form(energy_form)

    # -- private helpers ----------------------------------------------------

    def _patch_slider_to_x(self, slider_val):
        xmin = self._xmin_sb.value()
        xmax = self._xmax_sb.value()
        return xmin + slider_val * (xmax - xmin) / _PATCH_STEPS

    def _x_to_patch_slider(self, x):
        xmin = self._xmin_sb.value()
        xmax = self._xmax_sb.value()
        if xmax == xmin:
            return 0
        return int(round((x - xmin) / (xmax - xmin) * _PATCH_STEPS))

    def _update_patch_labels(self, _=None):
        x_left  = self._patch_slider_to_x(self._left_patch_slider.value())
        x_right = self._patch_slider_to_x(self._right_patch_slider.value())
        self._left_patch_val_lbl.setText(f'{x_left:.4f}')
        self._right_patch_val_lbl.setText(f'{x_right:.4f}')

    def _on_left_patch_combo_changed(self, _=None):
        """Track user's manual combo selection and emit patch_changed."""
        self._patch_left_phase_name = self._left_patch_phase_combo.currentText()
        self.patch_changed.emit()

    def _on_right_patch_combo_changed(self, _=None):
        """Track user's manual combo selection and emit patch_changed."""
        self._patch_right_phase_name = self._right_patch_phase_combo.currentText()
        self.patch_changed.emit()

    def update_patch_phase_list(self, names: list):
        """Repopulate both patch phase dropdowns with *names* (other phases).

        The stored _patch_*_phase_name (set at set_phase_data() time, or updated
        when the user changes the combo) takes priority over whatever Qt may have
        auto-selected.  This prevents Qt's automatic index-0 selection from
        overriding the intended target when phases are loaded in a particular order.
        """
        for combo, attr in (
                (self._left_patch_phase_combo, '_patch_left_phase_name'),
                (self._right_patch_phase_combo, '_patch_right_phase_name')):
            # Stored name has priority; fall back to current combo text only when
            # no name has been set (e.g. a brand-new phase with no patch).
            wanted = getattr(self, attr, '') or combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            idx = combo.findText(wanted)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _on_type_changed(self, phase_type):
        """Switch to VLE page when type becomes 'gas'; back to HS otherwise."""
        if phase_type == 'gas' and self._energy_form == 'HS':
            self._vle_mode = True
        else:
            self._vle_mode = False
        self._update_stack_page()

    def _switch_to_hs_page(self):
        """Explicitly switch away from the VLE page to raw H/S."""
        self._vle_mode = False
        self._update_stack_page()

    def _update_stack_page(self):
        """Select the correct stack page from
        (_energy_form, _vle_mode).
        """
        if self._energy_form == 'calphad':
            self._stack.setCurrentIndex(3)
        elif self._energy_form == 'polynomial':
            self._stack.setCurrentIndex(1)
        elif self._vle_mode:
            self._stack.setCurrentIndex(2)
        else:
            self._stack.setCurrentIndex(0)

    # -- public API ---------------------------------------------------------

    def set_energy_form(self, form):
        self._energy_form = form
        # Hide controls not applicable to
        # CALPHAD phases.
        is_calphad = (form == 'calphad')
        self._ideal_gas_cb.setVisible(
            not is_calphad)
        self._update_stack_page()

    def _update_eq_label(self, _=None):
        text = 'G(x,T) = H(x) \u2212 T\u00b7S(x)'
        if self._V_enable_cb.isChecked():
            text += ' + P\u00b7V(x)'
        if self._ideal_gas_cb.isChecked():
            text += ' + R\u00b7T\u00b7ln(P/P\u2080)'
        text += ('\n'
                 'H(x) = H\u2080 + H\u2081\u00b7x + H\u2082\u00b7x\u00b2 + \u2026'
                 '   \u2502   '
                 'S(x) = S\u2080 + S\u2081\u00b7x + S\u2082\u00b7x\u00b2 + \u2026')
        if self._V_enable_cb.isChecked():
            text += '   \u2502   V(x) = V\u2080 + V\u2081\u00b7x + V\u2082\u00b7x\u00b2 + \u2026'
        self._hint_lbl.setText(text)

    def get_phase_spec(self):
        """Collect widget state into a PhaseSpec.

        Determines model_type from the energy form and
        whether patches are enabled.  VLE gas phases use
        model_type='HS' with a vle_params sub-dict.
        """
        name = (self._name_edit.text()
                or 'phase')
        ptype = self._type_combo.currentText()
        xmin = self._xmin_sb.value()
        xmax = self._xmax_sb.value()
        ig = self._ideal_gas_cb.isChecked()

        # -- VLE gas --------------------------------
        if (self._vle_mode
                and ptype == 'gas'
                and self._energy_form == 'HS'):
            return PhaseSpec(
                name=name, phase_type=ptype,
                xmin=xmin, xmax=xmax,
                model_type='HS',
                model_params={
                    'ideal_gas': ig,
                    'vle_params': {
                        'liquid_phase': '',
                        'T_bp_A': (
                            self._T_bp_A_sb.value()),
                        'T_bp_B': (
                            self._T_bp_B_sb.value()),
                        'L_A': (
                            self._L_A_sb.value()),
                        'L_B': (
                            self._L_B_sb.value()),
                    },
                })

        # -- CALPHAD form ---------------------------
        if self._energy_form == 'calphad':
            return PhaseSpec(
                name=name, phase_type=ptype,
                xmin=xmin, xmax=xmax,
                model_type='calphad',
                model_params={
                    'calphad_phase': (
                        self._calphad_phase_edit
                        .text().strip()
                        or 'LIQUID'),
                })

        # -- HS form --------------------------------
        if self._energy_form == 'HS':
            V = (self._V_row.get_coeffs()
                 if self._V_enable_cb.isChecked()
                 else None)
            mp = {
                'H_coeffs': (
                    self._H_row.get_coeffs()),
                'S_coeffs': (
                    self._S_row.get_coeffs()),
                'V_coeffs': V,
                'ideal_gas': ig,
            }
            has_l = self._left_patch_cb.isChecked()
            has_r = self._right_patch_cb.isChecked()
            if has_l or has_r:
                if has_l:
                    mp['patch_left_x'] = (
                        self._patch_slider_to_x(
                            self._left_patch_slider
                            .value()))
                    mp['patch_left_phase'] = (
                        self._left_patch_phase_combo
                        .currentText())
                if has_r:
                    mp['patch_right_x'] = (
                        self._patch_slider_to_x(
                            self._right_patch_slider
                            .value()))
                    mp['patch_right_phase'] = (
                        self._right_patch_phase_combo
                        .currentText())
                return PhaseSpec(
                    name=name, phase_type=ptype,
                    xmin=xmin, xmax=xmax,
                    model_type='piecewise_patch',
                    model_params=mp)
            return PhaseSpec(
                name=name, phase_type=ptype,
                xmin=xmin, xmax=xmax,
                model_type='HS',
                model_params=mp)

        # -- Polynomial form ------------------------
        V = (self._V_row.get_coeffs()
             if self._V_enable_cb.isChecked()
             else None)
        return PhaseSpec(
            name=name, phase_type=ptype,
            xmin=xmin, xmax=xmax,
            model_type='polynomial',
            model_params={
                'poly_coeffs': (
                    self._poly_widget.get_coeffs()),
                'V_coeffs': V,
                'ideal_gas': ig,
            })

    def set_phase_spec(self, spec):
        """Populate widgets from a PhaseSpec."""
        self._name_edit.setText(spec.name)
        self._type_combo.blockSignals(True)
        idx = self._type_combo.findText(
            spec.phase_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._type_combo.blockSignals(False)

        self._xmin_sb.setValue(spec.xmin)
        self._xmax_sb.setValue(spec.xmax)
        self._ideal_gas_cb.setChecked(
            spec.ideal_gas)

        # Always populate H/S so they are ready if
        # the user switches from VLE to raw H/S.
        self._H_row.set_coeffs(
            spec.H_coeffs or [0.0])
        self._S_row.set_coeffs(
            spec.S_coeffs or [0.0])
        V = spec.V_coeffs
        if V:
            self._V_enable_cb.setChecked(True)
            self._V_row.set_coeffs(V)
        else:
            self._V_enable_cb.setChecked(False)
        self._poly_widget.set_coeffs(
            spec.poly_coeffs or [[0.0]])

        vle = spec.vle_params
        if vle is not None:
            self._vle_mode = True
            self._T_bp_A_sb.setValue(
                vle.get('T_bp_A', 350.0))
            self._T_bp_B_sb.setValue(
                vle.get('T_bp_B', 400.0))
            self._L_A_sb.setValue(
                vle.get('L_A', 1.0))
            self._L_B_sb.setValue(
                vle.get('L_B', 1.0))
        else:
            self._vle_mode = False

        # Restore patch state.
        plx = spec.patch_left_x
        if plx is not None:
            self._left_patch_cb.setChecked(True)
            self._left_patch_slider.setValue(
                self._x_to_patch_slider(plx))
        else:
            self._left_patch_cb.setChecked(False)
            self._left_patch_slider.setValue(0)
        prx = spec.patch_right_x
        if prx is not None:
            self._right_patch_cb.setChecked(True)
            self._right_patch_slider.setValue(
                self._x_to_patch_slider(prx))
        else:
            self._right_patch_cb.setChecked(False)
            self._right_patch_slider.setValue(
                _PATCH_STEPS)
        self._update_patch_labels()

        # Store patch-phase names; applied by
        # update_patch_phase_list().
        self._patch_left_phase_name = (
            spec.patch_left_phase or '')
        self._patch_right_phase_name = (
            spec.patch_right_phase or '')
        for combo, nm in (
                (self._left_patch_phase_combo,
                 self._patch_left_phase_name),
                (self._right_patch_phase_combo,
                 self._patch_right_phase_name)):
            if nm:
                idx = combo.findText(nm)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        # Populate CALPHAD phase name.
        calphad_ph = spec.calphad_phase
        if calphad_ph:
            self._calphad_phase_edit.setText(
                calphad_ph)

        self._update_stack_page()



# ---------------------------------------------------------------------------
# Builder window
# ---------------------------------------------------------------------------

class BuilderWindow(QDialog):
    """Non-modal dialog for building/editing the
    thermodynamic system.

    Works directly with SystemSpec / PhaseSpec — no
    intermediate data model.

    Signals
    -------
    system_applied : Signal(object, object)
        Emitted with (System, SystemSpec) on Apply.
    patch_changed  : Signal(object)
        Emitted with a preview System on patch
        slider move.
    """

    # (System, SystemSpec) on Apply.
    system_applied = Signal(object, object)
    # Preview System on patch slider move.
    patch_changed = Signal(object)

    def __init__(self, spec=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('PDE Builder')
        self.setWindowFlags(Qt.Window)   # independent (non-modal) window
        self.resize(700, 760)

        self._phase_editors = []   # list[PhaseEditorWidget]
        self._last_names    = {}   # PhaseEditorWidget → last accepted name
        self._comp_edits = []      # list[QLineEdit]
        self._current_T  = None   # kept in sync by MainWindow via set_current_T()

        root = QVBoxLayout(self)

        # ---- System group ----
        sys_group = QGroupBox('System')
        sys_layout = QHBoxLayout(sys_group)

        sys_layout.addWidget(QLabel('Title:'))
        self._title_edit = QLineEdit()
        self._title_edit.setMinimumWidth(260)
        self._title_edit.setMaximumWidth(300)
        sys_layout.addWidget(self._title_edit)

        sys_layout.addSpacing(12)
        sys_layout.addWidget(QLabel('Components:'))
        self._comp_container = QWidget()
        self._comp_layout = QHBoxLayout(self._comp_container)
        self._comp_layout.setContentsMargins(0, 0, 0, 0)
        self._comp_layout.setSpacing(2)
        sys_layout.addWidget(self._comp_container)

        add_comp = QPushButton('+')
        rem_comp = QPushButton('\u2212')
        add_comp.setFixedWidth(24)
        rem_comp.setFixedWidth(24)
        add_comp.setToolTip('Add component')
        rem_comp.setToolTip('Remove last component')
        add_comp.clicked.connect(self._add_component)
        rem_comp.clicked.connect(self._remove_component)
        sys_layout.addWidget(add_comp)
        sys_layout.addWidget(rem_comp)

        sys_layout.addSpacing(12)
        sys_layout.addWidget(QLabel('Energy form:'))
        self._form_combo = QComboBox()
        self._form_combo.addItem('HS')
        self._form_combo.addItem('polynomial')
        self._form_combo.addItem('calphad')
        self._form_combo.currentTextChanged.connect(
            self._on_form_changed)
        sys_layout.addWidget(self._form_combo)
        sys_layout.addStretch()

        root.addWidget(sys_group)

        # ---- TDB group (CALPHAD only) ----
        self._tdb_group = QGroupBox(
            'TDB Database (CALPHAD)')
        tdb_layout = QHBoxLayout(
            self._tdb_group)
        tdb_layout.addWidget(
            QLabel('TDB file:'))
        self._tdb_path_edit = QLineEdit()
        self._tdb_path_edit.setMinimumWidth(300)
        self._tdb_path_edit.setToolTip(
            'Path to a Thermo-Calc Database '
            '(.tdb) file')
        tdb_layout.addWidget(
            self._tdb_path_edit)
        browse_btn = QPushButton(
            'Browse\u2026')
        browse_btn.clicked.connect(
            self._on_browse_tdb)
        tdb_layout.addWidget(browse_btn)
        tdb_layout.addStretch()
        self._tdb_group.setVisible(False)
        root.addWidget(self._tdb_group)

        # ---- Units group ----
        self._units_group = QGroupBox('Units')
        units_lay = QHBoxLayout(
            self._units_group)
        units_lay.addWidget(
            QLabel('Energy:'))
        self._energy_unit_combo = QComboBox()
        self._energy_unit_combo.setEditable(True)
        for u in ('kJ/mol', 'J/mol',
                  'cal/mol', 'kcal/mol',
                  'eV/atom'):
            self._energy_unit_combo.addItem(u)
        self._energy_unit_combo.setMaximumWidth(
            100)
        units_lay.addWidget(
            self._energy_unit_combo)
        units_lay.addSpacing(8)
        units_lay.addWidget(
            QLabel('Temperature:'))
        self._temp_unit_combo = QComboBox()
        self._temp_unit_combo.setEditable(True)
        self._temp_unit_combo.addItem('K')
        self._temp_unit_combo.setMaximumWidth(60)
        units_lay.addWidget(
            self._temp_unit_combo)
        units_lay.addSpacing(8)
        units_lay.addWidget(
            QLabel('Pressure:'))
        self._pres_unit_combo = QComboBox()
        self._pres_unit_combo.setEditable(True)
        for u in ('atm', 'Pa', 'bar', 'kPa'):
            self._pres_unit_combo.addItem(u)
        self._pres_unit_combo.setMaximumWidth(80)
        units_lay.addWidget(
            self._pres_unit_combo)
        units_lay.addStretch()
        root.addWidget(self._units_group)

        # ---- Temperature group ----
        temp_group = QGroupBox('Temperature')
        temp_layout = QHBoxLayout(temp_group)
        for label, attr in (('Min:', '_T_min_sb'), ('Max:', '_T_max_sb'),
                             ('Initial:', '_T_init_sb')):
            temp_layout.addWidget(QLabel(label))
            sb = _FloatSpinBox()
            sb.setRange(0, 1e7)
            sb.setDecimals(1)
            setattr(self, attr, sb)
            temp_layout.addWidget(sb)
        self._T_min_sb.setValue(500.0)
        self._T_max_sb.setValue(1500.0)
        self._T_init_sb.setValue(1500.0)
        temp_layout.addWidget(QLabel('K'))
        temp_layout.addStretch()
        root.addWidget(temp_group)

        # ---- Pressure group ----
        pres_group = QGroupBox('Pressure')
        pres_layout = QHBoxLayout(pres_group)
        self._pres_enable_cb = QCheckBox('Enable')
        pres_layout.addWidget(self._pres_enable_cb)
        pres_layout.addSpacing(8)
        for label, attr, default in (
                ('Min:',     '_P_min_sb',  1.0),
                ('Max:',     '_P_max_sb',  5.0),
                ('Initial:', '_P_init_sb', 1.0)):
            pres_layout.addWidget(QLabel(label))
            sb = _FloatSpinBox()
            sb.setRange(-1e9, 1e9)
            sb.setValue(default)
            setattr(self, attr, sb)
            pres_layout.addWidget(sb)
        pres_layout.addWidget(QLabel('Unit:'))
        self._P_unit_edit = QLineEdit()
        self._P_unit_edit.setMinimumWidth(40)
        self._P_unit_edit.setMaximumWidth(60)
        pres_layout.addWidget(self._P_unit_edit)
        pres_layout.addSpacing(8)
        pres_layout.addWidget(QLabel('R_gas:'))
        self._R_gas_sb = _FloatSpinBox()
        pres_layout.addWidget(self._R_gas_sb)
        pres_layout.addWidget(QLabel('P_ref:'))
        self._P_ref_sb = _FloatSpinBox()
        self._P_ref_sb.setValue(1.0)
        pres_layout.addWidget(self._P_ref_sb)
        pres_layout.addStretch()
        root.addWidget(pres_group)

        # ---- Phases scroll area ----
        root.addWidget(QLabel('Phases:'))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(180)
        self._phases_container = QWidget()
        self._phases_layout = QVBoxLayout(self._phases_container)
        self._phases_layout.setSpacing(4)
        self._phases_layout.addStretch()   # stretch always stays at bottom
        scroll.setWidget(self._phases_container)
        root.addWidget(scroll)

        add_phase_btn = QPushButton('+ Add Phase')
        add_phase_btn.clicked.connect(lambda: self._add_phase())
        root.addWidget(add_phase_btn)

        # ---- Button row ----
        btn_row = QHBoxLayout()
        load_btn = QPushButton('Load XML\u2026')
        save_btn = QPushButton('Save XML\u2026')
        apply_btn = QPushButton('Apply')
        close_btn = QPushButton('Close')
        load_btn.clicked.connect(self._on_load)
        save_btn.clicked.connect(self._on_save)
        apply_btn.clicked.connect(self._on_apply)
        close_btn.clicked.connect(self._on_close)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # ---- Populate ----
        if spec is not None:
            self._populate_from_spec(spec)
        else:
            self._add_component_edit('A')
            self._add_component_edit('B')

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def _add_component_edit(self, name=''):
        edit = QLineEdit(name)
        edit.setMaximumWidth(48)
        self._comp_layout.addWidget(edit)
        self._comp_edits.append(edit)

    def _add_component(self):
        self._add_component_edit()

    def _remove_component(self):
        if len(self._comp_edits) > 2:
            edit = self._comp_edits.pop()
            edit.setParent(None)

    def _on_browse_tdb(self):
        """Open file dialog for TDB selection."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select TDB file', '',
            'TDB files (*.tdb);;'
            'All files (*)')
        if path:
            self._tdb_path_edit.setText(path)

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _on_form_changed(self, form):
        for editor in self._phase_editors:
            editor.set_energy_form(form)
        self._tdb_group.setVisible(
            form == 'calphad')
        # Lock units to SI for CALPHAD systems.
        if form == 'calphad':
            self._energy_unit_combo.setCurrentText(
                'J/mol')
            self._temp_unit_combo.setCurrentText(
                'K')
            self._pres_unit_combo.setCurrentText(
                'Pa')
            self._units_group.setEnabled(False)
        else:
            self._units_group.setEnabled(True)

    def _unique_phase_name(self, base='phase'):
        """Return *base* if unused, otherwise *base_2*, *base_3*, …"""
        existing = {e._name_edit.text() for e in self._phase_editors}
        if base not in existing:
            return base
        i = 2
        while f'{base}_{i}' in existing:
            i += 1
        return f'{base}_{i}'

    def _on_phase_name_changed(self, editor):
        """Reject a name edit that duplicates another phase; revert to previous."""
        new_name = editor._name_edit.text().strip()
        if not new_name:
            editor._name_edit.setText(self._last_names.get(editor, 'phase'))
            return
        others = {e._name_edit.text() for e in self._phase_editors if e is not editor}
        if new_name in others:
            QMessageBox.warning(self, 'Duplicate Phase Name',
                                f'A phase named "{new_name}" already exists.\n'
                                'The name has been reverted.')
            editor._name_edit.setText(self._last_names.get(editor, 'phase'))
        else:
            self._last_names[editor] = new_name
            self._refresh_patch_phase_lists()

    def _add_phase(self, spec=None):
        form = self._form_combo.currentText()
        editor = PhaseEditorWidget(
            energy_form=form)
        if spec is not None:
            spec.name = self._unique_phase_name(
                spec.name)
            editor.set_phase_spec(spec)
        else:
            editor._name_edit.setText(
                self._unique_phase_name('phase'))
        self._last_names[editor] = editor._name_edit.text()
        editor.remove_requested.connect(lambda e=editor: self._remove_phase(e))
        editor._name_edit.editingFinished.connect(
            lambda e=editor: self._on_phase_name_changed(e))
        editor.patch_changed.connect(self._on_patch_changed)
        # Insert before the bottom stretch.
        count = self._phases_layout.count()
        self._phases_layout.insertWidget(count - 1, editor)
        self._phase_editors.append(editor)
        self._refresh_patch_phase_lists()

    def _refresh_patch_phase_lists(self):
        """Update each phase editor's patch-phase dropdowns with all other phase names."""
        for editor in self._phase_editors:
            names = [e._name_edit.text() for e in self._phase_editors if e is not editor]
            editor.update_patch_phase_list(names)

    def _remove_phase(self, editor):
        self._last_names.pop(editor, None)
        if editor in self._phase_editors:
            self._phase_editors.remove(editor)
        self._phases_layout.removeWidget(editor)
        editor.setParent(None)
        self._refresh_patch_phase_lists()

    # ------------------------------------------------------------------
    # Data collection / population
    # ------------------------------------------------------------------

    def _collect_system_spec(self):
        """Build a SystemSpec from current widget state.

        Also resolves VLE liquid_phase references
        (first liquid phase in declaration order).
        """
        comps = [e.text()
                 for e in self._comp_edits
                 if e.text()]
        if not comps:
            comps = ['A', 'B']

        # -- Field specs --------------------------
        t_spec = FieldSpec(
            name='temperature', symbol='T',
            unit='K',
            min_val=self._T_min_sb.value(),
            max_val=self._T_max_sb.value(),
            initial_val=self._T_init_sb.value())
        fspecs = [t_spec]
        if self._pres_enable_cb.isChecked():
            extras = {}
            rg = self._R_gas_sb.value()
            if rg:
                extras['R_gas'] = rg
            extras['P_ref'] = (
                self._P_ref_sb.value())
            p_spec = FieldSpec(
                name='pressure', symbol='P',
                unit=self._P_unit_edit.text(),
                min_val=self._P_min_sb.value(),
                max_val=self._P_max_sb.value(),
                initial_val=(
                    self._P_init_sb.value()),
                extras=extras)
            fspecs.append(p_spec)

        # -- Phase specs --------------------------
        pspecs = [
            e.get_phase_spec()
            for e in self._phase_editors]

        # Resolve VLE liquid_phase references.
        from pde_input import _resolve_vle_liquid
        _resolve_vle_liquid(pspecs)

        energy_form = (
            self._form_combo.currentText())

        # -- Units --------------------------------
        # Collect declared units from the combos.
        # CALPHAD systems are locked to SI; native
        # systems read the user's selections.
        tdb_path = ''
        if energy_form == 'calphad':
            tdb_path = (
                self._tdb_path_edit.text()
                .strip())
            calphad_comps = [
                c.upper() for c in comps]
            for ps in pspecs:
                if ps.model_type == 'calphad':
                    ps.model_params[
                        'components'
                    ] = calphad_comps
        units = {}
        e_unit = (
            self._energy_unit_combo
            .currentText().strip())
        t_unit = (
            self._temp_unit_combo
            .currentText().strip())
        p_unit = (
            self._pres_unit_combo
            .currentText().strip())
        if e_unit:
            units['energy'] = e_unit
        if t_unit:
            units['temperature'] = t_unit
        if p_unit:
            units['pressure'] = p_unit

        return SystemSpec(
            title=self._title_edit.text(),
            components=comps,
            energy_form=energy_form,
            fields=fspecs,
            phases=pspecs,
            tdb_path=tdb_path,
            units=units,
        )

    def _populate_from_spec(self, spec):
        """Fill all widgets from a SystemSpec."""
        # Clear existing editors.
        for editor in list(self._phase_editors):
            self._remove_phase(editor)
        for edit in list(self._comp_edits):
            edit.setParent(None)
        self._comp_edits.clear()

        self._title_edit.setText(spec.title)
        for comp in spec.components:
            self._add_component_edit(comp)

        idx = self._form_combo.findText(
            spec.energy_form)
        if idx >= 0:
            self._form_combo.setCurrentIndex(idx)

        # TDB path (CALPHAD systems).
        self._tdb_path_edit.setText(
            spec.tdb_path or '')

        # Units combos.
        u = spec.units or {}
        if u.get('energy'):
            self._energy_unit_combo.setCurrentText(
                u['energy'])
        if u.get('temperature'):
            self._temp_unit_combo.setCurrentText(
                u['temperature'])
        if u.get('pressure'):
            self._pres_unit_combo.setCurrentText(
                u['pressure'])

        # Fill field widgets from FieldSpecs.
        has_pres = False
        for fs in spec.fields:
            if fs.name == 'temperature':
                self._T_min_sb.setValue(
                    fs.min_val)
                self._T_max_sb.setValue(
                    fs.max_val)
                self._T_init_sb.setValue(
                    fs.initial_val)
            elif fs.name == 'pressure':
                has_pres = True
                self._P_min_sb.setValue(
                    fs.min_val)
                self._P_max_sb.setValue(
                    fs.max_val)
                self._P_init_sb.setValue(
                    fs.initial_val)
                self._P_unit_edit.setText(
                    fs.unit)
                self._R_gas_sb.setValue(
                    fs.extras.get('R_gas', 0.0))
                self._P_ref_sb.setValue(
                    fs.extras.get('P_ref', 1.0))
        self._pres_enable_cb.setChecked(has_pres)

        for ps in spec.phases:
            self._add_phase(ps)
        self._refresh_patch_phase_lists()

    # ------------------------------------------------------------------
    # Button slots
    # ------------------------------------------------------------------

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load XML', '',
            'XML files (*.xml);;All files (*)')
        if path:
            try:
                from pde_input import (
                    parse_system_spec)
                spec = parse_system_spec(path)
                self._populate_from_spec(spec)
            except Exception as exc:
                QMessageBox.warning(
                    self, 'Load Error', str(exc))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save XML', '',
            'XML files (*.xml);;All files (*)')
        if path:
            try:
                spec = self._collect_system_spec()
                with open(path, 'w') as fh:
                    fh.write(spec.to_xml_str())
            except Exception as exc:
                QMessageBox.warning(
                    self, 'Save Error', str(exc))

    # ------------------------------------------------------
    # Patch slider live-preview support
    # ------------------------------------------------------

    def set_current_T(self, T: float):
        """Called by MainWindow on slider move."""
        self._current_T = float(T)

    def _build_system_from_spec(self, spec):
        """Build a System from *spec*, overriding T_ref
        with the current slider temperature when set.

        Returns the (System, spec) pair.
        """
        build = copy.deepcopy(spec)
        if self._current_T is not None:
            for fs in build.fields:
                if fs.name == 'temperature':
                    fs.initial_val = (
                        self._current_T)
                    break
        return build.to_system(), spec

    def _on_patch_changed(self):
        """Emit a preview System when a patch slider
        or combo changes.
        """
        try:
            spec = self._collect_system_spec()
            if not spec.phases:
                return
            system, _ = (
                self._build_system_from_spec(spec))
            self.patch_changed.emit(system)
        except Exception:
            pass

    # ------------------------------------------------------
    # Apply / close
    # ------------------------------------------------------

    def _on_apply(self):
        """Apply current state.  Returns True on
        success, False on error.
        """
        try:
            spec = self._collect_system_spec()
            if not spec.phases:
                QMessageBox.warning(
                    self, 'Apply Error',
                    'Add at least one phase '
                    'before applying.')
                return False
            system, spec = (
                self._build_system_from_spec(spec))
        except Exception as exc:
            QMessageBox.warning(
                self, 'Apply Error', str(exc))
            return False
        self.system_applied.emit(system, spec)
        return True

    def _on_close(self):
        self._closing = True
        if not self._on_apply():
            self._closing = False
            return
        self.close()

    def closeEvent(self, event):
        if not getattr(self, '_closing', False):
            try:
                spec = self._collect_system_spec()
                if spec.phases:
                    system, spec = (
                        self._build_system_from_spec(
                            spec))
                    self.system_applied.emit(
                        system, spec)
            except Exception:
                pass
        event.accept()

    # ------------------------------------------------------
    # Canvas-edit integration
    # ------------------------------------------------------

    def update_phase_spec(self, name, spec):
        """Update spinboxes for phase *name* from a
        PhaseSpec.  Called by GxCanvas via MainWindow
        after a handle drag.  Silently no-ops when the
        phase is not found.
        """
        for editor in self._phase_editors:
            if editor._name_edit.text() == name:
                editor.set_phase_spec(spec)
                return
