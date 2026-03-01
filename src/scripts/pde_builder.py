#!/usr/bin/env python3
"""
Graphical builder for PDE input systems.

Provides:
  PhaseData   — pure-Python data container for one phase
  SystemData  — pure-Python container for the full system, with
                to_system(), to_xml_str(), from_system(), from_xml()
  BuilderWindow — QDialog editor; emits system_applied(System) on Apply

Usage from pde_viz.py:
  from pde_builder import BuilderWindow
  builder = BuilderWindow(system=current_system)
  builder.system_applied.connect(main_window.reload_system)
  builder.show()
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import copy
from dataclasses import dataclass, field

from lxml import etree
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QStackedWidget,
    QVBoxLayout, QWidget,
)

from pde_energy import HSModel, PolyModel
from pde_phase import Phase, System


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _fmt(v):
    """Format a float for XML output (no trailing zeros, up to 10 sig figs)."""
    return f'{v:.10g}'


# ---------------------------------------------------------------------------
# Data model (pure Python, no Qt)
# ---------------------------------------------------------------------------

@dataclass
class PhaseData:
    """All data needed to describe one phase in either energy form."""
    name: str = 'phase'
    phase_type: str = 'solid'    # 'solid' | 'liquid' | 'gas' | 'end_member'
    xmin: float = 0.0
    xmax: float = 1.0
    ideal_gas: bool = False
    # HS form
    hs_H: list = field(default_factory=lambda: [0.0])
    hs_S: list = field(default_factory=lambda: [0.0])
    hs_V: list = field(default_factory=lambda: None)   # None → no PV term
    # Polynomial form: poly[i][j] = coefficient for x^i * T^j
    poly: list = field(default_factory=lambda: [[0.0]])


@dataclass
class SystemData:
    """Full system data container."""
    title: str = 'New System'
    components: list = field(default_factory=lambda: ['A', 'B'])
    energy_form: str = 'HS'
    T_min: float = 500.0
    T_max: float = 1500.0
    T_initial: float = 1500.0
    has_pressure: bool = False
    P_min: float = 1.0
    P_max: float = 5.0
    P_initial: float = 1.0
    R_gas: float = 0.0
    P_ref: float = 1.0
    P_unit: str = ''
    phases: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Build live objects
    # ------------------------------------------------------------------

    def to_system(self) -> System:
        """Construct and return a live System object."""
        R_gas = self.R_gas if self.has_pressure else 0.0
        P_ref = self.P_ref if self.has_pressure else 1.0

        phases = []
        for pd in self.phases:
            V_coeffs = pd.hs_V if pd.hs_V else None
            if self.energy_form == 'HS':
                model = HSModel(pd.hs_H, pd.hs_S,
                                V_coeffs=V_coeffs,
                                ideal_gas=pd.ideal_gas,
                                R_gas=R_gas, P_ref=P_ref)
            else:
                model = PolyModel(pd.poly,
                                  V_coeffs=V_coeffs,
                                  ideal_gas=pd.ideal_gas,
                                  R_gas=R_gas, P_ref=P_ref)
            phases.append(Phase(
                name=pd.name,
                phase_type=pd.phase_type,
                energy_model=model,
                xmin=pd.xmin,
                xmax=pd.xmax,
            ))

        return System(
            components=list(self.components),
            phases=phases,
            energy_form=self.energy_form,
            T_min=self.T_min,
            T_max=self.T_max,
            T_initial=self.T_initial,
            has_pressure=self.has_pressure,
            P_min=self.P_min if self.has_pressure else 1.0,
            P_max=self.P_max if self.has_pressure else 1.0,
            P_initial=self.P_initial if self.has_pressure else 1.0,
            R_gas=R_gas,
            P_ref=P_ref,
            P_unit=self.P_unit if self.has_pressure else '',
            title=self.title,
        )

    # ------------------------------------------------------------------
    # XML serialisation
    # ------------------------------------------------------------------

    def to_xml_str(self) -> str:
        """Return a pretty-printed XML string parseable by pde_input."""
        root = etree.Element('pde')

        if self.title:
            etree.SubElement(root, 'title').text = self.title

        sys_el = etree.SubElement(root, 'system')
        etree.SubElement(sys_el, 'components').text = ' '.join(self.components)
        etree.SubElement(sys_el, 'energy_form').text = self.energy_form

        temp_el = etree.SubElement(root, 'temperature')
        etree.SubElement(temp_el, 'min').text = _fmt(self.T_min)
        etree.SubElement(temp_el, 'max').text = _fmt(self.T_max)
        etree.SubElement(temp_el, 'initial').text = _fmt(self.T_initial)

        if self.has_pressure:
            pres_el = etree.SubElement(root, 'pressure')
            etree.SubElement(pres_el, 'min').text = _fmt(self.P_min)
            etree.SubElement(pres_el, 'max').text = _fmt(self.P_max)
            etree.SubElement(pres_el, 'initial').text = _fmt(self.P_initial)
            if self.R_gas:
                etree.SubElement(pres_el, 'R_gas').text = _fmt(self.R_gas)
            etree.SubElement(pres_el, 'P_ref').text = _fmt(self.P_ref)
            if self.P_unit:
                etree.SubElement(pres_el, 'unit').text = self.P_unit

        for pd in self.phases:
            phase_el = etree.SubElement(root, 'phase')
            phase_el.set('name', pd.name)
            phase_el.set('type', pd.phase_type)
            if pd.ideal_gas:
                phase_el.set('ideal_gas', 'true')

            is_point = (pd.xmin == pd.xmax)
            if pd.xmin != 0.0 or pd.xmax != 1.0:
                cr_el = etree.SubElement(phase_el, 'composition_range')
                cr_el.set('xmin', _fmt(pd.xmin))
                cr_el.set('xmax', _fmt(pd.xmax))

            energy_el = etree.SubElement(phase_el, 'energy')
            energy_el.set('model', 'point' if is_point else 'quadratic')

            if self.energy_form == 'HS':
                if is_point:
                    etree.SubElement(energy_el, 'H').text = _fmt(
                        pd.hs_H[0] if pd.hs_H else 0.0)
                    etree.SubElement(energy_el, 'S').text = _fmt(
                        pd.hs_S[0] if pd.hs_S else 0.0)
                else:
                    H_el = etree.SubElement(energy_el, 'H')
                    for i, c in enumerate(pd.hs_H):
                        H_el.set(f'x{i}', _fmt(c))
                    S_el = etree.SubElement(energy_el, 'S')
                    for i, c in enumerate(pd.hs_S):
                        S_el.set(f'x{i}', _fmt(c))
                if pd.hs_V:
                    V_el = etree.SubElement(energy_el, 'V')
                    for i, c in enumerate(pd.hs_V):
                        V_el.set(f'x{i}', _fmt(c))
            else:
                for i, t_coeffs in enumerate(pd.poly):
                    x_el = etree.SubElement(energy_el, f'x{i}')
                    for j, c in enumerate(t_coeffs):
                        x_el.set(f'a{j}', _fmt(c))
                if pd.hs_V:
                    V_el = etree.SubElement(energy_el, 'V')
                    for i, c in enumerate(pd.hs_V):
                        V_el.set(f'x{i}', _fmt(c))

        return etree.tostring(root, pretty_print=True, encoding='unicode')

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_system(cls, system) -> 'SystemData':
        """Populate a SystemData from a live System object."""
        sd = cls()
        sd.title = system.title
        sd.components = list(system.components)
        sd.energy_form = system.energy_form
        sd.T_min = system.T_min
        sd.T_max = system.T_max
        sd.T_initial = system.T_initial
        sd.has_pressure = system.has_pressure
        sd.P_min = system.P_min
        sd.P_max = system.P_max
        sd.P_initial = system.P_initial
        sd.R_gas = system.R_gas
        sd.P_ref = system.P_ref
        sd.P_unit = system.P_unit

        sd.phases = []
        for phase in system.phases:
            pd = PhaseData()
            pd.name = phase.name
            pd.phase_type = phase.phase_type
            pd.xmin = phase.xmin
            pd.xmax = phase.xmax
            model = phase.energy_model
            if isinstance(model, HSModel):
                pd.ideal_gas = model.ideal_gas
                pd.hs_H = model.H_coeffs.tolist()
                pd.hs_S = model.S_coeffs.tolist()
                pd.hs_V = (model.V_coeffs.tolist()
                           if model.V_coeffs is not None else None)
                pd.poly = [[0.0]]
            elif isinstance(model, PolyModel):
                pd.ideal_gas = model.ideal_gas
                pd.poly = [c.tolist() for c in model.coeffs]
                pd.hs_V = (model.V_coeffs.tolist()
                           if model.V_coeffs is not None else None)
                pd.hs_H = [0.0]
                pd.hs_S = [0.0]
            sd.phases.append(pd)

        return sd

    @classmethod
    def from_xml(cls, path) -> 'SystemData':
        """Parse an existing XML file and return a SystemData."""
        from pde_input import parse_system
        system = parse_system(path)
        return cls.from_system(system)


# ---------------------------------------------------------------------------
# Fitting layer — pure Python, no Qt
# ---------------------------------------------------------------------------

def apply_handle_drag(phase_data, drag_handle_idx,
                      handles_x, handles_G, T, energy_form):
    """Return updated PhaseData after a G(x) handle drag (Strategy 1: H-only).

    The function is a standalone, Qt-free fitting layer designed to be
    swappable as the interactive editing feature matures:

    Current implementation — Phase 2 (uniform translation):
        Any single-handle vertical drag shifts the whole G(x) curve by ΔG,
        implemented as H₀  +=  ΔG  (S held constant).

    Planned extensions (same signature, different body):
        Phase 4 — full 3-handle quadratic fit:
            Solve  H(xᵢ) = Gᵢ + T·S(xᵢ)  for i = 0,1,2  →  H₀, H₁, H₂
        Phase 8 — two-temperature H+S decomposition (Strategy 3):
            Collect drag snapshots at two T values, solve uniquely for ΔH, ΔS.

    Parameters
    ----------
    phase_data      : PhaseData   — original data (not modified in place)
    drag_handle_idx : int         — which handle was dragged (0=left, 1=mid, 2=right)
    handles_x       : list[float] — x positions of all handles (fixed during drag)
    handles_G       : list[float] — target G values after drag (new G at drag handle;
                                    original G at the others for Phase 4 constraints)
    T               : float       — temperature at which the drag occurred
    energy_form     : str         — 'HS' or 'polynomial'

    Returns
    -------
    PhaseData — deep copy of phase_data with updated coefficients.
    """
    import numpy as np

    new_data = copy.deepcopy(phase_data)

    if energy_form != 'HS':
        # Polynomial form editing deferred to a later phase.
        return new_data

    # ---- Phase 2: uniform translation of the G(x) curve ----
    # G(x, T) = H(x) − T·S(x).  Holding S fixed and dragging by ΔG means
    # absorbing ΔG entirely into H₀ (the constant term of H(x)).
    x_drag   = handles_x[drag_handle_idx]
    G_target = handles_G[drag_handle_idx]

    # Evaluate current G at the dragged x using the live coefficients.
    H_coeffs = np.asarray(phase_data.hs_H or [0.0])
    S_coeffs = np.asarray(phase_data.hs_S or [0.0])
    H_at_x   = np.polynomial.polynomial.polyval(x_drag, H_coeffs)
    S_at_x   = np.polynomial.polynomial.polyval(x_drag, S_coeffs)
    G_old    = H_at_x - T * S_at_x

    # Shift H₀ by the net ΔG so the curve translates uniformly.
    # TODO Phase 4: replace with a 3-point polynomial solve for H₀, H₁, H₂.
    delta_G    = G_target - G_old
    new_H      = list(phase_data.hs_H or [0.0])
    new_H[0]   = new_H[0] + delta_G
    new_data.hs_H = new_H

    return new_data


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------

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
        outer.addWidget(lbl)

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
        outer.addWidget(add_btn)
        outer.addWidget(rem_btn)

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

    def __init__(self, energy_form='HS', parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self._energy_form = energy_form

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

        self._ideal_gas_cb = QCheckBox('ideal gas')
        header.addWidget(self._ideal_gas_cb)

        header.addStretch()

        rem_btn = QPushButton('\u2715')   # ✕
        rem_btn.setFixedWidth(26)
        rem_btn.setToolTip('Remove phase')
        rem_btn.clicked.connect(self.remove_requested)
        header.addWidget(rem_btn)

        root.addLayout(header)

        # ---- stacked content: index 0 = HS, index 1 = poly ----
        self._stack = QStackedWidget()

        # HS page
        hs_widget = QWidget()
        hs_layout = QVBoxLayout(hs_widget)
        hs_layout.setContentsMargins(0, 0, 0, 0)
        hs_layout.setSpacing(4)
        # Hint: show the polynomial form so users see what each coefficient means
        hint_lbl = QLabel(
            'H(x) = H\u2080 + H\u2081\u00b7x + H\u2082\u00b7x\u00b2 + \u2026'
            '   \u2502   '
            'S(x) = S\u2080 + S\u2081\u00b7x + S\u2082\u00b7x\u00b2 + \u2026')
        hint_lbl.setStyleSheet('color: #777; font-style: italic;')
        hs_layout.addWidget(hint_lbl)
        self._H_row = CoeffRowWidget('H(x) =', [0.0], coeff_name='H')
        self._S_row = CoeffRowWidget('S(x) =', [0.0], coeff_name='S')
        self._V_enable_cb = QCheckBox('Enable PV term:')
        self._V_row = CoeffRowWidget('V(x) =', [0.0], coeff_name='V')
        self._V_row.setVisible(False)
        self._V_enable_cb.toggled.connect(self._V_row.setVisible)
        hs_layout.addWidget(self._H_row)
        hs_layout.addWidget(self._S_row)
        hs_layout.addWidget(self._V_enable_cb)
        hs_layout.addWidget(self._V_row)
        self._stack.addWidget(hs_widget)   # index 0

        # Poly page
        self._poly_widget = PolyPhaseCoeffWidget()
        self._stack.addWidget(self._poly_widget)  # index 1

        root.addWidget(self._stack)
        self.set_energy_form(energy_form)

    # -- public API ---------------------------------------------------------

    def set_energy_form(self, form):
        self._energy_form = form
        self._stack.setCurrentIndex(0 if form == 'HS' else 1)

    def get_phase_data(self) -> PhaseData:
        pd = PhaseData()
        pd.name = self._name_edit.text() or 'phase'
        pd.phase_type = self._type_combo.currentText()
        pd.xmin = self._xmin_sb.value()
        pd.xmax = self._xmax_sb.value()
        pd.ideal_gas = self._ideal_gas_cb.isChecked()
        pd.hs_H = self._H_row.get_coeffs()
        pd.hs_S = self._S_row.get_coeffs()
        pd.hs_V = self._V_row.get_coeffs() if self._V_enable_cb.isChecked() else None
        pd.poly = self._poly_widget.get_coeffs()
        return pd

    def set_phase_data(self, data: PhaseData):
        self._name_edit.setText(data.name)
        idx = self._type_combo.findText(data.phase_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._xmin_sb.setValue(data.xmin)
        self._xmax_sb.setValue(data.xmax)
        self._ideal_gas_cb.setChecked(data.ideal_gas)
        self._H_row.set_coeffs(data.hs_H or [0.0])
        self._S_row.set_coeffs(data.hs_S or [0.0])
        if data.hs_V:
            self._V_enable_cb.setChecked(True)
            self._V_row.set_coeffs(data.hs_V)
        else:
            self._V_enable_cb.setChecked(False)
        self._poly_widget.set_coeffs(data.poly or [[0.0]])


# ---------------------------------------------------------------------------
# Builder window
# ---------------------------------------------------------------------------

class BuilderWindow(QDialog):
    """Non-modal dialog for building / editing the thermodynamic system.

    Signals
    -------
    system_applied : Signal(object)
        Emitted with the newly constructed System when the user clicks Apply.
    """

    system_applied = Signal(object)

    def __init__(self, system=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('PDE Builder')
        self.setWindowFlags(Qt.Window)   # independent (non-modal) window
        self.resize(700, 600)

        self._phase_editors = []   # list[PhaseEditorWidget]
        self._comp_edits = []      # list[QLineEdit]

        root = QVBoxLayout(self)

        # ---- System group ----
        sys_group = QGroupBox('System')
        sys_layout = QHBoxLayout(sys_group)

        sys_layout.addWidget(QLabel('Title:'))
        self._title_edit = QLineEdit()
        self._title_edit.setMaximumWidth(160)
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
        self._form_combo.currentTextChanged.connect(self._on_form_changed)
        sys_layout.addWidget(self._form_combo)
        sys_layout.addStretch()

        root.addWidget(sys_group)

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
        self._P_unit_edit.setMaximumWidth(55)
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
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # ---- Populate ----
        if system is not None:
            sd = SystemData.from_system(system)
            self._populate_from_system_data(sd)
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

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _on_form_changed(self, form):
        for editor in self._phase_editors:
            editor.set_energy_form(form)

    def _add_phase(self, data=None):
        form = self._form_combo.currentText()
        editor = PhaseEditorWidget(energy_form=form)
        if data is not None:
            editor.set_phase_data(data)
        editor.remove_requested.connect(lambda e=editor: self._remove_phase(e))
        # Insert before the bottom stretch.
        count = self._phases_layout.count()
        self._phases_layout.insertWidget(count - 1, editor)
        self._phase_editors.append(editor)

    def _remove_phase(self, editor):
        if editor in self._phase_editors:
            self._phase_editors.remove(editor)
        self._phases_layout.removeWidget(editor)
        editor.setParent(None)

    # ------------------------------------------------------------------
    # Data collection / population
    # ------------------------------------------------------------------

    def _collect_system_data(self) -> SystemData:
        sd = SystemData()
        sd.title = self._title_edit.text()
        sd.components = [e.text() for e in self._comp_edits if e.text()]
        if not sd.components:
            sd.components = ['A', 'B']
        sd.energy_form = self._form_combo.currentText()
        sd.T_min = self._T_min_sb.value()
        sd.T_max = self._T_max_sb.value()
        sd.T_initial = self._T_init_sb.value()
        sd.has_pressure = self._pres_enable_cb.isChecked()
        sd.P_min = self._P_min_sb.value()
        sd.P_max = self._P_max_sb.value()
        sd.P_initial = self._P_init_sb.value()
        sd.P_unit = self._P_unit_edit.text()
        sd.R_gas = self._R_gas_sb.value()
        sd.P_ref = self._P_ref_sb.value()
        sd.phases = [e.get_phase_data() for e in self._phase_editors]
        return sd

    def _populate_from_system_data(self, sd: SystemData):
        # Clear existing phase editors.
        for editor in list(self._phase_editors):
            self._remove_phase(editor)
        # Clear existing component edits.
        for edit in list(self._comp_edits):
            edit.setParent(None)
        self._comp_edits.clear()

        self._title_edit.setText(sd.title)
        for comp in sd.components:
            self._add_component_edit(comp)

        idx = self._form_combo.findText(sd.energy_form)
        if idx >= 0:
            self._form_combo.setCurrentIndex(idx)

        self._T_min_sb.setValue(sd.T_min)
        self._T_max_sb.setValue(sd.T_max)
        self._T_init_sb.setValue(sd.T_initial)
        self._pres_enable_cb.setChecked(sd.has_pressure)
        self._P_min_sb.setValue(sd.P_min)
        self._P_max_sb.setValue(sd.P_max)
        self._P_init_sb.setValue(sd.P_initial)
        self._P_unit_edit.setText(sd.P_unit)
        self._R_gas_sb.setValue(sd.R_gas)
        self._P_ref_sb.setValue(sd.P_ref)

        for phase_data in sd.phases:
            self._add_phase(phase_data)

    # ------------------------------------------------------------------
    # Button slots
    # ------------------------------------------------------------------

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load XML', '', 'XML files (*.xml);;All files (*)')
        if path:
            try:
                sd = SystemData.from_xml(path)
                self._populate_from_system_data(sd)
            except Exception as exc:
                QMessageBox.warning(self, 'Load Error', str(exc))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save XML', '', 'XML files (*.xml);;All files (*)')
        if path:
            try:
                sd = self._collect_system_data()
                with open(path, 'w') as fh:
                    fh.write(sd.to_xml_str())
            except Exception as exc:
                QMessageBox.warning(self, 'Save Error', str(exc))

    def _on_apply(self):
        try:
            sd = self._collect_system_data()
            if not sd.phases:
                QMessageBox.warning(self, 'Apply Error',
                                    'Add at least one phase before applying.')
                return
            system = sd.to_system()
            self.system_applied.emit(system)
        except Exception as exc:
            QMessageBox.warning(self, 'Apply Error', str(exc))

    # ------------------------------------------------------------------
    # Canvas-edit integration (called by GxCanvas via MainWindow)
    # ------------------------------------------------------------------

    def update_phase_data(self, name: str, data: PhaseData):
        """Update the spinboxes for phase *name* with fresh PhaseData.

        Called by the G-x canvas (via MainWindow._on_phase_edited) whenever
        the user drags a handle.  Silently no-ops when the phase is not found.
        """
        for editor in self._phase_editors:
            if editor._name_edit.text() == name:
                editor.set_phase_data(data)
                return
