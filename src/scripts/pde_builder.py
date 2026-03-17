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
    QMessageBox, QPushButton, QScrollArea, QSlider, QStackedWidget,
    QVBoxLayout, QWidget,
)

from pde_energy import HSModel, PiecewisePatchModel, PolyModel, compute_vle_gas_hs
from pde_phase import Field, Phase, System


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
    # VLE reparameterisation (gas phases only)
    # When not None: {'T_bp_A': float, 'T_bp_B': float, 'L_A': float, 'L_B': float}
    vle: dict = field(default=None)
    # Patch cutoffs: composition value within [xmin, xmax], or None = no patch
    patch_left: object = field(default=None)   # float | None
    patch_right: object = field(default=None)  # float | None
    # Target phase name for each patch (empty string = none selected)
    patch_left_phase: str = field(default='')
    patch_right_phase: str = field(default='')

    @property
    def is_vle_gas(self) -> bool:
        """True when this is a gas phase using the VLE reparameterisation."""
        return self.phase_type == 'gas' and self.vle is not None


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

    def to_system(self, current_T=None) -> System:
        """Construct and return a live System object.

        Parameters
        ----------
        current_T : float or None
            Temperature to use when computing left-patch polynomials (G slope
            matching).  Defaults to self.T_initial when None.
        """
        T_patch = float(current_T) if current_T is not None else self.T_initial
        R_gas = self.R_gas if self.has_pressure else 0.0
        P_ref = self.P_ref if self.has_pressure else 1.0

        # Index PhaseData by name for patch target look-ups.
        phase_data_by_name = {pd.name: pd for pd in self.phases}

        # First pass: build all non-VLE phases; collect first liquid's H/S.
        phases = []
        liq_H = [0.0]
        liq_S = [0.0]
        liq_found = False
        pending_vle = []   # list of PhaseData for VLE gas phases

        for pd in self.phases:
            if pd.is_vle_gas:
                pending_vle.append(pd)
                continue

            V_coeffs = pd.hs_V if pd.hs_V else None

            # Build energy model — use PiecewisePatchModel when patches are set.
            has_left  = (self.energy_form == 'HS'
                         and pd.patch_left is not None
                         and pd.patch_left_phase
                         and pd.patch_left > pd.xmin + 1e-10)
            has_right = (self.energy_form == 'HS'
                         and pd.patch_right is not None
                         and pd.patch_right_phase
                         and pd.patch_right < pd.xmax - 1e-10)

            if has_left or has_right:
                H_left, x_cut_left   = None, None
                H_right, x_cut_right = None, None
                xmin_eff = pd.xmin
                xmax_eff = pd.xmax
                if has_left:
                    tpd = phase_data_by_name.get(pd.patch_left_phase)
                    if tpd is not None:
                        H_left = compute_left_patch_H(
                            pd, tpd, pd.patch_left, T=T_patch,
                            phase_data_by_name=phase_data_by_name)
                        x_cut_left = pd.patch_left
                        # Extend xmin to chain root's xmin.
                        _, xmin_eff = _resolve_left_patch_chain(tpd, phase_data_by_name)
                        xmin_eff = min(pd.xmin, xmin_eff)
                if has_right:
                    tpd = phase_data_by_name.get(pd.patch_right_phase)
                    if tpd is not None:
                        H_right = compute_right_patch_H(
                            pd, tpd, pd.patch_right, T=T_patch,
                            phase_data_by_name=phase_data_by_name)
                        x_cut_right = pd.patch_right
                        # Extend xmax to chain root's xmax.
                        _, xmax_eff = _resolve_right_patch_chain(tpd, phase_data_by_name)
                        xmax_eff = max(pd.xmax, xmax_eff)
                if H_left is not None or H_right is not None:
                    model = PiecewisePatchModel(
                        pd.hs_H, pd.hs_S,
                        H_left=H_left, x_cut_left=x_cut_left,
                        H_right=H_right, x_cut_right=x_cut_right,
                        V_coeffs=V_coeffs,
                        ideal_gas=pd.ideal_gas,
                        R_gas=R_gas, P_ref=P_ref,
                    )
                    if H_left  is not None:
                        model.patch_left_phase_name  = pd.patch_left_phase
                    if H_right is not None:
                        model.patch_right_phase_name = pd.patch_right_phase
                else:
                    model = HSModel(pd.hs_H, pd.hs_S,
                                    V_coeffs=V_coeffs,
                                    ideal_gas=pd.ideal_gas,
                                    R_gas=R_gas, P_ref=P_ref)
                    xmin_eff = pd.xmin
                    xmax_eff = pd.xmax
            elif self.energy_form == 'HS':
                model = HSModel(pd.hs_H, pd.hs_S,
                                V_coeffs=V_coeffs,
                                ideal_gas=pd.ideal_gas,
                                R_gas=R_gas, P_ref=P_ref)
                xmin_eff = pd.xmin
                xmax_eff = pd.xmax
            else:
                model = PolyModel(pd.poly,
                                  V_coeffs=V_coeffs,
                                  ideal_gas=pd.ideal_gas,
                                  R_gas=R_gas, P_ref=P_ref)
                xmin_eff = pd.xmin
                xmax_eff = pd.xmax
            phases.append(Phase(
                name=pd.name,
                phase_type=pd.phase_type,
                energy_model=model,
                xmin=xmin_eff,
                xmax=xmax_eff,
            ))
            if pd.phase_type == 'liquid' and not liq_found and self.energy_form == 'HS':
                liq_H = list(pd.hs_H) if pd.hs_H else [0.0]
                liq_S = list(pd.hs_S) if pd.hs_S else [0.0]
                liq_found = True

        # Second pass: build VLE gas phases using compute_vle_gas_hs().
        for pd in pending_vle:
            vp = pd.vle
            H_gas, S_gas = compute_vle_gas_hs(
                liq_H, liq_S,
                vp['T_bp_A'], vp['T_bp_B'],
                vp['L_A'],    vp['L_B'],
            )
            V_coeffs = pd.hs_V if pd.hs_V else None
            model = HSModel(H_gas, S_gas,
                            V_coeffs=V_coeffs,
                            ideal_gas=pd.ideal_gas,
                            R_gas=R_gas, P_ref=P_ref)
            model.vle_params = dict(vp)
            phases.append(Phase(
                name=pd.name,
                phase_type=pd.phase_type,
                energy_model=model,
                xmin=pd.xmin,
                xmax=pd.xmax,
            ))

        t_field = Field(name='temperature', symbol='T', unit='K',
                        min_val=self.T_min, max_val=self.T_max,
                        initial_val=self.T_initial)
        fields = [t_field]
        if self.has_pressure:
            p_field = Field(name='pressure', symbol='P',
                            unit=self.P_unit,
                            min_val=self.P_min, max_val=self.P_max,
                            initial_val=self.P_initial)
            fields.append(p_field)

        return System(
            components=list(self.components),
            phases=phases,
            energy_form=self.energy_form,
            fields=fields,
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

        fields_el = etree.SubElement(root, 'fields')
        t_el = etree.SubElement(fields_el, 'field')
        t_el.set('name', 'temperature')
        t_el.set('symbol', 'T')
        t_el.set('unit', 'K')
        t_el.set('min', _fmt(self.T_min))
        t_el.set('max', _fmt(self.T_max))
        t_el.set('initial', _fmt(self.T_initial))
        if self.has_pressure:
            p_el = etree.SubElement(fields_el, 'field')
            p_el.set('name', 'pressure')
            p_el.set('symbol', 'P')
            p_el.set('unit', self.P_unit)
            p_el.set('min', _fmt(self.P_min))
            p_el.set('max', _fmt(self.P_max))
            p_el.set('initial', _fmt(self.P_initial))
            if self.R_gas:
                p_el.set('R_gas', _fmt(self.R_gas))
            p_el.set('P_ref', _fmt(self.P_ref))

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

            # VLE gas phases: emit <vle> element instead of <energy>.
            if pd.is_vle_gas:
                vp = pd.vle
                vle_el = etree.SubElement(phase_el, 'vle')
                vle_el.set('T_bp_A', _fmt(vp['T_bp_A']))
                vle_el.set('T_bp_B', _fmt(vp['T_bp_B']))
                vle_el.set('L_A',    _fmt(vp['L_A']))
                vle_el.set('L_B',    _fmt(vp['L_B']))
                continue

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

            # Patch metadata (HS form only; patches not supported for polynomial form).
            if pd.patch_left is not None and pd.patch_left_phase:
                pl_el = etree.SubElement(phase_el, 'patch_left')
                pl_el.set('x_cut', _fmt(pd.patch_left))
                pl_el.set('phase', pd.patch_left_phase)
            if pd.patch_right is not None and pd.patch_right_phase:
                pr_el = etree.SubElement(phase_el, 'patch_right')
                pr_el.set('x_cut', _fmt(pd.patch_right))
                pr_el.set('phase', pd.patch_right_phase)

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
            if isinstance(model, PiecewisePatchModel):
                pd.ideal_gas = model.ideal_gas
                pd.hs_V = (model.V_coeffs.tolist()
                           if model.V_coeffs is not None else None)
                pd.hs_H = model.H_orig.tolist()
                pd.hs_S = model.S_coeffs.tolist()
                pd.poly = [[0.0]]
                if model.x_cut_left is not None:
                    pd.patch_left       = model.x_cut_left
                    pd.patch_left_phase = model.patch_left_phase_name
                if model.x_cut_right is not None:
                    pd.patch_right       = model.x_cut_right
                    pd.patch_right_phase = model.patch_right_phase_name
            elif isinstance(model, HSModel):
                pd.ideal_gas = model.ideal_gas
                pd.hs_V = (model.V_coeffs.tolist()
                           if model.V_coeffs is not None else None)
                # Recover VLE parameterisation if present.
                vp = getattr(model, 'vle_params', None)
                if vp is not None:
                    pd.vle = dict(vp)
                    pd.hs_H = [0.0]
                    pd.hs_S = [0.0]
                else:
                    pd.hs_H = model.H_coeffs.tolist()
                    pd.hs_S = model.S_coeffs.tolist()
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
                      handles_x, handles_G, T, energy_form,
                      P=0.0, R_gas=0.0, P_ref=1.0,
                      fit_target='H'):
    """Return updated PhaseData after a vertical G(x) handle drag.

    *fit_target* selects which polynomial is fitted:
      'H'  (default) — adjust H₀, H₁, H₂; keep S unchanged.
      'S'            — adjust S₀, S₁, S₂; keep H unchanged.

    In both cases higher-order coefficients (index ≥ 3) are preserved.

    Parameters
    ----------
    phase_data      : PhaseData   — original data (not modified in place)
    drag_handle_idx : int         — which handle was dragged (0=left, 1=mid, 2=right)
    handles_x       : list[float] — x positions of all handles (fixed during drag)
    handles_G       : list[float] — target full G values after drag (including pressure
                                    terms), new G at drag handle; original G at the
                                    other two as polynomial constraints
    T               : float       — temperature at which the drag occurred
    energy_form     : str         — 'HS' or 'polynomial'
    P               : float       — pressure at drag time (default 0; backward-compatible)
    R_gas           : float       — ideal-gas constant for the system (default 0)
    P_ref           : float       — reference pressure for ideal-gas term (default 1)
    fit_target      : str         — 'H' or 'S' (default 'H')

    Returns
    -------
    PhaseData — deep copy of phase_data with updated coefficients.
    """
    import numpy as np

    new_data = copy.deepcopy(phase_data)

    if energy_form != 'HS':
        # Polynomial form editing deferred to a later phase.
        return new_data

    xs = np.asarray(handles_x, dtype=float)

    # --- helper: remove PV and ideal-gas contributions from target G values ---
    net_G = np.asarray(handles_G, dtype=float).copy()
    if phase_data.hs_V and P != 0.0:
        V_vals = np.polynomial.polynomial.polyval(xs, np.asarray(phase_data.hs_V))
        net_G  = net_G - P * V_vals
    if phase_data.ideal_gas and R_gas != 0.0 and P > 0.0 and P_ref > 0.0:
        net_G = net_G - R_gas * T * np.log(P / P_ref)

    # Vandermonde matrix [1, x, x²] for each handle position.
    A = np.column_stack([np.ones(3), xs, xs ** 2])

    if fit_target == 'S':
        # ---- S-mode: keep H fixed, adjust S₀, S₁, S₂ ----
        # G = H(x) - T·S(x)  →  S(x) = (H(x) - G(x)) / T
        if abs(T) < 1e-10:
            return new_data   # cannot solve for S at T ≈ 0
        H_all  = np.asarray(phase_data.hs_H or [0.0])
        H_vals = np.polynomial.polynomial.polyval(xs, H_all)
        S_target = (H_vals - net_G) / T

        S_all  = np.asarray(phase_data.hs_S or [0.0])
        S_high = S_all[3:] if len(S_all) > 3 else np.array([])

        # Subtract high-order S contribution at handle positions.
        S_high_at_xs = np.zeros(3)
        for k, s_k in enumerate(S_high):
            S_high_at_xs += s_k * xs ** (k + 3)
        S_low_target = S_target - S_high_at_xs

        try:
            S_low_new    = np.linalg.solve(A, S_low_target)
            new_data.hs_S = list(S_low_new) + list(S_high)
        except np.linalg.LinAlgError:
            # Degenerate geometry — fall back to uniform S₀ shift.
            x_drag       = handles_x[drag_handle_idx]
            H_at_drag    = float(np.polynomial.polynomial.polyval(x_drag, H_all))
            net_G_drag   = float(net_G[drag_handle_idx])
            S_target_drag = (H_at_drag - net_G_drag) / T
            S_current_drag = float(np.polynomial.polynomial.polyval(x_drag, S_all))
            new_S        = list(phase_data.hs_S or [0.0])
            if not new_S:
                new_S = [0.0]
            new_S[0]    += S_target_drag - S_current_drag
            new_data.hs_S = new_S

    else:
        # ---- H-mode (default): keep S fixed, adjust H₀, H₁, H₂ ----
        # Invert G = H - T·S  →  H = net_G + T·S
        S_coeffs = np.asarray(phase_data.hs_S or [0.0])
        S_vals   = np.polynomial.polynomial.polyval(xs, S_coeffs)
        H_target = net_G + T * S_vals

        H_all  = np.asarray(phase_data.hs_H or [0.0])
        H_high = H_all[3:] if len(H_all) > 3 else np.array([])

        # Subtract high-order H contribution at handle positions.
        H_high_at_xs = np.zeros(3)
        for k, h_k in enumerate(H_high):
            H_high_at_xs += h_k * xs ** (k + 3)
        H_low_target = H_target - H_high_at_xs

        try:
            H_low_new    = np.linalg.solve(A, H_low_target)
            new_data.hs_H = list(H_low_new) + list(H_high)
        except np.linalg.LinAlgError:
            # Degenerate geometry — fall back to uniform H₀ shift.
            H_old  = np.asarray(phase_data.hs_H or [0.0])
            S_old  = np.asarray(phase_data.hs_S or [0.0])
            x_drag = handles_x[drag_handle_idx]
            G_old  = (np.polynomial.polynomial.polyval(x_drag, H_old)
                      - T * np.polynomial.polynomial.polyval(x_drag, S_old))
            if phase_data.hs_V and P != 0.0:
                G_old += P * np.polynomial.polynomial.polyval(
                    x_drag, np.asarray(phase_data.hs_V))
            if phase_data.ideal_gas and R_gas != 0.0 and P > 0.0 and P_ref > 0.0:
                G_old += R_gas * T * np.log(P / P_ref)
            delta_G    = handles_G[drag_handle_idx] - G_old
            new_H      = list(phase_data.hs_H or [0.0])
            new_H[0]  += delta_G
            new_data.hs_H = new_H

    return new_data


def apply_xrange_drag(phase_data, handle_idx, new_x):
    """Return updated PhaseData after a horizontal endpoint-handle drag (Phase 3).

    Adjusts xmin (handle_idx == 0) or xmax (handle_idx == 2) to *new_x*,
    clamped to keep a minimum separation of 0.02 between xmin and xmax and
    to stay within [0, 1].  The midpoint handle (idx 1) is not a valid target
    for horizontal drag and is silently ignored.

    Parameters
    ----------
    phase_data : PhaseData — original data (not modified in place)
    handle_idx : int       — 0 = left endpoint, 2 = right endpoint
    new_x      : float     — new x position from the drag

    Returns
    -------
    PhaseData — deep copy with updated xmin or xmax.
    """
    import numpy as np
    new_data = copy.deepcopy(phase_data)
    if handle_idx == 0:
        new_data.xmin = float(np.clip(new_x, 0.0, phase_data.xmax - 0.02))
    elif handle_idx == 2:
        new_data.xmax = float(np.clip(new_x, phase_data.xmin + 0.02, 1.0))
    return new_data


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


def apply_rigid_shift(phase_data, delta_G, delta_x=0.0):
    """Return updated PhaseData after a rigid G(x) shift (vertical and/or horizontal).

    *delta_G* is added to the constant H coefficient (hs_H[0]), shifting
    G(x, T) = H(x) − T·S(x) by exactly *delta_G* at every composition and
    temperature.

    *delta_x* translates the phase's x-range: both xmin and xmax are shifted
    by the same amount (clamped to keep them within [0, 1]).  The H and S
    polynomial coefficients are reparameterised via p(x) → p(x − δ) so that
    the curve shape is preserved at the new positions.

    Parameters
    ----------
    phase_data : PhaseData — original data (not modified in place)
    delta_G    : float     — vertical shift in G units (positive = up)
    delta_x    : float     — horizontal translation in composition units

    Returns
    -------
    PhaseData — deep copy with updated coefficients and/or xmin/xmax.
    """
    import numpy as np
    new_data = copy.deepcopy(phase_data)
    # Vertical shift: add delta_G to constant H term.
    H = list(new_data.hs_H or [0.0])
    H[0] = H[0] + float(delta_G)
    new_data.hs_H = H
    if delta_x != 0.0:
        width = phase_data.xmax - phase_data.xmin
        clamped_dx = float(np.clip(delta_x, -phase_data.xmin, 1.0 - phase_data.xmax))
        if clamped_dx != 0.0:
            new_data.xmin = phase_data.xmin + clamped_dx
            new_data.xmax = new_data.xmin + width
            # Reparameterise H and S: p(x) → p(x − clamped_dx).
            new_data.hs_H = _shift_poly_coeffs(new_data.hs_H, clamped_dx)
            if new_data.hs_S:
                new_data.hs_S = _shift_poly_coeffs(new_data.hs_S, clamped_dx)
    return new_data


def _resolve_right_patch_chain(target_pd, phase_data_by_name, _seen=None):
    """Follow the right patch chain from *target_pd* to find the root PhaseData.

    Returns ``(root_pd, xmax_eff)`` where *root_pd* is the first phase in the
    chain that has no further right patch (or whose right-patch target cannot
    be resolved), and *xmax_eff* = ``root_pd.xmax``.

    Example:  C→B→A  returns (A, A.xmax).
    """
    if _seen is None:
        _seen = set()
    if (target_pd.patch_right_phase
            and target_pd.patch_right is not None
            and target_pd.name not in _seen):
        _seen.add(target_pd.name)
        next_pd = phase_data_by_name.get(target_pd.patch_right_phase)
        if next_pd is not None:
            return _resolve_right_patch_chain(next_pd, phase_data_by_name, _seen)
    return target_pd, target_pd.xmax


def _resolve_left_patch_chain(target_pd, phase_data_by_name, _seen=None):
    """Follow the left patch chain from *target_pd* to find the root PhaseData.

    Returns ``(root_pd, xmin_eff)`` where *root_pd* is the first phase in the
    chain that has no further left patch, and *xmin_eff* = ``root_pd.xmin``.
    """
    if _seen is None:
        _seen = set()
    if (target_pd.patch_left_phase
            and target_pd.patch_left is not None
            and target_pd.name not in _seen):
        _seen.add(target_pd.name)
        next_pd = phase_data_by_name.get(target_pd.patch_left_phase)
        if next_pd is not None:
            return _resolve_left_patch_chain(next_pd, phase_data_by_name, _seen)
    return target_pd, target_pd.xmin


def compute_left_patch_H(phase_data, target_phase_data, x_cut, T=0.0,
                         phase_data_by_name=None):
    """Compute quadratic H coefficients Q(x) for the left patch region.

    Returns [q0, q1, q2] such that Q(x) = q0 + q1·x + q2·x² satisfies:

    1.  Q(x_cut)  = H_phase(x_cut)          — value continuity at the cut
    2.  Q'(x_cut) = H_phase'(x_cut)         — slope continuity at the cut
    3.  G_patch'(xmin_eff, T) = G_root'(xmin_eff, T)  — G slope match with
        the chain root at the effective left edge.

    Condition 3 expands (S is shared on the left of G_patch) to:
        Q'(xmin_eff) = H_root'(xmin_eff) + T·(S_phase'(xmin_eff) − S_root'(xmin_eff))

    When *phase_data_by_name* is supplied the left patch chain is followed
    (target → its target → …) so that chained patches share a common endpoint.
    When T = 0 the formula reduces to pure H-slope matching at xmin_eff.

    Parameters
    ----------
    phase_data        : PhaseData  — phase being patched
    target_phase_data : PhaseData  — target phase whose slope is matched at xmin
    x_cut             : float      — cut-off composition (left patch covers [xmin, x_cut])
    T                 : float      — temperature for G slope matching (default 0 → H only)
    phase_data_by_name : dict | None — full {name: PhaseData} mapping for chain resolution

    Returns
    -------
    list[float] — [q0, q1, q2] (three coefficients of the quadratic patch)
    """
    import numpy as np
    pv = np.polynomial.polynomial.polyval
    pd_d = np.polynomial.polynomial.polyder

    # Resolve patch chain: follow target's left-patch chain to the root.
    if phase_data_by_name is not None:
        root_pd, xmin_eff = _resolve_left_patch_chain(target_phase_data, phase_data_by_name)
    else:
        root_pd, xmin_eff = target_phase_data, phase_data.xmin

    xmin = xmin_eff
    H   = np.asarray(phase_data.hs_H   or [0.0])
    S   = np.asarray(phase_data.hs_S   or [0.0])
    H_t = np.asarray(root_pd.hs_H or [0.0])
    S_t = np.asarray(root_pd.hs_S or [0.0])

    H_val          = float(pv(x_cut, H))
    dH_at_cut      = float(pv(x_cut, pd_d(H))) if len(H) > 1 else 0.0
    dH_t_at_xmin   = float(pv(xmin,  pd_d(H_t))) if len(H_t) > 1 else 0.0
    dS_at_xmin     = float(pv(xmin,  pd_d(S)))  if len(S)   > 1 else 0.0
    dS_t_at_xmin   = float(pv(xmin,  pd_d(S_t))) if len(S_t) > 1 else 0.0

    # Target G-slope at xmin projected back to H-space:
    slope_target = dH_t_at_xmin + float(T) * (dS_at_xmin - dS_t_at_xmin)

    dx = x_cut - xmin
    if abs(dx) < 1e-10:
        # Degenerate — no patch region; return original H padded to 3 terms.
        h = list(H)
        return (h + [0.0] * max(0, 3 - len(h)))[:3]

    q2 = (dH_at_cut - slope_target) / (2.0 * dx)
    q1 = slope_target - 2.0 * q2 * xmin
    q0 = H_val - q1 * x_cut - q2 * x_cut ** 2
    return [q0, q1, q2]


def compute_right_patch_H(phase_data, target_phase_data, x_cut, T=0.0,
                          phase_data_by_name=None):
    """Compute quadratic H coefficients Q(x) for the right patch region.

    Returns [q0, q1, q2] such that Q(x) = q0 + q1·x + q2·x² satisfies:

    1.  Q(x_cut)   = H_phase(x_cut)           — value continuity at the cut
    2.  Q'(x_cut)  = H_phase'(x_cut)          — slope continuity at the cut
    3.  G_patch'(xmax_eff, T) = G_root'(xmax_eff, T)  — G slope match with
        the chain root at the effective right edge.

    Condition 3 expands to:
        Q'(xmax_eff) = H_root'(xmax_eff) + T·(S_phase'(xmax_eff) − S_root'(xmax_eff))

    When *phase_data_by_name* is supplied the right patch chain is followed
    (target → its target → …) so that chained patches share a common endpoint.
    When T = 0 the formula reduces to pure H-slope matching at xmax_eff.

    Parameters
    ----------
    phase_data        : PhaseData  — phase being patched
    target_phase_data : PhaseData  — target phase whose slope is matched at xmax
    x_cut             : float      — cut-off composition (right patch covers [x_cut, xmax_eff])
    T                 : float      — temperature for G slope matching (default 0 → H only)
    phase_data_by_name : dict | None — full {name: PhaseData} mapping for chain resolution

    Returns
    -------
    list[float] — [q0, q1, q2]
    """
    import numpy as np
    pv  = np.polynomial.polynomial.polyval
    pd_d = np.polynomial.polynomial.polyder

    # Resolve patch chain: follow target's right-patch chain to the root.
    if phase_data_by_name is not None:
        root_pd, xmax = _resolve_right_patch_chain(target_phase_data, phase_data_by_name)
    else:
        root_pd, xmax = target_phase_data, phase_data.xmax

    H   = np.asarray(phase_data.hs_H   or [0.0])
    S   = np.asarray(phase_data.hs_S   or [0.0])
    H_t = np.asarray(root_pd.hs_H or [0.0])
    S_t = np.asarray(root_pd.hs_S or [0.0])

    H_val          = float(pv(x_cut, H))
    dH_at_cut      = float(pv(x_cut, pd_d(H))) if len(H) > 1 else 0.0
    dH_t_at_xmax   = float(pv(xmax,  pd_d(H_t))) if len(H_t) > 1 else 0.0
    dS_at_xmax     = float(pv(xmax,  pd_d(S)))  if len(S)   > 1 else 0.0
    dS_t_at_xmax   = float(pv(xmax,  pd_d(S_t))) if len(S_t) > 1 else 0.0

    slope_target = dH_t_at_xmax + float(T) * (dS_at_xmax - dS_t_at_xmax)

    dx = xmax - x_cut
    if abs(dx) < 1e-10:
        h = list(H)
        return (h + [0.0] * max(0, 3 - len(h)))[:3]

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
        """Select the correct stack page from (_energy_form, _vle_mode)."""
        if self._energy_form == 'polynomial':
            self._stack.setCurrentIndex(1)
        elif self._vle_mode:
            self._stack.setCurrentIndex(2)
        else:
            self._stack.setCurrentIndex(0)

    # -- public API ---------------------------------------------------------

    def set_energy_form(self, form):
        self._energy_form = form
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

    def get_phase_data(self) -> PhaseData:
        pd = PhaseData()
        pd.name = self._name_edit.text() or 'phase'
        pd.phase_type = self._type_combo.currentText()
        pd.xmin = self._xmin_sb.value()
        pd.xmax = self._xmax_sb.value()
        pd.ideal_gas = self._ideal_gas_cb.isChecked()
        if self._vle_mode and pd.phase_type == 'gas' and self._energy_form == 'HS':
            pd.vle = {
                'T_bp_A': self._T_bp_A_sb.value(),
                'T_bp_B': self._T_bp_B_sb.value(),
                'L_A':    self._L_A_sb.value(),
                'L_B':    self._L_B_sb.value(),
            }
        else:
            pd.hs_H = self._H_row.get_coeffs()
            pd.hs_S = self._S_row.get_coeffs()
            pd.hs_V = self._V_row.get_coeffs() if self._V_enable_cb.isChecked() else None
        pd.poly = self._poly_widget.get_coeffs()
        pd.patch_left = (self._patch_slider_to_x(self._left_patch_slider.value())
                         if self._left_patch_cb.isChecked() else None)
        pd.patch_right = (self._patch_slider_to_x(self._right_patch_slider.value())
                          if self._right_patch_cb.isChecked() else None)
        pd.patch_left_phase = (self._left_patch_phase_combo.currentText()
                               if self._left_patch_cb.isChecked() else '')
        pd.patch_right_phase = (self._right_patch_phase_combo.currentText()
                                if self._right_patch_cb.isChecked() else '')
        return pd

    def set_phase_data(self, data: PhaseData):
        self._name_edit.setText(data.name)
        # Block type-combo signal — we handle mode switching explicitly below.
        self._type_combo.blockSignals(True)
        idx = self._type_combo.findText(data.phase_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._type_combo.blockSignals(False)

        self._xmin_sb.setValue(data.xmin)
        self._xmax_sb.setValue(data.xmax)
        self._ideal_gas_cb.setChecked(data.ideal_gas)

        # Always populate the H/S spinboxes so they hold correct values if the
        # user switches to "Use custom H/S instead".
        self._H_row.set_coeffs(data.hs_H or [0.0])
        self._S_row.set_coeffs(data.hs_S or [0.0])
        if data.hs_V:
            self._V_enable_cb.setChecked(True)
            self._V_row.set_coeffs(data.hs_V)
        else:
            self._V_enable_cb.setChecked(False)
        self._poly_widget.set_coeffs(data.poly or [[0.0]])

        if data.vle is not None:
            # Explicit VLE parameterisation — populate spinboxes and show VLE page.
            self._vle_mode = True
            self._T_bp_A_sb.setValue(data.vle.get('T_bp_A', 350.0))
            self._T_bp_B_sb.setValue(data.vle.get('T_bp_B', 400.0))
            self._L_A_sb.setValue(data.vle.get('L_A', 1.0))
            self._L_B_sb.setValue(data.vle.get('L_B', 1.0))
        else:
            # Raw H/S data (including old-style gas phases) — show HS page.
            self._vle_mode = False

        # Restore patch state.
        if data.patch_left is not None:
            self._left_patch_cb.setChecked(True)
            self._left_patch_slider.setValue(self._x_to_patch_slider(data.patch_left))
        else:
            self._left_patch_cb.setChecked(False)
            self._left_patch_slider.setValue(0)
        if data.patch_right is not None:
            self._right_patch_cb.setChecked(True)
            self._right_patch_slider.setValue(self._x_to_patch_slider(data.patch_right))
        else:
            self._right_patch_cb.setChecked(False)
            self._right_patch_slider.setValue(_PATCH_STEPS)
        self._update_patch_labels()

        # Store requested patch-phase names; applied by update_patch_phase_list().
        self._patch_left_phase_name = getattr(data, 'patch_left_phase', '') or ''
        self._patch_right_phase_name = getattr(data, 'patch_right_phase', '') or ''
        # Apply immediately if the combos are already populated.
        for combo, name in (
                (self._left_patch_phase_combo, self._patch_left_phase_name),
                (self._right_patch_phase_combo, self._patch_right_phase_name)):
            if name:
                idx = combo.findText(name)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        self._update_stack_page()



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
    patch_changed  = Signal(object)   # emitted with a preview System on patch slider move

    def __init__(self, system=None, parent=None):
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

    def _add_phase(self, data=None):
        form = self._form_combo.currentText()
        editor = PhaseEditorWidget(energy_form=form)
        if data is not None:
            data.name = self._unique_phase_name(data.name)
            editor.set_phase_data(data)
        else:
            editor._name_edit.setText(self._unique_phase_name('phase'))
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
        self._refresh_patch_phase_lists()

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

    # ------------------------------------------------------------------
    # Patch slider live-preview support
    # ------------------------------------------------------------------

    def set_current_T(self, T: float):
        """Called by MainWindow when the temperature slider moves."""
        self._current_T = float(T)

    def _on_patch_changed(self):
        """Emit a preview system when a left-patch slider or combo changes.

        The emitted system uses the current T for the G slope matching in the
        patch polynomial.  The main window listens and redraws GxCanvas without
        performing a full UI reload.
        """
        try:
            sd = self._collect_system_data()
            if not sd.phases:
                return
            t = self._current_T if self._current_T is not None else sd.T_initial
            system = sd.to_system(current_T=t)
            self.patch_changed.emit(system)
        except Exception:
            pass   # silently ignore during live drag

    # ------------------------------------------------------------------
    # Button slots
    # ------------------------------------------------------------------

    def _on_apply(self) -> bool:
        """Apply current state. Returns True on success, False on error."""
        try:
            sd = self._collect_system_data()
            if not sd.phases:
                QMessageBox.warning(self, 'Apply Error',
                                    'Add at least one phase before applying.')
                return False
            t = self._current_T if self._current_T is not None else sd.T_initial
            system = sd.to_system(current_T=t)
        except Exception as exc:
            QMessageBox.warning(self, 'Apply Error', str(exc))
            return False
        self.system_applied.emit(system)
        return True

    def _on_close(self):
        self._closing = True
        if not self._on_apply():
            self._closing = False
            return
        self.close()

    def closeEvent(self, event):
        if not getattr(self, '_closing', False):
            # Window X button: silently apply if possible, ignore errors
            try:
                sd = self._collect_system_data()
                if sd.phases:
                    t = self._current_T if self._current_T is not None else sd.T_initial
                    system = sd.to_system(current_T=t)
                    self.system_applied.emit(system)
            except Exception:
                pass
        event.accept()

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
