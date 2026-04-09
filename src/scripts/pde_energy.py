#!/usr/bin/env python3
"""
Energy model classes for PDE.

Hierarchy:
  EnergyModel  (ABC)
  ├── HSModel          G(x,T,P) = H(x) - T·S(x) + P·V(x) [+ R·T·ln(P/P°)]
  └── PolyModel        G(x,T,P) = Σᵢ cᵢ(T)·xⁱ + P·V(x) [+ R·T·ln(P/P°)]

Both classes use ascending-order coefficient conventions throughout:
  [c0, c1, c2, ...] means c0 + c1·x + c2·x² + ...
  [a0, a1, a2, ...] means a0 + a1·T + a2·T² + ...

End-member (point) phases are represented naturally:
  HSModel  with single-element H and S lists  → G(T) = H[0] - T·S[0]
  PolyModel with only an x⁰ T-coefficient list → G(T) = Σⱼ a₀ⱼ·Tʲ

Field-values interface
----------------------
EnergyModel.gibbs(x, field_values) takes a dict
mapping field names to scalar values, e.g.
{'temperature': 800.0, 'pressure': 1.5}.  This
dict is forwarded to _gibbs_impl() which each
subclass implements.

Each model also exposes a `couplings` property returning a list of
CouplingTerm objects that describes how each field enters G(x).  This
is used by Phase 3 (visualisation generalisation) to discover field
dependencies without parsing the energy model internals.

Variable pressure
-----------------
All models accept an optional pressure argument P (default 0.0, which
contributes zero to G so existing call sites are unaffected):

  G(x, T, P) = G_base(x, T)
             + P · V(x)                          if V_coeffs is given
             + R_gas · T · ln(P / P_ref)         if ideal_gas=True and P > 0

V(x) = V_coeffs[0] + V_coeffs[1]·x + V_coeffs[2]·x² + ...
R_gas and P_ref are set per-model from the system-level pressure block
by the parser (pde_input.py).

Physical guidance on which terms to use:
  Ideal gas phase:  set ideal_gas=True and omit V_coeffs.
    The R·T·ln(P/P°) term IS the full pressure dependence, derived from
    integrating (dG/dP)_T = V = RT/P.  Adding P·V(x) on top would
    double-count the pressure contribution.
  Condensed phase (liquid/solid):  set ideal_gas=False and supply V_coeffs.
    V is small and nearly P-independent; P·V(x) is the Poynting correction.
  Advanced non-ideal gas:  both terms can be used together if V_coeffs
    represents a correction beyond ideal (e.g. van der Waals co-volume),
    with the understanding that the user is responsible for avoiding
    double-counting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.polynomial import polyval, polyder


# ---------------------------------------------------------------------------
# CouplingTerm — describes how one field (or field combination) enters G(x)
# ---------------------------------------------------------------------------

@dataclass
class CouplingTerm:
    """One field-coupling contribution to the Gibbs energy.

    G(x, {λ}) += R(x) · f({λ})

    where R(x) is a composition polynomial (response function) and
    f({λ}) is a function of one or more field values.

    Attributes
    ----------
    response_coeffs : list[float]
        Coefficients of R(x) in ascending order; R(x) = Σᵢ cᵢ·xⁱ.
    coupling_type   : str
        How f({λ}) is computed:
          'linear'    — f = λ  (single field).  Covers −S·T, V·P, −M·H.
          'ideal_gas' — f = T · ln(P / params['P_ref']).
          'power'     — f = λʲ for a single field (T-power in PolyModel).
    field_names     : list[str]
        Names of the Field objects this term uses (e.g. ['temperature']).
    params          : dict
        Extra parameters (e.g. {'P_ref': 1.0} for ideal_gas coupling).
    """
    response_coeffs: list
    coupling_type:   str
    field_names:     list
    params:          dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EnergyModel(ABC):
    """Abstract base for all Gibbs energy models.

    Subclasses implement _gibbs_impl(x, field_values)
    where field_values is a dict mapping field names
    to scalar values (e.g. {'temperature': 800.0,
    'pressure': 1.5}).
    """

    @abstractmethod
    def _gibbs_impl(self, x, field_values):
        """Return G(x) as a numpy array (same shape
        as x) at the given field values.
        """

    def gibbs(self, x, field_values):
        """Evaluate G at composition *x* and the given
        *field_values* dict.

        Parameters
        ----------
        x            : array-like — composition
        field_values : dict[str, float]
        """
        return self._gibbs_impl(
            np.asarray(x, dtype=float),
            field_values)

    @property
    def couplings(self) -> list:
        """Return the list of CouplingTerm objects describing this model.

        Subclasses override this to provide inspectable coupling structure.
        Default: empty list (no explicit coupling description available).
        """
        return []


# ---------------------------------------------------------------------------
# HSModel
# ---------------------------------------------------------------------------

class HSModel(EnergyModel):
    """G(x, T, P) = H(x) - T·S(x) [+ P·V(x)] [+ R·T·ln(P/P°)]

    H(x) = H_coeffs[0] + H_coeffs[1]·x + H_coeffs[2]·x² + ...
    S(x) = S_coeffs[0] + S_coeffs[1]·x + S_coeffs[2]·x² + ...
    V(x) = V_coeffs[0] + V_coeffs[1]·x + V_coeffs[2]·x² + ...  (optional)

    For an end-member phase at a single composition, pass single-element lists:
      HSModel([H0], [S0])  →  G(T) = H0 - T·S0

    Parameters
    ----------
    H_coeffs  : array-like  — enthalpy polynomial coefficients (ascending x)
    S_coeffs  : array-like  — entropy polynomial coefficients (ascending x)
    V_coeffs  : array-like or None — molar volume polynomial; None → no PV term
    ideal_gas : bool — if True, add R·T·ln(P/P_ref) (ideal gas chemical potential)
    R_gas     : float — gas constant in the user's energy units (e.g. 8.314e-3 kJ/mol/K)
    P_ref     : float — reference pressure in the user's pressure units
    """

    def __init__(self, H_coeffs, S_coeffs,
                 V_coeffs=None, ideal_gas=False,
                 R_gas=0.0, P_ref=1.0):
        self.H_coeffs = np.asarray(H_coeffs, dtype=float)
        self.S_coeffs = np.asarray(S_coeffs, dtype=float)
        self.V_coeffs = np.asarray(V_coeffs, dtype=float) if V_coeffs is not None else None
        self.ideal_gas = bool(ideal_gas)
        self.R_gas = float(R_gas)
        self.P_ref = float(P_ref)
        self.vle_params = None   # set by callers using compute_vle_gas_hs()

    def _gibbs_impl(self, x, field_values: dict) -> np.ndarray:
        T = float(field_values.get('temperature', 0.0))
        P = float(field_values.get('pressure', 0.0))
        G = polyval(x, self.H_coeffs) - T * polyval(x, self.S_coeffs)
        if self.V_coeffs is not None:
            G = G + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            G = G + self.R_gas * T * np.log(P / self.P_ref)
        return G

    @property
    def couplings(self) -> list:
        """CouplingTerm list describing this model's field dependencies."""
        terms = [
            CouplingTerm(
                response_coeffs=(-self.S_coeffs).tolist(),
                coupling_type='linear',
                field_names=['temperature'],
            ),
        ]
        if self.V_coeffs is not None:
            terms.append(CouplingTerm(
                response_coeffs=self.V_coeffs.tolist(),
                coupling_type='linear',
                field_names=['pressure'],
            ))
        if self.ideal_gas:
            terms.append(CouplingTerm(
                response_coeffs=[self.R_gas],
                coupling_type='ideal_gas',
                field_names=['temperature', 'pressure'],
                params={'P_ref': self.P_ref},
            ))
        return terms


# ---------------------------------------------------------------------------
# PiecewisePatchModel — left-patch support
# ---------------------------------------------------------------------------

class PiecewisePatchModel(EnergyModel):
    """G(x, T) = H_eff(x) − T·S(x) where H_eff is piecewise:

        H_eff(x) = H_left(x)  for x ≤ x_cut_left   (left patch, when active)
                   H_orig(x)  in the interior        (always)
                   H_right(x) for x > x_cut_right    (right patch, when active)

    Either or both patches may be active.  S(x) is unchanged across the full
    composition range.

    Parameters
    ----------
    H_orig        : array-like        — original enthalpy polynomial (full range)
    S_coeffs      : array-like        — entropy polynomial (full range, unchanged)
    H_left        : array-like or None — left patch quadratic [q0, q1, q2]
    x_cut_left    : float or None      — cut-off for left patch  (patch covers x ≤ x_cut_left)
    H_right       : array-like or None — right patch quadratic [q0, q1, q2]
    x_cut_right   : float or None      — cut-off for right patch (patch covers x > x_cut_right)
    V_coeffs      : array-like or None
    ideal_gas     : bool
    R_gas, P_ref  : float
    """

    def __init__(self, H_orig, S_coeffs,
                 H_left=None, x_cut_left=None,
                 H_right=None, x_cut_right=None,
                 V_coeffs=None, ideal_gas=False, R_gas=0.0, P_ref=1.0):
        self.H_orig       = np.asarray(H_orig,   dtype=float)
        self.S_coeffs     = np.asarray(S_coeffs, dtype=float)
        self.H_left       = np.asarray(H_left,   dtype=float) if H_left  is not None else None
        self.H_right      = np.asarray(H_right,  dtype=float) if H_right is not None else None
        self.x_cut_left   = float(x_cut_left)  if x_cut_left  is not None else None
        self.x_cut_right  = float(x_cut_right) if x_cut_right is not None else None
        self.V_coeffs     = np.asarray(V_coeffs, dtype=float) if V_coeffs is not None else None
        self.ideal_gas    = bool(ideal_gas)
        self.R_gas        = float(R_gas)
        self.P_ref        = float(P_ref)
        # Metadata for round-trip through from_system().
        self.patch_left_phase_name  = ''
        self.patch_right_phase_name = ''

    def _gibbs_impl(self, x, field_values: dict) -> np.ndarray:
        T = float(field_values.get('temperature', 0.0))
        P = float(field_values.get('pressure', 0.0))
        # Start with the original H everywhere, then overwrite patched regions.
        H_vals = polyval(x, self.H_orig)
        if self.H_left is not None and self.x_cut_left is not None:
            H_vals = np.where(x <= self.x_cut_left,
                              polyval(x, self.H_left), H_vals)
        if self.H_right is not None and self.x_cut_right is not None:
            H_vals = np.where(x > self.x_cut_right,
                              polyval(x, self.H_right), H_vals)
        G = H_vals - T * polyval(x, self.S_coeffs)
        if self.V_coeffs is not None:
            G = G + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            G = G + self.R_gas * T * np.log(P / self.P_ref)
        return G

    @property
    def couplings(self) -> list:
        return [CouplingTerm(
            response_coeffs=(-self.S_coeffs).tolist(),
            coupling_type='linear',
            field_names=['temperature'],
        )]


# ---------------------------------------------------------------------------
# VLE reparameterisation helper
# ---------------------------------------------------------------------------

def compute_vle_gas_hs(liq_H, liq_S, T_bp_A, T_bp_B, L_A, L_B):
    """Derive (H_gas_coeffs, S_gas_coeffs) satisfying all VLE conditions.

    Given a liquid phase's H and S polynomials and the latent heats L_A, L_B
    at the pure-component boiling points T_bp_A (x=0) and T_bp_B (x=1),
    returns gas H and S coefficients such that all 6 VLE consistency conditions
    are satisfied by construction:
      - G_gas = G_liq and dG_gas/dx = dG_liq/dx  at  x=0  (T = T_bp_A)
      - G_gas = G_liq and dG_gas/dx = dG_liq/dx  at  x=1  (T = T_bp_B)
      - S_gas > S_liq  so that vapour is stable above the boiling point

    The ΔH and ΔS corrections follow from a 4-constraint derivation with
    a quadratic Ansatz ΔH(x) = L_A·(1−x)² + L_B·x²:

        ΔH = [L_A,  −2·L_A,   L_A+L_B]
        ΔS = [L_A/T_A, −2·L_A/T_A, L_A/T_A+L_B/T_B]

    Parameters
    ----------
    liq_H   : sequence  — liquid enthalpy coefficients (ascending x)
    liq_S   : sequence  — liquid entropy coefficients (ascending x)
    T_bp_A  : float     — boiling point of pure A (x=0), K
    T_bp_B  : float     — boiling point of pure B (x=1), K
    L_A     : float     — latent heat of A (same units as H)
    L_B     : float     — latent heat of B (same units as H)

    Returns
    -------
    (H_gas_coeffs, S_gas_coeffs) : (list[float], list[float])
        Length-3 lists ready to pass to HSModel.
    """
    # Pad liquid coefficients to at least 3 terms.
    H = list(liq_H) + [0.0] * max(0, 3 - len(liq_H))
    S = list(liq_S) + [0.0] * max(0, 3 - len(liq_S))
    dH = [L_A, -2.0 * L_A, L_A + L_B]
    dS = [L_A / T_bp_A, -2.0 * L_A / T_bp_A, L_A / T_bp_A + L_B / T_bp_B]
    H_gas = [H[i] + dH[i] for i in range(3)]
    S_gas = [S[i] + dS[i] for i in range(3)]
    return H_gas, S_gas


# ---------------------------------------------------------------------------
# Patch-H computation helpers (shared by parser and builder)
# ---------------------------------------------------------------------------

def compute_left_patch_H(H, S, H_target, S_target,
                         xmin, x_cut, T_ref):
    """Return left-patch enthalpy quadratic [q0, q1, q2].

    Constructs a quadratic enthalpy replacement for the left tail of a
    phase (x <= x_cut) that ensures both G and dG/dx continuity at the
    cut point.  At xmin the patch slope matches the target phase's dG/dx,
    ensuring a smooth blend.  Slope matching is evaluated at the reference
    temperature T_ref.

    Parameters
    ----------
    H, S               : array-like
        Own phase H(x) and S(x) coefficient lists in ascending order.
    H_target, S_target : array-like
        Target phase H(x) and S(x) coefficient lists in ascending order.
    xmin  : float — own phase lower composition bound
    x_cut : float — left-patch / interior boundary
    T_ref : float — reference temperature for slope matching
    """
    H = np.asarray(H)
    S = np.asarray(S)
    H_target = np.asarray(H_target)
    S_target = np.asarray(S_target)
    H_val = float(polyval(x_cut, H))
    dH_at_cut = (
        float(polyval(x_cut, polyder(H)))
        if len(H) > 1 else 0.0)
    dH_tgt_at_xmin = (
        float(polyval(xmin, polyder(H_target)))
        if len(H_target) > 1 else 0.0)
    dS_at_xmin = (
        float(polyval(xmin, polyder(S)))
        if len(S) > 1 else 0.0)
    dS_tgt_at_xmin = (
        float(polyval(xmin, polyder(S_target)))
        if len(S_target) > 1 else 0.0)
    slope_target = (
        dH_tgt_at_xmin
        + float(T_ref)
        * (dS_at_xmin - dS_tgt_at_xmin))
    dx = x_cut - xmin
    if abs(dx) < 1e-10:
        h = H.tolist()
        return (
            h + [0.0] * max(0, 3 - len(h))
        )[:3]
    q2 = (dH_at_cut - slope_target) / (2.0 * dx)
    q1 = slope_target - 2.0 * q2 * xmin
    q0 = H_val - q1 * x_cut - q2 * x_cut ** 2
    return [q0, q1, q2]


def compute_right_patch_H(H, S, H_target, S_target,
                          xmax, x_cut, T_ref):
    """Return right-patch enthalpy quadratic [q0, q1, q2].

    Constructs a quadratic enthalpy replacement for the right tail of a
    phase (x > x_cut) that ensures both G and dG/dx continuity at the
    cut point.  At xmax the patch slope matches the target phase's dG/dx,
    ensuring a smooth blend.  Mirror of compute_left_patch_H; the slope
    matching is evaluated at the reference temperature T_ref.

    Parameters
    ----------
    H, S               : array-like
        Own phase H(x) and S(x) coefficient lists in ascending order.
    H_target, S_target : array-like
        Target phase H(x) and S(x) coefficient lists in ascending order.
    xmax  : float — own phase upper composition bound
    x_cut : float — interior / right-patch boundary
    T_ref : float — reference temperature for slope matching
    """
    H = np.asarray(H)
    S = np.asarray(S)
    H_target = np.asarray(H_target)
    S_target = np.asarray(S_target)
    H_val = float(polyval(x_cut, H))
    dH_at_cut = (
        float(polyval(x_cut, polyder(H)))
        if len(H) > 1 else 0.0)
    dH_tgt_at_xmax = (
        float(polyval(xmax, polyder(H_target)))
        if len(H_target) > 1 else 0.0)
    dS_at_xmax = (
        float(polyval(xmax, polyder(S)))
        if len(S) > 1 else 0.0)
    dS_tgt_at_xmax = (
        float(polyval(xmax, polyder(S_target)))
        if len(S_target) > 1 else 0.0)
    slope_target = (
        dH_tgt_at_xmax
        + float(T_ref)
        * (dS_at_xmax - dS_tgt_at_xmax))
    dx = xmax - x_cut
    if abs(dx) < 1e-10:
        h = H.tolist()
        return (
            h + [0.0] * max(0, 3 - len(h))
        )[:3]
    q2 = (slope_target - dH_at_cut) / (2.0 * dx)
    q1 = dH_at_cut - 2.0 * q2 * x_cut
    q0 = H_val - q1 * x_cut - q2 * x_cut ** 2
    return [q0, q1, q2]


# ---------------------------------------------------------------------------
# PolyModel
# ---------------------------------------------------------------------------

class PolyModel(EnergyModel):
    """G(x, T, P) = Σᵢ cᵢ(T)·xⁱ [+ P·V(x)] [+ R·T·ln(P/P°)]

    t_poly_coeffs is a list indexed by x-power i. Each element is an array
    of T-polynomial coefficients in ascending order:
      t_poly_coeffs[i][j]  is the coefficient for  xⁱ·Tʲ

    For an end-member phase, pass a single-element outer list:
      PolyModel([[a00, a01, a02, ...]])  →  G(T) = a00 + a01·T + a02·T² + ...

    Parameters
    ----------
    t_poly_coeffs : list of array-like — T-polynomials for each x power
    V_coeffs      : array-like or None — molar volume polynomial; None → no PV term
    ideal_gas     : bool — if True, add R·T·ln(P/P_ref)
    R_gas         : float — gas constant in the user's energy units
    P_ref         : float — reference pressure in the user's pressure units
    """

    def __init__(self, t_poly_coeffs,
                 V_coeffs=None, ideal_gas=False,
                 R_gas=0.0, P_ref=1.0):
        self.coeffs = [np.asarray(tc, dtype=float) for tc in t_poly_coeffs]
        self.V_coeffs = np.asarray(V_coeffs, dtype=float) if V_coeffs is not None else None
        self.ideal_gas = bool(ideal_gas)
        self.R_gas = float(R_gas)
        self.P_ref = float(P_ref)

    def _gibbs_impl(self, x, field_values: dict) -> np.ndarray:
        T = float(field_values.get('temperature', 0.0))
        P = float(field_values.get('pressure', 0.0))
        result = np.zeros_like(x)
        for i, t_coeffs in enumerate(self.coeffs):
            c_i = polyval(T, t_coeffs)   # scalar: T-polynomial evaluated at T
            result = result + c_i * x**i
        if self.V_coeffs is not None:
            result = result + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            result = result + self.R_gas * T * np.log(P / self.P_ref)
        return result

    @property
    def couplings(self) -> list:
        """CouplingTerm list describing this model's field dependencies."""
        # Represent the full T-polynomial structure as a single 'poly_T' term.
        # The coefficient grid is stored in params for Phase 3 inspection.
        terms = [
            CouplingTerm(
                response_coeffs=[1.0],   # placeholder; full structure in params
                coupling_type='poly_T',
                field_names=['temperature'],
                params={'t_poly_coeffs': [c.tolist() for c in self.coeffs]},
            ),
        ]
        if self.V_coeffs is not None:
            terms.append(CouplingTerm(
                response_coeffs=self.V_coeffs.tolist(),
                coupling_type='linear',
                field_names=['pressure'],
            ))
        if self.ideal_gas:
            terms.append(CouplingTerm(
                response_coeffs=[self.R_gas],
                coupling_type='ideal_gas',
                field_names=['temperature', 'pressure'],
                params={'P_ref': self.P_ref},
            ))
        return terms


# -----------------------------------------------------------
# CALPHADModel — wraps pycalphad for assessed TDB data
# -----------------------------------------------------------

class CALPHADModel(EnergyModel):
    """G(x, T, P) from a pycalphad Database + phase.

    Evaluates the molar Gibbs energy of a single
    phase in a binary system using thermodynamic
    parameters from a TDB file.  pycalphad handles
    sublattice models, Redlich-Kister mixing, and
    magnetic contributions internally.

    All values are in SI: G in J/mol, T in K,
    P in Pa.

    The constructor inspects the phase's sublattice
    structure and precomputes a column layout for
    building site-fraction arrays from composition
    *x*.  For each sublattice the layout records
    which columns correspond to the A and B
    components and which are pure-vacancy (fixed
    at 1.0).  Species within each sublattice are
    sorted alphabetically, matching pycalphad's
    internal DOF ordering.

    Parameters
    ----------
    db : pycalphad.Database
        Pre-loaded TDB database (shared across
        phases in the same system; loaded once
        by SystemSpec.to_system()).
    phase_name : str
        Phase identifier in the TDB file, e.g.
        'LIQUID', 'FCC_A1', 'HCP_A3'.
    components : list[str]
        Binary component names in the order used
        by the System (e.g. ['AL', 'MG']).  PDE's
        composition x is the mole fraction of
        components[-1].  'VA' (vacancy) is appended
        automatically for pycalphad.
    P_default : float
        Default pressure in Pa when field_values
        has no 'pressure' key (default 101325 =
        1 atm).

    Notes
    -----
    pycalphad is lazy-imported in __init__ so the
    application works when pycalphad is not
    installed.
    """

    def __init__(self, db, phase_name, components,
                 P_default=101325.0):
        import pycalphad as _pc
        self._pc = _pc
        self._db = db
        self._phase_name = phase_name
        self._components = list(components)
        self._comps_va = (
            list(components) + ['VA'])
        self._P_default = P_default

        # Precompute the site-fraction column
        # layout from the phase's sublattice
        # structure.  Each entry in _sf_layout
        # is a sorted list of species names that
        # are active in our component set for
        # that sublattice.  We use
        # phase.constituents (not .sublattices,
        # which holds stoichiometric multipliers)
        # and extract the .name attribute from
        # each pycalphad Species object.
        comp_set = set(self._comps_va)
        phase_obj = db.phases[phase_name]
        self._sf_layout = []
        for subl in phase_obj.constituents:
            active = sorted(
                sp.name for sp in subl
                if sp.name in comp_set)
            self._sf_layout.append(active)

        # Precompute component-name → index map
        # for fast lookup in _gibbs_impl.
        self._comp_idx = {
            name: i for i, name
            in enumerate(self._components)}

    def _gibbs_impl(self, x, field_values):
        """Evaluate G(x) at the given field values.

        Binary: *x* is shape (n,) — mole fraction
        of components[-1].
        Multi-component: *x* is shape (n, N-1) —
        independent mole fractions [x_2, ..., x_N].
        x_1 = 1 - sum(x_i).

        Returns G values in J/mol as a 1-D array.
        """
        T = field_values.get(
            'temperature', 300.0)
        P = field_values.get(
            'pressure', self._P_default)

        x_arr = np.asarray(x, dtype=float)
        N = len(self._components)

        # Build full mole-fraction array (n, N)
        # from the independent coordinates.
        if x_arr.ndim == 1:
            # Binary or squeezed: (n,) → (n, 1)
            x_indep = x_arr.reshape(-1, 1)
        else:
            x_indep = x_arr        # (n, N-1)
        n_pts = x_indep.shape[0]
        x_first = (
            1.0 - x_indep.sum(axis=1))
        x_full = np.column_stack(
            [x_first, x_indep])    # (n, N)

        # Build site-fraction columns for each
        # sublattice in the order pycalphad
        # expects (alphabetical within each
        # sublattice).
        columns = []
        for species_list in self._sf_layout:
            for sp_name in species_list:
                ci = self._comp_idx.get(
                    sp_name)
                if ci is not None:
                    columns.append(
                        x_full[:, ci])
                elif sp_name == 'VA':
                    columns.append(
                        np.ones(n_pts))
                else:
                    columns.append(
                        np.zeros(n_pts))

        pts = np.column_stack(columns)

        result = self._pc.calculate(
            self._db,
            self._comps_va,
            self._phase_name,
            T=float(T),
            P=float(P),
            points=pts,
            output='GM')

        return result.GM.values.flatten()

    @property
    def couplings(self):
        """CALPHAD couplings are complex and not
        decomposable into simple CouplingTerm
        objects.
        """
        return []
